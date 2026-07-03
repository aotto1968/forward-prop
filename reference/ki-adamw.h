/*
 * cifar-1/ki-adamw.h — Float32/AdamW helpers für CIFAR-10
 * =========================================================
 *
 * Stellt bereit: ki_LinearLayer, ki_SetupInfo, ki_setup_show,
 * ki_linear_forward, ki_leaky_relu, ki_init_kaiming, ki_shuffle.
 *
 * Enthält NICHT die Otto-Score-Helfer (batch_correct etc.).
 * Für Otto Score: #include "ki-common.h" (shared, via Symlink).
 */
#ifndef KI_ADAMW_H
#define KI_ADAMW_H

/* ki-common.h brings all shared infrastructure (ki_xmalloc, lr_schedule, report, etc.) */
#include "ki-common.h"
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════════
 * FLOAT32 OPERATIONS
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    int in_features;
    int out_features;
    float *W;
} ki_LinearLayer;

static inline void ki_linear_forward(const ki_LinearLayer *layer,
                                      const float *x, float *output, int batchN) {
    #pragma omp parallel for collapse(2) if(batchN * layer->out_features >= 64)
    for (int b = 0; b < batchN; b++) {
        for (int o = 0; o < layer->out_features; o++) {
            float acc = 0.0f;
            for (int i = 0; i < layer->in_features; i++)
                acc += x[b * layer->in_features + i] * layer->W[o * layer->in_features + i];
            output[b * layer->out_features + o] = acc;
        }
    }
}

static inline void ki_leaky_relu(float *x, int n) {
    #pragma omp parallel for if(n >= 256)
    for (int i = 0; i < n; i++)
        if (x[i] < 0.0f) x[i] *= 0.05f;
}

static inline void ki_init_kaiming(ki_LinearLayer *layer, unsigned int seed) {
    srand(seed);
    float bound = 1.0f / sqrtf((float)layer->in_features);
    int n = layer->out_features * layer->in_features;
    for (int i = 0; i < n; i++)
        layer->W[i] = (float)rand() / (float)RAND_MAX * 2.0f * bound - bound;
}

#ifndef KI_COMMON_H
/* ki_shuffle now in ki-common.h — only define here if ki-common.h not included */
static inline void ki_shuffle(int *indices, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int t = indices[i]; indices[i] = indices[j]; indices[j] = t;
    }
}
#endif

/* ── ki_pack_blocks_float: pack selected blocks as float arrays ──
 * CIFAR-10 stores RGB planar: R[1024] G[1024] B[1024].
 * For each selected block (bit in --channels mask), produces floats:
 *   packed=1:  4px averaged → 1 float (KI_NC=256 containers per block)
 *   packed=0:  1px → 1 float (1024 containers per block, full resolution)
 *
 * Layout: [block_a_0..N] [block_b_0..N] ... in block order (0..7).
 *   Block 0=R, 1=G, 2=B, 3=Y(ITU-601), 4=LUM, 5=RG, 6=BY, 7=YL(ITU-709)
 *
 * mask < 0: fallback to r+g+b (0x07 = 3-channel color).
 * packed defaults to 1 (4px/cont) if not specified.
 * Uses the same ki_blocks_from_rgb() as Otto Score — unified block math.
 */
static float *ki_pack_blocks_float(const uint8_t *X_raw, int n_samples,
                                    int mask, int packed) {
    /* Collect active blocks in order */
    int active[8], n_active = 0;
    if (mask < 0) mask = 0x07;
    for (int b = 0; b < 8; b++)
        if (mask & (1 << b)) active[n_active++] = b;
    if (n_active == 0) { active[0]=0; active[1]=1; active[2]=2; n_active=3; }

    int nc = packed ? KI_NC : (KI_PX / 3);  /* 256 or 1024 per block */
    int total_nc = n_active * nc;
    size_t total = (size_t)n_samples * (size_t)total_nc;
    float *packed_out = (float *)malloc(total * sizeof(float));

    for (int s = 0; s < n_samples; s++) {
        size_t base = (size_t)s * (size_t)KI_PX;
        float *row = packed_out + (size_t)s * (size_t)total_nc;

        /* Compute all 8 block values per pixel via shared ki_blocks_from_rgb() */
        uint8_t block_px[8][1024];
        for (int p = 0; p < 1024; p++) {
            int r = (int)X_raw[base + (size_t)p];
            int g = (int)X_raw[base + 1024 + (size_t)p];
            int b = (int)X_raw[base + 2048 + (size_t)p];
            uint8_t blk[COLOR_NB];
            ki_blocks_from_rgb(r, g, b, blk);
            for (int i = 0; i < 8; i++)
                block_px[i][p] = blk[i];
        }

        /* For each active block: pack float(s) per pixel group */
        for (int ai = 0; ai < n_active; ai++) {
            uint8_t *src = block_px[active[ai]];
            float *dst = row + (size_t)ai * (size_t)nc;
            if (packed) {
                /* 4px averaged → 1 float (256 containers) */
                for (int c = 0; c < nc; c++) {
                    int sum = 0;
                    for (int k = 0; k < 4; k++)
                        sum += (int)src[(size_t)c * 4 + (size_t)k];
                    dst[c] = (float)sum / (4.0f * 127.5f) - 1.0f;
                }
            } else {
                /* 1px → 1 float (1024 containers, full resolution) */
                for (int p = 0; p < nc; p++)
                    dst[p] = (float)src[p] / 127.5f - 1.0f;
            }
        }
    }
    return packed_out;
}

/* AdamW default LR: 0.005 works better than the global 0.05 for float matmul.
 * Override via --lr. See: logs (2026-06-25) packed lr=0.005 → 44.8% vs 43.5%. */
#define ADAMW_DEFAULT_LR 0.005f

/* Legacy alias: r+g+b packed (3-channel color, 768 containers) */
#define KI_NC_ADAMW (KI_NC * 3)  /* 768 containers */

/* ═══════════════════════════════════════════════════════════════════════
 * DISPLAY — Architecture overview
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    const char *title;
    int   H;  int   epochs;
    int   bits_per_cont;  int   pixel_bits;
    unsigned int seed;
    int   N;  int   ne;  int   n_threads;
    int   px;  int   sizeof_bn;  int   nc;  int   C;
    size_t input_bit, hidden_bit, output_bit, w0_bit, w1_bit;
    int   batchN, maxFlips, no_close, w1_hidden_nrn, w0_hidden_nrn;
} ki_SetupInfo;

static inline void ki_setup_show(const ki_SetupInfo *s) {
    printf("══════════════════════════════════════════════════════════════════════\n");
    printf("══╡ %s ╞══  H=%-4d  Ep=%-2d  bitsPerCont=%-2d  pixelBits=%-2d  seed=%-4u",
           s->title, s->H, s->epochs, s->bits_per_cont, s->pixel_bits, s->seed);
    if (s->batchN > 0) printf("  batchN=%d", s->batchN);
    printf("\n══╡ SETUP ╞══════════════════════════════════════════════════════════\n");
    printf("  OMP:         %d threads\n", s->n_threads);
    printf("  Train/Eval:  %d / %d samples\n", s->N, s->ne);
    printf("  Input:       %d pixels (%dx%dx3)\n", s->px,
           s->px == 3072 ? 32 : 28, s->px == 3072 ? 32 : 28);
    printf("  sizeof(bn)   %d byte\n", s->sizeof_bn);
    printf("  ───────────────────────────────────────────────────────────\n");
    printf("  INPUT        %4d nrn x %2d bit  = %7zu bit  (%5.1f KB)\n",
           s->nc, s->bits_per_cont, s->input_bit, (double)s->input_bit/8/1024);
    printf("  HIDDEN       %4d nrn x %2d bit  = %7zu bit  (%5.1f KB)\n",
           s->H, s->bits_per_cont, s->hidden_bit, (double)s->hidden_bit/8/1024);
    printf("  OUTPUT       %4d nrn x %2d bit  = %7zu bit  (%5.1f KB)\n",
           10, s->bits_per_cont, s->output_bit, (double)s->output_bit/8/1024);
    printf("  ───────────────────────────────────────────────────────────\n");
    printf("  W0 = %4d x %4d x %2d bit  = %9zu bit  (%5.1f KB)\n",
           s->H, s->nc, s->bits_per_cont, s->w0_bit, (double)s->w0_bit/8/1024);
    printf("  W1 = %4d x %4d x %2d bit  = %9zu bit  (%5.1f KB)\n",
           10, s->H, s->bits_per_cont, s->w1_bit, (double)s->w1_bit/8/1024);
    printf("  ───────────────────────────────────────────────────────────\n");
    printf("  TOTAL (W0+W1)              %7zu bit  (%5.1f KB)\n",
           s->w0_bit + s->w1_bit, (double)(s->w0_bit + s->w1_bit)/8/1024);
    printf("══════════════════════════════════════════════════════════════════════\n");
}

#endif /* KI_ADAMW_H */
