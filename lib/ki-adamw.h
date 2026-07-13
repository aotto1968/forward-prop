/*
 * lib/ki-adamw.h — Float32/AdamW helpers for AdamW reference trainer
 * ==================================================================
 *
 * Provides: ki_LinearLayer, forward/relu, kaiming init, AdamW optimizer,
 * gradient clipping, shuffle, and display helpers.
 *
 * Dataset-independent — works for both MNIST and CIFAR.
 * Include AFTER ki-common.h:
 *   #include "ki-common.h"
 *   #include "ki-adamw.h"
 */
#ifndef KI_ADAMW_H
#define KI_ADAMW_H

#include <math.h>

/* ═══════════════════════════════════════════════════════════════════════
 * FLOAT32 LINEAR LAYER
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

/* ── Kaiming init mit w0_random() (konsistent mit Otto/Hebbian) ── */
#include "w0_random.h"
static inline void ki_init_kaiming_w0(ki_LinearLayer *layer, uint64_t seed) {
    w0_srandom(seed);
    float bound = 1.0f / sqrtf((float)layer->in_features);
    int n = layer->out_features * layer->in_features;
    for (int i = 0; i < n; i++)
        layer->W[i] = (float)w0_random() / (float)UINT32_MAX * 2.0f * bound - bound;
}

/* For ENS_SEED_ONCE: no reset, reads from current w0_random() stream */
static inline void ki_init_kaiming_w0_seq(ki_LinearLayer *layer) {
    float bound = 1.0f / sqrtf((float)layer->in_features);
    int n = layer->out_features * layer->in_features;
    for (int i = 0; i < n; i++)
        layer->W[i] = (float)w0_random() / (float)UINT32_MAX * 2.0f * bound - bound;
}

/* ═══════════════════════════════════════════════════════════════════════
 * ADAMW OPTIMIZER
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    size_t n;
    float lr, lr_min;
    float beta1, beta2, eps, wd;
    float *m, *v;
    int t;
} ki_AdamWState;

static inline ki_AdamWState ki_adamw_create(size_t n, float lr, float lr_min) {
    ki_AdamWState s;
    s.n = n;
    s.lr = lr;
    s.lr_min = lr_min;
    s.beta1 = 0.9f;
    s.beta2 = 0.999f;
    s.eps = 1e-8f;
    s.wd = 1e-4f;
    s.m = (float *)calloc(n, sizeof(float));
    s.v = (float *)calloc(n, sizeof(float));
    s.t = 0;
    if (!s.m || !s.v) { fprintf(stderr, "[FATAL] adamw_create OOM\n"); exit(1); }
    return s;
}

static inline void ki_adamw_step(ki_AdamWState *s, float *w, const float *grad) {
    s->t++;
    float lr_t = s->lr * sqrtf(1.0f - powf(s->beta2, (float)s->t))
                          / (1.0f - powf(s->beta1, (float)s->t));
    #pragma omp parallel for if(s->n >= 1024)
    for (size_t i = 0; i < s->n; i++) {
        float g = grad[i];
        s->m[i] = s->beta1 * s->m[i] + (1.0f - s->beta1) * g;
        s->v[i] = s->beta2 * s->v[i] + (1.0f - s->beta2) * g * g;
        w[i] -= lr_t * s->m[i] / (sqrtf(s->v[i]) + s->eps);
        w[i] -= lr_t * s->wd * w[i]; /* weight decay */
    }
}

static inline void ki_adamw_free(ki_AdamWState *s) {
    free(s->m);
    free(s->v);
}

static inline void ki_clip_grad(float *grad, size_t n) {
    float norm = 0.0f;
    for (size_t i = 0; i < n; i++) norm += grad[i] * grad[i];
    norm = sqrtf(norm);
    if (norm > 1.0f) {
        float inv = 1.0f / norm;
        for (size_t i = 0; i < n; i++) grad[i] *= inv;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * SHUFFLE
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void ki_shuffle_int(int *indices, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int t = indices[i]; indices[i] = indices[j]; indices[j] = t;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * ARCHITECTURE DISPLAY
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
