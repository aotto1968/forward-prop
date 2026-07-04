/*
 * mnist-1/mlp-bin32-hebbian-trn.c — Multi-Member Hebbian (MNIST + CIFAR-10)
 * ============================================================================
 *
 * Bitwise Hebbian training with multi-encoding support (like Otto Score).
 * Each enc_array entry becomes a "member" with independent W0+W1.
 * Final prediction: argmax of sum of scores across all members.
 *
 * Forward: MAJ3(~(in_slice ^ W0)) → h0 → XNOR + popcnt(W1, h0) → score
 * Training: counter-based co-occurrence sign-flip on W1.
 *
 * Dataset-specific via ki-local.h (KI_COLORS, KI_PX, KI_NC).
 * Symlink: cifar/mlp-bin32-hebbian-trn.c → ../mnist/mlp-bin32-hebbian-trn.c
 *
 * Usage:
 *   ./mlp-bin32-hebbian-trn-xnor.exe --hiddenN 256 --epochsN 3 --encoding exp
 *   ./mlp-bin32-hebbian-trn-xnor.exe --model models/hebbian --evalN 10000
 */
#define _POSIX_C_SOURCE 200809L
#define KI_COMMON_LOAD_INPUT
#include "ki-common.h"
#include "maj3.h"
#include "w0_random.h"
#include <inttypes.h>

#define N_CLASSES KI_NCLASSES
#define BITS 32

#ifdef H0_XOR
#  define H0_STR      "XOR"
#  define H0_OP(a,b)  ((~(a) ^ (b)))
#else
#  define H0_STR      "XNOR"
#  define H0_OP(a,b)  (~((a) ^ (b)))
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * CUSTOM load_input — applies encoding per enc_array entry
 * ═══════════════════════════════════════════════════════════════════════
 * Same approach as Otto Score: pack encoded pixels into uint32 containers.
 * The enc_array is populated by ki_parse_args() via --encoding.
 * Each container is one enc_array entry × 4px (MNIST) or R|G|B (CIFAR). */

static uint32_t *load_input(const uint8_t *X_raw, int n_samples) {
    int n_enc = aa.enc_count > 0 ? aa.enc_count : 1;
    int enc_off[KI_ENC_MAX], enc_nc[KI_ENC_MAX], eff_enc = 0;
    size_t stride = 0;
    for (int i = 0; i < aa.enc_count && i < KI_ENC_MAX; i++) {
        int w = (int)aa.enc_array[i].width;
        if (w < 1) w = KI_ENC_WIDTH_DEFAULT;
        enc_nc[eff_enc] = (KI_COLORS > 1 ? KI_NC : NC) * w / 8;
        enc_off[eff_enc] = (int)stride;
        stride += (size_t)enc_nc[eff_enc];
        eff_enc++;
    }
    if (eff_enc == 0) {
        /* Fallback: raw 4px packing (MNIST default) */
        size_t nc = (size_t)(KI_COLORS > 1 ? KI_NC * 3 : KI_NC);
        uint32_t *Xb = (uint32_t *)ki_xmalloc((size_t)n_samples * nc * sizeof(uint32_t));
        for (int s = 0; s < n_samples; s++) {
            uint32_t *row = Xb + (size_t)s * nc;
            if (KI_COLORS > 1) {
                for (int c = 0; c < 1024; c++) {
                    size_t base = (size_t)s * (size_t)KI_PX;
                    row[c] = (uint32_t)X_raw[base + (size_t)c]
                           | ((uint32_t)X_raw[base + 1024 + (size_t)c] << 8)
                           | ((uint32_t)X_raw[base + 2048 + (size_t)c] << 16);
                }
            } else {
                for (int c = 0; c < (int)nc; c++) {
                    uint32_t val = 0;
                    for (int k = 0; k < 4; k++) {
                        size_t p = (size_t)s * (size_t)KI_PX + (size_t)c * 4 + (size_t)k;
                        val |= ((uint32_t)X_raw[p] & 0xFFU) << (unsigned)(k * 8);
                    }
                    row[c] = val;
                }
            }
        }
        return Xb;
    }
    size_t total = (size_t)n_samples * stride;
    uint32_t *Xb = (uint32_t *)ki_xmalloc(total * sizeof(uint32_t));
    for (int s = 0; s < n_samples; s++) {
        uint32_t *row = Xb + (size_t)s * stride;
        uint8_t px[COLOR_NB][1024];
        size_t base = (size_t)s * (size_t)KI_PX;
        if (KI_COLORS > 1) {
            for (int p = 0; p < 1024; p++) {
                int r = (int)X_raw[base + (size_t)p];
                int g = (int)X_raw[base + 1024 + (size_t)p];
                int b = (int)X_raw[base + 2048 + (size_t)p];
                uint8_t blk[COLOR_NB];
                ki_blocks_from_rgb(r, g, b, blk);
                for (int bj = 0; bj < COLOR_NB; bj++) px[bj][p] = blk[bj];
            }
        }
        for (int ei = 0; ei < eff_enc; ei++) {
            int col = (int)aa.enc_array[ei].color;
            int typ = (int)aa.enc_array[ei].type;
            int w   = (int)aa.enc_array[ei].width;
            if (w < 1) w = KI_ENC_WIDTH_DEFAULT;
            if (typ < 0) typ = KI_ENC_LIN7;
            int pack = 32 / w, shift = w, off = enc_off[ei];
            if (KI_COLORS <= 1) {
                for (int c = 0; c < enc_nc[ei]; c++) {
                    uint32_t val = 0;
                    for (int k = 0; k < pack; k++) {
                        size_t p = base + (size_t)(c * pack + k);
                        uint8_t pv = X_raw[p];
                        val |= enc_lut_get(typ, w, pv) << (unsigned)(k * shift);
                    }
                    row[(size_t)off + (size_t)c] = val;
                }
            } else {
                for (int c = 0; c < enc_nc[ei]; c++) {
                    uint32_t val = 0;
                    for (int k = 0; k < pack; k++) {
                        uint8_t pv = (col >= 0 && col < COLOR_NB) ? px[col][c * pack + k] : 0;
                        val |= enc_lut_get(typ, w, pv) << (unsigned)(k * shift);
                    }
                    row[(size_t)off + (size_t)c] = val;
                }
            }
        }
    }
    (void)enc_off;
    return Xb;
}

/* ═══════════════════════════════════════════════════════════════════════
 * H0 via majority_tree (full MAJ3 over nc_slice containers)
 * ═══════════════════════════════════════════════════════════════════════ */
static inline void h0_compute(const uint32_t *in, const uint32_t *W0,
                               uint32_t *h0, int H, int nc) {
    for (int h = 0; h < H; h++) {
        const uint32_t *row = W0 + (size_t)h * (size_t)nc;
        uint32_t match[4096];
        for (int c = 0; c < nc; c++)
            match[c] = H0_OP(in[c], row[c]);
        h0[h] = majority_tree(match, nc);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * SCORING — single member
 * ═══════════════════════════════════════════════════════════════════════ */
static inline void member_score(const uint32_t *h0, const uint32_t *W1,
                                 int H, int64_t scores[N_CLASSES]) {
    for (int k = 0; k < N_CLASSES; k++) {
        int64_t sum = 0;
        for (int h = 0; h < H; h++)
            sum += (int64_t)__builtin_popcount(~(W1[(size_t)k * (size_t)H + h] ^ h0[h]));
        scores[k] = sum;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * ACCURACY — multi-member voting
 * ═══════════════════════════════════════════════════════════════════════ */
static float accuracy_multi(const uint32_t *X, const uint8_t *Y, int N,
                             const uint32_t *const *W0s, const uint32_t *const *W1s,
                             const int *ncs, const int *offsets, int H, int n_mem) {
    int ok = 0;
    #pragma omp parallel for reduction(+:ok) schedule(static)
    for (int s = 0; s < N; s++) {
        int64_t total[N_CLASSES];
        for (int k = 0; k < N_CLASSES; k++) total[k] = 0;
        for (int m = 0; m < n_mem; m++) {
            const uint32_t *slice = X + (size_t)s * (size_t)(offsets[n_mem - 1] + ncs[n_mem - 1])
                                    + offsets[m];
            uint32_t h0_buf[4096];
            h0_compute(slice, W0s[m], h0_buf, H, ncs[m]);
            int64_t scores[N_CLASSES];
            member_score(h0_buf, W1s[m], H, scores);
            for (int k = 0; k < N_CLASSES; k++) total[k] += scores[k];
        }
        int pred = 0;
        for (int k = 1; k < N_CLASSES; k++)
            if (total[k] > total[pred]) pred = k;
        if (pred == (int)Y[s]) ok++;
    }
    return 100.0f * (float)ok / (float)N;
}

/* ── Single-member accuracy (backward compat) ───────────────────── */
static float accuracy(const uint32_t *X, const uint8_t *Y, int N,
                       const uint32_t *W0, const uint32_t *W1, int H, int nc) {
    return accuracy_multi(X, Y, N, &W0, &W1, &nc, (int[]){0}, H, 1);
}

/* ═══════════════════════════════════════════════════════════════════════
 * HEBBIAN UPDATE — counter-based, single member
 * ═══════════════════════════════════════════════════════════════════════ */
static void hebbian_update(uint32_t *W1, const uint32_t *X, const uint8_t *Y,
                            int N, const uint32_t *W0, int H, int nc,
                            unsigned int seed, int *flips_out,
                            int hebbian_pct) {
    int K = N_CLASSES;
    size_t csize = (size_t)K * (size_t)H * (size_t)BITS;
    int *counter = (int *)calloc(csize, sizeof(int));
    int *class_cnt = (int *)calloc((size_t)K, sizeof(int));
    int *idx = (int *)malloc((size_t)N * sizeof(int));
    for (int i = 0; i < N; i++) idx[i] = i;
    srand(seed);
    for (int i = N - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }
    #pragma omp parallel
    {
        int *loc_ctr = (int *)calloc(csize, sizeof(int));
        int loc_cc[K]; memset(loc_cc, 0, sizeof(loc_cc));
        uint32_t *h0_buf = (uint32_t *)malloc((size_t)H * sizeof(uint32_t));
        #pragma omp for schedule(static)
        for (int si = 0; si < N; si++) {
            if (!loc_ctr || !h0_buf) continue;
            int s = idx[si], lb = (int)Y[s];
            loc_cc[lb]++;
            h0_compute(X + (size_t)s * (size_t)nc, W0, h0_buf, H, nc);
            for (int h = 0; h < H; h++) {
                uint32_t hv = h0_buf[h];
                if (hv == 0) continue;
                size_t base = ((size_t)lb * (size_t)H + (size_t)h) * (size_t)BITS;
                for (int b = 0; b < BITS; b++)
                    if (hv & (1U << (unsigned)b))
                        loc_ctr[base + (size_t)b]++;
            }
        }
        #pragma omp critical
        {
            for (size_t i = 0; i < csize; i++) counter[i] += loc_ctr[i];
            for (int k = 0; k < K; k++) class_cnt[k] += loc_cc[k];
        }
        free(loc_ctr); free(h0_buf);
    }
    int total_flips = 0;
    for (int k = 0; k < K; k++) {
        int th = (class_cnt[k] * hebbian_pct + 50) / 100;
        if (th < 1) th = 1;
        for (int h = 0; h < H; h++) {
            uint32_t old_w = W1[(size_t)k * (size_t)H + (size_t)h];
            uint32_t new_w = 0;
            size_t base = ((size_t)k * (size_t)H + (size_t)h) * (size_t)BITS;
            for (int b = 0; b < BITS; b++)
                if (counter[base + (size_t)b] > th)
                    new_w |= (1U << (unsigned)b);
            W1[(size_t)k * (size_t)H + (size_t)h] = new_w;
            total_flips += __builtin_popcount(old_w ^ new_w);
        }
    }
    free(counter); free(class_cnt); free(idx);
    if (flips_out) *flips_out = total_flips;
}

/* ═══════════════════════════════════════════════════════════════════════
 * EXPORT / IMPORT helpers
 * ═══════════════════════════════════════════════════════════════════════ */
typedef struct {
    int       H;
    int       nc;
    uint32_t *W0;  /* [H × nc] */
    uint32_t *W1;  /* [N_CLASSES × H] */
} Bin32Model;

static void member_export(const uint32_t *W0, const uint32_t *W1,
                           int H, int nc, const char *dir, int mid) {
    char cmd[512], path[1024];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", dir);
    if (system(cmd) != 0) return;
    snprintf(path, sizeof(path), "%s/weights-%d.meta", dir, mid);
    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "%d\n%d %d\n%d %d\n", 2, H, nc, N_CLASSES, H);
    fclose(f);
    /* Save encoding config for IFC */
    snprintf(path, sizeof(path), "%s/config.txt", dir);
    f = fopen(path, "w");
    if (f) { fprintf(f, "encoding=%s\n", ki_enc_name_short((int)aa.enc_array[mid].type));
             fprintf(f, "width=%d\n", (int)aa.enc_array[mid].width); fclose(f); }
    snprintf(path, sizeof(path), "%s/W0-%d.bin", dir, mid);
    f = fopen(path, "wb");
    if (f) { fwrite(W0, sizeof(uint32_t), (size_t)H * (size_t)nc, f); fclose(f); }
    snprintf(path, sizeof(path), "%s/W1-%d.bin", dir, mid);
    f = fopen(path, "wb");
    if (f) { fwrite(W1, sizeof(uint32_t), (size_t)N_CLASSES * (size_t)H, f); fclose(f); }
}

static Bin32Model *member_load(const char *dir, int mid) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/weights-%d.meta", dir, mid);
    FILE *f = fopen(path, "r");
    if (!f) return NULL;
    int layers, h, nc, cls, h2;
    if (fscanf(f, "%d\n%d %d\n%d %d\n", &layers, &h, &nc, &cls, &h2) != 5) { fclose(f); return NULL; }
    fclose(f);
    if (layers != 2 || cls != N_CLASSES || h2 != h) return NULL;
    Bin32Model *m = (Bin32Model *)malloc(sizeof(Bin32Model));
    m->H = h; m->nc = nc;
    m->W0 = (uint32_t *)malloc((size_t)h * (size_t)nc * sizeof(uint32_t));
    m->W1 = (uint32_t *)malloc((size_t)cls * (size_t)h * sizeof(uint32_t));
    snprintf(path, sizeof(path), "%s/W0-%d.bin", dir, mid);
    f = fopen(path, "rb");
    if (!f || fread(m->W0, sizeof(uint32_t), (size_t)h * (size_t)nc, f) != (size_t)h * (size_t)nc) return NULL;
    fclose(f);
    snprintf(path, sizeof(path), "%s/W1-%d.bin", dir, mid);
    f = fopen(path, "rb");
    if (!f || fread(m->W1, sizeof(uint32_t), (size_t)cls * (size_t)h, f) != (size_t)cls * (size_t)h) return NULL;
    fclose(f);
    return m;
}

static void model_free(Bin32Model *m) { if (m) { free(m->W0); free(m->W1); free(m); } }

/* ═══════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════ */

/* ── Global args (defaults, überschrieben durch --flags) ─────── */
ki_Args aa = {
    .hidden   = 256,
    .epochs   = 3,
    .batchN   = 256,
    .lr       = 0.05f,
    .trainN   = 50000,
    .evalN    = 10000,
    .threadN  = 8,
    .seed     = 42,
    .hebbian_pct = 50,
};

int main(int argc, char *argv[]) {
    ki_parse_args(argc, argv);
    omp_set_num_threads(aa.threadN);

    int H = aa.hidden, epochs = aa.epochs;

    /* ── Build member list from enc_array ───────────────────────── */
    int n_enc = aa.enc_count;
    if (n_enc == 0) {
        n_enc = 1;
    }
    /* Init encoding LUTs for all active encodings */
    for (int i = 0; i < aa.enc_count && i < KI_ENC_MAX; i++)
        enc_lut_init_enc((int)aa.enc_array[i].type, (int)aa.enc_array[i].width);
    if (aa.enc_count == 0) {
        enc_lut_init_enc(KI_ENC_EXP, 8);
        enc_lut_init_enc(KI_ENC_RAW, 8);
    }

    int mem_nc[KI_ENC_MAX], mem_off[KI_ENC_MAX];
    int total_stride = 0;
    for (int i = 0; i < n_enc && i < KI_ENC_MAX; i++) {
        int w = (aa.enc_count > 0) ? (int)aa.enc_array[i].width : KI_ENC_WIDTH_DEFAULT;
        if (w < 1) w = KI_ENC_WIDTH_DEFAULT;
        mem_nc[i] = (KI_COLORS > 1 ? KI_NC : KI_NC) * w / 8;
        mem_off[i] = total_stride;
        total_stride += mem_nc[i];
    }
    (void)mem_off; /* used via offs[] in training section */

    /* ── IFC: --model ──────────────────────────────────────────── */
    if (aa.model[0]) {
        /* Count members from directory */
        Bin32Model *models[KI_ENC_MAX];
        int n_loaded = 0;
        for (int i = 0; i < KI_ENC_MAX; i++) {
            models[i] = member_load(aa.model, i);
            if (models[i]) n_loaded++; else break;
        }
        if (n_loaded == 0) { fprintf(stderr, "[FATAL] No members in %s\n", aa.model); return 1; }

        ki_Dataset data;
        if (ki_dataset_read(&data) != 0 || data.pixels != KI_PX) return 1;
        uint32_t *X_all = load_input(data.X_raw, data.num_images);
        int te = aa.evalN > data.num_images ? data.num_images : aa.evalN;
        int off = data.num_images - te;

        /* Collect per-member nc from loaded models */
        int ncs[KI_ENC_MAX], offs[KI_ENC_MAX];
        uint32_t *W0s[KI_ENC_MAX], *W1s[KI_ENC_MAX];
        int cum_off = 0;
        for (int i = 0; i < n_loaded; i++) {
            ncs[i] = models[i]->nc;
            offs[i] = cum_off;
            cum_off += ncs[i];
            W0s[i] = models[i]->W0;
            W1s[i] = models[i]->W1;
        }
        uint32_t *X_te = X_all + (size_t)off * (size_t)cum_off;
        uint8_t  *y_te = data.y + off;

        struct timeval tv0, tv1; gettimeofday(&tv0, NULL);
        float acc = accuracy_multi(X_te, y_te, te, (const uint32_t *const *)W0s, (const uint32_t *const *)W1s, ncs, offs, models[0]->H, n_loaded);
        gettimeofday(&tv1, NULL);
        int el = (int)((tv1.tv_sec-tv0.tv_sec)*1000 + (tv1.tv_usec-tv0.tv_usec)/1000);

        printf("\n══╡ RESULT ╞══  %d members  H=%d  Hebbian %s  Eval: %.1f%%  time=%dms\n",
               n_loaded, models[0]->H, H0_STR, acc, el);
        ki_report_show(0, 0, (int)(acc * (float)te / 100.0f + 0.5f), te, el, aa.threadN, 0, 0.0f);
        free(X_all); ki_dataset_free(&data);
        for (int i = 0; i < n_loaded; i++) model_free(models[i]);
        return 0;
    }

    /* ═══════════════════════════════════════════════════════════════
     * TRAINING MODE
     * ═══════════════════════════════════════════════════════════════ */
    int total_all = aa.trainN + aa.evalN;
    ki_Dataset data;
    if (ki_dataset_read(&data) != 0 || data.pixels != KI_PX) return 1;
    if (total_all > data.num_images) total_all = data.num_images;

    uint32_t *X_all = load_input(data.X_raw, total_all);
    int te = aa.evalN > total_all ? total_all : aa.evalN;
    int off = total_all - te;
    uint8_t *y_te = data.y + off;

    /* ── Allocate per-member W0 + W1 ────────────────────────────── */
    uint32_t *W0s[KI_ENC_MAX], *W1s[KI_ENC_MAX];
    int ncs[KI_ENC_MAX], offs[KI_ENC_MAX];
    int cum_off = 0;
    for (int i = 0; i < n_enc && i < KI_ENC_MAX; i++) {
        ncs[i] = mem_nc[i];
        offs[i] = cum_off;
        cum_off += ncs[i];
        W0s[i] = (uint32_t *)malloc((size_t)H * (size_t)ncs[i] * sizeof(uint32_t));
        W1s[i] = (uint32_t *)calloc((size_t)N_CLASSES * (size_t)H, sizeof(uint32_t));
        /* W0 init */
        w0_srandom((unsigned int)(aa.seed + (unsigned)i));
        for (size_t j = 0; j < (size_t)H * (size_t)ncs[i]; j++)
            W0s[i][j] = w0_random();
    }

    if (aa.dry_run) {
        printf("══╡ Hebbian %s ╞══  H=%d  Ep=%d  members=%d  stride=%d  %s\n",
               H0_STR, H, epochs, n_enc, cum_off, KI_COLORS > 1 ? "CIFAR" : "MNIST");
        for (int i = 0; i < n_enc && i < KI_ENC_MAX; i++)
            printf("  member[%d]: nc=%d off=%d  enc=%s w=%d col=%d\n", i,
                   ncs[i], offs[i],
                   aa.enc_count > 0 ? ki_enc_name_short((int)aa.enc_array[i].type) : "exp",
                   aa.enc_count > 0 ? (int)aa.enc_array[i].width : 8,
                   aa.enc_count > 0 ? (int)aa.enc_array[i].color : -1);
        return 0;
    }

    printf("══╡ Hebbian %s ╞══  H=%-4d  Ep=%-2d  members=%-2d  stride=%-4d  %s%s\n",
           H0_STR, H, epochs, n_enc, cum_off, KI_COLORS > 1 ? "CIFAR" : "MNIST",
           aa.out[0] ? "  export" : "");
    fflush(stdout);

    /* ── Train ─────────────────────────────────────────────────── */
    float best_acc = 0.0f;
    /* Extract per-member data for training */
    uint32_t **X_mems = (uint32_t **)malloc((size_t)n_enc * sizeof(uint32_t *));
    for (int m = 0; m < n_enc && m < KI_ENC_MAX; m++) {
        X_mems[m] = (uint32_t *)malloc((size_t)aa.trainN * (size_t)ncs[m] * sizeof(uint32_t));
        for (int s = 0; s < aa.trainN; s++)
            memcpy(X_mems[m] + (size_t)s * (size_t)ncs[m],
                   X_all + (size_t)s * (size_t)cum_off + offs[m],
                   (size_t)ncs[m] * sizeof(uint32_t));
    }
    struct timeval tv0; gettimeofday(&tv0, NULL);
    for (int ep = 0; ep < epochs; ep++) {
        int total_flips = 0;
        for (int m = 0; m < n_enc && m < KI_ENC_MAX; m++) {
            int flips = 0;
            hebbian_update(W1s[m], X_mems[m], data.y, aa.trainN, W0s[m], H, ncs[m],
                           (unsigned int)(aa.seed + (unsigned)m + (unsigned)ep),
                           &flips, 50);
            total_flips += flips;
        }
        float acc = accuracy_multi(X_all + (size_t)off * (size_t)cum_off, y_te, te,
                                   (const uint32_t *const *)W0s, (const uint32_t *const *)W1s,
                                   ncs, offs, H, n_enc);
        if (acc > best_acc) best_acc = acc;
        struct timeval tv1; gettimeofday(&tv1, NULL);
        int ms = (int)((tv1.tv_sec-tv0.tv_sec)*1000 + (tv1.tv_usec-tv0.tv_usec)/1000);
        printf("  Ep %2d/%d  evl=%.1f%%  best=%.1f%%  flips=%d  time=%dms\n",
               ep + 1, epochs, acc, best_acc, total_flips, ms);
    }
    printf("\n  Best eval: %.1f%%\n", best_acc);

    if (aa.out[0])
        for (int m = 0; m < n_enc && m < KI_ENC_MAX; m++)
            member_export(W0s[m], W1s[m], H, ncs[m], aa.out, m);

    for (int i = 0; i < n_enc && i < KI_ENC_MAX; i++) { free(W0s[i]); free(W1s[i]); free(X_mems[i]); }
    free(X_mems); free(X_all); ki_dataset_free(&data);
    return 0;
}
