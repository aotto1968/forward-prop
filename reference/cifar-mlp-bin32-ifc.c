/*
 * cifar-1/mlp-bin32-ifc.c — Bin32 Hebbian Inference (CIFAR-10)
 * ==============================================================
 *
 * Loads a trained binary model (W0.bin + W1.bin) and classifies
 * CIFAR-10 test data. Pure XNOR+popcount, no float, no matmul.
 *
 * Derived from otto-score-ifc/mlp-bin32-ifc.c
 * Changes: CIFAR-10 input dimension
 */
#include "ki-common.h"
#include "maj3.h"
#include <inttypes.h>

#define N_CLASSES KI_NCLASSES

/* ── Model ─────────────────────────────────────────────────────── */
typedef struct {
    int       H;
    int       nc;       /* input containers */
    uint32_t *W0;       /* [H × nc] frozen random projection */
    uint32_t *W1;       /* [N_CLASSES × H] Hebbian-trained classifier */
    int       h0_mode;  /* 0=XNOR, 1=XOR */
} Bin32Model;

#ifdef H0_XOR
#  define H0_ALWAYS_XOR 1
#else
#  define H0_ALWAYS_XOR 0
#endif

/* ── Model loader ──────────────────────────────────────────────── */
static Bin32Model *model_load(const char *dir) {
    char path[512];
    snprintf(path, sizeof(path), "%s/weights.meta", dir);

    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[FATAL] Cannot open %s\n", path);
        return NULL;
    }

    int n_layers, in_feat, out_feat, H;
    if (fscanf(f, "%d\n%d %d\n%d %d\n",
               &n_layers, &H, &in_feat, &out_feat, &H) != 5) {
        fprintf(stderr, "[FATAL] Bad format in %s\n", path);
        fclose(f); return NULL;
    }
    fclose(f);

    if (n_layers != 2 || out_feat != N_CLASSES) {
        fprintf(stderr, "[FATAL] Unexpected shape: %d layers, %d outputs\n",
                n_layers, out_feat);
        return NULL;
    }

    Bin32Model *m = (Bin32Model *)calloc(1, sizeof(Bin32Model));
    if (!m) return NULL;
    m->H = H;
    m->nc = in_feat;
    m->h0_mode = 0;

    size_t n0 = (size_t)H * (size_t)in_feat;
    size_t n1 = (size_t)N_CLASSES * (size_t)H;

    m->W0 = (uint32_t *)malloc(n0 * sizeof(uint32_t));
    m->W1 = (uint32_t *)malloc(n1 * sizeof(uint32_t));
    if (!m->W0 || !m->W1) {
        free(m->W0); free(m->W1); free(m);
        fprintf(stderr, "[FATAL] Out of memory\n"); return NULL;
    }

    snprintf(path, sizeof(path), "%s/W0.bin", dir);
    f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[FATAL] Cannot open %s\n", path); free(m->W0); free(m->W1); free(m); return NULL; }
    if (fread(m->W0, sizeof(uint32_t), n0, f) != n0) { fprintf(stderr, "[FATAL] Short read %s\n", path); fclose(f); free(m->W0); free(m->W1); free(m); return NULL; }
    fclose(f);

    snprintf(path, sizeof(path), "%s/W1.bin", dir);
    f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[FATAL] Cannot open %s\n", path); free(m->W0); free(m->W1); free(m); return NULL; }
    if (fread(m->W1, sizeof(uint32_t), n1, f) != n1) { fprintf(stderr, "[FATAL] Short read %s\n", path); fclose(f); free(m->W0); free(m->W1); free(m); return NULL; }
    fclose(f);

    printf("  Model: H=%d  nc=%d  dir=%s\n", H, in_feat, dir);
    return m;
}

static void model_free(Bin32Model *m) {
    if (!m) return;
    free(m->W0); free(m->W1); free(m);
}

/* ── H0 computation ────────────────────────────────────────────── */
static inline uint32_t h0_compute(const uint32_t *in, const uint32_t *row, int nc, int mode) {
    uint32_t match[2048]; /* >= NC=768 */
    if (mode == 0) {
        for (int c = 0; c < nc; c++)
            match[c] = ~(in[c] ^ row[c]);  /* XNOR */
    } else {
        for (int c = 0; c < nc; c++)
            match[c] = in[c] ^ row[c];     /* XOR */
    }
    return majority_tree(match, nc);
}

/* ── Accuracy ──────────────────────────────────────────────────── */
static float accuracy_pct(const uint32_t *X, const uint8_t *Y, int N,
                           const Bin32Model *m) {
    int ok = 0;
    int mode = H0_ALWAYS_XOR ? 1 : m->h0_mode;
    #pragma omp parallel reduction(+:ok)
    {
        uint32_t *h0 = (uint32_t *)malloc((size_t)m->H * sizeof(uint32_t));
        if (!h0) { ok = -1; }
        else {
            #pragma omp for schedule(static)
            for (int i = 0; i < N; i++) {
                const uint32_t *in = X + (size_t)i * (size_t)m->nc;
                for (int h = 0; h < m->H; h++) {
                    h0[h] = h0_compute(in, m->W0 + (size_t)h * (size_t)m->nc,
                                       m->nc, mode);
                }
                /* Popcount argmax */
                int best = 0;
                uint32_t best_cnt = 0;
                for (int k = 0; k < N_CLASSES; k++) {
                    uint32_t cnt = 0;
                    for (int h = 0; h < m->H; h++) {
                        cnt += (uint32_t)__builtin_popcount(h0[h] & m->W1[(size_t)k * (size_t)m->H + (size_t)h]);
                    }
                    if (k == 0 || cnt > best_cnt) {
                        best_cnt = cnt;
                        best = k;
                    }
                }
                if (best == (int)Y[i]) ok++;
            }
            free(h0);
        }
    }
    return 100.0f * (float)ok / (float)N;
}

/* ═══════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[]) {
    const char *model_dir = NULL;
    int evalN = 10000;
    int threadN = 8;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s --model DIR [options]\n", argv[0]);
            printf("  --model DIR   Model directory (weights.meta + W0.bin + W1.bin)\n");
            printf("  --evalN N     CIFAR-10 eval samples (default: 10000)\n");
            printf("  --threadN N   OpenMP threads (default: 8)\n");
            return 0;
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "--evalN") == 0 && i + 1 < argc) {
            evalN = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threadN") == 0 && i + 1 < argc) {
            threadN = atoi(argv[++i]);
            if (threadN < 1) threadN = 1;
        } else {
            fprintf(stderr, "[ERROR] Unknown argument: %s\nTry --help\n", argv[i]);
            return 1;
        }
    }

    if (!model_dir) {
        fprintf(stderr, "[FATAL] --model DIR is required\n");
        return 1;
    }

    omp_set_num_threads(threadN);

    Bin32Model *model = model_load(model_dir);
    if (!model) return 1;

    /* Load CIFAR-10 */
    ki_ImageData data;
    if (ki_cifar_read(&data) != 0) { model_free(model); return 1; }
    if (data.pixels != KI_PX) {
        fprintf(stderr, "[FATAL] Expected %d pixels, got %d\n", KI_PX, data.pixels);
        ki_cifar_free(&data); model_free(model); return 1;
    }

    int total_eval = evalN;
    int offset = data.num_images - total_eval;
    if (offset < 0) { offset = 0; total_eval = data.num_images; }

    uint32_t *X_all = load_input(data.X_raw, data.num_images);
    uint32_t *X_te  = X_all + (size_t)offset * (size_t)model->nc;
    uint8_t  *y_te  = data.y + offset;

    printf("\n══╡ INFERENCE ╞══════════════════════════════════════════════════\n");
    printf("  Evaluating %d CIFAR-10 samples\n", total_eval);
    fflush(stdout);

    struct timeval tv0, tv1;
    gettimeofday(&tv0, NULL);

    float acc = accuracy_pct(X_te, y_te, total_eval, model);

    gettimeofday(&tv1, NULL);
    int elapsed = (int)((tv1.tv_sec - tv0.tv_sec) * 1000
                      + (tv1.tv_usec - tv0.tv_usec) / 1000);

    printf("\n══╡ RESULT ╞══════════════════════════════════════════════════════\n");
    printf("  Model:   H=%d  nc=%d  Bin32 Hebbian (CIFAR-10)\n", model->H, model->nc);
    printf("  Eval:    %.1f%%  (%d/%d)\n",
           acc, (int)(acc * (float)total_eval / 100.0f + 0.5f), total_eval);
    printf("  Time:    %dms  (%.1f µs/sample)\n",
           elapsed, (double)elapsed * 1000.0 / (double)total_eval);

    ki_report_show(0, 0,
                   (int)(acc * (float)total_eval / 100.0f + 0.5f), total_eval,
                   elapsed, threadN, 0, 0.0f);

    free(X_all);
    ki_cifar_free(&data);
    model_free(model);
    return 0;
}
