/*
 * cifar-1/mlp-bin32-trn-w1-hebbian.c — Majority + Hebbian (CIFAR-10)
 * ===================================================================
 *
 * Reference implementation: bitwise Hebbian training for CIFAR-10.
 * Uses XNOR/XOR + MAJ3 + popcount — NO floating point, NO matmul.
 *
 * Note: This trainer does NOT converge well (typically ~40-45% on CIFAR-10).
 * Included as reference for comparison with Otto Score and AdamW.
 *
 * Derived from otto-score-ifc/mlp-bin32-trn-w1-hebbian.c
 * Changes: CIFAR-10 input dimension (3072 px → 768 containers)
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#define KI_COMMON_LOAD_INPUT
#include "ki-common.h"
#include "w0_random.h"
#include "maj3.h"

/* ═══════════════════════════════════════════════════════════════════════
 * PACKING CONFIG — choose via -DPACKING=N
 * ═══════════════════════════════════════════════════════════════════════
 *   CIFAR-10: 3072 pixels
 *   PACKING=1  → NC=768   (4 px/cont)  [default]
 *   PACKING=4  → NC=3072  (1 px/cont)
 */

#ifndef PACKING
#  error "PACKING must be 1 (NC=768) or 4 (NC=3072)"
#elif PACKING != 1 && PACKING != 4
#  error "PACKING must be 1 or 4"
#elif PACKING == 4
   /* NOTE: NC comes from ki-local.h (=KI_NC=768). PACKING=4 not supported. */
#  error "PACKING=4 not supported (ki-local.h fixes NC=KI_NC=768)"
#endif

/* ── INVARIANT: NC (from ki-local.h) covers all 3072 pixels ──────
 *   KI_NC=1024, KI_PACK=3: 1024 × 3 = 3072
 */
_Static_assert(
    KI_NC * (KI_PX / KI_NC) == KI_PX,
    "KI_NC * KI_PACK != KI_PX (NC does not cover all pixels)"
);

#define BUF_MAJ   2048       /* majority_tree internal buffer (>= NC=768) */
#define H0_BUF    4096       /* h0 buffer (for H up to 4096) */

#define INPUT_PX  3072
#define INPUT_NC  NC
#define N_CLASSES KI_NCLASSES

#define H0_MODE_DEFAULT 0    /* 0=XNOR, 1=XOR */
#define H0_STR_DEFAULT  "XNOR"

#ifdef H0_XOR
#  define H0_STR   "XOR"
#  define H0_MODE  1
#else
#  define H0_STR   H0_STR_DEFAULT
#  define H0_MODE  H0_MODE_DEFAULT
#endif

/* ── load_input: uint8_t Pixel → uint32_t Container ──────────────
 *  (own version, guard KI_COMMON_LOAD_INPUT prevents ki-common.h
 *   from providing the default)
 */
static uint32_t *load_input(const uint8_t *X_raw, int n_samples) {
    uint32_t *Xb = (uint32_t *)malloc((size_t)n_samples * (size_t)INPUT_NC * sizeof(uint32_t));
#if PACKING == 1
    /* 4 px/cont: p0|p1<<8|p2<<16|p3<<24 */
    for (int s = 0; s < n_samples; s++) {
        uint32_t *row = Xb + (size_t)s * INPUT_NC;
        for (int c = 0; c < INPUT_NC; c++) {
            uint32_t val = 0;
            for (int k = 0; k < 4; k++) {
                size_t p = (size_t)s * (size_t)INPUT_PX + (size_t)c * 4 + (size_t)k;
                val |= ((uint32_t)X_raw[p] & 0xFFU) << (unsigned)(k * 8);
            }
            row[c] = val;
        }
    }
#elif PACKING == 4
    /* 1 px/cont: p0 * 0x01010101 */
    for (int s = 0; s < n_samples; s++) {
        uint32_t *row = Xb + (size_t)s * INPUT_NC;
        for (size_t p = 0; p < INPUT_PX; p++) {
            size_t off = (size_t)s * (size_t)INPUT_PX + p;
            row[p] = ((uint32_t)X_raw[off] & 0xFFU) * 0x01010101U;
        }
    }
#else
#  error "PACKING must be 1 or 4"
#endif
    return Xb;
}

/* ── H0: majority_tree over match array ─────────────────────────── */
static inline uint32_t h0_compute(const uint32_t *in, const uint32_t *row, int nc) {
    uint32_t match[2048]; /* >= NC=768 for PACKING=1 */
    if (H0_MODE == 0) {
        for (int c = 0; c < nc; c++)
            match[c] = ~(in[c] ^ row[c]);  /* XNOR: agrees are 1 */
    } else {
        for (int c = 0; c < nc; c++)
            match[c] = in[c] ^ row[c];     /* XOR: differences are 1 */
    }
    return majority_tree(match, nc);
}

/* ── Forward: compute h0 for all H neurons ─────────────────────── */
static void forward(const uint32_t *in, const uint32_t *W0,
                     uint32_t *h0, int H, int nc) {
    for (int h = 0; h < H; h++) {
        h0[h] = h0_compute(in, W0 + (size_t)h * (size_t)nc, nc);
    }
}

/* ── Popcount argmax ────────────────────────────────────────────── */
static int popcount_argmax(const uint32_t *h0, const uint32_t *W1,
                            int H) {
    int best = 0;
    uint32_t best_cnt = 0;
    for (int k = 0; k < N_CLASSES; k++) {
        const uint32_t *row = W1 + (size_t)k * (size_t)H;
        uint32_t cnt = 0;
        for (int h = 0; h < H; h++) {
            cnt += (uint32_t)__builtin_popcount(h0[h] & row[h]);
        }
        if (k == 0 || cnt > best_cnt) {
            best_cnt = cnt;
            best = k;
        }
    }
    return best;
}

/* ── Accuracy ──────────────────────────────────────────────────── */
static float accuracy_pct(const uint32_t *X, const uint8_t *Y, int N,
                           const uint32_t *W0, const uint32_t *W1,
                           int H, int nc) {
    int ok = 0;
    #pragma omp parallel reduction(+:ok)
    {
        uint32_t *h0 = (uint32_t *)malloc((size_t)H0_BUF * sizeof(uint32_t));
        if (!h0) { ok = -1; }
        else {
            #pragma omp for schedule(static)
            for (int i = 0; i < N; i++) {
                const uint32_t *in = X + (size_t)i * (size_t)nc;
                forward(in, W0, h0, H, nc);
                int pred = popcount_argmax(h0, W1, H);
                if (pred == (int)Y[i]) ok++;
            }
            free(h0);
        }
    }
    return 100.0f * (float)ok / (float)N;
}

/* ── Export weights ────────────────────────────────────────────── */
static int export_weights(const uint32_t *W0, const uint32_t *W1,
                           int H, int nc, const char *dir) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", dir);
    if (system(cmd) != 0) return -1;

    char path[512];
    snprintf(path, sizeof(path), "%s/weights.meta", dir);
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "%d\n%d %d\n%d %d\n", 2, H, nc, N_CLASSES, H);
    fclose(f);

    snprintf(path, sizeof(path), "%s/W0.bin", dir);
    f = fopen(path, "wb");
    if (!f) return -1;
    fwrite(W0, sizeof(uint32_t), (size_t)H * (size_t)nc, f);
    fclose(f);

    snprintf(path, sizeof(path), "%s/W1.bin", dir);
    f = fopen(path, "wb");
    if (!f) return -1;
    fwrite(W1, sizeof(uint32_t), (size_t)N_CLASSES * (size_t)H, f);
    fclose(f);

    printf("  Exported model → %s/\n", dir);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[]) {
    int H = 64, epochs = 3;
    int Ntrn = 50000, Nevl = 10000;
    int thrs = 8;
    char out_dir[256] = "";
    unsigned int seed = 42;
    int dry_run = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --hiddenN N     Hidden neurons (default: 64)\n");
            printf("  --epochsN N     Training epochs (default: 3)\n");
            printf("  --trainN N      Training samples (default: 50000)\n");
            printf("  --evalN N       Eval samples (default: 10000)\n");
            printf("  --out DIR       Export directory\n");
            printf("  --seed N        Random seed (default: 42)\n");
            printf("  --threadN N     OpenMP threads (default: 8)\n");
            printf("  --debug         Verbose output\n");
            printf("  --dry-run       Print architecture and exit\n");
            return 0;
        } else if (strcmp(argv[i], "--hiddenN") == 0 && i + 1 < argc) {
            H = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--epochsN") == 0 && i + 1 < argc) {
            epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--trainN") == 0 && i + 1 < argc) {
            Ntrn = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--evalN") == 0 && i + 1 < argc) {
            Nevl = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            strncpy(out_dir, argv[++i], sizeof(out_dir) - 1);
            out_dir[sizeof(out_dir) - 1] = '\0';
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (unsigned int)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threadN") == 0 && i + 1 < argc) {
            thrs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--dry-run") == 0) {
            dry_run = 1;
        } else {
            fprintf(stderr, "[ERROR] Unknown argument: %s\nTry --help\n", argv[i]);
            return 1;
        }
    }

    omp_set_num_threads(thrs);

    if (dry_run) {
        printf("══╡ Hebbian Trainer (CIFAR-10) ╞══\n");
        printf("  Hidden:   %d\n", H);
        printf("  Input:    %d px → %d containers (%s)\n", INPUT_PX, INPUT_NC,
               PACKING == 1 ? "4px/cont" : "1px/cont");
        printf("  Mode:     %s\n", H0_STR);
        return 0;
    }

    if (H > 4096) { fprintf(stderr, "[ERROR] H=%d > 4096\n", H); return 1; }

    printf("══╡ CIFAR-10 Hebbian Trainer ╞══  H=%-4d  Ep=%-2d  NC=%-3d  %s  packing=%d\n",
           H, epochs, INPUT_NC, H0_STR, PACKING);

    /* ── Load CIFAR-10 ─────────────────────────────────────────── */
    ki_ImageData data;
    if (ki_cifar_read(&data) != 0) return 1;
    if (data.pixels != INPUT_PX) {
        fprintf(stderr, "[FATAL] Expected %d pixels, got %d\n", INPUT_PX, data.pixels);
        ki_cifar_free(&data); return 1;
    }

    /* Pack input */
    uint32_t *Xtrn = load_input(data.X_raw, Ntrn);
    uint32_t *Xevl = load_input(data.X_raw + (size_t)Ntrn * (size_t)INPUT_PX, Nevl);
    uint8_t *ytrn = data.y;
    uint8_t *yevl = data.y + Ntrn;

    printf("  Train: %d  Eval: %d\n", Ntrn, Nevl);

    /* ── Create W0 (frozen random projection) ──────────────────── */
    size_t w0_n = (size_t)H * (size_t)INPUT_NC;
    uint32_t *W0 = (uint32_t *)malloc(w0_n * sizeof(uint32_t));
    srand(seed);
    for (size_t i = 0; i < w0_n; i++) W0[i] = w0_random();

    /* ── Create W1 (Hebbian-trained classifier) ───────────────── */
    size_t w1_n = (size_t)N_CLASSES * (size_t)H;
    uint32_t *W1 = (uint32_t *)calloc(w1_n, sizeof(uint32_t)); /* zero init */

    /* ── Training ──────────────────────────────────────────────── */
    struct timeval tv_start, tv_now;
    gettimeofday(&tv_start, NULL);

    float best_eval = 0.0f;
    uint32_t *best_W1 = (uint32_t *)malloc(w1_n * sizeof(uint32_t));

    for (int ep = 0; ep < epochs; ep++) {
        /* Shuffle */
        int *idx = (int *)malloc((size_t)Ntrn * sizeof(int));
        for (int i = 0; i < Ntrn; i++) idx[i] = i;
        ki_shuffle(idx, Ntrn);

        int ep_ok = 0;

        for (int si = 0; si < Ntrn; si++) {
            const uint32_t *in = Xtrn + (size_t)idx[si] * (size_t)INPUT_NC;
            uint8_t label = ytrn[idx[si]];

            uint32_t h0[H0_BUF];
            forward(in, W0, h0, H, INPUT_NC);

            /* Popcount prediction */
            int pred = popcount_argmax(h0, W1, H);
            if (pred == (int)label) ep_ok++;

            /* Error-driven Hebbian update */
            if (pred != (int)label) {
                uint32_t *w1_label = W1 + (size_t)label * (size_t)H;
                uint32_t *w1_pred  = W1 + (size_t)pred * (size_t)H;
                for (int h = 0; h < H; h++) {
                    /* Hebb: strengthen agreement with correct label */
                    uint32_t missing = h0[h] & ~w1_label[h];
                    if (missing) {
                        uint32_t bit = missing & (uint32_t)(-(int32_t)missing);
                        w1_label[h] |= bit;
                    }
                    /* Weaken agreement with wrong prediction */
                    uint32_t wrong_agree = h0[h] & w1_pred[h];
                    if (wrong_agree) {
                        uint32_t bit = wrong_agree & (uint32_t)(-(int32_t)wrong_agree);
                        w1_pred[h] ^= bit;
                    }
                }
            }
        }

        free(idx);

        float eval_acc = accuracy_pct(Xevl, yevl, Nevl, W0, W1, H, INPUT_NC);
        if (eval_acc > best_eval) {
            best_eval = eval_acc;
            memcpy(best_W1, W1, w1_n * sizeof(uint32_t));
        }

        gettimeofday(&tv_now, NULL);
        double elapsed = (double)(tv_now.tv_sec - tv_start.tv_sec)
                       + (double)(tv_now.tv_usec - tv_start.tv_usec) / 1e6;
        printf("  Ep %2d/%d  train=%.1f%%  eval=%.1f%%  time=%.0fs\n",
               ep + 1, epochs, (double)ep_ok * 100.0 / (double)Ntrn,
               (double)eval_acc, elapsed);
    }

    /* ── Export best model ─────────────────────────────────────── */
    memcpy(W1, best_W1, w1_n * sizeof(uint32_t));
    char def_dir[256];
    if (out_dir[0] != '\0') {
        snprintf(def_dir, sizeof(def_dir), "%s", out_dir);
    } else {
        snprintf(def_dir, sizeof(def_dir), KI_MODEL_DIR "/hebbian-h%d-b1-e%d", H, epochs);
    }
    export_weights(W0, W1, H, INPUT_NC, def_dir);

    /* ── Report ────────────────────────────────────────────────── */
    gettimeofday(&tv_now, NULL);
    int elapsed_ms = (int)((tv_now.tv_sec - tv_start.tv_sec) * 1000
                         + (tv_now.tv_usec - tv_start.tv_usec) / 1000);
    printf("\n══╡ RESULT ╞══════════════════════════════════════════════════════\n");
    printf("  Best eval: %.1f%%\n", (double)best_eval);
    printf("  Time:      %dms\n", elapsed_ms);

    /* ── Cleanup ───────────────────────────────────────────────── */
    free(Xtrn); free(Xevl);
    free(W0); free(W1); free(best_W1);
    ki_cifar_free(&data);
    return 0;
}
