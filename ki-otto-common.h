/*
 * otto-score-ifc/ki-otto-common.h — Otto Score shared infrastructure
 * ===================================================================
 *
 * Provides CLI parsing, MNIST loading, memory helpers, input packing,
 * batch correction, and report output for Otto Score programs.
 *
 * This is a copy of ki-w2/ki-common.h, renamed to coexist with
 * otto-score-ifc/ki-common.h (which has float32/AdamW/matmul helpers).
 */
#ifndef KI_OTTO_COMMON_H
#define KI_OTTO_COMMON_H

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <zlib.h>
#include <omp.h>

/* ═══════════════════════════════════════════════════════════════════════
 * w0_random.h — splitmix64 PRNG (included here for ki_rand_fill)
 * ═══════════════════════════════════════════════════════════════════════ */
#include "w0_random.h"

/* ═══════════════════════════════════════════════════════════════════════
 * CONSTANTS
 * ═══════════════════════════════════════════════════════════════════════ */

#define KI_PX       784
#define KI_NCLASSES 10
#ifndef KI_NC
#define KI_NC       196     /* Packed containers per image: 784/4 */
#endif
#ifndef KI_PACK
#define KI_PACK     (784 / KI_NC)  /* Pixels per container */
#endif

#ifndef OT_PRECISION
#define OT_PRECISION 10
#endif

#ifndef MAX
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#endif

/* Target-Index: [10][H][32] — class × neuron × bit */
#ifndef TGT_IDX
#define TGT_IDX(k, h, b, H) \
    ((size_t)(k) * (size_t)(H) * 32 + (size_t)(h) * 32 + (size_t)(b))
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * ARGS — CLI Parameters (Otto Score only)
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    int    hidden;          /* Hidden neurons (--hiddenN, default: 64) */
    int    epochs;          /* Iterations (--epochsN, default: 1) */
    int    batchN;          /* Mini-batch size (--batchN, default: 64) */
    int    trainN;          /* Training samples (--trainN, default: 50000) */
    int    evalN;           /* Eval samples (--evalN, default: 10000) */
    int    dry_run;         /* --dry-run: print arch and exit */
    int    debug;           /* --debug: verbose output */
    unsigned int seed;      /* Random seed (--seed, default: 42) */
    char   out[256];        /* --out DIR: export directory */
    float  lr;              /* Step size (--lr, default: 0.05) */
    int    lr_step;         /* round(a.lr * (1<<OT_PRECISION)) */
    int    threadN;         /* OpenMP threads (--threadN, default: 8) */
    int    debug_h0;        /* --debug-h0: per-neuron debug */
    int    shuffle;         /* --shuffle: randomize train/eval split */
    int    warmup_epochs;   /* --warmup N: linear warmup epochs (default: 2) */
    int    step_const;      /* --step-const: use constant step (no cosine decay) */
    int    ensembleN;       /* --ensembleN N: train N W0s and vote */
    char   random_file[256]; /* --random-file PATH: true random source */
    char   ensemble_seed[16]; /* --ensemble-seed const|incr */
} ki_Args;

static inline ki_Args ki_args_defaults(void) {
    ki_Args a;
    a.hidden        = 64;
    a.epochs        = 1;
    a.batchN        = 64;
    a.trainN        = 50000;
    a.evalN         = 10000;
    a.dry_run       = 0;
    a.debug         = 0;
    a.seed          = 42;
    a.out[0]        = '\0';
    a.lr            = 0.05f;
    a.threadN       = 8;
    a.debug_h0      = 0;
    a.shuffle       = 0;
    a.warmup_epochs = 2;
    a.step_const    = 0;
    a.ensembleN     = 1;
    a.random_file[0] = '\0';
    a.ensemble_seed[0] = '\0';
    return a;
}

/* ── Parse CLI ─────────────────────────────────────────────────── */
static inline void ki_parse_args(int argc, char *argv[], ki_Args *a) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --hiddenN N       Hidden neurons (default: 64)\n");
            printf("  --epochsN N       Iterations (default: 1)\n");
            printf("  --ensembleN N     Train N W0s and majority vote (default: 1)\n");
            printf("  --batchN N        Mini-batch size (default: 64)\n");
            printf("  --trainN N        Training samples (default: 50000)\n");
            printf("  --evalN N         Eval samples (default: 10000)\n");
            printf("  --quick           5000 train / 2000 eval\n");
            printf("  --qq              5000 train / 2000 eval / 3 epochs\n");
            printf("  --lr FLOAT        Step size (default: 0.05)\n");
            printf("  --threadN N       OpenMP threads (default: 8)\n");
            printf("  --warmup N        Linear warmup epochs (default: 2, 0=off)\n");
            printf("  --step-const      Use constant step (no cosine decay, old behavior)\n");
            printf("  --seed N          Random seed (default: 42)\n");
            printf("  --out DIR         Export directory\n");
            printf("  --dry-run         Print architecture and exit\n");
            printf("  --debug           Verbose output\n");
            printf("  --debug-h0        Per-neuron debug\n");
            printf("  --shuffle         Shuffle data before train/eval split\n");
            printf("  --random-file PATH-TO-RANDOM-FILE\n");
            printf("                    Use true random data from file instead of PRNG\n");
            printf("  --ensemble-seed const|incr|broken-31|fix-gnu|fix-splitmix\n");
            printf("                    Per-member seeding: const=same, incr=seed+m\n");
            exit(0);
        } else if (strcmp(argv[i], "--dry-run") == 0) {
            a->dry_run = 1;
        } else if (strcmp(argv[i], "--debug") == 0) {
            a->debug = 1;
        } else if (strcmp(argv[i], "--quick") == 0) {
            a->trainN = 5000; a->evalN = 2000;
        } else if (strcmp(argv[i], "--qq") == 0) {
            a->trainN = 5000; a->evalN = 2000; a->epochs = 3;
        } else if (strcmp(argv[i], "--hiddenN") == 0 && i + 1 < argc) {
            a->hidden = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--epochsN") == 0 && i + 1 < argc) {
            a->epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--batchN") == 0 && i + 1 < argc) {
            a->batchN = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--trainN") == 0 && i + 1 < argc) {
            a->trainN = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--evalN") == 0 && i + 1 < argc) {
            a->evalN = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            a->lr = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--threadN") == 0 && i + 1 < argc) {
            a->threadN = atoi(argv[++i]);
            if (a->threadN < 1) a->threadN = 1;
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            a->warmup_epochs = atoi(argv[++i]);
            if (a->warmup_epochs < 0) a->warmup_epochs = 0;
        } else if (strcmp(argv[i], "--step-const") == 0) {
            a->step_const = 1;
        } else if (strcmp(argv[i], "--iter") == 0 && i + 1 < argc) {
            i++;  /* ignored (BV32 compatibility) */
        } else if (strcmp(argv[i], "--lr-min") == 0 && i + 1 < argc) {
            i++;  /* ignored (scheduler compatibility) */
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            a->seed = (unsigned int)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            strncpy(a->out, argv[++i], sizeof(a->out) - 1);
            a->out[sizeof(a->out) - 1] = '\0';
        } else if (strcmp(argv[i], "--debug-h0") == 0) {
            a->debug_h0 = 1;
        } else if (strcmp(argv[i], "--shuffle") == 0) {
            a->shuffle = 1;
        } else if (strcmp(argv[i], "--ensembleN") == 0 && i + 1 < argc) {
            a->ensembleN = atoi(argv[++i]);
            if (a->ensembleN < 1) a->ensembleN = 1;
        } else if (strcmp(argv[i], "--random-file") == 0 && i + 1 < argc) {
            strncpy(a->random_file, argv[++i], sizeof(a->random_file) - 1);
            a->random_file[sizeof(a->random_file) - 1] = '\0';
        } else if (strcmp(argv[i], "--ensemble-seed") == 0 && i + 1 < argc) {
            const char *val = argv[++i];
            strncpy(a->ensemble_seed, val, sizeof(a->ensemble_seed) - 1);
            a->ensemble_seed[sizeof(a->ensemble_seed) - 1] = '\0';
            if (strcmp(val, "broken-31") != 0 &&
                strcmp(val, "fix-gnu") != 0 &&
                strcmp(val, "fix-splitmix") != 0 &&
                strcmp(val, "const") != 0 &&
                strcmp(val, "incr") != 0) {
                /* Not a keyword → treat as filename. Validate existence. */
                if (access(val, R_OK) != 0) {
                    fprintf(stderr, "[ERROR] --ensemble-seed: '%s' is not "
                            "'broken-31', 'fix-gnu', 'fix-splitmix', "
                            "'const', 'incr', or an existing file\n", val);
                    exit(1);
                }
                /* Set random_file too (shared by mlp-otto-score.c) */
                strncpy(a->random_file, val, sizeof(a->random_file) - 1);
                a->random_file[sizeof(a->random_file) - 1] = '\0';
            }
        } else {
            fprintf(stderr, "[ERROR] Unknown argument: %s\nTry --help\n", argv[i]);
            exit(1);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * LR SCHEDULE — Cosine Decay + Linear Warmup
 * ═══════════════════════════════════════════════════════════════════════
 * Returns a float multiplier (0..1) for the base LR.
 *
 *   warmup > 0 : linear increase 0→1 over first `warmup` epochs
 *   after warmup: cosine decay from base_lr down to lr_min
 *   no_decay=1: always returns base_lr (identity)
 */
static inline float ki_lr_schedule(int epoch, int total_epochs, int warmup,
                                    float base_lr, float lr_min, int no_decay) {
    if (no_decay) return base_lr;
    if (epoch < warmup)
        return base_lr * (float)(epoch + 1) / (float)warmup;
    int decay_epochs = total_epochs - warmup;
    if (decay_epochs <= 0) return base_lr;
    float progress = (float)(epoch - warmup) / (float)decay_epochs;
    float cosine = (float)(1.0 + cos(progress * 3.14159265358979323846)) / 2.0f;
    return lr_min + (base_lr - lr_min) * cosine;
}

/* ═══════════════════════════════════════════════════════════════════════
 * OT_PRECISION — Skalierungshilfe: in × F + 0.5-Rounding
 * ═══════════════════════════════════════════════════════════════════════
 * F = (1<<OT_PRECISION).  Alle logit/log(p)-Werte werden mit F skaliert
 * in int32/int64 gespeichert.  ot_precision() rundet kaufmännisch.
 */
static inline double ot_precision(double in) {
    return in * (double)(1 << OT_PRECISION) + (in >= 0 ? 0.5 : -0.5);
}

/* ═══════════════════════════════════════════════════════════════════════
 * SETUP PRINT — gemeinsame Otto-Score-Info
 * ═══════════════════════════════════════════════════════════════════════
 * Gibt OMP-Threads, Train/Eval, OT_PRECISION, lr/step/mode und warmup aus.
 * Jeder Trainer ruft das auf und ergänzt dann eigene Details.
 */
static inline void ki_print_setup(const ki_Args *a,
                                   int total_train, int total_eval) {
    int step = (int)(a->lr * (1 << OT_PRECISION) + 0.5f);
    printf("  OMP:         %d threads\n", a->threadN);
    printf("  Train/Eval:  %d / %d samples\n", total_train, total_eval);
    printf("  Precision:   OT_PRECISION=%d (F=%d)\n",
           OT_PRECISION, 1 << OT_PRECISION);
    printf("  lr=%.4f  step=%d  mode=%s",
           (double)a->lr, step,
           a->step_const ? "const" : "cosine");
    if (a->warmup_epochs > 0)
        printf("  warmup=%d", a->warmup_epochs);
    printf("  batch=%d", a->batchN);
    printf("\n");
}

/* ═══════════════════════════════════════════════════════════════════════
 * CORRECTION — atomare Target-Updates (über alle Trainer identisch)
 * ═══════════════════════════════════════════════════════════════════════
 * Für ein fehlklassifiziertes Sample (true_k ≠ pred):
 *   target[true_k][h][b] += step   für jedes aktive Bit
 *   target[pred][h][b]   -= step
 *
 * h0_s:    H0-Werte des Samples (precomputiert oder frisch berechnet)
 * target:  Zeiger auf das Target (mit Member-Offset für Ensemble)
 * H:       Anzahl Neuronen
 * step:    Korrektur-Schritt
 */
static inline void ki_correct_target(int32_t *target, const uint32_t *h0_s,
                                      int H, int true_k, int pred, int step) {
    for (int h = 0; h < H; h++) {
        uint32_t val = h0_s[h];
        for (int b = 0; b < 32; b++) {
            if (val & (1U << (unsigned)b)) {
                #pragma omp atomic
                target[TGT_IDX(true_k, h, b, H)] += step;
                #pragma omp atomic
                target[TGT_IDX(pred, h, b, H)] -= step;
            }
        }
    }
}


/* ═══════════════════════════════════════════════════════════════════════
 * MEMORY HELPERS
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void *ki_xmalloc(size_t size) {
    if (size == 0) return NULL;
    void *ptr = malloc(size);
    if (!ptr) { fprintf(stderr, "[FATAL] ki_xmalloc(%zu) failed\n", size); exit(1); }
    return ptr;
}

static inline void *ki_xcalloc(size_t nmemb, size_t size) {
    if (nmemb == 0 || size == 0) return NULL;
    void *ptr = calloc(nmemb, size);
    if (!ptr) { fprintf(stderr, "[FATAL] ki_xcalloc(%zu, %zu) failed\n", nmemb, size); exit(1); }
    return ptr;
}

/* ═══════════════════════════════════════════════════════════════════════
 * BATCH CORRECTION — parallel + deterministisch
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Phase 1 (parallel):  Jeder Thread schreibt in seine eigene delta_cache.
 * Phase 2 (sequential): Alle delta_caches werden summiert → target.
 *
 *   int nt = ki_omp_nthreads();
 *   int32_t **dc = ki_cache_alloc(nt, tgt_sz);
 *   #pragma omp parallel for reduction(+:corr) schedule(static)
 *   for (int s = 0; s < N; s++) {
 *       int tid = omp_get_thread_num();
 *       if (pred != true_k)
 *           for (h,b) if bit active:
 *               dc[tid][TGT_IDX(true_k, h, b, H)] += step;
 *               dc[tid][TGT_IDX(pred, h, b, H)]   -= step;
 *   }
 *   ki_cache_apply_free(dc, nt, target, tgt_sz);
 */
static inline int ki_omp_nthreads(void) {
    int n = 1;
    #pragma omp parallel
    #pragma omp single
    n = omp_get_num_threads();
    return n;
}

static inline int32_t **ki_cache_alloc(int n_threads, size_t tgt_sz) {
    int32_t **cache = (int32_t **)malloc((size_t)n_threads * sizeof(int32_t *));
    if (!cache) { fprintf(stderr, "[FATAL] ki_cache_alloc(%d) failed\n", n_threads); exit(1); }
    for (int t = 0; t < n_threads; t++)
        cache[t] = (int32_t *)ki_xcalloc(tgt_sz, sizeof(int32_t));
    return cache;
}

static inline void ki_cache_apply_free(int32_t **cache, int n_threads,
                                        int32_t *target, size_t tgt_sz) {
    for (int t = 0; t < n_threads; t++) {
        int32_t *ct = cache[t];
        for (size_t i = 0; i < tgt_sz; i++)
            target[i] += ct[i];
        free(ct);
    }
    free(cache);
}

/* ═══════════════════════════════════════════════════════════════════════
 * BATCH CORRECTION — parallel + deterministisch via Mini-Batches
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Phasen pro Batch:
 *   1. Parallel: Scores aus target berechnen, Deltas in Thread-Cache
 *   2. Sequentiell: Deltas auf target anwenden → nächster Batch sieht Änderung
 *
 * target:     Ziel-Target (mit Offset für Ensemble)
 * H:          Anzahl Neuronen
 * class_offset: Offset pro Klasse
 * h0_all:     Vorberechnete H0-Werte [N × H]
 * y:          Labels
 * N:          Anzahl Trainings-Samples
 * batch_size: Mini-Batch-Grösse (--batchN, default 64)
 * step:       Korrektur-Schritt
 * tgt_sz:     Grösse des Target-Arrays (H × 10 × 32)
 *
 * Returns:    Anzahl Korrekturen
 */
static inline int ki_batch_correct(int32_t *target, int H,
                                    const int64_t *class_offset,
                                    const uint32_t *h0_all,
                                    const uint8_t *y,
                                    int N, int batch_size,
                                    int step, size_t tgt_sz) {
    int n_threads = ki_omp_nthreads();
    int32_t **dc = ki_cache_alloc(n_threads, tgt_sz);
    int corrections = 0;

    for (int b_start = 0; b_start < N; b_start += batch_size) {
        int b_end = b_start + batch_size;
        if (b_end > N) b_end = N;
        int batch_corr = 0;

        #pragma omp parallel for reduction(+:batch_corr) schedule(static)
        for (int s = b_start; s < b_end; s++) {
            int tid = omp_get_thread_num();
            int true_k = (int)y[s];
            const uint32_t *h0_s = h0_all + (size_t)s * (size_t)H;
            int64_t sc[10];
            for (int k = 0; k < 10; k++) sc[k] = class_offset[k];
            for (int h = 0; h < H; h++) {
                uint32_t val = h0_s[h];
                for (int k = 0; k < 10; k++)
                    for (int b = 0; b < 32; b++)
                        if (val & (1U << (unsigned)b))
                            sc[k] += target[TGT_IDX(k, h, b, H)];
            }
            int pred = 0;
            for (int k = 1; k < 10; k++)
                if (sc[k] > sc[pred]) pred = k;

            if (pred != true_k) {
                batch_corr++;
                for (int h = 0; h < H; h++) {
                    uint32_t val = h0_s[h];
                    for (int b = 0; b < 32; b++) {
                        if (val & (1U << (unsigned)b)) {
                            dc[tid][TGT_IDX(true_k, h, b, H)] += step;
                            dc[tid][TGT_IDX(pred, h, b, H)] -= step;
                        }
                    }
                }
            }
        }

        /* Apply + clear cache */
        for (int t = 0; t < n_threads; t++) {
            int32_t *ct = dc[t];
            for (size_t i = 0; i < tgt_sz; i++) {
                int d = ct[i];
                if (d != 0) target[i] += d;
            }
            memset(ct, 0, tgt_sz * sizeof(int32_t));
        }
        corrections += batch_corr;
    }

    for (int t = 0; t < n_threads; t++) free(dc[t]);
    free(dc);
    return corrections;
}


/* ═══════════════════════════════════════════════════════════════════════
 * RANDOM SOURCE — true random file or splitmix64 PRNG
 * ═══════════════════════════════════════════════════════════════════════
 *
 * If --random-file is set, reads uint32_t values sequentially from a
 * raw binary file (e.g. from random.org). Falls back to w0_random().
 *
 * The file is just raw bytes — each fread reads 4 bytes → one uint32_t.
 * File position is tracked globally per call, NOT per-thread.
 * Only the main thread calls this during init (before OpenMP).
 */

/* ── Open random file once (cached) ────────────────────────────── */
static FILE *ki_rand_fp = NULL;

static inline void ki_rand_open(const char *path) {
    if (!path || !path[0]) return;
    if (ki_rand_fp) return;  /* already open */
    ki_rand_fp = fopen(path, "rb");
    if (!ki_rand_fp) {
        fprintf(stderr, "[FATAL] Cannot open random-file: %s\n", path);
        perror("  fopen");
        exit(1);
    }
    printf("  [RNG] Using true random source: %s\n", path);
}

/* ── Fill buffer with random uint32_t ────────────────────────────
 * If random_file is set, reads from file; otherwise uses w0_random().
 * w0_srandom() must be called before this if using PRNG mode.
 */
static inline void ki_rand_fill(uint32_t *buf, size_t count,
                                 const char *rpath) {
    if (rpath && rpath[0]) {
        ki_rand_open(rpath);
        size_t n = fread(buf, sizeof(uint32_t), count, ki_rand_fp);
        if (n != count) {
            fprintf(stderr, "[FATAL] Random file too short: "
                    "need %zu uint32, got %zu\n", count, n);
            exit(1);
        }
    } else {
        for (size_t i = 0; i < count; i++)
            buf[i] = w0_random();
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * MNIST DATA LOADER
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    int num_images;
    int rows;
    int cols;
    int pixels;
    float *X;         /* [num_images * pixels] normalized to [-1, +1] */
    uint8_t *X_raw;   /* [num_images * pixels] raw uint8 [0,255] */
    uint8_t *y;       /* [num_images] labels */
} ki_MNISTData;

/* ── GZIP decompression ────────────────────────────────────────── */
static int ki_decompress_gz(const char *path, uint8_t **out_data, size_t *out_size) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long gz_len = ftell(f);
    fseek(f, 0, SEEK_SET);

    uint8_t *gz_buf = (uint8_t *)malloc((size_t)gz_len);
    if (!gz_buf) { fclose(f); return -1; }
    if (fread(gz_buf, 1, (size_t)gz_len, f) != (size_t)gz_len) {
        free(gz_buf); fclose(f); return -1;
    }
    fclose(f);

    z_stream strm;
    memset(&strm, 0, sizeof(strm));
    strm.avail_in = (uInt)gz_len;
    strm.next_in = gz_buf;

    if (inflateInit2(&strm, 16 + MAX_WBITS) != Z_OK) {
        free(gz_buf); return -1;
    }

    size_t buf_cap = 65536;
    size_t buf_used = 0;
    uint8_t *buf = (uint8_t *)malloc(buf_cap);

    do {
        if (buf_used + 65536 > buf_cap) {
            buf_cap *= 2;
            uint8_t *newbuf = (uint8_t *)realloc(buf, buf_cap);
            if (!newbuf) { inflateEnd(&strm); free(gz_buf); free(buf); return -1; }
            buf = newbuf;
        }
        strm.avail_out = (uInt)(buf_cap - buf_used);
        strm.next_out = buf + buf_used;
        int ret = inflate(&strm, Z_NO_FLUSH);
        buf_used = buf_cap - strm.avail_out;
        if (ret == Z_STREAM_END) break;
        if (ret != Z_OK) {
            inflateEnd(&strm); free(gz_buf); free(buf); return -1;
        }
    } while (strm.avail_out == 0);

    inflateEnd(&strm);
    free(gz_buf);
    *out_data = buf;
    *out_size = buf_used;
    return 0;
}

/* ── MNIST read ──────────────────────────────────────────────────── */
static int ki_mnist_read(ki_MNISTData *out) {
    const char *candidates[] = {
        "data/mnist",
        "../data/mnist",
        "www/data/mnist",
        NULL
    };
    const char *data_dir = NULL;
    for (int i = 0; candidates[i]; i++) {
        char test[512];
        snprintf(test, sizeof(test), "%s/train-images-idx3-ubyte.gz", candidates[i]);
        if (access(test, R_OK) == 0) {
            data_dir = candidates[i];
            break;
        }
    }
    if (!data_dir) {
        fprintf(stderr, "[FATAL] Cannot find MNIST data.\n");
        return -1;
    }

    memset(out, 0, sizeof(*out));
    uint8_t *raw = NULL;
    size_t raw_size = 0;
    char path[512];

    /* Read images */
    snprintf(path, sizeof(path), "%s/train-images-idx3-ubyte.gz", data_dir);
    if (ki_decompress_gz(path, &raw, &raw_size) != 0) return -1;

    int magic    = (raw[0] << 24) | (raw[1] << 16) | (raw[2] << 8) | raw[3];
    int num_img  = (raw[4] << 24) | (raw[5] << 16) | (raw[6] << 8) | raw[7];
    int rows     = (raw[8] << 24) | (raw[9] << 16) | (raw[10] << 8) | raw[11];
    int cols     = (raw[12] << 24) | (raw[13] << 16) | (raw[14] << 8) | raw[15];
    int pixels   = rows * cols;

    printf("  [MNIST] images: magic=0x%08X %d x %d px=%d\n",
           magic, rows, cols, pixels);

    if (magic != 0x00000803) {
        fprintf(stderr, "[FATAL] Not MNIST image file (magic=0x%08X)\n", magic);
        free(raw); return -1;
    }

    out->num_images = num_img;
    out->rows = rows;
    out->cols = cols;
    out->pixels = pixels;
    size_t npix = (size_t)num_img * (size_t)pixels;
    out->X_raw = (uint8_t *)malloc(npix);
    memcpy(out->X_raw, raw + 16, npix);
    out->X = (float *)malloc(npix * sizeof(float));
    for (size_t i = 0; i < npix; i++)
        out->X[i] = ((float)raw[16 + i] / 255.0f) * 2.0f - 1.0f;
    free(raw);

    /* Read labels */
    snprintf(path, sizeof(path), "%s/train-labels-idx1-ubyte.gz", data_dir);
    if (ki_decompress_gz(path, &raw, &raw_size) != 0) { free(out->X); return -1; }

    int lbl_magic = (raw[0] << 24) | (raw[1] << 16) | (raw[2] << 8) | raw[3];
    int lbl_count = (raw[4] << 24) | (raw[5] << 16) | (raw[6] << 8) | raw[7];

    printf("  [MNIST] labels: magic=0x%08X count=%d\n", lbl_magic, lbl_count);

    if (lbl_magic != 0x00000801) {
        fprintf(stderr, "[FATAL] Not MNIST label file (magic=0x%08X)\n", lbl_magic);
        free(raw); free(out->X); return -1;
    }

    out->y = (uint8_t *)malloc((size_t)lbl_count);
    memcpy(out->y, raw + 8, (size_t)lbl_count);
    free(raw);

    printf("  [MNIST] Loaded %d samples (%d px)\n", out->num_images, out->pixels);
    return 0;
}

static inline void ki_mnist_free(ki_MNISTData *data) {
    free(data->X);
    free(data->X_raw);
    free(data->y);
    memset(data, 0, sizeof(*data));
}


/* ═══════════════════════════════════════════════════════════════════════
 * INPUT LOADING — pack uint8 pixels into uint32 containers
 * ═══════════════════════════════════════════════════════════════════════
 * KI_PACK=4 (KI_NC=196): 4 px/cont, p0|p1<<8|p2<<16|p3<<24
 * KI_PACK=1 (KI_NC=784): 1 px/cont, byte-repeat (*0x01010101)
 */
static uint32_t *load_input(const uint8_t *X_raw, int n_samples) {
    uint32_t *Xb = ki_xmalloc((size_t)n_samples * (size_t)KI_NC * sizeof(uint32_t));
#if KI_PACK == 4
    for (int s = 0; s < n_samples; s++) {
        uint32_t *row = Xb + (size_t)s * KI_NC;
        for (int c = 0; c < KI_NC; c++) {
            uint32_t val = 0;
            for (int k = 0; k < 4; k++) {
                size_t p = (size_t)s * (size_t)KI_PX + (size_t)c * 4 + (size_t)k;
                val |= ((uint32_t)X_raw[p] & 0xFFU) << (unsigned)(k * 8);
            }
            row[c] = val;
        }
    }
#elif KI_PACK == 1
    for (int s = 0; s < n_samples; s++) {
        uint32_t *row = Xb + (size_t)s * KI_NC;
        for (size_t p = 0; p < KI_PX; p++) {
            size_t off = (size_t)s * (size_t)KI_PX + p;
            row[p] = ((uint32_t)X_raw[off] & 0xFFU) * 0x01010101U;
        }
    }
#else
#  error "load_input: KI_PACK must be 4 (196) or 1 (784)"
#endif
    return Xb;
}


/* ═══════════════════════════════════════════════════════════════════════
 * REPORT — Machine-parseable result line
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void ki_report_show(int train_ok, int train_n,
                                   int eval_ok,  int eval_n,
                                   int elapsed_ms, int threadN) {
    float tp = (float)train_ok * 100.0f / (float)train_n;
    float ep = (float)eval_ok  * 100.0f / (float)eval_n;
    printf("\n============================================================\n");
    printf("REPORT train=%.1f%% (%d) eval=%.1f%% (%d) time=%dms threads=%d\n",
           tp, train_n, ep, eval_n, elapsed_ms, threadN);
    printf("============================================================\n");
}

#endif /* KI_OTTO_COMMON_H */
