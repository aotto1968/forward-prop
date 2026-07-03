/*
 * ki-common.h — Shared infrastructure for otto-score-ifc/
 * =========================================================
 *
 * Provides MNIST loading, float32 helpers (matmul, LReLU, AdamW),
 * CLI parsing, and report output for the public demo programs.
 *
 * Stripped of Hebbian-specific code — only what the three programs
 * in otto-score-ifc/ actually use.
 *
 * Usage:
 *   #include "ki-common.h"
 *   ki_Args a = ki_args_defaults();
 *   ki_parse_args(argc, argv, &a);
 *   ...
 *
 * DESIGN: One header, no .c files. Everything is static inline.
 */
#ifndef KI_COMMON_H
#define KI_COMMON_H

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
#define KI_BITS_PER_CONT ((int)sizeof(float)*8)

/* Default model export directory */
#ifndef KI_MODEL_DIR
#define KI_MODEL_DIR "models"
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * ARGS — Unified CLI Parameters
 * ═══════════════════════════════════════════════════════════════════════
 *
 * ALL fields have defaults set in ki_args_defaults().
 * Programs that don't use a field simply ignore it.
 */

typedef struct {
    /* Universal — used by ALL trainers */
    int    hidden;          /* Hidden neurons (--hiddenN, default: 64) */
    int    epochs;          /* Training epochs (--epochsN, default: 10) */
    int    batchN;          /* Mini-batch size (--batchN, default: 64) */
    int    trainN;          /* Training samples (--trainN, default: 50000) */
    int    evalN;           /* Eval samples (--evalN, default: 10000) */
    int    dry_run;         /* --dry-run: print arch and exit */
    int    debug;           /* --debug: verbose per-batch output */
    unsigned int seed;      /* Random seed (--seed, default: 42) */
    char   out[256];        /* --out DIR: export directory */
    float  lr;              /* Learning rate float (--lr, default: 0.002) */

    /* AdamW-specific — used by mlp-flt32-trn-w1-adam, ignored by others */
    uint32_t lr_min_uint;   /* Minimum LR uint32 (--lr-min-uint, default: lr>>3) */
    int      warmup_epochs; /* Warmup epochs (--warmup, default: epochs/5) */
    int      no_decay;      /* --no-decay: constant LR (no cosine decay) */

    /* Hebbian-specific — used by mlp-bin32-trn-w1-hebbian */
    int    hebbian_pct;     /* --hebbian-pct: flip threshold % (default: 50) */
    int    debug_detail;    /* --debug-detail: per-sample debug output */

    /* OpenMP threads */
    int    threadN;          /* OpenMP threads (--threadN, default: 2) */
} ki_Args;

/* ── Defaults ──────────────────────────────────────────────────── */
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
    a.lr            = 0.002f;
    a.lr_min_uint   = 0;
    a.warmup_epochs = 0;
    a.no_decay      = 0;
    a.hebbian_pct   = 50;   /* Hebbian: majority >50% flips */
    a.debug_detail  = 0;
    a.threadN       = 8;
    return a;
}

/* ── UINT32 ↔ float conversion ─────────────────────────────────── */
static inline float ki_lr_uint_to_float(uint32_t lr_uint) {
    return (float)lr_uint / (float)UINT32_MAX;
}

static inline uint32_t ki_float_to_lr_uint(float target) {
    if (target <= 0.0f) return 1U;
    float scaled = target * (float)UINT32_MAX;
    int N = (int)(log2f(scaled) + 0.5f);
    if (N < 0) N = 0;
    if (N > 31) N = 31;
    return 1U << (unsigned int)N;
}

/* ── Parse CLI ───────────────────────────────────────────────────
 *
 * Unified parser for ALL ki-w1 programs.
 * Programs that don't use a flag simply ignore the field.
 * Calls exit(0) on --help, exit(1) on unknown flag.
 */
static inline void ki_parse_args(int argc, char *argv[], ki_Args *a) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --hiddenN N       Hidden neurons (default: 64)\n");
            printf("  --epochsN N       Training epochs (default: 10)\n");
            printf("  --batchN N        Mini-batch size (default: 64)\n");
            printf("  --trainN N        Training samples (default: 50000)\n");
            printf("  --evalN N         Eval samples (default: 10000)\n");
            printf("  --warmup N        Warmup epochs (AdamW, default: auto)\n");
            printf("  --quick           5000 train / 2000 eval\n");
            printf("  --qq              5000 train / 2000 eval / 3 epochs\n");
            printf("  -----------------------------------------------\n");
            printf("  --lr FLOAT        Learning rate (default: 0.002)\n");
            printf("  --no-decay        Constant LR (AdamW)\n");
            printf("  -----------------------------------------------\n");
            printf("  --threadN N       OpenMP threads (default: 2)\n");
            printf("  --seed N          Random seed (default: 42)\n");
            printf("  --out DIR         Export directory\n");
            printf("  --dry-run         Print architecture and exit\n");
            printf("  --debug           Verbose output\n");
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
        } else if ((strcmp(argv[i], "--batchN") == 0) && i + 1 < argc) {
            a->batchN = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "--trainN") == 0 || strcmp(argv[i], "--n-train") == 0) && i + 1 < argc) {
            a->trainN = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "--evalN") == 0 || strcmp(argv[i], "--n-eval") == 0) && i + 1 < argc) {
            a->evalN = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            a->lr = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            a->warmup_epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-decay") == 0) {
            a->no_decay = 1;
        } else if (strcmp(argv[i], "--threadN") == 0 && i + 1 < argc) {
            a->threadN = atoi(argv[++i]);
            if (a->threadN < 1) a->threadN = 1;
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            a->seed = (unsigned int)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            strncpy(a->out, argv[++i], sizeof(a->out) - 1);
            a->out[sizeof(a->out) - 1] = '\0';
        } else if (strcmp(argv[i], "--hebbian-pct") == 0 && i + 1 < argc) {
            a->hebbian_pct = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--debug-detail") == 0) {
            a->debug_detail = 1;
        } else {
            fprintf(stderr, "[ERROR] Unknown argument: %s\nTry --help\n", argv[i]);
            exit(1);
        }
    }

    /* Default lr_min_uint = lr >> 3 (AdamW floor) */
    if (a->lr_min_uint == 0) {
        a->lr_min_uint = ki_float_to_lr_uint(a->lr) >> 3;
        if (a->lr_min_uint < 1) a->lr_min_uint = 1;
    }

    /* Default warmup: epochs/5 */
    if (a->warmup_epochs <= 0) {
        a->warmup_epochs = (a->epochs > 5) ? a->epochs / 5 : 1;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * MEMORY HELPERS (defined before MNIST loader which depends on them)
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
 * MNIST DATA LOADER — Self-contained (no external .c file needed)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Usage:
 *   ki_MNISTData data;
 *   if (ki_mnist_read(&a, &data) != 0) return 1;
 *   // data.X[0..trainN*pixels] normalized to [-1,+1]
 *   // data.y[0..trainN] labels
 *   ki_mnist_free(&data);
 */

typedef struct {
    int num_images;
    int rows;
    int cols;
    int pixels;        /* rows * cols */
    float *X;         /* [num_images * pixels] normalized to [-1, +1] */
    uint8_t *X_raw;   /* [num_images * pixels] raw uint8 [0,255] (for load_input) */
    uint8_t *y;       /* [num_images] labels */
} ki_MNISTData;

/* ── GZIP decompression (internal) ────────────────────────────── */
static __attribute__((unused)) int ki_decompress_gz(const char *path, uint8_t **out_data, size_t *out_size) {
    FILE *f = fopen(path, "rb");
    if (!f) { return -1; }
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

/* ── ki_mnist_read — try multiple data paths ────────────────────
 *
 * Searches for MNIST data in:
 *   1. <data_dir>/mnist/  (canonical)
 *   2. ../data/mnist/     (from ki-w1/)
 *   3. www/data/mnist/    (workspace root)
 * The data_dir is taken from args.out or searched automatically.
 */
static __attribute__((unused)) int ki_mnist_read(ki_MNISTData *out) {
    const char *candidates[] = {
        "data/mnist",
        "../data/mnist",
        "../mnist-research/data/mnist",
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
        fprintf(stderr, "[FATAL] Cannot find MNIST data. Tried: data/mnist, ../data/mnist, www/data/mnist\n");
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
    int expected = 16 + num_img * pixels;

    printf("  [MNIST] images: magic=0x%08X %d×%d×%d px=%d\n",
           magic, num_img, rows, cols, pixels);

    if (magic != 0x00000803) {
        fprintf(stderr, "[FATAL] Not MNIST image file (magic=0x%08X)\n", magic);
        free(raw); return -1;
    }
    if ((int)raw_size != expected) {
        fprintf(stderr, "[FATAL] Image file size mismatch: %zu vs %d\n", raw_size, expected);
        free(raw); return -1;
    }

    out->num_images = num_img;
    out->rows = rows;
    out->cols = cols;
    out->pixels = pixels;
    /* Save raw uint8 pixels (for load_input) */
    size_t npix = (size_t)num_img * (size_t)pixels;
    out->X_raw = (uint8_t *)malloc(npix);
    memcpy(out->X_raw, raw + 16, npix);
    /* Float version (for float32 trainer) */
    out->X = (float *)malloc(npix * sizeof(float));
    for (size_t i = 0; i < npix; i++) {
        out->X[i] = ((float)raw[16 + i] / 255.0f) * 2.0f - 1.0f;
    }
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
    if ((int)raw_size != 8 + lbl_count || lbl_count != num_img) {
        fprintf(stderr, "[FATAL] Label mismatch\n");
        free(raw); free(out->X); return -1;
    }

    out->y = (uint8_t *)malloc((size_t)lbl_count);
    for (int i = 0; i < lbl_count; i++) out->y[i] = raw[8 + i];
    free(raw);

    printf("  [MNIST] Loaded %d samples (%d px)\n", out->num_images, out->pixels);
    return 0;
}

/* ── ki_mnist_free ────────────────────────────────────────────── */
static inline void ki_mnist_free(ki_MNISTData *data) {
    free(data->X);
    free(data->X_raw);
    free(data->y);
    memset(data, 0, sizeof(*data));
}

/* ── Pack MNIST input: 784 px → 196 containers (4px/cont) ───────
 *
 * DESIGN: Takes the FIRST pixel of each 4-pixel group, retains [-1,+1]
 * normalization. Same bit-mass as binary trainer (nc=196), but
 * numerically stable for SGD (kaiming_init expects input variance ≈ 1,
 * not [0,255] raw bytes).
 *
 * AdamW could handle [0,255] (adaptive per-param LR), SGD cannot.
 * See: comments in mlp-flt32-trn-w1-sgd.c
 *
 * nc_out = 196, in_features = nc_out.
 */
/* ── ki_pack_packed_float: average 4 pixels → 1 float (for AdamW) ──
 *
 * Takes raw uint8 pixels (from md.X_raw), packs 4 pixels per container
 * (same grouping as Hebbian: p0|p1<<8|p2<<16|p3<<24) and returns
 * 196 floats [-1,+1] per sample (mean of 4 pixels).
 *
 * This is the float equivalent of the 196×uint32 packed format.
 * Same pixel grouping, same dimensionality — just as float.
 */
static __attribute__((unused)) float *ki_pack_packed_float(const uint8_t *X_raw, int n_samples) {
    size_t total = (size_t)n_samples * (size_t)KI_NC;
    float *packed = (float *)ki_xmalloc(total * sizeof(float));
    for (int s = 0; s < n_samples; s++) {
        for (int c = 0; c < KI_NC; c++) {
            int sum = 0;
            for (int k = 0; k < 4; k++) {
                size_t p = (size_t)s * (size_t)KI_PX + (size_t)c * 4 + (size_t)k;
                sum += (int)X_raw[p];
            }
            packed[(size_t)s * (size_t)KI_NC + (size_t)c] =
                (float)sum / (4.0f * 127.5f) - 1.0f;
        }
    }
    return packed;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  INPUT LOADING — PACKING-dependent encoding (unchanged)
 * ═══════════════════════════════════════════════════════════════════════
 *
 *  KI_PACK=4 (KI_NC=196): 4 px/Cont, p0|p1<<8|p2<<16|p3<<24
 *  KI_PACK=1 (KI_NC=784): 1 px/Cont, byte-repeat (*0x01010101)
 */

/* ── load_input: uint8_t Pixel → uint32_t Container ──────────────
 *  Dieses Makro erlaubt lokalen Dateien eine eigene load_input zu
 *  definieren (z.B. Hebbian mit PACKING=1,2,4 vs KI_PACK=4,1).
 *  Vor #include "ki-common.h" einfach #define KI_COMMON_LOAD_INPUT */
#ifndef KI_COMMON_LOAD_INPUT
static __attribute__((unused)) uint32_t *load_input(const uint8_t *X_raw, int n_samples) {
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
#endif /* KI_COMMON_LOAD_INPUT */

/* ═══════════════════════════════════════════════════════════════════════
 * SETUP DISPLAY — Unified architecture overview
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    const char *title;
    int   H;
    int   epochs;
    int   bits_per_cont;
    int   pixel_bits;
    unsigned int seed;
    int   N;
    int   ne;
    int   n_threads;
    int   px;
    int   sizeof_bn;
    int   nc;
    int   C;
    size_t input_bit;
    size_t hidden_bit;
    size_t output_bit;
    size_t w0_bit;
    size_t w1_bit;
    int   batchN;       /* updates per epoch (0 = N/A) */
    int   maxFlips;     /* flips per update (0 = N/A) */
    int   no_close;     /* don't close the separator line */
    int   w1_hidden_nrn; /* W1 hidden neurons (for 2b/biaxial) */
    int   w0_hidden_nrn; /* W0 hidden neurons (for biaxial) */
} ki_SetupInfo;

static inline void ki_setup_show(const ki_SetupInfo *s) {
    printf("══════════════════════════════════════════════════════════════════════\n");
    printf("══╡ %s ╞══  H=%-4d  Ep=%-2d  bitsPerCont=%-2d  pixelBits=%-2d  seed=%-4u",
           s->title, s->H, s->epochs, s->bits_per_cont, s->pixel_bits, s->seed);
    if (s->batchN > 0)
        printf("  batchN=%d  maxFlips=%d", s->batchN, s->maxFlips);
    printf("\n");
    printf("══╡ SETUP ╞══════════════════════════════════════════════════════════\n");
    printf("  OMP:         %d threads\n", s->n_threads);
    printf("  Train/Eval:  %d / %d samples\n", s->N, s->ne);
    printf("  sizeof(bn)   %d byte\n", s->sizeof_bn);
    printf("  ───────────────────────────────────────────────────────────\n");
    printf("  INPUT        %4d nrn × %2d bit  = %7zu bit  (%5.1f KB)\n",
           s->nc, s->bits_per_cont, s->input_bit, (double)s->input_bit / 8 / 1024);
    printf("  HIDDEN       %4d nrn × %2d bit  = %7zu bit  (%5.1f KB)\n",
           s->H, s->bits_per_cont, s->hidden_bit, (double)s->hidden_bit/8/1024);
    printf("  OUTPUT       %4d nrn × %2d bit  = %7zu bit  (%5.1f KB)\n",
           10, s->bits_per_cont, s->output_bit, (double)s->output_bit/8/1024);
    printf("  ───────────────────────────────────────────────────────────\n");
    printf("  W0 = %4d × %4d  × %2d bit  = %9zu bit  (%5.1f KB)\n",
           s->H, s->nc, s->bits_per_cont, s->w0_bit, (double)s->w0_bit/8/1024);
    printf("  W1 = %4d × %4d  × %2d bit  = %9zu bit  (%5.1f KB)\n",
           10, s->H, s->bits_per_cont, s->w1_bit, (double)s->w1_bit/8/1024);
    printf("  ───────────────────────────────────────────────────────────\n");
    printf("  TOTAL (W0+W1)              %7zu bit  (%5.1f KB)\n",
           s->w0_bit + s->w1_bit, (double)(s->w0_bit + s->w1_bit)/8/1024);
    printf("══════════════════════════════════════════════════════════════════════\n");
}

static inline void ki_dry_run_show(const ki_SetupInfo *s) {
    ki_setup_show(s);
    printf("══╡ INTERNALS ╞══════════════════════════════════════════════════════\n");
    printf("  %-18s %-12d  %s\n",  "nc",          s->nc,  "containers per image");
    printf("  %-18s %-12d  %s\n",  "C",           s->C,   "maj_bits voters");
    printf("  %-18s %-12d  %s\n",  "px",          s->px,  "input pixels");
    printf("  %-18s %-12d  %s\n",  "sizeof",      s->sizeof_bn, "sizeof(container)");
    printf("  %-18s %-12d  %s\n",  "BITS_PER_CONT", s->bits_per_cont, "bits per container");
    printf("  %-18s %-12d  %s\n",  "N",           s->N,   "train samples");
    printf("  %-18s %-12d  %s\n",  "ne",          s->ne,  "eval samples");
    printf("  %-18s %-12d  %s\n",  "epochs",      s->epochs, "training epochs");
    printf("  %-18s %-12u  %s\n",  "seed",        s->seed, "W0 random seed");
    printf("  %-18s %-12zu  %s\n", "input_bit",   s->input_bit,  "nc × BITS_PER_CONT");
    printf("  %-18s %-12zu  %s\n", "hidden_bit",  s->hidden_bit, "H × BITS_PER_CONT");
    printf("  %-18s %-12zu  %s\n", "output_bit",  s->output_bit, "10 × BITS_PER_CONT");
    printf("  %-18s %-12zu  %s\n", "w0_bit",      s->w0_bit,     "H × nc × bits_per_cont");
    printf("  %-18s %-12zu  %s\n", "w1_bit",      s->w1_bit,     "K × H × bits_per_cont");
    printf("══════════════════════════════════════════════════════════════════════\n");
}

/* ═══════════════════════════════════════════════════════════════════════
 * REPORT — Machine-parseable result line
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void ki_report_show(int train_ok, int train_n,
                                   int eval_ok,  int eval_n,
                                   int elapsed_ms, int threadN,
                                   int err, float lr) {
    float tp = (float)train_ok * 100.0f / (float)train_n;
    float ep = (float)eval_ok  * 100.0f / (float)eval_n;
    (void)err; (void)lr;
    printf("\n============================================================\n");
    printf("REPORT train=%.1f%% (%d) eval=%.1f%% (%d)"
           " err=%d lr=%.4f time=%dms threads=%d\n",
           tp, train_n, ep, eval_n, err, (double)lr, elapsed_ms, threadN);
    printf("============================================================\n");
}

/* ═══════════════════════════════════════════════════════════════════════
 * LR SCHEDULE — Cosine Decay + Linear Warmup
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Returns the learning rate for a given epoch.
 * - Warmup: linear von 0 → base_lr (epoch 0..warmup-1)
 * - Cosine: base_lr → lr_min (epoch warmup..total-1)
 * - no_decay: konstant base_lr
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
 * FLOAT32 OPERATIONS (for matmul-based trainers)
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    int in_features;
    int out_features;
    float *W;
} ki_LinearLayer;

/* ── Linear forward: output[b][o] = Σ_i x[b][i] × W[o][i] ───── */
static inline void ki_linear_forward(const ki_LinearLayer *layer,
                                      const float *x, float *output, int batchN) {
    #pragma omp parallel for collapse(2) if(batchN * layer->out_features >= 64)
    for (int b = 0; b < batchN; b++) {
        for (int o = 0; o < layer->out_features; o++) {
            float acc = 0.0f;
            for (int i = 0; i < layer->in_features; i++) {
                acc += x[b * layer->in_features + i] * layer->W[o * layer->in_features + i];
            }
            output[b * layer->out_features + o] = acc;
        }
    }
}

/* ── Leaky ReLU: x[i] = x[i] < 0 ? x[i] * 0.05 : x[i] ──────── */
static inline void ki_leaky_relu(float *x, int n) {
    #pragma omp parallel for if(n >= 256)
    for (int i = 0; i < n; i++) {
        if (x[i] < 0.0f) x[i] *= 0.05f;
    }
}

/* ── Kaiming uniform init ──────────────────────────────────────── */
static inline void ki_init_kaiming(ki_LinearLayer *layer, unsigned int seed) {
    srand(seed);
    float bound = 1.0f / sqrtf((float)layer->in_features);
    int n = layer->out_features * layer->in_features;
    for (int i = 0; i < n; i++) {
        layer->W[i] = (float)rand() / (float)RAND_MAX * 2.0f * bound - bound;
    }
}

/* ── Shuffle (Fisher-Yates) ────────────────────────────────────── */
static inline void ki_shuffle(int *indices, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i]; indices[i] = indices[j]; indices[j] = temp;
    }
}

#endif /* KI_COMMON_H */
