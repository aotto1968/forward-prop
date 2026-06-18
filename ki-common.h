/*
 * ki-w2/ki-common.h — Minimal shared infrastructure for Otto Score
 * ================================================================
 *
 * Provides CLI parsing, MNIST loading, memory helpers, input packing,
 * and report output for ki-w2/ Otto Score programs.
 * Minimal version — only what Otto Score actually uses.
 *
 * See: ki-w0/ki-common.h for the full version with Hebbian/float32.
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
    int    threadN;         /* OpenMP threads (--threadN, default: 8) */
    int    debug_h0;        /* --debug-h0: per-neuron debug */
    int    shuffle;         /* --shuffle: randomize train/eval split */
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
    return a;
}

/* ── Parse CLI ─────────────────────────────────────────────────── */
static inline void ki_parse_args(int argc, char *argv[], ki_Args *a) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --hiddenN N       Hidden neurons (default: 64)\n");
            printf("  --epochsN N       Iterations (default: 1)\n");
            printf("  --batchN N        Mini-batch size (default: 64)\n");
            printf("  --trainN N        Training samples (default: 50000)\n");
            printf("  --evalN N         Eval samples (default: 10000)\n");
            printf("  --quick           5000 train / 2000 eval\n");
            printf("  --qq              5000 train / 2000 eval / 3 epochs\n");
            printf("  --lr FLOAT        Step size (default: 0.05)\n");
            printf("  --threadN N       OpenMP threads (default: 8)\n");
            printf("  --seed N          Random seed (default: 42)\n");
            printf("  --out DIR         Export directory\n");
            printf("  --dry-run         Print architecture and exit\n");
            printf("  --debug           Verbose output\n");
            printf("  --debug-h0        Per-neuron debug\n");
            printf("  --shuffle         Shuffle data before train/eval split\n");
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
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            a->seed = (unsigned int)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            strncpy(a->out, argv[++i], sizeof(a->out) - 1);
            a->out[sizeof(a->out) - 1] = '\0';
        } else if (strcmp(argv[i], "--debug-h0") == 0) {
            a->debug_h0 = 1;
        } else if (strcmp(argv[i], "--shuffle") == 0) {
            a->shuffle = 1;
        } else {
            fprintf(stderr, "[ERROR] Unknown argument: %s\nTry --help\n", argv[i]);
            exit(1);
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

#endif /* KI_COMMON_H */
