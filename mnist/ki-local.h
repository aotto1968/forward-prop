/*
 * mnist-1/ki-local.h — MNIST-specific constants + data loader
 * ===========================================================
 *
 * Included by ki-common.h (shared between mnist-1/ and cifar-1/).
 * Provides dataset-specific definitions: pixel size, containers,
 * data struct and loader.
 */
#ifndef KI_LOCAL_H
#define KI_LOCAL_H

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <zlib.h>

/* ═══════════════════════════════════════════════════════════════════════
 * CONSTANTS — MNIST
 * ═══════════════════════════════════════════════════════════════════════ */

#define KI_DATASET_ID             0       /* unique for cache key */
#define KI_PX                   784
#define KI_NCLASSES             10
#define KI_DEFAULT_LR           0.05f   /* → step = 0.05 × 131072 = 6554 */
#define KI_DEFAULT_STEP_POWER   0.1f    /* höher erzeugt kleiners trn */
#define KI_DEFAULT_STEP_MODE    STEP_COS_TIME
#define KI_DEFAULT_BATCH_N      64      /* optimum */
#define KI_COLORS               1       /* MNIST is grayscale */
#define KI_DEFAULT_COLOR        (1<<COLOR_MNIST)  /* MNIST: single grayscale block */
#ifndef KI_NC
#define KI_NC                   196     /* Packed containers per image: 784/4 */
#endif
#define KI_NC_TOTAL             (KI_NC * KI_COLORS)  /* = 196 (same as KI_NC for grayscale) */
#ifndef KI_PACK
#define KI_PACK                 (784 / KI_NC)  /* Pixels per container */
#endif

#ifndef NC
#define NC  KI_NC
#endif

#ifndef OT_PRECISION
#define OT_PRECISION 17
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * MNIST DATA STRUCT + LOADER
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    int num_images;
    int rows;
    int cols;
    int pixels;
    float *X;         /* [num_images * pixels] normalized to [-1, +1] */
    uint8_t *X_raw;   /* [num_images * pixels] raw uint8 [0,255] */
    uint8_t *y;       /* [num_images] labels */
    int dry_run;      /* skip pixel data (fast metadata only) */
    int n_train;      /* default training count (set by loader) */
    int n_eval;       /* default eval count (set by loader) */
} ki_MNISTData;

/* Generic dataset aliases (used by ki-common.h) */
typedef ki_MNISTData ki_Dataset;

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

    int dry_run = out->dry_run;
    memset(out, 0, sizeof(*out));
    out->dry_run = dry_run;
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
    out->n_train = num_img;         /* all loaded images for training */
    out->n_eval = 10000;            /* test set (t10k) */
    out->rows = rows;
    out->cols = cols;
    out->pixels = pixels;

    /* Dry-run: header already parsed, skip pixel data */
    if (dry_run) {
        free(raw);
        printf("  [MNIST] dry-run: %d images, %d px each\n", num_img, pixels);
        return 0;
    }

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

/* ── PNG writer (dataset-specific: grayscale for MNIST) ────── */
__attribute__((unused))
static void _png_be32(FILE *f, uint32_t v) {
    uint8_t buf[4] = {
        (uint8_t)((v >> 24) & 0xFF),
        (uint8_t)((v >> 16) & 0xFF),
        (uint8_t)((v >>  8) & 0xFF),
        (uint8_t)( v        & 0xFF)
    };
    fwrite(buf, 1, 4, f);
}

__attribute__((unused))
static void _png_chunk(FILE *f, const char type[4],
                        const void *data, size_t len) {
    unsigned long crc = crc32(0L, NULL, 0);
    _png_be32(f, (uint32_t)len);
    crc = crc32(crc, (const unsigned char *)type, 4);
    fwrite(type, 1, 4, f);
    if (len > 0 && data) {
        crc = crc32(crc, (const unsigned char *)data, (unsigned int)len);
        fwrite(data, 1, len, f);
    }
    _png_be32(f, (uint32_t)(crc & 0xFFFFFFFFUL));
}

__attribute__((unused))
static void ki_write_png(const char *path,
                          const uint8_t *gray,
                          const uint8_t *plane1, const uint8_t *plane2,
                          int w, int h) {
    (void)plane1; (void)plane2;
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[ERROR] Cannot write %s\n", path); return; }

    uint8_t sig[8] = {137,80,78,71,13,10,26,10};
    fwrite(sig, 1, 8, f);

    uint8_t ihdr[13];
    for (int i = 0; i < 4; i++) {
        ihdr[3-i] = (uint8_t)((uint32_t)w >> (i*8));
        ihdr[7-i] = (uint8_t)((uint32_t)h >> (i*8));
    }
    ihdr[8]  = 8;  ihdr[9]  = 0;  ihdr[10] = 0;  ihdr[11] = 0;  ihdr[12] = 0;
    _png_chunk(f, "IHDR", ihdr, 13);

    size_t row_bytes = (size_t)w;
    size_t raw_size = (size_t)h * (1 + row_bytes);
    uint8_t *raw = (uint8_t *)malloc(raw_size);
    for (int y = 0; y < h; y++) {
        size_t y_off = (size_t)y * (1 + row_bytes);
        raw[y_off] = 0;
        memcpy(raw + y_off + 1, gray + (size_t)y * (size_t)w, row_bytes);
    }

    uLongf comp_len = compressBound(raw_size);
    uint8_t *comp = (uint8_t *)malloc(comp_len);
    if (compress(comp, &comp_len, raw, raw_size) != Z_OK) {
        free(raw); free(comp); fclose(f); return;
    }
    _png_chunk(f, "IDAT", comp, comp_len);
    _png_chunk(f, "IEND", NULL, 0);

    fclose(f);
    free(raw);
    free(comp);
}

/* Dataset function aliases */
#define ki_dataset_read ki_mnist_read
#define ki_dataset_free ki_mnist_free

/* ── Class names (dataset-specific) ────────────────────────────── */
__attribute__((unused))
static const char *ki_class_names[KI_NCLASSES] = {
    "0","1","2","3","4","5","6","7","8","9"
};

#endif /* KI_LOCAL_H */
