/*
 * cifar-1/ki-local.h — CIFAR-10-specific constants + data loader
 * ==============================================================
 *
 * Included by ki-common.h (symlinked from mnist-1/ki-common.h).
 * Provides CIFAR-10 definitions: pixel size, containers,
 * data struct and loader.
 */
#ifndef KI_LOCAL_H
#define KI_LOCAL_H

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/* ═══════════════════════════════════════════════════════════════════════
 * CONSTANTS — CIFAR-10
 * ═══════════════════════════════════════════════════════════════════════ */

#define KI_PX                     3072    /* 32 × 32 × 3 = 3072 pixels per image */
#define KI_NCLASSES               10
#define KI_DEFAULT_LR             0.01f   /* → step = 0.005 × 131072 = 655 (großes lr fürhrt zu starken trn und schwachem evl*/
#define KI_DEFAULT_STEP_POWER     7.0f    /* höher erzeugt kleiners trn */
#define KI_DEFAULT_STEP_MODE      STEP_COS_TIME
#define KI_DEFAULT_BATCH_N        128     /* optimum */
#define KI_DEFAULT_ENSEMBLE_SEED  ENS_SEED_ONCE
#define KI_COLORS                 3       /* R, G, B as independent samples, each packed 4px/cont */
#define KI_DEFAULT_COLOR          ((1<<COLOR_R)|(1<<COLOR_G)|(1<<COLOR_B))  /* CIFAR default: raw R+G+B (bits 1,2,3) */
#ifndef KI_NC
#define KI_NC                     256     /* Packed containers per color: 1024 px / 4 px/cont */
#endif
#define KI_NC_TOTAL               (KI_NC * KI_COLORS)  /* 768 containers per image */
#define KI_PACK                   4       /* 4 pixels packed per uint32 (same as MNIST) */

#ifndef NC
#define NC  KI_NC
#endif

#ifndef OT_PRECISION
#define OT_PRECISION      17
#endif

/* Für mlp-flt32-trn-*-adam.c (Alt-Trainer) */
#ifndef KI_BITS_PER_CONT
#define KI_BITS_PER_CONT  32
#endif
#ifndef KI_MODEL_DIR
#define KI_MODEL_DIR      "models"
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * 8 BLOCK COMPUTATIONS — shared between trainer + samples tool
 * ═══════════════════════════════════════════════════════════════════════
 *   Block | Name | Formel
 *   ------|------|----------------------------
 *     0   |  R   | r
 *     1   |  G   | g
 *     2   |  B   | b
 *     3   |  Y   | (r*77+g*150+b*29)>>8   (ITU-R BT.601)
 *     4   |  LUM | (r+g)>>1
 *     5   |  RG  | clamp(128 + (r-g))
 *     6   |  BY  | clamp(128 + b - (r+g)/2)
 *     7   |  YL  | (r*54+g*183+b*18)>>8   (ITU-R BT.709)
 */
#define KI_NBLOCKS 8

#ifndef KI_ENCODING_H
static inline int ki_clamp_u8(int v) {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return v;
}

/* Single pixel: r,g,b in 0..255 → 8 block values */
static inline void ki_blocks_from_rgb(int r, int g, int b, uint8_t blocks[COLOR_NB]) {
    unsigned int ru = (unsigned int)r;
    unsigned int gu = (unsigned int)g;
    unsigned int bu = (unsigned int)b;
    blocks[COLOR_MNIST]= 0;                                               /* MNIST=0 (unused, aber valide) */
    blocks[COLOR_R]  = (uint8_t)r;                                        /* R */
    blocks[COLOR_G]  = (uint8_t)g;                                        /* G */
    blocks[COLOR_B]  = (uint8_t)b;                                        /* B */

    blocks[COLOR_Y]  = (uint8_t)((ru*77U + gu*150U + bu*29U) >> 8U);      /* Y  */
    blocks[COLOR_YL] = (uint8_t)((ru*54U + gu*183U + bu*18U) >> 8U);      /* YL */

    blocks[COLOR_AL] = (uint8_t)((r + g) >> 1);                           /* AL */
    blocks[COLOR_AM] = (uint8_t)ki_clamp_u8(128 + (r - g));               /* AM */
    blocks[COLOR_AP] = (uint8_t)ki_clamp_u8(128 + (b - (r + g)/2));       /* AP */

    blocks[COLOR_BL] = (uint8_t)((r + b) >> 1);                           /* BL */
    blocks[COLOR_BM] = (uint8_t)ki_clamp_u8(128 + (r - b));               /* BM */
    blocks[COLOR_BP] = (uint8_t)ki_clamp_u8(128 + (g - (r + b)/2));       /* BP */

    blocks[COLOR_RG] = (uint8_t)ki_clamp_u8(128 + (r - g));               /* RG */
    blocks[COLOR_RB] = (uint8_t)ki_clamp_u8(128 + (r - b));               /* RB */
    blocks[COLOR_GB] = (uint8_t)ki_clamp_u8(128 + (g - b));               /* GB */

    /* Hue (Farbwinkel) — atan2(2R-G-B, G-B) normiert auf 0..255 */
    {   float dx = 2.0f * (float)r - (float)g - (float)b;
        float dy = 1.0f * ((float)g - (float)b);
        float hue_f = atan2f(dy, dx) / (2.0f * 3.14159265358979323846f);
        int hue = (int)((hue_f + 0.5f) * 255.0f);
        if (hue < 0) { hue = 0; }
        if (hue > 255) { hue = 255; }
        blocks[COLOR_H] = (uint8_t)hue;
    }
    /* Saturation (Farbsättigung) — max(R,G,B) - min(R,G,B) */
    {   int bv = b;  /* copy to avoid -Wshadow */
        int mx = r, mn = r;
        if (g > mx) { mx = g; } if (g < mn) { mn = g; }
        if (bv > mx) { mx = bv; } if (bv < mn) { mn = bv; }
        blocks[COLOR_S] = (uint8_t)(mx - mn);
    }
    /* Contrast (wird nach Sobel in load_input berechnet — hier Platzhalter) */
    blocks[COLOR_C] = (uint8_t)r;

    blocks[COLOR_CL] = (uint8_t)((g + b) >> 1);                           /* CL */
    blocks[COLOR_CM] = (uint8_t)ki_clamp_u8(128 + (g - b));               /* CM */
    blocks[COLOR_CP] = (uint8_t)ki_clamp_u8(128 + (r - (g + b)/2));       /* CP */
}
#endif /* KI_ENCODING_H */

/* ═══════════════════════════════════════════════════════════════════════
 * CIFAR-10 DATA STRUCT + LOADER
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    int num_images;
    int rows;
    int cols;
    int pixels;        /* rows * cols * channels (3072) */
    int channels;      /* 3 for CIFAR-10 */
    float *X;          /* [num_images * pixels] normalized to [-1, +1] */
    uint8_t *X_raw;    /* [num_images * pixels] raw uint8 [0,255] */
    uint8_t *y;        /* [num_images] labels */
    int dry_run;       /* skip pixel data (fast metadata only) */
} ki_ImageData;

/* Generic dataset aliases (used by ki-common.h) */
typedef ki_ImageData ki_Dataset;

/* ── Read a single CIFAR-10 batch file ────────────────────────── */
static uint8_t *ki_cifar_read_batch(const char *base_dir, const char *filename) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", base_dir, filename);
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    size_t file_size = 10000UL * 3073UL;
    uint8_t *buf = (uint8_t *)malloc(file_size);
    if (!buf) { fclose(f); return NULL; }
    size_t nread = fread(buf, 1, file_size, f);
    fclose(f);
    if (nread != file_size) {
        fprintf(stderr, "[ERROR] %s: expected %zu bytes, got %zu\n",
                path, file_size, nread);
        free(buf); return NULL;
    }
    return buf;
}

/* ── CIFAR-10 read (5 train + 1 test batch, 60000 total) ──────── */
static int ki_cifar_read(ki_ImageData *out) {
    const char *candidates[] = {
        "data/cifar-10-batches-bin",
        "cifar-1/data/cifar-10-batches-bin",    /* launch from PROJECT_ROOT */
        "../data/cifar-10-batches-bin",
        "../cifar-1/data/cifar-10-batches-bin",
        NULL
    };
    const char *data_dir = NULL;
    for (int i = 0; candidates[i]; i++) {
        char test[512];
        snprintf(test, sizeof(test), "%s/data_batch_1.bin", candidates[i]);
        if (access(test, R_OK) == 0) {
            data_dir = candidates[i];
            break;
        }
    }
    if (!data_dir) {
        fprintf(stderr, "[FATAL] Cannot find CIFAR-10 data.\n");
        fprintf(stderr, "  Run: bash fetch_cifar10.sh\n");
        return -1;
    }

    int dry_run = out->dry_run;
    memset(out, 0, sizeof(*out));
    out->dry_run = dry_run;
    out->rows     = 32;
    out->cols     = 32;
    out->channels = 3;
    out->pixels   = KI_PX;
    out->num_images = 60000;

    if (dry_run) {
        printf("  [CIFAR-10] dry-run: 60000 samples (%d px, 3 channels) — 5 train + 1 test batch\n", KI_PX);
        return 0;
    }

    const char *train_files[] = {
        "data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin",
        "data_batch_4.bin", "data_batch_5.bin"
    };
    int n_train = 5, train_total = 0;
    uint8_t *all_raw = NULL;
    size_t all_raw_cap = 0, all_raw_used = 0;

    for (int b = 0; b < n_train; b++) {
        uint8_t *batch = ki_cifar_read_batch(data_dir, train_files[b]);
        if (!batch) { free(all_raw); return -1; }
        size_t batch_bytes = 10000UL * 3073UL;
        if (all_raw_used + batch_bytes > all_raw_cap) {
            all_raw_cap = all_raw_cap ? all_raw_cap * 2 : batch_bytes;
            uint8_t *nb = (uint8_t *)realloc(all_raw, all_raw_cap);
            if (!nb) { free(all_raw); free(batch); return -1; }
            all_raw = nb;
        }
        memcpy(all_raw + all_raw_used, batch, batch_bytes);
        all_raw_used += batch_bytes;
        train_total += 10000;
        free(batch);
    }

    uint8_t *test_batch = ki_cifar_read_batch(data_dir, "test_batch.bin");
    if (!test_batch) { free(all_raw); return -1; }

    int total_images = train_total + 10000;
    out->num_images = total_images;
    size_t total_pixels = (size_t)total_images * (size_t)KI_PX;
    out->X_raw = (uint8_t *)malloc(total_pixels);
    out->X     = (float *)malloc(total_pixels * sizeof(float));
    out->y     = (uint8_t *)malloc((size_t)total_images);
    if (!out->X_raw || !out->X || !out->y) {
        free(out->X_raw); free(out->X); free(out->y);
        free(all_raw); free(test_batch); return -1;
    }

    uint8_t *rp = all_raw;
    for (int i = 0; i < train_total; i++) {
        out->y[i] = rp[0];
        memcpy(out->X_raw + (size_t)i * KI_PX, rp + 1, KI_PX);
        for (int p = 0; p < KI_PX; p++) {
            size_t off = (size_t)i * (size_t)KI_PX + (size_t)p;
            out->X[off] = ((float)rp[1 + p] / 127.5f) - 1.0f;
        }
        rp += 3073;
    }
    free(all_raw);

    rp = test_batch;
    for (int i = 0; i < 10000; i++) {
        int idx = train_total + i;
        out->y[idx] = rp[0];
        memcpy(out->X_raw + (size_t)idx * KI_PX, rp + 1, KI_PX);
        for (int p = 0; p < KI_PX; p++) {
            size_t off = (size_t)idx * (size_t)KI_PX + (size_t)p;
            out->X[off] = ((float)rp[1 + p] / 127.5f) - 1.0f;
        }
        rp += 3073;
    }
    free(test_batch);

    printf("  [CIFAR-10] Loaded %d samples (%d px, 3 channels) — 5 train + 1 test batch\n", total_images, KI_PX);
    return 0;
}

static inline void ki_cifar_free(ki_ImageData *data) {
    free(data->X);
    free(data->X_raw);
    free(data->y);
    memset(data, 0, sizeof(*data));
}

/* ── PNG writer (dataset-specific: color RGB for CIFAR-10) ── */
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
                          const uint8_t *r, const uint8_t *g,
                          const uint8_t *b, int w, int h) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[ERROR] Cannot write %s\n", path); return; }

    uint8_t sig[8] = {137,80,78,71,13,10,26,10};
    fwrite(sig, 1, 8, f);

    uint8_t ihdr[13];
    for (int i = 0; i < 4; i++) {
        ihdr[3-i] = (uint8_t)((uint32_t)w >> (i*8));
        ihdr[7-i] = (uint8_t)((uint32_t)h >> (i*8));
    }
    ihdr[8]  = 8;  ihdr[9]  = 2;  ihdr[10] = 0;  ihdr[11] = 0;  ihdr[12] = 0;
    _png_chunk(f, "IHDR", ihdr, 13);

    size_t row_bytes = (size_t)w * 3;
    size_t raw_size = (size_t)h * (1 + row_bytes);
    uint8_t *raw = (uint8_t *)malloc(raw_size);
    for (int y = 0; y < h; y++) {
        size_t y_off = (size_t)y * (1 + row_bytes);
        raw[y_off] = 0;
        for (int x = 0; x < w; x++) {
            size_t off = (size_t)y * (size_t)w + (size_t)x;
            size_t dst = y_off + 1 + (size_t)x * 3;
            raw[dst + 0] = r[off];
            raw[dst + 1] = g[off];
            raw[dst + 2] = b[off];
        }
    }

    uLongf comp_len = compressBound(raw_size);
    uint8_t *comp = (uint8_t *)malloc(comp_len);
    if (compress(comp, &comp_len, raw, raw_size) != Z_OK) {
        fprintf(stderr, "[ERROR] PNG compression failed: %s\n", path);
        free(raw); free(comp); fclose(f); return;
    }
    _png_chunk(f, "IDAT", comp, comp_len);
    _png_chunk(f, "IEND", NULL, 0);

    fclose(f);
    free(raw);
    free(comp);
}

/* Dataset function aliases */
#define ki_dataset_read ki_cifar_read
#define ki_dataset_free ki_cifar_free

/* ── Class names (truncated to 7 chars for table alignment) ──── */
__attribute__((unused))
static const char *ki_class_names[KI_NCLASSES] = {
    "airplan","automob","bird","cat","deer",
    "dog","frog","horse","ship","truck"
};

#endif /* KI_LOCAL_H */
