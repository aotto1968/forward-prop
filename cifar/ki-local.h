/*
 * cifar/ki-local.h — CIFAR-10-specific constants + data loader
 * ============================================================
 * For the public Otto Score distribution.
 * Derived from cifar-1/ki-local.h.
 *
 * API:
 *   int ki_dataset_read(ki_Dataset *out);   // 0 on success
 *   void ki_dataset_free(ki_Dataset *data);
 */
#ifndef KI_LOCAL_H
#define KI_LOCAL_H

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════════
 * CONSTANTS
 * ═══════════════════════════════════════════════════════════════════════ */

#define KI_PX                     3072
#define KI_NCLASSES               10
#define KI_DEFAULT_LR             0.01f
#define KI_DEFAULT_STEP_POWER     7.0f
#define KI_DEFAULT_STEP_MODE      STEP_COS_TIME
#define KI_DEFAULT_BATCH_N        128
#define KI_DEFAULT_ENSEMBLE_SEED  ENS_SEED_ONCE
#define KI_COLORS                 3
#define KI_DEFAULT_COLOR          ((1<<COLOR_R)|(1<<COLOR_G)|(1<<COLOR_B))
#ifndef KI_NC
#define KI_NC                     256
#endif
#define KI_NC_TOTAL               (KI_NC * KI_COLORS)
#define KI_PACK                   4

#ifndef NC
#define NC  KI_NC
#endif

#ifndef OT_PRECISION
#define OT_PRECISION      17
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * 8 BLOCK COMPUTATIONS — shared between trainer + samples tool
 * ═══════════════════════════════════════════════════════════════════════ */

#define KI_NBLOCKS 8

static inline int ki_clamp_u8(int v) {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return v;
}

static inline void ki_blocks_from_rgb(int r, int g, int b, uint8_t blocks[COLOR_NB]) {
    unsigned int ru = (unsigned int)r;
    unsigned int gu = (unsigned int)g;
    unsigned int bu = (unsigned int)b;
    blocks[COLOR_MNIST]= 0;
    blocks[COLOR_R]  = (uint8_t)r;
    blocks[COLOR_G]  = (uint8_t)g;
    blocks[COLOR_B]  = (uint8_t)b;
    blocks[COLOR_Y]  = (uint8_t)((ru*77U + gu*150U + bu*29U) >> 8U);
    blocks[COLOR_YL] = (uint8_t)((ru*54U + gu*183U + bu*18U) >> 8U);
    blocks[COLOR_AL] = (uint8_t)((r + g) >> 1);
    blocks[COLOR_AM] = (uint8_t)ki_clamp_u8(128 + (r - g));
    blocks[COLOR_AP] = (uint8_t)ki_clamp_u8(128 + (b - (r + g)/2));
    blocks[COLOR_BL] = (uint8_t)((r + b) >> 1);
    blocks[COLOR_BM] = (uint8_t)ki_clamp_u8(128 + (r - b));
    blocks[COLOR_BP] = (uint8_t)ki_clamp_u8(128 + (g - (r + b)/2));
    blocks[COLOR_RG] = (uint8_t)ki_clamp_u8(128 + (r - g));
    blocks[COLOR_RB] = (uint8_t)ki_clamp_u8(128 + (r - b));
    blocks[COLOR_GB] = (uint8_t)ki_clamp_u8(128 + (g - b));
    {   float dx = 2.0f * (float)r - (float)g - (float)b;
        float dy = 1.0f * ((float)g - (float)b);
        float hue_f = atan2f(dy, dx) / (2.0f * 3.14159265358979323846f);
        int hue = (int)((hue_f + 0.5f) * 255.0f);
        if (hue < 0) { hue = 0; }
        if (hue > 255) { hue = 255; }
        blocks[COLOR_H] = (uint8_t)hue;
    }
    {   int bv = b;
        int mx = r, mn = r;
        if (g > mx) { mx = g; } if (g < mn) { mn = g; }
        if (bv > mx) { mx = bv; } if (bv < mn) { mn = bv; }
        blocks[COLOR_S] = (uint8_t)(mx - mn);
    }
    blocks[COLOR_C] = (uint8_t)r;
    blocks[COLOR_CL] = (uint8_t)((g + b) >> 1);
    blocks[COLOR_CM] = (uint8_t)ki_clamp_u8(128 + (g - b));
    blocks[COLOR_CP] = (uint8_t)ki_clamp_u8(128 + (r - (g + b)/2));
}

/* ═══════════════════════════════════════════════════════════════════════
 * CIFAR-10 DATA STRUCT + LOADER
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    int num_images;
    int rows;
    int cols;
    int pixels;
    int channels;
    float *X;
    uint8_t *X_raw;
    uint8_t *y;
} ki_ImageData;

typedef ki_ImageData ki_Dataset;

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
    if (nread != file_size) { free(buf); return NULL; }
    return buf;
}

static int ki_cifar_read(ki_ImageData *out) {
    const char *candidates[] = {
        "data/cifar-10-batches-bin",
        "../data/cifar-10-batches-bin",
        "../../data/cifar-10-batches-bin",
        NULL
    };
    const char *data_dir = NULL;
    for (int i = 0; candidates[i]; i++) {
        char test[512];
        snprintf(test, sizeof(test), "%s/data_batch_1.bin", candidates[i]);
        if (access(test, R_OK) == 0) { data_dir = candidates[i]; break; }
    }
    if (!data_dir) {
        fprintf(stderr, "[FATAL] Cannot find CIFAR-10 data.\n");
        fprintf(stderr, "  Run: bash fetch_cifar10.sh\n");
        return -1;
    }
    memset(out, 0, sizeof(*out));
    out->rows = 32; out->cols = 32; out->channels = 3; out->pixels = KI_PX;

    const char *train_files[] = {
        "data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin",
        "data_batch_4.bin", "data_batch_5.bin"
    };
    int n_train = 5, train_total = 0;
    uint8_t *all_raw = NULL; size_t all_raw_cap = 0, all_raw_used = 0;

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
        for (int p = 0; p < KI_PX; p++)
            out->X[(size_t)i * (size_t)KI_PX + (size_t)p] = ((float)rp[1 + p] / 127.5f) - 1.0f;
        rp += 3073;
    }
    free(all_raw);

    rp = test_batch;
    for (int i = 0; i < 10000; i++) {
        int idx = train_total + i;
        out->y[idx] = rp[0];
        memcpy(out->X_raw + (size_t)idx * KI_PX, rp + 1, KI_PX);
        for (int p = 0; p < KI_PX; p++)
            out->X[(size_t)idx * (size_t)KI_PX + (size_t)p] = ((float)rp[1 + p] / 127.5f) - 1.0f;
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

#define ki_dataset_read ki_cifar_read
#define ki_dataset_free ki_cifar_free

#endif /* KI_LOCAL_H */
