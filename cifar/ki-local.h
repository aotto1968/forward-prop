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

/* ── Counter type for target/offset/step (overridable, default int32_t) ── */
#ifndef COUNTER_TYPE
#define COUNTER_TYPE int32_t   /* int32_t (fixed-point) or float */
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * CONSTANTS — CIFAR-10
 * ═══════════════════════════════════════════════════════════════════════ */

#ifndef KI_DATASET_ID
#define KI_DATASET_ID             1       /* unique for cache key (overridable) */
#define KI_DATASET_NAME           "CIFAR-10"
#ifndef KI_BIT_WIDTH
#define KI_BIT_WIDTH              8       /* bits per pixel (8/16/24/32) */
#endif
#define KI_PX_PER_CONT_W          (32 / KI_BIT_WIDTH)  /* 4 bei 8bit, 2 bei 16bit, 1 bei 32bit */
#define KI_ROWS                   32
#define KI_COLS                   32
#endif
#define KI_PX                     3072    /* 32 × 32 × 3 = 3072 pixels per image */
#define KI_NCLASSES               10
#define KI_DEFAULT_LR             0.01f   /* → step = 0.005 × 131072 = 655 (high lr leads to strong trn and weak evl */
#define KI_DEFAULT_STEP_POWER     7.0f    /* higher yields smaller trn */
#define KI_DEFAULT_STEP_MODE      STEP_COS_TIME
#define KI_DEFAULT_BATCH_N        128     /* optimum */
#define KI_DEFAULT_ENSEMBLE_SEED  ENS_SEED_ONCE
#define KI_COLORS                 3       /* R, G, B as independent samples, each packed 4px/cont */
#define KI_DEFAULT_COLOR          ((1<<COLOR_R)|(1<<COLOR_G)|(1<<COLOR_B))  /* CIFAR default: raw R+G+B (bits 1,2,3) */
#define KI_NC                     (KI_PX / KI_COLORS / KI_PX_PER_CONT_W)  /* Container per color: 1024/4=256 bei 8bit, 1024/1=1024 bei 32bit */
#define KI_NC_TOTAL               (KI_NC * KI_COLORS)
#define KI_PACK                   KI_PX_PER_CONT_W

#ifndef NC
#define NC  KI_NC
#endif

#ifndef OT_PRECISION
#define OT_PRECISION              17
#endif

/* For mlp-flt32-trn-*-adam.c (old trainer) */
#ifndef KI_BITS_PER_CONT
#define KI_BITS_PER_CONT          32
#endif
#ifndef KI_MODEL_DIR
#define KI_MODEL_DIR              "models"
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

    /* Hue — atan2(2R-G-B, G-B) normiert auf 0..255 */
    {   float dx = 2.0f * (float)r - (float)g - (float)b;
        float dy = 1.0f * ((float)g - (float)b);
        float hue_f = atan2f(dy, dx) / (2.0f * 3.14159265358979323846f);
        int hue = (int)((hue_f + 0.5f) * 255.0f);
        if (hue < 0) { hue = 0; }
        if (hue > 255) { hue = 255; }
        blocks[COLOR_H] = (uint8_t)hue;
    }
    /* Saturation — max(R,G,B) - min(R,G,B) */
    {   int bv = b;  /* copy to avoid -Wshadow */
        int mx = r, mn = r;
        if (g > mx) { mx = g; } if (g < mn) { mn = g; }
        if (bv > mx) { mx = bv; } if (bv < mn) { mn = bv; }
        blocks[COLOR_S] = (uint8_t)(mx - mn);
    }
    /* Contrast (computed by Sobel in load_input — placeholder here) */
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
    int n_train;       /* default training count (set by loader) */
    int n_eval;        /* default eval count (set by loader) */
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
        "www/data/cifar-10-batches-bin",        /* web-accessible copy */
        "../www/data/cifar-10-batches-bin",
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
    out->n_train  = 50000;
    out->n_eval   = 10000;

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

/* ── Encoding aliases (dataset-specific) ──────────────────────── */
#define KI_COMMON_ALIAS_LOOKUP
static const char *ki_encoding_alias_lookup(const char *name) {
    if (strcasecmp(name, "ey-c") == 0) return "r=up,cl=down,cm=sig,cp=sig";
    if (strcasecmp(name, "ey-c-2") == 0) return "cl=down,cm=sig,cp=sig";
    if (strcasecmp(name, "ey-s-2b") == 0) return "lbp=gamma,dog=sig,var=exp";
    if (strcasecmp(name, "ey-s-2c") == 0) return "range=exp,var=log,dir=tri";
    if (strcasecmp(name, "best-mnist") == 0) return "exp,log,log";
    if (strcasecmp(name, "top-rgb") == 0) return "r=down,g=down,b=down";
    if (strcasecmp(name, "latest-2") == 0) return "g=down,bl=gamma,bm=sig,bp=sig,b=sqrt,al=down,am=sig,ap=sig,h=lin,c=cbrt,gb=sig";
    /* old try */
    if (strcasecmp(name, "ey-b") == 0) return "g=up,bl=down,bm=sig,bp=sig";
    if (strcasecmp(name, "ey-a") == 0) return "b=up,al=down,am=sig,ap=sig";
    if (strcasecmp(name, "ey-h") == 0) return "h=down,c=exp,gb=sig";
    if (strcasecmp(name, "ey-s") == 0) return "lbp=up,dog=sig,var=exp";
    if (strcasecmp(name, "old") == 0) return "ey-b,ey-a,ey-h,ey-s";
    /* maj=3 optimized */  // --hiddenN 128 --epochsN 10 --ensembleN 1 --xform id
    if (strcasecmp(name, "ey-a-3") == 0) return "al=sqrt,am=sig,ap=sig";
    if (strcasecmp(name, "ey-b-3") == 0) return "bl=gamma,bm=down,bp=sig";
    if (strcasecmp(name, "ey-h-3") == 0) return "h=down,c=gamma,gb=sig";
    if (strcasecmp(name, "ey-s-3") == 0) return "dir=tri,range=sqrt,lbp-rg=cbrt";
    if (strcasecmp(name, "performance") == 0) return "ey-b-3,ey-a-3,ey-h-3,ey-s-3";
    /* maj=1 optimized by otto */
    if (strcasecmp(name, "ey-b-1") == 0) return "bl=gamma,bm=lin,bp=sig";
    if (strcasecmp(name, "ey-a-1") == 0) return "al=sqrt,am=sig,ap=sig";
    if (strcasecmp(name, "ey-h-1") == 0) return "h=up,c=gamma,gb=sig";
    if (strcasecmp(name, "ey-s-1") == 0) return "dir=lin,range=gamma,lbp-rg=exp";
    if (strcasecmp(name, "performance-1") == 0) return "ey-b-1,ey-a-1,ey-h-1,ey-s-1";
    /* maj=1-optimized variants (2026-07-18 sweep) */
    if (strcasecmp(name, "ey-m1-b") == 0) return "bl=gamma,bm=lin,bp=sig";
    if (strcasecmp(name, "ey-m1-a") == 0) return "al=sqrt,am=sig,ap=sig";
    if (strcasecmp(name, "ey-m1-h") == 0) return "h=up,c=gamma,gb=sig";
    if (strcasecmp(name, "ey-m1-s") == 0) return "dir=lin,range=gamma,lbp-rg=up";
    if (strcasecmp(name, "performance-maj1") == 0) return "ey-m1-b,ey-m1-a,ey-m1-h,ey-m1-s";
    /* summary */
    if (strcasecmp(name, "latest") == 0) return "performance";
    return NULL;
}



/* ── Random pixel shuffle map (fixed seed 42, pairwise permutation, per-plane 32×32) ── */
static const uint16_t ki_xform_shuffle_map[1024] = {
     964,   18,  184,  392,  149,  843,   53,  950,  355,  904,  933,  450,  744,  740,  985,  395,
     454,  483,  298,  202,  523,  622,  683,  231,  691,  509,  946,  213,  247,  866,  487,  899,
     241,  256,   85,   49,  428,  857,  582,  679,  635,  934,  185,  499,  188,  515,  847,   33,
     229,  770,  876,  581,  192,  221,  150,  374,  673,  924, 1009,  814,  625,  752,  198,  361,
     532,  793,  585,  907,  659,  600,  301,  360,  605,  485,  113,  419,  823,  619,  680,  262,
     919,  195,  290,  611,  848,   91,  676,  830,  554,    5,  494,   16,  902,  785,    1,   38,
     174,  193,  807,  398,  508,  404,  694,   76,  891,  411,  138,  885,  350,  211,  717,  251,
     342,  365,  901,  769,  177,  688,  988,  920,  525,  741,  264,  976,  330,  917,   34,  116,
     948,  297,  804,  328,  702,  678,  260,  571,  935,   56,  386,  403,    4,  513,  480,  121,
     369,  892,  446,  974,   61,  755,  860,  354,  449,  751,  522,  400,  799,  210,   66,  482,
     872,  958,  959,  637,  417,  941,  788,  495,  812,  418,  334,  775,  423,  652,  493,  507,
     651,  969,  926,  189,  539,  760,  596,  157,  287,  915,  456,  700,  577,  997,  731,  835,
     840,  364,   92,  536,  325,  436,  870,  798,  792,  572,  414,   17,  818,  327,  803,  136,
      88,  442,  200,  144,  813, 1006,  703,  722,  916,  381,  385,  670,  178,  895,  372,  474,
     415,  506,  457,  510,   50,  447,  424,  772,  308,  173,  147,  435,  986,  148, 1000,   79,
     272,  461,  831,   29,  542,  191,  795,  302,  560,  289,  900,  333,  496,  573,  119,  708,
     653,  425,   64,  709,  903,  318,  617,  353,  715,   20,  476,  599,  179,   26,  568,  914,
     922,  129,  629,  878,  282,  472, 1020,  232,  819,  466,  316,  938,  375,  862,  236,  647,
     734,  951,  293,  589,  201,   13,  538,  455,  239,   84,  444,  172,  311,   14,  349,  491,
     467,  624,  249,  380,  279,  782,  925,  833,  956,  180,  967,  628,  451,  850,   10,  299,
     199,  695,  874,   31,  329,  431,  277,  657,  586,  396,  742,  427,   78,  954,  834,  641,
     648,   41,  909,  537,  645,  139,  681,  439,  534,  750,  140,  644,  320,  728,  489,  865,
     326,  126,  132, 1013,  401,  939,  965,  443,  614,  258,  313,  636,  162,  304,  286,  884,
     761,  544,   68,  397,  662,  960,  815,  908,  237,  265,  952,   90,  859,  886,   54,  943,
     968,  579,  626,  124,  531, 1021,  771,  458,  970,  723,  294,  936,  212,  527,   37,  393,
     528,  897,  125,   23,   12, 1001,  101,  115,  463,  975,  120,  384,   47,  171,  317,  324,
     962,  263,  145,   43,  337,   86,  376,  407,  406,  547,  953,  664,  345,  430,  226,  937,
     164,  335,  501,  940,  504,  477,  107,  880,  155,  490,  426,  280,  207,  154,  606,  557,
    1010,  383,  756,  668,  475,  690,   98,  757,  638,  561,  356,  206,  912,  630,  575,  548,
     259,  153,  684,  727,   63,  434,  516,  503,  837,  810,  530,  601,   35,  977,  186,  693,
     343,  377,  927,  533,  368,  587,  275,  710,  839,  266,  805,  502,  615,   22,  559,   97,
     295,  460,  524,  765,  261,  310, 1004,  160,  402,  990,  176,  666,  817,  716,  801,  612,
     821,  631,  594,   28,  339,  409,  359,  864,  945,  921,  336,  767,  197,   24, 1018,  942,
     312,  358,  471,  204,  227,  481,   39,  391,  110,  856,  836,  707,  421,  340,  183, 1022,
     422,  181,  928,  766,  982,  984,  720,  105,  697,  588,  366,  639,  929,  315,   42,  130,
     440,  151,  607,   36,  994,  152,  842,  800,  620,  437,  564,  452,  809,  268,  133,  341,
     827,   45,  283,  672,  973,   52,  845,  802, 1002,  595,  208,  351,  215,  288,  278,  978,
     844,  109,  118,  966,  137,  106,  306,  753,    8,  846,  309,  556,  763,  816,  291,  378,
    1012,  649,  357,  732,  627,  518,  682,  550,   75,  102,  576,  881,    9,   74,  632,  468,
     307,  134,  883,  783,  244,  713,  209,  267,  894,  535,  578,  869,  993,   83,  634,   40,
     608,  592,  877,  583,  122,  413,  240,  743,  609,  789,  190,   82,  520,  660,  829,  806,
     514,  488,  871, 1014,   58,  321,  711,  555,  593,  796,  222,  303,  705,  158,  498,  569,
     719,  292,  465,  779,  399,    2,  910,  168,  745,  893,  218,  492,  888,  726,  170,  242,
    1005,   15,  553,  667,   51,  712,  563,  453,   77,  255,  911,  737,  187,  853,  143,  991,
     838,  725,  947,  254,  111,  347,  412,   62,  100,  661,  669,  689,   55,  478,  420,  776,
     362,  441,  674,  971,   96,  194,  416,  729,  248,  484,  590,  584,  135,  774,  219,  790,
     685,  285,  243,  526,  748,  338,  896,   69,  898,   60,  646,  724,   72,    7,  992, 1016,
     235,  567,  602,   21,  346,  851,  230,  253,  123,  462,  529,  448,  855,  996,  687,  868,
     319,  730,  205,  706,  963,  861,  808,  987,  433,  621,  540,  879,  169,  562,  486,  675,
     131,  128,  784,  545,  778,   70,  497,  749,   87,  957,  889,  246,   59,  245,  314,  371,
     832,   19,  500,  331,  613,    0,  543,  797,  552,  165,  822,  382,  858,  998,  918,  820,
     654,  305,  640,  108,  519,  182,  780,  512,  873,  736,  739,    3,  932,  161,  445,  955,
     114,  980,  656,  787,  923,  768,  961,  698,  117,  738,  930,   11,  566,  257,  541,  479,
     852,  390,  394,   65,  610,  981,  696,  811,  983,  642,  156,  112,   48,  773,   93,  505,
     521,  141,  905,  949,  408,  597,  438,  598,  764,  269,  551,  999,  762,  252, 1017,  271,
     146,  469,  658,  405,  511,  671,  217,  322,  735,  580,  216,   67,  274,  410,  323,  824,
    1007,  841,  234,   57,  794,  786,  863,  332,  701,  224,  570,  704,  655,  276,  388,  473,
     167, 1019,  746,  546,  175,  931,  623,   73,  663,  699,  995,  273,  686,  214,  363,  379,
     166,  373,  854,  650,  464,  979,  972,  103,  887, 1003,  875,   81,  296,  791,  233,  677,
      46,   71,  721,  196,  591,  370,  882,  906,  633,  643,  849,  300,  565,   80,  387,  944,
     127,  549,  470,  747,   44,  826,  270,  618,  352,  867,  367,   99,  389,   94, 1015,  344,
     781,  220,  159,  989,  348, 1008,  714,  163,  825,  777,    6,  890,  828,  284,  603,  459,
     225,  429, 1011,  718,  665,  733,  203,  574,   27,  616,  517,  238,  223,   95,   30,   32,
     432,  604,   89,  558,  913,  758,  692,  104,  754,  142, 1023,  250,  281,  759,   25,  228,
};

void ki_xform_shuffle_apply(uint8_t *restrict out,
                             const uint8_t *restrict in,
                             int n, int ch) {
    for (int pl = 0; pl < ch; pl++) {
        const uint8_t *s = in + (size_t)pl * (size_t)n;
        uint8_t *d = out + (size_t)pl * (size_t)n;
        for (int i = 0; i < n; i++)
            d[i] = s[ki_xform_shuffle_map[i]];
    }
}

/* ── Additional shuffle maps (seeds 1..10, per-plane 32×32) ── */

static const uint16_t ki_xform_shuffle1_map[1024] = {
     197,  714,  242,  169,  559,  137,  735,  503,  991,  155,  682,  493,   52,  444,  396,  419,
     704,  495,  676,  479,  928,  454,  655,  586,  757,   47,  703,  652,  787,  219,  233,  608,
     243,  363,  851,  820,  409,  506,   95,  619,  863,   34,  825,  294,   73,  827,  455,  239,
     246,  987,  930,  749,  633,  446,   58,  722,  362,  900,   54,  673,   82,  269,  394,  358,
     510,  345,  487,  476,  812,  892,  541,   49,  207,  776,  407,  899,  648,  764,  356,  864,
     397,  206, 1016,  822,  308,  532,  612,  826,  969,  146,  158,  370,  768,  254,  131,  798,
     107,    4,  737,  585,  121,  457, 1021,  904,  915,  241,  244,  647,  934,   50,  972,   13,
     783,  597,  474,  478,  624,  125,  613,  889,  357, 1007,  824,  360,  295,  878,  573,  523,
     919,  280,  610,  417,   37,  186,  306,  731,  730,  896,  992,  617,  893,  606,    7,  852,
     147,   56, 1006,  781,  371,  887,  966,  143,  100,   59,  918,  799,  611,  895,  715,  293,
     999,  266,  323,  542,  726,   19,  950,   68,  809,  651,  641,  135,   53,  484,  767,  447,
     180,  791,  311,   23,  176,  350,  881,  481,  706,  285,  113,  168,  267,  380,  964,  338,
     765,  260,   25,  594,  654,   11,  459,  437,  386,  659,  668,  921,  427,   77,  891,  996,
     316,  452,  271,   67,   89,  723,  154,  699,  208,  931,  530,  438,  216,  558,  200, 1003,
     199,  935,   39,  231,  184,  854,  467,  698,  744,  343,  793,  276,  393,  482,  740,  284,
     628,  412,  965,  941,  497,  336,  725,  525,  876,  148,   99,  289,  884,  134,  570,  262,
     535, 1009,  677,  976,  962,  705,  401,  945,  831,  590,  565,  643,  844,  894,  498,  839,
     156,  179,  763,  226,  385,    0,  716,  621,  901,  257,  223,  309,  548,  700,  709,   69,
     331,  596,    6,  910,  671,  581,   55,  879,  769,  366,  494,  946,   62,  215,  670,  182,
     509,  666,  477,  378,  299,  527,  139,  739,  472,   75,  145,  440,  843,  885,  124,  384,
      15,  933,  451,  544,  433,   97,  997,  415,  381,  127,  142,  151,  734,  209,  625,  307,
     475,   61,  108,  538,  318,  968,  429,  166,  418,  701,  489,  195,  942,  970, 1011,  421,
     337,  886,  795,  550,  982,   51,  321,  130,  616,  268,  539,  278,  580,  908,  196,  178,
     326,  529,  201,  653,  422,  983,  368,  291,  771,  661,  473,   79,  434,  193,  756,  263,
     546,   63,   43,  304,  869,  377,  391,  359,   84,  947,  282,  490,  453,  152,  159,  988,
     856,  405,   81,  188,  410,  656,  977,  759,  486,  800,  556,  140,  334,  372,  515,  286,
     790,  692,  568,  425,  882,  784,  253,  685,  122,  115,  374,  989,  775,  713,  332,  572,
     599,  270,   90,    3,  602,  119,   45,  729,  978,  684,  687,  562,  141,  833,  502,  450,
    1000,  753,   70,  957,  819,  835,  742,  980,  292,  213,  330,  916,  658,  939,  850,   87,
     973,  428,  202,  953,  829,  865,  435,  960,  277,  748,  281,  571, 1004,  745,  636,   32,
     398,  936,  126,  341,  441,  834,  778,   80,  801,  975,  949,  751,  817,  963,  382, 1012,
     430,  649,  464,  344,  537,  777,  462,  420,  161,  871,   16,  674,  718,  788,  534,  211,
     129,  300,  109,  312, 1022,  144,  153,  183,  911,  710,  631,  549,  513,  543,  157,  951,
     505,  367,  848,  630,  955,   76,  786,  660,  640,  688,  249,   20,  890,  250,   94,  103,
     342,  251,  929,  575,  191,  683,   42,  853,  695,  875,  912,   92,  458,   91,  803,  811,
     369,  252,  203,  314,  588,  981,  187,  265,  364,  888,  595,  220,  256,  114,  466,  324,
      65, 1018,  408,  320,  400, 1013,  327,  615,   10,   27,  609,  986,  212,  956,  322,  488,
     770,  634,  998,  555,  632,  445,  105,  724,  247,  906,  858,  118,  165,  774,  600,  245,
     804,  642,  766,  837,  815,  806,  874,  859,   46,  898,  283,  468,  635,  920,  577,  302,
     842,  273,   83,  579,  229,  818,  584,  727,  708,  927,  883,  117,  349,  732,  150,  389,
      74,  416,   40,  952,  697,  814,  593,  696,  500,  618,  669,  298,  959,  348,  346,  376,
     792,  754,  823,  940,  274,  678,  830,  170,  173,  136,   86,   41,   66,  240,  545,  836,
     547,  897,  560, 1010,   98,  994,  218,  439,  347,  138,  576,  335,  707,  160,  762,  288,
     411,  626,  333,  808,  925,  838,  943, 1002,  681,  106,  504,  198,  990, 1001,  905,  587,
     846,  101,  355,  205,  387,  717,  521,  637,  175,  471,  841,  222,  604,   38,  847,    8,
     133,  258,  578,  954,  162,  733,  317,   78,  810,  313,   48,  217,  128,  305,   60,  752,
     646,  861,  328,  589,  691,  404,  985,  664,  536,  228,  461,  528,  711,  645,  225,  557,
     845,  694,  518,  721,  995,  164,  736,   36,  149,  406,   18,  230,   21,  442,  620, 1008,
     522,  747,  259,  111,  264,  192,  431,  351,  395,  319,   24,  116,  485,  508,  329,  907,
     465,  301,  937,  663,  279,  672,  877,  971,  813,  163,  171,   71,  297,  924,  189,  639,
     112,  862,  255,  287,  773,  772,   14,  463,   17,   85,   72,  689,   33, 1017,  564,   93,
     185,  598,  563,  181,  650,  235,   28,  614,  469,  339,  627,  805,  638,  553,  551,    1,
     354,  913,  365,  832,  496,  423,  516,  872,  567,  583,  373,  492,   57,  436,  210,  574,
     796,  531,  132,  828,  524,  758,  840,  802,  392,    5,  746,  623,  870,  675, 1023,  958,
     361,  591,  352,  526,  414,  237,  909,  552,  204,  789,   12,  232,  514,  172,  174,  662,
     403,  592,  607,  629,  868,  720,  315,   44,  480,   30,  750,  501,  379,  922,  860,  533,
     167,  797,  110,  520,  679,  449,   88,  383,  755,  690,  794,  719,  926,  561,  375,  177,
     680,  424,  413,  816,  761,  248,  491,   35,  873,  603,  402,  517,  866,  511,  903,  601,
     290,  310,  194,  686,  932,  849,  519,  974,  512,  728,  993,  917,  340,  760,  123,  303,
     880,  741,  644,  190,  102,  657,  944,  569,  938,  857,  426,  984,  948,  296,  470, 1014,
     224,  693,  236,  353,  238,  566, 1015,  979,  448, 1019,  227,  540, 1005,  743,  432,  221,
     702,  390,  902,  961,    9,  554,  665,   26,   22,   31,  325,  923,  104,  967,  605,  234,
    1020,  738,  272,  456,  712,    2,  785,  780,  622,  443,  399,  855,  914,   29,  499,   96,
     214,  807,  388,  667,  483,  460,  779,  507,  120,  261,   64,  782,  821,  867,  582,  275,
};

void ki_xform_shuffle1_apply(uint8_t *restrict out,
         const uint8_t *restrict in,
         int n, int ch) {
    for (int pl = 0; pl < ch; pl++) {
        const uint8_t *s = in + (size_t)pl * (size_t)n;
        uint8_t *d = out + (size_t)pl * (size_t)n;
        for (int i = 0; i < n; i++)
            d[i] = s[ki_xform_shuffle1_map[i]];
    }
}

static const uint16_t ki_xform_shuffle2_map[1024] = {
     591,  838,  814,  443,  671,  643,  339,  376,  709,  463,  228,  646,  998,  440,  373,  508,
     937,   62,   22,  951,  714,  863,  718,  146,  406,  410,  397,  795,  301,  782,  901,  556,
      42,  730,  849,  663,   91,  978,  664,  684,   64,  513,  245,  574,  887,  568,  616,  158,
     214,  286,  266,  308,  666,  823,  920,  127,  341,   66,  219,  361,  924,  655,  862,    2,
     485,  506,   92,   99,  398,  328,  629,  843,  206,  448,  842,  729,  805,  290,  299,  609,
     300,  648,  720,  120,  715,  549,  515,  554,  421,  780,  779,  457,  152,  518,   43,  311,
     111,  198,  541,   56,  136,  532,  665,  733,  309,  662,  276,  926,  207,  248,  597,  166,
     704,  264,  417,  613,  816,  626,  593,  617,  608,  594,  917,  151, 1022,  579,  540,  350,
     931,  973,  269,  694,  902,  995,  298,  861,  106,  105,  488,  927,   70,  550,   40,  687,
     712,  436,  669,  454,  302,  956,  480,  405,    9,  189,  153,  390,  431,  923,  799,  845,
     450,    6,  950,  596,   10,  577,  211,   49,  544,  283,  829,  894,  294,  695,  976,  140,
     812,  452,  767,  396,  229,  801,  325,  936,  999,  627,  251,  388,  699,   80,  101,  307,
     425,  853,  941,   87,  446,  289,  889,  785,  900,  216,  825,   26,  932,  839,  691,  340,
     282,  764,  778,  100,  763,  461,  511,  770,  466,  867,  996,   13,  378,  601,  826,  656,
      63,   29,  727,   33,  490,  647,  357,  249,  792,  661,  135,  873,  492,  133,  208,  284,
     827,  235,  306,   51,  584,   15,  138,  322,  178,  754,    7,  386,   35,  331,  678,  123,
     243,  640, 1017,  314,  751,  218,  800,   12,  604,  112,  351,  569,  129,  944,  675,  156,
     787,  348,  363,  749,  871,  354,  961,   55,  717,  231,  776,  197,  210,  615,  865,  205,
    1001,  473,  835,  802,  278, 1005,  858,  242,  896,  660,  116,  377,  583,   59,  295,  447,
    1020,  918,  744,  979,  542,  755,   90,   73,  590,  190,  172,   23,  312,  122,  126,  547,
     387,  505,  870,  460, 1016,  807,  128,  169,  482,  611,   48,  587,  682,  356,  572,  881,
     946, 1000,  145,  344,  199,  384,  623,  379,  277,  736,  223,  333,  916,  183,  940,  179,
      53,   14,   65,  427,  582,  793,  637,  244,  225,  713,  907,  728,  551,  202, 1004,  830,
     820,  343,  983,  318,  859,  415,  634,  790,  360,  365,  711,  831,  693,  654,  635, 1012,
     304,  504, 1003,  175,  645,  539,  193,  453,  281,  137,  439,  689,   71,  423,   94,  191,
     347,  385,   97,  580,  962,  868,  517,  167,  860,  110,  327,  890,  578,  985,  997,  149,
     563,  586,   85, 1006,  438,  171,  469,  182,  163,  109,  899,  194,  507,  419,  705,  432,
      57,  809,  142,  267,  658,  287,  234,  119,  724,  479,   45,  321,  258,  408,   76,  486,
     434,  271,  305,  200,  641,  324,  649,   47,  430,  618,  201,  329,  334,  592,  407,  330,
     908,  821, 1009,  806,  132,   39,  735,  745,  804,  487,  914,  247,  702,  912,  775,  796,
     346,  445,  877,  226,  288,  499,  706,  817,  546,  187,    0,  239,  738,  632,  893,   79,
     948,  798,  614, 1014,  846,  559,   27,  841,  562,  442,  606,  403,  552,  811,  786,  566,
     529,  355,   81,  987,  471,  992,  273,  470,  644,  496,   67,  673,    3,  358,   74,  209,
     478,  968,  525,  565,  493,  707,  876,  141,  934,  659,  567,  204,  600,  939,  220,  395,
     221,  555,   17,  696,  974,  576,  864,  771,  320,   11,  391,  412,  192,  545,  794,   83,
     338,  560,  742,  483,  683,  766,  414,  280,  494,   54,  965,  783,  184,  392,   30,  451,
     477,  148,  367,  413,  240,  990,  161, 1018,  263,  963,  254,  528,  336,  612,  739,  553,
     159,  114,  293,  536,  847,  449,  337,  622,  399,  624,  131,   38,   88,   68,  426,  150,
     558,  970,  364,   72,  761,  589,  599, 1021,  437,  561,  852,  610,  810,  833,  797,  651,
     303,  503,  774,  489,  737,  495,  275,  383,  144,  468,  366,  297,  509,  824,  481,  633,
     719,  681,  710,  400,  342,  906,  888,  125,  759,  260,  124,  404,  353,  769,  672,  401,
     836,  585,  118,  196,  502,  224,   78,  680,  768,  788,  291,    5,  750,  628,  988,  213,
     474,  497,  652,  420,  909,  462,   16,  203,  313,  721,  103,  422,  692,  949,  381,  296,
     884,  625,  121,  531,   19,  374,  903,  679,   96,  185,  925,  984,  428,   52,  747, 1007,
     837,  523,  435,  444,  856,  701, 1010,  981,  657,  548,  317,  953,  475, 1015,  533,  945,
     222,  107,  758,  960,   20,  102,  789,  246,   89,  885,  113,  164,  418,  393,  989,  154,
     819,  977,  933,  854,  883,  262,  848,  726,  882,  639,  233,  866,   82,  256,   75,  993,
     731,  238,  252,  898,  725,  520,  237,  650,  464,   98,  174,  943,  134,   34,  259,  850,
     686,  143,  571,  910,   18,  869,  429, 1011,  268,  265,  878,  147,  335,  952,  959, 1002,
     598,  895,  875,  458,  857,  104,  323,  703,  676,   95,  230,  484,  157,  722,  636,  411,
     773,  270,   46,  757,  619,  784,  564,  459, 1013,   31,  886,  345,  292, 1023,  765,  760,
     642,  892,  352,    4,   37,  155,  253,  813,   44,  603,  394,    1,  708,  535,  188,  955,
     160,  832,  130,  261,  382,   21,  746,   41,   25,   69,  117,   84,  971,  688,  935,  176,
     964,  371,   58,   32,  777,  734,  980,   61,  215,  844,  250,  272,  967,  534,  942,  872,
     698,  232,  605,  279,   50,  668,  588,   60,  108,  762,  195,  911,    8,  919,  349,  840,
     803,   77,  638,  700,  375,  524,  500,  212,  748,  319,  416,  602,  630,  667,  519,  530,
     575,  516,  982,  723,  818,  310,  316,  491,  791,  991,  631,  897,  170, 1019,  716,  834,
     969,  227,  674,  498,  467,  741,  570,  743,  581,  359,  938,  921,  465,  904,  677,  947,
     957,  851,  527,  512,  510,  285,  501,  255,  543,  670,  472,  756,  732,  409,  772,  165,
     994,  879,  370,  362,  607,  808,  986,  781,  930,  928,  537,  752,  424,  815,  456,  915,
     186,  975,  690,  526,  368,  966,  522,  139,  177,  332,  180,   24,  236,  241,  181,  573,
     168,  538,  905,  913,  433,  389,  929,  326,  954,  476,  372,   28,  891, 1008,  922,  274,
     514,  455,  958,  557,  380,  972,  521,  880,  740,  822,  402,  653,  441,  162,  697,  595,
      36,  621,  217,  620,  257,  315,  874,  685,  828,  753,  173,  855,  369,   86,   93,  115,
};

void ki_xform_shuffle2_apply(uint8_t *restrict out,
         const uint8_t *restrict in,
         int n, int ch) {
    for (int pl = 0; pl < ch; pl++) {
        const uint8_t *s = in + (size_t)pl * (size_t)n;
        uint8_t *d = out + (size_t)pl * (size_t)n;
        for (int i = 0; i < n; i++)
            d[i] = s[ki_xform_shuffle2_map[i]];
    }
}

static const uint16_t ki_xform_shuffle3_map[1024] = {
     421,  225,  705,  207,  583,  923,  927,  253,  192,  766,  212,  479,  831,   83,  896,  469,
     450,  914,  201,  633,  979,  707,  473,  401,  256, 1013,  890,  677,  464,  451,  454,  563,
     631,  720,  740,  342,  210,  871,  287,  474,  699,  424,  387,  238, 1000,  722,  174,  185,
     516,  525,  312,  833,  528,  463,  457,  867,  571,  733,  324,  921,  412,  954,  704,  689,
     249,   48,  364,  434,  432,  337,  870,   25,  115,  534,  567,  570,  658,  666,  391,  855,
       2,   89,  898,  645,  162,  156,  507,  279,  262,  209,  566,  483,  894,  661,  199,  143,
     526,  218,   71,  975,  574,   38,  148,  842,  646,  145,  513,  491,  724,  825,  500,  511,
     601,  456,  999,   42,  924, 1021,  416,  991,  309,  355,  328,  142,  901,  365,  405,  683,
     976,  197,  186,  388,  947,  292, 1017,  785,  545,  656,  393, 1007,  415, 1008,  497,  147,
     266,  892,  838,  790,  377,  282,  413,  865,  834,  281,   40,  379,  316,  721,  719,  879,
      66,  576,    0,  476,  219,  518,  233,  443,  114,  290,  351,  684,  578,  674,  459,  112,
     503,  176,  946,  502,  612,  296,  664,  926,   53,  536,  907,   17,  179,  990,  836,  100,
      41,  809,  205,  302,  762,   10,  313,   80,  555, 1015,  193,  178,  439,  788,   16,  644,
     303,  849,  741,  217,  595,    8,  198,  267,  468,  680,  565,  220,  177,  259,  949,   84,
      82,  922,  920,  482,  713,   19,  639,   59,  919,  738,    6,   12,  236,  792,  804,  777,
     261,  960,  782,  228,  168,  371,  144,  750,  113,  242,  165,  931,  301,  272,    9,  108,
     596,  597,  235,  813,  251,  119,  988,  944,  885,  362,  472,  616,  913,  158,  504,  314,
     208,  709,  961,  726,  407,  394,  103,  370,  969,  786, 1009,  763,  801,   74,  982,  319,
     213,   96,  974,  928,  288,  418,  775,  835,   77,  608,  992,  356,  325,  651,  916,  206,
     170,  791,  505,  575,  315,  832,  466,  800,  778,  780, 1004,  138,  998,  636,  558,  392,
     294,  390,  283,   92,  366,  758,  681,  520,  486,  234,  965,  711,  153,  284,  215,  983,
     935,  703,  402,   72, 1010,   55,   63,  701,  769,  789,   91,   76,  628,  350,  815,  693,
     363,  408,  150,   30,  180,  260,  252,  986,   58,  862,  274,  289,  753,  806,  934,  414,
     458,  821,  346,  339,  496,  747,  188,  829,  768,  729,  932,   49,  672,  784,  224,   57,
     221,  814,  878,  140,  817,  981,  278,  904,  200,  591,  380,  498,  760,  781,   39,  248,
     194,  690,  169,  171,  985,  611,  172,  529,  449,  749,  231,  607,  460,  149,  135,  638,
      60,  794,  653,  373,  811,  911,   93,  627,  756,   86,  452,  592,  621,  667,  318,   95,
     844,  837,  470,   97,  686,  613,  490, 1016,  767,  345,  397,  367,  331,  120,  670,  462,
      23,  375,  400,  716,  761,  589,  973,  246,   14,  739,  918,  895,  427,  808, 1005,  569,
     550,  624, 1002,  357,  214,  908,  442,  966,  538,  663,  662,  679,  551,  816,  543,  617,
     411,  660,  956,  828,  157,  579,  957,  510,  489,  950,  552,  304,  211,  891,  810,  343,
     151,  987,  772,  725, 1003,  586,  118,  478,  161,  263,  299,   75,  560,  573,  818,  951,
     190,  453,  521,  509,   98,  166,  152,  410,  717,  900,  826,  164,  422,  240,  774,  559,
      85, 1014,  358,  997,  376,  912,  403,  465,  255,  540,  247,   79,  184,  332,  668,  102,
     109,  682,  541,  523,   37,  955,  306,  906,  903,  117,   26,  577,  673,   24,  183,  384,
     101,   94,  897,  887,  622,  963,  146,  655,    5,  845,  544,  326,  111,  822,  744,  869,
     727,  124,  268,  710,  383,  333,   28, 1020,  436,  757,  435,  863,  276,  868,  751,  696,
      18,  604,  428,  430,  675,  508,  381,  676,  840,   44,  610,  917,  852,  382,  859,  182,
     580,  754,  389,    3,  568,  130,  126,  438,  336,  743,  271,  369,  311,  880,  398,  850,
     936,   73,  742,  160,  323,  805,  514,  527,   88,  708,  494,   70,  652,  905,   47,  129,
      56,    1,  330,  971,   52,  433,  909,  539,  632,  771,  254,  728,   51,  409,  584,  945,
     499,  872,  131,  280,  245,  685,  847,  547,  125,  866,  226,  360,  802,  764, 1001,  258,
     128,  702,  116,  626,   54,  718,  549,  537,  823,  860,  770,  783,  515,  175,  746,  492,
    1011,   50,  958,  440,  203,  524,   35,  883,  250,  122,  293,  602,   69,  232,  349,  556,
     295,  531,  678,  796,   33,  773,  846,  444,  522,  637,  488,    4,  204,  861,  423,  902,
     419,  581,  429,  297,  426,  839,  354,  475,  748,  530,  864,  941,  132,  803,  353,  634,
      87,  657,  348,  827,  187,  588,  824,  195,  542,   34,  123,  230,  841,  461,  884,  223,
     765,  929,   81,  173,  448,  229,  691,  341,  329,  793,  104,  953,  714,  665,  445,  694,
     191,  335,  244,  972,  669,  227,  512,  873,  134,  752,  700,  915,   27,  107,  854,  307,
     270,  848,  799,  980,  320,  189,  372,  181,  327,  615, 1022,  305,  467,  643,  257, 1019,
      21,  692,   62,  959,   22,  501,  755,  877,  723,  623,  812,  695,  361,  477,  340,  642,
     910,   61,  136,  647,  603,  630,  939,  893,  322,  593,  889,  964,   11,  858,  561,  310,
     533,  989,  447,  968,  243,  732,  737,  649,  441,  277,  940,  830,  635,  105,  572,  697,
     970,  659,  938,  532,  471,  385,  882,  141,  368,  321,  347,  978,  706,  159,  269,  625,
     298,  933,  202,   32,  548,  614,  110,   78,    7,  317,  952,  241,  517,  285, 1006,  338,
     600,  735,  386,   46,  779,  629,  619,   45,  121,  425,  787,  962,  300,   20,  994,  420,
      68,  819,  352,   90,  495,  996,  874,  493,   64,  127,  291,  273, 1018,  851,  648,  216,
     671,  730,  106,  582,  585,  554,  334,  995,  715,  167,  712,  984, 1012,  286,  876,   29,
     698,  344,  925,  993,  598,  417,  599,  546,  359,  587,  395,  853,  519,  431,  977,  875,
     641,  797,  446,  688,  264,  222,  506,  139,   36,   99,  374,  967,  137,  455,  590,  820,
     745,  404,  437,  943,  807,  731,  396,  899,  942,  736,  609,  484,  275,  886,  843,   31,
     798,  308,   43,  605,  776,  163,   65,  795,  687,   15,  759,  399,  535,  948,  888,  155,
     650,  237,  154,  881,  654,  406, 1023,  562,  856,  553,  481,  734,  196,  239,  564,  265,
     480,  857,  930,   13,  620,   67,  594,  640,  485,  618,  937,  378,  133,  557,  606,  487,
};

void ki_xform_shuffle3_apply(uint8_t *restrict out,
         const uint8_t *restrict in,
         int n, int ch) {
    for (int pl = 0; pl < ch; pl++) {
        const uint8_t *s = in + (size_t)pl * (size_t)n;
        uint8_t *d = out + (size_t)pl * (size_t)n;
        for (int i = 0; i < n; i++)
            d[i] = s[ki_xform_shuffle3_map[i]];
    }
}

static const uint16_t ki_xform_shuffle4_map[1024] = {
     830,  675,   64,  289,   24,  362,  711,  680,  516,  177, 1003,  360,  343,  314,  577,  730,
     885,  951,  664,  789,  439,  833,  999,  946,  249,  945,   51,  786,  812,  834,  784,  780,
     769,  384,   16,  760,  122,  561,  572,   98,  258,  692, 1013,  712,  179,  690,   39,  327,
     790,  835,  598,   33,  766,  879,  765,  230,  270,  676,  928,  132,  576,  908,   49,  505,
     393,  191,  228,  854,  688,  463,  694,  242,  636,  546,  442,  713,  836,  654,  914,  866,
     273,  280,   65,  977,  618,   21,  497,  528,  697,  920,  262,  163,  989,  526,  101,  537,
     795,  218,  225,  748,  221,  175,  579,  128,  845,  979,  519,  386,  960,  150,  316,  315,
     214,  281,  580,  504,   55,  288,  350,  545,  761,  679,  922,  429, 1012,  326,  388,   11,
     551,  856,  921,  610,   72,  616,  912,   71,  548,  987,  487,  821,  746,  378,  403,   12,
     849,  170,  342,  972,  217,  347,  387,   40,  229,  294,  710,  934,  474,  126,   50,  193,
    1004,  301,  925,    8,  137,  197,  530,   66,  947,  146,  293,  900,  236,  408,  479,  607,
     592,  241,  629,  996,  189,   73,  154,  201,  222,  938,  515,  461,  149,  391,  111,  858,
     747,  398,  755,  250,  619,  627,  828,  740,  392,   46,  335,  896,  772,  232,  905,  597,
     817,  486,  359,  246,  325,  285,  525,  340,  787,  605,  481,  190,  955,  805,  775,  385,
     720,  744,  809,  352,  471, 1017,  485,  707,  804,  593,  493,  460,  204,  867,  695,  517,
     138,  240,  706,  187,  418,  402,   75,  395,   30,    6,  379,  824,  617,  929,  785,  600,
     681,  114,   57,  499,  862,  851,  927,  612,  665,  305,  112,  604,  480,  954,  400,  534,
     916,  511,  185,  414,  276,  966,  457,  646,  394,  732,  815,  363, 1001,   29,   82,  709,
      25,  494,  491,  807,  591,  492,  661,  366,  333, 1021, 1016,  564,  963,   38,  980,  843,
     567,  151,  436,  383,  671,  723,  988,  369,   85,  104,  291,  233,  552,  174,   58,  406,
     110,  129,  476,  931,  341,  645,  336,  728,  313,  544,  906,  614,  736,  134,  467,   15,
     933,  459,  143, 1020,  107,  458,  800,  209,  321,  208,  767,  141,  354,  245,  382,  976,
      89,  164,  731,  106,  701,  353,  750,  698,  389,  261,  529,  252,  770,  863,  907,  718,
     735,  435,  842,  509,  522,  788,  678,  109,   74,  417,  625,  651,   17,  375,  776,  686,
     140,  703,  705,  893,    0,  794,  590,  477,  428,  356,  818,  308,  673,  820,  995,  102,
     416,  127,  910,  759,  279,  936,  822,   22,  637,  595,  774,  660,  909,  269,  923,  778,
     437,  456,  603,   53, 1005,  120,  699,  648,  992,  568,  468,  623,  512,  302,  892,  569,
     983,  743,  998,  323,  808,   19,  377,  599,  211,  674,  902,  872,  967,  782,  628,  633,
      96,  638,  271,  426,  975,  944,  777,  194,  357,  496,  445,  361,  677,  231,  917,  453,
      94,  272,  594,  466,  573,  653, 1002,  472,  724,  956,  513,  311,  838,   59,  708,  884,
     588,  745,  926,  852,  991,  566,  700,   14,  729,   31,  986,  878,  985,  813,  941,  155,
     349,  742,  757,  756,  131,  455,  447,  570,   86,  797,   34,  871,  810,  521,  666,  758,
     157,  565,  514,  555,    2,  825,  527,  133,  424,  543,  768,  874,   62,  277,  124,  932,
     324,  796,  716,  841,  161,  640,  876,  997,  407,  167,   48,  135,  306,   70,  244,  659,
      67,  488,  584,  644,  655,   93,  737,  829,  959,   69,  523,  159,   28,  685,  733,  847,
     469,  553,  911,  827,  727,  639,  448,  290,  631,  410,  626,  682,  940,  542,  550,  601,
     235,  376,  984,  226,  501,  891,  415, 1022,  952,  814, 1000,  781,  606,  587,  507,  186,
     855,  100,  367,   90,  536, 1014,   23,  962,  178,  658,  973,  337,  183,  779,   78,  771,
     358,  589,  202,  801,  880,  981,  147,  881,  443,  136,  348,  370,  611,  257,  144,  753,
     452,  913,  969,  937,  581,  478,  557,  320,  726,  719,    1,  184,  897,  883,  864,  421,
     251,  533,   77,  791,  495,  615,  741,  113,  558,  123,  950,    5,  831,    4,  554,  264,
     935,  773,  877,  152,  180,  166,  380,  739,  904,  121,  432, 1018,  309,  652,  919,  643,
     918,  803,  425,  444,  524,  374,  994,  117,  974,   97,  409,  498,  239,  970,  318,  210,
     722,  464,  556,   87,  751,  355,  304,   10,  535,   81,  844,   42,  142,  715,  898,  574,
     500,  115,  608,   35,   80,   27,  450,  539,  763,  575,  338,  622,  971, 1011,  811,   54,
     563, 1006,  857, 1019,  118,  263,  624,   13,  160,  216,    3,  223,  399,  413, 1023,  582,
     475,  915,  704,  171,   61,  683,  275,  130,  634,  404,  850,  195,  684,  982,  964,   76,
     508,  401,  346,  274,  702,  670,  799,  540,  430,  895,  657,  924,  672,  303,  344,  169,
     853,  215,  440,   95,  396,  506,  990,  762,  734,  669,  668,  412,  583,  119,  578,  322,
      52,  332,  145,  267,  427,  237,  213,  538,  248,  259,  260,  806,   32,  192,  510,  220,
     438,  662,  968,  861,  431,  224,   37,  503,  869,  840,  247,  206,  489,  205,  870,  419,
     792,  958,  541,  890,   36,  873,  255,  894,  502,  559,   41,  621,  153,  693,  609,  172,
     754,   56,  899, 1007,  116,  839,  434,  212,  449,   99,  585,  887,  371,  823,  802,  373,
     173,  465,   45,  372,    9,  953,  889,  162,  943,  390,  256,  139,  299,  454,  196,  632,
     300,   79,  961,  696,  635,  420,  667,  793,  156,  329,  942,  330,   18,  334,  691,  752,
     292,   84,  714,  365,   43,  993,  470,  297,  181,  339,  125,  243,  882,  630,  596,  948,
     284,  234,  725,   63,  859,  103,  331,  203,  571,  649,  328,  602,  764,  446,  364,  188,
     282,  451,  647,  650,  207,  422, 1015,  200,  689,  148,  351,  717,  482,  663,  547,  531,
     287,  641,  473,   47,   83,   44,  816,  832,  265,  312,  238,  165,  462,  441,  295,  613,
     433,  423,  199,  520, 1008,  978,  319,  903,  721,  586,  298,  930,    7,  307,  860,  560,
     886,  837,  957,   91,  286,  484,  253,  182,  254,  518,  397,  687,  345,  620,  865,   88,
     381,  901,  868,  875,  749,  888,  642, 1010,  317,  168,  198,  278, 1009,  266,  826,  656,
     848,   26,  949,  965,  219,  268,  108,  846,  176,  798,  283,  368,  549,  532,  227,   60,
     783,  819,  296,  939,  562,  411,   20,   68,   92,  158,  490,  405,  738,  105,  310,  483,
};

void ki_xform_shuffle4_apply(uint8_t *restrict out,
         const uint8_t *restrict in,
         int n, int ch) {
    for (int pl = 0; pl < ch; pl++) {
        const uint8_t *s = in + (size_t)pl * (size_t)n;
        uint8_t *d = out + (size_t)pl * (size_t)n;
        for (int i = 0; i < n; i++)
            d[i] = s[ki_xform_shuffle4_map[i]];
    }
}

static const uint16_t ki_xform_shuffle5_map[1024] = {
     263,   10,   65,  404,  779,  410,  306,  802,  559,  750,  207,  283,  672,  803,  962,  880,
     624,  786,   52,  227,  467,  565,  718,  155,  234, 1021,   74,  268,  327,  470,  432,  670,
     105,  450,  397,  427,  857,   54,  477,  762,  616,  128,  247,   75,  935,  269,  657,  532,
     643,  539,  972,  581,  578,   39,  716,  440,  783,  412,  502,  405,  599,  621,  671,  332,
     116,  420,  140,  906,  658,   72,  901,  229,  751,  223,  770,  796,  248,  752,  226,  433,
     975,  875,  230,  586,  999,   88,  536,  101,  217,  881,  873,  614,  312,  168,  691,  717,
     290, 1023,   45,   90,   61,  131,  164,  685,  777, 1014,  674,  531, 1005,  568,  995,  620,
     597,  582,  521,  520,  328,  744,  981,  994,  355,    6,  574,  969,  939,   76,  753,  919,
     462,  840,   85,  276,  400,  362,  237,  703,  541,  611,  177,  275,  790,   41,  242,  154,
      84,  633,  379,  447,  215,  567,  411,  402,  943,  111,  998,  809,  852,  297,  775,  983,
     675,  253,  171,  993,  996,  819,  127,  225,  963,  918,  591,  337,  469,  441,  500,  609,
     465,  453,  828,  315,  421,    4,  513,  569,  988,  519,    7,  533,  813,  167,  712,  575,
     343,  403,  916,  222,  107,  375,  279,  855,  682,  804,  700,  320,  434,  288,  522,  422,
     550,  216,  399,  147, 1016,  692,   21,  952,   62,  409,  951,  518,  387,  244,  727,  774,
     680,  818,  349,  656, 1006,  950,  329,  776,  822,  141,  834,  211,  272,  452,  336,   78,
     555,  390,  517,  548,  832,  261,  827,  907,  655,  557,  947,  472,  267,  584,  903,    8,
     847,   38,  850,   44,  613,  649,  785,  119,   96,  206,  260,  445,   99,   19,  529,  156,
     228,  208,  345,  605,  723,  644,  254,   25,  720,  743,  330,  902,  408,  580,  351,   48,
     179,  576,  265,  593,  322,  917,   86,  210,  683,   31,  570,  547,  138,  967,  507,  106,
      89,  123,  622,  311,  510,  976,  795,  239,  914,  159,  955,  205,  853,  710,  118,  769,
      12,  102,   17,   83,  867,  836,  835,  166,  496,  354,  241,  233,  760,   30,  554,  188,
     741,  942,  368,  908,  553,  896,  848,  978,   56,  458,  739,  989,  890,  395,  479,  588,
      81,  334,   79,   98,   51,  909,  238,  457,  416,  807,  515,  240,  218,  728,  396,  583,
     246,  887,  612,  100,  103,   95,  810,  456,  535,  461,  202,   92,  715,  911,  438,  833,
     436,  884,  693,  540,  471,  870,  331,  793,  731,   69,  193,  197,  178,  361,  619,  883,
     651,  709, 1011,  192,  923,  957,   71,  108,  589,  340,  701,   59,  383,  114,  406,  764,
     219,  886,  571,  196,  871,  714,   15,  768,  407,  872,  791,  245,  236,  446,  122,  516,
     304,  259,   63,  606,  912,  772,  595,   43,  174,  882,  904,  991,  549,  788,  199,  377,
     232,  190,  627,  459,  798,  596,  264,  650,  634,  725,  130,  120,  868,  844,  134,  112,
     931,  281,  506,  668,  148,  564,  661,  925,  291,   77,  797,  357,  526,  926,  468,  191,
     335, 1000,  920,  484,  250,  630,  350,  161,   46,  990,  970,   68,  864,  449,  204, 1001,
     815,  294,  585, 1012,  143,  669,  859,  194,  538,  697,  284,  949,  831,  543,   34,  851,
     821,  924,  187,  153,  448,  299,  137,  695,  805,  125,  126,   50,  498,  414,  808,  690,
     341,   26,  781,   87,  273,  235,   70,  849,  435,  157,  338,  509,  489,  698,  607,  318,
    1004,  618,  877,  937,  310,  932,  755,  660,  898,  598,  487,  394,  493,  437, 1015,   14,
     195,  663,  637,  929,  258,    2,  590,  824,  782,  676,  475,   27,  734,  303,  344,  846,
     729,  307,  974,   16,  959,  736,  486,  504,  778,  292,  158,  274,  501,  602,  878, 1022,
     249,  746,  573,  545,  968,  992,  401,  139,   40,  745,  895,  454,  124,   24,  823,  146,
     293,  787,  314,  326,   20,  688,  257,  508,  940,  266,  841,  648,  162,  393,  388,  491,
     800,  830,  699,  442,  439,  384,  740,  636,  812,  960,  629,  579,  756,  885,  647,  677,
     726,  876,  623,  696,  732,  730,  654,  391,  936, 1003,  735,  478,  378,  551,  825,  673,
     429,  132,  360,  765,  722,  347,  891,  866,  879,   97,  941,  628,  563,  826,  771,  842,
     117,  419,  285,  845,  742,    0,  382,  843,  956,  604,  979,  626,  678,  997,  359, 1010,
     425,   36,  984,  198,  494,  706,  862,  113,  136,   55,  946,  231,  482,   42,  646,  789,
     863,  224,  900,  763,  639,  451,  986,  982,  200,   28,  277,  652,  566,  899,  309, 1020,
     681,  528,  761,  631,  175,  426,  686,  858,  708,  289,  488,  370,  961,  964, 1013,  145,
     530,  829,  373,  460,  704,  176,  358,  418,  353,  659,  754,  356,  365,  689,  662,  512,
     295,  474,  801,  687,  333,  894,  560,  485,   11,  594,  182,  665,  869,  300,  733,  954,
     271,  497,  287,  684,  921,  572,  372,  905,   35,  679,  601,  711,  364,   64,  948,  514,
     561,  527,  278,  503,   91,  184,  933,  511,  642,  183,  544,  386, 1002,  150,  913,  481,
     298,  280,  490,  325,  747,  854,  499,  811,  534,  173,  635,  464,  144,  301,  430,   60,
     856,  381,  444,  172,  109,  423,  910,  645,  973,  577,  749,  133,  212,  342,  505,  820,
      33,  653,   57,  945,  431,  666,  466,  985,  971,  608,  758,  243,   32,  415,  546,  615,
      82,   49,  737,  165,  495,  316,  705,  738,  152,  641,   80,  817,  181,  928,  324,  702,
     319,   94,  558,  719,  282,  874,  151,  371,  625,  525,  348,  213,   93,  443,   37,  302,
     376,  980,  799,  766,  603,  110, 1008, 1009,  256,  537,  930,  524,  352,  473,  839,  251,
     773,  121,  953,  638,  185,  934,   47,  463,    9, 1017,  592,  385,  374,  428,  806,  562,
      18,  413,  865,  366,  767,   23,  262,   58,  180,  721,  483,  927,  189,  323,  713,  492,
     313,  838,  363,  317,   67,  346,  724,  958,  610,    3,  600,  617,  837,  308,  339,   66,
     270,  149,  966,  424,  369,   22,  305,  392,  915,  201, 1019,  987,  209,  640,  694,  897,
     552,  203,  321,  296,  170,  893,  169,  220,  792,  214,    5,  892,    1,  135,  129,  455,
     977,  632,  142,   73,  816,  780,  163,  398,  784,  888,  938,  186,  286,  417,  221,  748,
      13, 1007,  587,  104,  556,  389,  252,  889,  480,  380,  115,  160,  922,   53,  664,  255,
    1018,  794,  476,  860,   29,  542,  944,  667,  757,  861,  965,  707,  814,  367,  759,  523,
};

void ki_xform_shuffle5_apply(uint8_t *restrict out,
         const uint8_t *restrict in,
         int n, int ch) {
    for (int pl = 0; pl < ch; pl++) {
        const uint8_t *s = in + (size_t)pl * (size_t)n;
        uint8_t *d = out + (size_t)pl * (size_t)n;
        for (int i = 0; i < n; i++)
            d[i] = s[ki_xform_shuffle5_map[i]];
    }
}

static const uint16_t ki_xform_shuffle6_map[1024] = {
     707,  215,  307,   43,  599,  412,  728,  232,  554, 1015,  757,  779,  733,  303,  185,  772,
     905,  826,  175,  968, 1003,  217,  992,  216,  106,  720,  930,   68,  494, 1019,  873,  645,
     229,  729,  863,  960,  520,  414,  422,  804,   86,  492,  328,  399,  211,  590,  657,  912,
     986,  762,  221,  315,  362,  899,  471,  208,  317,  689,  765,  797,  178, 1012,  820,  946,
     118,  309,  945,  610,  887,  997,   42,  424,   73,  837,   66,  709,  673,  346,  223,  160,
     535,  447,  522,  467,  394,  572,  507,  838,  533,  132,  242,  143,  174,  904,   24,  563,
     356,  470,  818,  708,  629,   19,  926,  517,  330,  843,  181,  442,  458,  159,  969,  612,
     336,   47,  910,   25,  665,  725,  355,  294,  651,   71,  597,  241, 1020,  817, 1008,   11,
     350, 1000,  949,  410,  742,  849,  334,  637,  769,  980,  605,  287,  883,  955,  575,  921,
     326,   59,   88,  384,  859,  103,  784,  569,  545,  236,  853,  890,  305,  195,  264,  213,
     425,  504,  524,   16,  361,  501,    3,  476,  756,  647,  366,  881,  225,  109,  323,  254,
      41,  209,  748,  387,  396,  120,  445,  187,  795,  981,  805,  907,  163,  105,   14, 1017,
     592,  227,  584,  463,   92,  999,  971,  353,  755,  222,  783,  668,   80,    8,  936,  798,
     354,  766,  562,  437,  329,  743,  676,  565,  827,  238,  252,  793,  719,  754, 1023,  630,
     413,  588, 1001,  691,  505,  732,  596,  210,  171,  876,  664,  291,   45,  265,   49,  693,
     157,   82,  464,  124,  146,  156,  121,  289,  768,  363,  401,  866,  642,  889,  928,  900,
      50,  428,  268,  466,  917,   35,  108,  940,  281,  658,  868,  614,  453,  967,   54,  404,
     461,  767,  359,  194,  263,   26,  953,  123,  290,  763,  406,  973,  144,  715,   39,  601,
     518,  468, 1021,  840,  835,  561,  523,  771,  983,  349,  273,  656,  342,  559,   57,  816,
     357,  542,  781,  831,  982,  682,   83,  212,  570,   48,  521,  218,  677,  184,  530,  266,
     892,  536,  681,  247,  758,  112,  639,  415,  937,  104,  316,  299,   27,  595,  846,  636,
     183,  374,  972,  477,  243,  113,  531,  877,   93,  472,  465,  295,  627,  759,  740,  304,
     528,  549,  405,  674, 1010,  943,  580,  870,   97,  544,  449,  998,    7,  546,  954,  895,
     920,  833,  285,  253,  269,    1,  761,  594,  127,  974,   23,  367,  598,  615,  502,  813,
     392,  150,  703,  277,  749,  169,  224,  855,  753,  101,  360,  204,  625,  706,   36,  568,
     606,  751,  347,  726,  803,  516,  526,  699,  778,  812,  737,  403,  207,  712,  240,  655,
     341,  860,  683,  688,   72,  191,  377,  659,  613,  923,  861,  513,  585,  911,  547,  438,
     162,  510,  322,  801,  782,  335,  579, 1018,  871,  744,  541,  134,  320,  170,   74,  950,
     679,  690,  100,  411,  233,  993,  314,  951,  493,  318,  234,  378,  245,  944,   64,  321,
     115,  618,  730,  125,  487,  724,   81,  906,  179,  110,  787,  880,  903,  426,  479,   44,
     894,  391,  581,  400,  800,  557,  258,  976,  832,  865,  823,  308,  301,   56,   21,   67,
     133,  721,  919,  408,  339,  332,  942,  444,  199,  302,  364,  430,  641,  789,  452,  373,
     808,  947,  151,  567,  421,  872,  952,  879,  962,  417,   84,  850,  312,    5,  670,  371,
     701,  723,  136,  811,  848,  427,  345,  620,  469,  785,  958,  220, 1014,  834,  485,  402,
     155,  916,  807,  525,  662,  111,  180,  760,  459,  786,  591,  306,  922,  259,  276,  491,
     727,  439,  914,  248,  799,  558,  666,  667,  135,  235, 1006,   89,  270,  474,    2, 1022,
      69,  984,   10,  607,  142,  352,   52,  446,  348,  845,   28,  231,  927,  977,  970,  704,
     979,  839,  338,  539,  604,   13,  284,  219,  705,  206,  776,  713,  261,  293,  435,  680,
      51,  182,  794,  957,  448,  172,  745,  587,  891,  867,   94,  300,  578,  623,  416,  230,
     555,   61,  700,  609,  376,  775,  553,  117,   60,  483,   40,  909,   62,  685,  393,  137,
     770,  168,  152,  888,  278,  381,  556,  239,  139,  750,  543,  809,    9,  734,  925,  390,
     333,  844,  395,  830,    4,  586,  574,  246,  324,  167,  429,  128,  577,  226,  918,  397,
     582,  519,  154,  288,  188,  153,  538,  190, 1011,  532,  358,   75,  131,   76,  654,  506,
     475,  138,  560,  409,  116,  692,  806,   79,  351,  443,  915,   53,  814,  774,  344,  177,
      33,  548,  244,  653,  619,  255,  862,   32,  380,  454,  441,  484,  280,  286,  282,  842,
    1002,   34,  368,  292,  372,  965,   65,  593,  488,  283,  902,  875,  450,  829,  882,  908,
     140,  652,   20,  710,  511,  841,  669, 1013,  646,  632,  164,   85,  631,  130,  564,  718,
      31,  644,   70,  741,  473,  489,  810,  963,  107,  731,  964,  684,  643,   38,  147,  325,
     621,  249,  640,  697,  148,  790,  791,  102,  237,  480,   77,  828,  929,  432,   15,  628,
     275,  158,  994,   63,  959,  509,  941,  379,  735,  996,  114,  583,  792,  985,   30,  186,
      58,  460,  638,    6,  573,  297,  161,  989,  856,  884,  661, 1009,  858,  878,  711,  385,
     634,  166,  675,  375,  331,  478,  649,  898,  508,  896,  738,  145,  503,  490,  672,  486,
     939,  961,  497,  626,  431,  420,  260,  869,  337,  857,  514,  262,  192,  956, 1004,  935,
     257,  126,  550, 1007,   18,  176,  874,  739,  499,  462,  272,  696,  885,  196,   87,  418,
     836,  722,  310,  482,  436,  529,   78,  966,  389,  851,  802,  457,  456,  686,  617,  933,
     825,  251,  451,   98,  407,  764,  608,  433,  189,   55, 1005,  695,  129,  296,  398,  540,
     386,  383,  313,  633,  635,  687,  571,  340,  663,   17,  173,  796,  193,   12,  534,  948,
     897,  886,  854,  214,  611,  141,  228,  495,  388,  200,  736,  319,  274,  987,  622,  119,
     203,  537,  694,  901,  512,   91,  122,  603,  365,  311,  440,  616,  250,  978,  648,   29,
     938,  515,  975,  913,  714,  660,  589,  527,  198,  931,  852,  498,  369,  934,  671,  847,
     205,  924,  602,   46,   99,  298,  815,  650,  717,  201,  773,  995,  991,  455,  256,  990,
     419,  819,  370,   95,  343,  434,  864,   90,  702,  624,  822, 1016,  271,  746,  824,  716,
     566,  576,  197,   96,  698,  552,  551,  932,  423,  893,  747,  202,  821,  500,  279,   22,
     788,  327,  382,  752,  777,  988,  481,  600,  678,  149,    0,   37,  267,  780,  496,  165,
};

void ki_xform_shuffle6_apply(uint8_t *restrict out,
         const uint8_t *restrict in,
         int n, int ch) {
    for (int pl = 0; pl < ch; pl++) {
        const uint8_t *s = in + (size_t)pl * (size_t)n;
        uint8_t *d = out + (size_t)pl * (size_t)n;
        for (int i = 0; i < n; i++)
            d[i] = s[ki_xform_shuffle6_map[i]];
    }
}

static const uint16_t ki_xform_shuffle7_map[1024] = {
     871,  473,  551,  933,   89,  218,  856, 1019,  751,  465,  855,  938,  739,   69,  394,  979,
     985,  445,  759,  664,  731,  202,  785,  163,   78,  705,  990,  995,  151,  793,  208,  832,
     391,  886,  607,  114,  978,  693,  143,  409,   24,  656,   37,  613,  845,  944,  881,  753,
     629,  131,  636,  417,  972,  160,  301,  341,  197,  771,  277,  964,  987,  716,  691,  652,
      98,  764,   10,  515,  704,  289,  447,  929,  297,  261,  402,   90,  617,  203,  779,  903,
     466,  868,  292,  273,  448,  809,  178,  179,  965,  812,   57,  482,   51,  116,  921,  955,
     366,  424,  755,  121,  550,  526,  205,  313,  167,  356,  188,  244,  239,  190,  916,  183,
     194,   67,  638,  446,  255,  433,  399,  211,  880,  501,  858,  390,  251,  369,  503, 1015,
      81,  585,  873,   44,  295,  497, 1018,  981,  181,  139,   85,  777,  702,  877,  241,  460,
     328,  844,   41,  383,  492,  749,  962,  618,  240,  956,  576,  423,  534,  436,  811,  762,
     983,  304,  359,   32,  621,  611,  312,  191,  164,  314,  950,  330,  303,  196,   20,   52,
     848,  141,  992,  936,  422,  773,  206,   48,  866,  852,  281,  828,  772,  552,  675,  371,
     538,   13,  781,  247,  833,  907,  193,  109,  353,  631,  679,  961,  525,   16,  894,  598,
     792,  189,    3,  471,  454,  222,  864,  214,  614,  327,  416,  266,  839,  215,   30,  957,
     960,  890,  803,  421,  396,   77,  669,  533,  989,  763,  386,  975,  287,  831,  559,  661,
     928,  377, 1011,  387,  850,  521,  227,  342,  290,  161,  843,  671,  309,  766,  382,  687,
     780,  878,  805,  260,  531,  891,  802,  230,  462,  293,  532,  558,  338,  308,  518,  475,
     823,  867,  511,  834,  836,  565,  778,  542,  488,  842,  984,  859,  248,   31,  609,    6,
     331,  719,  418,  169,   75,  298,  523, 1020,  761,  725,   39,  384,  252,  804,  768,  470,
     923,  885, 1007,  129,  242,   33,  665,  982,  274,  380,   35,  233,  398,  641,  250,  392,
     272,  420,  740,  700,  853,  516,  231,  963,  790,  939,  824,  797,  368,   15,  634,  999,
     349,  360,  509,  980,  587,  268,  849,  376, 1008,  765,  262,  486,  512,  329,  316,  145,
      76,  442,  991,  653,  743,  443,  912,  270,  648,  414,  869,  322,  630,  622,   11,  439,
     659,  300,  899,   73,  507,  767,   46,    2,  455,  127,  102,  335,  650,  655,  541,  720,
     412,  736,  788,  741,  876,  111,  567,   17,  667,  555,  280,  920,  830,  479,  695,  449,
     934,   80,  589,  235,  752,  887,  354,  578,  942,  620,  343,  502,  606,  681,  815,  946,
     935,  220,  673,  379,  320,   36,  581,  487,  745,  883,  582, 1003,  325,  818,  361,  821,
     101,  175,  117,  110,  814,  157, 1000,  701,  427,  279,  909,  336,  221,  332,  419,  730,
      54,  694,  612,  243,  263,   97,  282,  774,  737,  461,    7,  744,  819,    9,  677,  177,
     138,  966,  344,  256,  728,  148,  657,  388,  932,  156,  905, 1023,  784, 1001,   25,  640,
     747,   56,  510,  688,  872,  829,   22,  889,  318,  137,  468,  678,  796,  592,  173,    5,
     703,  882,  732,  135,   34,  113,  583,  311,   42,  898,  724,   45,  324,    8,  480,  275,
     910,  435,  870,  901,  847,   91,  334, 1010,  776,  415,  264,  432,  863,  893,  968,  914,
     278,  940,  689,  107,  234,  115,   65,  594,  302,  635,  339,  393,   18,  949,  637,  346,
     727,  566,   94,  820,  365,  200,  915,  347,  413,  817,  441,  754,  229,  794,  498,  806,
    1014,  224,  478,  787,  259,  146, 1013,  158,  769,  310,  919,  574,  438,  646,  924,  323,
     452,  807,  124,  426,  851,  458,  207,  908,  643,  535,  861,  602,  685,  546,  520,  601,
     605,  686,  922,  517,  791,  453,  958,  682,  575,  463, 1009,  100,   43,  283,  810,  941,
      58,  971,  108,  969,  568,  543,  721,  644,  799,  569,  123,  484,  144,  176,  153,  827,
     925,  187,  450,   19,  522,  536,  917,  951,  133,  647,  430,  529,  597,  469,  362,  888,
     660,  825,  557,  714,  333,  600, 1002,  513,  299,  217,  257,  672,  216,  726,  683,  142,
     539,  967,   14,   21,  134,  884,  561,  159,  902,  770,  610,  626,  826,  947,  604,  865,
     717,  130,  174,  162,  690,  854,  474,  405, 1006,  340,  959, 1005,  182,  489,  722,  397,
     122,  676,   86,  742,  710,  668,  862,    1,  822,  639,  494,  209,  345,  201,  481,  997,
     760,  225,  857,  373,  357,  457,  352,  619,  709,  198,  265,  483,  286,   28,   29,  735,
     504,  738,  204,  232,  800,  245,  199,  627,  734,  651,  337,  514,  554,  545,  993,  364,
     171,  375,  530,  267,  712,   93,  658,  305,  750,   27,  556,  706,  150,  943,  540,  954,
     874,  528,  165,  708,  490,  271,  758,  918,  789,  976,   87,  319,  495,  491,  477,  499,
     118,  125,  892,  783,  616,  875,  258,  801,  152,  385,  911,  212,   72,   26,  628,  372,
     103,  549,  906,  580,    0,  104,   53,  615,  904,  112,  166,  451,  213,   68,  195,  988,
     410,  649,  493,  106,  403,  408,  895,  401,  572,  952,  937,  897,   55,  757,  692,  670,
     632,  527,  707,  128,  326,  996,  624,  378,  547,  977,  149,    4,  288,  269,  186,  603,
     496,   12,  238,  674,  237, 1016,  180,   84, 1021,  236,  389,  699,  367,  425,  723,  285,
     563,  440,  838,  140,  284,  562,  411,  927,  170,   82,  900,  400,  407,  253,  756,  132,
     926,  786,  223,  998,  505,  119,  625,  172,  363,  472,   23,  355,  913,  395,  733,  291,
     456,  841,  697,  591,  662,  317,  718,  748,   62,   66,  680,  713,  485,  276,   95,  860,
      70,  467,  816,  593,  953,  608,  358,  711,  348,  948,  837,  930,  808,  586,  571,  782,
      79,  684,   40,  431,  500,  155,  350,  775,  168, 1004,  524,  974, 1017,  623,  294,  459,
     746,  351,  896,  506,  537,  307,  588,   83,  249,  798,  715,  184,  813,  254,  306,  370,
     464,  945,  986,  476,  321,  795,  437,  544,  696,  508,  210,  633,   61,  577,   64,  729,
     560,   99,  381,  192,  654,  973,  595,  105,  185,  698,  835,  573,  315,  584,  120,  553,
     147,  429,  296,  136,  879,  570,   47,  226,   50,  406,  599,  590,   63,  994, 1012,  642,
     645,  228, 1022,  126,  579,  846,   60,  434,  564,   92,  246,   71,  428,  444,   88,   38,
     219,  519,  931,   59,  596,  374,   96,  548,  840,   74,   49,  666,  404,  154,  970,  663,
};

void ki_xform_shuffle7_apply(uint8_t *restrict out,
         const uint8_t *restrict in,
         int n, int ch) {
    for (int pl = 0; pl < ch; pl++) {
        const uint8_t *s = in + (size_t)pl * (size_t)n;
        uint8_t *d = out + (size_t)pl * (size_t)n;
        for (int i = 0; i < n; i++)
            d[i] = s[ki_xform_shuffle7_map[i]];
    }
}

static const uint16_t ki_xform_shuffle8_map[1024] = {
     703,  562,  634,  131,  889,  422,  187,  767,  715,  507,  698,   15, 1016,  688,  801, 1010,
     330,  218,  317,   21,  936,  245,  769,  156,  999,  530,  505,  760,  828,  636,  776,  566,
     169,   43,  357,  438,  900,   77,  324,  641,  959,  100,  851,   19,  945,  862,  220,  929,
     545,   98,  964,  922,  604,  128,  609,  633,  118,  794,  948,  812,  992,  790,  231,  714,
     193,  534,  137,   53,  549,  559,  611,  188,  192,  199,  839,  981,   68,  431,  946,  478,
     493,  829,  314,  903,  543,  107,  840,  752,  571,  491,  581,  105,  758,   80,  420,  461,
     432,  201,  687,  792,    1,  262,  754,  705,  282,  716,  859,  443,  910,  913,  401,  310,
     606,  937,  504,   61,  798,  998,  503,  117,  622,  184,    9,  944,  329,  175,  950,  147,
     866,  912,  274,  643,  542,  252,  139,  527,  301,  826,  848,   56,  855,  645,  474,  668,
     770,  407,  975,  897,  257,  285,  425,  886,   54,  287,  837,  276,  243,   51,  583,  806,
     778,  268,  492,  846,  699,  235,  271,  895,  556,  353,  857,  648,  834,  931,  391,  799,
     498,  523,  644,  219,  973,  211,  521,  914,  208,  883,  858,  347,  332,  655,  901,  337,
      24,  149,  928,  954,  125,  132,  930,  719,   78,  701,  971,  323,  660,  575,  969,  759,
     617,  841,  343,  720,  888,  803,  183,  428,  221, 1007,  477, 1003, 1008,  791,  683, 1009,
     472,  525,  658,  166,   16,  278,  338,   48,  890,  520,  112,  891,  835,  318,  213,  480,
      63,  483,  781,  349,  155,  311,  502,  290,  713,   90,  996,  269,  591,  783,  669,  266,
     773,   93,  905,  453,  775,  150,  961,  893,  469,  448,  915,  242, 1022,  745,  458,  896,
     263,   29,  126,  681,  244,  286,  600,  686,  550,  796,  445,  479,  817,  171,   30,  411,
     563,  394,  756,  987,   41,  728,  800,   27,  613,  825,  642,  723,  247,  190,  630,  256,
     813,  625,  404,  786,  535,  804,  434, 1012,  724,  565, 1018,  371,   60,  382,  816,  204,
     408,   10,  554,  304,  121,   72,  941,  165,  653,   36,  136,  288,  962,  342,  623,  465,
     414,  344,  362,  284,  418,  312,  172,  265,  603,  423,  481,  601,  315,  702,  656,  186,
     313,  868,  729,  807,  281,  413,  561,  612,   42,  494,  122,  874,   79,  685,  335,  415,
      96, 1001,  358,  396,  737,  360,  106,  207,   11,  684,  141,  476,  847,  134,  805,  531,
     616,  553,  115,  861,  821,  161,  904,  741,  957,  631,  454,  339,  555,  209,  457,   25,
     942,  596,  894,  845,  361, 1004,  402, 1017,  475,  844,  471,  953,  460,  259,  217,   57,
     823,  449,  727,  246,  103,  628,  980,   49,  836,    0,  120,  960,   47,  869,  168,  852,
     924,  963,    5,  462,  757,  154,  574,  955,  489,    8,  677,    2,  551,  629,  934,  568,
     978,  486,  226,   18,    7,  970,  101,  409,  605,  654,  831,  325,   75,  774,  159,  984,
     340,  763,  694,  497,  838,  711,  258,   74,  567,  966,  607,  363,  267,  279,  853, 1000,
     293,  968,  557,  233,  341,  879,  367,  185,   55,  991,  940,   69,  680,  224,  597,  548,
     682,   45,  322,  336,  228,  254,  440,  882,  148,  949,  430,  232,  133,  873,  299,  588,
     451,  772,  578,  867,  308,  177,  564,  762,  495,  787,  238,  820,  590,  320,  144,  649,
     383,  439,  907,  354,  373,  908,  241,  818,  488,  104,  650,  675,  921,  377,   40,  764,
     381,  779,  435,  860,  589,  189,  819,   14,  146,   28, 1020,  355,  697,  573,  585,   35,
      32,  788,  695,  127,  584,  952,  947,  400,  974,  222,  751,  524,  546, 1011,  511,  691,
     130,  487,  272,  919,  191,  768,  646,  708,  789,  380,  447,  424,  294,  459,  822,  795,
     935,  173,  178,  988,  223,  608,  785,  135,  690,  620,  576,  113,  490,  512,  248,  733,
     637,  572,  671,  181,  569,   58,  881,  731,  203,  876,  180,  300,   84,  735,  547,  102,
      66,  814,  842,  398,  261,  160,  163,  444,  356,  664,  982,  755,  651,  334,  652,  983,
     227,  326,  765,   37,  236,   76,  375,  666,  811,   39,  327,   67,  366,  956,  797,   94,
     579,  289, 1006,  473,  884,   89,  270,  195,  624,   22, 1014,   46,  710,  580,  871,  771,
      86,  397,  780,  108,  508,  376,  709,  537,  610,  994,  255,  468,  693, 1013,  251,  712,
     250,  421,  744,  283,  704,  516,  696,  298,  280,  639,  878,  740,  277,  824,   70,  167,
     275,   83,  909,  916,  501,  730,  437,  689,  370,  726,  809,  692,  917,  967,  536,  700,
     392,  750,  406,  297,  210,  319,  599,  387,  739,  995,  753,  802,  667,  943,  926,  640,
     309,  528,  747,  378,  429,  560,  748,  749,  647,  808,  997,  328,   81,  303,  627,  296,
       6,  200,  662,  989,  393,  538,  331,  234,  513,  810,  870,  892,  514,  614,  766,   95,
     540,  442,  372,  670,   23,  626,  932,  843,    4,  793,  216,  972,  114,  864,   50,  920,
      52,  230,  466,  369,  403,  865,  482,  732,   92,  742,  526,  450,  736, 1019,  663,  463,
     738,  594,  111,  124,  390,  158,  484,  965,  229,  619,  205,   85,  832,  452,  441,  345,
     182,  176,  593,  993,   33,  632,   62,  215,  510,  661,    3,  374,  333,  359,  306,  761,
     123,  587,  674,  976,  348,  143,   17,  194,  977,  618,  152,  725, 1005,  517,  911,  157,
     368,  539,  533,  938,  109,  885, 1015,  898,   97,  734,  295,   71,  602,   82,  427,  416,
     456,  577,  419,  142,  237,  446,  582,  174,  856,  350,  389, 1021,  815,   13,  887,  307,
     164,  784,  206,   26,  872,  679,  170,  119,  405,  673,  179,  426,  722,  854,  305,  385,
     260,  433,   34,  990,  906,  351,  302,   12,  249,  352,  225,  558,  927,  519,  877,  958,
      73,  672,  522,  291,  541,  364,  153,  316,  202,  638,  515, 1002,  863,  659,  939,  951,
     321,  706,  544,  979,  615,  782,  162,  138,  436,  899,  875,  365,  570,  292,  455,  902,
     598,  746,  467,  592,  880,  151,  198,  718,  500,   38,  552,   65,  621,  145,  827,  595,
     212,  986,  529,  665,  925,  509,   88,  707,  240,  346,   59,  676,  110,  386,  849,  635,
     395,   64,  833,   99,  264,  678,  116,  743,  388,  933,  923,  485,  417,  532,  273,  717,
      20,  777,  239,  496,   91,  412,  850,  918,  196,  586,  506,  399, 1023,  499,  470,   31,
     657,  410,  214,  518,  830,  253,  140,   87,   44,  721,  197,  129,  384,  985,  379,  464,
};

void ki_xform_shuffle8_apply(uint8_t *restrict out,
         const uint8_t *restrict in,
         int n, int ch) {
    for (int pl = 0; pl < ch; pl++) {
        const uint8_t *s = in + (size_t)pl * (size_t)n;
        uint8_t *d = out + (size_t)pl * (size_t)n;
        for (int i = 0; i < n; i++)
            d[i] = s[ki_xform_shuffle8_map[i]];
    }
}

static const uint16_t ki_xform_shuffle9_map[1024] = {
     660,  986,  239,  352,  442,  402,  840,  801,  585,  312,  385,  794,  310,  798,  693,  713,
     860,  659,  710,  134,  535,  217,  890,  603,  697,  299,  875,  428,  992,  781,  233,  116,
     324,  824,  926,  653,  817,  445,  757,  441,  420,  333,  837,  901,  219,  184,  393,  319,
     109,  624,  117,  647,  739,  954,  353,  964,  418,  235,  911,  161,  551,  128,  198,  119,
     708,  417,  869,  350,  646,  780,  505,  855,  668,  306,  238,  368,  153,  833,  250,  409,
     769,  898,  456,  252,   47,  455,  527,   46,  856,  356,  247,  261,  245,  805,  403,  537,
     143,  731,  482,  422,   69,  759,  991,  960,  480,  200,  484,  170,  424,   58,  951,  371,
     848, 1002,  734,  311,  495,  591,  258,  565,  685,  290,  727,  966,  465,  590,  163,  730,
     230,  789,  448,  851,  979,   57,  423,  187,  983,  655,  154,  741,  542,  414,  282,  487,
     564,  500,   74,  497,  370,  292,   23,  376,  827,  447,  327,  902,  934,   10,  295,  724,
     609,   75,  358,  630,  391,   78,  846,  251,  100,  155,  439,  649,  775,  790,  878,  643,
     678,  201,  747,  705,  528,  795,   81,  228,  221,  882,  377,  317,  768,  711,  188,  549,
     648,    1,  688,  451,  689,  400,  955,  962,  425,  499,  375,  179,  870,  468,  334, 1006,
     613,  784,  302,  127,  137,  994,  318,  415,  288,  583,  496,  770,  511, 1003,  610,  635,
     691,  277,  814,  305,  618,  577,  976,  524,  629,  553,  650,  891,  162,  924,  359,  344,
     118,  220,  197,  698,  662,  596,  244,  164,  379,  192,  532,  121,  401,  544,   72,  886,
     156,  961,  963,  321,  463,  157,   37,  444,  574, 1005,  471,  642,  897,  405,    2,  289,
     810,    4,  149,  804,  633,  670,  638,   99,  996,  199,   40,  222,  360,  921,  748,  304,
     348,  584,   32,  547,  458,  208,   33,  640,  231,  320,  701,  211,  436,  335,  707,  651,
     476,  272,  903,  883,  906,  234,  661,  820,  504, 1014,  922,  330,  264,  563,  959,  543,
     842,  947,  832,  260,   79,  293,  889,  571,  847,  853,  973,  384,   38,  523,  751,  513,
     224,  656,  676,  612,  645,   45,  488,  750,  942,  595,  687,  286,  885,  178,  540,  189,
     139,  212,  386,  494,   91,  502,  115,  841,  158,  130,   63,  369,  774,  868,  628,  990,
     159,  714,  270,  995,  608,  746,  811,  893,  918,  525,  614,  129,  606,  266,  255,  636,
    1016,  315,  580,  340,   92,  510,  246,  597,  146,   71,  864,   16,  194,  566,  256, 1023,
     195,  821,  267,  671,  326,  479,  364,  294,  923,  825,  993,   21,  600,  460,  491,  125,
     399,  752,  615,  861,  634,  899,  169,  681,  362,  776,  771,  665,  545, 1004,  813,  452,
     470,   28,   14,  785,  854, 1007,  171,  533,  339, 1015,  271,  389,  498,  806,  240,  773,
     126,  133,  873,  675,  300,  744,   50,  473,   97,  372,  323,  223,  559,  453,  956,  733,
     760,  749,   31,  723,  829,  181,  366,  396,   25,  787,  433,  696,  483,  538,  351,  429,
     103,  263,  110,  967,  516,  515,  278,  706,  107,  581,  737,  562,  762,  838, 1008,  766,
     728,  275,  729,  296,  881,  807,  475,  815,  519,  673,  616,  702,  970,  347,  383,  150,
     940,  183,  237,  819,  831,  147,  888,  279,  909,  437,  490,  285,   20,  313,    9,  969,
     182,  186,  570,  328,  206,  850,  367,  365,  520,  138,  987,  830,  404,  413,  521,  822,
     167,  857,  664,   93,   43,  684,   11,  719,   73,  210,  546,  965,  936,  667,  862, 1013,
     526,  331,  232,  677,  998,   19,  672,  343,  554,  601,  863,  457,  345,   87,  718,  373,
     216,  686,   27,  556,  740,  174,   94,  849,  803,  879,  337, 1018,  394,  450,  466,  582,
     477,  472,  308,  761,   80,  797,  440,  408,  939,  576,  715,  669,  426,  253,  632,  325,
     680,   96,  309,    7, 1017,  361,  191,   39,  721,  988,  281,   60,  225, 1010,  322,  674,
     257,   59,  427,  531,  977,   22,  884,  259,  449,  874,  931,  530,  262,  953,  241,   83,
     816,  205,  877,  641,  478,  944,  106,  859,   84,   76,  122,   53,  989,  111,  226,  912,
     132,  777,  703,  978,  108, 1019,  489,  395,  895,  876,  459, 1012,  927,  329,  682,  341,
      77,  839,  486,  738,  560,  481,  700,  982,  809,   66,  694,  185,  858,   34,  193, 1009,
     914,  772,  416,  419,   48, 1022,  756,  767,  637,   36,  248,  507,  639, 1011,  380,  823,
     469,  249, 1021,  735,  617,  695,   54,  652,  501,   61,  464,  985,  102,  843,  786,  971,
      65,  307,  555,  587,  658,  779,  657,  949,  539,  536,  316,  557,  446,  599,  594,   29,
     666,   68,  548,  268,  607,  493,  561,  758,  925,  397,  558,  572,  605,  467,  485,  180,
     929,   13,  852,  166,  920,  845,  177,  131,  541,  283,  800,  101,  357,  506,  904,  937,
     280,  732,  679, 1000,  602,  980,  788,  999,  145,  867,  915,  835,  120,  712,  303,  569,
     765,  896,  140,  284,  575,  492,  144,  844,  363,   15,  626,  301,  398,  213,  461,   62,
      49,   12,  151,  621,  123,  168,  269,  905,   51,   30,  265,  709,  175,  297,  683,  945,
     378,  836,  381,  522,  410,  778,   98,  236,  663,  435,   67,  755,  579,  254,  796,  726,
     354,  509,  438,  826,  573,  799,  390,   26,  950,  802,  529,  105,  291,  644,  314,   90,
     834,  332,  952,  454,  227,  218,  938,  443,    0,  336,  622,  818, 1001,   18,  894,  588,
     203,  411,  215,  997,  880,  598,  112,  355,   88,  981,  793,  578,  984,  534,  972,  865,
     374,  975,  957,  517,  625,  165,  943,  412,  722,  935,   35,  589,  754,  274,  763,  152,
     930,  620,  699,  568,  276,   55,   42,  736,  623,  176,  148,  593,  717,   95,  783,  654,
     196,  407,   70,  430,  932,  550,  508,  586,   86,  512,   17,  552,  136,   85,  913,    8,
     338,  207,  872,  916,  202,  592,  782,  124,    3,  611,  114,  871,  928,  753,  434,  242,
      24,  142,  503,  866,  387,  917,   56,  421,  900,    5,  907,  725,  204,   44,  828,  349,
     287,  406,  214,  933, 1020,  791,   89,  910,  431,  812,  743,  229,  690,  209,  298,  941,
     104,  908,  764,  808,  392,  704,  792,   64,  604,  946,  892,  518,  135,  113,   52,  243,
     172,  160,  432,  742,  974,  462,  968,  720,  173,  388,  745,   41,  716,  631,  958,  567,
     342,   82,  619,  919,  474,  514,  346,    6,  692,  887,  190,  141,  273,  382,  627,  948,
};

void ki_xform_shuffle9_apply(uint8_t *restrict out,
         const uint8_t *restrict in,
         int n, int ch) {
    for (int pl = 0; pl < ch; pl++) {
        const uint8_t *s = in + (size_t)pl * (size_t)n;
        uint8_t *d = out + (size_t)pl * (size_t)n;
        for (int i = 0; i < n; i++)
            d[i] = s[ki_xform_shuffle9_map[i]];
    }
}

static const uint16_t ki_xform_shuffle10_map[1024] = {
     404,  386,  833,  442,  285,  536,  305,  134,  800,  185,  816,  293,  417,   86,  366,   55,
     722,  497,  716,  565,  804,  636,  295,  189,  705,  345,   70,  648,  547,  755,  331, 1018,
     379,  203,  903,  249,  377,   23,   83,  671,  257,  939,  726,  550,  317,  581,  787,  512,
     828,  719,  286,  594,  271,  993,  866,  279,  396,   49,  181,  370,  112,  158,  217,  651,
     220,  360,  143,  947,  460,  216,  527,  541,  521, 1019,  254,  812,  684,  675,  114,  910,
     587, 1012,   68,  207,  798, 1009,  809,  341,  974,  416,  420,  104,  261,   93,  772,  905,
     590,  892,  151,  981,  638,  354,  528,   60,  861,  558,  205,  262,  956,  463,  110,  176,
    1010,  858,  429,  447,   88,  960,  805,  333,  117,  944,  623,    0,  727,  544,  258,  720,
      20,  969,  566,  319,  183,  808,  163,  480,  282,  789,  105,  402,   58,  599,  904,  103,
     925,  165,  906,  226, 1017,   21,  831,  209,  372,  441,  221,  870,  476,  613,   63,  561,
     688,  100,  806,  184,  913,  915,  287,  737,  640,  470,  540,  935,  627,  929,  514,  148,
     559,  731,  811,  400,  352,  124,   92,  280,  630,  388,  887,  192,  242,  291,   84,  897,
     976,  141,  776,  797,  644,  423,  795,  579,  381,  131,   29,  510,  678,  538,  167,  556,
     704,  376,   39,  224,  770,  883,  426,  564,  238,  661,  407,  343,  483,  125,  421,  582,
     971,    9,   62,  492,   10,   80,  799,  398,  847,  608,  762,  822,  707,  694,  454,  937,
     529,    8,  298,  314,  871,  783,  138,  698,  821,  780,  493,  991,  672,  506,   57,  723,
     938,  836,  576,  188,  307,  277,  889,  154,  427,  239, 1004,  970,  116,  841,  230,  768,
     152,  299,  531,  389,  227,   11,  201,  982,  933,  304,  300,  215,  600,  872,  133,  911,
     403,    3,  380,  758,  712,  663,  932,  236,  732,  792,  475,  517,  773,  326,  489,  145,
     208,  928,  575,  256,  150,  900, 1014,  174,  560,  759,  882,  639,  387,  468,  730,  736,
     863,  634,  106,  323,  767,  437,   87,  419,  232,  717,  107,  973,  909,  202,  495,  747,
     275,  411,  965,  350,  635,  364,  687,  197,  884,  412,  306,  263,  835,  670,  471,  123,
     348,  771,  276,  869,  504,  322,  949,  715,  159,  219,  940,  756,   34,   52,  801,  338,
     693,  435,  337,  113,  222,  516,  526,  505,   65,   18,  603,  647,   41,  312,  626,   24,
     267,  270,  860,  344,   27,  957,  850,  898,  520,  378,  876,  328,  649,  619, 1013,  930,
     288,  699,  586,  266,  253,  453,  140,  359,  734,  820,  595,  302,  851,   22,  597,  604,
     438,  259,  961,  742,  530,  748,  778,  803,  231,  500,  914, 1008,  886,  954,  320,   89,
     631,  237,  926,  943,   59,  499,  342,  484,  657,   13,  855, 1005,  751,  515,  571,  567,
     584,  958,  952,  752,  478,  788,  810,  459,  311,  524,  324,   30,  498,   74,  764,  448,
     218,  264,  728,  325,  132,  555,    1,  710,  874,  413,  316,  361,  147,  641,  274,  612,
     392,   72,  813,  978,  465,   40, 1011,   78,    6,  502,   46,  646,  129,  135,  553,  844,
     340,  777,  365,  856,  539,  760,  509,  959,  745,  573,   95,  171,  233,  235,  570,  614,
     685,  173,  542,  681,  187,  444,   14,   69,  394,  127,  853,  741,   28,  126,  888,  128,
     650,  428,  962,  815,  896,  936,  574,  130,  210,   19,  656,  487,  659,  367,  666,  907,
     346,  440,   17,  941,  409,   76,   26,  543,  193,  744,  994,  294,  223,  408,  701,  485,
     522,  637,  234,  395,  655,   12,  297,  545,  250,  486,  765,  877,  990,  934,  924,  610,
     432,  109,  240,  121,  645,   96,  674, 1002,  834,  857,  606,  177,   38,  817,  479,  415,
     873,  979, 1015,  625,  652,   79,  708,  405,   43,  902,  989,  782,  662,  102,  895,  593,
      50,  228,  318,  849,   53,  899,  385,  629,  738, 1022,  679,  194,  191,   42,  156,  916,
      99,  774,  852,  819,  525,  848,  824,  823,  362,  179,  618,  724,  462,  695,  482,  455,
     149,   61,  740, 1020,  983,  660,   90,  721,  523,  964,   31,  901,  353,  692,  120,  161,
     175,  653,  927,  950,  696,  967,  578,  942,  839,  592,   48,  334,  713,  814,  918,  588,
     347,  206,  401,  985,  607,  842,  585,  683,  779,  992,  912,  583,  225,   54,  569,   97,
     769,  709,  246,  190,  620,  733,  382,  749,  406,   75,  329,  746,  891,  775,   16,  313,
     781,   67,  577,  845,  995,  172,  725,  375,  552,  706,   85,  894,  601,  784,  825,  867,
     283,  854,  349,  654,  921,  878,  269,  519,  865,  518,  301,  766,  477,  289,  802,  198,
     750,  980,  368,  260,  356,   56,  827, 1001,   71,  204,  667,  391,  418,  624,  434,  214,
       7,  754,  642,  118,  373,  893,  272,  213,  568,  680,  879, 1016,  739,  908,  711,  761,
     474,  315,  700,  922,  212,  436,  968,  605,  881,  186,  838,  196,  859,  609, 1000,  144,
     488,  686,  166,  292,  108,  785,  563,  611,   47,   91,  296,  735,  252,  890,  303,  119,
      25,  490,  643,  589,  602,  658,  200,  703,   33,  633,  330,  243,  757,  157,  535,  868,
     665,  336,  433,  829,  714,  807,  265,  743,  351,  457,  948,  793,  955,  469,  996,  496,
     818,  464,  622,  111,  155,   94,  358,   51,  953,  548,  449,    2,   36,  425,  101,  451,
     572,  169,  718,  946,  397,  551,  917,  951,  554,  998,  139,  410,  472,  690, 1007,  920,
      64,  511,  616, 1021,  532,  278,  537,  753,  247,  689,  507,  399,  162,  972,  414,  458,
     988,   81,  248,  327,  339,   44,  729,  557,    5,  945,  445, 1006,  355,  923,  491,  180,
     796,  986,  840,  664,  676,  168,  309,  466,  273,  122,  885,  182,  534,  632,  146,  281,
     697,  251,  456,  424,  443,  357,  195,  826,  170,  452,   98,  791,  794,  115,  966,  615,
     383,  580,  393,  153,   82,  875,  546,   73,  673,  621,  837,  308,  508,   32,   37,  244,
     422,  229,  862,  160,  513,  864,  332,  598,  668, 1023,  481,  446,  461,  562,  682,  321,
     977,  919,  790,  374,  549,  984,  931,  963,  199,  137,  241,    4,  596,  999,  384,  628,
     450,  245,  997,  786,  987,  136,  371,  677,  310,  702,  178,  467,  268,  691,  846,  290,
     431,  390,  363,  617,  142,  880,  430,   45,  369,  763, 1003,  975,  255,   77,  335,  501,
     533,   35,  164,  830,  669,  284,  843,  503,  832,  473,  211,   15,  591,  494,  439,   66,
};

void ki_xform_shuffle10_apply(uint8_t *restrict out,
         const uint8_t *restrict in,
         int n, int ch) {
    for (int pl = 0; pl < ch; pl++) {
        const uint8_t *s = in + (size_t)pl * (size_t)n;
        uint8_t *d = out + (size_t)pl * (size_t)n;
        for (int i = 0; i < n; i++)
            d[i] = s[ki_xform_shuffle10_map[i]];
    }
}


#endif /* KI_LOCAL_H */
