/*
 * lib/ki-load.h — Encoding-aware load_input (shared by Otto + experiments)
 * =======================================================================
 *
 * Loads raw pixels, applies encoding (exp8 etc.), packs into uint32 containers.
 *
 * Usage:
 *   1. Define ki_Args aa FIRST (before includes)
 *   2. #include "ki-load.h"   (includes ki-common.h, provides load_input)
 *
 * For CIFAR: 29 color channels + Sobel/LBP/DoG/Variance/Direction/Range
 * For MNIST: simple encoding via enc_lut_get
 */
#ifndef KI_LOAD_H
#define KI_LOAD_H

/* Suppress the default raw-packing load_input in ki-common.h */
#define KI_COMMON_LOAD_INPUT
#include "ki-common.h"
#undef KI_COMMON_LOAD_INPUT

/* ═══════════════════════════════════════════════════════════════════
 * ENCODING-AWARE load_input
 * ═══════════════════════════════════════════════════════════════════ */
static __attribute__((unused)) uint32_t *load_input(const uint8_t *X_raw,
                                                      int n_samples) {
    (void)n_samples;
#if KI_COLORS > 1
    /* ── CIFAR: via enc_array (color channels + Edge/LBP/DoG/...) ── */
    int n_enc = aa.enc_count;
    int enc_off[KI_ENC_MAX], enc_nc[KI_ENC_MAX];
    size_t stride = 0;
    for (int i = 0; i < n_enc && i < KI_ENC_MAX; i++) {
        int w = (int)aa.enc_array[i].width;
        if (w < 1) w = KI_ENC_WIDTH_DEFAULT;
        enc_nc[i] = KI_NC * w / KI_BIT_WIDTH;
        enc_off[i] = (int)stride;
        stride += (size_t)enc_nc[i];
    }
    size_t total_cont = (size_t)n_samples * stride;
    uint32_t *Xb = (uint32_t *)ki_xmalloc(total_cont * sizeof(uint32_t));
    for (int s = 0; s < n_samples; s++) {
        uint32_t *row = Xb + (size_t)s * stride;
        uint8_t px[COLOR_NB][1024];
        for (int px_i = 0; px_i < 1024; px_i++) {
            size_t base = (size_t)s * (size_t)KI_PX;
            int r_val = (int)X_raw[base + (size_t)px_i];
            int g_val = (int)X_raw[base + 1024 + (size_t)px_i];
            int b_val = (int)X_raw[base + 2048 + (size_t)px_i];
            uint8_t blk[COLOR_NB];
            ki_blocks_from_rgb(r_val, g_val, b_val, blk);
            for (int i = 0; i < COLOR_NB; i++) px[i][px_i] = blk[i];
        }
        ki_compute_edge(px, 32, 32);
        ki_compute_binary(px, 32, 32);
        ki_compute_lbp(px, 32, 32);
        ki_compute_dog(px, 32, 32);
        ki_compute_var(px, 32, 32);
        ki_compute_dir(px, 32, 32);
        ki_compute_range(px, 32, 32);
        ki_compute_lbp_rg(px, 32, 32);
        ki_compute_dist(px, 32, 32);
        for (int i = 0; i < n_enc; i++) {
            int col = (int)aa.enc_array[i].color;
            int typ = (int)aa.enc_array[i].type;
            int w   = (int)aa.enc_array[i].width;
            if (w < 1) w = KI_ENC_WIDTH_DEFAULT;
            if (typ < 0) typ = KI_ENC_LIN7;
            int pack = 32 / w, shift = w;
            int off = enc_off[i];
            for (int c = 0; c < enc_nc[i]; c++) {
                uint32_t val = 0;
                for (int k = 0; k < pack; k++) {
                    uint8_t pv = px[col][c * pack + k];
                    uint32_t ev = enc_lut_get(typ, w, pv);
                    val |= ev << (unsigned)(k * shift);
                }
                row[(size_t)off + (size_t)c] = val;
            }
        }
    }
    return Xb;
#else
    /* ── MNIST: via enc_array ── */
    int n_enc = aa.enc_count > 0 ? aa.enc_count : 1;
    int enc_w[KI_ENC_MAX], enc_pack[KI_ENC_MAX], enc_shift[KI_ENC_MAX];
    int enc_nc[KI_ENC_MAX], enc_type[KI_ENC_MAX];
    size_t stride = 0;
    int block_off[KI_ENC_MAX] = {0};
    for (int i = 0; i < n_enc && i < KI_ENC_MAX; i++) {
        int w  = aa.enc_array[i].width;
        int et = aa.enc_array[i].type;
        if (w < 1) w = KI_ENC_WIDTH_DEFAULT;
        if (et < 0) et = KI_ENC_LIN7;
        enc_w[i] = w; enc_pack[i] = 32 / w; enc_shift[i] = w;
        enc_nc[i] = KI_NC * w / KI_BIT_WIDTH; enc_type[i] = et;
        block_off[i] = (int)stride;
        stride += (size_t)enc_nc[i];
    }
    size_t total_cont = (size_t)n_samples * stride;
    uint32_t *Xb = (uint32_t *)ki_xmalloc(total_cont * sizeof(uint32_t));
    for (int s = 0; s < n_samples; s++) {
        uint32_t *row = Xb + (size_t)s * stride;
        for (int i = 0; i < n_enc; i++) {
            int off = block_off[i];
            for (int c = 0; c < enc_nc[i]; c++) {
                uint32_t val = 0;
                for (int k = 0; k < enc_pack[i]; k++) {
                    size_t p = (size_t)s * (size_t)KI_PX
                             + (size_t)c * (size_t)enc_pack[i] + (size_t)k;
                    uint8_t pv = X_raw[p];
                    uint32_t ev = enc_lut_get(enc_type[i], enc_w[i], pv);
                    val |= ev << (unsigned)(k * enc_shift[i]);
                }
                row[(size_t)off + (size_t)c] = val;
            }
        }
    }
    return Xb;
#endif
}

/* ═══════════════════════════════════════════════════════════════════════
 * INPUT CACHE — Precomputed uint32 containers saved to disk
 *
 * To avoid recomputing the encoding on every run, we cache the
 * load_input() output as a binary file:
 *   data/prepped/<hash>_<samples>x<stride>.pre
 *
 * Format: magic(4) ver(4) hash(4) samples(4) stride(4) data[...]
 * ═══════════════════════════════════════════════════════════════════════ */

/* Config hash for cache key (xform_id: 0=identity) */
static uint32_t input_cache_hash(int xform_id) {
    uint32_t h = 0;
    for (int i = 0; i < aa.enc_count; i++) {
        h = h * 31 + (uint32_t)(uint8_t)aa.enc_array[i].color;
        h = h * 31 + (uint32_t)(uint8_t)aa.enc_array[i].type;
        h = h * 31 + (uint32_t)(uint8_t)aa.enc_array[i].width;
    }
    h = h * 31 + (uint32_t)aa.packedB;
    h = h * 31 + (uint32_t)aa.channel;
    h = h * 31 + (uint32_t)KI_NC;
    h = h * 31 + (uint32_t)KI_PX;
    h = h * 31 + (uint32_t)KI_COLORS;
    h = h * 31 + (uint32_t)xform_id;        /* xform changes the pixel data cache */
#ifdef KI_DATASET_ID
    h = h * 31 + (uint32_t)KI_DATASET_ID;
#endif
    return h;
}

/* Build cache file path: data/prepped/<hash>_<N>x<S>.pre */
static void cache_path_build(char *buf, size_t bufsz, uint32_t hash, int n, size_t stride) {
    snprintf(buf, bufsz, "data/prepped/%08x_%dx%zu.pre", hash, n, stride);
}

/* Try to load cache, otherwise load_input() + save */
static uint32_t *load_input_cached(const uint8_t *X_raw, int n_samples,
                                    size_t stride) {
    uint32_t hash = input_cache_hash(0);  /* xform_id=0 = identity */
    char cache_dir[512], cache_path[1024];

    snprintf(cache_dir, sizeof(cache_dir), "data/prepped");
    snprintf(cache_path, sizeof(cache_path), "%s/%08x_%dx%zu.pre",
             cache_dir, hash, n_samples, stride);

    /* Try to load */
    FILE *cf = fopen(cache_path, "rb");
    if (cf) {
        uint32_t magic, ver, chk_hash, chk_samples, chk_stride32;
        if (fread(&magic, 4, 1, cf) == 1 && magic == 0x50524550 &&
            fread(&ver, 4, 1, cf) == 1 && ver == 1 &&
            fread(&chk_hash, 4, 1, cf) == 1 && chk_hash == hash &&
            fread(&chk_samples, 4, 1, cf) == 1 && (int)chk_samples == n_samples &&
            fread(&chk_stride32, 4, 1, cf) == 1 && (size_t)chk_stride32 == stride) {
            size_t total = (size_t)n_samples * stride;
            uint32_t *X = (uint32_t *)ki_xmalloc(total * sizeof(uint32_t));
            if (fread(X, sizeof(uint32_t), total, cf) == total) {
                fclose(cf);
                printf("  Input-cache: %s\n", cache_path);
                return X;
            }
            free(X);
        }
        fclose(cf);
        printf("  [CACHE] Invalid hash/config, recomputing\n");
    }

    /* Cache miss: load_input and save */
    uint32_t *X = load_input(X_raw, n_samples);

    /* Save cache */
    if (mkdir(cache_dir, 0755) == 0 || errno == EEXIST) {
        FILE *sf = fopen(cache_path, "wb");
        if (sf) {
            uint32_t magic = 0x50524550;
            uint32_t ver = 1;
            uint32_t samples = (uint32_t)n_samples;
            uint32_t stride32 = (uint32_t)stride;
            fwrite(&magic, 4, 1, sf);
            fwrite(&ver, 4, 1, sf);
            fwrite(&hash, 4, 1, sf);
            fwrite(&samples, 4, 1, sf);
            fwrite(&stride32, 4, 1, sf);
            size_t total = (size_t)n_samples * stride;
            fwrite(X, sizeof(uint32_t), total, sf);
            fclose(sf);
            printf("  Input-cache: %s  (saved)\n", cache_path);
        }
    }
    return X;
}

/* ── Xform-aware cache: xform_id ≠ 0 produces a different cache key ── */
static uint32_t *load_input_cached_xform(int xform_id,
                                          const uint8_t *X_raw,
                                          int n_samples, size_t stride) {
    if (xform_id == KI_XFORM_ID)
        return load_input_cached(X_raw, n_samples, stride);
    uint32_t hash = input_cache_hash(xform_id);
    char cpath[1024];
    cache_path_build(cpath, sizeof(cpath), hash, n_samples, stride);
    FILE *cf = fopen(cpath, "rb");
    if (cf) {
        uint32_t magic, ver, chk_hash, chk_samples, chk_stride32;
        if (fread(&magic, 4, 1, cf) == 1 && magic == 0x50524550 &&
            fread(&ver, 4, 1, cf) == 1 && ver == 1 &&
            fread(&chk_hash, 4, 1, cf) == 1 && chk_hash == hash &&
            fread(&chk_samples, 4, 1, cf) == 1 && (int)chk_samples == n_samples &&
            fread(&chk_stride32, 4, 1, cf) == 1 && (size_t)chk_stride32 == stride) {
            size_t total = (size_t)n_samples * stride;
            uint32_t *X = (uint32_t *)ki_xmalloc(total * sizeof(uint32_t));
            if (fread(X, sizeof(uint32_t), total, cf) == total) {
                fclose(cf);
                printf("  Input-cache: %s\n", cpath);
                return X;
            }
            free(X);
        }
        fclose(cf);
    }
    /* Cache miss: transform, load_input, save */
    uint32_t *X = load_input(X_raw, n_samples);
    mkdir("data/prepped", 0755);
    FILE *sf = fopen(cpath, "wb");
    if (sf) {
        uint32_t magic = 0x50524550, ver = 1, samples = (uint32_t)n_samples, stride32 = (uint32_t)stride;
        fwrite(&magic, 4, 1, sf);
        fwrite(&ver, 4, 1, sf);
        fwrite(&hash, 4, 1, sf);
        fwrite(&samples, 4, 1, sf);
        fwrite(&stride32, 4, 1, sf);
        size_t total = (size_t)n_samples * stride;
        fwrite(X, sizeof(uint32_t), total, sf);
        fclose(sf);
        printf("  Input-cache: %s  (saved)\n", cpath);
    }
    return X;
}

#endif /* KI_LOAD_H */