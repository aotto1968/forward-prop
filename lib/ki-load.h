/*
 * lib/ki-load.h — Encoding-aware load_input (shared by Otto + experiments)
 * =======================================================================
 *
 * Loads raw pixels, applies encoding (exp8 etc.), packs into uint32 containers.
 *
 * Usage:
 *   1. Define ki_Args aa FIRST (before includes)
 *   2. #include "ki-common.h" with KI_COMMON_LOAD_INPUT defined (suppresses the
 *      default raw-packing load_input) — the caller includes ki-common.h and
 *      ki-load.h EXPLICITLY (no hidden includes, see the .c files).
 *
 * For CIFAR: 29 color channels + Sobel/LBP/DoG/Variance/Direction/Range
 * For MNIST: simple encoding via enc_lut_get
 */
#ifndef KI_LOAD_H
#define KI_LOAD_H

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
        ki_compute_lbp_rb(px, 32, 32);
        ki_compute_lbp_gb(px, 32, 32);
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
 *
 * NOTE: Buffers larger than KI_PREPPED_MAX_BYTES are NEVER written to disk.
 * With --channel sweep one buffer is ~23 GB (60000 × 96512 × 4B) and 17
 * xforms would fill ~394 GB of disk for data that is loaded once and freed
 * again (sweep RAM control). Reading existing caches is unaffected.
 * ═══════════════════════════════════════════════════════════════════════ */

/* ═══════════════════════════════════════════════════════════════════════
 * CEX CACHE — one narrow buffer per (CHAN:ENC:XFORM)
 * ═══════════════════════════════════════════════════════════════════════
 * Replaces the broad per-xform cache: instead of one 4.4 GB buffer holding
 * ALL 71 CHAN:ENC slices, each member caches ONLY its own slice
 * (256 containers = 62 MB). File name carries the full member identity:
 *   data/prepped/cex_<chan>_<enc><w>_<xform>_v<implver>_<N>x<slice>.pre
 * → any member change OR xform implementation change produces a new file
 * name → stale caches are never served (bug 2026-08-04 colswap cache).
 * --sweep never writes (each member is trained once and never re-read).
 * X_raw here is the ALREADY xform-transformed raw buffer (the caller applies
 * ki_xform_apply_buf first, as the trainer does today).
 */
/* ═══════════════════════════════════════════════════════════════════════
 * IN-MEMORY XFORM CACHE — single slot (level 1 of the CEX pipeline)
 * ═══════════════════════════════════════════════════════════════════════
 * The CEX pipeline has 3 stages (see plans/plan-2026-08-05-xform-cache.md):
 *   1. xform transform  (raw → transformed raw, n_samples × KI_PX bytes)
 *   2. block computation (RGB → px[COLOR_NB][1024], expensive filters)
 *   3. channel encoding  (px[color] → member containers)
 * Stages 1+2 depend ONLY on the xform → identical for every member of one
 * xform group in the sweep. Members are xform-major sorted
 * (ki_member_spec_sort_xform) and the member loop is single-threaded (one
 * member computed at a time), so ONE slot suffices: a new xform_id simply
 * overwrites the previous buffer (auto-invalidation — no per-xform bookkeeping).
 *
 * Scope: this cache only engages on DISK-cache MISS inside
 * load_input_cex_cached() — if a cex_*.pre file exists, the buffer is loaded
 * and no transformation happens at all (nothing to cache). In --sweep every
 * member is a miss (sweep never writes .pre), so the slot hits for all
 * members of one xform group and is overwritten at the group switch.
 *
 * Ownership stays in the cache: the caller must NOT free the returned
 * pointer. ki_clear_cache() frees the slot explicitly at xform-group
 * boundaries (memory control, e.g. mini-devices < 8 GB). */
static uint8_t *_xf_cache_buf = NULL;   /* owned by the cache */
static int      _xf_cache_id  = -1;     /* xform_id of the cached buffer */
static int      _xf_cache_n   = 0;      /* n_samples the buffer was built for */

static const uint8_t *ki_xform_get(const uint8_t *X_raw, int n_samples,
                                   int xform_id) {
    if (xform_id == KI_XFORM_ID) return X_raw;  /* identity = raw, no slot */
    if (_xf_cache_id == xform_id && _xf_cache_n == n_samples)
        return _xf_cache_buf;                   /* slot hit (same xform group) */
    /* New xform (or different sample count): overwrite the previous slot.
     * Sorted members make this a group switch, so the old buffer is dead. */
    free(_xf_cache_buf);
    _xf_cache_buf = (uint8_t *)ki_xmalloc((size_t)n_samples * (size_t)KI_PX);
    ki_xform_apply_buf(_xf_cache_buf, X_raw, KI_COLS, KI_COLS, KI_COLORS,
                       n_samples, xform_id, KI_PX);
    _xf_cache_id = xform_id;
    _xf_cache_n  = n_samples;
    if (ki_debug_cache()) printf("  [XFORM-cache] computed %s (%dx%d)\n",
                            ki_xform_str(xform_id), n_samples, KI_PX);
    return _xf_cache_buf;
}

/* ═══════════════════════════════════════════════════════════════════════
 * CHANNEL CACHE — single slot (level 2 of the CEX pipeline, CIFAR only)
 * ═══════════════════════════════════════════════════════════════════════
 * Caches stage 2 PER (xform, channel): px[color][1024] per sample
 * (layout [n_samples][1024] = 61 MB at 60000 samples). Members are sorted
 * xform → channel → encoding, so one slot suffices: the R group computes +
 * caches R once, G overwrites it, etc. — one [CHANNEL-cache] computed per
 * channel group.
 *
 * NEED-ONLY (2026-08-05): stage 2 runs only the filters whose target block
 * the channel needs. Verified write-map: every filter reads ONLY basic
 * blocks (Y/AL/RG/RB/GB from ki_blocks_from_rgb) and writes exactly its own
 * block — except ki_compute_edge, which writes COLOR_EDGE AND COLOR_C.
 * ⇒ basic channels (R,G,B,Y,...) run NO filter at all; results are
 * byte-identical to the old full-block computation.
 *
 * Enabled only when aa.xform_cache_level >= 2 (default; --xform-cache-level
 * 1 uses the fused path without this buffer — mini-devices < 8 GB).
 * Same miss-only scope as ki_xform_get; freed by ki_clear_cache(). */
static uint8_t *_blk_cache_buf = NULL;   /* [n_samples][1024] = px[color], owned */
static int      _blk_cache_xf  = -1;     /* xform_id of the cached channel */
static int      _blk_cache_col = -1;     /* channel of the cached blocks */
static int      _blk_cache_n   = 0;      /* n_samples the blocks were built for */

/* Need-only stage 2 for ONE sample: fill dst[1024] with px[color].
 * src = the (transformed) raw buffer; identity src = X_raw. */
static __attribute__((unused)) void ki_sample_block_row(uint8_t *dst,
                                                        const uint8_t *src,
                                                        int s, int color) {
    uint8_t px[COLOR_NB][1024];
    for (int px_i = 0; px_i < 1024; px_i++) {
        size_t base = (size_t)s * (size_t)KI_PX;
        int r_val = (int)src[base + (size_t)px_i];
        int g_val = (int)src[base + 1024 + (size_t)px_i];
        int b_val = (int)src[base + 2048 + (size_t)px_i];
        uint8_t blk[COLOR_NB];
        ki_blocks_from_rgb(r_val, g_val, b_val, blk);
        for (int i = 0; i < COLOR_NB; i++) px[i][px_i] = blk[i];
    }
    /* Need-only filters — see write-map comment above */
    switch (color) {
    case COLOR_EDGE: case COLOR_C: ki_compute_edge(px, 32, 32); break;
    case COLOR_BIN:   ki_compute_binary(px, 32, 32); break;
    case COLOR_LBP:   ki_compute_lbp(px, 32, 32); break;
    case COLOR_DOG:   ki_compute_dog(px, 32, 32); break;
    case COLOR_VAR:   ki_compute_var(px, 32, 32); break;
    case COLOR_DIR:   ki_compute_dir(px, 32, 32); break;
    case COLOR_RANGE: ki_compute_range(px, 32, 32); break;
    case COLOR_LBP_RG: ki_compute_lbp_rg(px, 32, 32); break;
    case COLOR_LBP_RB: ki_compute_lbp_rb(px, 32, 32); break;
    case COLOR_LBP_GB: ki_compute_lbp_gb(px, 32, 32); break;
    default: break;   /* basic channel: blocks_from_rgb is sufficient */
    }
    memcpy(dst, px[color], 1024);
}

static __attribute__((unused)) const uint8_t *ki_channel_get(
        const uint8_t *src, int n_samples, int color, int xform_id) {
    if (aa.xform_cache_level < 2) return NULL;   /* level 1 only */
    if (_blk_cache_xf == xform_id && _blk_cache_col == color &&
        _blk_cache_n == n_samples)
        return _blk_cache_buf;                   /* slot hit (same channel group) */
    /* Miss: compute the channel's blocks for ALL samples into the slot.
     * Parallel over samples — each sample writes its own disjoint 1024-byte
     * row, px[] is thread-local → no race. */
    free(_blk_cache_buf);
    _blk_cache_buf = (uint8_t *)ki_xmalloc((size_t)n_samples * 1024);
    #pragma omp parallel for schedule(static) firstprivate(color)
    for (int s = 0; s < n_samples; s++)
        ki_sample_block_row(_blk_cache_buf + (size_t)s * 1024, src, s, color);
    _blk_cache_xf = xform_id;
    _blk_cache_col = color;
    _blk_cache_n  = n_samples;
    if (ki_debug_cache()) printf("  [CHANNEL-cache] computed %s:%s (%dx1024)\n",
                                 ki_xform_str(xform_id), ki_color_name(color),
                                 n_samples);
    return _blk_cache_buf;
}

/* Free both single-slot caches (level 1 transform + level 2 channel).
 * xform_id is accepted for API symmetry/debug output only — each slot holds
 * at most ONE buffer, so both are freed regardless. */
static void ki_clear_cache(int xform_id) {
    (void)xform_id;
    /* Channel cache first (level 2 is the dependent/last artifact built on
     * top of the transform) — release order matches the compute order. */
    if (_blk_cache_buf) {
        if (ki_debug_cache()) printf("  [CHANNEL-cache] free %s:%s\n",
                                ki_xform_str(_blk_cache_xf),
                                ki_color_name(_blk_cache_col));
        free(_blk_cache_buf);
        _blk_cache_buf = NULL;
    }
    _blk_cache_xf = -1;
    _blk_cache_col = -1;
    _blk_cache_n  = 0;
    if (_xf_cache_buf) {
        if (ki_debug_cache()) printf("  [XFORM-cache] free %s\n",
                                ki_xform_str(_xf_cache_id));
        free(_xf_cache_buf);
        _xf_cache_buf = NULL;
    }
    _xf_cache_id = -1;
    _xf_cache_n  = 0;
}
#if KI_COLORS > 1
static __attribute__((unused)) uint32_t *load_input_cex_cached(
        const uint8_t *X_raw, int n_samples,
        int color, int enc_type, int enc_width, int xform_id) {
    int w = enc_width > 0 ? enc_width : KI_ENC_WIDTH_DEFAULT;
    int typ = enc_type >= 0 ? enc_type : KI_ENC_LIN7;
    int n_cont = KI_NC * w / KI_BIT_WIDTH;
    size_t total = (size_t)n_samples * (size_t)n_cont;

    /* Cache key: channel/encoding/xform NAMES + impl version (stable) */
    uint32_t h = 0;
    {   const char *cn = ki_color_name(color);
        const char *en = ki_enc_name_short(typ);
        const char *xn = ki_xform_str(xform_id);
        while (*cn) h = h * 31 + (uint8_t)*cn++;
        while (*en) h = h * 31 + (uint8_t)*en++;
        while (*xn) h = h * 31 + (uint8_t)*xn++;
        h = h * 31 + (uint32_t)(uint8_t)w;
        h = h * 31 + (uint32_t)KI_XFORM_IMPL_VERSION;
        h = h * 31 + (uint32_t)KI_NC;
        h = h * 31 + (uint32_t)KI_PX;
        h = h * 31 + (uint32_t)KI_COLORS;
#ifdef KI_DATASET_ID
        h = h * 31 + (uint32_t)KI_DATASET_ID;
#endif
    }

    /* Human-readable path (names, not just hash) — easy to spot/delete */
    char path[1024];
    snprintf(path, sizeof(path), "data/prepped/cex_%s_%s%d_%s_v%d_%dx%d.pre",
              ki_color_name(color), ki_enc_name_short(typ), w,
              ki_xform_str(xform_id), KI_XFORM_IMPL_VERSION, n_samples, n_cont);

    /* Cache hit — skipped entirely with --no-ens-cache (real-live: compute
     * the inputs from the raw data every run, like the chip on new data;
     * the in-memory xform cache below still applies). */
    FILE *cf = NULL;
    if (!aa.no_ens_cache) cf = fopen(path, "rb");
    if (cf) {
        uint32_t magic, ver, chk_h, chk_n, chk_s;
        if (fread(&magic, 4, 1, cf) == 1 && magic == 0x43455850 &&  /* 'P XEC' */
            fread(&ver, 4, 1, cf) == 1 && ver == 1 &&
            fread(&chk_h, 4, 1, cf) == 1 && chk_h == h &&
            fread(&chk_n, 4, 1, cf) == 1 && (int)chk_n == n_samples &&
            fread(&chk_s, 4, 1, cf) == 1 && (int)chk_s == n_cont) {
            uint32_t *X = (uint32_t *)ki_xmalloc(total * sizeof(uint32_t));
            if (fread(X, sizeof(uint32_t), total, cf) == total) {
                fclose(cf);
                if (ki_debug_cache()) printf("  CEX-cache: %s  (hit)\n", path);
                return X;
            }
            free(X);
        }
        fclose(cf);
        if (ki_debug_cache()) printf("  CEX-cache: %s  (MISS, recomputed)\n", path);
    }

    /* Cache miss: compute ONLY this CHAN:ENC slice.
     * Xform transform happens HERE (only on miss) — the cache-hit path above
     * needs only the identity, never the transformed pixels. Without this
     * the trainer transformed raw pixels for EVERY member even on cache hit
     * (~20% slower with warm cache, 2026-08-04).
     * Parallel over samples — each sample writes its own disjoint row
     * X[s*n_cont..(s+1)*n_cont], px[] is thread-local, enc_lut_get is a
     * read-only LUT → no race. 16 threads ≈ 16× faster first-run/cache-miss. */
    int pack = 32 / w, shift = w;
    uint32_t *X = (uint32_t *)ki_xmalloc(total * sizeof(uint32_t));
    /* Single-slot in-memory xform cache: the transformed raw buffer is
     * computed ONCE per xform group (sorted members) and reused by all
     * members of that group. The cache owns the buffer — no free here.
     * Only reached on disk-miss, so a cex_*.pre hit never transforms. */
    const uint8_t *src = ki_xform_get(X_raw, n_samples, xform_id);
    if (aa.xform_cache_level >= 2) {
        /* Level 2: stage 2 is cached PER (xform, channel) via ki_channel_get
         * — one compute per channel group (R, G, B), on a HIT only the cheap
         * stage-3 encode runs. Layout [s][1024] = px[color] rows. */
        const uint8_t *blk = ki_channel_get(src, n_samples, color, xform_id);
        #pragma omp parallel for schedule(static) firstprivate(pack, shift, color, typ, w, n_cont)
        for (int s = 0; s < n_samples; s++) {
            const uint8_t *pb = blk + (size_t)s * 1024;
            uint32_t *row = X + (size_t)s * (size_t)n_cont;
            for (int c = 0; c < n_cont; c++) {
                uint32_t val = 0;
                for (int k = 0; k < pack; k++) {
                    uint32_t ev = enc_lut_get(typ, w, pb[c * pack + k]);
                    val |= ev << (unsigned)(k * shift);
                }
                row[c] = val;
            }
        }
    } else {
        /* Level 1 only (--xform-cache-level 1): fused path — stage 2 runs
         * per member with a thread-local px[color] (NO big buffer →
         * mini-devices < 8 GB). Need-only: basic channels run no filters. */
        #pragma omp parallel for schedule(static) firstprivate(pack, shift, color, typ, w, n_cont)
        for (int s = 0; s < n_samples; s++) {
            uint8_t pb[1024];
            ki_sample_block_row(pb, src, s, color);
            uint32_t *row = X + (size_t)s * (size_t)n_cont;
            for (int c = 0; c < n_cont; c++) {
                uint32_t val = 0;
                for (int k = 0; k < pack; k++) {
                    uint32_t ev = enc_lut_get(typ, w, pb[c * pack + k]);
                    val |= ev << (unsigned)(k * shift);
                }
                row[c] = val;
            }
        }
    }

    /* Save (never in --sweep: each member trained once, never re-read) */
    if (!aa.no_ens_cache && !aa.sweep && total * sizeof(uint32_t) <= KI_PREPPED_MAX_BYTES) {
        if (mkdir("data/prepped", 0755) == 0 || errno == EEXIST) {
            FILE *sf = fopen(path, "wb");
            if (sf) {
                uint32_t magic = 0x43455850, ver = 1, samples = (uint32_t)n_samples, stride = (uint32_t)n_cont;
                fwrite(&magic, 4, 1, sf);
                fwrite(&ver, 4, 1, sf);
                fwrite(&h, 4, 1, sf);
                fwrite(&samples, 4, 1, sf);
                fwrite(&stride, 4, 1, sf);
                fwrite(X, sizeof(uint32_t), total, sf);
                fclose(sf);
                if (ki_debug_cache()) printf("  CEX-cache: %s  (saved)\n", path);
            }
        }
    }
    return X;
}
#else
/* MNIST (KI_COLORS <= 1): single channel — the CEX slice is the whole buffer */
static __attribute__((unused)) uint32_t *load_input_cex_cached(
        const uint8_t *X_raw, int n_samples,
        int color, int enc_type, int enc_width, int xform_id) {
    (void)color;
    int w = enc_width > 0 ? enc_width : KI_ENC_WIDTH_DEFAULT;
    int typ = enc_type >= 0 ? enc_type : KI_ENC_LIN7;
    int n_cont = NC * w / KI_BIT_WIDTH;
    size_t total = (size_t)n_samples * (size_t)n_cont;

    /* Cache key: channel/encoding/xform NAMES + impl version (stable) */
    uint32_t h = 0;
    {   const char *en = ki_enc_name_short(typ);
        const char *xn = ki_xform_str(xform_id);
        while (*en) h = h * 31 + (uint8_t)*en++;
        while (*xn) h = h * 31 + (uint8_t)*xn++;
        h = h * 31 + (uint32_t)(uint8_t)w;
        h = h * 31 + (uint32_t)KI_XFORM_IMPL_VERSION;
        h = h * 31 + (uint32_t)NC;
        h = h * 31 + (uint32_t)KI_PX;
        h = h * 31 + (uint32_t)KI_COLORS;
#ifdef KI_DATASET_ID
        h = h * 31 + (uint32_t)KI_DATASET_ID;
#endif
    }

    /* Human-readable path (names, not just hash) — easy to spot/delete */
    char path[1024];
    snprintf(path, sizeof(path), "data/prepped/cex_%s_%s%d_%s_v%d_%dx%d.pre",
             ki_color_name(COLOR_MNIST), ki_enc_name_short(typ), w,
              ki_xform_str(xform_id), KI_XFORM_IMPL_VERSION, n_samples, n_cont);

    /* Cache hit — skipped with --no-ens-cache (real-live compute, see above) */
    FILE *cf = NULL;
    if (!aa.no_ens_cache) cf = fopen(path, "rb");
    if (cf) {
        uint32_t magic, ver, chk_h, chk_n, chk_s;
        if (fread(&magic, 4, 1, cf) == 1 && magic == 0x43455850 &&
            fread(&ver, 4, 1, cf) == 1 && ver == 1 &&
            fread(&chk_h, 4, 1, cf) == 1 && chk_h == h &&
            fread(&chk_n, 4, 1, cf) == 1 && (int)chk_n == n_samples &&
            fread(&chk_s, 4, 1, cf) == 1 && (int)chk_s == n_cont) {
            uint32_t *X = (uint32_t *)ki_xmalloc(total * sizeof(uint32_t));
            if (fread(X, sizeof(uint32_t), total, cf) == total) {
                fclose(cf);
                if (ki_debug_cache()) printf("  CEX-cache: %s  (hit)\n", path);
                return X;
            }
            free(X);
        }
        fclose(cf);
        if (ki_debug_cache()) printf("  CEX-cache: %s  (MISS, recomputed)\n", path);
    }

    /* Cache miss: transform only here (on miss), then encode the slice.
     * Single-slot in-memory xform cache — see CIFAR variant above. */
    uint32_t *X = (uint32_t *)ki_xmalloc(total * sizeof(uint32_t));
    int pack = 32 / w, shift = w;
    const uint8_t *src = ki_xform_get(X_raw, n_samples, xform_id);
    #pragma omp parallel for schedule(static) firstprivate(pack, shift, typ, w, n_cont)
    for (int s = 0; s < n_samples; s++) {
        uint32_t *row = X + (size_t)s * (size_t)n_cont;
        for (int c = 0; c < n_cont; c++) {
            uint32_t val = 0;
            for (int k = 0; k < pack; k++) {
                size_t p = (size_t)s * (size_t)KI_PX + (size_t)c * (size_t)pack + (size_t)k;
                uint8_t pv = src[p];
                uint32_t ev = enc_lut_get(typ, w, pv);
                val |= ev << (unsigned)(k * shift);
            }
            row[c] = val;
        }
    }

    /* Save (never in --sweep: each member trained once, never re-read) */
    if (!aa.no_ens_cache && !aa.sweep && total * sizeof(uint32_t) <= KI_PREPPED_MAX_BYTES) {
        if (mkdir("data/prepped", 0755) == 0 || errno == EEXIST) {
            FILE *sf = fopen(path, "wb");
            if (sf) {
                uint32_t magic = 0x43455850, ver = 1, samples = (uint32_t)n_samples, stride = (uint32_t)n_cont;
                fwrite(&magic, 4, 1, sf);
                fwrite(&ver, 4, 1, sf);
                fwrite(&h, 4, 1, sf);
                fwrite(&samples, 4, 1, sf);
                fwrite(&stride, 4, 1, sf);
                fwrite(X, sizeof(uint32_t), total, sf);
                fclose(sf);
                if (ki_debug_cache()) printf("  CEX-cache: %s  (saved)\n", path);
            }
        }
    }
    return X;
}
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * Prepped-cache write threshold (CEX format only)
 * ═══════════════════════════════════════════════════════════════════════
 * Buffers above KI_PREPPED_MAX_BYTES are never saved to data/prepped/.
 * Huge buffers (e.g. --channel sweep: 60000 × 96512 × 4B = 23 GB per
 * xform) are loaded once, used, and freed again — persisting them would
 * fill the disk (17 xforms ≈ 394 GB) without any benefit. Cache reads
 * are unaffected. See: plans/plan-2026-07-31-sweep-xform-memory.md
 * The threshold is defined PER DATASET in ki-local.h (included via
 * ki-common.h before this file's code is read); this is the fallback.
 *
 * INTENTIONAL (2026-08-05): the OLD hash-based broad cache
 * (input_cache_hash / input_cache_write_allowed / load_input_cached,
 * magic 0x50524550, files <hash>_<N>x<stride>.pre) was REMOVED — it had the
 * stale-cache bug (bug 2026-08-04) and the sweep kept writing it although
 * training uses per-member CEX buffers. The broad X_all is now computed
 * fresh via load_input() only when flat/shuffle/class-voting need it, and
 * the sweep runs without any .pre (new in-memory xform cache comes later,
 * see plans/plan-2026-08-05-xform-cache.md).
 * See: bugs/bug-2026-08-05-sweep-old-pre-format.md */
#ifndef KI_PREPPED_MAX_BYTES
#define KI_PREPPED_MAX_BYTES (2ULL * 1024ULL * 1024ULL * 1024ULL)
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * LOAD ONE SLOT — encode exactly one encoding slot from raw pixels
 * ═══════════════════════════════════════════════════════════════════════
 * Returns buffer of n_samples × n_cont[vi] uint32. Caller must free(). */
static __attribute__((unused)) uint32_t *load_input_slot(const uint8_t *X_raw,
                                                          int n_samples, int vi) {
    if (vi < 0 || vi >= aa.enc_count) return NULL;
    int typ = (int)aa.enc_array[vi].type;
    int w   = (int)aa.enc_array[vi].width;
    if (w < 1) w = KI_ENC_WIDTH_DEFAULT;
    if (typ < 0) typ = KI_ENC_LIN7;
    int n_cont = (KI_COLORS > 1 ? KI_NC : NC) * w / KI_BIT_WIDTH;
    int pack = 32 / w, shift = w;
    uint32_t *X = (uint32_t *)ki_xmalloc((size_t)n_samples * (size_t)n_cont * sizeof(uint32_t));
#if KI_COLORS > 1
    int col = (int)aa.enc_array[vi].color;
    for (int s = 0; s < n_samples; s++) {
        uint8_t px[COLOR_NB][1024];
        for (int px_i = 0; px_i < 1024; px_i++) {
            size_t base = (size_t)s * KI_PX;
            int r_val = (int)X_raw[base + (size_t)px_i];
            int g_val = (int)X_raw[base + 1024 + (size_t)px_i];
            int b_val = (int)X_raw[base + 2048 + (size_t)px_i];
            uint8_t blk[COLOR_NB];
            ki_blocks_from_rgb(r_val, g_val, b_val, blk);
            for (int ci = 0; ci < COLOR_NB; ci++) px[ci][px_i] = blk[ci];
        }
        ki_compute_edge(px, 32, 32); ki_compute_binary(px, 32, 32);
        ki_compute_lbp(px, 32, 32); ki_compute_dog(px, 32, 32);
        ki_compute_var(px, 32, 32); ki_compute_dir(px, 32, 32);
        ki_compute_range(px, 32, 32); ki_compute_lbp_rg(px, 32, 32);
        ki_compute_lbp_rb(px, 32, 32); ki_compute_lbp_gb(px, 32, 32);
        uint32_t *row_out = X + (size_t)s * (size_t)n_cont;
        for (int c = 0; c < n_cont; c++) {
            uint32_t val = 0;
            for (int k = 0; k < pack; k++) {
                uint8_t pv = px[col][c * pack + k];
                uint32_t ev = enc_lut_get(typ, w, pv);
                val |= ev << (unsigned)(k * shift);
            }
            row_out[c] = val;
        }
    }
#else
    for (int s = 0; s < n_samples; s++) {
        uint32_t *row_out = X + (size_t)s * (size_t)n_cont;
        for (int c = 0; c < n_cont; c++) {
            uint32_t val = 0;
            for (int k = 0; k < pack; k++) {
                size_t p = (size_t)s * KI_PX + (size_t)c * (size_t)pack + (size_t)k;
                uint8_t pv = X_raw[p];
                uint32_t ev = enc_lut_get(typ, w, pv);
                val |= ev << (unsigned)(k * shift);
            }
            row_out[c] = val;
        }
    }
#endif
    return X;
}

#endif /* KI_LOAD_H */
