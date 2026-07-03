/*
 * lib/enc-lut.h — Precomputed Encoding Lookup Table
 * ====================================================
 *
 * Replaces on-the-fly powf() / expf() in ki_apply_enc_w() with
 * a 256-entry lookup table per (encoding, width) combination.
 *
 * Usage:
 *   1. #include "ki-common.h" first (defines ki_apply_enc_w)
 *   2. #include "enc-lut.h"
 *   3. enc_lut_init_all()  — after ki_parse_args(), before load_input()
 *   4. enc_lut_get(enc, width, pv) — fast pixel encoding
 *
 * Tables are precomputed once and cached indefinitely.
 * Covers all 9 encodings × 3 widths (8, 16, 32) = 27 tables × 256 = 27 KB.
 */
#ifndef ENC_LUT_H
#define ENC_LUT_H

#include <stdint.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Internal state — static cache within each translation unit
 * ═══════════════════════════════════════════════════════════════════════ */

#define _ENC_LUT_NENC  9   /* enum ki_encoding: RAW..SIG = 0..8 */
#define _ENC_LUT_NWI   3   /* widths: 8, 16, 32 */

static uint32_t _enc_lut_tab[_ENC_LUT_NENC][_ENC_LUT_NWI][256];
static int      _enc_lut_rdy[_ENC_LUT_NENC][_ENC_LUT_NWI];  /* init to 0 via BSS */

/* ── Width → index (0,1,2) ────────────────────────────────────── */
static inline int _enc_lut_wi(int w) {
    return w == 8 ? 0 : (w == 16 ? 1 : 2);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Public API
 * ═══════════════════════════════════════════════════════════════════════ */

/* ── Initialize one (enc, width) combination ──────────────────── */
/* Requires ki_apply_enc_w() to be visible (include ki-common.h). */
static inline void enc_lut_init(int enc, int width) {
    int wi = _enc_lut_wi(width);
    if (_enc_lut_rdy[enc][wi]) return;
    _enc_lut_rdy[enc][wi] = 1;
    for (int pv = 0; pv < 256; pv++)
        _enc_lut_tab[enc][wi][pv] = ki_apply_enc_w((uint8_t)pv, enc, width);
}

/* ── Fast lookup (pre-initialized) ────────────────────────────── */
static inline uint32_t enc_lut_get(int enc, int width, uint8_t pv) {
    return _enc_lut_tab[enc][_enc_lut_wi(width)][pv];
}

/* ── Initialize all (enc, width) pairs currently in use ───────── */
/* Must be called after ki_parse_args() and before any load_input().
 * Requires aa.enc[] and aa.enc_width[] to be resolved. */
static inline void enc_lut_init_all(void) {
    /* Collect unique (enc, width) pairs from aa.enc[] */
    int done[_ENC_LUT_NENC][_ENC_LUT_NWI];
    memset(done, 0, sizeof(done));
    /* Per-block encodings */
    for (int b = 0; b < COLOR_NB; b++) {
        int e = (int)aa.enc[b];
        int w = (int)aa.enc_width[b];
        if (e < 0 || e >= _ENC_LUT_NENC) continue;
        if (w != 8 && w != 16 && w != 32) w = 8;
        int wi = _enc_lut_wi(w);
        if (done[e][wi]) continue;
        done[e][wi] = 1;
        enc_lut_init(e, w);
    }
    /* Multi-encodings (virtuelle Blöcke für MNIST) */
    for (int i = 0; i < aa.enc_count; i++) {
        int e = (int)aa.enc_array[i].type;
        int w = (int)aa.enc_array[i].width;
        if (e < 0 || e >= _ENC_LUT_NENC) continue;
        if (w != 8 && w != 16 && w != 32) w = 8;
        int wi = _enc_lut_wi(w);
        if (done[e][wi]) continue;
        done[e][wi] = 1;
        enc_lut_init(e, w);
    }
}

#endif /* ENC_LUT_H */
