/*
 * lib/ki-encoding.h — Thermometer Encoding for DRAM-native MLP
 * ===============================================================
 *
 * Self-contained Header for thermometer Encoding von Pixelwerten.
 * No dependency von ki_Args, ki-common.h oder Otto Score.
 *
 * Contains:
 *   enum ki_encoding         — Encoding-Typen (RAW, LIN7, …, SIG)
 *   ki_enc_parse()           — String → (encoding, width)
 *   ki_apply_enc_w()         — Pixel (0..255) → Thermometer-Bitmaske
 *   enc_lut_init_enc()       — LUT vorberechnen
 *   enc_lut_get()            — schneller LUT-Lookup
 *   enum ki_color_bit        — Farbblock-Definitionen
 *   COLOR_NB                 — number of color blocks
 *   ki_blocks_from_rgb()     — RGB → Block-Array
 *   ki_color_name()          — Blockname for display
 *
 * Usage:
 *   1. #include "ki-encoding.h"
 *   2. enc_lut_init_enc(KI_ENC_EXP, 8);  // once
 *   3. uint32_t bits = enc_lut_get(KI_ENC_EXP, 8, pixel);  // per pixel
 */
#ifndef KI_ENCODING_H
#define KI_ENCODING_H

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * IMAGE TRANSFORMS — Spiegelung an Achsen + Diagonalen
 * ═══════════════════════════════════════════════════════════════════════
 * Applied to raw uint8 pixel buffer BEFORE channel computation + encoding.
 * All transforms operate on square images (w == h).
 * CIFAR: 3 planes × 32×32, MNIST/Fashion: 1 plane × 28×28.
 * buffer: [plane0(w×h)][plane1(w×h)][...], each plane row-major.
 */
enum ki_xform {
    KI_XFORM_ID      = 0,   /* Identity */
    KI_XFORM_HFLIP   = 1,   /* Horizontal flip (left↔right) */
    KI_XFORM_VFLIP   = 2,   /* Vertical flip (top↔bottom) */
    KI_XFORM_DFLIP1  = 3,   /* Main diagonal flip (transpose) */
    KI_XFORM_DFLIP2  = 4,   /* Anti-diagonal flip */
    KI_XFORM_ROT90   = 5,   /* Rotate 90° clockwise */
    KI_XFORM_ROT180  = 6,   /* Rotate 180° (hflip+vflip) */
    KI_XFORM_ROT270  = 7,   /* Rotate 270° clockwise (≡ rot90⁻¹) */
    KI_XFORM_COUNT   = 8
};

/* ── Xform short name for display ──────────────────────────────── */
static inline const char *ki_xform_name(int xf) {
    static const char *names[] = {
        [KI_XFORM_ID]     = "id",
        [KI_XFORM_HFLIP]  = "hflip",
        [KI_XFORM_VFLIP]  = "vflip",
        [KI_XFORM_DFLIP1] = "dflip1",
        [KI_XFORM_DFLIP2] = "dflip2",
        [KI_XFORM_ROT90]  = "rot90",
        [KI_XFORM_ROT180] = "rot180",
        [KI_XFORM_ROT270] = "rot270",
    };
    if (xf >= 0 && xf < KI_XFORM_COUNT) return names[xf];
    return "?";
}

/* ── Apply image transform to raw uint8 pixel buffer ──────────────
 * out:   output buffer (may alias in if xform=ID)
 * in:    input buffer
 * w, h: image width and height (must be square for diagonal flips)
 * ch:    number of color planes (3 for CIFAR, 1 for MNIST/Fashion)
 * xf:    ki_xform enum value
 * The buffer layout is: [plane0[w*h]][plane1[w*h]]...[planeN[w*h]]
 * Each plane is row-major. */
static inline void ki_xform_raw(uint8_t *restrict out,
                                 const uint8_t *restrict in,
                                 int w, int h, int ch, int xf) {
    if (xf == KI_XFORM_ID) {
        if (out != in) memcpy(out, in, (size_t)(w * h * ch));
        return;
    }
    int stride = w * h;  /* pixels per plane */
    for (int pl = 0; pl < ch; pl++) {
        const uint8_t *src = in + (size_t)pl * (size_t)stride;
        uint8_t *dst = out + (size_t)pl * (size_t)stride;
        switch (xf) {
        case KI_XFORM_HFLIP:
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++)
                    dst[y * w + x] = src[y * w + (w - 1 - x)];
            }
            break;
        case KI_XFORM_VFLIP:
            for (int y = 0; y < h; y++) {
                memcpy(dst + y * w, src + (h - 1 - y) * w, (size_t)w);
            }
            break;
        case KI_XFORM_DFLIP1:
            /* Transpose: out[y][x] = in[x][y] (main diagonal) */
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    dst[y * w + x] = src[x * w + y];
            break;
        case KI_XFORM_DFLIP2:
            /* Anti-diagonal: out[y][x] = in[w-1-x][h-1-y] */
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    dst[y * w + x] = src[(w - 1 - x) * w + (h - 1 - y)];
            break;
        case KI_XFORM_ROT90:
            /* Rotate 90° clockwise: out[y][x] = in[x][h-1-y] */
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    dst[y * w + x] = src[x * w + (h - 1 - y)];
            break;
        case KI_XFORM_ROT180:
            /* Rotate 180°: out[y][x] = in[h-1-y][w-1-x] (= hflip(vflip)) */
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    dst[y * w + x] = src[(h - 1 - y) * w + (w - 1 - x)];
            break;
        case KI_XFORM_ROT270:
            /* Rotate 270° clockwise (= rot90⁻¹): out[y][x] = in[w-1-x][y] */
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    dst[y * w + x] = src[(w - 1 - x) * w + y];
            break;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * ENCODING TYPES
 * ═══════════════════════════════════════════════════════════════════════ */

/* ── Thermometer encoding: popcount(encode(pv)) ∝ brightness ──── */
enum ki_encoding {
    KI_ENC_RAW      = 0,   /* Raw pixel value (0..255), keine Transformation  */
    KI_ENC_LIN7     = 1,   /* Uniform, 7 Stufen (pv>>5 → popcount=1..7)       */
    KI_ENC_LIN8     = 2,   /* Uniform, 8 Stufen (pv*8>>8 → popcount=1..8)     */
    KI_ENC_DOWN     = 3,   /* Bottom-weighted (more resolution in shadows)    */
    KI_ENC_UP       = 4,   /* Top-weighted (more resolution in highlights)    */
    KI_ENC_MID      = 5,   /* Mid-weighted (more resolution in midtones)      */
    KI_ENC_LOG      = 6,   /* Logarithmisch (natural brightness perception)   */
    KI_ENC_EXP      = 7,   /* Exponentiell (heavily top-weighted)             */
    KI_ENC_SIG      = 8,   /* S-shaped (Sigmoid, smooth transition)           */
    KI_ENC_SQRT     = 9,   /* Square root — softer than exp, more bright res  */
    KI_ENC_CBRT     = 10,  /* Cube root — even softer, natural image curve    */
    KI_ENC_GAMMA    = 11,  /* Gamma 0.45 — tunable power-law (complementary)  */
    KI_ENC_TRIANGLE = 12,  /* Triangle — peaks at midtones (128), zero ends   */
    KI_ENC_INV_EXP  = 13,  /* Inverse exp — dark emphasis, 1-e^(-k·pv)        */
    KI_ENC_COUNT    = 14
};

#define KI_ENC_WIDTH_DEFAULT 8

/* ═══════════════════════════════════════════════════════════════════════
 * ENCODING NAMES (for display)
 * ═══════════════════════════════════════════════════════════════════════ */

/* ── Kurzname (ohne Width-Suffix) ──────────────────────────────── */
static inline const char *ki_enc_name_short(int enc) {
    static const char *names[] = {
        [KI_ENC_RAW]     = "raw",
        [KI_ENC_LIN7]    = "lin7",
        [KI_ENC_LIN8]    = "lin",
        [KI_ENC_DOWN]    = "down",
        [KI_ENC_UP]      = "up",
        [KI_ENC_MID]     = "mid",
        [KI_ENC_LOG]     = "log",
        [KI_ENC_EXP]     = "exp",
        [KI_ENC_SIG]     = "sig",
        [KI_ENC_SQRT]    = "sqrt",
        [KI_ENC_CBRT]    = "cbrt",
        [KI_ENC_GAMMA]   = "gamma",
        [KI_ENC_TRIANGLE]= "tri",
        [KI_ENC_INV_EXP] = "inv-exp",
    };
    if (enc >= 0 && enc < KI_ENC_COUNT) return names[enc];
    return "?";
}

/* ── All encoding names as comma-separated string (for error messages) ─ */
static inline const char *ki_enc_names_all(void) {
    static char _buf[256];
    if (_buf[0]) return _buf;  /* cached after first call */
    int pos = 0;
    for (int e = 0; e < KI_ENC_COUNT; e++) {
        const char *n = ki_enc_name_short(e);
        if (!n || n[0] == '?') continue;
        if (pos > 0) _buf[pos++] = ',';
        while (*n && pos < (int)sizeof(_buf) - 2) _buf[pos++] = *n++;
    }
    _buf[pos] = '\0';
    return _buf;
}

/* ── All color names as comma-separated string (for error messages) ─── */
/* ═══════════════════════════════════════════════════════════════════════
 * ENCODING PARSER — String → (encoding, width)
 * ═══════════════════════════════════════════════════════════════════════
 * Tokens: "exp16" → returns KI_ENC_EXP, *out_width=16
 *         "exp"   → returns KI_ENC_EXP, *out_width=KI_ENC_WIDTH_DEFAULT
 * Returns -1 on unknown encoding. */

/* Portable case-insensitive string comparison (avoids POSIX strcasecmp) */
static inline int ki_strcasecmp(const char *a, const char *b) {
    while (*a && *b) {
        unsigned char ca = (unsigned char)*a;
        unsigned char cb = (unsigned char)*b;
        if (ca >= 'A' && ca <= 'Z') ca += 32;
        if (cb >= 'A' && cb <= 'Z') cb += 32;
        if (ca != cb) return (int)ca - (int)cb;
        a++; b++;
    }
    return (int)(unsigned char)*a - (int)(unsigned char)*b;
}

static inline int ki_enc_parse(const char *tok, int *out_width) {
    int tok_len = (int)strlen(tok);
    if (tok_len > 32) return -1;
    static const int widths[] = {32, 16, 8};
    for (int wi = 0; wi < 3; wi++) {
        int w = widths[wi];
        char wstr[4];
        snprintf(wstr, sizeof(wstr), "%d", w);
        int wslen = (int)strlen(wstr);
        /* Number suffix: use strcmp (digits are case-insensitive by nature) */
        if (tok_len > wslen && memcmp(tok + tok_len - wslen, wstr, (size_t)wslen) == 0) {
            char prefix[64];  /* 64 > max tok_len 32, safe for any prefix */
            int plen = tok_len - wslen;
            memcpy(prefix, tok, (size_t)plen);
            prefix[plen] = '\0';
            if (ki_strcasecmp(prefix, "raw")  == 0) { if (out_width) *out_width = w; return KI_ENC_RAW; }
            if (ki_strcasecmp(prefix, "lin7") == 0) { if (out_width) *out_width = w; return KI_ENC_LIN7; }
            if (ki_strcasecmp(prefix, "lin8") == 0 || ki_strcasecmp(prefix, "lin") == 0) { if (out_width) *out_width = w; return KI_ENC_LIN8; }
            if (ki_strcasecmp(prefix, "down") == 0 || ki_strcasecmp(prefix, "unten") == 0) { if (out_width) *out_width = w; return KI_ENC_DOWN; }
            if (ki_strcasecmp(prefix, "up")   == 0 || ki_strcasecmp(prefix, "oben") == 0) { if (out_width) *out_width = w; return KI_ENC_UP; }
            if (ki_strcasecmp(prefix, "mid")  == 0 || ki_strcasecmp(prefix, "mitte") == 0) { if (out_width) *out_width = w; return KI_ENC_MID; }
            if (ki_strcasecmp(prefix, "log")  == 0) { if (out_width) *out_width = w; return KI_ENC_LOG; }
            if (ki_strcasecmp(prefix, "exp")  == 0) { if (out_width) *out_width = w; return KI_ENC_EXP; }
            if (ki_strcasecmp(prefix, "sig")     == 0) { if (out_width) *out_width = w; return KI_ENC_SIG; }
            if (ki_strcasecmp(prefix, "sqrt")    == 0) { if (out_width) *out_width = w; return KI_ENC_SQRT; }
            if (ki_strcasecmp(prefix, "cbrt")    == 0) { if (out_width) *out_width = w; return KI_ENC_CBRT; }
            if (ki_strcasecmp(prefix, "gamma")   == 0) { if (out_width) *out_width = w; return KI_ENC_GAMMA; }
            if (ki_strcasecmp(prefix, "tri")     == 0 || ki_strcasecmp(prefix, "triangle") == 0) { if (out_width) *out_width = w; return KI_ENC_TRIANGLE; }
            if (ki_strcasecmp(prefix, "inv-exp") == 0 || ki_strcasecmp(prefix, "invexp") == 0) { if (out_width) *out_width = w; return KI_ENC_INV_EXP; }
        }
    }
    if (out_width) *out_width = KI_ENC_WIDTH_DEFAULT;
    if (ki_strcasecmp(tok, "raw")  == 0) return KI_ENC_RAW;
    if (ki_strcasecmp(tok, "lin7") == 0) return KI_ENC_LIN7;
    if (ki_strcasecmp(tok, "lin8") == 0 || ki_strcasecmp(tok, "lin") == 0) return KI_ENC_LIN8;
    if (ki_strcasecmp(tok, "down") == 0 || ki_strcasecmp(tok, "unten") == 0) return KI_ENC_DOWN;
    if (ki_strcasecmp(tok, "up")   == 0 || ki_strcasecmp(tok, "oben") == 0) return KI_ENC_UP;
    if (ki_strcasecmp(tok, "mid")  == 0 || ki_strcasecmp(tok, "mitte") == 0) return KI_ENC_MID;
    if (ki_strcasecmp(tok, "log")  == 0 || ki_strcasecmp(tok, "logarithmisch") == 0) return KI_ENC_LOG;
    if (ki_strcasecmp(tok, "exp")  == 0 || ki_strcasecmp(tok, "exponentiell") == 0) return KI_ENC_EXP;
    if (ki_strcasecmp(tok, "sig")  == 0 || ki_strcasecmp(tok, "sigmoid") == 0) return KI_ENC_SIG;
    if (ki_strcasecmp(tok, "sqrt") == 0) return KI_ENC_SQRT;
    if (ki_strcasecmp(tok, "cbrt") == 0) return KI_ENC_CBRT;
    if (ki_strcasecmp(tok, "gamma") == 0) return KI_ENC_GAMMA;
    if (ki_strcasecmp(tok, "tri")  == 0 || ki_strcasecmp(tok, "triangle") == 0) return KI_ENC_TRIANGLE;
    if (ki_strcasecmp(tok, "inv-exp") == 0 || ki_strcasecmp(tok, "invexp") == 0) return KI_ENC_INV_EXP;
    return -1;
}

/* ═══════════════════════════════════════════════════════════════════════
 * APPLY ENCODING — Pixel (0..255) → Thermometer-Bitmaske
 * ═══════════════════════════════════════════════════════════════════════
 * Each encoding converts an 8-bit pixel value into a bitmask,
 * whose popcount is proportional to brightness.
 *
 * width = output width in bits (8, 16, 32)
 * LOG/EXP/SIG verwenden float — for host training (nicht DRAM). */

static inline uint32_t ki_apply_enc_w(uint8_t pv, int enc, int width) {
    int max_lev;
    switch (enc) {
      case KI_ENC_LIN7: max_lev = 7; break;
      default:          max_lev = width; break;
    }
    int level = 0;
    switch (enc) {
    case KI_ENC_RAW: return pv;
    case KI_ENC_LIN7: level = pv >> 5; break;
    case KI_ENC_LIN8:
        level = (int)(((uint32_t)pv * (uint32_t)max_lev) >> 8);
        break;
    case KI_ENC_DOWN: {
        int split = 64;
        int low_lev = max_lev * 3 / 7;
        if (pv < split)
            level = (int)(((uint32_t)pv * (uint32_t)low_lev) / (uint32_t)split);
        else
            level = low_lev + (int)(((uint32_t)(pv - split) * (uint32_t)(max_lev - low_lev))
                                   / (uint32_t)(256 - split));
        break;
    }
    case KI_ENC_UP: {
        int split = 192;
        int low_lev = max_lev * 4 / 7;
        if (pv > split)
            level = low_lev + (int)(((uint32_t)(pv - split) * (uint32_t)(max_lev - low_lev))
                                   / (uint32_t)(255 - split));
        else
            level = (int)(((uint32_t)pv * (uint32_t)low_lev) / (uint32_t)split);
        break;
    }
    case KI_ENC_MID: {
        int lo = 80, hi = 175;
        int low_lev = max_lev * 3 / 8;
        int mid_lev = max_lev - low_lev;
        if (pv < lo)
            level = (int)(((uint32_t)pv * (uint32_t)low_lev) / (uint32_t)lo);
        else if (pv > hi)
            level = low_lev + (int)(((uint32_t)(pv - hi) * (uint32_t)mid_lev)
                                   / (uint32_t)(255 - hi));
        else
            level = low_lev + (int)(((uint32_t)(pv - lo) * (uint32_t)mid_lev)
                                   / (uint32_t)(hi - lo));
        break;
    }
    case KI_ENC_LOG: {
        float l = log2f((float)(pv + 1)) * (float)max_lev / log2f(256.0f);
        level = (int)(l + 0.5f);
        break;
    }
    case KI_ENC_EXP: {
        float ex = (width == 16) ? 0.35f : 0.30f;
        level = (int)(powf(pv / 255.0f, ex) * (float)max_lev + 0.5f);
        break;
    }
    case KI_ENC_SIG: {
        float norm = pv / 255.0f;
        float s = 1.0f / (1.0f + expf(-12.0f * (norm - 0.5f)));
        level = (int)(s * (float)max_lev + 0.5f);
        break;
    }
    case KI_ENC_SQRT: {
        float s = sqrtf(pv / 255.0f);
        level = (int)(s * (float)max_lev + 0.5f);
        break;
    }
    case KI_ENC_CBRT: {
        float c = powf(pv / 255.0f, 1.0f / 3.0f);
        level = (int)(c * (float)max_lev + 0.5f);
        break;
    }
    case KI_ENC_GAMMA: {
        /* Gamma ≈ 0.45 — between exp(0.30) and sqrt(0.50) */
        float g = powf(pv / 255.0f, 0.45f);
        level = (int)(g * (float)max_lev + 0.5f);
        break;
    }
    case KI_ENC_TRIANGLE: {
        /* Triangle: peak at 128, zero at 0 and 255. min(pv, 255-pv) × 2 */
        int d = (pv < 128) ? pv : (255 - pv);
        level = (int)((uint32_t)d * (uint32_t)max_lev * 2U / 255U);
        break;
    }
    case KI_ENC_INV_EXP: {
        /* Inverse exp: 1 - e^(-k·pv/255). k≈4 gives good spread. */
        float ie = 1.0f - expf(-4.0f * pv / 255.0f);
        level = (int)(ie * (float)max_lev + 0.5f);
        break;
    }
    default: return pv;
    }
    if (level <= 0) return 0;
    if ((unsigned)level >= 32) return 0xFFFFFFFFu;
    return ((uint32_t)1u << (unsigned)level) - 1u;
}

/* ── 8-Bit Kurzform ──────────────────────────────────────────── */
static inline uint8_t ki_apply_enc(uint8_t pv, int enc) {
    return (uint8_t)ki_apply_enc_w(pv, enc, 8);
}

/* ═══════════════════════════════════════════════════════════════════════
 * ENCODING LOOKUP TABLE (LUT) — precomputed fast path
 * ═══════════════════════════════════════════════════════════════════════
 * Replaces on-the-fly powf()/expf() in ki_apply_enc_w() mit
 * einer 256-Eintrag-Tabelle pro (encoding, width)-Kombination.
 *
 * Covers all 9 encodings × 3 widths (8, 16, 32) = 27 tables × 256 = 27 KB.
 * Initialized via BSS = 0, hence needed. */

#define _KI_ENC_NENC  14
#define _KI_ENC_NWI   3

static uint32_t _enc_lut_tab[_KI_ENC_NENC][_KI_ENC_NWI][256];
static int      _enc_lut_rdy[_KI_ENC_NENC][_KI_ENC_NWI];

static inline int _enc_lut_wi(int w) {
    return w == 8 ? 0 : (w == 16 ? 1 : 2);
}

/* ── Eine (enc, width)-Tabelle initialisieren ───────────────── */
static inline void enc_lut_init_enc(int enc, int width) {
    int wi = _enc_lut_wi(width);
    if (_enc_lut_rdy[enc][wi]) return;
    _enc_lut_rdy[enc][wi] = 1;
    for (int pv = 0; pv < 256; pv++)
        _enc_lut_tab[enc][wi][pv] = ki_apply_enc_w((uint8_t)pv, enc, width);
}

/* ── Fast lookup (Tabelle must be initialized) ─────── */
static inline uint32_t enc_lut_get(int enc, int width, uint8_t pv) {
    return _enc_lut_tab[enc][_enc_lut_wi(width)][pv];
}

/* ═══════════════════════════════════════════════════════════════════════
 * COLOR DEFINITIONS — Bit positions for color analysis blocks
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Each analysis block has a fixed bit position (0..COLOR_NB-1).
 * Bit 0.*is the single grayscale block for MNIST.
 * All other blocks (R,G,B,Y,…) are shifted by 1.
 *
 * Default-Masken:
 *   CIFAR: r+g+b = (1<<COLOR_R)|(1<<COLOR_G)|(1<<COLOR_B) = 0x0E
 *   MNIST: mnist = (1<<COLOR_MNIST) = 1 */

enum ki_color_bit {
    COLOR_MNIST = 0,
    COLOR_R     = 1,
    COLOR_G     = 2,
    COLOR_B     = 3,
    COLOR_Y     = 4,   /* ITU-R BT.601: Y=(77R+150G+29B)>>8 */
    COLOR_YL    = 5,   /* ITU-R BT.709: Y=(54R+183G+18B)>>8 */
    COLOR_AL    = 6,   /* (R+G)>>1           (lum) */
    COLOR_AM    = 7,   /* R-G opponent       (rg)  */
    COLOR_AP    = 8,   /* B-(R+G)/2 opponent (by)  */
    COLOR_RG    = 9,   /* R-G opponent (clamp128)   */
    COLOR_RB    = 10,  /* R-B opponent (clamp128)   */
    COLOR_GB    = 11,  /* G-B opponent (clamp128)   */
    COLOR_BL    = 12,  /* (R+B)>>1 */
    COLOR_BM    = 13,  /* R-B opponent */
    COLOR_BP    = 14,  /* G-(R+B)/2 opponent */

    COLOR_H     = 15,  /* Hue */
    COLOR_S     = 16,  /* Saturation */
    COLOR_C     = 17,  /* Contrast (chromatische Varianz) */

    COLOR_CL    = 18,  /* (G+B)>>1 */
    COLOR_CM    = 19,  /* G-B opponent */
    COLOR_CP    = 20,  /* R-(G+B)/2 opponent */

    COLOR_EDGE  = 21,  /* Sobel edges on Y luminance (for --channels edge) */
    COLOR_BIN   = 22,  /* Otsu-binarized Y (filled black/white) */
    COLOR_LBP   = 23,  /* Local Binary Pattern (texture) */
    COLOR_DOG   = 24,  /* Difference of Gaussians (band-pass) */
    COLOR_VAR   = 25,  /* Local variance (roughness) */
    COLOR_DIR   = 26,  /* Gradient direction (8-bin quantized) */
    COLOR_RANGE = 27,  /* Local range (max-min in 3×3) */
    COLOR_LBP_RG = 28, /* LBP on RG opponent (chromatic texture) */
    COLOR_DIST  = 29,  /* Center distance (positional encoding) */

    COLOR_NB    = 30   /* number of Farben */
};

/* ── Block-Namen for display ────────────────────────────────── */
static inline const char *ki_color_name(int bit) {
    static const char *names[COLOR_NB] = {
        [COLOR_MNIST] = "mnist",
        [COLOR_R]     = "R",
        [COLOR_G]     = "G",
        [COLOR_B]     = "B",
        [COLOR_Y]     = "Y",
        [COLOR_YL]    = "YL",
        [COLOR_AL]    = "AL",
        [COLOR_AM]    = "AM",
        [COLOR_AP]    = "AP",
        [COLOR_RG]    = "RG",
        [COLOR_RB]    = "RB",
        [COLOR_GB]    = "GB",
        [COLOR_BL]    = "BL",
        [COLOR_BM]    = "BM",
        [COLOR_BP]    = "BP",
        [COLOR_H]     = "H",
        [COLOR_S]     = "S",
        [COLOR_C]     = "C",
        [COLOR_CL]    = "CL",
        [COLOR_CM]    = "CM",
        [COLOR_CP]    = "CP",
        [COLOR_EDGE]  = "edge",
        [COLOR_BIN]   = "bin",
        [COLOR_LBP]   = "lbp",
        [COLOR_DOG]   = "dog",
        [COLOR_VAR]   = "var",
        [COLOR_DIR]   = "dir",
        [COLOR_RANGE] = "range",
        [COLOR_LBP_RG]= "lbp-rg",
        [COLOR_DIST]  = "dist",
    };
    if ((unsigned)bit < COLOR_NB) return names[bit];
    return "?";
}

/* ── All color names as comma-separated string (for error messages) ─── */
static inline const char *ki_color_names_all(void) {
    static char _buf[512];
    if (_buf[0]) return _buf;
    int pos = 0;
    for (int c = 0; c < COLOR_NB; c++) {
        const char *n = ki_color_name(c);
        if (!n || n[0] == '?') continue;
        if (pos > 0) _buf[pos++] = ',';
        while (*n && pos < (int)sizeof(_buf) - 2) _buf[pos++] = *n++;
    }
    _buf[pos] = '\0';
    return _buf;
}

/* ── clamp to 0..255 (helper for ki_blocks_from_rgb) ───────── */
static inline int ki_clamp_u8(int x) {
    return (x < 0) ? 0 : (x > 255) ? 255 : x;
}

/* ── RGB.*blocks per pixel) ──────────
 * Computes all 21 color analysis blocks aus R,G,B (0..255).
 * The blocks are stored
 * indiziert via enum ki_color_bit. */
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

    /* Hue */
    {   float dx = 2.0f * (float)r - (float)g - (float)b;
        float dy = 1.0f * ((float)g - (float)b);
        float hue_f = atan2f(dy, dx) / (2.0f * 3.14159265358979323846f);
        int hue = (int)((hue_f + 0.5f) * 255.0f);
        if (hue < 0) hue = 0;
        if (hue > 255) hue = 255;
        blocks[COLOR_H] = (uint8_t)hue;
    }
    /* Saturation */
    {   int mx = r, mn = r;
        if (g > mx) { mx = g; } if (g < mn) { mn = g; }
        if (b > mx) { mx = b; } if (b < mn) { mn = b; }
        blocks[COLOR_S] = (uint8_t)(mx - mn);
    }
    /* Contrast.*computed by Sobel in load_input.*placeholder) */
    blocks[COLOR_C] = (uint8_t)r;
    /* Edge.*computed by Sobel in load_input.*placeholder) */
    blocks[COLOR_EDGE] = 0;
    /* Binary.*computed by Otsu in load_input.*placeholder) */
    blocks[COLOR_BIN] = 0;
    /* LBP, DoG, Variance — computed per image in post-processing */
    blocks[COLOR_LBP] = 0;
    blocks[COLOR_DOG] = 0;
    blocks[COLOR_VAR] = 0;
    blocks[COLOR_DIR] = 0;
    blocks[COLOR_RANGE] = 0;
    blocks[COLOR_LBP_RG] = 0;
    blocks[COLOR_DIST] = 0;

    blocks[COLOR_CL] = (uint8_t)((g + b) >> 1);
    blocks[COLOR_CM] = (uint8_t)ki_clamp_u8(128 + (g - b));
    blocks[COLOR_CP] = (uint8_t)ki_clamp_u8(128 + (r - (g + b)/2));
}

/* ═══════════════════════════════════════════════════════════════════════
 * PRINT MEMBER STRUCTURE — unified format for Otto/Hebbian/Adam
 * ═══════════════════════════════════════════════════════════════════════
 * Prints a consistent Member-Liste aus:
 *   Structure: M0(COL=ENC_WIDTH), M1(COL=ENC_WIDTH), ...
 *
 * Parameter:
 *   colors[]  — ki_color_bit for each member (-1 = unbekannt)
 *   types[]   - Encoding-Typ (-1 = kein Encoding)
 *   widths[]  - Encoding-Breite (-1 = unbekannt)
 *   n         - number of Members
 *   ens       - ensemble count (for display: × EN=ens)
 */
static inline void ki_print_member_structure(const int *colors,
                                               const int *types,
                                               const int *widths,
                                               int n, int ens) {
    printf("  Structure: ");
    for (int i = 0; i < n; i++) {
        if (i > 0) printf(", ");
        printf("M%d(", i);
        if (colors && colors[i] >= 0)
            printf("%s", ki_color_name(colors[i]));
        if (types && types[i] >= 0)
            printf("=%s%d", ki_enc_name_short(types[i]),
                   widths && widths[i] >= 0 ? widths[i] : 8);
        printf(")");
    }
    if (ens > 1) printf("  × EN=%d", ens);
    printf("\n");
}

/* ═══════════════════════════════════════════════════════════════════════
 * EDGE DETECTION — Sobel 3×3 auf Y-Luminanz
 * ═══════════════════════════════════════════════════════════════════════
 * Expects: px[COLOR_NB][1024] with valid COLOR_Y (ITU-601 Y).
 * Computes: COLOR_EDGE (Sobel-Magnitude) + COLOR_C (Sobel auf AL).
 * px is updated IN PLACE. Call AFTER ki_blocks_from_rgb().
 * w=32, h=32 (CIFAR-10 resolution).
 */
__attribute__((unused))
static inline void ki_compute_edge(uint8_t px[COLOR_NB][1024], int w, int h) {
    /* 3×3 Sobel kernel needs at least 3×3 pixels */
    if (w < 3 || h < 3) return;
    /* Sobel on Y (ITU-601) → COLOR_EDGE */
    {   int yn[1024];
        for (int p = 0; p < w * h; p++) yn[p] = px[COLOR_Y][p];
        for (int y = 1; y < h - 1; y++) {
            for (int x = 1; x < w - 1; x++) {
                int i = y * w + x;
                int gx = -yn[i-w-1] + yn[i-w+1]
                         -2*yn[i-1]  + 2*yn[i+1]
                         -yn[i+w-1] + yn[i+w+1];
                int gy = -yn[i-w-1] -2*yn[i-w] -yn[i-w+1]
                         +yn[i+w-1] +2*yn[i+w] +yn[i+w+1];
                int mag = (int)(sqrtf((float)(gx*gx + gy*gy)) / 6.0f + 0.5f);
                if (mag > 255) mag = 255;
                px[COLOR_EDGE][i] = (uint8_t)mag;
            }
        }
        /* 1) Left/right borders: copy from col 1 / w-2 (valid rows y=1..h-2) */
        for (int y = 1; y < h - 1; y++) {
            int ro = y * w;
            uint8_t lv = px[COLOR_EDGE][ro + 1];
            uint8_t rv = px[COLOR_EDGE][ro + w - 2];
            px[COLOR_EDGE][ro + 0]     = lv;
            px[COLOR_EDGE][ro + w - 1] = rv;
        }
        /* 2) Top/bottom rows: copy from row 1 / h-2 (full width, covers corners) */
        for (int x = 0; x < w; x++) {
            px[COLOR_EDGE][0 * w + x]       = px[COLOR_EDGE][1 * w + x];
            px[COLOR_EDGE][(h - 1) * w + x] = px[COLOR_EDGE][(h - 2) * w + x];
        }
    }
    /* Sobel on AL (R+G)/2 → COLOR_C (better scaling) */
    {   int ln[1024];
        for (int p = 0; p < w * h; p++) ln[p] = px[COLOR_AL][p];
        for (int y = 1; y < h - 1; y++) {
            for (int x = 1; x < w - 1; x++) {
                int i = y * w + x;
                int gx = -ln[i-w-1] + ln[i-w+1]
                         -2*ln[i-1]  + 2*ln[i+1]
                         -ln[i+w-1] + ln[i+w+1];
                int gy = -ln[i-w-1] -2*ln[i-w] -ln[i-w+1]
                         +ln[i+w-1] +2*ln[i+w] +ln[i+w+1];
                int mag = (int)(sqrtf((float)(gx*gx + gy*gy)) / 4.0f + 0.5f);
                if (mag > 255) mag = 255;
                px[COLOR_C][i] = (uint8_t)mag;
            }
        }
        /* 1) Left/right borders: copy from col 1 / w-2 (valid rows y=1..h-2) */
        for (int y = 1; y < h - 1; y++) {
            int ro = y * w;
            uint8_t lv = px[COLOR_C][ro + 1];
            uint8_t rv = px[COLOR_C][ro + w - 2];
            px[COLOR_C][ro + 0]     = lv;
            px[COLOR_C][ro + w - 1] = rv;
        }
        /* 2) Top/bottom rows: copy from row 1 / h-2 (full width, covers corners) */
        for (int x = 0; x < w; x++) {
            px[COLOR_C][0 * w + x]       = px[COLOR_C][1 * w + x];
            px[COLOR_C][(h - 1) * w + x] = px[COLOR_C][(h - 2) * w + x];
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * BINARY THRESHOLD — Otsu auf Y-Luminanz → filled black/white
 * ═══════════════════════════════════════════════════════════════════════
 * Expects: px[COLOR_NB][1024] with valid COLOR_Y (ITU-601 Y).
 * Computes: COLOR_BIN (Otsu-binarisiert: 0 oder 255).
 * px is updated IN PLACE. Call AFTER ki_blocks_from_rgb().
 * w=32, h=32 (CIFAR-10 resolution).
 *
 * Otsu's Method: Finds threshold T that minimizes intra-class variance
 * minimiert (bzw. inter-class Varianz maximiert). Returns a
 * "filled" Schwarz/Weiss-Bild — Object regions sind 255, background 0.
 */
__attribute__((unused))
static inline void ki_compute_binary(uint8_t px[COLOR_NB][1024], int w, int h) {
    int N = w * h;
    /* Histogram der Y-Werte (0..255) */
    int hist[256] = {0};
    for (int p = 0; p < N; p++)
        hist[px[COLOR_Y][p]]++;

    /* Otsu: find threshold T that maximizes sigma_b² */
    int total = N;
    float sum = 0.0f;
    for (int i = 0; i < 256; i++) sum += (float)i * (float)hist[i];

    float sumB = 0.0f;
    int wB = 0, wF = 0;
    float max_var = 0.0f;
    int threshold = 128;  /* fallback */

    for (int i = 0; i < 256; i++) {
        wB += hist[i];
        if (wB == 0) continue;
        wF = total - wB;
        if (wF == 0) break;

        sumB += (float)i * (float)hist[i];
        float mB = sumB / (float)wB;
        float mF = (sum - sumB) / (float)wF;

        float var = (float)wB * (float)wF * (mB - mF) * (mB - mF);
        if (var > max_var) {
            max_var = var;
            threshold = i;
        }
    }

    /* Binarize: Y > threshold → 255, otherwise 0 */
    for (int p = 0; p < N; p++)
        px[COLOR_BIN][p] = (px[COLOR_Y][p] > threshold) ? 255 : 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * LOCAL BINARY PATTERN — texture descriptor on Y luminance
 * ═══════════════════════════════════════════════════════════════════════
 * Expects: px[COLOR_NB][1024] with valid COLOR_Y (ITU-601 Y).
 * Computes: COLOR_LBP — 8-bit LBP code per pixel.
 * Pure comparisons, no multiply-accumulate. */
__attribute__((unused))
static inline void ki_compute_lbp(uint8_t px[COLOR_NB][1024], int w, int h) {
    /* 3×3 kernel needs at least 3×3 pixels */
    if (w < 3 || h < 3) return;
    /* LBP for inner region (y=1..h-2, x=1..w-2) */
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int i = y * w + x;
            uint8_t center = px[COLOR_Y][i];
            uint8_t code = 0;
            /* Clockwise from top-left (standard LBP ordering) */
            if (px[COLOR_Y][i-w-1] >= center) code |= 0x80;  /* top-left      */
            if (px[COLOR_Y][i-w  ] >= center) code |= 0x40;  /* top           */
            if (px[COLOR_Y][i-w+1] >= center) code |= 0x20;  /* top-right     */
            if (px[COLOR_Y][i+1  ] >= center) code |= 0x10;  /* right         */
            if (px[COLOR_Y][i+w+1] >= center) code |= 0x08;  /* bottom-right  */
            if (px[COLOR_Y][i+w  ] >= center) code |= 0x04;  /* bottom        */
            if (px[COLOR_Y][i+w-1] >= center) code |= 0x02;  /* bottom-left   */
            if (px[COLOR_Y][i-1  ] >= center) code |= 0x01;  /* left          */
            px[COLOR_LBP][i] = code;
        }
    }
    /* 1) Left/right borders: copy from col 1 / w-2 (valid rows y=1..h-2) */
    for (int y = 1; y < h - 1; y++) {
        int ro = y * w;
        uint8_t lv = px[COLOR_LBP][ro + 1];
        uint8_t rv = px[COLOR_LBP][ro + w - 2];
        px[COLOR_LBP][ro + 0]     = lv;
        px[COLOR_LBP][ro + w - 1] = rv;
    }
    /* 2) Top/bottom rows: copy from row 1 / h-2 (full width, covers corners) */
    for (int x = 0; x < w; x++) {
        px[COLOR_LBP][0 * w + x]       = px[COLOR_LBP][1 * w + x];
        px[COLOR_LBP][(h - 1) * w + x] = px[COLOR_LBP][(h - 2) * w + x];
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * DIFFERENCE OF GAUSSIANS — band-pass edge detection on Y luminance
 * ═══════════════════════════════════════════════════════════════════════
 * Expects: px[COLOR_NB][1024] with valid COLOR_Y (ITU-601 Y).
 * Computes: COLOR_DOG — DoG(3,5) band-pass.
 * Uses integer box-blur approximations (no exp).
 * Border 2px: DoG not defined → repeat nearest valid value. */
__attribute__((unused))
static inline void ki_compute_dog(uint8_t px[COLOR_NB][1024], int w, int h) {
    /* DoG(3,5) needs at least 5×5 pixels */
    if (w < 5 || h < 5) return;
    /* Box blur r=1 (3×3) and r=2 (5×5), both on Y */
    int blur3[1024], blur5[1024];
    memset(blur3, 0, sizeof(blur3));
    memset(blur5, 0, sizeof(blur5));
    /* blur3: valid for y=1..h-2, x=1..w-2 */
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int i = y * w + x;
            int s3 = (int)px[COLOR_Y][i-w-1] + (int)px[COLOR_Y][i-w] + (int)px[COLOR_Y][i-w+1]
                   + (int)px[COLOR_Y][i-1]   + (int)px[COLOR_Y][i]   + (int)px[COLOR_Y][i+1]
                   + (int)px[COLOR_Y][i+w-1] + (int)px[COLOR_Y][i+w] + (int)px[COLOR_Y][i+w+1];
            blur3[i] = s3 / 9;
        }
    }
    /* blur5: valid for y=2..h-3, x=2..w-3 */
    for (int y = 2; y < h - 2; y++) {
        for (int x = 2; x < w - 2; x++) {
            int i = y * w + x;
            int s5 = 0;
            for (int dy = -2; dy <= 2; dy++)
                for (int dx = -2; dx <= 2; dx++)
                    s5 += (int)px[COLOR_Y][i + dy*w + dx];
            blur5[i] = s5 / 25;
        }
    }
    /* DoG: only where blur5 is valid (y=2..h-3, x=2..w-3) */
    for (int y = 2; y < h - 2; y++) {
        for (int x = 2; x < w - 2; x++) {
            int i = y * w + x;
            int dog = blur3[i] - blur5[i] + 128;
            if (dog < 0) dog = 0;
            if (dog > 255) dog = 255;
            px[COLOR_DOG][i] = (uint8_t)dog;
        }
    }
    /* 1) Left/right borders: copy from column 2 / w-3 (only valid rows y=2..h-3) */
    for (int y = 2; y < h - 2; y++) {
        int ro = y * w;
        uint8_t lv = px[COLOR_DOG][ro + 2];
        uint8_t rv = px[COLOR_DOG][ro + w - 3];
        px[COLOR_DOG][ro + 0] = lv;
        px[COLOR_DOG][ro + 1] = lv;
        px[COLOR_DOG][ro + w - 2] = rv;
        px[COLOR_DOG][ro + w - 1] = rv;
    }
    /* 2) Top/bottom 2 rows: copy from row 2 / h-3 (covers corners too) */
    for (int x = 0; x < w; x++) {
        uint8_t tv = px[COLOR_DOG][2 * w + x];
        uint8_t bv = px[COLOR_DOG][(h - 3) * w + x];
        px[COLOR_DOG][0 * w + x] = tv;
        px[COLOR_DOG][1 * w + x] = tv;
        px[COLOR_DOG][(h - 2) * w + x] = bv;
        px[COLOR_DOG][(h - 1) * w + x] = bv;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * LOCAL VARIANCE — texture roughness on Y luminance
 * ═══════════════════════════════════════════════════════════════════════
 * Expects: px[COLOR_NB][1024] with valid COLOR_Y (ITU-601 Y).
 * Computes: COLOR_VAR — mean absolute deviation in 3×3 patch.
 * Uses integer absolute diff, no multiply (cheaper than variance). */
__attribute__((unused))
static inline void ki_compute_var(uint8_t px[COLOR_NB][1024], int w, int h) {
    /* 3×3 kernel needs at least 3×3 pixels */
    if (w < 3 || h < 3) return;
    /* Variance only for inner region (y=1..h-2, x=1..w-2) */
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int i = y * w + x;
            /* Local mean (3×3) */
            int sum = (int)px[COLOR_Y][i-w-1] + (int)px[COLOR_Y][i-w] + (int)px[COLOR_Y][i-w+1]
                    + (int)px[COLOR_Y][i-1]   + (int)px[COLOR_Y][i]   + (int)px[COLOR_Y][i+1]
                    + (int)px[COLOR_Y][i+w-1] + (int)px[COLOR_Y][i+w] + (int)px[COLOR_Y][i+w+1];
            int mean = sum / 9;
            /* Mean absolute deviation */
            int mad = abs((int)px[COLOR_Y][i-w-1] - mean)
                    + abs((int)px[COLOR_Y][i-w]   - mean)
                    + abs((int)px[COLOR_Y][i-w+1] - mean)
                    + abs((int)px[COLOR_Y][i-1]   - mean)
                    + abs((int)px[COLOR_Y][i]     - mean)
                    + abs((int)px[COLOR_Y][i+1]   - mean)
                    + abs((int)px[COLOR_Y][i+w-1] - mean)
                    + abs((int)px[COLOR_Y][i+w]   - mean)
                    + abs((int)px[COLOR_Y][i+w+1] - mean);
            int v = mad / 9;  /* normalize to 0..255 (max=9*255/9=255) */
            px[COLOR_VAR][i] = (uint8_t)v;
        }
    }
    /* 1) Left/right borders: copy from col 1 / w-2 (valid rows y=1..h-2) */
    for (int y = 1; y < h - 1; y++) {
        int ro = y * w;
        uint8_t lv = px[COLOR_VAR][ro + 1];
        uint8_t rv = px[COLOR_VAR][ro + w - 2];
        px[COLOR_VAR][ro + 0]     = lv;
        px[COLOR_VAR][ro + w - 1] = rv;
    }
    /* 2) Top/bottom rows: copy from row 1 / h-2 (full width, covers corners) */
    for (int x = 0; x < w; x++) {
        px[COLOR_VAR][0 * w + x]       = px[COLOR_VAR][1 * w + x];
        px[COLOR_VAR][(h - 1) * w + x] = px[COLOR_VAR][(h - 2) * w + x];
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * GRADIENT DIRECTION — 8-bin quantized orientation on Sobel gx/gy
 * ═══════════════════════════════════════════════════════════════════════
 * Expects: px[COLOR_NB][1024] with valid COLOR_EDGE (Sobel computed).
 * Computes: COLOR_DIR — 8-bin direction (0..7) mapped to 0..255.
 * No atan2: uses gx/gy sign ratios for speed. */
__attribute__((unused))
static inline void ki_compute_dir(uint8_t px[COLOR_NB][1024], int w, int h) {
    if (w < 3 || h < 3) return;
    /* Recompute Sobel gx/gy on Y, then quantize direction */
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int i = y * w + x;
            int gx = -(int)px[COLOR_Y][i-w-1] + (int)px[COLOR_Y][i-w+1]
                     -2*(int)px[COLOR_Y][i-1]  + 2*(int)px[COLOR_Y][i+1]
                     -(int)px[COLOR_Y][i+w-1] + (int)px[COLOR_Y][i+w+1];
            int gy = -(int)px[COLOR_Y][i-w-1] -2*(int)px[COLOR_Y][i-w] -(int)px[COLOR_Y][i-w+1]
                     +(int)px[COLOR_Y][i+w-1] +2*(int)px[COLOR_Y][i+w] +(int)px[COLOR_Y][i+w+1];
            /* Quantize to 8 bins based on sign ratios (no atan2) */
            int bin = 0;
            int agx = abs(gx), agy = abs(gy);
            if (agx == 0 && agy == 0) { bin = 0; }
            else if (gx >= 0 && gy >= 0) {
                if (agx > 2*agy) bin = 0; else if (agy > 2*agx) bin = 2; else bin = 1;
            } else if (gx < 0 && gy >= 0) {
                if (agx > 2*agy) bin = 4; else if (agy > 2*agx) bin = 2; else bin = 3;
            } else if (gx < 0 && gy < 0) {
                if (agx > 2*agy) bin = 4; else if (agy > 2*agx) bin = 6; else bin = 5;
            } else {
                if (agx > 2*agy) bin = 0; else if (agy > 2*agx) bin = 6; else bin = 7;
            }
            px[COLOR_DIR][i] = (uint8_t)(bin * 32);  /* map 0..7 → 0..248 */
        }
    }
    /* 1) Left/right borders */
    for (int y = 1; y < h - 1; y++) {
        int ro = y * w;
        uint8_t lv = px[COLOR_DIR][ro + 1];
        uint8_t rv = px[COLOR_DIR][ro + w - 2];
        px[COLOR_DIR][ro + 0]     = lv;
        px[COLOR_DIR][ro + w - 1] = rv;
    }
    /* 2) Top/bottom rows */
    for (int x = 0; x < w; x++) {
        px[COLOR_DIR][0 * w + x]       = px[COLOR_DIR][1 * w + x];
        px[COLOR_DIR][(h - 1) * w + x] = px[COLOR_DIR][(h - 2) * w + x];
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * LOCAL RANGE — max-min in 3×3 patch (texture sharpness)
 * ═══════════════════════════════════════════════════════════════════════
 * Expects: px[COLOR_NB][1024] with valid COLOR_Y (ITU-601 Y).
 * Computes: COLOR_RANGE — max-min, 0..255, no division. */
__attribute__((unused))
static inline void ki_compute_range(uint8_t px[COLOR_NB][1024], int w, int h) {
    if (w < 3 || h < 3) return;
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int i = y * w + x;
            uint8_t mn = 255, mx = 0;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    uint8_t val = px[COLOR_Y][i + dy*w + dx];
                    if (val < mn) mn = val;
                    if (val > mx) mx = val;
                }
            }
            px[COLOR_RANGE][i] = (uint8_t)(mx - mn);
        }
    }
    /* 1) Left/right borders */
    for (int y = 1; y < h - 1; y++) {
        int ro = y * w;
        uint8_t lv = px[COLOR_RANGE][ro + 1];
        uint8_t rv = px[COLOR_RANGE][ro + w - 2];
        px[COLOR_RANGE][ro + 0]     = lv;
        px[COLOR_RANGE][ro + w - 1] = rv;
    }
    /* 2) Top/bottom rows */
    for (int x = 0; x < w; x++) {
        px[COLOR_RANGE][0 * w + x]       = px[COLOR_RANGE][1 * w + x];
        px[COLOR_RANGE][(h - 1) * w + x] = px[COLOR_RANGE][(h - 2) * w + x];
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * COLOR LBP — LBP on RG opponent channel (chromatic texture)
 * ═══════════════════════════════════════════════════════════════════════
 * Expects: px[COLOR_NB][1024] with valid COLOR_RG (R-G opponent).
 * Computes: COLOR_LBP_RG — 8-bit LBP code on color opponent.
 * Captures chromatic textures invisible in luminance LBP. */
__attribute__((unused))
static inline void ki_compute_lbp_rg(uint8_t px[COLOR_NB][1024], int w, int h) {
    if (w < 3 || h < 3) return;
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int i = y * w + x;
            uint8_t center = px[COLOR_RG][i];
            uint8_t code = 0;
            if (px[COLOR_RG][i-w-1] >= center) code |= 0x80;
            if (px[COLOR_RG][i-w  ] >= center) code |= 0x40;
            if (px[COLOR_RG][i-w+1] >= center) code |= 0x20;
            if (px[COLOR_RG][i+1  ] >= center) code |= 0x10;
            if (px[COLOR_RG][i+w+1] >= center) code |= 0x08;
            if (px[COLOR_RG][i+w  ] >= center) code |= 0x04;
            if (px[COLOR_RG][i+w-1] >= center) code |= 0x02;
            if (px[COLOR_RG][i-1  ] >= center) code |= 0x01;
            px[COLOR_LBP_RG][i] = code;
        }
    }
    for (int y = 1; y < h - 1; y++) {
        int ro = y * w;
        uint8_t lv = px[COLOR_LBP_RG][ro + 1];
        uint8_t rv = px[COLOR_LBP_RG][ro + w - 2];
        px[COLOR_LBP_RG][ro + 0]     = lv;
        px[COLOR_LBP_RG][ro + w - 1] = rv;
    }
    for (int x = 0; x < w; x++) {
        px[COLOR_LBP_RG][0 * w + x]       = px[COLOR_LBP_RG][1 * w + x];
        px[COLOR_LBP_RG][(h - 1) * w + x] = px[COLOR_LBP_RG][(h - 2) * w + x];
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * CENTER DISTANCE — positional encoding for spatial context
 * ═══════════════════════════════════════════════════════════════════════
 * Computes: COLOR_DIST — Manhattan distance from center (15.5,15.5).
 * Inverted: center=255, border=0. Static map, same for all images. */
__attribute__((unused))
static inline void ki_compute_dist(uint8_t px[COLOR_NB][1024], int w, int h) {
    int cx = w / 2, cy = h / 2;  /* center pixel */
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int d = abs(x - cx) + abs(y - cy);
            int v = 255 - (d * 255 / (cx + cy));  /* invert: center=255 */
            if (v < 0) v = 0;
            px[COLOR_DIST][y * w + x] = (uint8_t)v;
        }
    }
}

#ifdef __cplusplus
}
#endif

#endif /* KI_ENCODING_H */
