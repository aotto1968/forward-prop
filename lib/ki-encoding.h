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
 *   COLOR_NB                 — number of Blöcke
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
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * ENCODING TYPES
 * ═══════════════════════════════════════════════════════════════════════ */

/* ── Thermometer encoding: popcount(encode(pv)) ∝ brightness ──── */
enum ki_encoding {
    KI_ENC_RAW  = 0,   /* Raw pixel value (0..255), keine Transformation */
    KI_ENC_LIN7 = 1,   /* Uniform, 7 Stufen (pv>>5 → popcount=1..7)  */
    KI_ENC_LIN8 = 2,   /* Uniform, 8 Stufen (pv*8>>8 → popcount=1..8) */
    KI_ENC_DOWN = 3,   /* Bottom-weighted (more resolution in shadows)       */
    KI_ENC_UP   = 4,   /* Top-weighted (more resolution in highlights)        */
    KI_ENC_MID  = 5,   /* Mid-weighted (more resolution in midtones)   */
    KI_ENC_LOG  = 6,   /* Logarithmisch (natural brightness perception) */
    KI_ENC_EXP  = 7,   /* Exponentiell (heavily top-weighted)                */
    KI_ENC_SIG  = 8,   /* S-shaped (Sigmoid, smooth transition)            */
    KI_ENC_COUNT = 9
};

#define KI_ENC_WIDTH_DEFAULT 8

/* ═══════════════════════════════════════════════════════════════════════
 * ENCODING NAMES (for display)
 * ═══════════════════════════════════════════════════════════════════════ */

/* ── Kurzname (ohne Width-Suffix) ──────────────────────────────── */
static inline const char *ki_enc_name_short(int enc) {
    static const char *names[] = {
        [KI_ENC_RAW]  = "raw",
        [KI_ENC_LIN7] = "lin7",
        [KI_ENC_LIN8] = "lin",
        [KI_ENC_DOWN] = "down",
        [KI_ENC_UP]   = "up",
        [KI_ENC_MID]  = "mid",
        [KI_ENC_LOG]  = "log",
        [KI_ENC_EXP]  = "exp",
        [KI_ENC_SIG]  = "sig",
    };
    if (enc >= 0 && enc < KI_ENC_COUNT) return names[enc];
    return "?";
}

/* ═══════════════════════════════════════════════════════════════════════
 * ENCODING PARSER — String → (encoding, width)
 * ═══════════════════════════════════════════════════════════════════════
 * Tokens: "exp16" → returns KI_ENC_EXP, *out_width=16
 *         "exp"   → returns KI_ENC_EXP, *out_width=KI_ENC_WIDTH_DEFAULT
 * Returns -1 on unknown encoding. */

static inline int ki_enc_parse(const char *tok, int *out_width) {
    int tok_len = (int)strlen(tok);
    if (tok_len > 32) return -1;
    static const int widths[] = {32, 16, 8};
    for (int wi = 0; wi < 3; wi++) {
        int w = widths[wi];
        char wstr[4];
        snprintf(wstr, sizeof(wstr), "%d", w);
        int wslen = (int)strlen(wstr);
        if (tok_len > wslen && strcasecmp(tok + tok_len - wslen, wstr) == 0) {
            char prefix[32];
            int plen = tok_len - wslen;
            memcpy(prefix, tok, (size_t)plen);
            prefix[plen] = '\0';
            if (strcasecmp(prefix, "raw")  == 0) { if (out_width) *out_width = w; return KI_ENC_RAW; }
            if (strcasecmp(prefix, "lin7") == 0) { if (out_width) *out_width = w; return KI_ENC_LIN7; }
            if (strcasecmp(prefix, "lin8") == 0 || strcasecmp(prefix, "lin") == 0) { if (out_width) *out_width = w; return KI_ENC_LIN8; }
            if (strcasecmp(prefix, "down") == 0 || strcasecmp(prefix, "unten") == 0) { if (out_width) *out_width = w; return KI_ENC_DOWN; }
            if (strcasecmp(prefix, "up")   == 0 || strcasecmp(prefix, "oben") == 0) { if (out_width) *out_width = w; return KI_ENC_UP; }
            if (strcasecmp(prefix, "mid")  == 0 || strcasecmp(prefix, "mitte") == 0) { if (out_width) *out_width = w; return KI_ENC_MID; }
            if (strcasecmp(prefix, "log")  == 0) { if (out_width) *out_width = w; return KI_ENC_LOG; }
            if (strcasecmp(prefix, "exp")  == 0) { if (out_width) *out_width = w; return KI_ENC_EXP; }
            if (strcasecmp(prefix, "sig")  == 0) { if (out_width) *out_width = w; return KI_ENC_SIG; }
        }
    }
    if (out_width) *out_width = KI_ENC_WIDTH_DEFAULT;
    if (strcasecmp(tok, "raw")  == 0) return KI_ENC_RAW;
    if (strcasecmp(tok, "lin7") == 0) return KI_ENC_LIN7;
    if (strcasecmp(tok, "lin8") == 0 || strcasecmp(tok, "lin") == 0) return KI_ENC_LIN8;
    if (strcasecmp(tok, "down") == 0 || strcasecmp(tok, "unten") == 0) return KI_ENC_DOWN;
    if (strcasecmp(tok, "up")   == 0 || strcasecmp(tok, "oben") == 0) return KI_ENC_UP;
    if (strcasecmp(tok, "mid")  == 0 || strcasecmp(tok, "mitte") == 0) return KI_ENC_MID;
    if (strcasecmp(tok, "log")  == 0 || strcasecmp(tok, "logarithmisch") == 0) return KI_ENC_LOG;
    if (strcasecmp(tok, "exp")  == 0 || strcasecmp(tok, "exponentiell") == 0) return KI_ENC_EXP;
    if (strcasecmp(tok, "sig")  == 0 || strcasecmp(tok, "sigmoid") == 0) return KI_ENC_SIG;
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

#define _KI_ENC_NENC  9
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
 * Alle anderen Blöcke (R,G,B,Y,…) are shifted by 1.
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

    COLOR_EDGE  = 21,  /* Sobel edges on Y luminance (für --channels edge) */
    COLOR_BIN   = 22,  /* Otsu-binarized Y (filled black/white) */

    COLOR_NB    = 23   /* number of Farben */
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
    };
    if ((unsigned)bit < COLOR_NB) return names[bit];
    return "?";
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

    blocks[COLOR_CL] = (uint8_t)((g + b) >> 1);
    blocks[COLOR_CM] = (uint8_t)ki_clamp_u8(128 + (g - b));
    blocks[COLOR_CP] = (uint8_t)ki_clamp_u8(128 + (r - (g + b)/2));
}

/* ═══════════════════════════════════════════════════════════════════════
 * PRINT MEMBER STRUCTURE — unified format für Otto/Hebbian/Adam
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
            printf("=%s%d", ki_enc_name_short((int8_t)types[i]),
                   widths && widths[i] >= 0 ? widths[i] : 8);
        printf(")");
    }
    if (ens > 1) printf("  × EN=%d", ens);
    printf("\n");
}

/* ═══════════════════════════════════════════════════════════════════════
 * EDGE DETECTION — Sobel 3×3 auf Y-Luminanz
 * ═══════════════════════════════════════════════════════════════════════
 * Expects: px[COLOR_NB][1024] mit gültigem COLOR_Y (ITU-601 Y).
 * Computes: COLOR_EDGE (Sobel-Magnitude) + COLOR_C (Sobel auf AL).
 * px is updated IN PLACE. Call AFTER ki_blocks_from_rgb().
 * w=32, h=32 (CIFAR-10 resolution).
 */
__attribute__((unused))
static inline void ki_compute_edge(uint8_t px[COLOR_NB][1024], int w, int h) {
    /* Sobel auf Y (ITU-601) → COLOR_EDGE */
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
        /* Border: copy nearest pixel */
        for (int y = 0; y < h; y++) {
            px[COLOR_EDGE][y*w]     = px[COLOR_EDGE][y*w+1];
            px[COLOR_EDGE][y*w+w-1] = px[COLOR_EDGE][y*w+w-2];
        }
        for (int x = 0; x < w; x++) {
            px[COLOR_EDGE][x]       = px[COLOR_EDGE][w+x];
            px[COLOR_EDGE][(h-1)*w + x] = px[COLOR_EDGE][(h-2)*w + x];
        }
    }
    /* Sobel auf AL (R+G)/2 → COLOR_C (bessere Skalierung) */
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
        /* Border: copy nearest pixel */
        for (int y = 0; y < h; y++) {
            px[COLOR_C][y*w]     = px[COLOR_C][y*w+1];
            px[COLOR_C][y*w+w-1] = px[COLOR_C][y*w+w-2];
        }
        for (int x = 0; x < w; x++) {
            px[COLOR_C][x]       = px[COLOR_C][w+x];
            px[COLOR_C][(h-1)*w + x] = px[COLOR_C][(h-2)*w + x];
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * BINARY THRESHOLD — Otsu auf Y-Luminanz → filled black/white
 * ═══════════════════════════════════════════════════════════════════════
 * Expects: px[COLOR_NB][1024] mit gültigem COLOR_Y (ITU-601 Y).
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

#ifdef __cplusplus
}
#endif

#endif /* KI_ENCODING_H */
