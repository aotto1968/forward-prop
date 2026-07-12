/*
 * mnist-1/mlp-bin32-otto-trn.c — Otto Score Ensemble Trainer
 * =============================================================
 *
 * Extension of Otto Score with ensemble voting.
 * Train N independent W0s and combine via score averaging.
 *
 * --ensembleN N : number of W0s to train in parallel (default 1)
 *
 * Model format version 5:
 *   magic, version=5, mode, ensembleN, H, NC
 *   For each m in 0..ensembleN-1:
 *     W0[m]:    uint32[H * NC]
 *     Target[m]: int32[KI_NCLASSES * H * 32]
 *     Offset[m]: int64[KI_NCLASSES]
 */
#define KI_COMMON_LOAD_INPUT   /* override load_input: color-split for CIFAR */
#include "ki-common.h"
#include "maj3.h"
#include "../lib/ki-encoding.h"
#include <inttypes.h>
#include <sys/stat.h>
#include <errno.h>

/* ═══════════════════════════════════════════════════════════════════════
 * TGT_IDX — Target-Index: [H][V][KI_NCLASSES]
 * V = VN_GROUPS = 32 / splitVN. k-last = contiguous scores[] = 1 cache line. */
#ifndef TGT_IDX
#define TGT_IDX(k, h, v, H, V) \
    ((size_t)(h) * (size_t)(V) * KI_NCLASSES + (size_t)(v) * KI_NCLASSES + (size_t)(k))
#endif

/* ── VN-Dispatch-Makro (generiert switch über splitVN-Werte) ── */
#define VN_DISPATCH(func, G, args...) do {                          \
    switch (G) {                                                    \
        case 1:  func ## _vn1(args); break;                         \
        case 2:  func ## _vn2(args); break;                         \
        case 3:  func ## _vn3(args); break;                         \
        case 4:  func ## _vn4(args); break;                         \
        case 8:  func ## _vn8(args); break;                         \
        case 16: func ## _vn16(args); break;                        \
        case 32: func ## _vn32(args); break;                        \
        default: fprintf(stderr, "[FATAL] invalid --splitVN %d\n", G); exit(1); \
    }                                                               \
} while (0)
#define VN_DISPATCH_R(func, G, args...) ({                          \
    __typeof__(func ## _vn1 args) _r;                               \
    switch (G) {                                                    \
        case 1:  _r = func ## _vn1(args); break;                    \
        case 2:  _r = func ## _vn2(args); break;                    \
        case 3:  _r = func ## _vn3(args); break;                    \
        case 4:  _r = func ## _vn4(args); break;                    \
        case 8:  _r = func ## _vn8(args); break;                    \
        case 16: _r = func ## _vn16(args); break;                   \
        case 32: _r = func ## _vn32(args); break;                   \
        default: fprintf(stderr, "[FATAL] invalid --splitVN %d\n", G); exit(1); \
    }                                                               \
    _r;                                                             \
})

/* ── VN=1: 32 groups of 1 bit (every set bit = active group) ── */
#define VN_SCORE_1(h0, h, H, TGT, SC) do { \
    uint32_t _b = (h0); \
    while (_b) { int _v = __builtin_ctz(_b); \
        for (int _k = 0; _k < KI_NCLASSES; _k++) \
            (SC)[_k] += (TGT)[TGT_IDX(_k, (h), _v, H, 32)]; \
        _b &= _b - 1; } \
} while (0)
#define VN_CORRECT_1(h0, h, H, DC, TK, PK, SI) do { \
    uint32_t _b = (h0); \
    while (_b) { int _v = __builtin_ctz(_b); \
        (DC)[TGT_IDX((TK), (h), _v, H, 32)] += (SI); \
        (DC)[TGT_IDX((PK), (h), _v, H, 32)] -= (SI); \
        _b &= _b - 1; } \
} while (0)

/* ── VN=2: 16 groups of 2 bits, TH=1 (strict: only 11) ──────── */
#define VN_SCORE_2(h0, h, H, TGT, SC) do { \
    uint32_t _gb = 0; \
    _gb |= (__builtin_popcount((h0) & 0x3u) > 1) ? 0x0001u : 0; \
    _gb |= (__builtin_popcount(((h0)>>2) & 0x3u) > 1) ? 0x0002u : 0; \
    _gb |= (__builtin_popcount(((h0)>>4) & 0x3u) > 1) ? 0x0004u : 0; \
    _gb |= (__builtin_popcount(((h0)>>6) & 0x3u) > 1) ? 0x0008u : 0; \
    _gb |= (__builtin_popcount(((h0)>>8) & 0x3u) > 1) ? 0x0010u : 0; \
    _gb |= (__builtin_popcount(((h0)>>10) & 0x3u) > 1) ? 0x0020u : 0; \
    _gb |= (__builtin_popcount(((h0)>>12) & 0x3u) > 1) ? 0x0040u : 0; \
    _gb |= (__builtin_popcount(((h0)>>14) & 0x3u) > 1) ? 0x0080u : 0; \
    _gb |= (__builtin_popcount(((h0)>>16) & 0x3u) > 1) ? 0x0100u : 0; \
    _gb |= (__builtin_popcount(((h0)>>18) & 0x3u) > 1) ? 0x0200u : 0; \
    _gb |= (__builtin_popcount(((h0)>>20) & 0x3u) > 1) ? 0x0400u : 0; \
    _gb |= (__builtin_popcount(((h0)>>22) & 0x3u) > 1) ? 0x0800u : 0; \
    _gb |= (__builtin_popcount(((h0)>>24) & 0x3u) > 1) ? 0x1000u : 0; \
    _gb |= (__builtin_popcount(((h0)>>26) & 0x3u) > 1) ? 0x2000u : 0; \
    _gb |= (__builtin_popcount(((h0)>>28) & 0x3u) > 1) ? 0x4000u : 0; \
    _gb |= (__builtin_popcount(((h0)>>30) & 0x3u) > 1) ? 0x8000u : 0; \
    while (_gb) { int _v = __builtin_ctz(_gb); \
        for (int _k = 0; _k < KI_NCLASSES; _k++) \
            (SC)[_k] += (TGT)[TGT_IDX(_k, (h), _v, H, 16)]; \
        _gb &= _gb - 1; } \
} while (0)
#define VN_CORRECT_2(h0, h, H, DC, TK, PK, SI) do { \
    uint32_t _gb = 0; \
    _gb |= (__builtin_popcount((h0) & 0x3u) > 1) ? 0x0001u : 0; \
    _gb |= (__builtin_popcount(((h0)>>2) & 0x3u) > 1) ? 0x0002u : 0; \
    _gb |= (__builtin_popcount(((h0)>>4) & 0x3u) > 1) ? 0x0004u : 0; \
    _gb |= (__builtin_popcount(((h0)>>6) & 0x3u) > 1) ? 0x0008u : 0; \
    _gb |= (__builtin_popcount(((h0)>>8) & 0x3u) > 1) ? 0x0010u : 0; \
    _gb |= (__builtin_popcount(((h0)>>10) & 0x3u) > 1) ? 0x0020u : 0; \
    _gb |= (__builtin_popcount(((h0)>>12) & 0x3u) > 1) ? 0x0040u : 0; \
    _gb |= (__builtin_popcount(((h0)>>14) & 0x3u) > 1) ? 0x0080u : 0; \
    _gb |= (__builtin_popcount(((h0)>>16) & 0x3u) > 1) ? 0x0100u : 0; \
    _gb |= (__builtin_popcount(((h0)>>18) & 0x3u) > 1) ? 0x0200u : 0; \
    _gb |= (__builtin_popcount(((h0)>>20) & 0x3u) > 1) ? 0x0400u : 0; \
    _gb |= (__builtin_popcount(((h0)>>22) & 0x3u) > 1) ? 0x0800u : 0; \
    _gb |= (__builtin_popcount(((h0)>>24) & 0x3u) > 1) ? 0x1000u : 0; \
    _gb |= (__builtin_popcount(((h0)>>26) & 0x3u) > 1) ? 0x2000u : 0; \
    _gb |= (__builtin_popcount(((h0)>>28) & 0x3u) > 1) ? 0x4000u : 0; \
    _gb |= (__builtin_popcount(((h0)>>30) & 0x3u) > 1) ? 0x8000u : 0; \
    while (_gb) { int _v = __builtin_ctz(_gb); \
        (DC)[TGT_IDX((TK), (h), _v, H, 16)] += (SI); \
        (DC)[TGT_IDX((PK), (h), _v, H, 16)] -= (SI); \
        _gb &= _gb - 1; } \
} while (0)

/* ── VN=3: 10 groups of 3 bits, TH=2 (strict: only 111) ──────── */
#define VN_SCORE_3(h0, h, H, TGT, SC) do { \
    uint32_t _gb = 0; \
    _gb |= (__builtin_popcount((h0) & 0x7u) > 2) ? 1u<<0 : 0; \
    _gb |= (__builtin_popcount(((h0)>>3) & 0x7u) > 2) ? 1u<<1 : 0; \
    _gb |= (__builtin_popcount(((h0)>>6) & 0x7u) > 2) ? 1u<<2 : 0; \
    _gb |= (__builtin_popcount(((h0)>>9) & 0x7u) > 2) ? 1u<<3 : 0; \
    _gb |= (__builtin_popcount(((h0)>>12) & 0x7u) > 2) ? 1u<<4 : 0; \
    _gb |= (__builtin_popcount(((h0)>>15) & 0x7u) > 2) ? 1u<<5 : 0; \
    _gb |= (__builtin_popcount(((h0)>>18) & 0x7u) > 2) ? 1u<<6 : 0; \
    _gb |= (__builtin_popcount(((h0)>>21) & 0x7u) > 2) ? 1u<<7 : 0; \
    _gb |= (__builtin_popcount(((h0)>>24) & 0x7u) > 2) ? 1u<<8 : 0; \
    _gb |= (__builtin_popcount(((h0)>>27) & 0x7u) > 2) ? 1u<<9 : 0; \
    while (_gb) { int _v = __builtin_ctz(_gb); \
        for (int _k = 0; _k < KI_NCLASSES; _k++) \
            (SC)[_k] += (TGT)[TGT_IDX(_k, (h), _v, H, 10)]; \
        _gb &= _gb - 1; } \
} while (0)
#define VN_CORRECT_3(h0, h, H, DC, TK, PK, SI) do { \
    uint32_t _gb = 0; \
    _gb |= (__builtin_popcount((h0) & 0x7u) > 2) ? 1u<<0 : 0; \
    _gb |= (__builtin_popcount(((h0)>>3) & 0x7u) > 2) ? 1u<<1 : 0; \
    _gb |= (__builtin_popcount(((h0)>>6) & 0x7u) > 2) ? 1u<<2 : 0; \
    _gb |= (__builtin_popcount(((h0)>>9) & 0x7u) > 2) ? 1u<<3 : 0; \
    _gb |= (__builtin_popcount(((h0)>>12) & 0x7u) > 2) ? 1u<<4 : 0; \
    _gb |= (__builtin_popcount(((h0)>>15) & 0x7u) > 2) ? 1u<<5 : 0; \
    _gb |= (__builtin_popcount(((h0)>>18) & 0x7u) > 2) ? 1u<<6 : 0; \
    _gb |= (__builtin_popcount(((h0)>>21) & 0x7u) > 2) ? 1u<<7 : 0; \
    _gb |= (__builtin_popcount(((h0)>>24) & 0x7u) > 2) ? 1u<<8 : 0; \
    _gb |= (__builtin_popcount(((h0)>>27) & 0x7u) > 2) ? 1u<<9 : 0; \
    while (_gb) { int _v = __builtin_ctz(_gb); \
        (DC)[TGT_IDX((TK), (h), _v, H, 10)] += (SI); \
        (DC)[TGT_IDX((PK), (h), _v, H, 10)] -= (SI); \
        _gb &= _gb - 1; } \
} while (0)

/* ── VN=4: 8 groups of 4 bits, TH=3 (strict: only 1111) ──────── */
#define VN_SCORE_4(h0, h, H, TGT, SC) do { \
    uint32_t _gb = 0; \
    _gb |= (__builtin_popcount((h0) & 0xFu) > 3) ? 1u<<0 : 0; \
    _gb |= (__builtin_popcount(((h0)>>4) & 0xFu) > 3) ? 1u<<1 : 0; \
    _gb |= (__builtin_popcount(((h0)>>8) & 0xFu) > 3) ? 1u<<2 : 0; \
    _gb |= (__builtin_popcount(((h0)>>12) & 0xFu) > 3) ? 1u<<3 : 0; \
    _gb |= (__builtin_popcount(((h0)>>16) & 0xFu) > 3) ? 1u<<4 : 0; \
    _gb |= (__builtin_popcount(((h0)>>20) & 0xFu) > 3) ? 1u<<5 : 0; \
    _gb |= (__builtin_popcount(((h0)>>24) & 0xFu) > 3) ? 1u<<6 : 0; \
    _gb |= (__builtin_popcount(((h0)>>28) & 0xFu) > 3) ? 1u<<7 : 0; \
    while (_gb) { int _v = __builtin_ctz(_gb); \
        for (int _k = 0; _k < KI_NCLASSES; _k++) \
            (SC)[_k] += (TGT)[TGT_IDX(_k, (h), _v, H, 8)]; \
        _gb &= _gb - 1; } \
} while (0)
#define VN_CORRECT_4(h0, h, H, DC, TK, PK, SI) do { \
    uint32_t _gb = 0; \
    _gb |= (__builtin_popcount((h0) & 0xFu) > 3) ? 1u<<0 : 0; \
    _gb |= (__builtin_popcount(((h0)>>4) & 0xFu) > 3) ? 1u<<1 : 0; \
    _gb |= (__builtin_popcount(((h0)>>8) & 0xFu) > 3) ? 1u<<2 : 0; \
    _gb |= (__builtin_popcount(((h0)>>12) & 0xFu) > 3) ? 1u<<3 : 0; \
    _gb |= (__builtin_popcount(((h0)>>16) & 0xFu) > 3) ? 1u<<4 : 0; \
    _gb |= (__builtin_popcount(((h0)>>20) & 0xFu) > 3) ? 1u<<5 : 0; \
    _gb |= (__builtin_popcount(((h0)>>24) & 0xFu) > 3) ? 1u<<6 : 0; \
    _gb |= (__builtin_popcount(((h0)>>28) & 0xFu) > 3) ? 1u<<7 : 0; \
    while (_gb) { int _v = __builtin_ctz(_gb); \
        (DC)[TGT_IDX((TK), (h), _v, H, 8)] += (SI); \
        (DC)[TGT_IDX((PK), (h), _v, H, 8)] -= (SI); \
        _gb &= _gb - 1; } \
} while (0)

/* ── VN=8: 4 groups of 8 bits, TH=7 (strict: only 0xFF) ──────── */
#define VN_SCORE_8(h0, h, H, TGT, SC) do { \
    uint32_t _gb = 0; \
    _gb |= (__builtin_popcount((h0) & 0xFFu) > 7) ? 1u<<0 : 0; \
    _gb |= (__builtin_popcount(((h0)>>8) & 0xFFu) > 7) ? 1u<<1 : 0; \
    _gb |= (__builtin_popcount(((h0)>>16) & 0xFFu) > 7) ? 1u<<2 : 0; \
    _gb |= (__builtin_popcount(((h0)>>24) & 0xFFu) > 7) ? 1u<<3 : 0; \
    while (_gb) { int _v = __builtin_ctz(_gb); \
        for (int _k = 0; _k < KI_NCLASSES; _k++) \
            (SC)[_k] += (TGT)[TGT_IDX(_k, (h), _v, H, 4)]; \
        _gb &= _gb - 1; } \
} while (0)
#define VN_CORRECT_8(h0, h, H, DC, TK, PK, SI) do { \
    uint32_t _gb = 0; \
    _gb |= (__builtin_popcount((h0) & 0xFFu) > 7) ? 1u<<0 : 0; \
    _gb |= (__builtin_popcount(((h0)>>8) & 0xFFu) > 7) ? 1u<<1 : 0; \
    _gb |= (__builtin_popcount(((h0)>>16) & 0xFFu) > 7) ? 1u<<2 : 0; \
    _gb |= (__builtin_popcount(((h0)>>24) & 0xFFu) > 7) ? 1u<<3 : 0; \
    while (_gb) { int _v = __builtin_ctz(_gb); \
        (DC)[TGT_IDX((TK), (h), _v, H, 4)] += (SI); \
        (DC)[TGT_IDX((PK), (h), _v, H, 4)] -= (SI); \
        _gb &= _gb - 1; } \
} while (0)

/* ── VN=16: 2 groups of 16 bits, TH=15 (strict: only 0xFFFF) ─── */
#define VN_SCORE_16(h0, h, H, TGT, SC) do { \
    uint32_t _gb = 0; \
    if (__builtin_popcount((h0) & 0xFFFFu) > 15) _gb |= 1u<<0; \
    if (__builtin_popcount((h0) >> 16) > 15) _gb |= 1u<<1; \
    while (_gb) { int _v = __builtin_ctz(_gb); \
        for (int _k = 0; _k < KI_NCLASSES; _k++) \
            (SC)[_k] += (TGT)[TGT_IDX(_k, (h), _v, H, 2)]; \
        _gb &= _gb - 1; } \
} while (0)
#define VN_CORRECT_16(h0, h, H, DC, TK, PK, SI) do { \
    uint32_t _gb = 0; \
    if (__builtin_popcount((h0) & 0xFFFFu) > 15) _gb |= 1u<<0; \
    if (__builtin_popcount((h0) >> 16) > 15) _gb |= 1u<<1; \
    while (_gb) { int _v = __builtin_ctz(_gb); \
        (DC)[TGT_IDX((TK), (h), _v, H, 2)] += (SI); \
        (DC)[TGT_IDX((PK), (h), _v, H, 2)] -= (SI); \
        _gb &= _gb - 1; } \
} while (0)

/* ── VN=32: 1 group of 32 bits, TH=31 (strict: only 0xFFFFFFFF) ─ */
#define VN_SCORE_32(h0, h, H, TGT, SC) do { \
    if (__builtin_popcount(h0) > 31) { \
        for (int _k = 0; _k < KI_NCLASSES; _k++) \
            (SC)[_k] += (TGT)[TGT_IDX(_k, (h), 0, H, 1)]; \
    } \
} while (0)
#define VN_CORRECT_32(h0, h, H, DC, TK, PK, SI) do { \
    if (__builtin_popcount(h0) > 31) { \
        (DC)[TGT_IDX((TK), (h), 0, H, 1)] += (SI); \
        (DC)[TGT_IDX((PK), (h), 0, H, 1)] -= (SI); \
    } \
} while (0)

/* ── Forward declaration für ki_Member (Struct-Definition folgt
 * weiter unten, ki_evaluate_member steht davor) ───────────── */
typedef struct ki_Member ki_Member;

/* ── Global args (initialisiert in main) ────────────────────── */
ki_Args aa = {
    .hidden             = 64,
    .epochs             = 1,
    .batchN             = KI_DEFAULT_BATCH_N,
    .trainN             = 0,      /* auto: set from dataset default */
    .evalN              = 0,      /* auto: set from dataset default */
    .seed               = 42,
    .lr                 = KI_DEFAULT_LR,
    .threadN            = 8,
    .warmup_epochs      = 2,
    .step_power         = KI_DEFAULT_STEP_POWER,
    .target_err         = 0.0f,
    .step_mode          = KI_DEFAULT_STEP_MODE,
    .ensembleN          = 1,
    .splitVN            = 1,
    .splitHN            = 1,
    .channel            = KI_DEFAULT_COLOR,/* CIFAR: r+g+b, MNIST: nur Block 0 */
    .packedB            = 1,
    .enc_default_type   = -1,    /* -1 = auto: falls bin→KI_ENC_LIN7, sonst KI_ENC_RAW */
    .enc_default_width  = KI_ENC_WIDTH_DEFAULT,
    .enc_count          = 0,     /* 0 = kein enc_array (legacy single) */
    .opt_target_norm    = KI_DEFAULT_TARGET_NORM,
    .ensemble_seed      = ENS_SEED_ONCE,
    .multi_correct      = 0,
    .seed_splitmix      = 1,
};

/* ═══════════════════════════════════════════════════════════════════════
 * CUSTOM load_input — 7-channel input buffer for CIFAR (KI_COLORS=3),
 * passthrough for MNIST (KI_COLORS=1, KI_PACK=4)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * ── INPUT-BUFFER LAYOUT (CIFAR, linear, 1 Bild = 7 × KI_NC uint32) ──
 *
 *   Block | Bit | Name | Formel (pro Pixel)         | Mapping auf 0..255
 *   ------|-----|------|----------------------------|-------------------
 *     0   |  0  |  R   | roter Rohpixel             | r
 *     1   |  1  |  G   | grüner Rohpixel            | g
 *     2   |  2  |  B   | blauer Rohpixel            | b
 *     3   |  3  |  Y   | ITU-R BT.601               | (r*77+g*150+b*29)>>8
 *     4   |  4  |  LUM | R+G Luminanz               | (r+g)>>1
 *     5   |  5  |  RG  | R-G Rot-Grün Opponent      | (r-g+255)>>1
 *     6   |  6  |  BY  | B-(R+G)/2 Blau-Gelb Opp.   | (2b-r-g+510)>>2
 *
 *   Buffer: [R(256)][G(256)][B(256)][Y(256)][LUM(256)][RG(256)][BY(256)][YL(256)]
 *           ↑0      ↑256   ↑512   ↑768   ↑1024    ↑1280    ↑1536    ↑1792
 *   n_cont = 8×256 = 2048 (FIXED, immer alle Blöcke)
 *
 *   Block | Bit | Name | Formel
 *   ------|-----|------|----------------------------
 *     0   |  0  |  R   | r
 *     1   |  1  |  G   | g
 *     2   |  2  |  B   | b
 *     3   |  3  |  Y   | (r*77+g*150+b*29)>>8   (ITU-R BT.601)
 *     4   |  4  |  LUM | (r+g)>>1
 *     5   |  5  |  RG  | (r-g+255)>>1
 *     6   |  6  |  BY  | (2b-r-g+510)>>2
 *     7   |  7  |  YL  | (r*54+g*183+b*18)>>8   (ITU-R BT.709)
 *
 * ── MEMBER-INDIZIERUNG ──────────────────────────────────────────────
 *   active_chans[] = 1:1 Mapping aus Bitmaske (b→Block b).
 *   Für jeden Member m:
 *     seq_chan  = (m / splitHN) % eff_colors
 *     block     = active_chans[seq_chan]            // 0..7
 *     h_idx     = m % splitHN
 *     nc_off    = block * KI_NC + h_idx * NC_slice
 */
/* Anzahl Container-Blöcke = Anzahl Farben (COLOR_NB, dynamisch) */
#define KI_NB COLOR_NB

static __attribute__((unused)) uint32_t *load_input(const uint8_t *X_raw,
                                                     int n_samples) {
    (void)n_samples;
#if KI_COLORS > 1
    /* ── CIFAR: immer über enc_array (nach Expansion haben alle Einträge color>=0) ── */
    int n_enc = aa.enc_count;
    int enc_off[KI_ENC_MAX], enc_nc[KI_ENC_MAX];
    size_t stride = 0;
    for (int i = 0; i < n_enc && i < KI_ENC_MAX; i++) {
        int w = (int)aa.enc_array[i].width;
        if (w < 1) w = KI_ENC_WIDTH_DEFAULT;
        enc_nc[i] = KI_NC * w / 8;
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
        /* Sobel-Kanten auf Y (ITU-601) + LUM: 3×3-Gradienten-Magnitude */
        ki_compute_edge(px, 32, 32);
        /* Otsu-Binarisierung auf Y → filled black/white */
        ki_compute_binary(px, 32, 32);
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
    /* ── MNIST: immer über enc_array ── */
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
        enc_nc[i] = KI_NC * w / 8; enc_type[i] = et;
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
                    size_t p = (size_t)s * (size_t)KI_PX + (size_t)c * (size_t)enc_pack[i] + (size_t)k;
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

/* ── Konstanten ────────────────────────────────────────────────── */
#ifndef NC
#define NC        196     /* Default MNIST — override via -DNC or ki-local.h */
#endif

/* Globale Channel-Parameter (werden nach --channels in main() gesetzt) */
static int eff_colors = 3;              /* Anzahl aktiver Channels / Mitglieder (popcount mask) */
static int active_chans[KI_ENC_MAX];    /* Mapping: seq_idx → Bit-Position (0..8) */
#define BITS       32
#define N_CLASSES KI_NCLASSES

/* Export file magic + version */
#define OTTO_MAGIC   0x4F54544FU   /* "OTTO" */
#define OTTO_VERSION 5U            /* v5 = ensemble (no precision field) */
#define OTTO_VERSION_V6 6U         /* v6 = ensemble + precision field */

/* Index: [KI_NCLASSES][H][32] — klasse × neuron × bit */


// === Zentraler Skalierungsfaktor OT_PRECISION ===
// Alle logit/log(p)-Werte werden mit F = (1<<OT_PRECISION) skaliert
// in int32/int64 gespeichert.  Der Korrektur-Step und die lr-Anzeige
// leiten sich daraus ab → eine Änderung wirkt auf alle abhängigen Stellen.
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#ifndef OT_PRECISION
#endif

// #pragma message(">>> OT_PRECISION = " TOSTRING(OT_PRECISION))


/* ═══════════════════════════════════════════════════════════════════
 * H0 MODE — XNOR (default) or XOR (via -DH0_XOR)
 * ═══════════════════════════════════════════════════════════════════ */
#ifdef H0_XOR
#  define H0_STR     "XOR"
#  define H0_MATCH(in, W0_row, c)  ((in)[c] ^ (W0_row)[c])
#  define H0_MODE_VAL 1U
#else
#  define H0_STR     "XNOR"
#  define H0_MATCH(in, W0_row, c)  (~((in)[c] ^ (W0_row)[c]))
#  define H0_MODE_VAL 0U
#endif


/* ═══════════════════════════════════════════════════════════════════
 * H0 FORWARD — MAJ3 über nc_local Container
 * ═══════════════════════════════════════════════════════════════════
 * in_offset: Start des Slices im Input-Array
 * nc_local:  Anzahl Container für diesen Member
 */
static uint32_t h0_neuron(const uint32_t *in, const uint32_t *W0_row, int nc_local) {
    uint32_t match[MAJ3_BUF]; /* aus maj3.h, ≥ MAJ3_BUF */
    for (int c = 0; c < nc_local; c++)
        match[c] = H0_MATCH(in, W0_row, c);
    return majority_tree(match, nc_local);
}


/* ═══════════════════════════════════════════════════════════════════
 * TARGET BUILD — count class-k VN firings per (h,v)
 * ═══════════════════════════════════════════════════════════════════
 * Returns raw counts. Caller must run logit_convert + class_offset
 * + logit_convert to produce the final target matrix.
 *
 * W0:     pointer to start of this member's W0 [H_local × NC_slice]
 * H_local: neurons for this member
 * NC_slice: containers for this member (NC / splitHN)
 * nc_off:  container offset in input (member_idx × NC_slice)
 */
static int32_t *ki_build_target(const uint32_t *X, const uint8_t *Y, int N,
                               const uint32_t *W0, int H_local, int NC_slice,
                               int nc_off, int stride, int silent) {
    int V = VN_GROUPS_, G = aa.splitVN, TH = VN_THRESH_;
    size_t sz = (size_t)H_local * KI_NCLASSES * (size_t)V;
    int32_t *target = (int32_t *)ki_xcalloc(sz, sizeof(int32_t));

    if (!silent) {
        printf("\n=== OTTO TARGET ===\n");
        printf("  Target[%d][%d][%d] = %zu KB\n",
               KI_NCLASSES, H_local, V, sz * sizeof(int32_t) / 1024);
        fflush(stdout);
    }

    #pragma omp parallel
    {
        int32_t *lt = (int32_t *)ki_xcalloc(sz, sizeof(int32_t));
        #pragma omp for schedule(static)
        for (int s = 0; s < N; s++) {
            int k = (int)Y[s];
            const uint32_t *in = X + (size_t)s * (size_t)stride + nc_off;
            for (int h = 0; h < H_local; h++) {
                uint32_t h0 = h0_neuron(in, W0 + (size_t)h * NC_slice, NC_slice);
                uint32_t gbits;
                if (G == 1) {
                    gbits = h0;
                } else {
                    gbits = 0;
                    for (int v = 0; v < V; v++) {
                        uint32_t slice = (h0 >> (v * G)) & ((1u << G) - 1u);
                        if (__builtin_popcount(slice) > TH) gbits |= (1u << v);
                    }
                }
                while (gbits) {
                    int v = __builtin_ctz(gbits);
                    lt[TGT_IDX(k, h, v, H_local, V)]++;
                    gbits &= gbits - 1;
                }
            }
        }
        #pragma omp critical
        { for (size_t i = 0; i < sz; i++) target[i] += lt[i]; }
        free(lt);
    }
    return target;
}

/* ── ki_build_target_from_gb: Target-Zählung aus gb_buf (KEIN h0_neuron) ── *
 * Nutzt vorberechnete gb_buf-Daten statt h0_neuron + VN-Reduktion.
 * Doppelt so schnell wie ki_build_target, weil h0_neuron UND die
 * VN-Gruppen-Berechnung entfallen (beides ist bereits in gb_buf). */
static int32_t *ki_build_target_from_gb(const uint8_t *Y, int N,
    const uint32_t *gb_buf, int H_local, int V) {
    size_t sz = (size_t)H_local * KI_NCLASSES * (size_t)V;
    int32_t *target = (int32_t *)ki_xcalloc(sz, sizeof(int32_t));
    #pragma omp parallel
    {
        int32_t *lt = (int32_t *)ki_xcalloc(sz, sizeof(int32_t));
        #pragma omp for schedule(static)
        for (int s = 0; s < N; s++) {
            int k = (int)Y[s];
            const uint32_t *gb_row = gb_buf + (size_t)s * (size_t)H_local;
            for (int h = 0; h < H_local; h++) {
                uint32_t gbits = gb_row[h];
                while (gbits) {
                    int v = __builtin_ctz(gbits);
                    lt[TGT_IDX(k, h, v, H_local, V)]++;
                    gbits &= gbits - 1;
                }
            }
        }
        #pragma omp critical
        { for (size_t i = 0; i < sz; i++) target[i] += lt[i]; }
        free(lt);
    }
    return target;
}


/* ═══════════════════════════════════════════════════════════════════
 * LOGIT CONVERT — Target-Counts → log-odds (in-place)
 * ═══════════════════════════════════════════════════════════════════
 *   target[k][h][b] = round( ln((t+1)/(N_k-t+1)) × F )
 *   mit F = (1<<OT_PRECISION) (default 10 → F=1024)
 *
 * Abhängigkeit: ot_precision() definiert die int32-Skalierung.
 */
static void logit_convert(int32_t *target, int H_local, const int class_counts[KI_NCLASSES]) {
    int V = VN_GROUPS_;
    for (int k = 0; k < KI_NCLASSES; k++) {
        int nk = class_counts[k];
        if (nk <= 0) continue;
        for (int h = 0; h < H_local; h++) {
            for (int v = 0; v < V; v++) {
                size_t idx = TGT_IDX(k, h, v, H_local, V);
                int t = target[idx];
                double p = (double)(t + 1) / (double)(nk + 2);
                target[idx] = (int32_t)ot_precision(log(p / (1.0 - p)));
            }
        }
    }
}

/* ── VN_SCORE_FROM_GB / VN_CORRECT_FROM_GB ───────────────────── *
 * Nutzen vorberechnete gb-Maske statt h0 → popcount-Reduktion.     */
#define VN_SCORE_FROM_GB(gb, h, H, NG, TGT, SC) do { \
    uint32_t _b = (gb); \
    while (_b) { int _v = __builtin_ctz(_b); \
        for (int _k = 0; _k < KI_NCLASSES; _k++) \
            (SC)[_k] += (TGT)[TGT_IDX(_k, (h), _v, H, NG)]; \
        _b &= _b - 1; } \
} while (0)

#define VN_CORRECT_FROM_GB(gb, h, H, NG, DC, TK, PK, SI) do { \
    uint32_t _b = (gb); \
    while (_b) { int _v = __builtin_ctz(_b); \
        (DC)[TGT_IDX((TK), (h), _v, H, NG)] += (SI); \
        (DC)[TGT_IDX((PK), (h), _v, H, NG)] -= (SI); \
        _b &= _b - 1; } \
} while (0)

/* ═══════════════════════════════════════════════════════════════════
 * CLASS OFFSET — Σ log(1-P_k) × F  (F = 1<<OT_PRECISION)
 * ═══════════════════════════════════════════════════════════════════
 * Muss VOR logit_convert berechnet werden (braucht raw counts).
 * Gemeinsame Skalierung: target und offset teilen F.
 *
 * Note: +-0.5 rounding removed because log(p1)*F is always negative
 * and the 0.5 correction (~0.001%) is below measurement noise (±0.3pp).
 * See: 2026-06-18 experiment — 3 runs with/without 0.5 gave identical
 * results within normal OpenMP scheduling noise.
 */
static void compute_class_offset(int64_t class_offset[KI_NCLASSES],
                                  const int32_t *target, int H_local,
                                  const int class_counts[KI_NCLASSES]) {
    int V = VN_GROUPS_;
    for (int k = 0; k < KI_NCLASSES; k++) {
        int64_t sum = 0;
        int nk = class_counts[k];
        if (nk <= 0) { class_offset[k] = 0; continue; }
        for (int h = 0; h < H_local; h++) {
            for (int v = 0; v < V; v++) {
                int t = target[TGT_IDX(k, h, v, H_local, V)];
                double p1 = (double)(nk - t + 1) / (double)(nk + 2);
                sum += (int64_t)ot_precision(log(p1));
            }
        }
        class_offset[k] = sum;
    }
}
/* ═══════════════════════════════════════════════════════════════════
 * SCORE — Bayes log-Score (mit Slice)
 * in:    Input + nc_off (zum Member-Slice verschoben)
 * W0:    Member-W0 [H_local × NC_slice]
 * H_local, NC_slice, nc_off: Slice-Parameter
 */
static void scores_otto(const uint32_t *in, const uint32_t *W0,
                         int H_local, int NC_slice,
                         const int32_t *target,
                         const int64_t class_offset[KI_NCLASSES],
                         int64_t scores[KI_NCLASSES]) {
    for (int k = 0; k < KI_NCLASSES; k++)
        scores[k] = class_offset[k];

    for (int h = 0; h < H_local; h++) {
        uint32_t h0 = h0_neuron(in, W0 + (size_t)h * NC_slice, NC_slice);
        /* VN-grouped: compile-time-optimierte Makros */
        switch (aa.splitVN) {
            case 1:  VN_SCORE_1(h0, h, H_local, target, scores); break;
            case 2:  VN_SCORE_2(h0, h, H_local, target, scores); break;
            case 3:  VN_SCORE_3(h0, h, H_local, target, scores); break;
            case 4:  VN_SCORE_4(h0, h, H_local, target, scores); break;
            case 8:  VN_SCORE_8(h0, h, H_local, target, scores); break;
            case 16: VN_SCORE_16(h0, h, H_local, target, scores); break;
            case 32: VN_SCORE_32(h0, h, H_local, target, scores); break;
        }
    }
}

/* ── scores_otto aus vorberechneter VN-Gruppenmaske (KEIN h0_neuron) ── *
 * Wird für Training-Eval verwendet (gb_buf ist 1× berechnet).
 * Nimmt Einzelfelder statt ki_Member* (Struct-Definition kommt später).
 * Für Test-Eval ohne Cache gibt es scores_otto (oben). */
static void scores_otto_from_gb(int s, int H_local,
                                 const uint32_t *gb_buf,
                                 const int32_t *target,
                                 const int64_t class_offset[KI_NCLASSES],
                                 int64_t scores[KI_NCLASSES]) {
    const uint32_t *gb_row = gb_buf + (size_t)s * (size_t)H_local;
    for (int k = 0; k < KI_NCLASSES; k++)
        scores[k] = class_offset[k];

    for (int h = 0; h < H_local; h++) {
        switch (aa.splitVN) {
            case 1:  VN_SCORE_FROM_GB(gb_row[h], h, H_local, 32, target, scores); break;
            case 2:  VN_SCORE_FROM_GB(gb_row[h], h, H_local, 16, target, scores); break;
            case 3:  VN_SCORE_FROM_GB(gb_row[h], h, H_local, 10, target, scores); break;
            case 4:  VN_SCORE_FROM_GB(gb_row[h], h, H_local,  8, target, scores); break;
            case 8:  VN_SCORE_FROM_GB(gb_row[h], h, H_local,  4, target, scores); break;
            case 16: VN_SCORE_FROM_GB(gb_row[h], h, H_local,  2, target, scores); break;
            case 32: VN_SCORE_FROM_GB(gb_row[h], h, H_local,  1, target, scores); break;
        }
    }
}


/* ═══════════════════════════════════════════════════════════════════
 * EVALUATE — Members außen, Samples innen (cache-optimal)
 * ═══════════════════════════════════════════════════════════════════
 *
 * Members seriell außen → target[m] (80 KB) bleibt im D1-Cache für
 * alle N Samples. Reduziert D1mr von 55% auf <1% (vorher: samples
 * außen → jeder member-Wechsel evicted target aus dem Cache).
 *
 * Nutzt direkt die ki_Member-Structs (keine flachen Arrays mehr).
 * Jeder Member hat eigene W0, target, offset, slc_off.
 *
 * Votes-Zwischenspeicher: N × KI_NCLASSES × 8 Byte (4 MB für 50000 Samples).
 * n_cont:   Container pro Sample (NC, für Stride)
 * Returns:  Anzahl korrekt klassifizierte Samples
 */

/* (body moved after struct ki_Member definition, siehe forward decl) */


/* ═══════════════════════════════════════════════════════════════════
 * EXPORT — W0 + Target + Offset als model.otto
 * ═══════════════════════════════════════════════════════════════════
 *
 * Format (Einzeldatei):
 *   Header: 20 Byte
 *     uint32 magic      = 0x4F544F54 ('OTTO')
 *     uint32 version    = 1
 *     uint32 h0_mode    = 0 (XNOR) / 1 (XOR)
 *     uint32 H          = hidden neurons
 *     uint32 NC         = containers per image
 *   W0:      uint32[H * NC]
 *   Target:  int32[KI_NCLASSES * H * 32]
 *   Offset:  int64[KI_NCLASSES]
 */
static void export_model(const char *out_dir,
                          const uint32_t *W0, int H,
                          const int32_t *target,
                          const int64_t class_offset[KI_NCLASSES]) {
    /* Not used in ensemble mode — use export_ensemble instead */
    (void)W0; (void)target; (void)class_offset;
    fprintf(stderr, "[ERROR] Use export_ensemble with --ensembleN (independent copies)\n");
}


/* ── Export ensemble: all N models into one file ──────────────── */

/* ═══════════════════════════════════════════════════════════════════
 * SAVE SCORES ARCHIVE — per-member scores for merge-ensemble
 * ═══════════════════════════════════════════════════════════════════
 *
 * Called after training.  Computes scores for each member on the
 * eval set and writes them to a single archive file.
 *
 * Archive naming: H{hidden}_EP{epochs}_VN{splitVN}_HN{splitHN}_TE{te}_SD{seed}.ens
 *   where te = (int)(target_err * 100 + 0.5f)
 *
 * Archive format:
 *   Header: magic(4) ver=4(4) n_test(4) n_classes(4) n_members(4)
 *           hidden(4) epochs(4) split_vn(4) split_hn(4)
 *           target_err(4) seed(4) timestamp(8) ensemble_eval(4)
 *   Per-member metadata:
 *   Data:   n_members . int64[n_test . n_classes]
 *           uint8[n_test]  (ground truth labels)
 *
 * Requires ki_Member to be fully defined (call after struct).
 * Returns 0 on success, -1 on error.
 */


/* ═══════════════════════════════════════════════════════════════════
 * SETUP
 * ═══════════════════════════════════════════════════════════════════ */
static void print_setup(int H, int epochs, int trainN, int evalN,
                          int threadN, unsigned int seed, int batchN,
                           int splitVN, int splitHN, int NC_slice, int H_local,
                           int ensembleN, int channel, int nc_per_blk, int nc_total) {
    int V = 32 / splitVN;                /* virtual neurons per container */
    size_t bit_per_cont = 32;
    size_t hidden_bit = (size_t)H * bit_per_cont;
    size_t w0_bit = (size_t)H_local * (size_t)NC_slice * bit_per_cont;
    size_t w1_bit = (size_t)KI_NCLASSES * (size_t)H_local * (size_t)V * sizeof(int32_t) * 8;
    int total_slots = ensembleN * eff_colors * splitHN;    /* VN no longer multiplies members */
    size_t tgt_total = (size_t)H_local * KI_NCLASSES * (size_t)V * (size_t)total_slots;
    printf("══════════════════════════════════════════════════════════════════════\n");
    printf("══╡ OTTO-SCORE ╞══  %s\n", H0_STR);
    printf("  Args:        H=%d  B=%d  Ep=%d  NC=%d V=%d  HN=%d  H_sub=%d  NC_sub=%d\n", 
            H, batchN, epochs, nc_per_blk, V, splitHN, H_local, NC_slice);
    printf("\n");

    printf("══╡ SETUP ╞══════════════════════════════════════════════════════════\n");
    int disp_cols = 0;  /* actual number of selected blocks (for display) */
    for (int _b = 0; _b < COLOR_NB; _b++)
        if (aa.channel & (1 << _b)) disp_cols++;
    if (disp_cols < 1) disp_cols = 1;
    printf("  Input:       %d px → %d/%d blocks (%s) × %d = %d total  (channels)\n",
           KI_PX, disp_cols, KI_NB, color_str(), nc_per_blk, nc_total);

    printf("  ───────────────────────────────────────────────────────────\n");
    printf("  Input H0:    %-4d nrn x %2zu bit  = %7zu bit  (%5.1f KB)\n",
           H, bit_per_cont, hidden_bit, (double)hidden_bit / 8 / 1024);
    printf("  Output:      %-4d nrn x %2d bit  = %7zu bit  (%5.1f KB)\n",
           KI_NCLASSES, 32, (size_t)KI_NCLASSES * 32, (double)(KI_NCLASSES * 32) / 8 / 1024);

    printf("  ───────────────────────────────────────────────────────────\n");
    printf("  W0:          H0[%3d] x I0[%3d] / HN[%3d] x bin%2zu = %9zu bit  (%5.1f KB)  per member, frozen\n",
           H_local, NC_slice, splitHN, bit_per_cont,
           w0_bit, (double)w0_bit / 8 / 1024);
    printf("  W1:          C1[%3d] × H0[%3d] ×  V[%3d] x int32 = %9zu bit  (%5.1f KB)  per member, target+offset\n",
           KI_NCLASSES, H_local, V, w1_bit, (double)w1_bit / 8 / 1024);
    printf("  ───────────────────────────────────────────────────────────\n");
    printf("  TOTAL:       (W0+W1) x (EN[%d]×CO[%d]×HN[%d]=%d) = %9zu bit  (%5.1f KB)\n",
           ensembleN, eff_colors, splitHN, total_slots,
           (w0_bit + w1_bit) * (size_t)total_slots,
           (double)((w0_bit + w1_bit) * (size_t)total_slots) / 8 / 1024);
    printf("                                                + %zu KB target/offset (all members)\n",
           tgt_total / 1024);
    printf("  OMP:         %d threads\n", threadN);
    printf("  Train/Eval:  %d / %d samples  batch=%d\n", trainN, evalN, batchN);
    printf("  Score:       Σ_h Σ_b [ y×log(P_k) + (1-y)×log(1-P_k) ]\n");
    printf("  Predict:     argmax  (NO training, NO AdamW)\n");
    printf("  ───────────────────────────────────────────────────────────\n");
    const char *rng_src;
    if (aa.seed_file[0])
        rng_src = "true random file";
    else
        rng_src = "splitmix64";
    printf("  Seed:        seed=%u  %s", aa.seed, rng_src);
    if (aa.seed_file[0]) {
        printf("  from %s", aa.seed_file);
    }
    printf("  seed-member: %s", ensemble_seed_str());
    printf("  multi-correct: %s", aa.multi_correct ? "on" : "off");
    if (aa.filter_mask) {
        printf("  filter:");
        for (int _k = 0; _k < KI_NCLASSES; _k++)
            if ((aa.filter_mask >> _k) & 1)
                printf(" %s(%d)", ki_class_names[_k], _k);
    }
    if (aa.save_scores[0])
        printf("  Save-scores: %s\n", aa.save_scores);
    printf("\n");
}


/* Forward declaration */
static const char *opp_name(int ch);

/* ═══════════════════════════════════════════════════════════════════════
 * PRINT MEMBER STRUCTURE — zeigt Grid + Per-Member + H/C-Struktur
 * ═══════════════════════════════════════════════════════════════════════
 * Wird von dry-run und main() aufgerufen — keine Datenabhängigkeit.
 */
static void print_member_structure(int ensembleN, int splitVN, int splitHN,
                                    int H_local, int NC_slice, int channel) {
    int total = ensembleN * eff_colors * splitHN;
    (void)splitVN;
    printf("\n══╡ MEMBER ╞══════════════════════════════════════════════════\n");
    printf("  Grid: ENSEMBLE[%d] × COLOR[%d] × HN[%d] = %d members\n",
           ensembleN, eff_colors, splitHN, total);
    printf("  Per member: W0[H=%d × I=%d], Target[K=%d × H=%d × V=%d]\n",
           H_local, NC_slice, KI_NCLASSES, H_local, 32 / splitVN);
    int max_col = eff_colors;
    /* Build arrays for ki_print_member_structure, lookup encoding */
    int _c[64], _t[64], _w[64];
    int _n = 0;
    for (int ci = 0; ci < max_col && _n < 64; ci++) {
        int col = active_chans[ci];
        for (int hi = 0; hi < splitHN && _n < 64; hi++) {
            _c[_n] = col;
            _t[_n] = -1; _w[_n] = -1;
            /* Nachschlagen: enc_array für diese Farbe (oder default -1) */
            for (int ei = 0; ei < aa.enc_count && ei < KI_ENC_MAX; ei++) {
                int ec = (int)aa.enc_array[ei].color;
                if (ec == col || ec < 0) {  /* explizite Farbe oder default */
                    _t[_n] = (int)aa.enc_array[ei].type;
                    _w[_n] = (int)aa.enc_array[ei].width;
                    break;
                }
            }
            _n++;
        }
    }
    ki_print_member_structure(_c, _t, _w, _n, ensembleN);
    if (ensembleN > 1) {
        if (aa.ensemble_seed == ENS_SEED_CONST) {
            printf("  → ENSEMBLE x%d: alle Channel-Member teilen W0 (const)\n",
                   ensembleN);
            if (aa.seed_file[0])
                printf("    W0 from %s, 1 chunk per ensemble\n", aa.seed_file);
        } else if (aa.ensemble_seed == ENS_SEED_INCR) {
            printf("  → ENSEMBLE x%d: %d independent W0, seed=incr\n",
                   ensembleN, ensembleN);
        } else if (aa.seed_file[0]) {
            printf("  → ENSEMBLE x%d: %d independent W0 from %s (once)\n",
                   ensembleN, ensembleN, aa.seed_file);
        } else {
            printf("  → ENSEMBLE x%d: %d independent W0, seed=once\n",
                   ensembleN, ensembleN);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * BLOCK NAME — block index (0..6) → "R"|"G"|"B"|"Y"|"LUM"|"RG"|"BY"
 * ═══════════════════════════════════════════════════════════════════════
 */
static const char *opp_name(int ch) {
    return ki_color_name(ch);
}

/* ═══════════════════════════════════════════════════════════════════
 * AUTONOMOUS MEMBER — eigener Speicher, eigene Fehler, eigener Schritt
 * ═══════════════════════════════════════════════════════════════════
 *
 * Jeder Member verwaltet seine Ressourcen selbst. Kein malloc/free
 * pro Epoche → kein Cache-Bouncing durch Allokator-Overhead.
 */
typedef struct ki_Member {
    /* Dimensionen (aus CLI, konstant) */
    int H_local;            /* Neurons (H, no vertical split) */
    int NC_slice;           /* Container (KI_NC / splitHN) */
    int slc_off;            /* Input-Offset für diesen Member */
    int vi;                 /* Encoding-Index in enc_array (für stats/debug) */

    /* Zeiger auf externe Daten (Member besitzt target+offset, teilt W0) */
    const uint32_t *W0;     /* W0 row start (geteilt oder eigen) */
    int32_t *target;        /* [H_local × KI_NCLASSES × 32] int32 — eigener Speicher */
    int64_t *offset;        /* [KI_NCLASSES] int64 — eigener Speicher */

    /* Trainings-Puffer (einmal alloziert, jede Epoche wiederverwendet) */
    uint32_t *h0_buf;       /* [total_train × H_local] — rohe H0-Werte */
    uint32_t *gb_buf;       /* [total_train × H_local] — VN-Gruppenmaske (h0_to_gb) */
    int orig_m;             /* Index im Export (für korrekte Reihenfolge) */
    uint32_t *gb_buf_te;    /* [total_eval × H_local] — eval-Cache (NULL wenn kein eval) */

    /* Best-State (für Export bei bestem eval) */
    int32_t *best_target;   /* [H_local × KI_NCLASSES × V] — Snapshot bei bestem eval */
    int64_t *best_offset;   /* [KI_NCLASSES] */
    float   best_evl;       /* eval bei bestem Snapshot */

    /* Err-State (für rollback, nur wenn aa.err_rollback) */
    int32_t *err_target;    /* [H_local × KI_NCLASSES × V] — Snapshot bei bestem train-err */
    int64_t *err_offset;    /* [KI_NCLASSES] */

    /* Member-Zustand (pro Epoche aktualisiert) */
    int step;               /* aktueller Schritt */
    int last_err;           /* letzter Fehler */
    int ep;                 /* eigene Epoche (für Cosine) */
    int evl_ok;             /* eval correct count (--debug) */
} ki_Member;

static void export_ensemble(const char *out_dir,
                             const uint32_t *W0_ens, int total_members,
                             ki_Member **members, int active_members,
                             int H_local, int NC_slice, int nc_total) {
    char cmd[512], path[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", out_dir);
    if (system(cmd) != 0) return;
    snprintf(path, sizeof(path), "%s/model.otto", out_dir);

    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[ERROR] Cannot write %s\n", path); return; }

    uint32_t magic = OTTO_MAGIC, ver = OTTO_VERSION_V6, mode = H0_MODE_VAL;
    uint32_t n_mem = (uint32_t)total_members;
    uint32_t hh = (uint32_t)((size_t)H_local * (size_t)NC_slice);
    uint32_t ncc = (uint32_t)nc_total;
    uint32_t prec = (uint32_t)OT_PRECISION;
    uint32_t hl = (uint32_t)H_local, ncs = (uint32_t)NC_slice;

    fwrite(&magic,4,1,f); fwrite(&ver,4,1,f);
    fwrite(&mode,4,1,f); fwrite(&n_mem,4,1,f);
    fwrite(&hh,4,1,f); fwrite(&ncc,4,1,f);
    fwrite(&hl,4,1,f); fwrite(&ncs,4,1,f);
    fwrite(&prec,4,1,f);

    size_t w0_bytes = (size_t)H_local * (size_t)NC_slice * 4;
    size_t tgt_bytes = (size_t)H_local * KI_NCLASSES * VN_GROUPS_ * 4;
    size_t off_bytes = KI_NCLASSES * 8;
    size_t total = 0;

    for (int m = 0; m < total_members; m++) {
        fwrite(W0_ens + (size_t)m * (size_t)H_local * NC_slice, sizeof(uint32_t), (size_t)H_local * NC_slice, f);
        /* Find member with orig_m == m, or write zeros if filtered out */
        int found = 0;
        for (int b = 0; b < active_members && !found; b++) {
            if (members[b]->orig_m == m) {
                fwrite(members[b]->target, sizeof(int32_t), (size_t)H_local * KI_NCLASSES * VN_GROUPS_, f);
                fwrite(members[b]->offset, sizeof(int64_t), KI_NCLASSES, f);
                found = 1;
            }
        }
        if (!found) {
            int32_t *zeros = (int32_t *)calloc((size_t)H_local * KI_NCLASSES * VN_GROUPS_, sizeof(int32_t));
            fwrite(zeros, sizeof(int32_t), (size_t)H_local * KI_NCLASSES * VN_GROUPS_, f);
            free(zeros);
            int64_t *oz = (int64_t *)calloc(KI_NCLASSES, sizeof(int64_t));
            fwrite(oz, sizeof(int64_t), KI_NCLASSES, f);
            free(oz);
        }
        total += w0_bytes + tgt_bytes + off_bytes;
    }

    fclose(f);

    printf("\n══╡ EXPORT ╞═══════════════════════════════════════════════════\n");
    printf("  Model:  %s  (v7, %d members, H_local=%d, NC_slice=%d, F=%d)\n",
           path, total_members, H_local, NC_slice, 1<<OT_PRECISION);
    printf("  Total:  %zu KB (%d × (W0=%zuKB + Tgt=%zuKB + Off=%zuB))\n",
           (24 + total) / 1024, total_members,
           w0_bytes / 1024, tgt_bytes / 1024, off_bytes);
    fflush(stdout);
}

/* ── Member erzeugen: alloziert target, offset, h0_buf, gb_buf* ── */
static ki_Member *ki_member_create(int H_local, int NC_slice, int slc_off,
                                    const uint32_t *W0, int total_train,
                                    int total_eval) {
    ki_Member *m = (ki_Member *)malloc(sizeof(ki_Member));
    if (!m) { fprintf(stderr, "[FATAL] ki_member_create OOM\n"); exit(1); }
    m->H_local  = H_local;
    m->NC_slice = NC_slice;
    m->slc_off  = slc_off;
    m->W0       = W0;
    m->step     = 0;
    m->last_err = 0;
    m->ep       = 0;

    size_t tgt_sz = (size_t)H_local * KI_NCLASSES * VN_GROUPS_;
    m->target = (int32_t *)ki_xcalloc(tgt_sz, sizeof(int32_t));
    m->offset = (int64_t *)ki_xcalloc(KI_NCLASSES, sizeof(int64_t));
    m->best_target = (int32_t *)ki_xcalloc(tgt_sz, sizeof(int32_t));
    m->best_offset = (int64_t *)ki_xcalloc(KI_NCLASSES, sizeof(int64_t));
    m->best_evl = 0.0f;
    if (aa.err_rollback) {
        m->err_target = (int32_t *)ki_xcalloc(tgt_sz, sizeof(int32_t));
        m->err_offset = (int64_t *)ki_xcalloc(KI_NCLASSES, sizeof(int64_t));
    } else {
        m->err_target = NULL;
        m->err_offset = NULL;
    }

    size_t h0_sz = (size_t)total_train * (size_t)H_local;
    m->h0_buf = (uint32_t *)ki_xmalloc(h0_sz * sizeof(uint32_t));
    m->gb_buf = (uint32_t *)ki_xmalloc(h0_sz * sizeof(uint32_t));
    if (total_eval > 0) {
        size_t te_sz = (size_t)total_eval * (size_t)H_local;
        m->gb_buf_te = (uint32_t *)ki_xmalloc(te_sz * sizeof(uint32_t));
    } else {
        m->gb_buf_te = NULL;
    }
    return m;
}

/* ── Member freigeben ─────────────────────────────────────────── */
static void ki_member_destroy(ki_Member *m) {
    if (!m) return;
    free(m->target);
    free(m->offset);
    free(m->best_target);
    free(m->best_offset);
    free(m->err_target);
    free(m->err_offset);
    free(m->h0_buf);
    free(m->gb_buf);
    free(m->gb_buf_te);
    free(m);
}

/* forward declaration (defined below after ki_batch_correct helpers) */
static inline uint32_t h0_to_gb(uint32_t h0);

/* ── Member: Test-Eval gb vorberechnen (einmalig) ───────────────── *
 * Schreibt direkt in gb_buf_te, kein h0_buf nötig (nur gb wird
 * für evaluate_member gebraucht). */
static void ki_member_compute_gb_te(ki_Member *m, const uint32_t *X,
                                     int N, int n_cont) {
    if (!m->gb_buf_te || N <= 0) return;
    const uint32_t *in_base = X + (size_t)m->slc_off;
    #pragma omp parallel for schedule(static)
    for (int s = 0; s < N; s++) {
        const uint32_t *in = in_base + (size_t)s * (size_t)n_cont;
        for (int h = 0; h < m->H_local; h++) {
            size_t idx = (size_t)s * (size_t)m->H_local + (size_t)h;
            m->gb_buf_te[idx] = h0_to_gb(
                h0_neuron(in, m->W0 + (size_t)h * (size_t)m->NC_slice, m->NC_slice));
        }
    }
}

/* ── Member: h0 + gb vorberechnen (einmalig, jede Epoche wiederverwendet) ── *
 * gb_buf = VN-Gruppenmaske aus h0 (vermeidet popcount-Reduktion im Training). */
static void ki_member_compute_h0(ki_Member *m, const uint32_t *X, int N,
                                  int n_cont) {
    const uint32_t *in_base = X + (size_t)m->slc_off;
    #pragma omp parallel for schedule(static)
    for (int s = 0; s < N; s++) {
        const uint32_t *in = in_base + (size_t)s * (size_t)n_cont;
        for (int h = 0; h < m->H_local; h++) {
            size_t idx = (size_t)s * (size_t)m->H_local + (size_t)h;
            uint32_t hv = h0_neuron(in, m->W0 + (size_t)h * (size_t)m->NC_slice, m->NC_slice);
            m->h0_buf[idx] = hv;
            m->gb_buf[idx] = h0_to_gb(hv);
        }
    }
}
static int save_scores_archive(const char *out_dir, const uint32_t *X,
                                const uint8_t *y, int N,
                                ki_Member **members, int active_members,
                                int n_cont, int evl_ok)
{
    if (N <= 0 || active_members <= 0) return -1;

    /* Build archive filename from parameters */
    int te_int = (int)(aa.target_err * 100.0f + 0.5f);
    int64_t stamp = (int64_t)time(NULL);  /* creation timestamp, in file AND filename */
    char fname[512];
    snprintf(fname, sizeof(fname), "%s/H%d_EP%d_VN%d_HN%d_TE%d_SD%u_F4_TS%" PRId64 ".ens",
             out_dir, aa.hidden, aa.epochs, aa.splitVN, aa.splitHN,
             te_int, aa.seed, stamp);
    /*        ↑↑
     * F4 = Format version 4 (embedded in filename for at-a-glance readability) */

    /* Create output directory */
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", out_dir);
    if (system(cmd) != 0) return -1;

    FILE *f = fopen(fname, "wb");
    if (!f) { fprintf(stderr, "[ERROR] Cannot write %s\n", fname); return -1; }

    /* ── Header ─────────────────────────────────────────── */
    uint32_t magic = 0x454E534D;  /* 'ENSM' */
    uint32_t ver   = 4;          /* v4: per-member eval accuracy */
    uint32_t n_test    = (uint32_t)N;
    uint32_t n_classes = (uint32_t)KI_NCLASSES;
    uint32_t n_mem     = (uint32_t)active_members;
    uint32_t hidden    = (uint32_t)aa.hidden;
    uint32_t epochs    = (uint32_t)aa.epochs;
    uint32_t split_vn  = (uint32_t)aa.splitVN;
    uint32_t split_hn  = (uint32_t)aa.splitHN;
    float    tgt_err   = aa.target_err;
    uint32_t seed      = aa.seed;

    fwrite(&magic, 4, 1, f); fwrite(&ver, 4, 1, f);
    fwrite(&n_test, 4, 1, f); fwrite(&n_classes, 4, 1, f);
    fwrite(&n_mem, 4, 1, f);
    fwrite(&hidden, 4, 1, f); fwrite(&epochs, 4, 1, f);
    fwrite(&split_vn, 4, 1, f); fwrite(&split_hn, 4, 1, f);
    fwrite(&tgt_err, 4, 1, f); fwrite(&seed, 4, 1, f);
    fwrite(&stamp, 8, 1, f);   /* v3: embedded timestamp */

    /* v4: ensemble eval accuracy (from evl_ok parameter) */
    float ensemble_eval_f = (float)evl_ok * 100.0f / (float)N;
    fwrite(&ensemble_eval_f, 4, 1, f);

    /* ── Per-member metadata (v2/v3/v4) ───────────────────── */
    for (int m = 0; m < active_members; m++) {
        ki_Member *mem = members[m];
        uint8_t color    = (uint8_t)aa.enc_array[mem->vi].color;
        uint8_t enc_type = (uint8_t)aa.enc_array[mem->vi].type;
        uint8_t enc_wid  = (uint8_t)aa.enc_array[mem->vi].width;
        uint8_t pad      = 0;
        fwrite(&color, 1, 1, f); fwrite(&enc_type, 1, 1, f);
        fwrite(&enc_wid, 1, 1, f); fwrite(&pad, 1, 1, f);
    }

    /* ── Per-member scores ──────────────────────────────── */
    size_t score_sz = (size_t)N * (size_t)KI_NCLASSES;
    for (int m = 0; m < active_members; m++) {
        ki_Member *mem = members[m];
        int64_t *sc = (int64_t *)calloc(score_sz, sizeof(int64_t));
        if (!sc) { fprintf(stderr, "[FATAL] save_scores: OOM\n"); fclose(f); return -1; }

        #pragma omp parallel for schedule(static)
        for (int s = 0; s < N; s++) {
            int64_t scc[KI_NCLASSES];
            scores_otto(X + (size_t)s * (size_t)n_cont + mem->slc_off,
                        mem->W0, mem->H_local, mem->NC_slice,
                        mem->target, mem->offset, scc);
            for (int k = 0; k < KI_NCLASSES; k++)
                sc[(size_t)s * KI_NCLASSES + (size_t)k] = scc[k];
        }

        fwrite(sc, sizeof(int64_t), score_sz, f);
        free(sc);
    }

    /* Append ground truth labels (uint8[N]) */
    fwrite(y, 1, (size_t)N, f);

    fclose(f);
    printf("  Archive:  %s  (%d members x %d samples, %zu KB)\n",
           fname, active_members, N,
           (sizeof(int64_t) * score_sz * (size_t)active_members + 44) / 1024);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * HIP ACCELERATION — Multi-member GPU h0 (via cifar-1/hip-mem.cu)
 * ═══════════════════════════════════════════════════════════════════════
 * Compile with -DUSE_HIP and link against hip-mem.o -lamdhip64.
 * h0 is computed ONCE before training (cached for all epochs).
 */
#ifdef USE_HIP
extern int hip_mem_init(int N, int H, int NC_c, int stride, int members);
extern void hip_mem_upload_X(const uint32_t *X);
extern void hip_mem_upload_W0_all(const uint32_t *W0_flat, int total_W0_entries);
extern void hip_mem_compute_h0_all(uint32_t *h0_flat);
extern void hip_mem_done(void);
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * VN-GROUP BITMASK — h0 → gb conversion (cached, not per-epoch recomputed)
 * ═══════════════════════════════════════════════════════════════════════
 * Für VN=1: gb = h0 (jedes Bit = eigene Gruppe).
 * Für VN≥2: gb hat ein Bit pro Gruppe, gesetzt wenn popcount >
 *            N_BITS-1 (strict AND: alle Bits müssen 1 sein). */
static inline uint32_t h0_to_gb(uint32_t h0) {
    switch (aa.splitVN) {
        case 1:  return h0;
        case 2: { uint32_t g = 0;
            g |= (__builtin_popcount((h0) & 0x3u) > 1) ? 1u<<0 : 0;
            g |= (__builtin_popcount(((h0)>>2) & 0x3u) > 1) ? 1u<<1 : 0;
            g |= (__builtin_popcount(((h0)>>4) & 0x3u) > 1) ? 1u<<2 : 0;
            g |= (__builtin_popcount(((h0)>>6) & 0x3u) > 1) ? 1u<<3 : 0;
            g |= (__builtin_popcount(((h0)>>8) & 0x3u) > 1) ? 1u<<4 : 0;
            g |= (__builtin_popcount(((h0)>>10) & 0x3u) > 1) ? 1u<<5 : 0;
            g |= (__builtin_popcount(((h0)>>12) & 0x3u) > 1) ? 1u<<6 : 0;
            g |= (__builtin_popcount(((h0)>>14) & 0x3u) > 1) ? 1u<<7 : 0;
            g |= (__builtin_popcount(((h0)>>16) & 0x3u) > 1) ? 1u<<8 : 0;
            g |= (__builtin_popcount(((h0)>>18) & 0x3u) > 1) ? 1u<<9 : 0;
            g |= (__builtin_popcount(((h0)>>20) & 0x3u) > 1) ? 1u<<10 : 0;
            g |= (__builtin_popcount(((h0)>>22) & 0x3u) > 1) ? 1u<<11 : 0;
            g |= (__builtin_popcount(((h0)>>24) & 0x3u) > 1) ? 1u<<12 : 0;
            g |= (__builtin_popcount(((h0)>>26) & 0x3u) > 1) ? 1u<<13 : 0;
            g |= (__builtin_popcount(((h0)>>28) & 0x3u) > 1) ? 1u<<14 : 0;
            g |= (__builtin_popcount(((h0)>>30) & 0x3u) > 1) ? 1u<<15 : 0;
            return g; }
        case 3: { uint32_t g = 0;
            g |= (__builtin_popcount((h0) & 0x7u) > 2) ? 1u<<0 : 0;
            g |= (__builtin_popcount(((h0)>>3) & 0x7u) > 2) ? 1u<<1 : 0;
            g |= (__builtin_popcount(((h0)>>6) & 0x7u) > 2) ? 1u<<2 : 0;
            g |= (__builtin_popcount(((h0)>>9) & 0x7u) > 2) ? 1u<<3 : 0;
            g |= (__builtin_popcount(((h0)>>12) & 0x7u) > 2) ? 1u<<4 : 0;
            g |= (__builtin_popcount(((h0)>>15) & 0x7u) > 2) ? 1u<<5 : 0;
            g |= (__builtin_popcount(((h0)>>18) & 0x7u) > 2) ? 1u<<6 : 0;
            g |= (__builtin_popcount(((h0)>>21) & 0x7u) > 2) ? 1u<<7 : 0;
            g |= (__builtin_popcount(((h0)>>24) & 0x7u) > 2) ? 1u<<8 : 0;
            g |= (__builtin_popcount(((h0)>>27) & 0x7u) > 2) ? 1u<<9 : 0;
            return g; }
        case 4: { uint32_t g = 0;
            g |= (__builtin_popcount((h0) & 0xFu) > 3) ? 1u<<0 : 0;
            g |= (__builtin_popcount(((h0)>>4) & 0xFu) > 3) ? 1u<<1 : 0;
            g |= (__builtin_popcount(((h0)>>8) & 0xFu) > 3) ? 1u<<2 : 0;
            g |= (__builtin_popcount(((h0)>>12) & 0xFu) > 3) ? 1u<<3 : 0;
            g |= (__builtin_popcount(((h0)>>16) & 0xFu) > 3) ? 1u<<4 : 0;
            g |= (__builtin_popcount(((h0)>>20) & 0xFu) > 3) ? 1u<<5 : 0;
            g |= (__builtin_popcount(((h0)>>24) & 0xFu) > 3) ? 1u<<6 : 0;
            g |= (__builtin_popcount(((h0)>>28) & 0xFu) > 3) ? 1u<<7 : 0;
            return g; }
        case 8: { uint32_t g = 0;
            g |= (__builtin_popcount((h0) & 0xFFu) > 7) ? 1u<<0 : 0;
            g |= (__builtin_popcount(((h0)>>8) & 0xFFu) > 7) ? 1u<<1 : 0;
            g |= (__builtin_popcount(((h0)>>16) & 0xFFu) > 7) ? 1u<<2 : 0;
            g |= (__builtin_popcount(((h0)>>24) & 0xFFu) > 7) ? 1u<<3 : 0;
            return g; }
        case 16: { uint32_t g = 0;
            if (__builtin_popcount((h0) & 0xFFFFu) > 15) g |= 1u<<0;
            if (__builtin_popcount(h0 >> 16) > 15) g |= 1u<<1;
            return g; }
        case 32: return (__builtin_popcount(h0) > 31) ? 1u<<0 : 0;
        default: return 0;
    }
}

static inline int ki_omp_nthreads(void) {
    int n = 1;
    #pragma omp parallel
    #pragma omp single
    n = omp_get_num_threads();
    return n;
}

static inline int32_t **ki_cache_alloc(int n_threads, size_t tgt_sz) {
    int32_t **cache = (int32_t **)malloc((size_t)n_threads * sizeof(int32_t *));
    if (!cache) { fprintf(stderr, "[FATAL] ki_cache_alloc(%d) failed\n", n_threads); exit(1); }
    for (int t = 0; t < n_threads; t++)
        cache[t] = (int32_t *)ki_xcalloc(tgt_sz, sizeof(int32_t));
    return cache;
}

static inline void ki_cache_apply_free(int32_t **cache, int n_threads,
                                        int32_t *target, size_t tgt_sz) {
    for (int t = 0; t < n_threads; t++) {
        int32_t *ct = cache[t];
        for (size_t i = 0; i < tgt_sz; i++)
            target[i] += ct[i];
        free(ct);
    }
    free(cache);
}

/* ═══════════════════════════════════════════════════════════════════════
 * BATCH CORRECTION — parallel + deterministisch via Mini-Batches
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Phasen pro Batch:
 *   1. Parallel: Scores aus target berechnen, Deltas in Thread-Cache
 *   2. Sequentiell: Deltas auf target anwenden → nächster Batch sieht Änderung
 *
 * Schrittgrösse pro Sample:
 *   gap = sc[pred] - sc[true_k]   (Member-eigener Score-Abstand)
 *   gap > 0 → Korrektur, step skaliert proportional: step × gap / F
 *   gap ≥ F → vollen Schritt
 *   gap ≤ 0 → kein Update (Member lag richtig)
 *
 * target:     Ziel-Target (mit Offset für Ensemble)
 * H:          Anzahl Neuronen
 * class_offset: Offset pro Klasse
 * gb_all:     Vorberechnete VN-Gruppenmasken [N × H] (aus h0_to_gb)
 * y:          Labels
 * N:          Anzahl Trainings-Samples
 * step:       Basis-Schritt (Obergrenze, Member skaliert via gap)
 * tgt_sz:     Grösse des Target-Arrays (H × KI_NCLASSES × V)
 *
 * Returns:    Anzahl Korrekturen
 */
static inline int ki_batch_correct(int32_t *target, int H,
                                    const int64_t *class_offset,
                                    const uint32_t *gb_all,
                                    const uint8_t *y,
                                    int N, int step, size_t tgt_sz,
                                    int filter_mask) {
    int n_threads = ki_omp_nthreads();
    int32_t **dc = ki_cache_alloc(n_threads, tgt_sz);
    int corrections = 0;

    for (int b_start = 0; b_start < N; b_start += aa.batchN) {
        int b_end = b_start + aa.batchN;
        if (b_end > N) b_end = N;
        int batch_corr = 0;
        int batch_multi = 0;
        int64_t sc[KI_NCLASSES];

        #pragma omp parallel for reduction(+:batch_corr,batch_multi) firstprivate(sc) schedule(static)
        for (int s = b_start; s < b_end; s++) {
            if (filter_mask && !((filter_mask >> (int)y[s]) & 1))
                continue;
            int tid = omp_get_thread_num();
            int true_k = (int)y[s];
            const uint32_t *gb_s = gb_all + (size_t)s * (size_t)H;
            for (int k = 0; k < KI_NCLASSES; k++) sc[k] = class_offset[k];

            /* Score mit vorberechneter VN-Gruppenmaske */
#if 0
            for (int h = 0; h < H; h++)
              switch (aa.splitVN) {
                  case 1:  VN_SCORE_FROM_GB(gb_s[h], h, H, 32, target, sc); break;
                  case 2:  VN_SCORE_FROM_GB(gb_s[h], h, H, 16, target, sc); break;
                  case 3:  VN_SCORE_FROM_GB(gb_s[h], h, H, 10, target, sc); break;
                  case 4:  VN_SCORE_FROM_GB(gb_s[h], h, H,  8, target, sc); break;
                  case 8:  VN_SCORE_FROM_GB(gb_s[h], h, H,  4, target, sc); break;
                  case 16: VN_SCORE_FROM_GB(gb_s[h], h, H,  2, target, sc); break;
                  case 32: VN_SCORE_FROM_GB(gb_s[h], h, H,  1, target, sc); break;
              }
#else
            switch (aa.splitVN) {
                case 1:  for (int h = 0; h < H; h++) VN_SCORE_FROM_GB(gb_s[h], h, H, 32, target, sc); break;
                case 2:  for (int h = 0; h < H; h++) VN_SCORE_FROM_GB(gb_s[h], h, H, 16, target, sc); break;
                case 3:  for (int h = 0; h < H; h++) VN_SCORE_FROM_GB(gb_s[h], h, H, 10, target, sc); break;
                case 4:  for (int h = 0; h < H; h++) VN_SCORE_FROM_GB(gb_s[h], h, H,  8, target, sc); break;
                case 8:  for (int h = 0; h < H; h++) VN_SCORE_FROM_GB(gb_s[h], h, H,  4, target, sc); break;
                case 16: for (int h = 0; h < H; h++) VN_SCORE_FROM_GB(gb_s[h], h, H,  2, target, sc); break;
                case 32: for (int h = 0; h < H; h++) VN_SCORE_FROM_GB(gb_s[h], h, H,  1, target, sc); break;
            }
#endif

            int pred = 0;
            for (int k = 1; k < KI_NCLASSES; k++)
                if (sc[k] > sc[pred]) pred = k;

            if (pred != true_k) {
                int64_t gap = sc[pred] - sc[true_k];
                if (gap > 0) {
                    int step_i;
                    if (gap < (int64_t)OT_F) {
                        int64_t scaled = ((int64_t)step * gap) >> OT_PRECISION;
                        step_i = (int)scaled;
                        if (step_i < 1) step_i = 1;
                    } else {
                        step_i = step;
                    }
                    if (aa.multi_correct) {
                        int n_over = 0;
                        int over_k[KI_NCLASSES];
                        int64_t over_gap[KI_NCLASSES];
                        int64_t total_gap = 0;

                        for (int k = 0; k < KI_NCLASSES; k++) {
                            if (k == true_k) continue;
                            int64_t g = sc[k] - sc[true_k];
                            if (g > 0) {
                                over_k[n_over] = k;
                                over_gap[n_over] = g;
                                total_gap += g;
                                n_over++;
                            }
                        }

                        if (n_over > 0) {
                            batch_corr++;
                            if (n_over > 1) {
                                batch_multi++;
                                if (total_gap < (int64_t)OT_F) {
                                    int64_t scaled = ((int64_t)step * total_gap) >> OT_PRECISION;
                                    if (scaled > (int64_t)step) scaled = step;
                                    step_i = (int)scaled;
                                    if (step_i < 1) step_i = 1;
                                } else {
                                    step_i = step;
                                }
                            }
                            int64_t assigned = 0;
                            for (int oi = 0; oi < n_over; oi++) {
                                int k = over_k[oi];
                                int64_t step_k = (oi == n_over - 1)
                                    ? (int64_t)step_i - assigned
                                    : ((int64_t)step_i * over_gap[oi]) / total_gap;
                                if (step_k < 1) step_k = 1;
                                if (assigned + step_k > step_i)
                                    step_k = step_i - assigned;
                                assigned += step_k;
                                if (step_k > 0) {
#if 0
                                    for (int h = 0; h < H; h++)
                                      switch (aa.splitVN) {
                                          case 1:  VN_CORRECT_FROM_GB(gb_s[h], h, H, 32, dc[tid], true_k, k, (int)step_k); break;
                                          case 2:  VN_CORRECT_FROM_GB(gb_s[h], h, H, 16, dc[tid], true_k, k, (int)step_k); break;
                                          case 3:  VN_CORRECT_FROM_GB(gb_s[h], h, H, 10, dc[tid], true_k, k, (int)step_k); break;
                                          case 4:  VN_CORRECT_FROM_GB(gb_s[h], h, H,  8, dc[tid], true_k, k, (int)step_k); break;
                                          case 8:  VN_CORRECT_FROM_GB(gb_s[h], h, H,  4, dc[tid], true_k, k, (int)step_k); break;
                                          case 16: VN_CORRECT_FROM_GB(gb_s[h], h, H,  2, dc[tid], true_k, k, (int)step_k); break;
                                          case 32: VN_CORRECT_FROM_GB(gb_s[h], h, H,  1, dc[tid], true_k, k, (int)step_k); break;
                                      }
#else
                                    switch (aa.splitVN) {
                                        case 1:  for (int h = 0; h < H; h++) VN_CORRECT_FROM_GB(gb_s[h], h, H, 32, dc[tid], true_k, k, (int)step_k); break;
                                        case 2:  for (int h = 0; h < H; h++) VN_CORRECT_FROM_GB(gb_s[h], h, H, 16, dc[tid], true_k, k, (int)step_k); break;
                                        case 3:  for (int h = 0; h < H; h++) VN_CORRECT_FROM_GB(gb_s[h], h, H, 10, dc[tid], true_k, k, (int)step_k); break;
                                        case 4:  for (int h = 0; h < H; h++) VN_CORRECT_FROM_GB(gb_s[h], h, H,  8, dc[tid], true_k, k, (int)step_k); break;
                                        case 8:  for (int h = 0; h < H; h++) VN_CORRECT_FROM_GB(gb_s[h], h, H,  4, dc[tid], true_k, k, (int)step_k); break;
                                        case 16: for (int h = 0; h < H; h++) VN_CORRECT_FROM_GB(gb_s[h], h, H,  2, dc[tid], true_k, k, (int)step_k); break;
                                        case 32: for (int h = 0; h < H; h++) VN_CORRECT_FROM_GB(gb_s[h], h, H,  1, dc[tid], true_k, k, (int)step_k); break;
                                    }
#endif
                                }
                            }
                        }
                    } else {
                        batch_corr++;
#if 0
                        for (int h = 0; h < H; h++)
                          switch (aa.splitVN) {
                                  case 1:  VN_CORRECT_FROM_GB(gb_s[h], h, H, 32, dc[tid], true_k, pred, step_i); break;
                                  case 2:  VN_CORRECT_FROM_GB(gb_s[h], h, H, 16, dc[tid], true_k, pred, step_i); break;
                                  case 3:  VN_CORRECT_FROM_GB(gb_s[h], h, H, 10, dc[tid], true_k, pred, step_i); break;
                                  case 4:  VN_CORRECT_FROM_GB(gb_s[h], h, H,  8, dc[tid], true_k, pred, step_i); break;
                                  case 8:  VN_CORRECT_FROM_GB(gb_s[h], h, H,  4, dc[tid], true_k, pred, step_i); break;
                                  case 16: VN_CORRECT_FROM_GB(gb_s[h], h, H,  2, dc[tid], true_k, pred, step_i); break;
                                  case 32: VN_CORRECT_FROM_GB(gb_s[h], h, H,  1, dc[tid], true_k, pred, step_i); break;
                          }
#else
                        switch (aa.splitVN) {
                            case 1:  for (int h = 0; h < H; h++) VN_CORRECT_FROM_GB(gb_s[h], h, H, 32, dc[tid], true_k, pred, step_i); break;
                            case 2:  for (int h = 0; h < H; h++) VN_CORRECT_FROM_GB(gb_s[h], h, H, 16, dc[tid], true_k, pred, step_i); break;
                            case 3:  for (int h = 0; h < H; h++) VN_CORRECT_FROM_GB(gb_s[h], h, H, 10, dc[tid], true_k, pred, step_i); break;
                            case 4:  for (int h = 0; h < H; h++) VN_CORRECT_FROM_GB(gb_s[h], h, H,  8, dc[tid], true_k, pred, step_i); break;
                            case 8:  for (int h = 0; h < H; h++) VN_CORRECT_FROM_GB(gb_s[h], h, H,  4, dc[tid], true_k, pred, step_i); break;
                            case 16: for (int h = 0; h < H; h++) VN_CORRECT_FROM_GB(gb_s[h], h, H,  2, dc[tid], true_k, pred, step_i); break;
                            case 32: for (int h = 0; h < H; h++) VN_CORRECT_FROM_GB(gb_s[h], h, H,  1, dc[tid], true_k, pred, step_i); break;
                        }
#endif
                    }
                }
            }
        }

        /* Apply + clear cache */
        for (int t = 0; t < n_threads; t++) {
            int32_t *ct = dc[t];
            for (size_t i = 0; i < tgt_sz; i++) {
                int d = ct[i];
                if (d != 0) target[i] += d;
            }
            memset(ct, 0, tgt_sz * sizeof(int32_t));
        }
        corrections += batch_corr;
    }

    for (int t = 0; t < n_threads; t++) free(dc[t]);
    free(dc);
    return corrections;
}

/* ── Member: batch correct (nutzt gb_buf + target + offset) ────────── *
 * gb_buf = vorberechnete VN-Gruppenmaske (keine popcount-Reduktion nötig). */
static inline int ki_member_batch_correct(ki_Member *m, const uint8_t *y, int N, int step) {
    m->step = step;
    int err = ki_batch_correct(m->target, m->H_local, m->offset,
                                m->gb_buf, y, N, step, (size_t)m->H_local * KI_NCLASSES * 32,
                                aa.filter_mask);
    m->last_err = err;
    m->ep++;
    return err;
}

/* ═══════════════════════════════════════════════════════════════════
 * EVALUATE — Members außen, Samples innen (cache-optimal)
 * ═══════════════════════════════════════════════════════════════════
 *
 * Members seriell außen → target[m] (80 KB) bleibt im D1-Cache für
 * alle N Samples. Reduziert D1mr von 55% auf <1% (vorher: samples
 * außen → jeder member-Wechsel evicted target aus dem Cache).
 *
 * Nutzt direkt die ki_Member-Structs (keine flachen Arrays mehr).
 * Jeder Member hat eigene W0, target, offset, slc_off.
 *
 * Votes-Zwischenspeicher: N × KI_NCLASSES × 8 Byte (4 MB für 50000 Samples).
 * n_cont:   Container pro Sample (NC, für Stride)
 * Returns:  Anzahl korrekt klassifizierte Samples
 */
static int ki_evaluate_member(const uint32_t *X, const uint8_t *y, int N,
                               ki_Member **members, int active_members,
                               int n_cont, uint8_t *pred_out, int use_gb)
{
    if (N <= 0) return 0;

    /* Votes-Accumulator: N Samples × KI_NCLASSES Klassen */
    int64_t (*votes)[KI_NCLASSES] = (int64_t (*)[KI_NCLASSES])calloc((size_t)N, sizeof(int64_t[KI_NCLASSES]));
    if (!votes) { fprintf(stderr, "[FATAL] evaluate: votes OOM\n"); exit(1); }

    /* Jeder Member bekommt gleiches Stimmrecht: Skaliere sc[] so dass
     * max|sc[k]| ≤ SCALE_MAX. Verhindert dass Member mit grossen
     * Target-Gewichten (mehr Korrekturen, andere Channel) dominieren. */
    #define VOTE_SCALE ((int64_t)1 << 24)  /* 16.7 Mio, passt ×16 in int64 */

    /* Members außen: target[m] bleibt warm im D1-Cache */
    //#pragma omp parallel for schedule(static) if(active_members > 8)
    for (int m = 0; m < active_members; m++) {
        ki_Member *mem = members[m];

        #pragma omp parallel for schedule(static)
        for (int s = 0; s < N; s++) {
            int64_t sc[KI_NCLASSES];
            if (use_gb == 2) {
                /* ── Test-Eval: gb_buf_te ist 1× berechnet und gecached. */
                scores_otto_from_gb(s, mem->H_local, mem->gb_buf_te,
                                   mem->target, mem->offset, sc);
            } else if (use_gb == 1) {
                /* ── Training-Eval: gb_buf ist 1× berechnet und gecached. */
                scores_otto_from_gb(s, mem->H_local, mem->gb_buf,
                                   mem->target, mem->offset, sc);
            } else {
                /* ── Rohpixel → h0_neuron (Test-Set ohne gb_buf) */
                scores_otto(X + (size_t)s * (size_t)n_cont + mem->slc_off,
                           mem->W0, mem->H_local, mem->NC_slice,
                           mem->target, mem->offset, sc);
            }

            /* Vote-Normalisierung: --optional target-norm
             * Jeder Member wird auf max|sc| = VOTE_SCALE normiert.
             * Alle Member haben dadurch gleiches Stimmrecht,
             * unabhängig von Target-Grösse oder Kanal. */
            if (aa.opt_target_norm) {
                int64_t max_abs = 0;
                for (int k = 0; k < KI_NCLASSES; k++) {
                    int64_t a = (sc[k] >= 0) ? sc[k] : -sc[k];
                    if (a > max_abs) max_abs = a;
                }
                if (max_abs > 0) {
                    for (int k = 0; k < KI_NCLASSES; k++)
                        votes[s][k] += sc[k] * VOTE_SCALE / max_abs;
                } else {
                    for (int k = 0; k < KI_NCLASSES; k++)
                        votes[s][k] += sc[k];
                }
            } else {
                for (int k = 0; k < KI_NCLASSES; k++)
                    votes[s][k] += sc[k];
            }
        }
    }
    #undef VOTE_SCALE

    /* Merge: argmax pro Sample */
    int ok = 0;
    for (int s = 0; s < N; s++) {
        int pred = 0;
        for (int k = 1; k < KI_NCLASSES; k++)
            if (votes[s][k] > votes[s][pred]) pred = k;
        if (pred_out) pred_out[s] = (uint8_t)pred;
        if (pred == (int)y[s]) ok++;
    }
    free(votes);
    return ok;
}

/* ═══════════════════════════════════════════════════════════════════
 * PER-MEMBER DEBUG STATS — Tabelle nach jeder Epoche
 * ═══════════════════════════════════════════════════════════════════
 * Nur bei --debug aktiv.  Zeigt pro Member:
 *   Channel=Encoding · Target min/max · Step · last_err · Eigenständige eval
 * Ermöglicht zu erkennen, welche Member helfen/schaden.
 */
static void print_member_debug(ki_Member **members, int active_members,
                                const uint32_t *X, const uint8_t *y, int N,
                                int n_cont, int ep) {
    if (!aa.debug) return;

    /* ── Kopfzeile ────────────────────────────────────────────── */
    printf("\n  ── Member stats (Ep %d) ─────────────────────────────\n", ep + 1);
    printf("  %-4s  %-18s  %9s  %8s  %9s  %8s  %5s  %7s  %6s  %6s\n",
           "idx", "channel=encoding", "tgt_min", "min@k:h:v", "tgt_max", "max@k:h:v", "step",
           "trn_err", "evl_acc", "evl_err");

    int V = VN_GROUPS_;
    for (int mi = 0; mi < active_members; mi++) {
        ki_Member *mem = members[mi];

        /* ── Channel=Encoding Name aus enc_array[vi] ──────────── */
        char label[32];
        if (aa.enc_count > 0 && mem->vi < aa.enc_count) {
            int col = (int)aa.enc_array[mem->vi].color;
            int typ = (int)aa.enc_array[mem->vi].type;
            int w   = (int)aa.enc_array[mem->vi].width;
            const char *cn = (col >= 0) ? ki_color_name(col) : "?";
            const char *en = ki_enc_name_short((int8_t)typ);
            snprintf(label, sizeof(label), "%s=%s%d", cn, en, w);
        } else {
            snprintf(label, sizeof(label), "member#%d", mi);
        }

        /* ── Target min/max mit Position (Klasse, Neuron, VN) ── */
        size_t tgt_sz = (size_t)mem->H_local * KI_NCLASSES * (size_t)V;
        int tgt_min = 0, tgt_max = 0;
        size_t pos_min = 0, pos_max = 0;
        if (tgt_sz > 0) {
            tgt_min = tgt_max = mem->target[0];
            for (size_t i = 1; i < tgt_sz; i++) {
                int v = mem->target[i];
                if (v < tgt_min) { tgt_min = v; pos_min = i; }
                if (v > tgt_max) { tgt_max = v; pos_max = i; }
            }
        }
        int _hv = mem->H_local * V;
        int min_k = (int)(pos_min / (size_t)_hv);
        int min_h = (int)((pos_min % (size_t)_hv) / (size_t)V);
        int min_v = (int)(pos_min % (size_t)V);
        int max_k = (int)(pos_max / (size_t)_hv);
        int max_h = (int)((pos_max % (size_t)_hv) / (size_t)V);
        int max_v = (int)(pos_max % (size_t)V);

        /* ── Eigenständige eval (nur dieser Member) ───────────── */
        int member_ok = 0;
        if (N > 0) {
            int64_t *sc = (int64_t *)calloc((size_t)N * KI_NCLASSES, sizeof(int64_t));
            if (sc) {
                for (int s = 0; s < N; s++) {
                    int64_t scc[KI_NCLASSES];
                    scores_otto(X + (size_t)s * (size_t)n_cont + mem->slc_off,
                                mem->W0, mem->H_local, mem->NC_slice,
                                mem->target, mem->offset, scc);
                    /* Vote ohne target-norm (wie Default) */
                    for (int k = 0; k < KI_NCLASSES; k++)
                        sc[(size_t)s * KI_NCLASSES + (size_t)k] = scc[k];
                }
                for (int s = 0; s < N; s++) {
                    int pred = 0;
                    int64_t *row = sc + (size_t)s * KI_NCLASSES;
                    for (int k = 1; k < KI_NCLASSES; k++)
                        if (row[k] > row[pred]) pred = k;
                    if (pred == (int)y[s]) member_ok++;
                }
                free(sc);
            }
        }
        mem->evl_ok = member_ok;

        /* ── Zeile ausgeben ───────────────────────────────────── */
        int evl_err = N - member_ok;
        char _minpos[16], _maxpos[16];
        snprintf(_minpos, sizeof(_minpos), "%d:%d:%d", min_k, min_h, min_v);
        snprintf(_maxpos, sizeof(_maxpos), "%d:%d:%d", max_k, max_h, max_v);
        printf("  %-4d  %-18s  %9d  %8s  %9d  %8s  %5d  %7d  %5.1f%%  %6d\n",
               mi, label, tgt_min, _minpos, tgt_max, _maxpos, mem->step,
               mem->last_err,
               (double)member_ok * 100.0 / (double)N, evl_err);
    }
    printf("  ─────────────────────────────────────────────────────\n\n");
    fflush(stdout);
}

/* ═══════════════════════════════════════════════════════════════════
 * CLASS-VOTING DEBUG — Member × Klasse Trefferquote auf trainN
 * ═══════════════════════════════════════════════════════════════════
 * Nur bei --debug-class-voting aktiv.  Zeigt pro Member und Klasse
 * wie oft der Member korrekt lag (pred == y[s]) geteilt durch
 * Anzahl Samples dieser Klasse.
 *
 * Zeilen = Member (mit Channel=Encoding-Name)
 * Spalten = Klassen 0..K-1  + avg
 */
static void print_class_voting_debug(ki_Member **members, int active_members,
                                      const uint32_t *X, const uint8_t *y,
                                      int N, int n_cont, int ep) {
    if (N <= 0 || active_members <= 0) return;

    /* ── Accumulatoren ─────────────────────────────────────────── */
    int *total = (int *)calloc((size_t)KI_NCLASSES, sizeof(int));
    int (*correct)[KI_NCLASSES] = (int (*)[KI_NCLASSES])
        calloc((size_t)active_members, sizeof(int[KI_NCLASSES]));
    if (!total || !correct) {
        fprintf(stderr, "[FATAL] print_class_voting_debug OOM\n");
        free(total); free(correct); exit(1);
    }

    /* ── Erster Pass: Samples pro Klasse zählen ────────────────── */
    for (int s = 0; s < N; s++) {
        int k = (int)y[s];
        if (k >= 0 && k < KI_NCLASSES) total[k]++;
    }

    /* ── Zweiter Pass: pro Member Scores berechnen, argmax, vergleich ── */
    for (int m = 0; m < active_members; m++) {
        ki_Member *mem = members[m];
        for (int s = 0; s < N; s++) {
            int64_t sc[KI_NCLASSES];
            scores_otto(X + (size_t)s * (size_t)n_cont + mem->slc_off,
                        mem->W0, mem->H_local, mem->NC_slice,
                        mem->target, mem->offset, sc);
            int pred = 0;
            for (int k = 1; k < KI_NCLASSES; k++)
                if (sc[k] > sc[pred]) pred = k;
            if (pred == (int)y[s]) {
                int true_k = (int)y[s];
                if (true_k >= 0 && true_k < KI_NCLASSES)
                    correct[m][true_k]++;
            }
        }
    }

    /* ── Tabelle ausgeben ──────────────────────────────────────── */
    /* Nur Klassen mit Samples anzeigen (wichtig bei --filter) */
    int active_cols[KI_NCLASSES], n_active = 0;
    for (int k = 0; k < KI_NCLASSES; k++)
        if (total[k] > 0) active_cols[n_active++] = k;

    if (n_active == 0) { free(total); free(correct); return; }

    #define FORMAT_TEXT           "  %-14s"
    #define FORMAT_FLT            "  %6.1f%%"
    printf("\n  ── Class-voting stats (Ep %d) ──────────────────────────────\n", ep + 1);
    printf(FORMAT_TEXT, "member");
    for (int ai = 0; ai < n_active; ai++) {
        int k = active_cols[ai];
        printf("  %-7s", ki_class_names[k]);
    }
    printf("  %-7s\n", "avg");

    /* Trennlinie */
    printf(FORMAT_TEXT,"──────────────");
    for (int ai = 0; ai < n_active; ai++)
        printf("  %-7s", "───────");
    printf("  %-7s\n", "───────");

    int *ok_member = (int *)calloc((size_t)active_members, sizeof(int));
    for (int m = 0; m < active_members; m++) {
        /* ── Member-Label ─────────────────────────────────────── */
        char label[24];
        ki_Member *mem = members[m];
        if (aa.enc_count > 0 && mem->vi < aa.enc_count) {
            int col = (int)aa.enc_array[mem->vi].color;
            int typ = (int)aa.enc_array[mem->vi].type;
            int w   = (int)aa.enc_array[mem->vi].width;
            const char *cn = (col >= 0) ? ki_color_name(col) : "?";
            const char *en = ki_enc_name_short((int8_t)typ);
            snprintf(label, sizeof(label), "#%d %s=%s%d", m, cn, en, w);
        } else {
            snprintf(label, sizeof(label), "#%d", m);
        }

        printf(FORMAT_TEXT, label);
        int member_ok = 0;
        for (int ai = 0; ai < n_active; ai++) {
            int k = active_cols[ai];
            if (total[k] > 0) {
                double pct = (double)correct[m][k] * 100.0 / (double)total[k];
                printf(FORMAT_FLT, pct);
                member_ok += correct[m][k];
            } else {
                printf("       ");
            }
        }
        ok_member[m] = member_ok;
        double avg = (double)member_ok * 100.0 / (double)N;
        printf(FORMAT_FLT "\n", avg);
    }

    printf("  ──────────────────────────────────────────────────────────────\n\n");
    fflush(stdout);

    free(total); free(correct); free(ok_member);
    #undef FORMAT_TEXT
    #undef FORMAT_FLT
}

/* ═══════════════════════════════════════════════════════════════════
 * IFC MODEL LOAD — read exported .otto file
 * ═══════════════════════════════════════════════════════════════════
 * Returns arrays allocated by the caller (must free).
 */
static int ifc_load_model(const char *path,
                           uint32_t **W0_out, int32_t **tgt_out, int64_t **off_out,
                           int *n_mem, int *H_local, int *NC_slice) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[FATAL] Cannot open %s\n", path); return -1; }
    uint32_t magic, ver, mode, n_members, h_total, nc_total, h_local, ncs_slice, prec;
    if (fread(&magic,4,1,f)!=1 || fread(&ver,4,1,f)!=1 || fread(&mode,4,1,f)!=1 ||
        fread(&n_members,4,1,f)!=1 || fread(&h_total,4,1,f)!=1 || fread(&nc_total,4,1,f)!=1 ||
        fread(&h_local,4,1,f)!=1 || fread(&ncs_slice,4,1,f)!=1 || fread(&prec,4,1,f)!=1) {
        fprintf(stderr, "[FATAL] Cannot read header from %s\n", path);
        fclose(f); return -1;
    }
    if (magic != OTTO_MAGIC) { fprintf(stderr, "[FATAL] Bad magic\n"); fclose(f); return -1; }
    *n_mem    = (int)n_members;
    *H_local  = (int)h_local;
    *NC_slice = (int)ncs_slice;
    size_t w0_sz  = (size_t)n_members * (size_t)h_local * (size_t)ncs_slice;
    size_t w0_msz = (size_t)h_local * (size_t)ncs_slice;
    size_t tgt_msz = (size_t)h_local * KI_NCLASSES * 32;
    size_t off_msz = KI_NCLASSES;
    *W0_out  = (uint32_t *)malloc(w0_sz * sizeof(uint32_t));
    *tgt_out = (int32_t *)malloc((size_t)n_members * tgt_msz * sizeof(int32_t));
    *off_out = (int64_t *)calloc((size_t)n_members * off_msz, sizeof(int64_t));
    if (!*W0_out || !*tgt_out || !*off_out) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
    /* Datei-Layout (wie export_ensemble): per-member interleaved
     *   W0_m0  TGT_m0  OFF_m0  W0_m1  TGT_m1  OFF_m1  ...
     * Import muss im gleichen Loop lesen. */
    for (uint32_t m = 0; m < n_members; m++) {
        if (fread(*W0_out + (size_t)m * w0_msz, sizeof(uint32_t), w0_msz, f) != w0_msz ||
            fread(*tgt_out + (size_t)m * tgt_msz, sizeof(int32_t), tgt_msz, f) != tgt_msz ||
            fread(*off_out + (size_t)m * off_msz, sizeof(int64_t), off_msz, f) != off_msz) {
            fprintf(stderr, "[FATAL] Short read (member %u) from %s\n", m, path);
            free(*W0_out); free(*tgt_out); free(*off_out);
            fclose(f); return -1;
        }
    }
    fclose(f);
    printf("  Model: v%u  H=%d  NC=%d  ensemble=%d  mode=%s  F=%d\n",
           ver, h_local, ncs_slice, n_members, mode==0?"XNOR":"XOR", 1<<prec);
    return (int)n_members;
}

/* ═══════════════════════════════════════════════════════════════════
 * PREPROCESSED INPUT CACHE — load_input() mit Datei-Cache
 * ═══════════════════════════════════════════════════════════════════
 *
 * load_input() ist teuer (Farbraum + Encoding f. Samples).
 * Da die Encoding-Konfiguration pro Projekt konstant ist, koennen wir
 * die uint32-Container einmal berechnen und als Cache-Datei speichern.
 *
 * Format: data/prepped/<hash>_<samples>x<stride>.pre
 *   magic(4) ver(4) hash(4) samples(4) stride(4) data[...]
 */

/* Config-Hash fr Cache-Key */
static uint32_t input_cache_hash(void) {
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
#ifdef KI_DATASET_ID
    h = h * 31 + (uint32_t)KI_DATASET_ID;
#endif
    return h;
}

/* Versuche Cache zu laden, sonst load_input() + speichern */
static uint32_t *load_input_cached(const uint8_t *X_raw, int n_samples,
                                    size_t stride) {
    uint32_t hash = input_cache_hash();
    char cache_dir[512], cache_path[1024];

    snprintf(cache_dir, sizeof(cache_dir), "data/prepped");
    snprintf(cache_path, sizeof(cache_path), "%s/%08x_%dx%zu.pre",
             cache_dir, hash, n_samples, stride);

    /* Versuche zu laden */
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

    /* Cache miss: load_input und speichern */
    uint32_t *X = load_input(X_raw, n_samples);

    /* Cache speichern */
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


/* ═══════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[]) {
    aa.lr_step = (int)round(aa.lr * (1<<OT_PRECISION));
    ki_parse_args(argc, argv);
    /* Precompute encoding LUTs for each active (enc, width) pair */
    for (int _ei = 0; _ei < aa.enc_count; _ei++)
        enc_lut_init_enc((int)aa.enc_array[_ei].type, (int)aa.enc_array[_ei].width);
    if (aa.enc_count == 0) {
        int def_enc = aa.debug_binarize ? KI_ENC_LIN7 : KI_ENC_RAW;
        enc_lut_init_enc(def_enc, KI_ENC_WIDTH_DEFAULT);
    }
    if (KI_COLORS <= 1)
        aa.channel = KI_DEFAULT_COLOR;  /* MNIST: ignore --channels */
    /* 1:1 Mapping: Bit b = COLOR_BIT direkt (COLOR_MNIST=0, R=1, G=2, …, GB=10)
     * active_chans speichert die Bit-Position aus enum ki_color_bit. */
    {   int mask = aa.channel;
        int n = 0;
        for (int b = 0; b < COLOR_NB; b++)
            if (mask & (1 << b))
                active_chans[n++] = b;
        if (n == 0) { fprintf(stderr, "[FATAL] --channels: no channels selected\n"); return 1; }
        eff_colors = (aa.debug_flat && n > 1) ? 1 : n;
    }
    int eff_colors_orig = 0;  /* actual block count (for flat NC computation) */
    {   int cnt = 0;
        for (int b = 0; b < COLOR_NB; b++)
            if (aa.channel & (1 << b)) cnt++;
        eff_colors_orig = cnt;
    }
    /* Multi-Encoding: enc_array-Einträge = virtuelle Blöcke */
    if (aa.enc_count > 1) {
        eff_colors = aa.enc_count;
        /* active_chans mit den Farben aus enc_array füllen */
        for (int i = 0; i < aa.enc_count && i < KI_ENC_MAX; i++) {
            int col = (int)aa.enc_array[i].color;
            if (KI_COLORS > 1 && col >= 0 && col < COLOR_NB) {
                active_chans[i] = col;
            } else if (eff_colors_orig > 0) {
                active_chans[i] = active_chans[0];  /* MNIST: alle = gleiche Farbe */
            }
        }
    }
    omp_set_num_threads(aa.threadN);

    int H = aa.hidden;

    /* ── Slice configuration checks (before data loading!) ────── */
    int ensembleN = aa.ensembleN;
    if (ensembleN < 1) ensembleN = 1;
    int splitVN = aa.splitVN;
    if (splitVN < 1) splitVN = 1;
    int splitHN = aa.splitHN;
    if (splitHN < 1) splitHN = 1;

    /* ── Effektive Container pro Block (Encoding-Breite aus enc_array) ── */
    /* SplitHN-check: jeder enc_array-Eintrag muss teilbar sein */
    for (int i = 0; i < aa.enc_count && i < KI_ENC_MAX; i++) {
        int w = (int)aa.enc_array[i].width;
        if (w < 1) w = KI_ENC_WIDTH_DEFAULT;
        int ncc = (KI_COLORS > 1) ? KI_NC * w / 8 : NC * w / 8;
        if (ncc % splitHN != 0) {
            fprintf(stderr, "[FATAL] NC=%d not divisible by splitHN=%d\n", ncc, splitHN);
            fprintf(stderr, "  Valid splitHN values (divisors of %d): ", ncc);
            for (int d = 1; d <= ncc; d++)
                if (ncc % d == 0) fprintf(stderr, "%s%d", (d == 1) ? "" : ", ", d);
            fprintf(stderr, "\n");
            return 1;
        }
    }
    /* nc_blk (breitester Block, für step-validation) */
    int nc_blk = 0;
    for (int i = 0; i < aa.enc_count && i < KI_ENC_MAX; i++) {
        int w = (int)aa.enc_array[i].width;
        if (w < 1) w = KI_ENC_WIDTH_DEFAULT;
        int ncc = (KI_COLORS > 1) ? KI_NC * w / 8 : NC * w / 8;
        if (ncc > nc_blk) nc_blk = ncc;
    }
    if (nc_blk == 0) nc_blk = (KI_COLORS > 1 ? KI_NC : NC) * KI_ENC_WIDTH_DEFAULT / 8;

    /* ── step validation (before data loading!) ────────────────── */
    {
        int ncs = nc_blk / splitHN;           /* containers per slice (per-color) */
        int step_scale_div = H * nc_blk;      /* total neuron-container pairs */
        int sc = (H * ncs * 100) / step_scale_div;
        if (sc < 1) sc = 1;
        int minN = (100 + sc - 1) / sc;
        if (aa.stepN > 0 && aa.stepN < minN) {
            fprintf(stderr, "[FATAL] --step-const %d too small (step_norm=0)\n", aa.stepN);
            fprintf(stderr, "  H=%d  NC_slice=%d  KI_NC_TOTAL=%d\n", H, ncs, nc_blk * KI_COLORS);
            fprintf(stderr, "  step_scale=%d%%  min_stepN=%d\n", sc, minN);
            return 1;
        }
    }

    /* ── Load dataset (MNIST or CIFAR-10, ki-local.h adapts) ── */
    struct timeval tv_start, tv_end;
    gettimeofday(&tv_start, NULL);
    ki_Dataset data = { .dry_run = aa.dry_run };
    if (ki_dataset_read(&data) != 0) return 1;
    /* NOTE: --filter (aa.filter_mask) wirkt NUR auf Training.
     * Evaluation verwendet immer ALLE Klassen.  Kein
     * ki_filter_dataset() nötig — das Training skippt
     * nicht-matching Samples in ki_batch_correct(). */
    if (data.pixels != KI_PX) {
        fprintf(stderr, "[FATAL] Expected %d pixels, got %d\n", KI_PX, data.pixels);
        ki_dataset_free(&data); return 1;
    }

    /* ── Use dataset defaults if not explicitly set ── */
    if (aa.trainN <= 0 && data.n_train > 0) aa.trainN = data.n_train;
    if (aa.evalN  <= 0 && data.n_eval  > 0) aa.evalN  = data.n_eval;

    int total_train = aa.trainN;
    int total_eval  = aa.evalN;
    int total_all   = total_train + total_eval;

    /* ── Compute total stride = sum over all enc_array entries ── */
    size_t n_cont = 0;
    for (int i = 0; i < aa.enc_count; i++) {
        int w = (int)aa.enc_array[i].width;
        if (w < 1) w = KI_ENC_WIDTH_DEFAULT;
        n_cont += (size_t)((KI_COLORS > 1 ? KI_NC : NC) * w / 8);
    }
    /* ── Pixel-data-dependent init (skipped for dry-run) ────── */
    uint32_t *X_all = NULL;
    uint32_t *X_flat_free = NULL;
    uint32_t *X_perm = NULL;
    uint8_t  *y_perm = NULL;
    int own_eval_data = 0;
    if (!aa.dry_run) {
        X_all = load_input_cached(data.X_raw, total_all, n_cont);

        /* ── Flat mode: concat selected blocks into contiguous array ── */
        if (aa.debug_flat && eff_colors_orig > 1) {
            int *block_off_f = (int *)calloc((size_t)eff_colors_orig, sizeof(int));
            size_t flat_cont = 0;
            for (int bi = 0; bi < eff_colors_orig; bi++) {
                int bit = active_chans[bi];
                int w = KI_ENC_WIDTH_DEFAULT;
                for (int ei = 0; ei < aa.enc_count && ei < KI_ENC_MAX; ei++)
                    if ((int)aa.enc_array[ei].color == bit) { w = (int)aa.enc_array[ei].width; break; }
                if (w < 1) w = KI_ENC_WIDTH_DEFAULT;
                int ncb = KI_NC * w / 8;
                block_off_f[bi] = (int)flat_cont;
                flat_cont += (size_t)ncb;
            }
            X_flat_free = (uint32_t *)malloc((size_t)total_all * flat_cont * sizeof(uint32_t));
            if (!X_flat_free) { free(block_off_f); fprintf(stderr, "[FATAL] X_flat OOM\n"); return 1; }
            for (int s = 0; s < total_all; s++) {
                uint32_t *dst = X_flat_free + (size_t)s * flat_cont;
                for (int bi = 0; bi < eff_colors_orig; bi++) {
                    int bit = active_chans[bi];
                    int w = KI_ENC_WIDTH_DEFAULT;
                    for (int ei = 0; ei < aa.enc_count && ei < KI_ENC_MAX; ei++)
                        if ((int)aa.enc_array[ei].color == bit) { w = (int)aa.enc_array[ei].width; break; }
                    if (w < 1) w = KI_ENC_WIDTH_DEFAULT;
                    int ncb = KI_NC * w / 8;
                    memcpy(dst + (size_t)block_off_f[bi],
                           X_all + (size_t)s * n_cont + (size_t)block_off_f[bi],
                           (size_t)ncb * sizeof(uint32_t));
                }
            }
            free(block_off_f);
            free(X_all);
            X_all = X_flat_free;
            n_cont = flat_cont;
        }

        /* ── Optional: shuffle indices before train/eval split ──────── */
        if (aa.shuffle) {
            printf("  Shuffling %d samples before %d/%d split...\n",
                   total_all, total_train, total_eval);
            int *idx = (int *)malloc((size_t)total_all * sizeof(int));
            for (int i = 0; i < total_all; i++) idx[i] = i;
            srand(aa.seed);
            for (int i = total_all - 1; i > 0; i--) {
                int j = rand() % (i + 1);
                int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
            }
            X_perm = (uint32_t *)malloc((size_t)total_all * n_cont * sizeof(uint32_t));
            y_perm = (uint8_t *)malloc((size_t)total_all * sizeof(uint8_t));
            for (int i = 0; i < total_all; i++) {
                int src = idx[i];
                memcpy(X_perm + (size_t)i * n_cont, X_all + (size_t)src * n_cont,
                       n_cont * sizeof(uint32_t));
                y_perm[i] = data.y[src];
            }
            free(idx);
        }
    }
    uint32_t *X_tr  = X_all;
    uint32_t *X_te  = X_all ? X_all + (size_t)total_train * n_cont : NULL;
    uint8_t  *y_tr  = aa.dry_run ? NULL : data.y;
    uint8_t  *y_te  = aa.dry_run ? NULL : data.y + total_train;

    /* ── If shuffle occurred, switch to permuted pointers ─────── */
    if (X_perm) {
        X_tr = X_perm;
        X_te = X_perm + (size_t)total_train * n_cont;
        y_tr = y_perm;
        y_te = y_perm + total_train;
        own_eval_data = 1;
    }

    /* ── Compute per-block nc and offsets from enc_array ── */
    int multi_enc_blk_off[KI_ENC_MAX] = {0};
    int multi_enc_nc[KI_ENC_MAX] = {0};
    {   int off = 0;
        for (int i = 0; i < aa.enc_count && i < KI_ENC_MAX; i++) {
            int w = (int)aa.enc_array[i].width;
            if (w < 1) w = KI_ENC_WIDTH_DEFAULT;
            multi_enc_nc[i] = (KI_COLORS > 1 ? KI_NC : NC) * w / 8;
            multi_enc_blk_off[i] = off;
            off += multi_enc_nc[i];
        }
    }

    /* ── Compute slice dimensions ────────────────────────────── */
    int H_local   = H;
    int NC_slice  = nc_blk / splitHN;  /* base slice (from default width) */
    int total_members = ensembleN * splitHN * eff_colors;

    /* ── Default W0-Quelle: splitmix64 PRNG (kein auto-search mehr) ─── */
    /* --seed-file override → w0_rand_set_file() in W0 init.
     * Mit seed_splitmix=1 (default) wird immer splitmix64 genutzt. */

    /* ── IFC MODE: --import → evaluieren statt trainieren ───────────── */
    if (aa.importD[0]) {
        printf("\n══╡ INFERENCE ╞══════════════════════════════════════════════════\n");
        char model_path[1024];
        snprintf(model_path, sizeof(model_path), "%s/model.otto", aa.importD);
        uint32_t *W0_ifc; int32_t *tgt_ifc; int64_t *off_ifc;
        int n_mifc, hl_ifc, ns_ifc;
        if (ifc_load_model(model_path, &W0_ifc, &tgt_ifc, &off_ifc,
                           &n_mifc, &hl_ifc, &ns_ifc) < 0) return 1;
        /* Create ki_Member array from model data */
        int K = KI_NCLASSES, V = 32;
        ki_Member **mems = (ki_Member **)malloc((size_t)n_mifc * sizeof(ki_Member *));
        for (int i = 0; i < n_mifc; i++) {
            size_t w0_off = (size_t)i * (size_t)hl_ifc * (size_t)ns_ifc;
            mems[i] = ki_member_create(hl_ifc, ns_ifc, (int)((size_t)i * (size_t)ns_ifc),
                                       W0_ifc + w0_off, total_eval, 0);
            memcpy(mems[i]->target, tgt_ifc + (size_t)i * (size_t)hl_ifc * K * V,
                   (size_t)hl_ifc * K * V * sizeof(int32_t));
            memcpy(mems[i]->offset, off_ifc + (size_t)i * K, (size_t)K * sizeof(int64_t));
        }
        /* Evaluate (no correction, just forward) */
        uint8_t *pred_eval = aa.predictions[0]
            ? (uint8_t *)ki_xcalloc((size_t)total_eval, sizeof(uint8_t))
            : NULL;
        struct timeval tv0, tv1; gettimeofday(&tv0, NULL);
        /* BUG 2026-07-10: use_gb=2 crashed because gb_buf_te was never allocated
         * in IFC mode (ki_member_create gets total_eval=0). Use use_gb=0 to compute
         * scores from raw pixels — X_te and W0 are available. */
        int evl_ok = ki_evaluate_member(X_te, y_te, total_eval, mems, n_mifc, (int)n_cont, pred_eval, 0);
        gettimeofday(&tv1, NULL);
        int el = (int)((tv1.tv_sec-tv0.tv_sec)*1000 + (tv1.tv_usec-tv0.tv_usec)/1000);
        float acc = (float)evl_ok * 100.0f / (float)total_eval;
        printf("\n══╡ RESULT ╞══════════════════════════════════════════════════════\n");
        printf("  Eval:    %.1f%%  (%d/%d)\n", acc, evl_ok, total_eval);
        printf("  Time:    %dms\n", el);
        ki_report_show(0, 0, evl_ok, total_eval, el, aa.threadN, 0, 0.0f);
        /* ── Export per-sample predictions (for vis-errors) ─ */
        if (aa.predictions[0] && pred_eval) {
            FILE *pf = fopen(aa.predictions, "wb");
            if (pf) {
                uint32_t magic = 0x44455250;
                uint32_t n_eval = (uint32_t)total_eval;
                uint32_t off = 0;
                fwrite(&magic, 4, 1, pf);
                fwrite(&n_eval, 4, 1, pf);
                fwrite(&off, 4, 1, pf);
                fwrite(pred_eval, 1, (size_t)total_eval, pf);
                fclose(pf);
                printf("  Predictions: %s  (%d eval samples)\n",
                       aa.predictions, total_eval);
            } else {
                fprintf(stderr, "[ERROR] Cannot write %s\n", aa.predictions);
            }
        }
        free(pred_eval);
        for (int i = 0; i < n_mifc; i++) ki_member_destroy(mems[i]);
        free(mems); free(W0_ifc); free(tgt_ifc); free(off_ifc);
        ki_dataset_free(&data); free(X_all);
        if (X_perm) { free(X_perm); free(y_perm); }
        return 0;
    }

    print_setup(H, aa.epochs, total_train, total_eval, aa.threadN, aa.seed, aa.batchN,
                splitVN, splitHN, NC_slice, H_local, ensembleN, aa.channel, nc_blk, (int)n_cont);

    /* ── W0: random uint32[total_members][H_local][NC_slice] (frozen) ── */
    size_t w0_m_sz = (size_t)H_local * (size_t)NC_slice;
    size_t w0_sz = (size_t)total_members * w0_m_sz;
    uint32_t *W0_ens = (uint32_t *)ki_xmalloc(w0_sz * sizeof(uint32_t));

    /* Transparent: w0_random() liest aus Datei (falls --seed-file),
     * sonst aus splitmix64 PRNG.  Die member-strategy (const/incr/once)
     * arbeitet identisch in beiden Modi. */
    if (aa.seed_file[0])
        w0_rand_set_file(aa.seed_file);

    if (aa.ensemble_seed == ENS_SEED_CONST) {
        /* const: Jedes Ensemble bekommt eigenen W0-Chunk,
         * alle Channel-Member teilen ihn sich. */
        int memb_per_ens = eff_colors * splitHN;
        for (int e = 0; e < ensembleN; e++) {
            w0_srandom((unsigned int)(aa.seed + e));
            for (size_t i = 0; i < w0_m_sz; i++) {
                uint32_t v = w0_random();
                for (int mm = 0; mm < memb_per_ens; mm++) {
                    int m = e * memb_per_ens + mm;
                    W0_ens[(size_t)m * w0_m_sz + i] = v;
                }
            }
        }
    } else if (aa.ensemble_seed == ENS_SEED_INCR) {
        /* incr: Jeder Member bekommt eigenen Seed → anderen W0 */
        for (int m = 0; m < total_members; m++) {
            w0_srandom((unsigned int)(aa.seed + m));
            for (size_t i = 0; i < w0_m_sz; i++)
                W0_ens[(size_t)m * w0_m_sz + i] = w0_random();
        }
    } else {
        /* once (default): Ein Seed, alle Member sequentiell */
        w0_srandom((unsigned int)aa.seed);
        for (size_t i = 0; i < w0_sz; i++)
            W0_ens[i] = w0_random();
    }

#ifdef USE_HIP
    /* ── GPU init: upload all W0 once ── */
    int gpu_ok = 0;
    if (!aa.dry_run) {
        if (hip_mem_init(total_train, H_local, NC_slice, nc_blk, total_members) == 0) {
            hip_mem_upload_W0_all(W0_ens, total_members * H_local * NC_slice);
            printf("  [HIP] GPU enabled (%d members)\n", total_members);
            gpu_ok = 1;
        } else printf("  [HIP] Init failed, using CPU\n");
    }
#endif

    /* ── Target + Offset for each member ─────────────────────────────── */
    int V = VN_GROUPS_;
    size_t tgt_sz_m = (size_t)H_local * KI_NCLASSES * (size_t)V;  /* per member */

    int class_counts[KI_NCLASSES] = {0};
    if (!aa.dry_run)
        for (int s = 0; s < total_train; s++)
            class_counts[(int)y_tr[s]]++;

    print_member_structure(ensembleN, splitVN, splitHN, H_local, NC_slice, aa.channel);
    /* Target wird NACH h0-compute aus gb_buf gebaut (s.u.). */
    printf("\n");
    fflush(stdout);

    {
        int step = (int)(aa.lr * (float)OT_F + 0.5f);
        printf("══╡ TRAINING ╞══  lr=%.4f  step=%d  mode=%s  F=%d",
             (double)aa.lr, step, mode_str(), OT_F);
        if (aa.opt_target_norm)
            printf("  tgt-nrm=%d", aa.opt_target_norm ? 1 : 0);
        if (aa.target_err > 0.0f)
            printf("  tgt-err=%.2f", (double)aa.target_err);
        printf("\n");
        fflush(stdout);
    }
    struct timeval _tv0, _tv1, _target0, _target1, _logit0, _logit1, _total0, _total1;
    gettimeofday(&_total0, NULL);

    /* ── Iterative target tuning ──────────────────────────────────── */
    int step_init = (aa.lr > 0) ? (int)ot_precision(aa.lr) : aa.lr_step;
    int warmup = aa.warmup_epochs;
    int epochs = aa.epochs;

    /* ── Member-Array anlegen: jeder Member verwaltet sich selbst ─── */
    int active_members = 0;
    for (int m = 0; m < total_members; m++)
        if (!(aa.channel >= 0 && !(aa.channel & (1 << active_chans[(m / splitHN) % eff_colors]))))
            active_members++;
    ki_Member **members = (ki_Member **)malloc((size_t)active_members * sizeof(ki_Member *));
    if (!members) { fprintf(stderr, "[FATAL] members OOM\n"); return 1; }

    {
        int mem_idx = 0;
        for (int m = 0; m < total_members; m++) {
            int vi = (m / splitHN) % eff_colors;  /* virtual block index */
            int color = (KI_COLORS > 1) ? active_chans[vi] : COLOR_MNIST;
            if (aa.channel >= 0 && !(aa.channel & (1 << color)) && KI_COLORS > 1) continue;
            int slc_off;
            int mem_nc;
            if (aa.debug_flat) {
                mem_nc = NC_slice;
                slc_off = (m % splitHN) * NC_slice;
            } else if (aa.enc_count > 0) {
                mem_nc = multi_enc_nc[vi] / splitHN;
                slc_off = multi_enc_blk_off[vi] + (m % splitHN) * mem_nc;
            } else {
                mem_nc = NC_slice;
                slc_off = (m % splitHN) * NC_slice;
            }
            const uint32_t *W0_m =
                W0_ens + (size_t)m * w0_m_sz;
            members[mem_idx] = ki_member_create(H_local, mem_nc, slc_off,
                                            W0_m, total_train, total_eval);
            members[mem_idx]->orig_m = m;
            members[mem_idx]->vi = vi;
            /* target + offset werden NACH h0-compute aus gb_buf befüllt */
            members[mem_idx]->last_err = total_train;
            mem_idx++;
        }
    }

    /* Best-Snapshots (flat arrays, für Export) */
    float best_evl = 0.0f;
    int best_evl_ok = 0;        /* Anzahl korrekt beim besten eval */
    int best_err = total_train;
    int last_avg_err = total_train;

    /* pred_epoch: reusable prediction buffer (debug_confusion_all or filter needs it) */
    int need_pred = (aa.debug_confusion_all || aa.filter_mask) && !aa.dry_run;
    uint8_t *pred_epoch = need_pred ? (uint8_t *)malloc((size_t)total_train) : NULL ;

    /* ── h0 precompute: ONCE before training, cached for ALL epochs ──
     * h0 depends ONLY on X (fixed input) and W0 (frozen). It NEVER
     * changes between epochs — recomputing each epoch is wasted work! */
    gettimeofday(&_tv0, NULL);
    if (!aa.dry_run) {
#ifdef USE_HIP
        if (gpu_ok) {
            uint32_t *h0_all = (uint32_t *)malloc(
                (size_t)active_members * (size_t)total_train * (size_t)H_local * sizeof(uint32_t));
            hip_mem_upload_X(X_tr);
            hip_mem_compute_h0_all(h0_all);
            for (int _b = 0; _b < active_members; _b++) {
                size_t off = (size_t)_b * (size_t)total_train * (size_t)H_local;
                ki_Member *mem = members[_b];
                size_t n_pairs = (size_t)total_train * (size_t)H_local;
                memcpy(mem->h0_buf, h0_all + off, n_pairs * sizeof(uint32_t));
                /* gb_buf aus h0_buf ableiten */
                for (size_t i = 0; i < n_pairs; i++)
                    mem->gb_buf[i] = h0_to_gb(mem->h0_buf[i]);
            }
            free(h0_all);
        } else
#endif
        {
            //#pragma omp parallel for schedule(static) if(active_members > 8)
            for (int _b = 0; _b < active_members; _b++)
                ki_member_compute_h0(members[_b], X_tr, total_train, (int)n_cont);
        }
        /* Test-Eval gb einmal vorberechnen (wenn test data existiert) */
        if (total_eval > 0) {
            for (int _b = 0; _b < active_members; _b++)
                ki_member_compute_gb_te(members[_b], X_te, total_eval, (int)n_cont);
        }
        /* h0_buf nach gb-Berechnung freigeben — wird danach nie mehr gebraucht */
        for (int _b = 0; _b < active_members; _b++) {
            ki_Member *mem = members[_b];
            if (mem->h0_buf) { free(mem->h0_buf); mem->h0_buf = NULL; }
        }
    }
    gettimeofday(&_tv1, NULL);

    /* ── Target aus gb_buf bauen (KEIN h0_neuron mehr) ──────────── *
     * Überschreibt die uniformen Init-Werte (aus dem TARGET-INIT-Loop
     * oben) mit korrekten Counts. Anschliessend logit_convert + offset. */
    gettimeofday(&_target0, NULL);
    if (!aa.dry_run && V > 1) {
        for (int _b = 0; _b < active_members; _b++) {
            ki_Member *mem = members[_b];
            if (!mem->gb_buf) continue;
            int32_t *tgt = ki_build_target_from_gb(y_tr, total_train,
                mem->gb_buf, mem->H_local, V);
            /* Überschreibt die uniformen Init-Werte */
            memcpy(mem->target, tgt, (size_t)mem->H_local * KI_NCLASSES * (size_t)V * sizeof(int32_t));
            free(tgt);
        }
    }
    gettimeofday(&_target1, NULL);

    /* logit_convert + class_offset für alle Member (basierend auf neuen Targets) */
    gettimeofday(&_logit0, NULL);
    if (!aa.dry_run) {
        for (int _b = 0; _b < active_members; _b++) {
            ki_Member *mem = members[_b];
            size_t m_tgt_sz = (size_t)mem->H_local * KI_NCLASSES * (size_t)V;
            int64_t off_m[KI_NCLASSES];
            compute_class_offset(off_m, mem->target, mem->H_local, class_counts);
            memcpy(mem->offset, off_m, KI_NCLASSES * sizeof(int64_t));
            logit_convert(mem->target, mem->H_local, class_counts);
            /* Ersten Snapshot in best/err (für Export / Rollback) */
            memcpy(mem->best_target, mem->target, m_tgt_sz * sizeof(int32_t));
            memcpy(mem->best_offset, mem->offset, KI_NCLASSES * sizeof(int64_t));
            if (aa.err_rollback) {
                memcpy(mem->err_target, mem->target, m_tgt_sz * sizeof(int32_t));
                memcpy(mem->err_offset, mem->offset, KI_NCLASSES * sizeof(int64_t));
            }
        }
    }
    gettimeofday(&_logit1, NULL);
    gettimeofday(&_total1, NULL);

    /* Timing: Setup (h0+gb/gbl precompute) */
    int setup_target_ms = (int)((_target1.tv_sec-_target0.tv_sec)*1000 + (_target1.tv_usec-_target0.tv_usec)/1000);
    int setup_tv_ms = (int)((_tv1.tv_sec-_tv0.tv_sec)*1000 + (_tv1.tv_usec-_tv0.tv_usec)/1000);
    int setup_logit_ms = (int)((_logit1.tv_sec-_logit0.tv_sec)*1000 + (_logit1.tv_usec-_logit0.tv_usec)/1000);
    int setup_total_ms = (int)((_total1.tv_sec-_total0.tv_sec)*1000 + (_total1.tv_usec-_total0.tv_usec)/1000);
    if (setup_total_ms > 50)
        printf("  Setup:    %d ms (h0+gb) + %d ms (target) + %d ms (logit) = %d ms (total)\n", 
          setup_tv_ms, setup_target_ms, setup_logit_ms, setup_total_ms);

    for (int ep = 0; ep < epochs; ep++) {
        /* Step für jeden Member berechnen (jeder hat seinen eigenen) */
        int display_step = 0;
        //#pragma omp parallel for reduction(max:display_step) schedule(static) if(active_members > 1)
        for (int _a = 0; _a < active_members; _a++) {
            int s;
            switch (aa.step_mode) {
               case STEP_POW:
                   {
                       /* Power-law: step = step_init × (err/total)^step_power
                        * Fällt sanfter ab als cos-err, bleibt bei kleinen Fehlern aktiv.
                        * --step-power N steuert die Abfallgeschwindigkeit. */
                       float ratio = (float)members[_a]->last_err / (float)total_train;
                       float p = powf(ratio, aa.step_power);
                       s = (int)((float)step_init * p + 0.5f);
                       if (s < 1) s = 1;
                   }
                   break;
               case STEP_COS_ERR:
                  {
                      float err_ratio = (float)members[_a]->last_err / (float)total_train;
                      float progress = 1.0f - err_ratio;
                      if (progress < 0.0f) progress = 0.0f;
                      if (progress > 1.0f) progress = 1.0f;
                      float cosine = (1.0f + cosf(progress * (float)3.14159265358979323846f)) / 2.0f;
                      float lr_min_f = (aa.lr_min > 0.0f) ? aa.lr_min : 0.0f;
                      s = (int)((float)step_init * (lr_min_f + (1.0f - lr_min_f) * cosine) + 0.5f);
                  }
                  break;
               case STEP_CONST:
                   s = (aa.stepN > 0) ? aa.stepN : step_init;
                   break;
               default:  /* STEP_COS_TIME (or unknown → fallback cosine) */
                  {
                      /* Per-Member Cosine */
                      if (members[_a]->ep < warmup) {
                          float scale = (float)(members[_a]->ep + 1) / (float)warmup;
                          s = (int)((float)step_init * scale + 0.5f);
                      } else {
                          float progress = (float)(members[_a]->ep - warmup) / (float)((epochs+0) - warmup);
                          if (progress > 1.0f) progress = 1.0f;
                          float cosine = (1.0f + cosf(progress * (float)3.14159265358979323846f)) / 2.0f;
                          float lr_min_f = (aa.lr_min > 0.0f) ? aa.lr_min : 0.0f;
                          s = (int)((float)step_init * (lr_min_f + (1.0f - lr_min_f) * cosine) + 0.5f);
                      }
                  }
                  break;
             }
             /* Soft target-err: step skaliert wenn err_rate sich target nähert.
              * Überspringen in ep=0 (last_err noch nicht gesetzt). */
             if (aa.target_err > 0.0f && members[_a]->last_err > 0) {
                 float err_rate = (float)members[_a]->last_err / (float)total_train;
                 if (err_rate <= aa.target_err) {
                     s = 0;
                 } else {
                     float excess = (err_rate - aa.target_err) / aa.target_err;
                     if (excess > 1.0f) excess = 1.0f;
                     s = (int)((float)s * excess + 0.5f);
                 }
             }
             members[_a]->step = (s < 0) ? 0 : s;
            if (members[_a]->step > display_step) display_step = members[_a]->step;
        }

        /* Evaluate: Members außen, Samples innen (cache-optimal) */
        int trn_ok = ki_evaluate_member(X_tr, y_tr, total_train,
                        members, active_members, (int)n_cont, pred_epoch, 1);
        int evl_ok = 0;
        if (total_eval > 0) {
            evl_ok = ki_evaluate_member(X_te, y_te, total_eval,
                        members, active_members, (int)n_cont, NULL, 2);
        }
        float evl_acc = (total_eval > 0)
            ? (float)evl_ok * 100.0f / (float)total_eval : 0.0f;

        if (evl_acc > best_evl) {
            best_evl = evl_acc;
            best_evl_ok = evl_ok;
            /* Jetzt auch member->best_target setzen (für export) */
            for (int _b = 0; _b < active_members; _b++) {
                ki_Member *mem = members[_b];
                memcpy(mem->best_target, mem->target, tgt_sz_m * sizeof(int32_t));
                memcpy(mem->best_offset, mem->offset, KI_NCLASSES * sizeof(int64_t));
            }
        }

        gettimeofday(&tv_end, NULL);
        int elapsed = (int)((tv_end.tv_sec - tv_start.tv_sec) * 1000
                          + (tv_end.tv_usec - tv_start.tv_usec) / 1000);

        /* ── trn % / err: bei --filter NUR auf den trainierten Klassen ── */
        int trn_ok_filtered = 0, n_filtered_train = total_train;
        if (aa.filter_mask && pred_epoch) {
            n_filtered_train = 0;
            for (int s = 0; s < total_train; s++) {
                if ((aa.filter_mask >> y_tr[s]) & 1) {
                    n_filtered_train++;
                    if (pred_epoch[s] == y_tr[s]) trn_ok_filtered++;
                }
            }
        } else {
            trn_ok_filtered = trn_ok;
        }

        float trn_disp_acc = (n_filtered_train > 0)
            ? (float)trn_ok_filtered * 100.0f / (float)n_filtered_train : 0.0f;
        int trn_disp_err = n_filtered_train - trn_ok_filtered;

        printf("  Ep %2d  trn=%.1f%%  evl=%.1f%%  best=%.1f%%  step=%d  time=%dms",
               ep + 1, trn_disp_acc, evl_acc, best_evl, display_step, elapsed);

        if (aa.filter_mask && pred_epoch) {
            int trn_all_err = total_train - trn_ok;
            printf("  err=%d/%d  trnall=%.1f%%(%d)", trn_disp_err, n_filtered_train,
                   (float)trn_ok * 100.0f / (float)total_train, trn_all_err);
            last_avg_err = trn_disp_err;  /* step-decay basiert auf gefiltertem Fehler */
        } else {
            last_avg_err = total_train - trn_ok;
            printf("  err=%d", last_avg_err);
        }
        printf("\n");

        int _is_done = (display_step == 0 && aa.target_err > 0.0f && ep > 0);

        /* Jede Epoche korrigiert — auch die letzte! */
        if (!_is_done) {
            int member_err_sum = 0;
            /* h0 is PRE-COMPUTED once before training (see above).
             * Only correction runs each epoch — h0_buf is reusable. */
            for (int _b = 0; _b < active_members; _b++) {
                ki_Member *mem = members[_b];
                member_err_sum += ki_member_batch_correct(mem, y_tr, total_train, mem->step);
            }
            int trn_err = (active_members > 0) ? (member_err_sum / active_members) : 0;

            if (aa.err_rollback && trn_err > best_err && ep > 0) {
                printf("  ↑ err=%d best=%d  rollback", trn_err, best_err);
                {
                    for (int _b = 0; _b < active_members; _b++) {
                        ki_Member *mem = members[_b];
                        memcpy(mem->target, mem->err_target,
                               tgt_sz_m * sizeof(int32_t));
                        memcpy(mem->offset, mem->err_offset,
                               KI_NCLASSES * sizeof(int64_t));
                    }
                }
                float reduction = 2.0f / 3.0f;
                step_init = (int)((float)step_init * reduction + 0.5f);
                if (step_init < 1) step_init = 1;
                printf(", step_init %d", step_init);
                ep--; printf("\n"); continue;
            }
            if (aa.err_rollback && trn_err < best_err) {
                best_err = trn_err;
                {
                    for (int _b = 0; _b < active_members; _b++) {
                        ki_Member *mem = members[_b];
                        memcpy(mem->err_target, mem->target,
                               tgt_sz_m * sizeof(int32_t));
                        memcpy(mem->err_offset, mem->offset,
                               KI_NCLASSES * sizeof(int64_t));
                    }
                }
            }
            /* Early stopping: bei --filter den Ensemble-Fehler nutzen
             * (nicht die Per-Member-Korrekturen, die laufen weiter
             * während das Ensemble schon 100% erreicht hat). */
            int early_stop_err = aa.filter_mask ? trn_disp_err : trn_err;
            if (early_stop_err == 0) { printf("  ✓\n"); break; }
        }

        /* --debug: per-member stats NACH Korrektur */
        if (aa.debug) {
            int _n = (total_eval > 0) ? total_eval : total_train;
            const uint32_t *_x = (total_eval > 0) ? X_te : X_tr;
            const uint8_t  *_y2 = (total_eval > 0) ? y_te : y_tr;
            print_member_debug(members, active_members, _x, _y2, _n, (int)n_cont, ep);
        }

        /* --debug-class-voting(-all): IMMER auf trainN (niemals eval) */
        if (aa.debug_class_voting_all) {
            print_class_voting_debug(members, active_members,
                                     X_tr, y_tr, total_train, (int)n_cont, ep);
        }

        /* --debug-confusion-matrix-all: per-epoch confusion matrix on trainN */
        if (aa.debug_confusion_all && pred_epoch) {
            print_confusion_debug(y_tr, pred_epoch, total_train, ep, 0);
        }

        if (_is_done) break;
    }


    free(pred_epoch);
    /* Member-Destruktion (in ki_member_destroy freed: target, offset, best/err, h0, gb, gbl) */

    gettimeofday(&tv_end, NULL);
    int elapsed_ms = (int)((tv_end.tv_sec - tv_start.tv_sec) * 1000
                         + (tv_end.tv_usec - tv_start.tv_usec) / 1000);

    /* Final evaluation: mit AKTUELLEN Targets (nach letzter Korrektur) */
    int trn_ok=0, evl_ok = 0;
    uint8_t *pred_eval = aa.predictions[0] ?  (uint8_t *)ki_xcalloc((size_t)total_eval, sizeof(uint8_t)) : NULL ;
    /* pred_tr für debug_confusion — ki_evaluate_member nur 1× auf trainN */
    uint8_t *pred_tr = (aa.debug_confusion && !aa.dry_run)
        ? (uint8_t *)ki_xcalloc((size_t)total_train, sizeof(uint8_t)) : NULL ;
    if (!aa.dry_run) {

      trn_ok = ki_evaluate_member(X_tr, y_tr, total_train,
                      members, active_members, (int)n_cont, pred_tr, 1);
      if (total_eval > 0)
          evl_ok = ki_evaluate_member(X_te, y_te, total_eval,
                      members, active_members, (int)n_cont, pred_eval, 2);
    }

    /* ── Save per-member scores (f�r merge-ensemble) ────────── */
    if (aa.save_scores[0] && !aa.dry_run && total_eval > 0) {
        save_scores_archive(aa.save_scores, X_te, y_te, total_eval,
                            members, active_members, (int)n_cont, evl_ok);
    }

    /* ── Final debug output (vor member destruction) ───────── */
    if (aa.debug_class_voting && !aa.dry_run) {
        print_class_voting_debug(members, active_members,
                                 X_tr, y_tr, total_train, (int)n_cont, epochs - 1);
    }
    if (aa.debug_confusion && !aa.dry_run) {
        print_confusion_debug(y_tr, pred_tr, total_train, epochs - 1, 1);
    }
    free(pred_tr);

    /* Export MUSS vor Member-Destruktion erfolgen (liest members[b]->target/offset) */
    if (aa.exportD[0] != '\0')
    {
        export_ensemble(aa.exportD, W0_ens, total_members,
                        members, active_members,
                        H_local, NC_slice, (int)n_cont);
    }

    /* Members zerstören (nach finaler Evaluation + Export) */
    for (int _z = 0; _z < active_members; _z++)
        ki_member_destroy(members[_z]);

    float fin_trn = (float)trn_ok * 100.0f / (float)total_train;
    float fin_evl = (total_eval > 0)
        ? (float)evl_ok * 100.0f / (float)total_eval : 0.0f;
    int fin_err = total_train - trn_ok;  /* Fehler passend zu train=/eval= */

    printf("\n══╡ RESULT ╞══════════════════════════════════════════════════════\n");
    float final_best = (best_evl > fin_evl) ? best_evl : fin_evl;
    printf("  H=%d  ens=%d  v_split=%d  h_split=%d  ep=%d  trn=%.1f%%  evl=%.1f%%  best=%.1f%%  lr=%.4f  time=%dms\n",
           H, ensembleN, splitVN, splitHN, epochs, fin_trn, fin_evl, final_best,
           (double)aa.lr, elapsed_ms);

    /* REPORT uses best eval (whichever is higher: training loop best or final eval) */
    int report_evl_ok = (evl_ok > best_evl_ok) ? evl_ok : best_evl_ok;
    ki_report_show(trn_ok, total_train, report_evl_ok, total_eval,
                   elapsed_ms, aa.threadN, fin_err, aa.lr);

    /* ── Export per-sample predictions (eval only, for vis-errors) ─ */
    if (aa.predictions[0]) {
        FILE *pf = fopen(aa.predictions, "wb");
        if (pf) {
            uint32_t magic = 0x44455250;  /* 'PRED' in LE */
            uint32_t n_eval = (uint32_t)total_eval;
            uint32_t off = (uint32_t)total_train;  /* start offset in dataset */
            fwrite(&magic, 4, 1, pf);
            fwrite(&n_eval, 4, 1, pf);
            fwrite(&off, 4, 1, pf);
            fwrite(pred_eval, 1, (size_t)total_eval, pf);
            fclose(pf);
            printf("  Predictions: %s  (%d eval samples, offset=%d)\n",
                   aa.predictions, total_eval, total_train);
        } else {
            fprintf(stderr, "[ERROR] Cannot write %s\n", aa.predictions);
        }
        free(pred_eval);
    }

    if (own_eval_data) { free(X_perm); free(y_perm); }
    free(members);
    free(X_all);
    free(W0_ens);
#ifdef USE_HIP
    hip_mem_done();
#endif
    ki_dataset_free(&data);
    return 0;
}
