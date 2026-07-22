/*
 * otto-score-ifc/mnist/mlp-bin32-otto-trn-seq.c — SEQUENTIAL MEMBER TRAINING
 * ==========================================================================
 *
 * Same as mlp-bin32-otto-trn.c but processes members sequentially:
 *   One member at a time → all epochs → accumulate scores → next member.
 *
 * This reduces peak memory from ~55 GB to ~2 GB at H=512 + augmentation
 * because gb_buf is allocated for ONE member at a time, not all 240.
 *
 * Each member computes its gb_buf once, runs all epochs, evaluates,
 * accumulates votes, then frees gb_buf. Members never share memory.
 */
#include "ki-load.h"           /* encoding-aware load_input (CIFAR + MNIST) */
#include <stdio.h>
#include <unistd.h>
#include "maj3.h"
#include "maj1.h"
#include "maj7.h"
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

/* ── VN dispatch macro (generates switch over splitVN values) ── */
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

/* ── Forward declaration for ki_Member (struct definition follows
 * below, ki_evaluate_member is declared before) ───────────── */
typedef struct ki_Member ki_Member;

/* ── Global args (initialisiert in main) ────────────────────── */
/* ── --debug-epoch flag (local, not in ki-common.h) ── */
static int debug_epoch = 0;

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
    .gap_k              = 0.0f,
    .step_mode          = KI_DEFAULT_STEP_MODE,
    .ensembleN          = 1,
    .splitVN            = 1,
    .splitHN            = 1,
    .channel            = KI_DEFAULT_COLOR,/* CIFAR: r+g+b, MNIST: only block 0 */
    .packedB            = 1,
    .enc_default_type   = -1,    /* -1 = auto: falls bin→KI_ENC_LIN7, otherwise KI_ENC_RAW */
    .enc_default_width  = KI_ENC_WIDTH_DEFAULT,
    .enc_size           = KI_ENC_WIDTH_DEFAULT,
    .enc_count          = 0,     /* 0 = kein enc_array (legacy single) */
    .opt_target_norm    = KI_DEFAULT_TARGET_NORM,
    .ensemble_seed      = ENS_SEED_ONCE,
    .target_init_mode = KI_TARGET_COUNT,
    .multi_correct      = 0,
    .seed_splitmix      = 1,
    .maj_mode           = KI_MAJ_3,  /* --maj 3: tree approximation (faster + more accurate than 1) */
    .maj_step           = 0,      /* 0=auto (KI_PX_PER_CONT) */
    .debug_maj          = 0,      /* 0=auto, 1=container, 2=pixel */
    .rows_mode          = 0,      /* 0=flat, 1=per-row members */
    .member_threshold   = 0,      /* 0=disabled, else min trn%% to participate */
    .xforms             = (1 << KI_XFORM_ID),  /* default: identity only */
};

/* ═══════════════════════════════════════════════════════════════════════
 * CUSTOM load_input — 7-channel input buffer for CIFAR (KI_COLORS=3),
 * passthrough for MNIST (KI_COLORS=1, KI_PACK=4)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * ── INPUT-BUFFER LAYOUT (CIFAR, linear, 1 Bild = 7 × KI_NC uint32) ──
 *
 *   Block | Bit | Name | Formel (per pixel)         | Mapping auf 0..255
 *   ------|-----|------|----------------------------|-------------------
 *     0   |  0  |  R   | roter Rohpixel             | r
 *     1   |  1  |  G   | green raw pixel            | g
 *     2   |  2  |  B   | blauer Rohpixel            | b
 *     3   |  3  |  Y   | ITU-R BT.601               | (r*77+g*150+b*29)>>8
 *     4   |  4  |  LUM | R+G Luminanz               | (r+g)>>1
 *     5   |  5  |  RG  | R-G red-green opponent      | (r-g+255)>>1
 *     6   |  6  |  BY  | B-(R+G)/2 Blau-Gelb Opp.   | (2b-r-g+510)>>2
 *
 *   Buffer: [R(256)][G(256)][B(256)][Y(256)][LUM(256)][RG(256)][BY(256)][YL(256)]
 *           ↑0      ↑256   ↑512   ↑768   ↑1024    ↑1280    ↑1536    ↑1792
 *   n_cont = 8×256 = 2048 (FIXED, always all blocks)
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
 *   For each member m:
 *     seq_chan  = (m / splitHN) % eff_colors
 *     block     = active_chans[seq_chan]            // 0..7
 *     h_idx     = m % splitHN
 *     nc_off    = block * KI_NC + h_idx * NC_slice
 */
/* number of container blocks = number of colors (COLOR_NB, dynamic) */
#define KI_NB COLOR_NB


/* ── Konstanten ────────────────────────────────────────────────── */
#ifndef NC
#define NC        196     /* Default MNIST — override via -DNC or ki-local.h */
#endif

/* Global channel parameters set after --channels in main) */
static int eff_colors = 3;              /* number of active channels / members (popcount mask) */
static int active_chans[KI_ENC_MAX];    /* Mapping: seq_idx → Bit-Position (0..8) */
#define BITS       32
#define N_CLASSES KI_NCLASSES

/* Export file magic + version */
#define OTTO_MAGIC   0x4F54544FU   /* "OTTO" */
#define OTTO_VERSION 5U            /* v5 = ensemble (no precision field) */
#define OTTO_VERSION_V6 6U         /* v6 = ensemble + precision field */

/* Index: [KI_NCLASSES][H][32] — klasse × neuron × bit */


// === Zentraler Skalierungsfaktor OT_PRECISION ===
// All logit values are scaled by F
// stored in int32/int64.  The correction step and lr display
// derived from it.*a change affects all dependent places.
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
 * H0 FORWARD — MAJ3 over nc_local containers
 * ═══════════════════════════════════════════════════════════════════
 * in_offset: Start des Slices im Input-Array
 * nc_local:  number of Container for this member
 */
static uint32_t h0_neuron(const uint32_t *in, const uint32_t *W0_row, int nc_local) {
    uint32_t match[4096] = {0}; /* max nc_local */
    switch (aa.maj_mode) {
        case KI_MAJ_1: {
            /* Container-level flat (original majority_tree1) */
            for (int c = 0; c < nc_local; c++)
                match[c] = H0_MATCH(in, W0_row, c);
            return majority_tree1(match, nc_local);
        }
        case KI_MAJ_1R: {
            /* Container-level row-wise (old rowwise) */
            for (int c = 0; c < nc_local; c++)
                match[c] = H0_MATCH(in, W0_row, c);
            int cpr = KI_COLS / KI_PX_PER_CONT;
            return majority_tree1_rowwise(match, nc_local, cpr);
        }
        case KI_MAJ_1P: {
            /* Pixel-accurate flat (current default) */
            int half = nc_local * KI_PX_PER_CONT / 2;
            uint32_t r = 0;
            for (int g = 0; g < KI_PIXEL_GROUPS; g++) {
                for (int c = 0; c < nc_local; c++)
                    match[c] = H0_MATCH(in, W0_row + g * nc_local, c);
                r |= (majority_tree1_pixel(match, nc_local, half) << (g * KI_BIT_POS));
            }
            return r;
        }
        case KI_MAJ_1RP: {
            /* Pixel-accurate row-wise: per-row pixel-accurate, then cross-row majority */
            int cpr = KI_COLS / KI_PX_PER_CONT;  /* Container pro Zeile */
            int rows = nc_local / cpr;
            int half_row = cpr * KI_PX_PER_CONT / 2;
            uint32_t result = 0;
            for (int g = 0; g < KI_PIXEL_GROUPS; g++) {
                uint32_t row_results[256];
                const uint32_t *W0_group = W0_row + g * nc_local;
                for (int r = 0; r < rows; r++) {
                    const uint32_t *W0_row_r = W0_group + r * cpr;
                    const uint32_t *in_row = in + r * cpr;
                    for (int c = 0; c < cpr; c++)
                        match[c] = H0_MATCH(in_row, W0_row_r, c);
                    row_results[r] = majority_tree1_pixel(match, cpr, half_row);
                }
                uint32_t cross = majority_tree1(row_results, rows);
                result |= (cross << (g * KI_BIT_POS));
            }
            return result;
        }
        case KI_MAJ_7: {
            for (int c = 0; c < nc_local; c++)
                match[c] = H0_MATCH(in, W0_row, c);
            return majority_tree7(match, nc_local);
        }
        default: {
            for (int c = 0; c < nc_local; c++)
                match[c] = H0_MATCH(in, W0_row, c);
            /* KI_MAJ_3 with optional pixel step (0=auto=KI_PX_PER_CONT) */
            int _step = aa.maj_step;
            if (_step == 0) _step = KI_PX_PER_CONT;
            /* --debug-maj overrides auto-detection */
            if (aa.debug_maj == 1) {
                return majority_tree3(match, nc_local);
            } else if (aa.debug_maj == 2) {
                return majority_tree3_pixel_step(match, nc_local, _step);
            }
            /* Default auto: fast path for container-aligned steps, pixel for others */
            if (_step == KI_PX_PER_CONT) {
                return majority_tree3(match, nc_local);
            }
            return majority_tree3_pixel_step(match, nc_local, _step);
        }
    }
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
static COUNTER_TYPE *ki_build_target(const uint32_t *X, const uint8_t *Y, int N,
                               const uint32_t *W0, int H_local, int NC_slice,
                               int nc_off, int stride, int silent) {
    int V = VN_GROUPS_, G = aa.splitVN, TH = VN_THRESH_;
    size_t sz = (size_t)H_local * KI_NCLASSES * (size_t)V;
    COUNTER_TYPE *target = (COUNTER_TYPE *)ki_xcalloc(sz, sizeof(COUNTER_TYPE));

    if (!silent) {
        printf("\n=== OTTO TARGET ===\n");
        printf("  Target[%d][%d][%d] = %zu KB\n",
               KI_NCLASSES, H_local, V, sz * sizeof(COUNTER_TYPE) / 1024);
        fflush(stdout);
    }

    #pragma omp parallel
    {
        int32_t *lt = (int32_t *)ki_xcalloc(sz, sizeof(COUNTER_TYPE));
        #pragma omp for schedule(static)
        for (int s = 0; s < N; s++) {
            int k = (int)Y[s];
            const uint32_t *in = X + (size_t)s * (size_t)stride + nc_off;
            for (int h = 0; h < H_local; h++) {
                int _ws1 = aa.rows_mode && aa.maj_mode == KI_MAJ_1 ? NC_slice * 4 : NC_slice;
                uint32_t h0 = h0_neuron(in, W0 + (size_t)h * _ws1, NC_slice);
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
        { for (size_t i = 0; i < sz; i++) target[i] += (COUNTER_TYPE)lt[i]; }
        free(lt);
    }
    return target;
}

/* ── ki_build_target_from_gb: target counting from gb_buf (no h0_neuron) ── *
 * Uses precomputed gb_buf data instead of h0_neuron + VN reduction.
 * Twice as fast as ki_build_target because both h0_neuron AND
 * VN group computation are eliminated (both are already in gb_buf). */
static COUNTER_TYPE *ki_build_target_from_gb(const uint8_t *Y, int N,
    const uint32_t *gb_buf, int H_local, int V,
    const int class_counts[KI_NCLASSES]) {
    size_t sz = (size_t)H_local * KI_NCLASSES * (size_t)V;
    COUNTER_TYPE *target = (COUNTER_TYPE *)ki_xcalloc(sz, sizeof(COUNTER_TYPE));
    if (aa.target_init_mode == KI_TARGET_RANDOM) {
        /* Random init: target = uniform [0, nk] per class.
         * class_counts is provided by the caller (1× in main). */
        for (int k = 0; k < KI_NCLASSES; k++) {
            int nk = class_counts[k];
            if (nk <= 0) continue;
            for (int h = 0; h < H_local; h++)
                for (int v = 0; v < V; v++)
                    target[TGT_IDX(k, h, v, H_local, V)] =
                        (COUNTER_TYPE)(w0_random() % (uint32_t)(nk + 1));
        }
    } else if (aa.target_init_mode == KI_TARGET_UNIFORM) {
        /* Uniform: all raw counts = 1 (constant, no per-class or
         * per-neuron variation). After logit: same logit for all
         * entries — only the class prior n_k differentiates. */
        for (size_t _i = 0; _i < sz; _i++)
            target[_i] = 1;
    } else if (aa.target_init_mode == KI_TARGET_PRIOR) {
        /* Prior: per-class constant = class_count[k] for all (h,v).
         * After logit: class-specific constant logit, no per-neuron
         * variation — tests whether per-neuron structure matters. */
        for (int k = 0; k < KI_NCLASSES; k++) {
            int nk = class_counts[k];
            if (nk <= 0) continue;
            for (int h = 0; h < H_local; h++)
                for (int v = 0; v < V; v++)
                    target[TGT_IDX(k, h, v, H_local, V)] = (COUNTER_TYPE)nk;
        }
    } else if (aa.target_init_mode == KI_TARGET_INVERSE) {
        /* Inverse: count-mode raw counts (same as count below), then
         * logits are NEGATED after logit_convert in the caller.
         * This works because inverse in logit-space = -count_logit,
         * and using count-mode raw counts keeps logit_convert() /
         * compute_class_offset() valid (avoids t > nk overflow). */
        #pragma omp parallel
        {
            int32_t *lt = (int32_t *)ki_xcalloc(sz, sizeof(COUNTER_TYPE));
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
            { for (size_t i = 0; i < sz; i++) target[i] += (COUNTER_TYPE)lt[i]; }
            free(lt);
        }
    } else if (aa.target_init_mode == KI_TARGET_LAPLACE) {
        /* Laplace: count-mode raw counts, then +1 per entry (additive smoothing).
         * Clamped to n_k to avoid p = 1 overflow in logit_convert. */
        #pragma omp parallel
        {
            int32_t *lt = (int32_t *)ki_xcalloc(sz, sizeof(COUNTER_TYPE));
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
            { for (size_t i = 0; i < sz; i++) target[i] += (COUNTER_TYPE)lt[i]; }
            free(lt);
        }
        /* Laplace +1: add 1 to each (k, h, v) but never exceed n_k. */
        for (int k = 0; k < KI_NCLASSES; k++) {
            int nk = class_counts[k];
            if (nk <= 0) continue;
            for (int h = 0; h < H_local; h++)
                for (int v = 0; v < V; v++) {
                    size_t idx = TGT_IDX(k, h, v, H_local, V);
                    if (target[idx] < nk) target[idx]++;
                }
        }
    } else if (aa.target_init_mode == KI_TARGET_DAMPEN) {
        /* Dampen: count-mode raw counts, then right-shift by 1 (÷2).
         * Preserves the "mountain range" shape but halves peak/valley
         * amplitude — initial log-odds are less extreme. */
        #pragma omp parallel
        {
            int32_t *lt = (int32_t *)ki_xcalloc(sz, sizeof(COUNTER_TYPE));
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
            { for (size_t i = 0; i < sz; i++) target[i] += (COUNTER_TYPE)lt[i]; }
            free(lt);
        }
        /* Dampen: divide all raw counts by 2. */
        for (size_t _i = 0; _i < sz; _i++) {
            int32_t _tmp = (int32_t)(target[_i]);
            target[_i] = (COUNTER_TYPE)(_tmp >> 1);
        }
    } else {
        #pragma omp parallel
        {
            int32_t *lt = (int32_t *)ki_xcalloc(sz, sizeof(COUNTER_TYPE));
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
            { for (size_t i = 0; i < sz; i++) target[i] += (COUNTER_TYPE)lt[i]; }
            free(lt);
        }
    }
    return target;
}


/* ═══════════════════════════════════════════════════════════════════
 * LOGIT CONVERT — Target-Counts → log-odds (in-place)
 * ═══════════════════════════════════════════════════════════════════
 *   target[k][h][b] = round( ln((t+1)/(N_k-t+1)) × F )
 *   mit F = (1<<OT_PRECISION) (default 10 → F=1024)
 *
 * Dependency: ot_precision defines the int32 scaling.
 */
static void logit_convert(COUNTER_TYPE *target, int H_local, const int class_counts[KI_NCLASSES]) {
    int V = VN_GROUPS_;
    for (int k = 0; k < KI_NCLASSES; k++) {
        int nk = class_counts[k];
        if (nk <= 0) continue;
        for (int h = 0; h < H_local; h++) {
            for (int v = 0; v < V; v++) {
                size_t idx = TGT_IDX(k, h, v, H_local, V);
                int t = (int)target[idx];
                double p = (double)(t + 1) / (double)(nk + 2);
                target[idx] = (COUNTER_TYPE)ot_precision(log(p / (1.0 - p)));
            }
        }
    }
}

/* ── VN_SCORE_FROM_GB — Use precomputed gb mask (scores_otto_from_gb uses this) ── */
#define VN_SCORE_FROM_GB(gb, h, H, NG, TGT, SC) do { \
    uint32_t _b = (gb); \
    while (_b) { int _v = __builtin_ctz(_b); \
        for (int _k = 0; _k < KI_NCLASSES; _k++) \
            (SC)[_k] += (TGT)[TGT_IDX(_k, (h), _v, H, NG)]; \
        _b &= _b - 1; } \
} while (0)

/* ki_batch_correct comes from ki-train.h (shared with bitvoting) */

/* ═══════════════════════════════════════════════════════════════════
 * CLASS OFFSET — Σ log(1-P_k) × F  (F = 1<<OT_PRECISION)
 * ═══════════════════════════════════════════════════════════════════
 * Must be computed BEFORE logit_convert (needs raw counts).
 * Gemeinsame Skalierung: target und offset teilen F.
 *
 * Note: +-0.5 rounding removed because log(p1)*F is always negative
 * and the 0.5 correction (~0.001%) is below measurement noise (±0.3pp).
 * See: 2026-06-18 experiment — 3 runs with/without 0.5 gave identical
 * results within normal OpenMP scheduling noise.
 */
static void compute_class_offset(SCORE_TYPE class_offset[KI_NCLASSES],
                                  const COUNTER_TYPE *target, int H_local,
                                  const int class_counts[KI_NCLASSES]) {
    int V = VN_GROUPS_;
    for (int k = 0; k < KI_NCLASSES; k++) {
        SCORE_TYPE sum = 0;
        int nk = class_counts[k];
        if (nk <= 0) { class_offset[k] = (SCORE_TYPE)0; continue; }
        for (int h = 0; h < H_local; h++) {
            for (int v = 0; v < V; v++) {
                int t = (int)target[TGT_IDX(k, h, v, H_local, V)];
                double p1 = (double)(nk - t + 1) / (double)(nk + 2);
                sum += (SCORE_TYPE)ot_precision(log(p1));
            }
        }
        class_offset[k] = sum;
    }
}
/* ═══════════════════════════════════════════════════════════════════
 * SCORE — Bayes log-Score (mit Slice)
 * in: Input shifted to member slice)
 * W0:    Member-W0 [H_local × NC_slice]
 * H_local, NC_slice, nc_off: Slice-Parameter
 */
static void scores_otto(const uint32_t *in, const uint32_t *W0,
                         int H_local, int NC_slice,
                         const COUNTER_TYPE *target,
                         const SCORE_TYPE class_offset[KI_NCLASSES],
                         SCORE_TYPE scores[KI_NCLASSES]) {
    for (int k = 0; k < KI_NCLASSES; k++)
        scores[k] = (SCORE_TYPE)class_offset[k];

    int _ws2 = aa.rows_mode && aa.maj_mode == KI_MAJ_1 ? NC_slice * 4 : NC_slice;
    for (int h = 0; h < H_local; h++) {
        uint32_t h0 = h0_neuron(in, W0 + (size_t)h * _ws2, NC_slice);
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

/* ── scores_otto from precomputed VN group mask (no h0_neuron) ── *
 * Used for training eval (gb_buf is computed once).
 * Takes individual fields instead of ki_Member* (struct definition comes later).
 * For test eval without cache there is scores_otto (oben). */
static void scores_otto_from_gb(int s, int H_local,
                                 const uint32_t *gb_buf,
                                 const COUNTER_TYPE *target,
                         const SCORE_TYPE class_offset[KI_NCLASSES],
                                 SCORE_TYPE scores[KI_NCLASSES]) {
    const uint32_t *gb_row = gb_buf + (size_t)s * (size_t)H_local;
    for (int k = 0; k < KI_NCLASSES; k++)
        scores[k] = (SCORE_TYPE)class_offset[k];

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
 * EVALUATE — Members outer, samples inner (cache-optimal)
 * ═══════════════════════════════════════════════════════════════════
 *
 * Members serial outer.*stays in L1 cache for
 * all N samples. Reduces D1mr from 55% to <1% (previously: samples
 * outer → each member switch evicted target from cache).
 *
 * Uses ki_Member structs directly (no more flat arrays).
 * Each member has its own W0, target, offset, slc_off.
 *
 * Votes buffer.*for.*samples).
 * n_cont:   Containers per sample (NC, for stride)
 * Returns:  number of korrekt klassifizierte Samples
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
                          const COUNTER_TYPE *target,
                          const SCORE_TYPE class_offset[KI_NCLASSES]) {
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
    size_t w1_bit = (size_t)KI_NCLASSES * (size_t)H_local * (size_t)V * sizeof(COUNTER_TYPE) * 8;
    int n_xf_active = 0;
    for (int _x = 0; _x < KI_XFORM_COUNT; _x++)
        if (aa.xforms & (1 << _x)) n_xf_active++;
    if (n_xf_active < 1) n_xf_active = 1;
    int total_slots = ensembleN * n_xf_active * eff_colors * splitHN;    /* VN no longer multiplies members */
    size_t tgt_total = (size_t)H_local * KI_NCLASSES * (size_t)V * (size_t)total_slots;
    printf("══════════════════════════════════════════════════════════════════════\n");
    printf("══╡ OTTO-SCORE ╞══  %s  %s\n", KI_DATASET_NAME, H0_STR);
    printf("  Args:        H=%d  B=%d  Ep=%d  NC=%d V=%d  HN=%d  H_sub=%d  NC_sub=%d  Maj=%s\n",
           H, batchN, epochs, nc_per_blk, V, splitHN, H_local, NC_slice, maj_str());
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
    if (n_xf_active > 1) {
        printf("  TOTAL:       (W0+W1) x (EN[%d]×XF[%d]×CO[%d]×HN[%d]=%d) = %9zu bit  (%5.1f KB)\n",
               ensembleN, n_xf_active, eff_colors, splitHN, total_slots,
               (w0_bit + w1_bit) * (size_t)total_slots,
               (double)((w0_bit + w1_bit) * (size_t)total_slots) / 8 / 1024);
    } else {
        printf("  TOTAL:       (W0+W1) x (EN[%d]×CO[%d]×HN[%d]=%d) = %9zu bit  (%5.1f KB)\n",
               ensembleN, eff_colors, splitHN, total_slots,
               (w0_bit + w1_bit) * (size_t)total_slots,
               (double)((w0_bit + w1_bit) * (size_t)total_slots) / 8 / 1024);
    }
    printf("                                                + %zu KB target/offset (all members)\n",
           tgt_total / 1024);
    printf("  OMP:         %d threads\n", threadN);
    {
        const char *maj_name;
        switch (aa.maj_mode) {
            case KI_MAJ_1:   maj_name = "container flat"; break;
            case KI_MAJ_1R:  maj_name = "container row"; break;
            case KI_MAJ_1P:  maj_name = "pixel flat"; break;
            case KI_MAJ_1RP: maj_name = "pixel+row"; break;
            case KI_MAJ_3:   maj_name = "3-tree"; break;
            case KI_MAJ_7:   maj_name = "7-tree"; break;
            default:         maj_name = "?"; break;
        }
        printf("  Majority:    %s (%d)\n", maj_name, aa.maj_mode);
    }
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
    if (aa.filter_mask) {
        printf("  filter:");
        for (int _k = 0; _k < KI_NCLASSES; _k++)
            if ((aa.filter_mask >> _k) & 1)
                printf(" %s(%d)", ki_class_names[_k], _k);
    }
    if (aa.export_merge_scores[0])
        printf("  Save-scores: %s\n", aa.export_merge_scores);
    /* ── Show xforms if more than identity ── */
    if (n_xf_active > 1) {
        printf("\n  ───────────────────────────────────────────────────────────\n");
        printf("  Xform:       %s  (%d× ensemble multiplier)\n", xform_str(), n_xf_active);
    }
    printf("\n");
}


/* Forward declaration */
static const char *opp_name(int ch);

/* ═══════════════════════════════════════════════════════════════════════
 * PRINT MEMBER STRUCTURE — zeigt Grid + Per-Member + H/C-Struktur
 * ═══════════════════════════════════════════════════════════════════════
 * Called by dry-run and main.*no data dependency.
 */
static void print_member_structure(int ensembleN, int splitVN, int splitHN,
                                    int H_local, int NC_slice, int channel,
                                    int n_xforms_eff) {
    int rows_factor = aa.rows_mode ? KI_ROWS : 1;
    int total = ensembleN * n_xforms_eff * eff_colors * splitHN * rows_factor;
    (void)splitVN;
    printf("\n══╡ MEMBER ╞══════════════════════════════════════════════════\n");
    if (n_xforms_eff > 1)
        printf("  Grid: ENSEMBLE[%d] × XFORM[%d] × COLOR[%d] × HN[%d] × ROW[%d] = %d members\n",
               ensembleN, n_xforms_eff, eff_colors, splitHN, rows_factor, total);
    else
        printf("  Grid: ENSEMBLE[%d] × COLOR[%d] × HN[%d] × ROW[%d] = %d members\n",
               ensembleN, eff_colors, splitHN, rows_factor, total);
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
            /* Multi-encoding: use enc_array index directly (not color match)
             * so same-channel members with different encodings display correctly */
            if (ci < aa.enc_count) {
                _t[_n] = (int)aa.enc_array[ci].type;
                _w[_n] = (int)aa.enc_array[ci].width;
            }
            _n++;
        }
    }
    ki_print_member_structure(_c, _t, _w, _n, ensembleN);
    if (ensembleN > 1) {
        if (aa.ensemble_seed == ENS_SEED_CONST) {
            printf("  → ENSEMBLE x%d: all channel members share W0 (const)\n",
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
 * AUTONOMOUS MEMBER — own memory, eigene Fehler, eigener Schritt
 * ═══════════════════════════════════════════════════════════════════
 *
 * Each member manages its own resources. Kein malloc/free
 * per epoch → no cache bouncing from allocator overhead.
 */
typedef struct ki_Member {
    /* Dimensionen (aus CLI, konstant) */
    int H_local;            /* Neurons (H, no vertical split) */
    int NC_slice;           /* Container (KI_NC / splitHN) */
    int w0_step;            /* W0 stride per neuron (uint32), = NC_slice * pixel_groups */
    int slc_off;            /* Input-Offset for this member */
    int vi;                 /* Encoding-Index in enc_array (for stats/debug) */
    int xform_id;           /* Active xform index (for debug) */
    int color_bit;          /* Color channel bit (COLOR_R, etc., for debug) */

    /* Pointer to external data (Member owns target+offset, shares W0) */
    const uint32_t *W0;     /* W0 row start (geteilt oder eigen) */
    const uint32_t *input_buf;    /* X buffer for this member's xform (train) */
    const uint32_t *input_buf_te; /* X buffer for this member's xform (eval) */
    COUNTER_TYPE *target;   /* [H_local × KI_NCLASSES × V] — own memory */
    SCORE_TYPE *offset;        /* [KI_NCLASSES] SCORE_TYPE — own memory */

    /* Training buffers.*allocated once.*reused each epoch) */
    uint32_t *h0_buf;       /* [total_train × H_local] — rohe H0-Werte */
    uint32_t *gb_buf;       /* [total_train × H_local] — VN-Gruppenmaske (h0_to_gb) */
    int orig_m;             /* Index im Export (for correct ordering) */
    uint32_t *gb_buf_te;    /* [total_eval × H_local] — eval cache.*NULL if no eval) */

    /* Best-State (for export bei bestem eval) */
    COUNTER_TYPE *best_target;  /* [H_local × KI_NCLASSES × V] — Snapshot bei bestem eval */
    SCORE_TYPE *best_offset;   /* [KI_NCLASSES] SCORE_TYPE — Snapshot bei bestem eval */
    float   fin_evl;       /* eval bei bestem Snapshot */

    /* Err state.*for rollback.*only when aa.err_rollback) */
    COUNTER_TYPE *err_target;   /* [H_local × KI_NCLASSES × V] — Snapshot bei bestem train-err */
    SCORE_TYPE *err_offset;    /* [KI_NCLASSES] SCORE_TYPE — Snapshot bei bestem train-err */

    /* Member-Zustand (per epoch aktualisiert) */
    int step;               /* aktueller Schritt */
    int last_err;           /* letzter Fehler */
    int ep;                 /* own epoch (for cosine) */
    int evl_ok;             /* eval correct count (--debug) */
    float trn_acc;          /* training accuracy after last epoch (0..100), for --member-threshold */
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
                fwrite(members[b]->target, sizeof(COUNTER_TYPE), (size_t)H_local * KI_NCLASSES * VN_GROUPS_, f);
                fwrite(members[b]->offset, sizeof(SCORE_TYPE), KI_NCLASSES, f);
                found = 1;
            }
        }
        if (!found) {
            COUNTER_TYPE *zeros = (COUNTER_TYPE *)calloc((size_t)H_local * KI_NCLASSES * VN_GROUPS_, sizeof(COUNTER_TYPE));
            fwrite(zeros, sizeof(COUNTER_TYPE), (size_t)H_local * KI_NCLASSES * VN_GROUPS_, f);
            free(zeros);
            SCORE_TYPE *oz = (SCORE_TYPE *)calloc(KI_NCLASSES, sizeof(SCORE_TYPE));
            fwrite(oz, sizeof(SCORE_TYPE), KI_NCLASSES, f);
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
    m->w0_step  = NC_slice;  /* default: stride = containers (can be overridden for pixel-maj) */
    m->slc_off  = slc_off;
    m->W0       = W0;
    m->step     = 0;
    m->last_err = 0;
    m->ep       = 0;
    m->trn_acc  = 100.0f;  /* initially: all members participate */

    size_t tgt_sz = (size_t)H_local * KI_NCLASSES * VN_GROUPS_;
    m->target = (COUNTER_TYPE *)ki_xcalloc(tgt_sz, sizeof(COUNTER_TYPE));
    m->offset = (SCORE_TYPE *)ki_xcalloc(KI_NCLASSES, sizeof(SCORE_TYPE));
    m->best_target = (COUNTER_TYPE *)ki_xcalloc(tgt_sz, sizeof(COUNTER_TYPE));
    m->best_offset = (SCORE_TYPE *)ki_xcalloc(KI_NCLASSES, sizeof(SCORE_TYPE));
    m->fin_evl = 0.0f;
    if (aa.err_rollback) {
        m->err_target = (COUNTER_TYPE *)ki_xcalloc(tgt_sz, sizeof(COUNTER_TYPE));
        m->err_offset = (SCORE_TYPE *)ki_xcalloc(KI_NCLASSES, sizeof(SCORE_TYPE));
    } else {
        m->err_target = NULL;
        m->err_offset = NULL;
    }

    if (!aa.no_precompute) {
        size_t h0_sz = (size_t)total_train * (size_t)H_local;
        m->h0_buf = (uint32_t *)ki_xmalloc(h0_sz * sizeof(uint32_t));
        m->gb_buf = (uint32_t *)ki_xmalloc(h0_sz * sizeof(uint32_t));
        if (total_eval > 0) {
            size_t te_sz = (size_t)total_eval * (size_t)H_local;
            m->gb_buf_te = (uint32_t *)ki_xmalloc(te_sz * sizeof(uint32_t));
        } else {
            m->gb_buf_te = NULL;
        }
    } else {
        m->h0_buf = NULL;
        m->gb_buf = NULL;
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

/* ── Member: Test-Eval gb vorberechnen (once) ───────────────── *
 * Writes directly to gb_buf_te, no h0_buf needed (only gb is
 * needed for evaluate_member). */
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
                h0_neuron(in, m->W0 + (size_t)h * (size_t)m->w0_step, m->NC_slice));
        }
    }
}

/* ── Member: h0 + gb vorberechnen (once, jede Epoche wiederverwendet) ── *
 * gb_buf = VN group mask from h0 (vermeidet popcount reduction im Training). */
static void ki_member_compute_h0(ki_Member *m, const uint32_t *X, int N,
                                  int n_cont) {
    const uint32_t *in_base = X + (size_t)m->slc_off;
    #pragma omp parallel for schedule(static)
    for (int s = 0; s < N; s++) {
        const uint32_t *in = in_base + (size_t)s * (size_t)n_cont;
        for (int h = 0; h < m->H_local; h++) {
            size_t idx = (size_t)s * (size_t)m->H_local + (size_t)h;
            uint32_t hv = h0_neuron(in, m->W0 + (size_t)h * (size_t)m->w0_step, m->NC_slice);
            m->h0_buf[idx] = hv;
            m->gb_buf[idx] = h0_to_gb(hv);
        }
    }
}
static int export_merge_scores_archive(const char *out_dir, const uint32_t *X,
                                const uint8_t *y, int N,
                                ki_Member **members, int active_members,
                                int n_cont, int evl_ok)
{
    if (N <= 0 || active_members <= 0) return -1;

    /* Build archive filename from parameters */
    int te_int = 0;  /* target_err removed, kept 0 for archive format compat */
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
    float    tgt_err   = 0.0f;  /* target_err removed */
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
        SCORE_TYPE *sc = (SCORE_TYPE *)calloc(score_sz, sizeof(SCORE_TYPE));
        if (!sc) { fprintf(stderr, "[FATAL] export_merge_scores: OOM\n"); fclose(f); return -1; }

        #pragma omp parallel for schedule(static)
        for (int s = 0; s < N; s++) {
            SCORE_TYPE scc[KI_NCLASSES];
            scores_otto(X + (size_t)s * (size_t)n_cont + mem->slc_off,
                        mem->W0, mem->H_local, mem->NC_slice,
                        mem->target, mem->offset, scc);
            for (int k = 0; k < KI_NCLASSES; k++)
                sc[(size_t)s * KI_NCLASSES + (size_t)k] = scc[k];
        }

        fwrite(sc, sizeof(SCORE_TYPE), score_sz, f);
        free(sc);
    }

    /* Append ground truth labels (uint8[N]) */
    fwrite(y, 1, (size_t)N, f);

    fclose(f);
    printf("  Archive:  %s  (%d members x %d samples, %zu KB)\n",
           fname, active_members, N,
           (sizeof(SCORE_TYPE) * score_sz * (size_t)active_members + 44) / 1024);
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
 * For VN=1: gb = h0.*each bit = its own group).
 * For VN.*gb has one bit per group, set when popcount >
 *            N_BITS-1 (strict AND: all bits must be 1). */
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

static inline COUNTER_TYPE **ki_cache_alloc(int n_threads, size_t tgt_sz) {
    COUNTER_TYPE **cache = (COUNTER_TYPE **)malloc((size_t)n_threads * sizeof(COUNTER_TYPE *));
    if (!cache) { fprintf(stderr, "[FATAL] ki_cache_alloc(%d) failed\n", n_threads); exit(1); }
    for (int t = 0; t < n_threads; t++)
        cache[t] = (COUNTER_TYPE *)ki_xcalloc(tgt_sz, sizeof(COUNTER_TYPE));
    return cache;
}

static inline void ki_cache_apply_free(COUNTER_TYPE **cache, int n_threads,
                                        size_t tgt_sz, COUNTER_TYPE *target) {
    for (int t = 0; t < n_threads; t++) {
        COUNTER_TYPE *ct = cache[t];
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
 *   1. Parallel: compute scores from target, deltas in thread cache
 *   2. Sequentiell: apply deltas to target → next batch sees the change
 *
 * step size per sample:
 *   gap = sc[pred] - sc[true_k]   (Member-eigener Score-Abstand)
 *   gap > 0 → Korrektur, step skaliert proportional: step × gap / F
 *   gap ≥ F → vollen Schritt
 *   gap.*no update.*member was correct)
 *
 * target:     target.*with offset for ensemble)
 * H:          number of Neuronen
 * class_offset: Offset per class
 * gb_all:     Vorberechnete VN-Gruppenmasken [N × H] (aus h0_to_gb)
 * y:          Labels
 * N:          number of Trainings-Samples
 * step:       Basis-Schritt (Obergrenze, Member skaliert via gap)
 * tgt_sz:     size des Target-Arrays (H × KI_NCLASSES × V)
 *
 * Returns:    number of Korrekturen
 */
#include "../lib/ki-train.h"   /* shared batch_correct (ki_batch_correct) */

/* ── Member: batch correct (nutzt gb_buf + target + offset) ────────── *
 * gb_buf = precomputed VN group mask (no popcount reduction needed). */
static inline int ki_member_batch_correct(ki_Member *m, const uint8_t *y, int N, int step) {
    m->step = step;
    int err = ki_batch_correct(m->target, m->H_local, m->offset,
                                 m->gb_buf, y, N, (COUNTER_TYPE)step, (size_t)m->H_local * KI_NCLASSES * 32,
                                aa.filter_mask, m->H_local, 0);
    m->last_err = err;
    m->ep++;
    return err;
}

/* ═══════════════════════════════════════════════════════════════════
 * EVALUATE — Members outer, samples inner (cache-optimal)
 * ═══════════════════════════════════════════════════════════════════
 *
 * Members serial outer.*stays in L1 cache for
 * all N samples. Reduces D1mr from 55% to <1% (previously: samples
 * outer → each member switch evicted target from cache).
 *
 * Uses ki_Member structs directly (no more flat arrays).
 * Each member has its own W0, target, offset, slc_off.
 *
 * Votes buffer.*for.*samples).
 * n_cont:   Containers per sample (NC, for stride)
 * Returns:  number of korrekt klassifizierte Samples
 */
static int ki_evaluate_member(const uint32_t *X, const uint8_t *y, int N,
                               ki_Member **members, int active_members,
                               int n_cont, uint8_t *pred_out, int use_gb)
{
    if (N <= 0) return 0;

    /* Votes-Accumulator: N Samples × KI_NCLASSES Klassen */
    SCORE_TYPE (*votes)[KI_NCLASSES] = (SCORE_TYPE (*)[KI_NCLASSES])calloc((size_t)N, sizeof(SCORE_TYPE[KI_NCLASSES]));
    if (!votes) { fprintf(stderr, "[FATAL] evaluate: votes OOM\n"); exit(1); }

    /* Each member gets equal voting power: Scale sc.*so that
     * max|sc[k]| ≤ SCALE_MAX. Prevents members with large
     * target weights (more corrections, different channels) from dominating. */
    #if COUNTER_TYPE_IS_FLOAT
    #define VOTE_SCALE ((SCORE_TYPE)16777216.0f)
    #else
    #define VOTE_SCALE ((SCORE_TYPE)1 << 24)
    #endif

    /* Members outer: target.*stays warm in L1 cache */
    //#pragma omp parallel for schedule(static) if(active_members > 8)
    for (int m = 0; m < active_members; m++) {
        ki_Member *mem = members[m];

        /* ── Skip members below accuracy threshold ────────────── */
        if (aa.member_threshold > 0 && mem->trn_acc < (float)aa.member_threshold)
            continue;

        #pragma omp parallel for schedule(static)
        for (int s = 0; s < N; s++) {
            SCORE_TYPE sc[KI_NCLASSES];
            if (use_gb == 2) {
                /* ── Test eval: gb_buf_te is computed once and cached. */
                scores_otto_from_gb(s, mem->H_local, mem->gb_buf_te,
                                   mem->target, mem->offset, sc);
            } else if (use_gb == 1) {
                /* ── Training-Eval: gb_buf is computed once und gecached. */
                scores_otto_from_gb(s, mem->H_local, mem->gb_buf,
                                   mem->target, mem->offset, sc);
            } else {
                /* ── Rohpixel → h0_neuron (Test-Set ohne gb_buf) */
                scores_otto(X + (size_t)s * (size_t)n_cont + mem->slc_off,
                           mem->W0, mem->H_local, mem->NC_slice,
                           mem->target, mem->offset, sc);
            }

            /* Vote-Normalisierung: --optional target-norm
             * Each member is normalized to max.*VOTE_SCALE.
             * All members thus have equal voting power,
             * regardless of target size or channel. */
            if (aa.opt_target_norm) {
                SCORE_TYPE max_abs = 0;
                for (int k = 0; k < KI_NCLASSES; k++) {
                    SCORE_TYPE a = (SCORE_TYPE)((sc[k] >= 0) ? sc[k] : -sc[k]);
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

    /* Merge: argmax per sample (pred=-1 when all votes=0, counted as wrong) */
    int ok = 0;
    for (int s = 0; s < N; s++) {
        int pred = -1;
        for (int k = 0; k < KI_NCLASSES; k++)
            if ((votes[s][k] > 0 || votes[s][k] < 0) && (pred < 0 || votes[s][k] > votes[s][pred]))
                pred = k;
        if (pred_out) pred_out[s] = (uint8_t)(pred >= 0 ? pred : 0);
        if (pred >= 0 && pred == (int)y[s]) ok++;
    }
    free(votes);
    return ok;
}

/* ═══════════════════════════════════════════════════════════════════
 * Per-member debug stats.*table after each epoch
 * ═══════════════════════════════════════════════════════════════════
 * Active only with --debug.*Shows per member:
 *   Channel=Encoding · Target min/max · Step · last_err · Independent eval
 * Shows which members help or hurt, which members help or hurt.
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
            tgt_min = tgt_max = (int)mem->target[0];
            for (size_t i = 1; i < tgt_sz; i++) {
                int v = (int)mem->target[i];
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

        /* ── Independent eval (this member only) ───────────── */
        int member_ok = 0;
        if (N > 0) {
            SCORE_TYPE *sc = (SCORE_TYPE *)calloc((size_t)N * KI_NCLASSES, sizeof(SCORE_TYPE));
            if (sc) {
                for (int s = 0; s < N; s++) {
                    SCORE_TYPE scc[KI_NCLASSES];
                    scores_otto(X + (size_t)s * (size_t)n_cont + mem->slc_off,
                                mem->W0, mem->H_local, mem->NC_slice,
                                mem->target, mem->offset, scc);
                    /* Vote ohne target-norm (wie Default) */
                    for (int k = 0; k < KI_NCLASSES; k++)
                        sc[(size_t)s * KI_NCLASSES + (size_t)k] = scc[k];
                }
                for (int s = 0; s < N; s++) {
                    int pred = 0;
                    SCORE_TYPE *row = sc + (size_t)s * KI_NCLASSES;
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
 * Active only with --debug-class-voting.*Shows per member and class
 * how often the member was correct (pred == y[s]) divided by
 * the number of samples of this class.
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

    /* ── First pass: count samples per class ────────────────── */
    for (int s = 0; s < N; s++) {
        int k = (int)y[s];
        if (k >= 0 && k < KI_NCLASSES) total[k]++;
    }

    /* ── Zweiter Pass: per member Scores berechnen, argmax, vergleich ── */
    for (int m = 0; m < active_members; m++) {
        ki_Member *mem = members[m];
        for (int s = 0; s < N; s++) {
            SCORE_TYPE sc[KI_NCLASSES];
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
    /* Only classes with samples anzeigen (important with --filter) */
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
                           uint32_t **W0_out, COUNTER_TYPE **tgt_out, SCORE_TYPE **off_out,
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
    *tgt_out = (COUNTER_TYPE *)malloc((size_t)n_members * tgt_msz * sizeof(COUNTER_TYPE));
    *off_out = (SCORE_TYPE *)calloc((size_t)n_members * off_msz, sizeof(SCORE_TYPE));
    if (!*W0_out || !*tgt_out || !*off_out) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
    /* Datei-Layout (wie export_ensemble): per-member interleaved
     *   W0_m0  TGT_m0  OFF_m0  W0_m1  TGT_m1  OFF_m1  ...
     * Import must read in the same loop. */
    for (uint32_t m = 0; m < n_members; m++) {
        if (fread(*W0_out + (size_t)m * w0_msz, sizeof(uint32_t), w0_msz, f) != w0_msz ||
            fread(*tgt_out + (size_t)m * tgt_msz, sizeof(COUNTER_TYPE), tgt_msz, f) != tgt_msz ||
            fread(*off_out + (size_t)m * off_msz, sizeof(SCORE_TYPE), off_msz, f) != off_msz) {
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

/* Input cache functions (load_input_cached, load_input_cached_xform) are
 * defined in lib/ki-load.h — shared by Otto and bitvoting trainers. */


/* ═══════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[]) {
    /* Filter out --debug-epoch before ki_parse_args */
    const char **debug_av = (const char **)malloc((size_t)(argc + 1) * sizeof(char *));
    int debug_ac = 0;
    debug_av[debug_ac++] = argv[0];
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--debug-epoch") == 0) { debug_epoch = 1; }
        else { debug_av[debug_ac++] = argv[i]; }
    }
    debug_av[debug_ac] = NULL;
    aa.lr_step = (int)round(aa.lr * (1<<OT_PRECISION));
    ki_parse_args(debug_ac, (char **)debug_av);
    free(debug_av);
    aa.no_precompute = 1;  /* sequential: compute gb per-member */
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
     * active_chans stores the bit position from enum ki_color_bit. */
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
    /* Multi-encoding: enc_array entries = virtual blocks */
    if (aa.enc_count > 1) {
        eff_colors = aa.enc_count;
        /* fill active_chans with colors from enc_array */
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
    /* SplitHN check: each enc_array entry must be divisible */
    for (int i = 0; i < aa.enc_count && i < KI_ENC_MAX; i++) {
        int w = (int)aa.enc_array[i].width;
        if (w < 1) w = KI_ENC_WIDTH_DEFAULT;
        int ncc = (KI_COLORS > 1) ? KI_NC * w / KI_BIT_WIDTH : NC * w / KI_BIT_WIDTH;
        if (ncc % splitHN != 0) {
            fprintf(stderr, "[FATAL] NC=%d not divisible by splitHN=%d\n", ncc, splitHN);
            fprintf(stderr, "  Valid splitHN values (divisors of %d): ", ncc);
            for (int d = 1; d <= ncc; d++)
                if (ncc % d == 0) fprintf(stderr, "%s%d", (d == 1) ? "" : ", ", d);
            fprintf(stderr, "\n");
            return 1;
        }
    }
    /* nc_blk.*widest block.*for step validation) */
    int nc_blk = 0;
    for (int i = 0; i < aa.enc_count && i < KI_ENC_MAX; i++) {
        int w = (int)aa.enc_array[i].width;
        if (w < 1) w = KI_ENC_WIDTH_DEFAULT;
        int ncc = (KI_COLORS > 1) ? KI_NC * w / KI_BIT_WIDTH : NC * w / KI_BIT_WIDTH;
        if (ncc > nc_blk) nc_blk = ncc;
    }
    if (nc_blk == 0) nc_blk = (KI_COLORS > 1 ? KI_NC : NC) * KI_ENC_WIDTH_DEFAULT / KI_BIT_WIDTH;

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
    fflush(stdout);
    /* NOTE: --filter affects training only.
     * Evaluation always uses ALL classes.  No
     * ki_filter_dataset() needed — training skips
     * non-matching samples in ki_batch_correct(). */
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
    /* Upper bound: use dataset's n_train+n_eval (handles dry-run where
     * num_images = training set only, but n_eval is still set) */
    int total_max  = data.n_train > 0 ? data.n_train + data.n_eval : data.num_images;
    if (total_train + total_eval > total_max) {
        fprintf(stderr, "  [WARN] Requested %d+%d=%d > %d available, adjusting eval\n",
                total_train, total_eval, total_all, total_max);
        total_eval = total_max - total_train;
        if (total_eval < 0) { total_eval = 0; total_train = total_max; }
        if (total_eval == 0 && total_train < total_max) {
            total_train = total_max / 2; /* fallback */
            total_eval  = total_max - total_train;
        }
        total_all = total_train + total_eval;
    }

    /* ── Compute total stride = sum over all enc_array entries ── */
    size_t n_cont;
    if (aa.enc_count > 0) {
        n_cont = 0;
        for (int i = 0; i < aa.enc_count; i++) {
            int w = (int)aa.enc_array[i].width;
            if (w < 1) w = KI_ENC_WIDTH_DEFAULT;
            n_cont += (size_t)((KI_COLORS > 1 ? KI_NC : NC) * w / KI_BIT_WIDTH);
        }
    } else {
        /* Default single encoding: NC containers (KI_ENC_WIDTH_DEFAULT bits/px) */
        int w = (aa.enc_default_width > 0) ? (int)aa.enc_default_width : KI_ENC_WIDTH_DEFAULT;
        n_cont = (size_t)((KI_COLORS > 1 ? KI_NC : NC) * w / KI_BIT_WIDTH);
    }
    /* ── Pixel-data-dependent init (skipped for dry-run) ────── */
    uint32_t *X_all = NULL;
    uint32_t *X_flat_free = NULL;
    uint32_t *X_perm = NULL;
    uint8_t  *y_perm = NULL;
    int own_eval_data = 0;
    if (!aa.dry_run) {
        /* Only load X_all if identity (id) is in the xform list */
        int _need_identity = (aa.xform_list_count == 0);
        if (!_need_identity) {
            for (int _li = 0; _li < aa.xform_list_count; _li++)
                if (aa.xform_list[_li] == KI_XFORM_ID) { _need_identity = 1; break; }
        }
        if (_need_identity)
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
                int ncb = KI_NC * w / KI_BIT_WIDTH;
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
                    int ncb = KI_NC * w / KI_BIT_WIDTH;
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

    /* ── Xform input buffers — per-xform containers for member training ──
     * Each active xform produces a separate encoded buffer from transformed
     * raw pixels.  Members pick their buffer via input_buf/input_buf_te. */
    uint32_t *X_xform[KI_XFORM_COUNT] = {NULL};
    uint32_t *X_xform_te[KI_XFORM_COUNT];  /* non-owning views into train/eval splits */
    memset(X_xform_te, 0, sizeof(X_xform_te));
    int n_xforms_eff = aa.xform_list_count;
    if (n_xforms_eff < 1) {
        /* Default: identity only */
        n_xforms_eff = 1;
        aa.xform_list[0] = KI_XFORM_ID;
        aa.xform_list_count = 1;
    }
    if (!aa.dry_run) {
        int img_size = data.rows * data.cols;        /* pixels per plane */
        int channels = KI_COLORS;                     /* 3 for CIFAR, 1 for MNIST */
        /* Load each UNIQUE xform once from the list (duplicates share data) */
        for (int li = 0; li < n_xforms_eff; li++) {
            int xf = aa.xform_list[li];
            if (X_xform[xf]) continue;  /* already loaded */
            if (xf == KI_XFORM_ID) {
                X_xform[xf] = X_all;  /* identity reuses main buffer */
            } else {
                uint8_t *raw_xform = (uint8_t *)ki_xmalloc((size_t)total_all * (size_t)img_size * (size_t)channels);
                for (int s = 0; s < total_all; s++) {
                    ki_xform_raw(raw_xform + (size_t)s * (size_t)img_size * (size_t)channels,
                                data.X_raw + (size_t)s * (size_t)KI_PX,
                                data.cols, data.rows, channels, xf);
                }
                X_xform[xf] = load_input_cached_xform(xf, raw_xform, total_all, n_cont);
                free(raw_xform);
            }
        }
    }
    /* Build eval pointers for each xform (split train/eval within each buffer) */
    for (int xf = 0; xf < KI_XFORM_COUNT; xf++)
        X_xform_te[xf] = X_xform[xf] ? X_xform[xf] + (size_t)total_train * n_cont : NULL;

    /* ── Compute per-block nc and offsets from enc_array ── */
    int multi_enc_blk_off[KI_ENC_MAX] = {0};
    int multi_enc_nc[KI_ENC_MAX] = {0};
    {   int off = 0;
        for (int i = 0; i < aa.enc_count && i < KI_ENC_MAX; i++) {
            int w = (int)aa.enc_array[i].width;
            if (w < 1) w = KI_ENC_WIDTH_DEFAULT;
            multi_enc_nc[i] = (KI_COLORS > 1 ? KI_NC : NC) * w / KI_BIT_WIDTH;
            multi_enc_blk_off[i] = off;
            off += multi_enc_nc[i];
        }
    }

    /* ── Compute cont_per_row and rows factor ────────────────── */
    /* cont_per_row = containers per image row = KI_COLS * width / 32 */
    /* Derived from nc_blk / KI_ROWS since nc_blk = KI_ROWS * cont_per_row */
    int cont_per_row = nc_blk / KI_ROWS;
    if (cont_per_row < 1) cont_per_row = 1;
    int rows_factor = aa.rows_mode ? KI_ROWS : 1;

    /* ── Compute slice dimensions ────────────────────────────── */
    int H_local   = H;
    int NC_slice  = aa.rows_mode ? cont_per_row : (nc_blk / splitHN);  /* base slice */
    int total_members = ensembleN * n_xforms_eff * splitHN * eff_colors * rows_factor;

    /* ── Default W0 source: splitmix64 PRNG.*no more auto search) ─── */
    /* --seed-file override → w0_rand_set_file() in W0 init.
     * With seed_splitmix=1.*splitmix64 is always used. */

    /* ── IFC MODE: --import → evaluieren statt trainieren ───────────── */
    if (aa.importD[0]) {
        printf("\n══╡ INFERENCE ╞══════════════════════════════════════════════════\n");
        char model_path[1024];
        snprintf(model_path, sizeof(model_path), "%s/model.otto", aa.importD);
        uint32_t *W0_ifc; COUNTER_TYPE *tgt_ifc; SCORE_TYPE *off_ifc;
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
                   (size_t)hl_ifc * K * V * sizeof(COUNTER_TYPE));
            memcpy(mems[i]->offset, off_ifc + (size_t)i * K, (size_t)K * sizeof(SCORE_TYPE));
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
        ki_report_show(0, 0, evl_ok, total_eval, el, aa.threadN, 0, 0.0f, n_mifc);
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
        for (int _xf = 0; _xf < KI_XFORM_COUNT; _xf++) {
            if (_xf == KI_XFORM_ID) continue;
            if (X_xform[_xf] && X_xform[_xf] != X_all) free(X_xform[_xf]);
        }
        ki_dataset_free(&data); free(X_all);
        if (X_perm) { free(X_perm); free(y_perm); }
        return 0;
    }

    print_setup(H, aa.epochs, total_train, total_eval, aa.threadN, aa.seed, aa.batchN,
                splitVN, splitHN, NC_slice, H_local, ensembleN, aa.channel, nc_blk, (int)n_cont);

    /* ── W0: random uint32[total_members][H_local][NC_slice*pixel_groups] (frozen) ── */
    int uses_pixel = (aa.maj_mode == KI_MAJ_1P || aa.maj_mode == KI_MAJ_1RP);
    int pixel_groups = uses_pixel ? KI_PIXEL_GROUPS : 1;
    size_t w0_m_sz = (size_t)H_local * (size_t)NC_slice * (size_t)pixel_groups;
    size_t w0_sz = (size_t)total_members * w0_m_sz;
    uint32_t *W0_ens = (uint32_t *)ki_xmalloc(w0_sz * sizeof(uint32_t));

    /* Transparent: w0_random() liest aus Datei (falls --seed-file),
     * otherwise aus splitmix64 PRNG.  Die member-strategy (const/incr/once)
     * works identically in both modes. */
    if (aa.seed_file[0])
        w0_rand_set_file(aa.seed_file);

    if (aa.ensemble_seed == ENS_SEED_CONST) {
        /* const: each ensemble+xform gets its own W0 chunk,
         * all channel+hn members share it. */
        int memb_per_ens = n_xforms_eff * eff_colors * splitHN;
        for (int e = 0; e < ensembleN; e++) {
            for (int xf_idx = 0; xf_idx < n_xforms_eff; xf_idx++) {
                w0_srandom((unsigned int)(aa.seed + e * n_xforms_eff + xf_idx));
                for (size_t i = 0; i < w0_m_sz; i++) {
                    uint32_t v = w0_random();
                    for (int mm = 0; mm < eff_colors * splitHN; mm++) {
                        int m = e * memb_per_ens + xf_idx * (eff_colors * splitHN) + mm;
                        W0_ens[(size_t)m * w0_m_sz + i] = v;
                    }
                }
            }
        }
    } else if (aa.ensemble_seed == ENS_SEED_INCR) {
        /* incr: Each member gets own seed → anderen W0 */
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

    int class_counts[KI_NCLASSES] = {0};
    if (!aa.dry_run)
        for (int s = 0; s < total_train; s++)
            class_counts[(int)y_tr[s]]++;

    print_member_structure(ensembleN, splitVN, splitHN, H_local, NC_slice, aa.channel, n_xforms_eff);
    /* Target is built from gb_buf AFTER h0-compute (s.u.). */
    printf("\n");
    fflush(stdout);

    {
        int step = (int)(aa.lr * (float)OT_F + 0.5f);
        int n_xf_active = 0;
        for (int _x = 0; _x < KI_XFORM_COUNT; _x++)
            if (aa.xforms & (1 << _x)) n_xf_active++;
        printf("══╡ TRAINING ╞══  lr=%.4f  step=%d  mode=%s  F=%d",
             (double)aa.lr, step, mode_str(), OT_F);
        printf("  tgt-init=%s", target_init_str());
        printf("  multi-correct=%s", aa.multi_correct ? "on" : "off");
        if (aa.opt_target_norm)
            printf("  tgt-nrm=%d", aa.opt_target_norm ? 1 : 0);
        if (aa.gap_k > 0.0f)
            printf("  gap-k=%.1f", (double)aa.gap_k);
        if (aa.member_threshold > 0)
            printf("  mth=%d", aa.member_threshold);
        if (n_xf_active > 1)
            printf("  xform=%s", xform_str());
        printf("\n");
        fflush(stdout);
    }
    if (aa.dry_run) {
        printf("\n  (dry-run, exiting)\n");
        free(W0_ens);
        ki_dataset_free(&data);
        return 1;  /* INTENTIONAL: non-zero so run-research.sh suppresses logging */
    }
    struct timeval _tv0, _tv1, _target0, _target1, _logit0, _logit1, _total0, _total1;
    gettimeofday(&_total0, NULL);

    /* ── Iterative target tuning ──────────────────────────────────── */
    COUNTER_TYPE step_init = (aa.lr > 0) ? (COUNTER_TYPE)ot_precision(aa.lr) : (COUNTER_TYPE)aa.lr_step;
    int epochs = aa.epochs;

    /* ── Create member array: each member manages itself ─── */
    /* ── Active member count (accounting for row-mode) ───── */
    int active_members = 0;
    int base_total = aa.rows_mode ? (total_members / KI_ROWS) : total_members;
    for (int bm = 0; bm < base_total; bm++) {
        if (!(aa.channel >= 0 && !(aa.channel & (1 << active_chans[(bm / splitHN) % eff_colors]))))
            active_members += rows_factor;  /* all rows of this base member */
    }
    ki_Member **members = (ki_Member **)malloc((size_t)active_members * sizeof(ki_Member *));
    if (!members) { fprintf(stderr, "[FATAL] members OOM\n"); return 1; }

    /* ── Create members with optional row expansion ──────── */
    {
        int mem_idx = 0;
        for (int bm = 0; bm < base_total; bm++) {
            int vi = (bm / splitHN) % eff_colors;  /* virtual block index */
            int xf_idx = (bm / (eff_colors * splitHN)) % n_xforms_eff;
            int color = (KI_COLORS > 1) ? active_chans[vi] : COLOR_MNIST;
            if (aa.channel >= 0 && !(aa.channel & (1 << color)) && KI_COLORS > 1) continue;

            /* ── Base container offset (for this color/encoding) ── */
            int base_slc_off;
            int base_mem_nc;
            if (aa.debug_flat) {
                base_mem_nc = nc_blk / splitHN;
                base_slc_off = (bm % splitHN) * base_mem_nc;
            } else if (aa.enc_count > 0) {
                base_mem_nc = multi_enc_nc[vi] / splitHN;
                base_slc_off = multi_enc_blk_off[vi] + (bm % splitHN) * base_mem_nc;
            } else {
                base_mem_nc = nc_blk / splitHN;
                base_slc_off = (bm % splitHN) * base_mem_nc;
            }

            /* ── Iterate over rows (1 for flat, KI_ROWS for row-mode) ── */
            for (int r = 0; r < rows_factor; r++) {
                int mem_nc   = aa.rows_mode ? cont_per_row : base_mem_nc;
                int slc_off  = aa.rows_mode
                               ? (base_slc_off + r * cont_per_row)
                               : base_slc_off;

                int m_idx = mem_idx;  /* unique linear index */
                const uint32_t *W0_m = W0_ens + (size_t)m_idx * w0_m_sz;
                members[mem_idx] = ki_member_create(H_local, mem_nc, slc_off,
                                                W0_m, total_train, total_eval);
                members[mem_idx]->w0_step = mem_nc * pixel_groups;
                int xf_id = aa.xform_list[xf_idx];
                members[mem_idx]->input_buf    = X_xform[xf_id];
                members[mem_idx]->input_buf_te = X_xform_te[xf_id];
                members[mem_idx]->orig_m = m_idx;
                members[mem_idx]->vi = vi;
                members[mem_idx]->xform_id = xf_id;
                members[mem_idx]->color_bit = color;
                members[mem_idx]->last_err = total_train;
                mem_idx++;
            }
        }
    }

    /* Best-Snapshots (flat arrays, for export) */

    /* pred_epoch: reusable prediction buffer (debug_confusion_all or filter needs it) */

    /* ── h0 precompute: ONCE before training, cached for ALL epochs ──
     * h0 depends ONLY on X (fixed input) and W0 (frozen). It NEVER
     * changes between epochs — recomputing each epoch is wasted work! */
    gettimeofday(&_tv0, NULL);
    if (!aa.dry_run) {
    if (!aa.no_precompute) {
#ifdef USE_HIP
            if (gpu_ok && n_xforms_eff <= 1) {
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
                for (int _b = 0; _b < active_members; _b++) {
                    ki_Member *mem = members[_b];
                    ki_member_compute_h0(mem, mem->input_buf, total_train, (int)n_cont);
                }
            }
            /* Precompute test eval gb once (if test data exists) */
            if (total_eval > 0) {
                for (int _b = 0; _b < active_members; _b++) {
                    ki_Member *mem = members[_b];
                    ki_member_compute_gb_te(mem, mem->input_buf_te, total_eval, (int)n_cont);
                }
            }
            /* free h0_buf after gb computation — is never needed again */
            for (int _b = 0; _b < active_members; _b++) {
                ki_Member *mem = members[_b];
                if (mem->h0_buf) { free(mem->h0_buf); mem->h0_buf = NULL; }
            }
        }
    }
    gettimeofday(&_tv1, NULL);

    /* ── Build target from gb_buf (no more h0_neuron) ──────────── *
     * Overwrites the uniform init values (from the TARGET-INIT loop
     * above) with correct counts. Then logit_convert + offset. */
    gettimeofday(&_target0, NULL);
    if (!aa.dry_run && V > 1 && !aa.no_precompute) {
        for (int _b = 0; _b < active_members; _b++) {
            ki_Member *mem = members[_b];
            uint32_t *tg_gb = mem->gb_buf;
            int free_tg_gb = 0;
            if (!tg_gb && aa.no_precompute) {
                /* --no-precompute: compute temp gb for target building */
                size_t ng = (size_t)total_train * (size_t)H_local;
                tg_gb = (uint32_t *)malloc(ng * sizeof(uint32_t));
                free_tg_gb = 1;
                for (int s = 0; s < total_train; s++) {
                    const uint32_t *in = X_tr + (size_t)s * (size_t)n_cont + mem->slc_off;
                    for (int h = 0; h < H_local; h++) {
                        uint32_t h0 = h0_neuron(in, mem->W0 + (size_t)h * (size_t)mem->w0_step, mem->NC_slice);
                        tg_gb[(size_t)s * (size_t)H_local + h] = h0_to_gb(h0);
                    }
                }
            }
            if (tg_gb) {
                COUNTER_TYPE *tgt = ki_build_target_from_gb(y_tr, total_train,
                    tg_gb, mem->H_local, V, class_counts);
                /* Overwrites the uniform init values */
                memcpy(mem->target, tgt, (size_t)mem->H_local * KI_NCLASSES * (size_t)V * sizeof(COUNTER_TYPE));
                free(tgt);
                if (free_tg_gb) free(tg_gb);
            }
        }
    }
    gettimeofday(&_target1, NULL);

    /* logit_convert.*class_offset for all members (based on new targets) */
    gettimeofday(&_logit0, NULL);
    if (!aa.dry_run && !aa.no_precompute) {
        for (int _b = 0; _b < active_members; _b++) {
            ki_Member *mem = members[_b];
            size_t m_tgt_sz = (size_t)mem->H_local * KI_NCLASSES * (size_t)V;
            SCORE_TYPE off_m[KI_NCLASSES];
            compute_class_offset(off_m, mem->target, mem->H_local, class_counts);
            memcpy(mem->offset, off_m, KI_NCLASSES * sizeof(SCORE_TYPE));
            logit_convert(mem->target, mem->H_local, class_counts);
            /* Inverse: negate logits → target = -count_logit.
             * Offset stays unchanged (class prior, not per-neuron). */
            if (aa.target_init_mode == KI_TARGET_INVERSE) {
                for (size_t _i = 0; _i < m_tgt_sz; _i++)
                    mem->target[_i] = -mem->target[_i];
            }
            /* First snapshot in best/err (for export / Rollback) */
            memcpy(mem->best_target, mem->target, m_tgt_sz * sizeof(COUNTER_TYPE));
            memcpy(mem->best_offset, mem->offset, KI_NCLASSES * sizeof(SCORE_TYPE));
            if (aa.err_rollback) {
                memcpy(mem->err_target, mem->target, m_tgt_sz * sizeof(COUNTER_TYPE));
                memcpy(mem->err_offset, mem->offset, KI_NCLASSES * sizeof(SCORE_TYPE));
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

    /* ── SEQUENTIAL TRAINING: one member at a time, all epochs ───── */
    SCORE_TYPE (*acc_votes_tr)[KI_NCLASSES] = (SCORE_TYPE (*)[KI_NCLASSES])calloc(
        (size_t)total_train, sizeof(SCORE_TYPE[KI_NCLASSES]));
    SCORE_TYPE (*acc_votes_te)[KI_NCLASSES] = total_eval > 0
        ? (SCORE_TYPE (*)[KI_NCLASSES])calloc((size_t)total_eval, sizeof(SCORE_TYPE[KI_NCLASSES]))
        : NULL;
    int final_trn_ok = 0, final_evl_ok = 0, best_evl_ok = 0;

    /* Member stats for --debug-member-stats */
    typedef struct {
        int   mb_idx, vi, color_bit, xform_id, enc_type, enc_width;
        float mem_trn, mem_evl;       /* standalone accuracy */
        float ens_before_trn, ens_before_evl; /* ensemble accuracy before this member */
        float gain_trn, gain_evl;     /* delta */
        float agree_trn, disagree_trn; /* member corrects / introduces errors (train) */
        float agree_evl, disagree_evl; /* member corrects / introduces errors (eval) */
    } _MemberStat;
    _MemberStat *_member_stats = NULL;
    int _member_stats_n = 0;
    float _ens_before_trn = 0.0f, _ens_before_evl = 0.0f;
    int _agree_trn = 0, _disagree_trn = 0, _agree_evl = 0, _disagree_evl = 0;

    /* Export buffers for per-member data (filled during member loop) */
    SCORE_TYPE *_export_scores_buf = NULL;
    int _export_scores_nm = 0;

    /* Member scores file (--debug-member-stats) */
    FILE *_ms_fp = NULL;
    char _ms_path[1024] = "";
    uint8_t *_ms_meta = NULL;  /* metadata buffer: 4 bytes per member */
    int _ms_meta_n = 0;
    if (aa.debug_member_stats) {
        snprintf(_ms_path, sizeof(_ms_path), "%s", aa.member_scores_path[0]
                 ? aa.member_scores_path : "member-scores.bin");
        _ms_fp = fopen(_ms_path, "wb");
        if (_ms_fp) {
            uint32_t _ms_hdr[4] = {(uint32_t)active_members, (uint32_t)total_train,
                                   (uint32_t)total_eval, (uint32_t)KI_NCLASSES};
            fwrite(_ms_hdr, sizeof(uint32_t), 4, _ms_fp);
        }
    }

    gettimeofday(&tv_start, NULL);
    int _report_int;
    if (aa.debug_member) {
        _report_int = 1;
    } else if (active_members <= epochs) {
        _report_int = 1;  /* fewer members than epochs → show all */
    } else if (epochs > 0) {
        _report_int = (active_members + epochs - 1) / epochs;  /* genau epochs Zeilen */
    } else {
        _report_int = 1;  /* epochs=0: still show report (initial eval) */
    }
    for (int mb = 0; mb < active_members; mb++) {
        ki_Member *mem = members[mb];
        /* Reset per-member agreement/disagreement counters */
        _agree_trn = _disagree_trn = _agree_evl = _disagree_evl = 0;

        /* ── Compute gb for this member once ── */
        size_t gb_sz = (size_t)total_train * (size_t)H_local;
        mem->gb_buf = (uint32_t *)malloc(gb_sz * sizeof(uint32_t));
        #pragma omp parallel for schedule(static)
        for (int s = 0; s < total_train; s++) {
            const uint32_t *in = mem->input_buf + (size_t)s * (size_t)n_cont + mem->slc_off;
            for (int h = 0; h < H_local; h++) {
                uint32_t h0 = h0_neuron(in, mem->W0 + (size_t)h * (size_t)mem->w0_step, mem->NC_slice);
                mem->gb_buf[(size_t)s * (size_t)H_local + h] = h0_to_gb(h0);
            }
        }
        mem->gb_buf_te = NULL;
        if (total_eval > 0) {
            size_t te_sz = (size_t)total_eval * (size_t)H_local;
            mem->gb_buf_te = (uint32_t *)malloc(te_sz * sizeof(uint32_t));
            #pragma omp parallel for schedule(static)
            for (int s = 0; s < total_eval; s++) {
                const uint32_t *in = mem->input_buf_te + (size_t)s * (size_t)n_cont + mem->slc_off;
                for (int h = 0; h < H_local; h++) {
                    uint32_t h0 = h0_neuron(in, mem->W0 + (size_t)h * (size_t)mem->w0_step, mem->NC_slice);
                    mem->gb_buf_te[(size_t)s * (size_t)H_local + h] = h0_to_gb(h0);
                }
            }
        }

        /* ── Build target from gb ── */
        if (!aa.dry_run && V > 1) {
            COUNTER_TYPE *tgt = ki_build_target_from_gb(y_tr, total_train,
                mem->gb_buf, mem->H_local, V, class_counts);
            memcpy(mem->target, tgt, (size_t)mem->H_local * KI_NCLASSES * (size_t)V * sizeof(COUNTER_TYPE));
            free(tgt);
            SCORE_TYPE off_m[KI_NCLASSES];
            compute_class_offset(off_m, mem->target, mem->H_local, class_counts);
            memcpy(mem->offset, off_m, KI_NCLASSES * sizeof(SCORE_TYPE));
            logit_convert(mem->target, mem->H_local, class_counts);
            if (aa.target_init_mode == KI_TARGET_INVERSE) {
                size_t m_tgt_sz = (size_t)mem->H_local * KI_NCLASSES * (size_t)V;
                for (size_t _i = 0; _i < m_tgt_sz; _i++)
                    mem->target[_i] = -mem->target[_i];
            }
            memcpy(mem->best_target, mem->target, (size_t)mem->H_local * KI_NCLASSES * (size_t)V * sizeof(COUNTER_TYPE));
            memcpy(mem->best_offset, mem->offset, KI_NCLASSES * sizeof(SCORE_TYPE));
            if (aa.err_rollback && mem->err_target) {
                memcpy(mem->err_target, mem->target, (size_t)mem->H_local * KI_NCLASSES * (size_t)V * sizeof(COUNTER_TYPE));
                memcpy(mem->err_offset, mem->offset, KI_NCLASSES * sizeof(SCORE_TYPE));
            }
        }

        /* ── Train all epochs for this member ── */
        int member_best_err = total_train;
        COUNTER_TYPE step_init_local = step_init;  /* per-member step, reduced by rollbacks */
        int rb_depth = 0;  /* rollback counter per member */
        float member_gap = 0.0f;  /* train/eval gap for step damping */
        SCORE_TYPE _gap_sc[KI_NCLASSES];
        for (int ep = 0; ep < epochs; ep++) {
            int s_step;
            if (aa.warmup_epochs > 0 && mem->ep < aa.warmup_epochs) {
                float scale = (float)(mem->ep + 1) / (float)aa.warmup_epochs;
                s_step = (int)((float)step_init_local * scale + 0.5f);
            } else {
                float progress = (float)(mem->ep - aa.warmup_epochs) / (float)((epochs + 0) - aa.warmup_epochs);
                if (progress > 1.0f) progress = 1.0f;
                float cosine = (1.0f + cosf(progress * (float)3.14159265358979323846f)) / 2.0f;
                float lr_min_f = (aa.lr_min > 0.0f) ? aa.lr_min : 0.0f;
                s_step = (int)((float)step_init_local * (lr_min_f + (1.0f - lr_min_f) * cosine) + 0.5f);
            }
            /* Gap damping: exp(-K × gap) reduces step when overfitting gap widens */
            if (aa.gap_k > 0.0f && member_gap > 0.0f) {
                float gap_factor = expf(-aa.gap_k * member_gap);
                s_step = (int)((float)s_step * gap_factor + 0.5f);
            }
            if (s_step < 2) s_step = 2;
            mem->step = s_step;

             int err = ki_batch_correct(mem->target, mem->H_local, mem->offset,
                         mem->gb_buf, y_tr, total_train, (COUNTER_TYPE)s_step,
                         (size_t)mem->H_local * KI_NCLASSES * 32, aa.filter_mask,
                         mem->H_local, 0);
            mem->last_err = err;
            /* trn_acc is set correctly AFTER evaluation (see below) */
            mem->ep++;

            /* ── Compute member gap (eval_err - train_err) for step damping ── */
            if (aa.gap_k > 0.0f && total_eval > 0 && mem->gb_buf_te) {
                int _evl_err = 0;
                #pragma omp parallel for firstprivate(_gap_sc) reduction(+:_evl_err) schedule(static)
                for (int s = 0; s < total_eval; s++) {
                    scores_otto_from_gb(s, mem->H_local, mem->gb_buf_te,
                                       mem->target, mem->offset, _gap_sc);
                    int pred = 0;
                    for (int k = 1; k < KI_NCLASSES; k++)
                        if (_gap_sc[k] > _gap_sc[pred]) pred = k;
                    if (pred != (int)y_te[s]) _evl_err++;
                }
                float trn_rate = (float)err / (float)total_train;
                float evl_rate = (float)_evl_err / (float)total_eval;
                member_gap = evl_rate - trn_rate;
                if (member_gap < 0.0f) member_gap = 0.0f;
            }

            /* ── debug-epoch: pro Epoche anzeigen ── */
            if (debug_epoch) {
                int _dep_trn = total_train - err;
                int _dep_evl = 0;
                if (total_eval > 0 && mem->gb_buf_te) {
                    SCORE_TYPE _dep_sc[KI_NCLASSES];
                    #pragma omp parallel for firstprivate(_dep_sc) reduction(+:_dep_evl) schedule(static)
                    for (int s = 0; s < total_eval; s++) {
                        scores_otto_from_gb(s, mem->H_local, mem->gb_buf_te,
                                           mem->target, mem->offset, _dep_sc);
                        int _pred = 0;
                        for (int k = 1; k < KI_NCLASSES; k++)
                            if (_dep_sc[k] > _dep_sc[_pred]) _pred = k;
                        if (_pred != (int)y_te[s]) _dep_evl++;
                    }
                }
                /* Member-Info (encoding/xform/channel) wie --debug-member */
                char _dep_info[80] = "";
                int _dep_vi = mem->vi;
                if (_dep_vi >= 0 && _dep_vi < aa.enc_count) {
                    const char *_dep_en = ki_enc_name_short(aa.enc_array[_dep_vi].type);
                    int _dep_ew = (int)aa.enc_array[_dep_vi].width;
                    if (KI_COLORS > 1) {
                        const char *_dep_cn = ki_color_name(mem->color_bit);
                        snprintf(_dep_info, sizeof(_dep_info), " %s=%s%d", _dep_cn, _dep_en, _dep_ew);
                    } else {
                        snprintf(_dep_info, sizeof(_dep_info), " %s%d", _dep_en, _dep_ew);
                    }
                }
                const char *_dep_xn = ki_xform_name(mem->xform_id);
                printf("      [%3d/%d] trn=%5.1f%%  evl=%5.1f%%  err=%d  step=%d%s xf=%s\n",
                       mem->ep, epochs,
                       (float)_dep_trn * 100.0f / (float)total_train,
                       (float)(total_eval - _dep_evl) * 100.0f / (float)(total_eval > 0 ? total_eval : 1),
                       err, mem->step, _dep_info, _dep_xn);
                fflush(stdout);
            }

            /* ── err-rollback: revert targets if error increased ── */
            if (aa.err_rollback && err > member_best_err && rb_depth < 5) {
                memcpy(mem->target, mem->err_target,
                       (size_t)mem->H_local * KI_NCLASSES * (size_t)V * sizeof(COUNTER_TYPE));
                memcpy(mem->offset, mem->err_offset, KI_NCLASSES * sizeof(SCORE_TYPE));
                float reduction = 2.0f / 3.0f;
                step_init_local = (int)((float)step_init_local * reduction + 0.5f);
                if (step_init_local < (COUNTER_TYPE)2) step_init_local = (COUNTER_TYPE)2;
                mem->step = (int)step_init_local;
                rb_depth++;
                ep--;  /* retry this epoch */
                continue;
            }
            if (aa.err_rollback && err < member_best_err) {
                member_best_err = err;
                memcpy(mem->err_target, mem->target,
                       (size_t)mem->H_local * KI_NCLASSES * (size_t)V * sizeof(COUNTER_TYPE));
                memcpy(mem->err_offset, mem->offset, KI_NCLASSES * sizeof(SCORE_TYPE));
            }
        }

        /* ── Phase 1: Pre-Eval — member accuracy (needed for threshold) ── */
        int _member_trn = 0, _member_evl = 0;
        int _skip_member = 0;
        if (aa.member_threshold > 0) {
            {   SCORE_TYPE _sc[KI_NCLASSES];
                for (int s = 0; s < total_train + total_eval; s++) {
                    int _is_eval = (s >= total_train);
                    int _ns = _is_eval ? s - total_train : s;
                    const uint32_t *_gb = _is_eval ? mem->gb_buf_te : mem->gb_buf;
                    if (!_gb) continue;
                    scores_otto_from_gb(_ns, mem->H_local, _gb, mem->target, mem->offset, _sc);
                    int _pred = 0;
                    for (int k = 1; k < KI_NCLASSES; k++)
                        if (_sc[k] > _sc[_pred]) _pred = k;
                    int _true_k = (int)(_is_eval ? y_te : y_tr)[_ns];
                    if (_pred == _true_k) { if (_is_eval) _member_evl++; else _member_trn++; }
                }
            }
            mem->trn_acc = (float)_member_trn * 100.0f / (float)(total_train > 0 ? total_train : 1);
            _skip_member = (mem->trn_acc < (float)aa.member_threshold);
        }

        /* ── Phase 2: Main Eval — add votes (only if not skipped) ── */
        if (!_skip_member) {
            /* Reset _member_trn if Phase 1 already set it (avoid double-count) */
            _member_trn = 0; _member_evl = 0;
            {   SCORE_TYPE sc[KI_NCLASSES];
                #pragma omp parallel for firstprivate(sc) reduction(+:final_trn_ok,final_evl_ok,_member_trn,_member_evl,_agree_trn,_disagree_trn,_agree_evl,_disagree_evl) schedule(static)
                for (int s = 0; s < total_train + total_eval; s++) {
                    int is_eval = (s >= total_train);
                    int ns = is_eval ? s - total_train : s;
                    const uint32_t *gb = is_eval ? mem->gb_buf_te : mem->gb_buf;
                    SCORE_TYPE *acc = is_eval ? acc_votes_te[ns] : acc_votes_tr[s];
                    if (!gb) continue;
                    scores_otto_from_gb(ns, mem->H_local, gb, mem->target, mem->offset, sc);
                    /* Member accuracy */
                    int _true_k = (int)(is_eval ? y_te : y_tr)[ns];
                    int mem_pred = 0;
                    for (int k = 1; k < KI_NCLASSES; k++)
                        if (sc[k] > sc[mem_pred]) mem_pred = k;
                    int mem_correct = (mem_pred == _true_k);
                    if (mem_correct) { if (is_eval) _member_evl++; else _member_trn++; }
                    /* Ensemble prediction BEFORE adding this member */
                    int _ens_pred = 0;
                    for (int k = 1; k < KI_NCLASSES; k++)
                        if (acc[k] > acc[_ens_pred]) _ens_pred = k;
                    int _ens_was_wrong = (_ens_pred != _true_k);
                    /* Now add votes to accumulator */
                    for (int k = 0; k < KI_NCLASSES; k++) acc[k] += sc[k];
                    /* Agreement/disagreement: did member correct ensemble error? */
                    if (aa.debug_member_stats) {
                        if (mem_correct && _ens_was_wrong)
                            { if (is_eval) _agree_evl++; else _agree_trn++; }
                        else if (!mem_correct && !_ens_was_wrong)
                            { if (is_eval) _disagree_evl++; else _disagree_trn++; }
                    }
                }
            }
        }

        /* ── Compute cumulative accuracy ── */
        {   int _trn_ok = 0, _evl_ok = 0;
            for (int s = 0; s < total_train; s++) {
                int pred = -1;
                for (int k = 0; k < KI_NCLASSES; k++)
                    if ((acc_votes_tr[s][k] > 0 || acc_votes_tr[s][k] < 0) && (pred < 0 || acc_votes_tr[s][k] > acc_votes_tr[s][pred]))
                        pred = k;
                if (pred >= 0 && pred == (int)y_tr[s]) _trn_ok++;
            }
            for (int s = 0; s < total_eval; s++) {
                int pred = -1;
                for (int k = 0; k < KI_NCLASSES; k++)
                    if ((acc_votes_te[s][k] > 0 || acc_votes_te[s][k] < 0) && (pred < 0 || acc_votes_te[s][k] > acc_votes_te[s][pred]))
                        pred = k;
                if (pred >= 0 && pred == (int)y_te[s]) _evl_ok++;
            }
            final_trn_ok = _trn_ok;
            final_evl_ok = _evl_ok;
            if (_evl_ok > best_evl_ok) best_evl_ok = _evl_ok;

            /* ══ Collect member stats for --debug-member-stats ══ */
            if (aa.debug_member_stats) {
                _MemberStat _ms;
                memset(&_ms, 0, sizeof(_ms));
                _ms.mb_idx = mb;
                _ms.vi = mem->vi;
                _ms.color_bit = mem->color_bit;
                _ms.xform_id = mem->xform_id;
                if (mem->vi >= 0 && mem->vi < aa.enc_count) {
                    _ms.enc_type = aa.enc_array[mem->vi].type;
                    _ms.enc_width = aa.enc_array[mem->vi].width;
                }
                _ms.mem_trn = (float)_member_trn * 100.0f / (float)total_train;
                _ms.mem_evl = (float)_member_evl * 100.0f / (float)(total_eval > 0 ? total_eval : 1);
                _ms.ens_before_trn = _ens_before_trn;
                _ms.ens_before_evl = _ens_before_evl;
                _ms.gain_trn = (float)_trn_ok * 100.0f / (float)total_train - _ens_before_trn;
                _ms.gain_evl = (float)_evl_ok * 100.0f / (float)(total_eval > 0 ? total_eval : 1) - _ens_before_evl;
                /* Compute agreement/disagreement from accumulator sample data */
                /* We compute this inline during evaluation: see above */
                _ms.agree_trn = (float)_agree_trn; _ms.disagree_trn = (float)_disagree_trn;
                _ms.agree_evl = (float)_agree_evl; _ms.disagree_evl = (float)_disagree_evl;

                _MemberStat *_new = realloc(_member_stats, (size_t)(_member_stats_n + 1) * sizeof(_MemberStat));
                if (_new) { _member_stats = _new; _member_stats[_member_stats_n++] = _ms; }
            }
            /* Save current ensemble accuracy as "before" for NEXT member */
            _ens_before_trn = (float)_trn_ok * 100.0f / (float)total_train;
            _ens_before_evl = (float)_evl_ok * 100.0f / (float)(total_eval > 0 ? total_eval : 1);
            gettimeofday(&tv_end, NULL);
            int _el = (int)((tv_end.tv_sec - tv_start.tv_sec) * 1000 + (tv_end.tv_usec - tv_start.tv_usec) / 1000);
            /* Print progress with cumulative + per-member accuracy */
            int _last_mb = (mb == active_members - 1);
            if (_report_int > 0 && ((mb + 1) % _report_int == 0 || _last_mb)) {
                /* Build member info (xform, channel, encoding) for debug output */
                char _info[128] = "";
                if (aa.debug_member || debug_epoch || (_report_int == 1 && active_members > 1)) {
                    int _vi = mem->vi;
                    const char *_cn = ki_color_name(mem->color_bit);
                    const char *_xn = ki_xform_name(mem->xform_id);
                    if (_vi >= 0 && _vi < aa.enc_count) {
                        const char *_en = ki_enc_name_short(aa.enc_array[_vi].type);
                        int _ew = aa.enc_array[_vi].width;
                        snprintf(_info, sizeof(_info), "  %s=%s%d xf=%s", _cn, _en, _ew, _xn);
                    } else {
                        snprintf(_info, sizeof(_info), "  ch=%s xf=%s", _cn, _xn);
                    }
                }
                int _filtered = (aa.member_threshold > 0 && mem->trn_acc < (float)aa.member_threshold);
                if (aa.debug_member || debug_epoch || (_report_int == 1 && active_members > 1)) {
                    printf("  [%3d/%d] ens=%.1f%%/%.1f%%  mem=%.1f%%/%.1f%%  err=%d  time=%dms%s%s\n",
                           mb + 1, active_members,
                           (float)_trn_ok * 100.0f / (float)total_train,
                           (float)_evl_ok * 100.0f / (float)(total_eval > 0 ? total_eval : 1),
                           (float)_member_trn * 100.0f / (float)total_train,
                           (float)_member_evl * 100.0f / (float)(total_eval > 0 ? total_eval : 1),
                           total_train - _trn_ok, _el, _info,
                           _filtered ? "  S" : "");
                } else {
                    printf("  [%3d/%d] trn=%5.1f%%  evl=%5.1f%%  err=%d  time=%dms%s%s\n",
                           mb + 1, active_members,
                           (float)_trn_ok * 100.0f / (float)total_train,
                           (float)_evl_ok * 100.0f / (float)(total_eval > 0 ? total_eval : 1),
                           total_train - _trn_ok, _el, _info,
                           _filtered ? "  S" : "");
                }
                fflush(stdout);
            }
        }

        /* ── Export per-member data (before freeing gb_buf) ── */
        /* --export-scores: accumulate per-member scores */
        if (aa.export_scores[0] && !aa.dry_run) {
            size_t _member_sz = (size_t)(total_train + total_eval) * KI_NCLASSES;
            SCORE_TYPE *_m_sc = (SCORE_TYPE *)calloc(_member_sz, sizeof(SCORE_TYPE));
            SCORE_TYPE sc[KI_NCLASSES];
            #pragma omp parallel for firstprivate(sc) schedule(static)
            for (int s = 0; s < total_train; s++) {
                scores_otto_from_gb(s, mem->H_local, mem->gb_buf,
                                   mem->target, mem->offset, sc);
                SCORE_TYPE *dst = _m_sc + (size_t)s * KI_NCLASSES;
                for (int k = 0; k < KI_NCLASSES; k++) dst[k] = sc[k];
            }
            if (total_eval > 0) {
                #pragma omp parallel for firstprivate(sc) schedule(static)
                for (int s = 0; s < total_eval; s++) {
                    scores_otto_from_gb(s, mem->H_local, mem->gb_buf_te,
                                       mem->target, mem->offset, sc);
                    SCORE_TYPE *dst = _m_sc + (size_t)(total_train + s) * KI_NCLASSES;
                    for (int k = 0; k < KI_NCLASSES; k++) dst[k] = sc[k];
                }
            }
            /* Accumulate into global export buffer */
            if (!_export_scores_buf) {
                _export_scores_buf = _m_sc;
                _export_scores_nm = 1;
            } else {
                size_t _es = (size_t)(total_train + total_eval) * KI_NCLASSES;
                _export_scores_buf = (SCORE_TYPE *)realloc(_export_scores_buf,
                    (size_t)(_export_scores_nm + 1) * _es * sizeof(SCORE_TYPE));
                memcpy(_export_scores_buf + (size_t)_export_scores_nm * _es, _m_sc, _es * sizeof(SCORE_TYPE));
                _export_scores_nm++;
                free(_m_sc);
            }
        }

        /* --export-neurons: write gb_buf + target + offset */
        if (aa.export_neurons[0] && !aa.dry_run && mem->gb_buf && mem->gb_buf_te) {
            int Hl = mem->H_local;
            int V_vn = VN_GROUPS_;
            size_t gb_tr_sz = (size_t)total_train * (size_t)Hl;
            size_t gb_te_sz = (size_t)total_eval * (size_t)Hl;
            size_t tgt_sz = (size_t)Hl * (size_t)V_vn * (size_t)KI_NCLASSES;
            /* Write to a per-member temp file, will be merged at end */
            char _np[1024];
            snprintf(_np, sizeof(_np), "%s.m%d", aa.export_neurons, mb);
            FILE *nf = fopen(_np, "wb");
            if (nf) {
                fwrite(mem->gb_buf, sizeof(uint32_t), gb_tr_sz, nf);
                fwrite(mem->gb_buf_te, sizeof(uint32_t), gb_te_sz, nf);
                fwrite(mem->target, sizeof(COUNTER_TYPE), tgt_sz, nf);
                fwrite(mem->offset, sizeof(SCORE_TYPE), KI_NCLASSES, nf);
                fclose(nf);
            }
        }

        /* ── Write member scores for --debug-member-stats ── */
        if (_ms_fp) {
            SCORE_TYPE _ms_sc[KI_NCLASSES];
            size_t _ms_total = (size_t)(total_train + total_eval);
            for (size_t _ms_s = 0; _ms_s < _ms_total; _ms_s++) {
                int _ms_is_eval = (_ms_s >= (size_t)total_train);
                size_t _ms_ns = _ms_is_eval ? _ms_s - (size_t)total_train : _ms_s;
                const uint32_t *_ms_gb = _ms_is_eval ? mem->gb_buf_te : mem->gb_buf;
                scores_otto_from_gb((int)_ms_ns, mem->H_local, _ms_gb,
                                   mem->target, mem->offset, _ms_sc);
                fwrite(_ms_sc, sizeof(COUNTER_TYPE), KI_NCLASSES, _ms_fp);
            }
            /* Collect metadata: write member name as string (color=encN xf=name) */
            char _ms_m[64] = "";
            if (mem->vi >= 0 && mem->vi < aa.enc_count) {
                int _ms_vi = mem->vi;
                const char *_cn = ki_color_name(aa.enc_array[_ms_vi].color);
                const char *_en = ki_enc_name_short(aa.enc_array[_ms_vi].type);
                int _ew = aa.enc_array[_ms_vi].width;
                const char *_xn = ki_xform_name(mem->xform_id);
                snprintf(_ms_m, sizeof(_ms_m), "%s=%s%d-%s", _cn, _en, _ew, _xn);
            } else {
                snprintf(_ms_m, sizeof(_ms_m), "%s=%s-%s",
                    ki_color_name(mem->color_bit), "?", ki_xform_name(mem->xform_id));
            }
            size_t _ms_namelen = strlen(_ms_m) + 1;
            uint8_t *_new = realloc(_ms_meta, (size_t)_ms_meta_n * 64 + _ms_namelen);
            if (_new) {
                _ms_meta = _new;
                memcpy(_ms_meta + (size_t)_ms_meta_n * 64, _ms_m, _ms_namelen);
                _ms_meta_n++;
            }
        }

        /* ── Free per-member gb ── */
        free(mem->gb_buf); mem->gb_buf = NULL;
        free(mem->gb_buf_te); mem->gb_buf_te = NULL;
    }

    /* Close member-scores.bin (--debug-member-stats) */
    if (_ms_fp) {
        /* Metadata: 4 bytes per member (color_bit, enc_type, enc_width, xform_id) */
        if (_ms_meta) {
            fwrite(_ms_meta, 64, (size_t)_ms_meta_n, _ms_fp);
            free(_ms_meta);
        }
        /* Labels */
        fwrite(y_tr, 1, (size_t)total_train, _ms_fp);
        if (total_eval > 0) fwrite(y_te, 1, (size_t)total_eval, _ms_fp);
        fclose(_ms_fp);
        char _ms_abspath[1024];
        if (_ms_path[0] == '/') {
            snprintf(_ms_abspath, sizeof(_ms_abspath), "%s", _ms_path);
        } else if (getcwd(_ms_abspath, sizeof(_ms_abspath) - 64)) {
            strcat(_ms_abspath, "/");
            strcat(_ms_abspath, _ms_path);
        } else {
            snprintf(_ms_abspath, sizeof(_ms_abspath), "%s", _ms_path);
        }
        printf("  Member scores: %s  (%d members, %d+%d samples)\n",
               _ms_abspath, active_members, total_train, total_eval);
        fflush(stdout);
    }

    gettimeofday(&tv_end, NULL);
    int elapsed_ms = (int)((tv_end.tv_sec - tv_start.tv_sec) * 1000
                         + (tv_end.tv_usec - tv_start.tv_usec) / 1000);
    printf("\n══╡ DONE ╞══  sequential training complete (%d members, %dms)\n",
           active_members, elapsed_ms);


    /* ── Final report ── */
    int trn_ok = final_trn_ok, evl_ok = final_evl_ok;
    uint8_t *pred_eval = aa.predictions[0] ?  (uint8_t *)ki_xcalloc((size_t)total_eval, sizeof(uint8_t)) : NULL ;
    uint8_t *pred_tr = (aa.debug_confusion && !aa.dry_run)
        ? (uint8_t *)ki_xcalloc((size_t)total_train, sizeof(uint8_t)) : NULL ;
    if (!aa.dry_run) {
        if (pred_tr) {
            for (int s = 0; s < total_train; s++) {
                int pred = -1;
                for (int k = 0; k < KI_NCLASSES; k++)
                    if ((acc_votes_tr[s][k] > 0 || acc_votes_tr[s][k] < 0) && (pred < 0 || acc_votes_tr[s][k] > acc_votes_tr[s][pred]))
                        pred = k;
                pred_tr[s] = (uint8_t)(pred >= 0 ? pred : 0);
            }
        }
        if (pred_eval) {
            for (int s = 0; s < total_eval; s++) {
                int pred = -1;
                for (int k = 0; k < KI_NCLASSES; k++)
                    if ((acc_votes_te[s][k] > 0 || acc_votes_te[s][k] < 0) && (pred < 0 || acc_votes_te[s][k] > acc_votes_te[s][pred]))
                        pred = k;
                pred_eval[s] = (uint8_t)(pred >= 0 ? pred : 0);
            }
        }
    }

    /* Member-Destruktion (in ki_member_destroy freed: target, offset, best/err, h0, gb, gbl) */
    /* Final evaluation: with CURRENT targets (after last correction) */
    /* ── Save per-member scores (f�r merge-ensemble) ────────── */
    if (aa.export_merge_scores[0] && !aa.dry_run && total_eval > 0) {
        export_merge_scores_archive(aa.export_merge_scores, X_te, y_te, total_eval,
                            members, active_members, (int)n_cont, evl_ok);
    }

    /* ── Export per-sample gb_buf + Target + Offset (for Adam-on-neurons) ── */
    /* In seq mode, data was written per-member during the training loop */
    if (aa.export_neurons[0] && !aa.dry_run) {
        /* Check if per-member files exist (written during seq loop) */
        char _np[1024];
        snprintf(_np, sizeof(_np), "%s.m0", aa.export_neurons);
        if (access(_np, F_OK) == 0) {
            /* Merge per-member temp files into one final file */
            int total_all_ex = total_train + total_eval;
            int n_m = active_members;
            int V_vn = VN_GROUPS_;
            int H0 = members[0] ? members[0]->H_local : 0;
            size_t gb_tr_sz = (size_t)total_train * (size_t)H0;
            size_t gb_te_sz = (size_t)total_eval * (size_t)H0;
            size_t tgt_sz = (size_t)H0 * (size_t)V_vn * (size_t)KI_NCLASSES;
            FILE *nf = fopen(aa.export_neurons, "wb");
            if (nf) {
                uint32_t hdr[8] = {
                    (uint32_t)total_all_ex, (uint32_t)KI_NCLASSES, (uint32_t)n_m, (uint32_t)OT_PRECISION,
                    (uint32_t)H0, (uint32_t)V_vn, 3, 0
                };
                fwrite(hdr, sizeof(uint32_t), 8, nf);
                for (int m = 0; m < active_members; m++) {
                    snprintf(_np, sizeof(_np), "%s.m%d", aa.export_neurons, m);
                    FILE *_tf = fopen(_np, "rb");
                    if (!_tf) continue;
                    uint32_t *_gb_tr = (uint32_t *)malloc(gb_tr_sz * sizeof(uint32_t));
                    uint32_t *_gb_te = (uint32_t *)malloc(gb_te_sz * sizeof(uint32_t));
                    COUNTER_TYPE *_tgt = (COUNTER_TYPE *)malloc(tgt_sz * sizeof(COUNTER_TYPE));
                    SCORE_TYPE *_off = (SCORE_TYPE *)malloc(KI_NCLASSES * sizeof(SCORE_TYPE));
                    if (_gb_tr && _gb_te && _tgt && _off) {
                        fread(_gb_tr, sizeof(uint32_t), gb_tr_sz, _tf);
                        fread(_gb_te, sizeof(uint32_t), gb_te_sz, _tf);
                        fread(_tgt, sizeof(COUNTER_TYPE), tgt_sz, _tf);
                        fread(_off, sizeof(SCORE_TYPE), KI_NCLASSES, _tf);
                        fwrite(_gb_tr, sizeof(uint32_t), gb_tr_sz, nf);
                        fwrite(_gb_te, sizeof(uint32_t), gb_te_sz, nf);
                        fwrite(_tgt, sizeof(COUNTER_TYPE), tgt_sz, nf);
                        fwrite(_off, sizeof(SCORE_TYPE), KI_NCLASSES, nf);
                    }
                    free(_gb_tr); free(_gb_te); free(_tgt); free(_off);
                    fclose(_tf);
                    remove(_np);  /* cleanup temp file */
                }
                fclose(nf);
                /* Append labels */
                FILE *lf = fopen(aa.export_neurons, "ab");
                if (lf) {
                    fwrite(y_tr, 1, (size_t)total_train, lf);
                    if (total_eval > 0) fwrite(y_te, 1, (size_t)total_eval, lf);
                    fclose(lf);
                }
                printf("  Export neurons: %s  (%d members x H=%d, %d samples)\n",
                       aa.export_neurons, n_m, H0, total_all_ex);
                fflush(stdout);
            }
        } else {
            /* No per-member files: run original code (non-seq mode) */
            /* (kept for gcc compatibility — not reached in seq mode) */
        }
    }

    if (aa.debug_class_voting && !aa.dry_run) {
        print_class_voting_debug(members, active_members,
                                 X_tr, y_tr, total_train, (int)n_cont, epochs - 1);
    }
    if (aa.debug_confusion && !aa.dry_run) {
        print_confusion_debug(y_tr, pred_tr, total_train, epochs - 1, 1);
    }
    free(pred_tr);

    /* Export MUST happen before member destruction (liest members[b]->target/offset) */
    if (aa.exportD[0] != '\0')
    {
        export_ensemble(aa.exportD, W0_ens, total_members,
                        members, active_members,
                        H_local, NC_slice, (int)n_cont);
    }

    /* Destroy members.*after final evaluation + Export) */
    for (int _z = 0; _z < active_members; _z++)
        ki_member_destroy(members[_z]);

    float fin_trn = (float)trn_ok * 100.0f / (float)total_train;
    float fin_evl = (total_eval > 0)
        ? (float)evl_ok * 100.0f / (float)total_eval : 0.0f;
    int fin_err = total_train - trn_ok;  /* Fehler passend zu train=/eval= */

    printf("\n══╡ RESULT ╞══════════════════════════════════════════════════════\n");
    float final_best = (best_evl_ok > evl_ok) ? (float)best_evl_ok : (float)evl_ok;
    final_best = final_best * 100.0f / (float)total_eval;
    printf("  H=%d  ens=%d  v_split=%d  h_split=%d  ep=%d  trn=%.1f%%  evl=%.1f%%  best=%.1f%%  lr=%.4f  time=%dms\n",
           H, ensembleN, splitVN, splitHN, epochs, fin_trn, fin_evl, final_best,
           (double)aa.lr, elapsed_ms);

    /* REPORT uses best eval across all member evaluations */
    int report_evl_ok = (best_evl_ok > 0) ? best_evl_ok : final_evl_ok;
    ki_report_show(trn_ok, total_train, report_evl_ok, total_eval,
                   elapsed_ms, aa.threadN, fin_err, aa.lr, active_members);

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
    /* Free X_xform buffers (identity reuses X_all, already freed above) */
    for (int _xf = 0; _xf < KI_XFORM_COUNT; _xf++) {
        if (_xf == KI_XFORM_ID) continue;  /* freed via X_all */
        if (X_xform[_xf] && X_xform[_xf] != X_all)
            free(X_xform[_xf]);
    }
    free(acc_votes_tr);
    if (acc_votes_te) free(acc_votes_te);
    free(members);
    free(X_all);
    free(W0_ens);
#ifdef USE_HIP
    hip_mem_done();
#endif
    ki_dataset_free(&data);
    return 0;
}
