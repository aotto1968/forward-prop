/*
 * ── FINAL ARCHITECTURE (2026-08-08) ────────────────────────────────────────
 * otto-score-ifc/mnist/mlp-bin32-otto-trn-seq-prof.c — PRF trainer.
 * Runs N members in parallel, each with 1 thread (distributed simulation:
 * N members × 1 thread instead of 1 member × N threads). Evolved from the
 * cache-blocking study (plans/plan-2026-08-07-sweep-cache-blocking.md);
 * the old sequential trainer is archived as mlp-bin32-otto-trn-seq.c
 * (still buildable via the -seq-8-* Makefile targets for A/B testing).
 * Verified: ~2× faster than the sequential trainer at same thread count,
 * int32 results bit-identical (integer addition is associative → no
 * thread-order drift; the DRAM chip accumulates popcounts in integer
 * arithmetic). --threadN N = parallel member width; per-member progress
 * line is the default, W0/MIN/MAX/pxz only with explicit --debug-member.
 * ──────────────────────────────────────────────────────────────────────────
 *
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

#include <stdio.h>
#include <unistd.h>
#include <inttypes.h>
#include <sys/stat.h>
#include <errno.h>
#include <dlfcn.h>
/* LeakSanitizer cache marker for intentional process-lifetime allocations.
 * Resolved at RUNTIME via dlsym so it works in every build mode:
 *   - LD_PRELOAD of libasan (run-research.sh --debug) → symbol found, call active
 *   - built with -fsanitize=address → symbol found, call active
 *   - plain release build → dlsym returns NULL, call is a no-op
 * (A compile-time weak symbol would be optimized away at -O3 and fail to
 *  link the release build. See: bug report LeakSanitizer 10976 bytes.) */
static void ki_lsan_ignore(const void *p) {
    static void (*fn)(const void *) = NULL;
    if (!fn) {
        /* benign race: multiple threads may call dlsym concurrently on first
         * use — all store the same function pointer (or all store NULL). */
        fn = (void (*)(const void *))dlsym(RTLD_DEFAULT, "__lsan_ignore_object");
    }
    if (fn) fn(p);
}

#include "ki-config.h"
/* libtprint BEFORE ki-common.h: print_confusion_debug() in ki-common.h uses
 * TPrint for the confusion table (2026-08-12). */
/* ki-common.h with KI_COMMON_LOAD_INPUT: suppress the default raw-packing
 * load_input (ki-load.h provides the encoding-aware one). Explicit include —
 * ki-load.h does NOT pull ki-common.h in anymore (flat include level). */
#define KI_COMMON_LOAD_INPUT
#include "ki-common.h"
#undef KI_COMMON_LOAD_INPUT
#include "../lib/ki-load.h"           /* encoding-aware load_input (CIFAR + MNIST) */
#include "../lib/ki-ens.h"            /* .ens archive format: write + roundtrip verify */
#include "../lib/maj3.h"
#include "../lib/maj1.h"
#include "../lib/maj7.h"
#include "../lib/ki-encoding.h"

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
            (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h), _v, H, 32)]; \
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
            (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h), _v, H, 16)]; \
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
            (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h), _v, H, 10)]; \
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
            (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h), _v, H, 8)]; \
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
            (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h), _v, H, 4)]; \
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
            (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h), _v, H, 2)]; \
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
            (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h), 0, H, 1)]; \
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

/* ── [PROF] phase profiling (local, prof trainer only, 2026-08-08) ──
 * --prof: time the per-member phases (CEX-load / gb-train / gb-eval /
 * target / train / eval) and print one [PROF] line per member plus a
 * summed summary. Off by default → the binary stays A/B-identical to
 * the production flt32 trainer without the flag. */
static int g_prof = 0;

/* ── [EXPORT] --export-default (2026-08-18) ──
 * Derive the export directory name from ki_default_tag() (H512-E10-OT8-M1-105-INT32),
 * analog zum MERGE --member-out-default. Off by default → only an explicit
 * --export DIR exports. */
static int g_export_default = 0;

/* ── [DBG] explicit --debug-member (local, prof trainer only) ──
 * Per-member progress (mem=... id:...) is ALWAYS printed (PRF default,
 * visual feedback). The extra fields W0 / MIN / MAX / pxz are only shown
 * when the user passes --debug-member explicitly — g_dbg_member tracks
 * that explicit request (aa.debug_member is forced to 1 in main). */
static int g_dbg_member = 0;

/* ── [PRF] parallel member execution (prof trainer ONLY mode) ──
 * The prof trainer runs N members in parallel, each with 1 thread
 * (simulates distributed computing: N members × 1 thread instead of
 * 1 member × N threads). This is the ONLY mode of the prof trainer —
 * the production trainer (mlp-bin32-otto-trn-seq.c) remains the
 * sequential baseline for A/B comparisons. OpenMP nested parallelism is
 * disabled by default, so inner per-sample regions must NOT spawn their
 * own teams — see the if(!g_prf) removals below (g_prf is constant 1). */
static const int g_prf = 1;         /* always on in the prof trainer */
static int g_prf_width = 0;        /* --threadN: parallel member width
                                     (num_threads for the OUTER member loop).
                                     Defaults to the number of HW threads;
                                     --threadN N caps it. */
static int g_parallel_members = 0; /* EXCLUSIVE parallelism (2026-08-09):
                                     1 = member loop parallel (active >= width)
                                        → sample loops single
                                     0 = member loop single (few members)
                                        → sample loops use all cores.
                                     Never both at once → no nested OpenMP,
                                     no deadlock (bug 2026-08-09: 1 member ran
                                     single-threaded = 13.6s vs 3.6s). */
static int g_user_threadn = 0;     /* 1 = user passed --threadN explicitly */

/* Trainer executable name (basename of argv[0], set in main). Written to
 * the --export-merge-scores dir .meta as EXE= and checked on re-export:
 * a scores archive must not be re-populated by a DIFFERENT trainer binary
 * (e.g. 8-bit xnor vs 16-bit bitvoting) — the .meta H/BITS/COUNTER_TYPE
 * guards catch most mismatches, but EXE is the explicit one. */
static char g_exe_name[256] = "";

ki_Args aa = {
    .hidden             = 64,
    .epochs             = 1,
    .cfg_epochs         = 1,
    .batchN             = KI_DEFAULT_BATCH_N,
    .trainN             = 0,      /* auto: set from dataset default */
    .evalN              = 0,      /* auto: set from dataset default */
    .seed               = 42,
    .lr                 = KI_DEFAULT_LR,
    .threadN            = 0,      /* PRF: 0 = auto (all HW threads) = parallel
                                     member width; --threadN N changes it.
                                     The INNER training parallelism is capped
                                     at 8 separately (ki_batch_correct with
                                     batch=64 does NOT scale beyond 8, bug
                                     2026-08-10) — so the effective 1-member
                                     behavior matches the historical seq
                                     default of 8 threads. */
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
    .ensemble_seed      = ENS_SEED_CONST,
    .target_init_mode = KI_TARGET_COUNT,
    .multi_correct      = 0,
    .seed_splitmix      = 1,
    .maj_mode           = KI_MAJ_1,  /* --maj 1: exact per-bit majority (DRAM-native, replaces old maj3 default) */
    .eff_lambda         = 0.02f,     /* --eff-lambda: member penalty for the eff= score (eff = eval - lambda*(members-1)) */
    .maj1_thresh        = -2,     /* --maj1-thresh: -2 = auto per encoding (n*135/256 for 8-bit) */
    .maj_step           = 0,      /* 0=auto (KI_PX_PER_CONT) */
    .debug_maj          = 0,      /* 0=auto, 1=container, 2=pixel */
    .rows_mode          = 0,      /* 0=flat, 1=per-row members */
    .member_threshold   = 0,      /* 0=disabled, else min trn%% to participate */
    .sweep              = 0,      /* --sweep: off by default */
    .xform_cache_level  = 2,      /* --xform-cache-level: 2=transform+block cache (default, ~2 GB slot on CIFAR) */
    .xforms             = (1ull << KI_XFORM_ID),  /* default: identity only */
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

/* Global channel parameters set after --channel in main) */
static int eff_colors = 3;              /* number of active channels / members (popcount mask) */
static int active_chans[KI_ENC_MAX];    /* Mapping: seq_idx → Bit-Position (0..8) */
#define BITS       32
#define N_CLASSES KI_NCLASSES

/* Export file magic + version */
#define OTTO_MAGIC   0x4F54544FU   /* "OTTO" */
#define OTTO_VERSION 5U            /* v5 = ensemble (no precision field) */
#define OTTO_VERSION_V6 6U         /* v6 = ensemble + precision field */
#define OTTO_VERSION_V7 7U         /* v7 = v6 + per-member channel/enc/width/xform metadata */
#define OTTO_VERSION_V8 8U         /* v8 = v7 + maj_mode/maj1_thresh/splitVN/splitHN in the
                                      header (self-describing IFC — import no longer starts
                                      with CLI defaults, bug 2026-08-09: 36.6% without
                                      --maj1-thresh 104) */

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
static uint32_t __attribute__((unused)) h0_neuron(const uint32_t *in, const uint32_t *W0_row, int nc_local, int half __attribute__((unused))) {
    (void)half; /* suppress -Wunused-parameter (half used only in KI_MAJ_1/1R) */
    /* THREAD-LOCAL match buffer (was uint32_t match[4096] on the stack = 16KB,
     * then heap-malloc per call). The inner per-sample loops call h0_neuron
     * H_local× per sample — 512×60000 = 30M calls. malloc per call was 6.5s
     * (gb phase). A static __thread buffer is allocated ONCE per thread and
     * reused: no stack overflow (16KB × threads is fine) and no malloc storm.
     * Size: nc_local ≤ 4096 (CIFAR worst case). */
    static __thread uint32_t *t_match = NULL;
    static __thread int t_match_sz = 0;
    if (!t_match || t_match_sz < nc_local) {
        free(t_match);
        t_match = (uint32_t *)malloc((size_t)nc_local * sizeof(uint32_t));
        if (!t_match) { fprintf(stderr, "[FATAL] h0_neuron OOM\n"); exit(1); }
        t_match_sz = nc_local;
        /* INTENTIONAL: process-lifetime per-thread cache (freed never by design).
         * Allocated once per thread, reused across all 30M h0_neuron calls.
         * LeakSanitizer would flag it as a leak — mark it as intentional
         * (no-op when libasan is not loaded). */
        ki_lsan_ignore(t_match);
    }
    uint32_t *match = t_match;
    uint32_t _result;
    switch (aa.maj_mode) {
        case KI_MAJ_1: {
            /* Container-level flat (original majority_tree1) */
            for (int c = 0; c < nc_local; c++)
                match[c] = H0_MATCH(in, W0_row, c);
            _result = majority_tree1(match, nc_local, half);
            break;
        }
        case KI_MAJ_1R: {
            /* Container-level row-wise (old rowwise) */
            for (int c = 0; c < nc_local; c++)
                match[c] = H0_MATCH(in, W0_row, c);
            int cpr = KI_COLS / KI_PX_PER_CONT;
            _result = majority_tree1_rowwise(match, nc_local, cpr, half);
            break;
        }
        case KI_MAJ_1P: {
            /* Pixel-accurate flat */
            int pix_half = nc_local * KI_PX_PER_CONT / 2;
            uint32_t r = 0;
            for (int g = 0; g < KI_PIXEL_GROUPS; g++) {
                for (int c = 0; c < nc_local; c++)
                    match[c] = H0_MATCH(in, W0_row + g * nc_local, c);
                r |= (majority_tree1_pixel(match, nc_local, pix_half) << (g * KI_BIT_POS));
            }
            _result = r;
            break;
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
                uint32_t cross = majority_tree1(row_results, rows, rows / 2);
                result |= (cross << (g * KI_BIT_POS));
            }
            _result = result;
            break;
        }
        case KI_MAJ_7: {
            for (int c = 0; c < nc_local; c++)
                match[c] = H0_MATCH(in, W0_row, c);
            _result = majority_tree7(match, nc_local);
            break;
        }
        default: {
            for (int c = 0; c < nc_local; c++)
                match[c] = H0_MATCH(in, W0_row, c);
            /* KI_MAJ_3 with optional pixel step (0=auto=KI_PX_PER_CONT) */
            int _step = aa.maj_step;
            if (_step == 0) _step = KI_PX_PER_CONT;
            /* --debug-maj overrides auto-detection */
            if (aa.debug_maj == 1) {
                _result = majority_tree3(match, nc_local);
            } else if (aa.debug_maj == 2) {
                _result = majority_tree3_pixel_step(match, nc_local, _step);
            } else if (_step == KI_PX_PER_CONT) {
                /* Default auto: fast path for container-aligned steps */
                _result = majority_tree3(match, nc_local);
            } else {
                _result = majority_tree3_pixel_step(match, nc_local, _step);
            }
            break;
        }
    }
    return _result;
}

/* ── H0/GB value for neuron h of a member ──────────────────────────
 * Normal Otto: h0_neuron (XNOR with W0 row + majority) → h0_to_gb.
 * KI_BITVOTING: identity — the input container h IS the gb value
 * (direct pixel-bit → class voting, no W0, no majority).
 * in: points at the member's slice start (slc_off applied).
 * Forward decl — definition sits after h0_to_gb (which it calls). */
static inline uint32_t ki_gb_for_neuron(const uint32_t *in, const uint32_t *W0_row,
                                        int h, int w0_step, int NC_slice, int half);

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
static __attribute__((unused)) COUNTER_TYPE *ki_build_target(const uint32_t *X, const uint8_t *Y, int N,
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
        int32_t *lt = (int32_t *)ki_xcalloc(sz, sizeof(*lt));
        #pragma omp for schedule(static)
        for (int s = 0; s < N; s++) {
            int k = (int)Y[s];
            const uint32_t *in = X + (size_t)s * (size_t)stride + nc_off;
            for (int h = 0; h < H_local; h++) {
                int _ws1 = aa.rows_mode && aa.maj_mode == KI_MAJ_1 ? NC_slice * 4 : NC_slice;
                int _half = ki_compute_half(NC_slice);
                uint32_t h0 = ki_gb_for_neuron(in, W0, h, _ws1, NC_slice, _half);
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
static __attribute__((unused)) COUNTER_TYPE *ki_build_target_from_gb(const uint8_t *Y, int N,
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
            int32_t *lt = (int32_t *)ki_xcalloc(sz, sizeof(*lt));
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
            int32_t *lt = (int32_t *)ki_xcalloc(sz, sizeof(*lt));
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
            int32_t *lt = (int32_t *)ki_xcalloc(sz, sizeof(*lt));
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
            int32_t *lt = (int32_t *)ki_xcalloc(sz, sizeof(*lt));
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
#if 1
/* Old version: __builtin_ctz-based (3-cycle TZCNT + data-dependent loop) */
#define VN_SCORE_FROM_GB(gb, h, H, NG, TGT, SC) do { \
    uint32_t _b = (gb); \
    while (_b) { int _v = __builtin_ctz(_b); \
        for (int _k = 0; _k < KI_NCLASSES; _k++) \
            (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h), _v, H, NG)]; \
        _b &= _b - 1; } \
} while (0)
#else
/* Unrolled 32-bit test: no builtin, no data-dependent loop, always 32 iterations */
#define VN_SCORE_FROM_GB(gb, h, H, NG, TGT, SC) do { \
    uint32_t _b = (gb); \
    if (_b & 0x00000001U) { for (int _k = 0; _k < KI_NCLASSES; _k++) (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h),  0, H, NG)]; } \
    if (_b & 0x00000002U) { for (int _k = 0; _k < KI_NCLASSES; _k++) (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h),  1, H, NG)]; } \
    if (_b & 0x00000004U) { for (int _k = 0; _k < KI_NCLASSES; _k++) (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h),  2, H, NG)]; } \
    if (_b & 0x00000008U) { for (int _k = 0; _k < KI_NCLASSES; _k++) (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h),  3, H, NG)]; } \
    if (_b & 0x00000010U) { for (int _k = 0; _k < KI_NCLASSES; _k++) (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h),  4, H, NG)]; } \
    if (_b & 0x00000020U) { for (int _k = 0; _k < KI_NCLASSES; _k++) (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h),  5, H, NG)]; } \
    if (_b & 0x00000040U) { for (int _k = 0; _k < KI_NCLASSES; _k++) (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h),  6, H, NG)]; } \
    if (_b & 0x00000080U) { for (int _k = 0; _k < KI_NCLASSES; _k++) (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h),  7, H, NG)]; } \
    if (_b & 0x00000100U) { for (int _k = 0; _k < KI_NCLASSES; _k++) (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h),  8, H, NG)]; } \
    if (_b & 0x00000200U) { for (int _k = 0; _k < KI_NCLASSES; _k++) (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h),  9, H, NG)]; } \
    if (_b & 0x00000400U) { for (int _k = 0; _k < KI_NCLASSES; _k++) (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h), 10, H, NG)]; } \
    if (_b & 0x00000800U) { for (int _k = 0; _k < KI_NCLASSES; _k++) (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h), 11, H, NG)]; } \
    if (_b & 0x00001000U) { for (int _k = 0; _k < KI_NCLASSES; _k++) (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h), 12, H, NG)]; } \
    if (_b & 0x00002000U) { for (int _k = 0; _k < KI_NCLASSES; _k++) (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h), 13, H, NG)]; } \
    if (_b & 0x00004000U) { for (int _k = 0; _k < KI_NCLASSES; _k++) (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h), 14, H, NG)]; } \
    if (_b & 0x00008000U) { for (int _k = 0; _k < KI_NCLASSES; _k++) (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h), 15, H, NG)]; } \
    if (_b & 0x00010000U) { for (int _k = 0; _k < KI_NCLASSES; _k++) (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h), 16, H, NG)]; } \
    if (_b & 0x00020000U) { for (int _k = 0; _k < KI_NCLASSES; _k++) (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h), 17, H, NG)]; } \
    if (_b & 0x00040000U) { for (int _k = 0; _k < KI_NCLASSES; _k++) (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h), 18, H, NG)]; } \
    if (_b & 0x00080000U) { for (int _k = 0; _k < KI_NCLASSES; _k++) (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h), 19, H, NG)]; } \
    if (_b & 0x00100000U) { for (int _k = 0; _k < KI_NCLASSES; _k++) (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h), 20, H, NG)]; } \
    if (_b & 0x00200000U) { for (int _k = 0; _k < KI_NCLASSES; _k++) (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h), 21, H, NG)]; } \
    if (_b & 0x00400000U) { for (int _k = 0; _k < KI_NCLASSES; _k++) (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h), 22, H, NG)]; } \
    if (_b & 0x00800000U) { for (int _k = 0; _k < KI_NCLASSES; _k++) (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h), 23, H, NG)]; } \
    if (_b & 0x01000000U) { for (int _k = 0; _k < KI_NCLASSES; _k++) (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h), 24, H, NG)]; } \
    if (_b & 0x02000000U) { for (int _k = 0; _k < KI_NCLASSES; _k++) (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h), 25, H, NG)]; } \
    if (_b & 0x04000000U) { for (int _k = 0; _k < KI_NCLASSES; _k++) (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h), 26, H, NG)]; } \
    if (_b & 0x08000000U) { for (int _k = 0; _k < KI_NCLASSES; _k++) (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h), 27, H, NG)]; } \
    if (_b & 0x10000000U) { for (int _k = 0; _k < KI_NCLASSES; _k++) (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h), 28, H, NG)]; } \
    if (_b & 0x20000000U) { for (int _k = 0; _k < KI_NCLASSES; _k++) (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h), 29, H, NG)]; } \
    if (_b & 0x40000000U) { for (int _k = 0; _k < KI_NCLASSES; _k++) (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h), 30, H, NG)]; } \
    if (_b & 0x80000000U) { for (int _k = 0; _k < KI_NCLASSES; _k++) (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, (h), 31, H, NG)]; } \
} while (0)
#endif

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
        int nk = class_counts[k];
        if (nk <= 0) { class_offset[k] = (SCORE_TYPE)0; continue; }
        /* INTENTIONAL (2026-08-16): accumulate in double, quantize ONCE at the
         * end. The old code applied ot_precision() (×F + int-round) per term —
         * H×V = 8192 separate roundings at H=256, each ±0.5 F → accumulated
         * error ±0.06 logit (~20% of measurement noise). A double sum keeps the
         * exact value and the single final cast loses only ±0.5 F. Same result
         * as before for H=1 (one term, one rounding), strictly better for H>1.
         * Verified: argmax decisions with top-2 gap < 0.06 logit are affected. */
        double acc = 0.0;
        for (int h = 0; h < H_local; h++) {
            for (int v = 0; v < V; v++) {
                int t = (int)target[TGT_IDX(k, h, v, H_local, V)];
                double p1 = (double)(nk - t + 1) / (double)(nk + 2);
                acc += log(p1);   /* full double precision, no intermediate rounding */
            }
        }
        class_offset[k] = (SCORE_TYPE)ot_precision(acc);   /* genau 1 Rundung */
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
    int _half = ki_compute_half(NC_slice);
    for (int h = 0; h < H_local; h++) {
        uint32_t h0 = ki_gb_for_neuron(in, W0, h, _ws2, NC_slice, _half);
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
static __attribute__((unused)) void export_model(const char *out_dir,
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

/* Real-type labels (ki_counter_type_str/ki_score_type_str) are now shared
 * static inline in ki-common.h — the merge needs them too (2026-08-06). */

/* Print score range (--debug-member) — SCORE_TYPE can be overridden via -D
 * (float/double/int64_t); the format must match the actual type.
 * (Was previously split by COUNTER_TYPE_IS_FLOAT — wrong for mixed builds like
 * -DSCORE_TYPE=double or float-counter+int64-scores, 2026-08-05.) */
static void ki_print_scr_minmax(SCORE_TYPE mn, SCORE_TYPE mx, int pxz) {
    _Generic((SCORE_TYPE)0,
        int64_t: printf("  MIN=%" PRId64 "  MAX=%" PRId64 "  pxz=%d\n",
                        (int64_t)mn, (int64_t)mx, pxz),
        default: printf("  MIN=%g  MAX=%g  pxz=%d\n",
                        (double)mn, (double)mx, pxz));
}

static void print_setup(int H, int epochs, int trainN, int evalN,
                          int threadN, unsigned int seed, int batchN,
                           int splitVN, int splitHN, int NC_slice, int H_local,
                           int ensembleN, int channel, int nc_per_blk, int nc_total) {
    int V = 32 / splitVN;                /* virtual neurons per container */
    size_t bit_per_cont = 32;
    size_t hidden_bit = (size_t)H * bit_per_cont;
    size_t w0_bit = (size_t)H_local * (size_t)NC_slice * bit_per_cont;
    size_t w1_bit = (size_t)KI_NCLASSES * (size_t)H_local * (size_t)V * sizeof(COUNTER_TYPE) * 8;
    int n_xf_active = aa.xform_list_count > 0 ? aa.xform_list_count : 1;
    /* Show XFORM display if any non-ID xform or pipe is active.
     * List-based, NOT bitmask: pipe IDs >= 64 are never set in aa.xforms
     * (ki_xform_bit_set guard, fix 2026-08-09). */
    int show_xform = 0;
    for (int _xi = 0; _xi < aa.xform_list_count && !show_xform; _xi++)
        if (aa.xform_list[_xi] != KI_XFORM_ID) show_xform = 1;
    int total_slots = ensembleN * n_xf_active * eff_colors * splitHN;  /* VN no longer multiplies members */
    size_t tgt_total = (size_t)H_local * KI_NCLASSES * (size_t)V * (size_t)total_slots;
    printf("══════════════════════════════════════════════════════════════════════\n");
#ifdef KI_BITVOTING
    printf("══╡ BIT-VOTING ╞══  %s  (direct pixel-bit → class, no W0/majX)\n", KI_DATASET_NAME);
#else
    printf("══╡ OTTO-SCORE ╞══  %s  %s\n", KI_DATASET_NAME, H0_STR);
#endif
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
#ifdef KI_BITVOTING
    printf("  W0:          none  (identity: gb = input, no projection)\n");
    printf("  W1:          C1[%3d] × I[%3d] ×  V[%3d] x int32 = %9zu bit  (%5.1f KB)  per member, target+offset\n",
           KI_NCLASSES, H_local, V, w1_bit, (double)w1_bit / 8 / 1024);
    printf("  ───────────────────────────────────────────────────────────\n");
    printf("  TOTAL:       W1 x (EN[%d]×CO[%d]×HN[%d]=%d) = %9zu bit  (%5.1f KB)\n",
           ensembleN, eff_colors, splitHN, total_slots,
           w1_bit * (size_t)total_slots,
           (double)(w1_bit * (size_t)total_slots) / 8 / 1024);
#else
    printf("  W0:          H0[%3d] x I0[%3d] / HN[%3d] x bin%2zu = %9zu bit  (%5.1f KB)  per member, frozen\n",
           H_local, NC_slice, splitHN, bit_per_cont,
           w0_bit, (double)w0_bit / 8 / 1024);
    printf("  W1:          C1[%3d] × H0[%3d] ×  V[%3d] x int32 = %9zu bit  (%5.1f KB)  per member, target+offset\n",
           KI_NCLASSES, H_local, V, w1_bit, (double)w1_bit / 8 / 1024);
    printf("  ───────────────────────────────────────────────────────────\n");
    if (show_xform) {
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
#endif
    printf("                                                + %zu KB target/offset (all members)\n",
           tgt_total / 1024);
    printf("  OMP:         %d threads\n", threadN);
    printf("  Score Mode:  %s  (%s counter + %s scores)\n",
           ki_score_type_str(),
               ki_counter_type_str(),
           ki_score_type_str());
    {
#ifdef KI_BITVOTING
        printf("  Majority:    none  (direct bit vote)\n");
#else
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
        printf("  Majority:    %s (%d)", maj_name, aa.maj_mode);
        if (aa.maj_mode == KI_MAJ_1 || aa.maj_mode == KI_MAJ_1R) {
            int half_v;
            half_v = ki_compute_half(NC_slice);
            printf("  half=%d (thresh=%d, nc=%d)", half_v, aa.maj1_thresh, NC_slice);
        }
        printf("\n");
#endif
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
    if (show_xform) {
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
                                    int H_local, int NC_slice, int channel) {
    int rows_factor = aa.rows_mode ? KI_ROWS : 1;
    int total = aa.member_spec_count * rows_factor;
    (void)splitVN; (void)splitHN;
    printf("\n══╡ MEMBER ╞══════════════════════════════════════════════════\n");
    printf("  Members:     %d  (%d specs × %d row%s)\n",
           total, aa.member_spec_count, rows_factor,
           rows_factor > 1 ? "s" : "");
#ifdef KI_BITVOTING
    printf("  Per member: I=%d containers × 32 bit = %d bits, Target[K=%d × I=%d × V=%d]\n",
           H_local, H_local * 32, KI_NCLASSES, H_local, 32 / splitVN);
#else
    printf("  Per member: W0[H=%d × I=%d], Target[K=%d × H=%d × V=%d]\n",
           H_local, NC_slice, KI_NCLASSES, H_local, 32 / splitVN);
#endif
    /* 3-line display from member specs (unique channels/encodings/xforms) */
    {
        char ch_buf[256] = "", en_buf[256] = "", xf_buf[256] = "";
        int ch_seen[COLOR_NB], en_seen[256], xf_seen[64];
        memset(ch_seen, 0, sizeof(ch_seen));
        memset(en_seen, 0, sizeof(en_seen));
        memset(xf_seen, 0, sizeof(xf_seen));
        for (int mi = 0; mi < aa.member_spec_count; mi++) {
            ki_MemberSpec *sp = &aa.member_spec[mi];
            /* Channel */
            if (sp->color >= 0 && sp->color < COLOR_NB && !ch_seen[sp->color]) {
                ch_seen[sp->color] = 1;
                if (ch_buf[0]) strncat(ch_buf, ", ", sizeof(ch_buf) - strlen(ch_buf) - 1);
                strncat(ch_buf, ki_color_name(sp->color), sizeof(ch_buf) - strlen(ch_buf) - 1);
            }
            /* Encoding (type+width composite) */
            int en_key = (sp->enc_type ^ (sp->enc_width << 4)) & 255;
            if (!en_seen[en_key]) {
                en_seen[en_key] = 1;
                if (en_buf[0]) strncat(en_buf, ", ", sizeof(en_buf) - strlen(en_buf) - 1);
                strncat(en_buf, ki_enc_name_short(sp->enc_type), sizeof(en_buf) - strlen(en_buf) - 1);
                char wbuf[8]; snprintf(wbuf, sizeof(wbuf), "%d", sp->enc_width);
                strncat(en_buf, wbuf, sizeof(en_buf) - strlen(en_buf) - 1);
            }
            /* Xform */
            if (sp->xform_id >= 0 && sp->xform_id < 64 && !xf_seen[sp->xform_id]) {
                xf_seen[sp->xform_id] = 1;
                if (xf_buf[0]) strncat(xf_buf, ", ", sizeof(xf_buf) - strlen(xf_buf) - 1);
                const char *n = ki_xform_str(sp->xform_id);
                strncat(xf_buf, n, sizeof(xf_buf) - strlen(xf_buf) - 1);
            }
        }
        printf("  Channels:    %s\n", ch_buf[0] ? ch_buf : "-");
        printf("  Encodings:   %s\n", en_buf[0] ? en_buf : "-");
        printf("  Xform:       %s\n", xf_buf[0] ? xf_buf : "-");
    }
    if (ensembleN > 1) {
        if (aa.ensemble_seed == ENS_SEED_CONST) {
            printf("  → ENSEMBLE x%d: one W0 shared across ALL members (const)\n",
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
static __attribute__((unused)) const char *opp_name(int ch) {
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
    int half;               /* Precomputed half for majority_tree1 (n*135/256 or exact) */
    int w0_step;            /* W0 stride per neuron (uint32), = NC_slice * pixel_groups */
    int slc_off;            /* Input-Offset for this member */
    int vi;                 /* Encoding-Index in enc_array (for stats/debug) */
    int xform_id;           /* Active xform index (for debug) */
    int color_bit;          /* Color channel bit (COLOR_R, etc., for debug) */
    int enc_type;           /* Encoding type (KI_ENC_RAW, KI_ENC_EXP, …) */
    int enc_width;          /* Encoding bit-width (8, 16, 32) */

    /* Pointer to external data (Member owns target+offset, shares W0) */
    const uint32_t *W0;     /* W0 row start (geteilt oder eigen) */
    uint32_t *input_buf;    /* owned CEX buffer for this member (train) — freed in destroy */
    uint32_t *input_buf_te; /* non-owning view into input_buf (eval split) */
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

/* ═══════════════════════════════════════════════════════════════════
 * BUILD .ENS MEMBER PATH — per-member .ens filename for --sweep
 * ═══════════════════════════════════════════════════════════════════
 * Format: {dir}/{xf_name}:{color_name}:{enc_name}{width}_{idx}.ens
 */
static int build_ens_member_path(char *buf, size_t sz,
    const char *dir, ki_Member *mem, int member_idx)
{
    /* Xform name */
    const char *xf_name = ki_xform_str(mem->xform_id);

    /* Color, encoding, width from member fields */
    const char *color_name = ki_color_name(mem->color_bit >= 0 ? mem->color_bit : 0);
    const char *enc_name   = ki_enc_name_short(mem->enc_type);
    int         enc_width  = mem->enc_width;
    char width_str[16];
    snprintf(width_str, sizeof(width_str), "%d", enc_width);
    snprintf(buf, sz, "%s/%s:%s:%s%s.ens",
             dir, xf_name, color_name, enc_name, width_str);
    return 0;
}


/* ═══════════════════════════════════════════════════════════════════
 * .META VALIDATION — check/create directory identity
 * ═══════════════════════════════════════════════════════════════════
 * Returns 0 on success (valid or created), -1 on config mismatch.
 *
 * The .meta file stores the training config that produced the archive
 * directory. Besides H/EPOCHS/VN/HN/SEED it records the majority mode
 * and maj1-thresh so a later sweep cannot silently mix incompatible
 * members into the same directory. MAJ/MAJ1_THRESH are optional fields:
 * older .meta files (written before they existed) still validate.
 */
/* CLI token for --maj (inverse of the parser in ki-common.h) */
static const char *maj_mode_token(int mode) {
#ifdef KI_BITVOTING
    /* INTENTIONAL: Bit-Voting never applies a majority (identity W0,
     * gb = input, linear voter). "-1" is the explicit "not used" marker
     * written to .meta (MAJ=-1) and .ens v11 headers (maj_token="-1")
     * so merge-ensemble can distinguish it from a real maj=1 config. */
    return "-1";
#else
    switch (mode) {
        case KI_MAJ_1:   return "1";
        case KI_MAJ_3:   return "3";
        case KI_MAJ_7:   return "7";
        case KI_MAJ_1R:  return "1r";
        case KI_MAJ_1P:  return "1p";
        case KI_MAJ_1RP: return "1rp";
        default:         return "?";
    }
#endif
}

/* ── storebackup exclusion flag (2026-08-11) ─────────────────────
 * The scores-* directories are large, regenerable archives (thousands of
 * .ens files). storebackup (the user's backup tool) must NOT mirror them.
 * Every --export-merge-scores dir gets a .storebackup_dont_backup marker.
 * Written AFTER mkdir, both on fresh creation and on existing dirs, so a
 * directory created by an older binary is flagged on the next run too. */
static void storebackup_flag_set(const char *dir) {
    char flag[1024];
    snprintf(flag, sizeof(flag), "%s/.storebackup_dont_backup", dir);
    FILE *ff = fopen(flag, "w");
    if (ff) {
        fprintf(ff, "# storebackup: do not back up this directory\n"
                    "# reason: regenerable score archive (--export-merge-scores)\n");
        fclose(ff);
    }
}

static int export_merge_scores_meta_check(const char *dir) {
    char meta_path[1024];
    snprintf(meta_path, sizeof(meta_path), "%s/.meta", dir);
    FILE *mf = fopen(meta_path, "r");
    if (mf) {
        /* Validate existing .meta */
        int m_h = 0, m_ep = 0, m_vn = 0, m_hn = 0, m_seed = 0;
        char m_maj[8] = "";        /* empty = field absent */
        int  m_majt = -999;        /* -999 = field absent */
        char m_ct[64] = "";        /* COUNTER_TYPE label (empty = absent) */
        char m_st[64] = "";        /* SCORE_TYPE label (empty = absent) */
        char m_exe[256] = "";      /* trainer exe basename (empty = absent) */
        int  m_bits = 0;           /* KI_BIT_WIDTH (0 = absent) */
        int  m_otp = 0;            /* OT_PRECISION (0 = absent) */
        char line[128];
        while (fgets(line, sizeof(line), mf)) {
            if (sscanf(line, "H=%d", &m_h) == 1) continue;
            if (sscanf(line, "EPOCHS=%d", &m_ep) == 1) continue;
            if (sscanf(line, "VN=%d", &m_vn) == 1) continue;
            if (sscanf(line, "HN=%d", &m_hn) == 1) continue;
            if (sscanf(line, "SEED=%d", &m_seed) == 1) continue;
            if (sscanf(line, "MAJ=%7s", m_maj) == 1) continue;
            if (sscanf(line, "MAJ1_THRESH=%d", &m_majt) == 1) continue;
            if (sscanf(line, "BITS=%d", &m_bits) == 1) continue;
            if (sscanf(line, "OT_PRECISION=%d", &m_otp) == 1) continue;
            /* labels contain spaces ("IEEE 754 double") → read rest of line */
            if (sscanf(line, "COUNTER_TYPE=%63[^\n]", m_ct) == 1) continue;
            if (sscanf(line, "SCORE_TYPE=%63[^\n]", m_st) == 1) continue;
            if (sscanf(line, "EXE=%255[^\n]", m_exe) == 1) continue;
        }
        fclose(mf);
        if (m_h != aa.hidden || m_ep != aa.epochs ||
            m_vn != aa.splitVN || m_hn != aa.splitHN) {
            fprintf(stderr, "[ERROR] %s: config mismatch "
                    "(H=%d/%d EP=%d/%d VN=%d/%d HN=%d/%d)\n",
                    meta_path, m_h, aa.hidden, m_ep, aa.epochs,
                    m_vn, aa.splitVN, m_hn, aa.splitHN);
            return -1;
        }
        /* Container bit width + logit scaling precision (2026-08-12):
         * BITS=KI_BIT_WIDTH and OT_PRECISION=F-scaling (F=1<<OT_PRECISION)
         * both affect the stored logits — an archive must not mix them.
         * Enforced only when present (older .meta files lack the fields). */
        if (m_bits != 0 && m_bits != KI_BIT_WIDTH) {
            fprintf(stderr, "[ERROR] %s: bit-width mismatch (BITS=%d/%d)\n",
                    meta_path, m_bits, KI_BIT_WIDTH);
            return -1;
        }
        if (m_otp != 0 && m_otp != OT_PRECISION) {
            fprintf(stderr, "[ERROR] %s: OT_PRECISION mismatch (%d/%d)\n",
                    meta_path, m_otp, OT_PRECISION);
            return -1;
        }
        /* Optional fields: only enforced when present in the file, so
         * directories created by older binaries still work. */
        /* Trainer exe guard (2026-08-11): a scores archive must not be
         * re-populated by a DIFFERENT trainer binary (e.g. 8-bit xnor vs
         * 16-bit bitvoting with identical H/BITS meta). The exe name is
         * basename(argv[0]) — stable across ./ prefix and paths. */
        if (m_exe[0] && g_exe_name[0] && strcmp(m_exe, g_exe_name) != 0) {
            fprintf(stderr, "[ERROR] %s: trainer exe mismatch "
                    "(archive=%s, this=%s)\n",
                    meta_path, m_exe, g_exe_name);
            return -1;
        }
#ifndef KI_BITVOTING
        /* Otto: majority is a real config — must match. */
        if (m_maj[0] && strcmp(m_maj, maj_mode_token(aa.maj_mode)) != 0) {
            fprintf(stderr, "[ERROR] %s: maj mismatch (maj=%s/%s)\n",
                    meta_path, m_maj, maj_mode_token(aa.maj_mode));
            return -1;
        }
        if (m_majt != -999 && m_majt != aa.maj1_thresh) {
            fprintf(stderr, "[ERROR] %s: maj1-thresh mismatch (maj1-thresh=%d/%d)\n",
                    meta_path, m_majt, aa.maj1_thresh);
            return -1;
        }
        /* Arithmetic types (2026-06-06): only enforced when present in the
         * file — the precision of the score accumulation matters (float32
         * drifts, selection 29→44 members). A directory must not mix
         * float32/double/int64-built archives. */
        if (m_ct[0] && strcmp(m_ct, ki_counter_type_str()) != 0) {
            fprintf(stderr, "[ERROR] %s: COUNTER_TYPE mismatch (%s/%s)\n",
                    meta_path, m_ct, ki_counter_type_str());
            return -1;
        }
        if (m_st[0] && strcmp(m_st, ki_score_type_str()) != 0) {
            fprintf(stderr, "[ERROR] %s: SCORE_TYPE mismatch (%s/%s)\n",
                    meta_path, m_st, ki_score_type_str());
            return -1;
        }
#else
        /* INTENTIONAL: Bit-Voting never applies a majority — the .meta MAJ
         * field is informational only. Older BV directories carry MAJ=1
         * (trainer default); new ones MAJ=-1 (explicit no-majority marker).
         * Neither is a config mismatch here. */
        (void)m_maj; (void)m_majt;
#endif
        /* Existing dir: ensure the storebackup exclusion flag is present
         * (older binaries created the dir without it). */
        storebackup_flag_set(dir);
        return 0;
    }
    /* Create .meta + directory */
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", dir);
    if (system(cmd) != 0) return -1;
    /* Fresh dir: set the storebackup exclusion flag BEFORE any .ens files
     * are written, so the backup tool never sees a partial archive. */
    storebackup_flag_set(dir);
    mf = fopen(meta_path, "w");
    if (mf) {
        fprintf(mf, "H=%d\nEPOCHS=%d\nVN=%d\nHN=%d\nSEED=%d\n"
                    "MAJ=%s\nMAJ1_THRESH=%d\n"
                    "BITS=%d\n"
                    "OT_PRECISION=%d\n"
                    "COUNTER_TYPE=%s\nSCORE_TYPE=%s\n"
                    "EXE=%s\n",
                aa.hidden, aa.epochs, aa.splitVN, aa.splitHN, aa.seed,
                maj_mode_token(aa.maj_mode), aa.maj1_thresh,
                KI_BIT_WIDTH, OT_PRECISION,
                ki_counter_type_str(), ki_score_type_str(),
                g_exe_name[0] ? g_exe_name : "(unknown)");
        fclose(mf);
    }
    return 0;
}


/* ═══════════════════════════════════════════════════════════════════
 * EXPORT ONE .ENS FILE — exactly one member to one .ens file
 * ═══════════════════════════════════════════════════════════════════
 *
 * Writes a single .ens file (v7 format) for one member immediately after
 * training.  Called per-member, NOT batched at end.
 *
 * Parameters:
 *   mem          — trained member (W0, target, offset, gb_buf_te must be valid)
 *   member_idx   — 0-based global member index (for unique filename)
 *   dir          — output directory
 *   y            — ground truth labels [N]
 *   N            — number of eval samples
 *   n_cont       — containers per sample (stride)
 *
 * Returns 0 on success, -1 on error.
 */
static int export_one_member_ens(ki_Member *mem, int member_idx,
    const char *dir, const uint8_t *y, int N, int n_cont)
{
    if (N <= 0) return -1;

    /* Validate .meta first (creates dir if needed) */
    if (export_merge_scores_meta_check(dir) != 0) return -1;

    /* Build filename */
    char fname[512];
    build_ens_member_path(fname, sizeof(fname), dir, mem, member_idx);

    /* ── Write via the shared .ens library (lib/ki-ens.h) ──
     * The version follows the internal SCORE_TYPE (v11=float, v12=double,
     * v13=int64) — the archive preserves the computation precision
     * (decision 2026-08-06). ens_write + ens_verify (roundtrip) make the
     * writer self-checking: a version/width regression is caught HERE at
     * write time, not at merge time (bug 2026-08-06: "0 blocks filled"
     * SEGV — the version bound was missed in one reader path). */
    size_t score_sz = (size_t)N * (size_t)KI_NCLASSES;
    EnsWriteCfg cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_test    = (uint32_t)N;
    cfg.n_classes = (uint32_t)KI_NCLASSES;
    cfg.n_members = 1;
    /* KI_BITVOTING: aa.hidden is repurposed as the I=H container count
     * (set in main) — .meta and .ens headers stay consistent, and
     * merge-ensemble validates H across the directory. */
    cfg.hidden    = (uint32_t)aa.hidden;
    cfg.epochs    = (uint32_t)aa.epochs;
    cfg.split_vn  = (uint32_t)aa.splitVN;
    cfg.split_hn  = (uint32_t)aa.splitHN;
    cfg.seed      = aa.seed;
    cfg.target_err = 0.0f;
    cfg.w0_marker = mem->W0 ? mem->W0[0] : 0;
    strncpy(cfg.maj_token, maj_mode_token(aa.maj_mode), sizeof(cfg.maj_token) - 1);
    cfg.maj1_thresh = aa.maj1_thresh;
    /* v14+ precision block: how the stored logits were computed (the merge
     * --check validates archives against these AND against the .meta). */
    cfg.ot_precision = (int32_t)OT_PRECISION;
    cfg.bit_width    = (int32_t)KI_BIT_WIDTH;
    snprintf(cfg.counter_type, sizeof(cfg.counter_type), "%.23s",
             ki_counter_type_str());

    char wid_str[16];
    snprintf(wid_str, sizeof(wid_str), "%d", mem->enc_width);
    cfg.member_fields[0] = ki_color_name(mem->color_bit >= 0 ? mem->color_bit : 0);
    cfg.member_fields[1] = ki_enc_name_short(mem->enc_type);
    cfg.member_fields[2] = wid_str;
    cfg.member_fields[3] = ki_xform_str(mem->xform_id);

    /* ── Compute member eval accuracy from scores ── */
    SCORE_TYPE *sc = (SCORE_TYPE *)calloc(score_sz, sizeof(SCORE_TYPE));
    if (!sc) { fprintf(stderr, "[FATAL] OOM\n"); return -1; }

    int member_ok = 0;
    if (mem->gb_buf_te) {
        /* Fast path: use cached gb_buf_te */
        #pragma omp parallel for schedule(static) reduction(+:member_ok) \
            if(!g_parallel_members)
        for (int s = 0; s < N; s++) {
            SCORE_TYPE scc[KI_NCLASSES];
            scores_otto_from_gb(s, mem->H_local, mem->gb_buf_te,
                               mem->target, mem->offset, scc);
            for (int k = 0; k < KI_NCLASSES; k++)
                sc[(size_t)s * KI_NCLASSES + (size_t)k] = scc[k];
            int pred = 0;
            for (int k = 1; k < KI_NCLASSES; k++)
                if (scc[k] > scc[pred]) pred = k;
            if (pred == (int)y[s]) member_ok++;
        }
    } else {
        /* Fallback: raw pixel path */
        #pragma omp parallel for schedule(static) reduction(+:member_ok) \
            if(!g_parallel_members)
        for (int s = 0; s < N; s++) {
            SCORE_TYPE scc[KI_NCLASSES];
            scores_otto(mem->input_buf_te + (size_t)s * (size_t)mem->NC_slice + mem->slc_off,
                       mem->W0, mem->H_local, mem->NC_slice,
                       mem->target, mem->offset, scc);
            for (int k = 0; k < KI_NCLASSES; k++)
                sc[(size_t)s * KI_NCLASSES + (size_t)k] = scc[k];
            int pred = 0;
            for (int k = 1; k < KI_NCLASSES; k++)
                if (scc[k] > scc[pred]) pred = k;
            if (pred == (int)y[s]) member_ok++;
        }
    }

    cfg.member_eval = (float)member_ok * 100.0f / (float)N;

    /* ── Write + roundtrip-verify via the shared library ──
     * ens_write stores the internal SCORE_TYPE natively (float→v11 4 B,
     * double→v12 8 B, int64→v13 8 B); ens_verify reads it back and checks
     * version/width/config/scores/labels — a format regression fails HERE
     * on the first member, not after a full corpus (bug 2026-08-06). */
    int wrc = ens_write(fname, &cfg, y, N, sc, score_sz);
    if (wrc == 0)
        wrc = ens_verify(fname, &cfg, y, N, sc, score_sz);
    free(sc);
    if (wrc != 0) {
        fprintf(stderr, "[WARN] .ens export/verify failed: %s\n", fname);
        return -1;
    }
    return 0;
}


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

    uint32_t magic = OTTO_MAGIC, ver = OTTO_VERSION_V8,
        /* v8 = v7 + maj_mode/maj1_thresh/splitVN/splitHN header fields
         * (self-describing IFC — the import must know the exact majority
         * calibration to reconstruct h0 bits, bug 2026-08-09: 36.6% import
         * without the CLI --maj1-thresh 104 that the training used).
         * Layout/weights unchanged vs v7. */
        mode = H0_MODE_VAL;
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
    /* v8: majority + split calibration — the import overrides aa.* with these
     * (model wins; CLI only for legacy v5-v7 models). */
    uint32_t majm = (uint32_t)aa.maj_mode;
    uint32_t majt = (uint32_t)aa.maj1_thresh;
    uint32_t svn  = (uint32_t)aa.splitVN;
    uint32_t shn  = (uint32_t)aa.splitHN;
    fwrite(&majm,4,1,f); fwrite(&majt,4,1,f);
    fwrite(&svn,4,1,f);  fwrite(&shn,4,1,f);

    /* ── v7: per-member metadata (4 length-prefixed strings: channel,
     * encoding, encoding-width, xform) — mirrors the .ens v7+ strings. */
    for (int m = 0; m < total_members; m++) {
        const char *col = "", *enc = "", *wid = "", *xf = "";
        char wid_buf[16];
        for (int b = 0; b < active_members; b++) {
            if (members[b]->orig_m == m) {
                col = ki_color_name(members[b]->color_bit >= 0 ? members[b]->color_bit : 0);
                enc = ki_enc_name_short(members[b]->enc_type);
                snprintf(wid_buf, sizeof(wid_buf), "%d", members[b]->enc_width);
                wid = wid_buf;
                xf  = ki_xform_str(members[b]->xform_id);
                break;
            }
        }
        const char *meta[4] = { col, enc, wid, xf };
        for (int i = 0; i < 4; i++) {
            uint8_t l = (uint8_t)strlen(meta[i]);
            fwrite(&l, 1, 1, f);
            fwrite(meta[i], 1, l, f);
        }
    }

    size_t w0_bytes = (size_t)H_local * (size_t)NC_slice * 4;
    size_t tgt_bytes = (size_t)H_local * KI_NCLASSES * (size_t)VN_GROUPS_ * 4;
    size_t off_bytes = KI_NCLASSES * 8;
    size_t total = 0;

    for (int m = 0; m < total_members; m++) {
#ifdef KI_BITVOTING
        /* BIT-VOTING: no W0 — write zeros (format-compatible; the IFC
         * forward path reads gb = input directly and ignores W0). */
        COUNTER_TYPE *_w0z = (COUNTER_TYPE *)calloc((size_t)H_local * (size_t)NC_slice, sizeof(COUNTER_TYPE));
        fwrite(_w0z, sizeof(COUNTER_TYPE), (size_t)H_local * (size_t)NC_slice, f);
        free(_w0z);
#else
        fwrite(W0_ens + (size_t)m * (size_t)H_local * (size_t)NC_slice, sizeof(uint32_t), (size_t)H_local * (size_t)NC_slice, f);
#endif
        /* Find member with orig_m == m, or write zeros if filtered out */
        int found = 0;
        for (int b = 0; b < active_members && !found; b++) {
            if (members[b]->orig_m == m) {
                fwrite(members[b]->target, sizeof(COUNTER_TYPE), (size_t)H_local * KI_NCLASSES * (size_t)VN_GROUPS_, f);
                fwrite(members[b]->offset, sizeof(SCORE_TYPE), KI_NCLASSES, f);
                found = 1;
            }
        }
        if (!found) {
            COUNTER_TYPE *zeros = (COUNTER_TYPE *)calloc((size_t)H_local * KI_NCLASSES * (size_t)VN_GROUPS_, sizeof(COUNTER_TYPE));
            fwrite(zeros, sizeof(COUNTER_TYPE), (size_t)H_local * KI_NCLASSES * (size_t)VN_GROUPS_, f);
            free(zeros);
            SCORE_TYPE *oz = (SCORE_TYPE *)calloc(KI_NCLASSES, sizeof(SCORE_TYPE));
            fwrite(oz, sizeof(SCORE_TYPE), KI_NCLASSES, f);
            free(oz);
        }
        total += w0_bytes + tgt_bytes + off_bytes;
    }

    fclose(f);

    printf("\n══╡ EXPORT ╞═══════════════════════════════════════════════════\n");
    printf("  Model:  %s  (v%u, %d members, H_local=%d, NC_slice=%d, F=%d)\n",
           path, ver, total_members, H_local, NC_slice, 1<<OT_PRECISION);
    printf("  Total:  %zu KB (%d × (W0=%zuKB + Tgt=%zuKB + Off=%zuB))\n",
           (24 + total) / 1024, total_members,
           w0_bytes / 1024, tgt_bytes / 1024, off_bytes);
    fflush(stdout);
}

/* ── Inline helper: allocate all member buffers (target, offset, h0/gb) ──
 * Extracted from ki_member_create to reduce boilerplate and centralize
 * allocation logic. Called once per member at creation. */
static inline void ki_member_alloc_buffers(ki_Member *m, int H_local, int NC_slice,
                                           int total_train, int total_eval) {
    int V = 32 / aa.splitVN;  /* VN_GROUPS_ computed from aa.splitVN */
    size_t tgt_sz = (size_t)H_local * KI_NCLASSES * (size_t)V;

    m->target       = (COUNTER_TYPE *)ki_xcalloc(tgt_sz, sizeof(COUNTER_TYPE));
    m->offset       = (SCORE_TYPE *)ki_xcalloc(KI_NCLASSES, sizeof(SCORE_TYPE));
    m->best_target  = (COUNTER_TYPE *)ki_xcalloc(tgt_sz, sizeof(COUNTER_TYPE));
    m->best_offset  = (SCORE_TYPE *)ki_xcalloc(KI_NCLASSES, sizeof(SCORE_TYPE));

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
        m->h0_buf     = NULL;
        m->gb_buf     = NULL;
        m->gb_buf_te  = NULL;
    }
}

/* ── Inline helper: free all member buffers (except input_buf/input_buf_te) ──
 * Extracted from ki_member_destroy. input_buf/input_buf_te are NOT freed here
 * because they have different ownership semantics (see ki_member_destroy). */
static inline void ki_member_free_buffers(ki_Member *m) {
    free(m->target);       m->target = NULL;
    free(m->offset);       m->offset = NULL;
    free(m->best_target);  m->best_target = NULL;
    free(m->best_offset);  m->best_offset = NULL;
    free(m->err_target);   m->err_target = NULL;
    free(m->err_offset);   m->err_offset = NULL;
    free(m->h0_buf);       m->h0_buf = NULL;
    free(m->gb_buf);       m->gb_buf = NULL;
    free(m->gb_buf_te);    m->gb_buf_te = NULL;
}

/* ── Member erzeugen: alloziert target, offset, h0_buf, gb_buf* ── */
static ki_Member *ki_member_create(int H_local, int NC_slice, int slc_off,
                                    const uint32_t *W0, int total_train,
                                    int total_eval) {
    ki_Member *m = (ki_Member *)malloc(sizeof(ki_Member));
    if (!m) { fprintf(stderr, "[FATAL] ki_member_create OOM\n"); exit(1); }
    m->H_local  = H_local;
    m->NC_slice = NC_slice;
    m->half     = ki_compute_half(NC_slice);
    m->w0_step  = NC_slice;  /* default: stride = containers (can be overridden for pixel-maj) */
    m->slc_off  = slc_off;
    m->W0       = W0;
    m->step     = 0;
    m->last_err = 0;
    m->ep       = 0;
    m->trn_acc  = 100.0f;  /* initially: all members participate */

    /* CEX input buffers: the member does NOT own them at creation — the
     * trainer's member loop sets them via load_input_cex_cached() later.
     * Must be NULL here: ki_member_destroy() frees input_buf unconditionally,
     * and the IFC/--import path never sets it (created + destroyed without
     * loading) — without NULL init it freed garbage → ASan SEGV in
     * ki_member_destroy (repro: make test-import, 0xbebebebe pattern).
     * See: bugs/bug-2026-08-05-import-uninit-input-buf.md */
    m->input_buf    = NULL;
    m->input_buf_te = NULL;

    /* Allocate all other buffers via inline helper */
    ki_member_alloc_buffers(m, H_local, NC_slice, total_train, total_eval);

    return m;
}

/* ── Member freigeben ─────────────────────────────────────────── */
static void ki_member_destroy(ki_Member *m) {
    if (!m) return;
    /* CEX (2026-08-04): each member owns its narrow input buffer
     * (load_input_cex_cached). input_buf_te is a non-owning view into the
     * same allocation (train/eval split) — freed once via input_buf. */
    if (m->input_buf) { free(m->input_buf); m->input_buf = NULL; }
    m->input_buf_te = NULL;

    /* Free all other buffers via inline helper */
    ki_member_free_buffers(m);

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
    /* PRE-COMPUTE phase: runs OUTSIDE the PRF member loop (sequentially over
     * members) — safe to parallelize the inner sample loop with all cores.
     * Only when the member loop is single (few members): gate on
     * !g_parallel_members, thread count = g_prf_width locally (the global
     * limit stays 1 for the member loop). Bug 2026-08-09: 1 member was
     * single-threaded = 13.6s vs 3.6s. */
    #pragma omp parallel for schedule(static) num_threads(g_prf_width) \
        if(!g_parallel_members)
    for (int s = 0; s < N; s++) {
        const uint32_t *in = in_base + (size_t)s * (size_t)n_cont;
        for (int h = 0; h < m->H_local; h++) {
            size_t idx = (size_t)s * (size_t)m->H_local + (size_t)h;
            m->gb_buf_te[idx] = h0_to_gb(
                ki_gb_for_neuron(in, m->W0, h, m->w0_step, m->NC_slice, m->half));
        }
    }
}

/* ── Member: h0 + gb vorberechnen (once, jede Epoche wiederverwendet) ── *
 * gb_buf = VN group mask from h0 (vermeidet popcount reduction im Training). */
static void ki_member_compute_h0(ki_Member *m, const uint32_t *X, int N,
                                  int n_cont) {
    const uint32_t *in_base = X + (size_t)m->slc_off;
    /* PRE-COMPUTE phase: outside the PRF member loop — sample-parallel with
     * all cores only when the member loop is single (g_parallel_members==0) */
    #pragma omp parallel for schedule(static) num_threads(g_prf_width) \
        if(!g_parallel_members)
    for (int s = 0; s < N; s++) {
        const uint32_t *in = in_base + (size_t)s * (size_t)n_cont;
        for (int h = 0; h < m->H_local; h++) {
            size_t idx = (size_t)s * (size_t)m->H_local + (size_t)h;
            uint32_t hv = ki_gb_for_neuron(in, m->W0, h, m->w0_step, m->NC_slice, m->half);
            m->h0_buf[idx] = hv;
            m->gb_buf[idx] = h0_to_gb(hv);
        }
    }
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

/* ── H0/GB value for neuron h of a member (definition) ────────────
 * Normal Otto: h0_neuron (XNOR with W0 row + majority) → h0_to_gb.
 * KI_BITVOTING: identity — the input container h IS the gb value
 * (direct pixel-bit → class voting, no W0, no majority). */
static inline uint32_t ki_gb_for_neuron(const uint32_t *in, const uint32_t *W0_row,
                                        int h, int w0_step, int NC_slice, int half) {
#ifdef KI_BITVOTING
    (void)W0_row; (void)w0_step; (void)NC_slice; (void)half;
    return in[h];
#else
    return h0_to_gb(h0_neuron(in, W0_row + (size_t)h * (size_t)w0_step, NC_slice, half));
#endif
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

    /* EXCLUSIVE PARALLELISM (bug 2026-08-10) — SAME logic as the training
     * member loop: members parallel XOR samples parallel, never nested.
     * With 3+ members the MEMBER loop runs parallel (1 thread per member)
     * and each member-thread accumulates into its OWN vote block (no race);
     * with 1-2 members the member loop runs single and the SAMPLE loop
     * uses all cores (each s writes a disjoint votes[s] row). */
    g_parallel_members = (active_members >= 3) ? 1 : 0;
    int n_threads_mem = g_parallel_members
        ? (active_members < g_prf_width ? active_members : g_prf_width) : 1;
    if (n_threads_mem < 1) n_threads_mem = 1;

    /* Votes-Accumulator: n_threads_mem blocks of N × KI_NCLASSES.
     * Block 0 is the merge target; in the member-parallel case blocks
     * 1..n_threads_mem-1 hold the other threads' partial votes. */
    SCORE_TYPE (*votes)[KI_NCLASSES] = (SCORE_TYPE (*)[KI_NCLASSES])calloc(
        (size_t)n_threads_mem * (size_t)N, sizeof(SCORE_TYPE[KI_NCLASSES]));
    if (!votes) { fprintf(stderr, "[FATAL] evaluate: votes OOM\n"); exit(1); }

    /* Each member gets equal voting power: Scale sc.*so that
     * max|sc[k]| ≤ SCALE_MAX. Prevents members with large
     * target weights (more corrections, different channels) from dominating.
     * Type-generic: 16777216 is value-identical for float/double/int64
     * (was split by COUNTER_TYPE_IS_FLOAT; the int branch's `<< 24` breaks
     * with -DSCORE_TYPE=double, 2026-08-05). */
    #define VOTE_SCALE ((SCORE_TYPE)16777216)

    /* Members outer: target.*stays warm in L1 cache */
    #pragma omp parallel for schedule(static) if(g_parallel_members) \
        num_threads(n_threads_mem)
    for (int m = 0; m < active_members; m++) {
        ki_Member *mem = members[m];
        int tid = g_parallel_members ? omp_get_thread_num() : 0;
        SCORE_TYPE (*votes_m)[KI_NCLASSES] = votes + (size_t)tid * (size_t)N;

        /* ── Skip members below accuracy threshold ────────────── */
        if (aa.member_threshold > 0 && mem->trn_acc < (float)aa.member_threshold)
            continue;

        #pragma omp parallel for schedule(static) if(!g_parallel_members)
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
                /* ── Rohpixel → h0_neuron (Test-Set ohne gb_buf).
                 * v7 IFC import: per-member input buffer (input_buf set,
                 * X==NULL, slc_off==0); otherwise the shared X + slc_off
                 * (v6 fallback / no-precompute path). */
                const uint32_t *in = mem->input_buf
                    ? mem->input_buf + (size_t)s * (size_t)mem->NC_slice
                    : X + (size_t)s * (size_t)n_cont + mem->slc_off;
                scores_otto(in, mem->W0, mem->H_local, mem->NC_slice,
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
                        votes_m[s][k] += sc[k] * VOTE_SCALE / max_abs;
                } else {
                    for (int k = 0; k < KI_NCLASSES; k++)
                        votes_m[s][k] += sc[k];
                }
            } else {
                for (int k = 0; k < KI_NCLASSES; k++)
                    votes_m[s][k] += sc[k];
            }
        }
    }
    #undef VOTE_SCALE

    /* Merge partial vote blocks (member-parallel case) into block 0 */
    for (int t = 1; t < n_threads_mem; t++) {
        SCORE_TYPE (*vt)[KI_NCLASSES] = votes + (size_t)t * (size_t)N;
        for (int s = 0; s < N; s++)
            for (int k = 0; k < KI_NCLASSES; k++)
                votes[s][k] += vt[s][k];
    }

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
static __attribute__((unused)) void print_member_debug(ki_Member **members, int active_members,
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
    int *correct_ens = (int *)calloc((size_t)KI_NCLASSES, sizeof(int));
    if (!total || !correct || !correct_ens) {
        fprintf(stderr, "[FATAL] print_class_voting_debug OOM\n");
        free(total); free(correct); free(correct_ens); exit(1);
    }

    /* ── First pass: count samples per class ────────────────── */
    for (int s = 0; s < N; s++) {
        int k = (int)y[s];
        if (k >= 0 && k < KI_NCLASSES) total[k]++;
    }

    /* ── Zweiter Pass: per Sample alle Member + ENSEMBLE (2026-08-12) ──
     * Sample-outer loop so the per-class score SUM over all members can be
     * accumulated — the ensemble decision is argmax of that sum, NOT the
     * argmax of any single member (a member with low individual recall can
     * still tip the sum, e.g. avg4@spiral:sig8 +3 Bag hits in member-8.out).
     * The "ens-total" table row shows this ensemble recall per class.
     * FIX (2026-08-12): in PRF mode every member has its OWN CEX input
     * (mem->gb_buf_te) because members differ in channel/encoding width —
     * the global X buffer (X_te) is wrong per member. That made ens-total
     * disagree with the confusion matrix (Pullover 0% vs 81.9%). Use the
     * member's precomputed gb when available; fall back to X (seq mode). */
    for (int s = 0; s < N; s++) {
        SCORE_TYPE ens_sum[KI_NCLASSES];
        for (int k = 0; k < KI_NCLASSES; k++) ens_sum[k] = 0;
        for (int m = 0; m < active_members; m++) {
            ki_Member *mem = members[m];
            SCORE_TYPE sc[KI_NCLASSES];
            if (mem->gb_buf_te) {
                scores_otto_from_gb(s, mem->H_local, mem->gb_buf_te,
                                    mem->target, mem->offset, sc);
            } else {
                scores_otto(X + (size_t)s * (size_t)n_cont + mem->slc_off,
                            mem->W0, mem->H_local, mem->NC_slice,
                            mem->target, mem->offset, sc);
            }
            int pred = 0;
            for (int k = 1; k < KI_NCLASSES; k++)
                if (sc[k] > sc[pred]) pred = k;
            if (pred == (int)y[s]) {
                int true_k = (int)y[s];
                if (true_k >= 0 && true_k < KI_NCLASSES)
                    correct[m][true_k]++;
            }
            for (int k = 0; k < KI_NCLASSES; k++)
                ens_sum[k] += sc[k];
        }
        int ens_pred = 0;
        for (int k = 1; k < KI_NCLASSES; k++)
            if (ens_sum[k] > ens_sum[ens_pred]) ens_pred = k;
        if (ens_pred == (int)y[s]) {
            int true_k = (int)y[s];
            if (true_k >= 0 && true_k < KI_NCLASSES)
                correct_ens[true_k]++;
        }
    }

    /* ── Tabelle ausgeben ──────────────────────────────────────── */
    /* Only classes with samples anzeigen (important with --filter) */
    int active_cols[KI_NCLASSES], n_active = 0;
    for (int k = 0; k < KI_NCLASSES; k++)
        if (total[k] > 0) active_cols[n_active++] = k;

    if (n_active == 0) { free(total); free(correct); return; }

    /* ASCII table via libtprint (GPL-3.0, vendored otto-score-ifc/lib/
     * tprint.c) — the library computes the per-column width from the
     * widest header/value cell itself, so the table is ALWAYS exactly
     * column-aligned (fix 2026-08-12: hand-rolled %-7s broke on long
     * class names like "T-shirt/top" / "Ankle boot"). The specialist's
     * target column(s) from "# META: ... TARGET=3,7" are marked with
     * "*" around the name (feature 2026-08-12: "one specialist = one
     * question" — target must beat all other specialists on its column). */
    TPrint *tp = tprint_create(stdout, TRUE, TRUE, 2, 2);
    tprint_set_double_fmt(tp, "%5.1f%%");
    tprint_column_add(tp, "member", TPAlign_center, TPAlign_left);
    for (int ai = 0; ai < n_active; ai++) {
        int k = active_cols[ai];
        char cap[32];
        if (aa.member_target_mask & (1 << k))
            snprintf(cap, sizeof(cap), "*%s*", ki_class_names[k]);
        else
            snprintf(cap, sizeof(cap), "%s", ki_class_names[k]);
        tprint_column_add(tp, cap, TPAlign_center, TPAlign_right);
    }
    tprint_column_add(tp, "avg", TPAlign_center, TPAlign_right);

    int *ok_member = (int *)calloc((size_t)active_members, sizeof(int));
    for (int m = 0; m < active_members; m++) {
        /* ── Member-Label (XF:CHAN:ENC format, fix 2026-08-12) ──
         * Old format "#%d mnist=sig8" dropped the xform — two members with
         * different xforms (avg4@spiral vs sft-u3@spiral) both showed as
         * "mnist=sig8" and were indistinguishable. Now matches the
         * --member-out spec: "#%d <xform>:<channel>:<enc><width>", e.g.
         * "#0 avg4@spiral:mnist:sig8". Uses the debug fields that the
         * member-file loader / xform generator fill (Z. 3451-3454). */
        char label[128];
        ki_Member *mem = members[m];
        const char *xn = (mem->xform_id >= 0) ? ki_xform_str(mem->xform_id) : "?";
        const char *cn = (mem->color_bit >= 0) ? ki_color_name(mem->color_bit) : "?";
        const char *en = ki_enc_name_short((int8_t)mem->enc_type);
        snprintf(label, sizeof(label), "#%d %s:%s:%s%d",
                 m, xn, cn, en, mem->enc_width);
        tprint_data_add_str(tp, 0, label);

        int member_ok = 0;
        for (int ai = 0; ai < n_active; ai++) {
            int k = active_cols[ai];
            if (total[k] > 0) {
                double pct = (double)correct[m][k] * 100.0 / (double)total[k];
                tprint_data_add_double(tp, ai + 1, pct);
                member_ok += correct[m][k];
            } else {
                tprint_data_add_str(tp, ai + 1, "-");
            }
        }
        ok_member[m] = member_ok;
        double avg = (double)member_ok * 100.0 / (double)N;
        tprint_data_add_double(tp, n_active + 1, avg);
    }

    /* ── ENSEMBLE total row (2026-08-12): the SUM of all member scores is
     * the ensemble decision (same semantics as merge-ensemble). Per-class
     * recall of the ensemble — the single-member evl% values above are NOT
     * the relevant measure for "does the ensemble win this class". */
    {
        int ens_ok = 0;
        tprint_data_add_str(tp, 0, "ens-total");
        for (int ai = 0; ai < n_active; ai++) {
            int k = active_cols[ai];
            if (total[k] > 0) {
                double pct = (double)correct_ens[k] * 100.0 / (double)total[k];
                tprint_data_add_double(tp, ai + 1, pct);
                ens_ok += correct_ens[k];
            } else {
                tprint_data_add_str(tp, ai + 1, "-");
            }
        }
        double avg = (double)ens_ok * 100.0 / (double)N;
        tprint_data_add_double(tp, n_active + 1, avg);
    }

    printf("\n  ── Class-voting stats (Ep %d) ──────────────────────────────\n", ep + 1);
    tprint_print(tp);
    tprint_free(tp);
    printf("\n");
    fflush(stdout);

    free(total); free(correct); free(correct_ens); free(ok_member);
}

/* ═══════════════════════════════════════════════════════════════════
 * IFC MODEL LOAD — read exported .otto file
 * ═══════════════════════════════════════════════════════════════════
 * Returns arrays allocated by the caller (must free).
 */
typedef char EnsMemberMeta[4][64];   /* v7 per-member channel/enc/width/xform strings */

/* name→ID lookups for the IFC import (v7 per-member metadata). The tables
 * live in ki-encoding.h, the xform parse in ki-common.h — kept local here to
 * avoid include-order coupling (duplicates merge-ensemble.c helpers). */
static int ifc_color_id(const char *name) {
    for (int i = 0; i < (int)(sizeof(ki_color_table)/sizeof(ki_color_table[0])); i++)
        if (strcmp(ki_color_table[i].name, name) == 0) return ki_color_table[i].id;
    return -1;
}
static int ifc_enc_id(const char *name) {
    for (int i = 0; i < (int)(sizeof(ki_enc_table)/sizeof(ki_enc_table[0])); i++)
        if (strcmp(ki_enc_table[i].name, name) == 0) return ki_enc_table[i].id;
    return -1;
}
static int ifc_xform_id(const char *name) {
    int id = ki_xform_parse_or_pipe(name);
    if (id < 0) {
        for (int i = 0; i < (int)(sizeof(ki_xform_table)/sizeof(ki_xform_table[0])); i++)
            if (strcmp(ki_xform_table[i].name, name) == 0) return ki_xform_table[i].id;
    }
    return id;
}

static int ifc_load_model(const char *path,
                          uint32_t **W0_out, COUNTER_TYPE **tgt_out, SCORE_TYPE **off_out,
                          int *n_mem, int *H_local, int *NC_slice,
                          EnsMemberMeta **meta_out) {
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
    /* v8: majority + split calibration — the MODEL is authoritative for the
     * h0/gb computation (maj1_thresh drives the half threshold). Without it
     * the import fell back to CLI defaults (maj1-thresh -2 = auto 107) while
     * the model was trained with e.g. 104 → every h0 bit differed → 36.6%
     * (bug 2026-08-09). Legacy v5-v7 models keep CLI/aa defaults. */
    {
        uint32_t majm = (uint32_t)aa.maj_mode;
        uint32_t majt = (uint32_t)aa.maj1_thresh;
        uint32_t svn  = (uint32_t)aa.splitVN;
        uint32_t shn  = (uint32_t)aa.splitHN;
        if (ver >= OTTO_VERSION_V8) {
            if (fread(&majm,4,1,f)!=1 || fread(&majt,4,1,f)!=1 ||
                fread(&svn,4,1,f)!=1  || fread(&shn,4,1,f)!=1) {
                fprintf(stderr, "[FATAL] Truncated v8 header from %s\n", path);
                fclose(f); return -1;
            }
        }
        aa.maj_mode    = (int)majm;
        aa.maj1_thresh = (int)majt;
        aa.splitVN     = (int)svn;
        aa.splitHN     = (int)shn;
    }
    /* v7: per-member metadata section (channel, enc, width, xform strings) —
     * needed to reconstruct heterogeneous ensembles (bug 2026-08-06: CIFAR
     * import gave 12.81% because the model carried no per-member channel). */
    *meta_out = NULL;
    if (ver >= OTTO_VERSION_V7) {
        EnsMemberMeta *meta = (EnsMemberMeta *)malloc((size_t)n_members * sizeof(EnsMemberMeta));
        if (!meta) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
        char buf[64];
        for (uint32_t m = 0; m < n_members; m++) {
            for (int i = 0; i < 4; i++) {
                uint8_t slen;
                if (fread(&slen, 1, 1, f) != 1 || slen >= 64) {
                    fprintf(stderr, "[FATAL] Bad v7 metadata (member %u, field %d)\n", m, i);
                    free(meta); fclose(f); return -1;
                }
                if (fread(buf, 1, slen, f) != slen) {
                    fprintf(stderr, "[FATAL] Bad v7 metadata (member %u, field %d)\n", m, i);
                    free(meta); fclose(f); return -1;
                }
                buf[slen] = '\0';
                snprintf(meta[m][i], 64, "%s", buf);
            }
        }
        *meta_out = meta;
    }
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
            if (*meta_out) free(*meta_out);
            fclose(f); return -1;
        }
    }
    fclose(f);
    printf("  Model: v%u  H=%d  NC=%d  ensemble=%d  mode=%s  F=%d%s\n",
           ver, h_local, ncs_slice, n_members, mode==0?"XNOR":"XOR", 1<<prec,
           ver >= OTTO_VERSION_V7 ? (ver >= OTTO_VERSION_V8 ? "  (v8: self-describing)" : "  (v7: per-member metadata)") : "");
    /* v8: show the majority/split calibration the import will use — the
     * exact half threshold drives every h0 bit (bug 2026-08-09). */
    printf("  Maj:    maj=%d  maj1-thresh=%d  splitVN=%d  splitHN=%d\n",
           aa.maj_mode, aa.maj1_thresh, aa.splitVN, aa.splitHN);
    return (int)n_members;
}

/* CEX input cache (load_input_cex_cached) is defined in lib/ki-load.h —
 * shared by Otto and bitvoting trainers. The OLD hash-based load_input_cached
 * / load_input_cached_xform were removed (2026-08-05): their <hash>_*.pre
 * format had the stale-cache bug (bug 2026-08-04) and is retired.
 * See: bugs/bug-2026-08-05-sweep-old-pre-format.md */


/* ═══════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[]) {
    /* Record the trainer executable basename (for the .meta EXE= guard).
     * basename(): strips any ./ or path prefix so the same binary invoked
     * via ./fashion-... and /abs/path/fashion-... matches. */
    {
        const char *b = strrchr(argv[0], '/');
        const char *exe = b ? b + 1 : argv[0];
        snprintf(g_exe_name, sizeof(g_exe_name), "%s", exe);
    }
    /* Filter out --debug-epoch and --prof before ki_parse_args.
     * NOTE: no --prf / --seq handling — the prof trainer runs in PRF mode
     * (N members × 1 thread) ALWAYS; the production trainer is the
     * sequential baseline. A stray --prf is left for ki_parse_args to
     * reject (it is not a known option there), keeping the CLI honest. */
    const char **debug_av = (const char **)malloc((size_t)(argc + 1) * sizeof(char *));
    int debug_ac = 0;
    debug_av[debug_ac++] = argv[0];
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--debug-epoch") == 0) { debug_epoch = 1; }
        else if (strcmp(argv[i], "--prof") == 0) { g_prof = 1; }
        else if (strcmp(argv[i], "--export-default") == 0) { g_export_default = 1; }
        else if (strcmp(argv[i], "--debug-member") == 0) { g_dbg_member = 1; }
        else if (strcmp(argv[i], "--threadN") == 0 && i + 1 < argc) {
            /* PRF: --threadN N = parallel member width (N members at once,
             * each on 1 thread). Keep the arg for ki_parse_args (it also
             * sets aa.threadN which we overwrite to 1 per member later). */
            g_user_threadn = 1;
            debug_av[debug_ac++] = argv[i];
            debug_av[debug_ac++] = argv[++i];
        }
        else if (strcmp(argv[i], "--member-file") == 0 && i + 1 < argc) {
            /* EXPLICIT --member-file: skip BOTH the flag and its value here.
             * The value ends in ".out" and exists, so the implicit scan below
             * would otherwise filter it out of debug_av while leaving the flag
             * behind — ki_parse_args would then see "--member-file" with no
             * value and fail with "[ERROR] Unknown argument: --member-file".
             * The explicit member-file scan (further down) sets aa.member_file,
             * so the flag does not need to reach ki_parse_args at all. */
            i++;  /* skip the value */
        }
        else {
            /* IMPLICIT --member-file (2026-08-18): ein Positions-Argument das auf
             * ".out" and exists as a file is NOT passed through to ki_parse_args
             * (it is not a CLI flag) — the member-file scan below
             * sets aa.member_file from it. */
            size_t _dl = strlen(argv[i]);
            int _is_implicit_mf = 0;
            if (argv[i][0] != '-' && _dl > 4 && strcmp(argv[i] + _dl - 4, ".out") == 0) {
                struct stat _st;
                if (stat(argv[i], &_st) == 0 && S_ISREG(_st.st_mode)) _is_implicit_mf = 1;
            }
            if (!_is_implicit_mf) debug_av[debug_ac++] = argv[i];
        }
    }
    debug_av[debug_ac] = NULL;
    aa.lr_step = (int)round(aa.lr * (1<<OT_PRECISION));
    /* ROUNDTRIP (feature 2026-08-10): adopt ensemble DEFAULTS from the
     * --member-file META header BEFORE the CLI parse, so the file sets the
     * defaults and explicit user options still win (they are applied by
     * ki_parse_args afterwards). The path must be extracted from argv here
     * because ki_parse_args has not run yet — it sets aa.member_file itself
     * (ki-common.h:1491), which is why the parse happens after it. */
    {
        /* IMPLICIT --member-file (2026-08-18): ein Positions-Argument, das auf
         * ".out" and exists as a file implies --member-file <arg>.
         * e.g.: run-research.sh ...x.exe member-H512-E10-OT8-M1-105-INT32.out.
         * An explicit --member-file (next scan) always wins. */
        for (int _ai = 1; _ai < argc; _ai++) {
            if (argv[_ai][0] == '-') continue;          /* Optionen ueberspringen */
            size_t _l = strlen(argv[_ai]);
            if (_l > 4 && strcmp(argv[_ai] + _l - 4, ".out") == 0) {
                struct stat _st;
                if (stat(argv[_ai], &_st) == 0 && S_ISREG(_st.st_mode)) {
                    strncpy(aa.member_file, argv[_ai], sizeof(aa.member_file) - 1);
                    aa.member_file[sizeof(aa.member_file) - 1] = '\0';
                    break;
                }
            }
        }
        /* EXPLICIT --member-file: wins over the implicit .out detection above. */
        for (int _ai = 1; _ai + 1 < argc; _ai++)
            if (strcmp(argv[_ai], "--member-file") == 0) {
                strncpy(aa.member_file, argv[_ai + 1], sizeof(aa.member_file) - 1);
                aa.member_file[sizeof(aa.member_file) - 1] = '\0';
                break;
            }
        /* Apply META defaults once, for whichever scan (implicit or explicit)
         * set aa.member_file. */
        if (aa.member_file[0])
            ki_member_file_apply_meta();
    }
    ki_parse_args(debug_ac, (char **)debug_av);
    free(debug_av);
    aa.no_precompute = 1;  /* sequential: compute gb per-member */
    /* ── --export-default (2026-08-18): derive export-{STAMM} dir name from
     * ki_default_tag(), analog zum MERGE --member-out-default. The maj1_thresh
     * suffix is ALWAYS resolved: explicit value, or ki_default_half(NC_slice)
     * when auto (-2). Only if no explicit --export DIR was given. */
    if (g_export_default && aa.exportD[0] == '\0') {
        int _nc_slice = KI_NC / (aa.splitHN > 0 ? aa.splitHN : 1);
        int _m1t = (aa.maj_mode == KI_MAJ_1) ? aa.maj1_thresh : -1;
        if (_m1t == -2) _m1t = ki_default_half(_nc_slice);
        char _exp[512];
        ki_default_tag(_exp + 7, sizeof(_exp) - 8, aa.hidden, aa.epochs, aa.maj_mode, _m1t);
        memcpy(_exp, "export-", 7);
        strncpy(aa.exportD, _exp, sizeof(aa.exportD) - 1);
        aa.exportD[sizeof(aa.exportD) - 1] = '\0';
        printf("  Export-Default: %s  (--export-default)\n", aa.exportD);
    }
    /* Precompute encoding LUTs for each active (enc, width) pair */
    for (int _ei = 0; _ei < aa.enc_count; _ei++)
        enc_lut_init_enc((int)aa.enc_array[_ei].type, (int)aa.enc_array[_ei].width);
    if (aa.enc_count == 0) {
        int def_enc = aa.debug_binarize ? KI_ENC_LIN7 : KI_ENC_RAW;
        enc_lut_init_enc(def_enc, KI_ENC_WIDTH_DEFAULT);
    }
    if (KI_COLORS <= 1) {
        /* MNIST/Fashion grayscale: the RGB-derived channels (COLOR_R..CP,
         * chromatic LBP) do not exist — fold them to COLOR_MNIST. The
         * grayscale structure channels (edge/bin/lbp/dog/var/dir/range)
         * are KEPT, so --channel mnist,lbp produces a real cross-product
         * (plan-2026-08-13-fashion-edge-channel.md). Legacy behavior
         * (--channel not given) stays: channel stays KI_DEFAULT_COLOR. */
        if (!aa.channel_explicit)
            aa.channel = KI_DEFAULT_COLOR;
        /* Reassign enc_array: CIFAR channel → COLOR_MNIST, deduplicate.
         * Structure channels keep their id. */
        int wi = 0;
        for (int i = 0; i < aa.enc_count; i++) {
            int col = (int)aa.enc_array[i].color;
            int new_col = (col >= 0 && ki_color_is_cifar_rgb(col)) ? COLOR_MNIST : col;
            /* Check for duplicate with already-kept entries */
            int dup = 0;
            for (int j = 0; j < wi; j++) {
                if (aa.enc_array[j].type  == aa.enc_array[i].type &&
                    aa.enc_array[j].width == aa.enc_array[i].width &&
                    aa.enc_array[j].color == new_col) { dup = 1; break; }
            }
            if (!dup) {
                aa.enc_array[wi] = aa.enc_array[i];
                aa.enc_array[wi].color = (int8_t)new_col;
                wi++;
            }
        }
        aa.enc_count = wi;
    }

    /* Parse --member/--member-file EARLY (vor n_cont/load_input),
     * so load_input packs all required encodings. */
    {
        int n_mems = 0;
        n_mems = ki_member_file_parse();
        if (n_mems < 0) return 1;
        if (n_mems == 0) { n_mems = ki_member_parse(); if (n_mems < 0) return 1; }
        if (n_mems > 0) {
            rebuild_enc_array();
            for (int _ei = 0; _ei < aa.enc_count; _ei++)
                enc_lut_init_enc((int)aa.enc_array[_ei].type, (int)aa.enc_array[_ei].width);
        }
    }

    /* 1:1 Mapping: Bit b = COLOR_BIT direkt (COLOR_MNIST=0, R=1, G=2, …, GB=10)
     * active_chans stores the bit position from enum ki_color_bit. */
    {   int mask = aa.channel;
        int n = 0;
        for (int b = 0; b < COLOR_NB; b++)
            if (mask & (1 << b))
                active_chans[n++] = b;
        if (n == 0) { fprintf(stderr, "[FATAL] --channel: no channels selected\n"); return 1; }
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
    /* PRF mode (always on): N members in parallel, each with 1 thread — the
     * member loop below gets num_threads(min(active_members, g_prf_width)).
     * g_prf_width is the PARALLEL MEMBER WIDTH: --threadN N changes it;
     * without --threadN it defaults to the number of HW threads. The INNER
     * training parallelism is capped at 8 (see the member-loop comment) so
     * the 1-member case matches the historical seq default of 8 threads. */
    int _hw = omp_get_max_threads();   /* BEFORE omp_set_num_threads */
    g_prf_width = aa.threadN > 0 ? aa.threadN : _hw;
    if (g_prf_width < 1) g_prf_width = 1;
    printf("  [PRF] parallel members (distributed simulation, width=%d)%s\n",
           g_prf_width,
           g_user_threadn ? "  (--threadN)" : "  (auto)");
    /* PRF default: per-member progress (mem=... id:...) always on — visual
     * feedback for the user. The extra W0/MIN/MAX/pxz fields only appear
     * with an explicit --debug-member (tracked by g_dbg_member). */
    aa.debug_member = 1;
    /* PRF threads: the global limit stays 1 (per-member) — the OUTER member
     * loop caps itself via num_threads(min(active, g_prf_width)). The PRE-COMPUTE
     * functions (compute_h0/gb_te, outside the member loop) set their own
     * num_threads(g_prf_width) locally — sample-parallel only when the member
     * loop is single (few members). Never nested: the member loop and the
     * precompute loops are disjoint in time (bug 2026-08-09: 1 member ran
     * single-threaded = 13.6s vs 3.6s). */
    omp_set_num_threads(1);

#ifdef KI_BITVOTING
    /* ── BIT-VOTING mode: direct pixel-bit → class voting ──────────
     * Bit-Voting is Otto with identity W0 (gb = input) and I=H width:
     *   - splitVN forced to 1  → V=32, h0_to_gb is identity
     *   - --hiddenN ignored    → H_local = NC_slice (number of input
     *     containers); aa.hidden is set to NC_slice after the slice
     *     computation so .meta/.ens headers stay consistent.
     * Everything else (target init, ki_batch_correct, scores, export,
     * IFC) runs unchanged — see plans/plan-2026-08-02-ki-bitvoting-flag.md */
    if (aa.splitVN != 1)
        fprintf(stderr, "  [WARN] --splitVN ignored in Bit-Voting mode "
                        "(forced to 1, V=32)\n");
    aa.splitVN = 1;
#endif

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
        /* KI_BITVOTING: H (=aa.hidden) is 0/ignored — validate against the
         * container count (I=H). Normal Otto uses the hidden neuron count. */
        int _h_val = aa.hidden;
#ifdef KI_BITVOTING
        _h_val = nc_blk / splitHN;
#endif
        int ncs = nc_blk / splitHN;           /* containers per slice (per-color) */
        int step_scale_div = _h_val * nc_blk; /* total neuron-container pairs */
        int sc = (_h_val * ncs * 100) / step_scale_div;
        if (sc < 1) sc = 1;
        int minN = (100 + sc - 1) / sc;
        if (aa.stepN > 0 && aa.stepN < minN) {
            fprintf(stderr, "[FATAL] --step-const %d too small (step_norm=0)\n", aa.stepN);
            fprintf(stderr, "  H=%d  NC_slice=%d  KI_NC_TOTAL=%d\n", _h_val, ncs, nc_blk * KI_COLORS);
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
        if (_need_identity) {
            /* The broad X_all buffer is only read by --debug-flat, --shuffle
             * and --debug-class-voting; the training loop uses per-member CEX
             * buffers (mem->input_buf via load_input_cex_cached) and the
             * sweep ensemble eval uses vote accumulators. In --sweep and in
             * default runs X_all is dead weight — skip it entirely (this also
             * stops writing the retired OLD-format <hash>_<N>x<stride>.pre
             * cache, bug 2026-08-05). Where a consumer needs it, compute it
             * fresh via load_input() — no disk cache, the old .pre format is
             * gone. See: bugs/bug-2026-08-05-sweep-old-pre-format.md */
            int _need_broad = aa.debug_flat || aa.shuffle || aa.debug_class_voting;
            X_all = (aa.sweep || !_need_broad)
                ? NULL
                : load_input(data.X_raw, total_all);
        }

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
    uint32_t *X_te  __attribute__((unused)) = X_all ? X_all + (size_t)total_train * n_cont : NULL;
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

    /* ── CEX input buffers ──
     * Each member owns ONE narrow buffer for its (CHAN:ENC:XFORM) slice,
     * loaded lazily via load_input_cex_cached() on first gb-cache MISS.
     * Identity members (xform_id = KI_XFORM_ID) also get their own CEX
     * buffer — the broad X_all is NOT used by training anymore (NULL unless
     * --debug-flat/--shuffle/--debug-class-voting, bug 2026-08-05). */
    int n_xforms_eff = aa.xform_list_count;
    if (n_xforms_eff < 1) {
        /* Default: identity only */
        n_xforms_eff = 1;
        aa.xform_list[0] = KI_XFORM_ID;
        aa.xform_list_count = 1;
    }

    /* ── Compute cont_per_row and rows factor ────────────────── */
    /* cont_per_row = containers per image row = KI_COLS * width / 32 */
    /* Derived from nc_blk / KI_ROWS since nc_blk = KI_ROWS * cont_per_row */
    int cont_per_row = nc_blk / KI_ROWS;
    if (cont_per_row < 1) cont_per_row = 1;
    int rows_factor = aa.rows_mode ? KI_ROWS : 1;

    /* ── Compute slice dimensions ────────────────────────────── */
    int NC_slice  = aa.rows_mode ? cont_per_row : (nc_blk / splitHN);  /* base slice */
#ifdef KI_BITVOTING
    /* I=H width: H_local = number of input containers per member
     * slice (splitVN is forced to 1, so V=32 → target[b][k] with
     * b = container×32 + bit, exactly the Bit-Voting layout).
     * aa.hidden is repurposed as the container count so .meta/.ens
     * headers record the I=H width (merge-ensemble validates H).
     * H (local display var) is pulled along so the setup/RESULT
     * headers show the real I=H width, not the ignored --hiddenN. */
    if (aa.hidden != 64)
        fprintf(stderr, "  [WARN] --hiddenN ignored in Bit-Voting mode "
                        "(I=H width, H=%d)\n", NC_slice);
    aa.hidden = NC_slice < 1 ? 1 : NC_slice;
    H = aa.hidden;
    int H_local   = aa.hidden;
#else
    int H_local   = H;
#endif
    int total_members = 0;  /* recomputed after member_spec generation */

    /* ── Pre-flight: verify/create the --export-merge-scores dir BEFORE
     * training. A mismatched trainer exe (or config) must abort the WHOLE
     * run here — otherwise 221 members train for hours and only then every
     * per-member export fails silently (export_one_member_ens ignores its
     * return value). This is the .meta EXE= guard (2026-08-11).
     * NOTE: must run AFTER the aa.hidden repurpose above (Bit-Voting sets
     * aa.hidden = NC_slice = the I=H container count, e.g. 784 at 32-bit) —
     * before it, aa.hidden still holds the CLI default (64) and the .meta
     * would be created with the WRONG H (bug 2026-08-11: BV32 sweep failed
     * with "config mismatch (H=64/784)"). */
    if (!aa.dry_run && aa.export_merge_scores[0]) {
        if (export_merge_scores_meta_check(aa.export_merge_scores) != 0) {
            ki_dataset_free(&data);
            return 1;
        }
    }

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
        EnsMemberMeta *ifc_meta = NULL;
        if (ifc_load_model(model_path, &W0_ifc, &tgt_ifc, &off_ifc,
                           &n_mifc, &hl_ifc, &ns_ifc, &ifc_meta) < 0) return 1;
        printf("  Score Mode:  %s  (%s counter + %s scores)\n",
               ki_score_type_str(),
           ki_counter_type_str(),
               ki_score_type_str());
        /* Show member details — from the MODEL metadata, not the CLI args:
         * in IFC mode the ensemble specs live in ifc_meta (channel/enc/wid/
         * xform per member), the CLI enc_array/xform_list are empty defaults
         * and would print wrong data (bug 2026-08-10: showed "exp"/"id"
         * although the members are gamma8/sqrt8 + rot90@colswap-1-4 etc.). */
        printf("  Members:     %d\n", n_mifc);
        printf("  Channels:    %s\n", color_str());
        {
            char en_buf[256] = "";
            for (int mi = 0; mi < n_mifc && ifc_meta; mi++) {
                if (en_buf[0]) strncat(en_buf, ",", sizeof(en_buf) - strlen(en_buf) - 1);
                strncat(en_buf, ifc_meta[mi][1], sizeof(en_buf) - strlen(en_buf) - 1);
                strncat(en_buf, ifc_meta[mi][2], sizeof(en_buf) - strlen(en_buf) - 1);
            }
            if (!en_buf[0]) {
                for (int ei = 0; ei < aa.enc_count && ei < KI_ENC_MAX; ei++) {
                    if (en_buf[0]) strncat(en_buf, ",", sizeof(en_buf) - strlen(en_buf) - 1);
                    strncat(en_buf, ki_enc_name_short((int)aa.enc_array[ei].type),
                            sizeof(en_buf) - strlen(en_buf) - 1);
                }
            }
            printf("  Encodings:   %s\n", en_buf[0] ? en_buf : "-");
        }
        {
            char xf_buf[256] = "";
            for (int mi = 0; mi < n_mifc && ifc_meta; mi++) {
                if (xf_buf[0]) strncat(xf_buf, ", ", sizeof(xf_buf) - strlen(xf_buf) - 1);
                strncat(xf_buf, ifc_meta[mi][3], sizeof(xf_buf) - strlen(xf_buf) - 1);
            }
            if (!xf_buf[0]) {
                for (int xi = 0; xi < aa.xform_list_count; xi++) {
                    if (xf_buf[0]) strncat(xf_buf, ", ", sizeof(xf_buf) - strlen(xf_buf) - 1);
                    const char *n = ki_xform_str(aa.xform_list[xi]);
                    strncat(xf_buf, n, sizeof(xf_buf) - strlen(xf_buf) - 1);
                }
            }
            printf("  Xform:       %s\n", xf_buf[0] ? xf_buf : "id");
        }
        /* Create ki_Member array from model data */
        size_t K = KI_NCLASSES, V = 32;
        ki_Member **mems = (ki_Member **)malloc((size_t)n_mifc * sizeof(ki_Member *));
        int shared_input = (ifc_meta == NULL);   /* v6: shared CLI-config buffer */
        /* --dry-run (debugging): no pixel data is loaded (data.X_raw == NULL,
         * ki-local.h:151) — the per-member input buffers CANNOT be built. Show
         * the SETUP (members/metadata above) and return; the actual inference
         * is skipped. Without this guard load_input_cex_cached() SEGV'd on the
         * NULL X_raw (bug 2026-08-09). Frees the model buffers exactly like the
         * normal path (Z.2894-2895) — no leak on the early return. */
        if (aa.dry_run) {
            printf("  Dry-run:      setup only (no pixel data loaded, inference skipped)\n");
            printf("  Members:     %d  H=%d  NC=%d  mode=%s\n",
                   n_mifc, hl_ifc, ns_ifc, ki_xform_str(aa.xform_list_count ? aa.xform_list[0] : KI_XFORM_ID));
            free(mems);
            free(W0_ifc); free(tgt_ifc); free(off_ifc); free(ifc_meta);
            ki_dataset_free(&data);
            return 1;
        }
        for (int i = 0; i < n_mifc; i++) {
            size_t w0_off = (size_t)i * (size_t)hl_ifc * (size_t)ns_ifc;
            mems[i] = ki_member_create(hl_ifc, ns_ifc,
                                       shared_input ? (int)((size_t)i * (size_t)ns_ifc) : 0,
                                       W0_ifc + w0_off, total_eval, 0);
            memcpy(mems[i]->target, tgt_ifc + (size_t)i * (size_t)hl_ifc * K * V,
                   (size_t)hl_ifc * K * V * sizeof(COUNTER_TYPE));
            memcpy(mems[i]->offset, off_ifc + (size_t)i * K, (size_t)K * sizeof(SCORE_TYPE));
            if (!shared_input) {
                /* v7: rebuild the member's OWN input slice from the model's
                 * channel/encoding/xform metadata — the shared CLI-config
                 * buffer mismatched heterogeneous ensembles (bug 2026-08-06:
                 * CIFAR import gave 12.81%). */
                int col = ifc_color_id(ifc_meta[i][0]);
                int enc = ifc_enc_id(ifc_meta[i][1]);
                int wid = atoi(ifc_meta[i][2]);
                int xf  = ifc_xform_id(ifc_meta[i][3]);
                /* MUST init the encoding LUT before load_input_cex_cached —
                 * otherwise enc_lut_get() returns garbage (bug 2026-08-06:
                 * import gave exactly 10.0% = random; the training inits the
                 * member encodings via rebuild_enc_array(), the import
                 * skipped it). */
                enc_lut_init_enc(enc, wid);
                mems[i]->color_bit = col;
                mems[i]->enc_type  = enc;
                mems[i]->enc_width = wid;
                mems[i]->xform_id  = xf;
                mems[i]->input_buf = load_input_cex_cached(
                    data.X_raw + (size_t)total_train * KI_PX, total_eval,
                    col, enc, wid, xf);
            }
        }
        /* Evaluate — v7: per-member input buffers (X=NULL, mem->input_buf);
         * v6 fallback: one shared CLI-config buffer. */
        uint8_t *pred_eval = (!aa.dry_run && aa.predictions[0])
            ? (uint8_t *)ki_xcalloc((size_t)total_eval, sizeof(uint8_t))
            : NULL;
        struct timeval tv0, tv1; gettimeofday(&tv0, NULL);
        int evl_ok = 0;
        if (total_eval > 0 && !aa.dry_run) {
            /* IFC inference: restore the OpenMP thread count. The PRF trainer
             * forced omp_set_num_threads(1) (per-member threads) — but the
             * import evaluates ALL members in ONE parallel pass (ki_evaluate_member
             * parallelizes over samples). omp_get_max_threads() returns 1 after
             * the PRF limit, so use g_prf_width (the saved HW width). Without
             * this reset the inference ran single-threaded: 17.8s vs ~2s at 16
             * threads (bug 2026-08-09). */
            int _ifc_hw = g_prf_width > 0 ? g_prf_width : omp_get_num_procs();
            if (_ifc_hw < 1) _ifc_hw = 1;
            omp_set_num_threads(_ifc_hw);
            aa.threadN = _ifc_hw;
            if (shared_input) {
                /* Lade nur eval-Samples (offset = total_train * KI_PX) */
                uint32_t *eval_buf = load_input(data.X_raw + (size_t)total_train * KI_PX,
                                               total_eval);
                if (eval_buf) {
                    evl_ok = ki_evaluate_member(eval_buf, y_te, total_eval, mems, n_mifc,
                                               (int)n_cont, pred_eval, 0);
                    free(eval_buf);
                }
            } else {
                evl_ok = ki_evaluate_member(NULL, y_te, total_eval, mems, n_mifc,
                                            (int)n_cont, pred_eval, 0);
            }
        }
        /* Per-member evals (explicit --debug-member, v7 path) — lets us see
         * whether the reconstructed per-member inputs match the training
         * members. Shows the running ensemble accuracy (ens=) as members
         * are added, plus each member's individual accuracy (mem=) and its
         * spec. Gate on g_dbg_member (the EXPLICIT flag), NOT aa.debug_member
         * which the PRF trainer forces to 1 for the training progress line
         * (bug 2026-08-10: without --debug-member this loop still ran and
         * doubled the eval work — 16 members × 10000 samples). */
        if (g_dbg_member && !shared_input && total_eval > 0 && !aa.dry_run) {
            printf("  Per-member evals (v7 reconstructed inputs):\n");
            SCORE_TYPE (*run_votes)[KI_NCLASSES] = (SCORE_TYPE (*)[KI_NCLASSES])
                calloc((size_t)total_eval, sizeof(SCORE_TYPE[KI_NCLASSES]));
            if (!run_votes) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
            for (int i = 0; i < n_mifc; i++) {
                int ok = 0, cum_ok = 0;
                /* EXCLUSIVE PARALLELISM (bug 2026-08-10): this per-member
                 * debug eval was fully SEQUENTIAL (16 members × 10000
                 * samples) → +6.6s on top of the parallel ki_evaluate_member.
                 * The MEMBER loop must stay sequential (run_votes is an
                 * incremental ensemble accumulator), but the sample loop is
                 * data-parallel: each s writes its OWN run_votes[s] row
                 * (disjoint), ok/cum_ok are reductions. No race.
                 * NOTE: no if(!g_parallel_members) gate here — this block
                 * runs at TOP LEVEL (no enclosing parallel region), so the
                 * sample loop can ALWAYS use all cores regardless of how
                 * ki_evaluate_member set g_parallel_members (bug 2026-08-10:
                 * the gate made this loop serial again after the member-
                 * parallel eval → 6.9s vs 1.4s). */
                #pragma omp parallel for schedule(static) reduction(+:ok,cum_ok)
                for (int s = 0; s < total_eval; s++) {
                    SCORE_TYPE sc[KI_NCLASSES];
                    scores_otto(mems[i]->input_buf + (size_t)s * (size_t)mems[i]->NC_slice,
                                mems[i]->W0, mems[i]->H_local, mems[i]->NC_slice,
                                mems[i]->target, mems[i]->offset, sc);
                    int pred = 0;
                    for (int k = 1; k < KI_NCLASSES; k++)
                        if (sc[k] > sc[pred]) pred = k;
                    if (pred == (int)y_te[s]) ok++;
                    /* running ensemble vote */
                    int cpred = 0;
                    for (int k = 0; k < KI_NCLASSES; k++) {
                        run_votes[s][k] += sc[k];
                        if (k > 0 && run_votes[s][k] > run_votes[s][cpred]) cpred = k;
                    }
                    if (cpred == (int)y_te[s]) cum_ok++;
                }
                printf("  [%3d/%d] ens=%5.1f%%  mem=%5.1f%%  %s:%s%d  xf=%s\n",
                       i + 1, n_mifc,
                       100.0f * (float)cum_ok / (float)total_eval,
                       100.0f * (float)ok / (float)total_eval,
                       ifc_meta[i][0], ifc_meta[i][1], atoi(ifc_meta[i][2]),
                       ifc_meta[i][3]);
            }
            free(run_votes);
            fflush(stdout);
        }
        gettimeofday(&tv1, NULL);
        int el = (int)((tv1.tv_sec-tv0.tv_sec)*1000 + (tv1.tv_usec-tv0.tv_usec)/1000);
        float acc = (float)evl_ok * 100.0f / (float)total_eval;
        printf("\n══╡ RESULT ╞══════════════════════════════════════════════════════\n");
        printf("  Eval:    %.1f%%  (%d/%d)\n", acc, evl_ok, total_eval);
        printf("  Time:    %dms\n", el);
        ki_report_show(0, 0, evl_ok, total_eval, el, aa.threadN,
                       total_eval - evl_ok, 0.0f, n_mifc, 0);
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
        free(mems); free(W0_ifc); free(tgt_ifc); free(off_ifc); free(ifc_meta);
        ki_dataset_free(&data); free(X_all);
        if (X_perm) { free(X_perm); free(y_perm); }
        return 0;
    }

    print_setup(H, aa.cfg_epochs, total_train, total_eval, aa.threadN, aa.seed, aa.batchN,
                splitVN, splitHN, NC_slice, H_local, ensembleN, aa.channel, nc_blk, (int)n_cont);

    /* ── gb-cache status (GUARANTEED message, bug 2026-08-10): the STATUS
     * line is always shown — without it, --debug-cache logged nothing when
     * data/gb/ did not exist, silently hiding that the gb_buf (h0→gb, the
     * dominant phase at H=512, ~785ms) is recomputed every run. The
     * detailed hit/miss/saved lines below stay behind --debug-cache. */
    if (!aa.import_gb && !aa.export_gb) {
        printf("  gb-cache: DISABLED  (use --export-gb once to write data/gb/, then --import-gb to load)\n");
    } else if (aa.import_gb) {
        printf("  gb-cache: enabled    (import: data/gb/*.gb)\n");
    } else {
        printf("  gb-cache: enabled    (export: data/gb/*.gb)\n");
    }

    /* ── W0: random uint32[total_members][H_local][NC_slice*pixel_groups] (frozen) ── */
    int uses_pixel = (aa.maj_mode == KI_MAJ_1P || aa.maj_mode == KI_MAJ_1RP);
    int pixel_groups = uses_pixel ? KI_PIXEL_GROUPS : 1;
    size_t w0_m_sz = (size_t)H_local * (size_t)NC_slice * (size_t)pixel_groups;
    /* W0_ens is allocated + filled after member_spec generation (see below).
     * Placeholder here only — the actual W0 seeding comes after member_spec. */
    uint32_t *W0_ens = NULL;
#ifndef KI_BITVOTING
    size_t w0_sz = 0;
#else
    size_t w0_sz __attribute__((unused)) = 0;
#endif

#ifdef USE_HIP
    /* ── GPU init: upload all W0 once (W0_ens is allocated later) ── */
    int gpu_ok = 0;
#endif

    /* ── Target + Offset for each member ─────────────────────────────── */
    int V = VN_GROUPS_;

    int class_counts[KI_NCLASSES] = {0};
    if (!aa.dry_run)
        for (int s = 0; s < total_train; s++)
            class_counts[(int)y_tr[s]]++;

    /* Generate member specs from product (nur wenn kein --member/--member-file) */
    if (aa.member_spec_count == 0) {
        if (ki_member_spec_generate_from_product() < 0) return 1;
    }
    /* Group specs xform-major (stable) so the training loop holds only one
     * input buffer per xform group. Without this, --member-file order (file
     * order, unsorted) makes every lazy-loaded X_xform buffer stay resident
     * → OOM on large member files (461 members × ~20 xforms × ~2.6 GB).
     * See: plans/plan-2026-07-31-sweep-xform-memory.md */
    ki_member_spec_sort_xform();
    rebuild_enc_array();
    total_members = aa.member_spec_count * ensembleN * rows_factor;

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

    /* ── W0: allocate and fill (member_spec_count is now known) ── */
#ifndef KI_BITVOTING
    w0_sz = (size_t)total_members * w0_m_sz;
    if (w0_sz == 0) w0_sz = (size_t)w0_m_sz;
    if (W0_ens) free(W0_ens);
    W0_ens = (uint32_t *)ki_xmalloc(w0_sz * sizeof(uint32_t));
    {
        if (aa.seed_file[0])
            w0_rand_set_file(aa.seed_file);
        if (aa.ensemble_seed == ENS_SEED_CONST) {
            /* const: ONE W0 per ensemble, shared across ALL members.
             * All xform+channel+encoding+hn variants get the SAME W0 —
             * only --ensembleN produces different W0 per ensemble. */
            int memb_per_ens = aa.member_spec_count * rows_factor;
            for (int e = 0; e < ensembleN; e++) {
                w0_srandom((unsigned int)(aa.seed + (unsigned int)e));
                int mb_start = e * memb_per_ens;
                for (size_t i = 0; i < w0_m_sz; i++) {
                    uint32_t v = w0_random();
                    for (int mm = 0; mm < memb_per_ens; mm++)
                        W0_ens[(size_t)(mb_start + mm) * w0_m_sz + i] = v;
                }
            }
        } else if (aa.ensemble_seed == ENS_SEED_INCR) {
            for (int m = 0; m < total_members; m++) {
                w0_srandom((unsigned int)(aa.seed + (unsigned int)m));
                for (size_t i = 0; i < w0_m_sz; i++)
                    W0_ens[(size_t)m * w0_m_sz + i] = w0_random();
            }
        } else {
            w0_srandom((unsigned int)aa.seed);
            for (size_t i = 0; i < w0_sz; i++)
                W0_ens[i] = w0_random();
        }
    }
#else
    /* BIT-VOTING: no W0 (identity projection) — nothing to allocate.
     * w0_marker in the .ens export is 0 (mem->W0 == NULL), which
     * merge-ensemble uses to keep Bit-Voting archives separate from
     * Otto archives (see plans/plan-2026-08-02-ki-bitvoting-flag.md). */
    (void)w0_m_sz;
    W0_ens = NULL;
#endif

#ifdef USE_HIP
    /* ── GPU init: upload all W0 once (W0_ens is now allocated) ── */
    int gpu_ok = 0;
    if (!aa.dry_run) {
        if (hip_mem_init(total_train, H_local, NC_slice, nc_blk, total_members) == 0) {
            hip_mem_upload_W0_all(W0_ens, total_members * H_local * NC_slice);
            printf("  [HIP] GPU enabled (%d members)\n", total_members);
            gpu_ok = 1;
        } else printf("  [HIP] Init failed, using CPU\n");
    }
#endif

    print_member_structure(ensembleN, splitVN, splitHN, H_local, NC_slice, aa.channel);
    /* Target is built from gb_buf AFTER h0-compute (s.u.). */
    printf("\n");
    fflush(stdout);

    {
#if COUNTER_TYPE_IS_FLOAT
        /* Lossless float mode (OT_F=1): the step IS the lr — an int cast
         * would show 0 for lr=0.05. Display it as a float. 2026-08-16. */
        printf("══╡ TRAINING ╞══  lr=%.4f  step=%.4f  mode=%s  F=%d",
             (double)aa.lr, (double)ot_precision(aa.lr), mode_str(), OT_F);
#else
        int step = (int)(aa.lr * (float)OT_F + 0.5f);
        printf("══╡ TRAINING ╞══  lr=%.4f  step=%d  mode=%s  F=%d",
             (double)aa.lr, step, mode_str(), OT_F);
#endif
        printf("  tgt-init=%s", target_init_str());
        if (aa.multi_correct)
          printf("  multi-correct=on");
        if (aa.opt_target_norm)
            printf("  tgt-nrm=%d", aa.opt_target_norm ? 1 : 0);
        if (aa.gap_k > 0.0f)
            printf("  gap-k=%.1f", (double)aa.gap_k);
        if (aa.member_threshold > 0)
            printf("  mth=%d", aa.member_threshold);
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
    /* ── All specs are active (no channel filtering needed with member_spec) ─── */
    int active_members = total_members;  /* = aa.member_spec_count * ensembleN * rows_factor */
    /* Exclusive parallelism switch: many members → member loop parallel;
     * few members → sample loops use all cores (bug 2026-08-09).
     * FIXED threshold of 3 members (bug 2026-08-10): even width/2 was too
     * strict — 4 members on width=16 ran sequential (8648ms) but parallel
     * (--threadN 4) took 6565ms (cifar-1 2026-08-10_00-59-28 vs 00-59-52).
     * With 3+ members the member loop parallelizes (1 thread per member);
     * with 1-2 members the sample loops use all cores (cap 8) instead. */
    g_parallel_members = (active_members >= 3) ? 1 : 0;
    /* EXCLUSIVE PARALLELISM (bug 2026-08-10): when the member loop runs
     * SINGLE (few members, g_parallel_members==0), ALL inner sample loops
     * must run with all cores. ki_batch_correct (the training loop) has
     * `#pragma omp parallel for` WITHOUT num_threads (ki-train.h:85) and
     * inherits the GLOBAL team — with the old omp_set_num_threads(1) the
     * training ran on 1 thread: prof 5920ms vs seq 3682ms for the same job
     * (logs 2026-08-10_00-14-47 vs 00-09-35). aa.threadN must match the
     * team (it sizes the dc[] thread buffers in ki_batch_correct).
     * With many members (g_parallel_members==1) the outer loop owns the
     * parallelism: aa.threadN is forced to 1 (per-member thread count, the
     * PRF design) so ki_batch_correct allocates a SINGLE dc[] buffer —
     * otherwise the serial merge cost O(n_threads × tgt_sz) per batch
     * explodes (16 members with threadN=8 → 25s, bug 2026-08-10). */
    int _save_threadn = aa.threadN;
    if (!g_parallel_members) {
        /* Few members: use all cores inside the member — capped at 8
         * because ki_batch_correct (batch=64) does NOT scale beyond 8
         * (16 threads → 4 samples/thread + spawn overhead = 13s, measured
         * 2026-08-10). seq's historical default was 8 (aa.threadN=8). */
        int _inner = g_prf_width > 8 ? 8 : g_prf_width;
        aa.threadN = _inner;
        omp_set_num_threads(_inner);
    } else {
        /* Many members: 1 thread per member, member loop is the parallel
         * dimension. Matches the old auto default (aa.threadN=0 → 1). */
        aa.threadN = 1;
        omp_set_num_threads(1);
    }
    ki_Member **members = (ki_Member **)malloc((size_t)active_members * sizeof(ki_Member *));
    if (!members) { fprintf(stderr, "[FATAL] members OOM\n"); return 1; }

    /* ── Create members from member_spec[] with row expansion ── */
    {
        int mem_idx = 0;
        for (int e = 0; e < ensembleN; e++) {
            for (int sp = 0; sp < aa.member_spec_count; sp++) {
                ki_MemberSpec *spec = &aa.member_spec[sp];
                int color = spec->color;
                int xf_id = spec->xform_id;
                int hn_idx = spec->hn_idx;

                /* ── Container offset from encoding index in enc_array ── */
                int base_slc_off;
                int base_mem_nc;
                /* Find enc_array index matching this spec's encoding
                 * (type + width + color). On MNIST (KI_COLORS <= 1) the
                 * input buffer has one block per encoding, each packed
                 * with its own encoding LUT → vi selects the correct block. */
                int vi = -1;
                for (int ei = 0; ei < aa.enc_count && vi < 0; ei++) {
                    if ((int)aa.enc_array[ei].type == spec->enc_type &&
                        (int)aa.enc_array[ei].width == spec->enc_width &&
                        (int)aa.enc_array[ei].color == spec->color) {
                        vi = ei;
                    }
                }
                if (vi < 0) vi = 0;  /* fallback */
                if (aa.debug_flat) {
                    base_mem_nc = nc_blk / splitHN;
                    base_slc_off = hn_idx * base_mem_nc;
                } else if (aa.enc_count > 0 && vi < aa.enc_count) {
                    base_mem_nc = multi_enc_nc[vi] / splitHN;
                    base_slc_off = multi_enc_blk_off[vi] + hn_idx * base_mem_nc;
                } else {
                    base_mem_nc = nc_blk / splitHN;
                    base_slc_off = hn_idx * base_mem_nc;
                }

                /* ── Iterate over rows (1 for flat, KI_ROWS for row-mode) ── */
                for (int r = 0; r < rows_factor; r++) {
                    int mem_nc   = aa.rows_mode ? cont_per_row : base_mem_nc;
                    int slc_off  = aa.rows_mode
                                   ? (base_slc_off + r * cont_per_row)
                                   : base_slc_off;

    const uint32_t *W0_m;
#ifdef KI_BITVOTING
    (void)W0_m;  /* no W0 in Bit-Voting mode */
#else
    W0_m = W0_ens + (size_t)mem_idx * w0_m_sz;
#endif
#ifdef KI_BITVOTING
                    /* I=H width per member: H_local = this member's own
                     * container count (encodings differ in width). */
                    members[mem_idx] = ki_member_create(mem_nc, mem_nc, slc_off,
                                                    NULL, total_train, total_eval);
#else
                    members[mem_idx] = ki_member_create(H_local, mem_nc, slc_off,
                                                    W0_m, total_train, total_eval);
#endif
                    members[mem_idx]->w0_step = mem_nc * pixel_groups;
                    /* CEX: buffer loaded lazily per member (see gb-cache MISS).
                     * input_buf stays NULL until then; slc_off becomes 0
                     * because the CEX buffer IS the member's slice. */
                    members[mem_idx]->input_buf    = NULL;
                    members[mem_idx]->input_buf_te = NULL;
                    members[mem_idx]->slc_off = 0;
                    members[mem_idx]->orig_m = mem_idx;
                    members[mem_idx]->vi = vi;
                    members[mem_idx]->xform_id = xf_id;
                    members[mem_idx]->color_bit = color;
                    members[mem_idx]->enc_type  = spec->enc_type;
                    members[mem_idx]->enc_width = spec->enc_width;
                    members[mem_idx]->last_err = total_train;
                    mem_idx++;
                }
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
                    for (int h = 0; h < mem->H_local; h++) {
                        uint32_t h0 = ki_gb_for_neuron(in, mem->W0, h, mem->w0_step, mem->NC_slice, mem->half);
                        tg_gb[(size_t)s * (size_t)mem->H_local + (size_t)h] = h0_to_gb(h0);
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

    /* Member stats for --debug-member-stats — the collection block was
     * removed in PRF mode (member order undefined); only the per-member
     * agree/disagree counters remain (filled during Phase 2 eval). */
    __attribute__((unused))
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

    /* ── --sweep mode: print warnings for incompatible options ── */
    if (aa.sweep) {
        if (!aa.export_merge_scores[0]) {
            fprintf(stderr, "[FATAL] --sweep requires --export-merge-scores DIR\n");
            return 1;
        }
        if (aa.debug_member_stats)
            fprintf(stderr, "[WARN] --debug-member-stats ignored in sweep mode\n");
        if (aa.export_scores[0])
            fprintf(stderr, "[WARN] --export-scores ignored in sweep mode\n");
        if (aa.export_neurons[0])
            fprintf(stderr, "[WARN] --export-neurons ignored in sweep mode\n");
        if (aa.predictions[0])
            fprintf(stderr, "[WARN] --predictions ignored in sweep mode\n");
        if (aa.debug_confusion || aa.debug_class_voting)
            fprintf(stderr, "[WARN] --debug-confusion/--debug-class-voting ignored in sweep\n");
        if (aa.ensembleN > 1)
            fprintf(stderr, "[WARN] --ensembleN > 1 not recommended in sweep mode (use 1)\n");
    }

    int _sweep_trained = 0, _sweep_skipped = 0;
    int _sweep_done = 0; /* progress lines emitted (ETA base; atomic via critical(progress)) */
    /* FIX (2026-08-19): ETA on a restarted (aborted) sweep — many members are
     * SKIPped up front (already done). Those SKIP passes cost ~0 time but inflate
     * tv_start-relative elapsed, so the per-member rate must be measured from the
     * FIRST actual TRAIN, not from loop start. tv_work_start is set on the first
     * TRAIN progress line (in the critical(progress) block below). See:
     * logs/fashion-...-2026-08-19_17-41-23.log (2843 SKIPs -> bogus 7h09m ETA). */
    struct timeval tv_work_start;
    int tv_work_start_set = 0;
    int _last_xf = -1;   /* xform of the previously processed member (cache clear) */

    /* [PROF] phase accumulators (--prof, prof trainer only, 2026-08-08) */
    double _prof_cex = 0, _prof_gb = 0, _prof_target = 0, _prof_train = 0, _prof_eval = 0;
    int _prof_n = 0;

    /* PRF: N members in parallel, each with 1 thread. omp_set_num_threads(1)
     * above makes every inner per-sample region single-threaded; the outer
     * loop explicitly spawns min(active_members, cores) threads. Shared
     * state (ensemble votes, xform cache, counters) is protected with
     * critical/atomic sections — the vote phase is only ~1.6% of runtime,
     * so serializing it costs almost nothing. The cumulative `ens=`
     * progress line is skipped in PRF mode (member order is not defined),
     * only the final REPORT is emitted. */
    /* EXCLUSIVE PARALLELISM (bug 2026-08-09): the member loop is parallel
     * with 3+ members (g_parallel_members, fixed threshold 2026-08-10:
     * width/2 was too strict, 4 members on 16 threads ran sequential =
     * 8648ms vs parallel 6565ms, cifar-1 00-59-28 vs 00-59-52). With 1-2
     * members this region is DISABLED (`if` false → sequential), so the
     * inner per-sample loops (which carry `if(!g_parallel_members)`) can
     * run with all cores — no nested OpenMP, no deadlock. The `if`-gate
     * is the key: without it, the outer region was still "parallel" with
     * 1 thread and the inner loops could not spawn. */
    #pragma omp parallel for schedule(dynamic,1) \
        if(g_parallel_members) \
        num_threads(active_members > 0 && active_members < g_prf_width \
                    ? active_members : g_prf_width)
    for (int mb = 0; mb < active_members; mb++) {
        ki_Member *mem = members[mb];
        struct timeval _pt0, _pt1;
        double _p_cex = 0, _p_gb = 0, _p_target = 0, _p_train = 0, _p_eval = 0;

        /* Xform-group switch (all modes): release the single-slot in-memory
         * xform cache (ki-load.h) of the previous group. Members are sorted
         * xform-major, so the switch happens once per group and the slot is
         * overwritten automatically on the next miss anyway — this explicit
         * clear only frees the buffer EARLIER (peak-RAM control, e.g. for a
         * next group that is fully skipped/disk-hit). Plain int comparison —
         * never touches member structs (avoids the f8007d4 prev_mem
         * use-after-free pattern, bug-2026-08-05-sweep-prev-mem-double-free).
         * See: plans/plan-2026-08-05-xform-cache.md
         * PRF: the xform cache is a GLOBAL single slot — clearing it here
         * while another parallel member reads it is a race. The cache is
         * protected by the critical(cex) section around load_input_cex_cached
         * below; the explicit early clear is skipped (RAM stays slightly
         * higher, correctness first). */
        (void)_last_xf;   /* xform-group clear is disabled in PRF mode */

        /* Sweep memory control happens at the END of each member's own
         * iteration (ki_member_destroy below, incl. free(m)), so every sweep
         * member is destroyed exactly once — no prev-member free is needed
         * here. INTENTIONAL: the old prev_mem block (added in f8007d4)
         * dereferenced ALREADY destroyed members — prev_mem pointed at the
         * freed ki_Member struct → use-after-free / double-free of input_buf
         * (SIGSEGV + ASan double-free, repro: --sweep id + r,g,b + raw,lin,exp
         * on CIFAR). Removed 2026-08-05.
         * See: bugs/bug-2026-08-05-sweep-prev-mem-double-free.md */

        /* ── --sweep: skip if .ens already exists ── */
        if (aa.sweep) {
            char ens_path[512];
            build_ens_member_path(ens_path, sizeof(ens_path),
                aa.export_merge_scores, mem, mb);
            if (!aa.force && access(ens_path, F_OK) == 0) {
                printf("  [%3d/%d] SKIP  %s\n", mb+1, active_members, ens_path);
                fflush(stdout);
                #pragma omp atomic
                _sweep_skipped++;
                /* Free the skipped member immediately. Without this every
                 * skipped member keeps its target/best buffers (allocated
                 * for ALL members in the setup loop) resident until process
                 * exit — on a re-run sweep with many members that is GBs of
                 * wasted RAM. A skipped member is never trained/exported,
                 * and members[] is not accessed after the loop in sweep mode.
                 * See: plans/plan-2026-07-31-sweep-xform-memory.md */
                ki_member_destroy(mem);
                members[mb] = NULL;
                continue;
            }
        }

        /* Reset per-member agreement/disagreement counters */
        _agree_trn = _disagree_trn = _agree_evl = _disagree_evl = 0;

        /* ── Compute gb for this member once ── */
        /* Use the member's own H_local — identical to the global value in
         * normal Otto, but per-member in KI_BITVOTING (I=H per encoding). */
        size_t gb_sz = (size_t)total_train * (size_t)mem->H_local;
        size_t te_sz = (size_t)total_eval * (size_t)mem->H_local;

        /* Compute gb cache key + path (used by both --import-gb and --export-gb) */
        int gb_loaded = 0;
        uint32_t gb_key = 0;
        char gb_path[512] = "";
        if (aa.import_gb || aa.export_gb) {
            const char *_cn = ki_color_name(mem->color_bit);
            const char *_en = ki_enc_name_short((int)aa.enc_array[mem->vi].type);
            int _ew = (int)aa.enc_array[mem->vi].width;
            const char *_xn = ki_xform_str(mem->xform_id);
            gb_key = gb_cache_hash(mem->W0, total_train, total_eval,
                                    H_local, mem->NC_slice, n_cont,
                                    _cn, _en, _ew, _xn);
            snprintf(gb_path, sizeof(gb_path), "data/gb/%08x_%dx%dx%d.gb",
                     gb_key, total_train, total_eval, H_local);
        }
        if (aa.import_gb) {
            FILE *gb_f = fopen(gb_path, "rb");
            uint32_t magic, ver, chk_key, chk_Ntr, chk_Nev, chk_H;
            if (fread(&magic, 4, 1, gb_f) == 1 && magic == 0x47425F50 &&
                fread(&ver,   4, 1, gb_f) == 1 && ver == 1 &&
                fread(&chk_key,4, 1, gb_f) == 1 && chk_key == gb_key &&
                fread(&chk_Ntr,4, 1, gb_f) == 1 && (int)chk_Ntr == total_train &&
                fread(&chk_Nev,4, 1, gb_f) == 1 && (int)chk_Nev == total_eval &&
                fread(&chk_H,  4, 1, gb_f) == 1 && (int)chk_H == H_local) {
                mem->gb_buf = (uint32_t *)malloc(gb_sz * sizeof(uint32_t));
                if (fread(mem->gb_buf, sizeof(uint32_t), gb_sz, gb_f) == gb_sz) {
                    mem->gb_buf_te = NULL;
                    if (total_eval > 0) {
                        mem->gb_buf_te = (uint32_t *)malloc(te_sz * sizeof(uint32_t));
                        if (fread(mem->gb_buf_te, sizeof(uint32_t), te_sz, gb_f) == te_sz) {
                            gb_loaded = 1;
                            if (ki_debug_cache()) printf("  gb-cache: %s  (%d+%d samples, hit)\n", gb_path, total_train, total_eval);
                        } else { free(mem->gb_buf_te); mem->gb_buf_te = NULL; }
                    } else {
                        gb_loaded = 1;
                        if (ki_debug_cache()) printf("  gb-cache: %s  (%d samples, hit)\n", gb_path, total_train);
                    }
                }
                if (!gb_loaded) { free(mem->gb_buf); mem->gb_buf = NULL; }
            }
            fclose(gb_f);
        }

        if (!gb_loaded) {
            /* ── CEX input loading: one narrow buffer per member ──
             * Each member loads ONLY its own (CHAN:ENC:XFORM) slice via
             * load_input_cex_cached() — the cache file is keyed by the full
             * member identity, so member or xform changes never serve stale
             * data. slc_off is 0: the buffer IS the member's slice. */
            if (!mem->input_buf) {
                /* CEX input: the cache function handles the xform transform
                 * INTERNALLY and only on cache MISS. On hit it just loads the
                 * narrow file — no redundant per-member transformation
                 * (~20% slower with warm cache, 2026-08-04).
                 * PRF: the xform/channel caches in ki-load.h are GLOBAL
                 * single slots — serialize the load (0.9% of runtime) to
                 * avoid races between parallel members. */
                if (g_prof) gettimeofday(&_pt0, NULL);
                uint32_t *Xcex;
                #pragma omp critical(cex)
                {
                    Xcex = load_input_cex_cached(
                            data.X_raw, total_all,
                            mem->color_bit, mem->enc_type, mem->enc_width, mem->xform_id);
                }
                if (g_prof) { gettimeofday(&_pt1, NULL);
                    _p_cex += (double)(_pt1.tv_sec - _pt0.tv_sec) * 1000.0
                            + (double)(_pt1.tv_usec - _pt0.tv_usec) / 1000.0; }
                mem->input_buf    = Xcex;
                mem->input_buf_te = Xcex + (size_t)total_train * (size_t)mem->NC_slice;
                mem->slc_off = 0;   /* buffer is exactly this member's slice */
            }
            if (g_prof) gettimeofday(&_pt0, NULL);
            mem->gb_buf = (uint32_t *)malloc(gb_sz * sizeof(uint32_t));
            /* EXCLUSIVE PARALLELISM (bug 2026-08-09): this gb loop is the
             * dominant cost (88% at H=512, 1 member). It parallelizes over
             * samples ONLY when the member loop is single (few members,
             * g_parallel_members==0). With many members the outer loop does
             * the parallel and this stays single. */
            #pragma omp parallel for schedule(static) num_threads(g_prf_width) \
                if(!g_parallel_members)
            for (int s = 0; s < total_train; s++) {
                const uint32_t *in = mem->input_buf + (size_t)s * (size_t)mem->NC_slice + mem->slc_off;
                for (int h = 0; h < mem->H_local; h++) {
                    uint32_t h0 = ki_gb_for_neuron(in, mem->W0, h, mem->w0_step, mem->NC_slice, mem->half);
                    mem->gb_buf[(size_t)s * (size_t)mem->H_local + (size_t)h] = h0_to_gb(h0);
                }
            }
            mem->gb_buf_te = NULL;
            if (total_eval > 0) {
                mem->gb_buf_te = (uint32_t *)malloc(te_sz * sizeof(uint32_t));
                #pragma omp parallel for schedule(static) num_threads(g_prf_width) \
                    if(!g_parallel_members)
                for (int s = 0; s < total_eval; s++) {
                    const uint32_t *in = mem->input_buf_te + (size_t)s * (size_t)mem->NC_slice + mem->slc_off;
                    for (int h = 0; h < mem->H_local; h++) {
                        uint32_t h0 = ki_gb_for_neuron(in, mem->W0, h, mem->w0_step, mem->NC_slice, mem->half);
                        mem->gb_buf_te[(size_t)s * (size_t)mem->H_local + (size_t)h] = h0_to_gb(h0);
                    }
                }
            }
            if (g_prof) { gettimeofday(&_pt1, NULL);
                _p_gb += (double)(_pt1.tv_sec - _pt0.tv_sec) * 1000.0
                       + (double)(_pt1.tv_usec - _pt0.tv_usec) / 1000.0; }

            /* Save to cache if --export-gb */
            if (aa.export_gb) {
                mkdir("data/gb", 0755);
                FILE *sf = fopen(gb_path, "wb");
                if (sf) {
                    uint32_t magic = 0x47425F50, ver = 1;
                    uint32_t _Ntr = (uint32_t)total_train, _Nev = (uint32_t)total_eval, _H = (uint32_t)H_local;
                    fwrite(&magic, 4, 1, sf);
                    fwrite(&ver,   4, 1, sf);
                    fwrite(&gb_key,4, 1, sf);
                    fwrite(&_Ntr,  4, 1, sf);
                    fwrite(&_Nev,  4, 1, sf);
                    fwrite(&_H,    4, 1, sf);
                    fwrite(mem->gb_buf, sizeof(uint32_t), gb_sz, sf);
                    if (total_eval > 0 && mem->gb_buf_te)
                        fwrite(mem->gb_buf_te, sizeof(uint32_t), te_sz, sf);
                    fclose(sf);
                    if (ki_debug_cache()) printf("  gb-cache: %s  (saved)\n", gb_path);
                }
            }
        }

        /* ── Build target from gb ── */
        if (g_prof) gettimeofday(&_pt0, NULL);
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
        if (g_prof) { gettimeofday(&_pt1, NULL);
            _p_target += (double)(_pt1.tv_sec - _pt0.tv_sec) * 1000.0
                       + (double)(_pt1.tv_usec - _pt0.tv_usec) / 1000.0; }

        /* ── Train all epochs for this member ── */
        if (g_prof) gettimeofday(&_pt0, NULL);
        int member_best_err = total_train;
        COUNTER_TYPE step_init_local = step_init;  /* per-member step, reduced by rollbacks */
        int rb_depth = 0;  /* rollback counter per member */
        float member_gap = 0.0f;  /* train/eval gap for step damping */
        SCORE_TYPE _gap_sc[KI_NCLASSES];
        for (int ep = 0; ep < epochs; ep++) {
#if COUNTER_TYPE_IS_FLOAT
            /* Lossless float mode (2026-08-16): the correction step IS the
             * (damped) learning rate — no int cast, no min-2 clamp. In the
             * int32 path step_init_local = lr×F (~6554) and an int step with
             * min 2 is fine; here step_init_local = lr (0.05) and the int
             * cast + "s_step<2 → 2" clamp would force every correction to
             * 2.0 → targets explode (10^300, NaN scores). */
            COUNTER_TYPE s_step = step_init_local;
            if (aa.warmup_epochs > 0 && mem->ep < aa.warmup_epochs) {
                float scale = (float)(mem->ep + 1) / (float)aa.warmup_epochs;
                s_step = (COUNTER_TYPE)((double)step_init_local * (double)scale);
            } else {
                float progress = (float)(mem->ep - aa.warmup_epochs) / (float)((epochs + 0) - aa.warmup_epochs);
                if (progress > 1.0f) progress = 1.0f;
                float cosine = (1.0f + cosf(progress * (float)3.14159265358979323846f)) / 2.0f;
                float lr_min_f = (aa.lr_min > 0.0f) ? aa.lr_min : 0.0f;
                s_step = (COUNTER_TYPE)((double)step_init_local *
                          (double)(lr_min_f + (1.0f - lr_min_f) * cosine));
            }
            /* Gap damping: exp(-K × gap) reduces step when overfitting gap widens */
            if (aa.gap_k > 0.0f && member_gap > 0.0f) {
                float gap_factor = expf(-aa.gap_k * member_gap);
                s_step = (COUNTER_TYPE)((double)s_step * (double)gap_factor);
            }
            if (s_step < (COUNTER_TYPE)0) s_step = (COUNTER_TYPE)0;
            mem->step = (int)s_step;

             int err = ki_batch_correct(mem->target, mem->H_local, mem->offset,
                         mem->gb_buf, y_tr, total_train, (COUNTER_TYPE)s_step,
                         (size_t)mem->H_local * KI_NCLASSES * 32, aa.filter_mask,
                         mem->H_local, 0);
#else
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
#endif
            mem->last_err = err;
            /* trn_acc is set correctly AFTER evaluation (see below) */
            mem->ep++;

            /* ── Compute member gap (eval_err - train_err) for step damping ── */
            int _evl_err = -1;
            if (aa.gap_k > 0.0f && total_eval > 0 && mem->gb_buf_te) {
                _evl_err = 0;
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
                int _xid_xf = mem->xform_id;
                const char *_dep_xn = ki_xform_str(_xid_xf);
int _half = ki_compute_half(mem->NC_slice);
                printf("      [%3d/%d] trn=%5.1f%%  evl=%5.1f%%  err=%d  step=%d  half=%d%s xf=%s\n",
                       mem->ep, epochs,
                       (float)_dep_trn * 100.0f / (float)total_train,
                       (float)(total_eval - _dep_evl) * 100.0f / (float)(total_eval > 0 ? total_eval : 1),
                       err, mem->step, _half, _dep_info, _dep_xn);
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
        if (g_prof) { gettimeofday(&_pt1, NULL);
            _p_train += (double)(_pt1.tv_sec - _pt0.tv_sec) * 1000.0
                      + (double)(_pt1.tv_usec - _pt0.tv_usec) / 1000.0;
            gettimeofday(&_pt0, NULL); }

        /* ── Phase 1: Pre-Eval — member accuracy (needed for threshold) ── */
        int _member_trn = 0, _member_evl = 0;
        int _skip_member = 0;
        SCORE_TYPE _scr_min = 0, _scr_max = 0;  /* score range (--debug-member) */
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
            /* PRF: the vote accumulation writes SHARED state (acc_votes_tr/te,
             * final_trn_ok, final_evl_ok, best_evl_ok, _agree_* etc.) and the
             * inner omp-parallel-for reduction targets global variables —
             * a race with parallel members. The vote phase is ~1.6% of the
             * runtime, so serializing it costs almost nothing; the heavy
             * phases (gb, train) stay fully parallel. */
            #pragma omp critical(votes)
            {   SCORE_TYPE sc[KI_NCLASSES];
                /* score range (--debug-member): type-generic limits —
                 * float/double → ±INFINITY, int64 → INT64 extremes.
                 * (Was split by COUNTER_TYPE_IS_FLOAT; with
                 * -DMODE_INT32 -DSCORE_TYPE=double the INT64_MAX assign
                 * warned, 2026-08-05.) */
                _scr_min = _Generic((SCORE_TYPE)0,
                    int64_t: (SCORE_TYPE)INT64_MAX,
                    default: (SCORE_TYPE)INFINITY);
                _scr_max = _Generic((SCORE_TYPE)0,
                    int64_t: (SCORE_TYPE)INT64_MIN,
                    default: (SCORE_TYPE)-INFINITY);
                /* EXCLUSIVE PARALLELISM (bug 2026-08-10): with FEW members
                 * (!g_parallel_members) this eval runs with all cores — the
                 * seq trainer already did (Z.3626), the prof was serial in
                 * the critical(votes) block → +400ms (4059 vs 3655ms).
                 * acc_votes[s] is written by exactly one thread (disjoint s),
                 * agree/disagree and member counters are reduction targets.
                 * With MANY members the `if` disables the parallel region →
                 * the outer critical(votes) still serializes the shared
                 * counters. */
                #pragma omp parallel for schedule(static) if(!g_parallel_members) firstprivate(sc) reduction(+:_member_trn,_member_evl,_agree_trn,_disagree_trn,_agree_evl,_disagree_evl) reduction(min:_scr_min) reduction(max:_scr_max)
                for (int s = 0; s < total_train + total_eval; s++) {
                    int is_eval = (s >= total_train);
                    int ns = is_eval ? s - total_train : s;
                    const uint32_t *gb = is_eval ? mem->gb_buf_te : mem->gb_buf;
                    SCORE_TYPE *acc = is_eval ? acc_votes_te[ns] : acc_votes_tr[s];
                    if (!gb) continue;
                    scores_otto_from_gb(ns, mem->H_local, gb, mem->target, mem->offset, sc);
                    if (g_dbg_member && is_eval) {   /* W0/MIN/MAX/pxz only */
                        for (int k = 0; k < KI_NCLASSES; k++) {
                            if (sc[k] < _scr_min) _scr_min = sc[k];
                            if (sc[k] > _scr_max) _scr_max = sc[k];
                        }
                    }
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
        }   /* end if !_skip_member; the SCORE_TYPE sc block above is the
             * critical(votes) body, no extra closing brace. */

        /* DEBUG REMOVED */

        /* ── [PROF] per-member phase times (--prof only) ── */
        if (g_prof) { gettimeofday(&_pt1, NULL);
            _p_eval += (double)(_pt1.tv_sec - _pt0.tv_sec) * 1000.0
                     + (double)(_pt1.tv_usec - _pt0.tv_usec) / 1000.0;
            double _ptot = _p_cex + _p_gb + _p_target + _p_train + _p_eval;
            /* PRF: members run in parallel → serialize the print + the
             * accumulator updates (small, one line per member). */
            #pragma omp critical(prof)
            {
                printf("[PROF] mb=%d/%d  cex=%8.1fms  gb=%8.1fms  target=%7.1fms  "
                       "train=%8.1fms  eval=%8.1fms  total=%8.1fms\n",
                       mb + 1, active_members, _p_cex, _p_gb, _p_target, _p_train, _p_eval, _ptot);
                _prof_cex += _p_cex; _prof_gb += _p_gb; _prof_target += _p_target;
                _prof_train += _p_train; _prof_eval += _p_eval; _prof_n++;
                fflush(stdout);
            }
        }

        /* ── Per-member progress (--debug-member / --debug-epoch) ──
         * PRF (always on): the cumulative `ens=` line depends on sequential
         * member order and is NOT computed (the final REPORT after the loop
         * carries the ensemble accuracy). The per-member info (mem=, member
         * spec, MIN/MAX/pxz) is member-local and shown here, serialized via
         * critical(prof) because members run in parallel. */
        if (!aa.sweep && (aa.debug_member || debug_epoch)) {
            char _info[128] = "";
            if (aa.debug_member || debug_epoch || (_report_int == 1 && active_members > 1)) {
                const char *_cn = ki_color_name(mem->color_bit >= 0 ? mem->color_bit : 0);
                const char *_en = ki_enc_name_short(mem->enc_type);
                int _ew = mem->enc_width;
                int _xtmp = mem->xform_id;
                const char *_xn = ki_xform_str(_xtmp);
                /* W0 only with explicit --debug-member (PRF default shows
                 * the spec without the weight marker, faster + less noise). */
                if (g_dbg_member)
                    snprintf(_info, sizeof(_info), "  %s:%s:%s%d  W0=0x%08X",
                             _xn, _cn, _en, _ew,
                             mem->W0 ? mem->W0[0] : 0);   /* NULL in KI_BITVOTING (no W0) */
                else
                    snprintf(_info, sizeof(_info), "  %s:%s:%s%d",
                             _xn, _cn, _en, _ew);
            }
            int _filtered = (aa.member_threshold > 0 && mem->trn_acc < (float)aa.member_threshold);
            /* pxz only needed for the explicit --debug-member line */
            int _pxz = 0;
            if (g_dbg_member && mem->input_buf_te && mem->enc_width > 0) {
                int _pack = 32 / mem->enc_width;
                int _cnts[4] = {495 / _pack, 496 / _pack, 527 / _pack, 528 / _pack};
                int _poss[4] = {495 % _pack, 496 % _pack, 527 % _pack, 528 % _pack};
                uint32_t _mask = (1u << mem->enc_width) - 1u;
                for (int _pi = 0; _pi < 4; _pi++)
                    _pxz += (int)((mem->input_buf_te[_cnts[_pi]] >> (_poss[_pi] * mem->enc_width)) & _mask);
                _pxz /= 4;
            }
            gettimeofday(&tv_end, NULL);
            int _el = (int)((tv_end.tv_sec - tv_start.tv_sec) * 1000 + (tv_end.tv_usec - tv_start.tv_usec) / 1000);
            #pragma omp critical(prof)
            {
                if (aa.debug_member || debug_epoch || (_report_int == 1 && active_members > 1)) {
                    /* PRF: no cumulative ens= (member order undefined); the
                     * member-local accuracy mem= and the spec are shown.
                     * W0/MIN/MAX/pxz only with explicit --debug-member. */
                    printf("  [%3d/%d] mem=%.1f%%/%.1f%%/E=%d  time=%dms%s%s",
                           mb + 1, active_members,
                           (float)_member_trn * 100.0f / (float)total_train,
                           (float)_member_evl * 100.0f / (float)(total_eval > 0 ? total_eval : 1),
                           total_eval - _member_evl,
                           _el, _info,
                           _filtered ? "  S" : "");
                    if (g_dbg_member)
                        ki_print_scr_minmax(_scr_min, _scr_max, _pxz);
                    else
                        printf("\n");
                    fflush(stdout);
                }
            }   /* end #pragma omp critical(prof) */
        }   /* end if (!aa.sweep && (debug_member || debug_epoch)) */

        /* ── Export per-member data (before freeing gb_buf) ── */
        /* --export-scores: accumulate per-member scores.
         * PRF: the per-member score computation is local; only the append to
         * the global _export_scores_buf is shared → serialize the append. */
        if (aa.export_scores[0] && !aa.dry_run) {
            size_t _member_sz = (size_t)(total_train + total_eval) * KI_NCLASSES;
            SCORE_TYPE *_m_sc = (SCORE_TYPE *)calloc(_member_sz, sizeof(SCORE_TYPE));
            SCORE_TYPE sc[KI_NCLASSES];
            for (int s = 0; s < total_train; s++) {
                scores_otto_from_gb(s, mem->H_local, mem->gb_buf,
                                   mem->target, mem->offset, sc);
                SCORE_TYPE *dst = _m_sc + (size_t)s * KI_NCLASSES;
                for (int k = 0; k < KI_NCLASSES; k++) dst[k] = sc[k];
            }
            if (total_eval > 0) {
                for (int s = 0; s < total_eval; s++) {
                    scores_otto_from_gb(s, mem->H_local, mem->gb_buf_te,
                                       mem->target, mem->offset, sc);
                    SCORE_TYPE *dst = _m_sc + (size_t)(total_train + s) * KI_NCLASSES;
                    for (int k = 0; k < KI_NCLASSES; k++) dst[k] = sc[k];
                }
            }
            /* Accumulate into global export buffer */
            #pragma omp critical(export)
            {
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
            }   /* #pragma omp critical(export) */
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

        /* ── Write member scores for --debug-member-stats ──
         * PRF: _ms_fp is shared → serialize the whole block (debug-only). */
        if (_ms_fp) {
            #pragma omp critical(msfp)
            {
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
                int _xt = mem->xform_id;
                const char *_xn = ki_xform_str(_xt);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
                snprintf(_ms_m, sizeof(_ms_m), "%s=%s%d-%s", _cn, _en, _ew, _xn);
#pragma GCC diagnostic pop
            } else {
                int _xt = mem->xform_id;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
                snprintf(_ms_m, sizeof(_ms_m), "%s=%s-%s",
                    ki_color_name(mem->color_bit), "?", ki_xform_str(_xt));
#pragma GCC diagnostic pop
            }
            size_t _ms_namelen = strlen(_ms_m) + 1;
            uint8_t *_new = realloc(_ms_meta, (size_t)_ms_meta_n * 64 + _ms_namelen);
            if (_new) {
                _ms_meta = _new;
                memcpy(_ms_meta + (size_t)_ms_meta_n * 64, _ms_m, _ms_namelen);
                _ms_meta_n++;
            }
            }   /* #pragma omp critical(msfp) */
        }

        /* ── Sweep mode: Member-Progress nach Training ── */
        if (aa.sweep && !aa.dry_run) {
            char _mi[128] = "";
            {
                const char *_cn = ki_color_name(mem->color_bit >= 0 ? mem->color_bit : 0);
                const char *_en = ki_enc_name_short(mem->enc_type);
                int _ew = mem->enc_width;
                const char *_xn = ki_xform_str(mem->xform_id);
                snprintf(_mi, sizeof(_mi), "  %s:%s:%s:%d", _xn, _cn, _en, _ew);
            }
            /* PRF: serialize the progress line (shared stdout) */
            #pragma omp critical(progress)
            {
                _sweep_done++;
                /* FIX (2026-08-19): anchor the ETA rate at the FIRST actual TRAIN,
                 * not at loop start. On a restarted sweep the leading SKIP passes
                 * (already-done members) cost ~0 time but would otherwise inflate
                 * the elapsed divisor and deflate the rate. */
                if (!tv_work_start_set) {
                    gettimeofday(&tv_work_start, NULL);
                    tv_work_start_set = 1;
                }
                printf("  [%3d/%d] TRAIN%s  err=%d/%d  evl=%d/%d (%.1f%%)",
                       mb+1, active_members, _mi,
                       mem->last_err, total_train,
                       _member_evl, total_eval,
                       (float)_member_evl * 100.0f / (float)(total_eval > 0 ? total_eval : 1));
                /* ETA from elapsed TRAIN time and _sweep_done (2026-08-16, fixed
                 * 2026-08-19): rate = done / train-elapsed. Remaining work is only
                 * the members neither SKIPped (already done) nor trained yet:
                 * active_members - _sweep_skipped - _sweep_done. The old formula
                 * (active_members - _sweep_done) counted every skipped member as
                 * still-to-do → ETA far too high on a restarted sweep. */
                if (_sweep_done >= 20) {
                    struct timeval _eta_tv;
                    gettimeofday(&_eta_tv, NULL);
                    double _el = (double)(_eta_tv.tv_sec - tv_work_start.tv_sec)
                               + (double)(_eta_tv.tv_usec - tv_work_start.tv_usec) / 1e6;
                    if (_el > 0.0) {
                        double _rate = (double)_sweep_done / _el; /* members/s */
                        double _remaining = (double)(active_members - _sweep_skipped - _sweep_done);
                        if (_remaining < 0.0) _remaining = 0.0;
                        double _eta = _remaining / _rate;
                        int _eh = (int)(_eta / 3600.0);
                        int _em = (int)(_eta / 60.0) % 60;
                        printf("  ETA %dh%02dm (%.1f/min)", _eh, _em, _rate * 60.0);
                    }
                } else {
                    printf("  ETA ...");
                }
                printf("\n");
                fflush(stdout);
            }
        }

        /* ── Per-member .ens export (immer nach Training, vor gb-free) ── */
        if (!aa.dry_run && total_eval > 0 && aa.export_merge_scores[0]) {
            /* PRF: export writes its OWN file per member — thread-safe by
             * construction (unique path per member), no lock needed. */
            export_one_member_ens(mem, mb, aa.export_merge_scores,
                                  y_te, total_eval, (int)n_cont);
            if (aa.sweep) {
                #pragma omp atomic
                _sweep_trained++;
            }
        }

        /* ── Free per-member gb ──
         * gb_buf_te is KEPT in the non-sweep case: print_class_voting_debug()
         * runs AFTER this loop and reads mem->gb_buf_te (the member's own CEX
         * input) to compute the ens-total row — freeing it here forced the
         * X_te fallback, which is wrong per-member in PRF mode (Pullover 0%
         * vs 81.9% in the confusion matrix, bug 2026-08-12). It is freed
         * after the debug outputs (see the end of the RESULT block). In
         * sweep mode the debug flags are ignored and ki_member_destroy
         * frees the member anyway. */
        free(mem->gb_buf); mem->gb_buf = NULL;
        if (aa.sweep) { free(mem->gb_buf_te); mem->gb_buf_te = NULL; }

        /* ── Sweep mode: destroy member immediately ── */
        if (aa.sweep) {
            ki_member_destroy(mem);
            members[mb] = NULL;
        }
    }

    /* Restore the global team (exclusive case only — see the comment before
     * the member loop): the ensemble-vote accumulation after the loop is
     * sequential anyway, but anything else after this point must not
     * accidentally run nested-parallel on a stale team. */
    if (!g_parallel_members) {
        aa.threadN = _save_threadn;
        omp_set_num_threads(1);
    }

    /* ── [PROF] summed phase summary (--prof only) ── */
    if (g_prof && _prof_n > 0) {
        double _ptot = _prof_cex + _prof_gb + _prof_target + _prof_train + _prof_eval;
        printf("[PROF] SUM over %d members:  cex=%8.1fms (%5.1f%%)  gb=%8.1fms (%5.1f%%)  "
               "target=%7.1fms (%5.1f%%)  train=%8.1fms (%5.1f%%)  eval=%8.1fms (%5.1f%%)  "
               "total=%8.1fms\n",
               _prof_n,
               _prof_cex,   100.0 * _prof_cex    / _ptot,
               _prof_gb,    100.0 * _prof_gb     / _ptot,
               _prof_target,100.0 * _prof_target / _ptot,
               _prof_train, 100.0 * _prof_train  / _ptot,
               _prof_eval,  100.0 * _prof_eval   / _ptot,
               _ptot);
        fflush(stdout);
    }
    /* Final release of the single-slot xform cache (last xform group).
     * Freeing here keeps peak RAM at 1× transformed buffer even before the
     * post-loop ensemble eval (non-sweep) or process exit (sweep). */
    ki_clear_cache(-1);

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
    if (aa.sweep) {
        printf("\n══╡ SWEEP COMPLETE ╞══════════════════════════════════════════════\n");
        printf("  %d members: %d trained, %d skipped (time=%dms)\n",
               active_members, _sweep_trained, _sweep_skipped, elapsed_ms);

        /* ── Evaulate ensemble from per-member .ens files ── */
        int _ens_trn = 0, _ens_evl = 0;
        for (int s = 0; s < total_train; s++) {
            int pred = -1;
            for (int k = 0; k < KI_NCLASSES; k++)
                if ((acc_votes_tr[s][k] > 0 || acc_votes_tr[s][k] < 0) && (pred < 0 || acc_votes_tr[s][k] > acc_votes_tr[s][pred]))
                    pred = k;
            if (pred >= 0 && pred == (int)y_tr[s]) _ens_trn++;
        }
        for (int s = 0; s < total_eval; s++) {
            int pred = -1;
            for (int k = 0; k < KI_NCLASSES; k++)
                if ((acc_votes_te[s][k] > 0 || acc_votes_te[s][k] < 0) && (pred < 0 || acc_votes_te[s][k] > acc_votes_te[s][pred]))
                    pred = k;
            if (pred >= 0 && pred == (int)y_te[s]) _ens_evl++;
        }
        final_trn_ok = _ens_trn;
        final_evl_ok = _ens_evl;
    }

    /* ── PRF (--prf): the cumulative ens= line is skipped inside the loop
     * (member order undefined in parallel), so the ensemble accuracy must be
     * recomputed here from the accumulated votes — same as the sweep path. */
    if (g_prf && !aa.sweep) {
        int _ens_trn = 0, _ens_evl = 0;
        for (int s = 0; s < total_train; s++) {
            int pred = -1;
            for (int k = 0; k < KI_NCLASSES; k++)
                if ((acc_votes_tr[s][k] > 0 || acc_votes_tr[s][k] < 0) && (pred < 0 || acc_votes_tr[s][k] > acc_votes_tr[s][pred]))
                    pred = k;
            if (pred >= 0 && pred == (int)y_tr[s]) _ens_trn++;
        }
        for (int s = 0; s < total_eval; s++) {
            int pred = -1;
            for (int k = 0; k < KI_NCLASSES; k++)
                if ((acc_votes_te[s][k] > 0 || acc_votes_te[s][k] < 0) && (pred < 0 || acc_votes_te[s][k] > acc_votes_te[s][pred]))
                    pred = k;
            if (pred >= 0 && pred == (int)y_te[s]) _ens_evl++;
        }
        final_trn_ok = _ens_trn;
        final_evl_ok = _ens_evl;
    }

    if (!aa.sweep)
        printf("\n══╡ DONE ╞══  %s training complete (%d members, %dms)\n",
               g_prf ? "parallel (PRF)" : "sequential",
               active_members, elapsed_ms);

    /* ── Final report ── */
    int trn_ok = final_trn_ok, evl_ok = final_evl_ok;
    /* pred_eval: built when predictions are requested OR when the confusion
     * matrix runs on the eval split (2026-08-12: --debug-confusion-matrix now
     * shows evl% data, same split as --debug-class-voting). */
    uint8_t *pred_eval = (aa.predictions[0] || (aa.debug_confusion && !aa.dry_run))
        ? (uint8_t *)ki_xcalloc((size_t)total_eval, sizeof(uint8_t)) : NULL ;
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
        /* EVAL-split (evl%), not train — the specialist question is about
         * generalisation ("does S_k beat all other specialists on class k?"),
         * train% is meaningless for that (fix 2026-08-12: table showed trn%
         * of freshly retrained members, which drifted far from the .ens
         * eval-scores the merge-ensemble selection is based on). */
        print_class_voting_debug(members, active_members,
                                 X_te, y_te, total_eval, (int)n_cont, epochs - 1);
    }
    if (aa.debug_confusion && !aa.dry_run) {
        /* EVAL split + ENSEMBLE (2026-08-12): the confusion matrix now shows
         * the ensemble decision on the eval data (y_te / pred_eval, same
         * split as --debug-class-voting) — train% was meaningless for the
         * K2/K6 similarity question. */
        print_confusion_debug(y_te, pred_eval, total_eval, epochs - 1, 1);
    }
    free(pred_tr);
    if (pred_eval && !aa.predictions[0]) free(pred_eval);

    /* Free the eval-gb buffers kept for the debug outputs above
     * (non-sweep: print_class_voting_debug read mem->gb_buf_te; see the
     * member-loop free comment, bug 2026-08-12). */
    if (!aa.sweep) {
        for (int _z = 0; _z < active_members; _z++)
            if (members[_z]) { free(members[_z]->gb_buf_te); members[_z]->gb_buf_te = NULL; }
    }

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

    if (!aa.sweep) {
    printf("\n══╡ RESULT ╞══════════════════════════════════════════════════════\n");
    float final_best = (best_evl_ok > evl_ok) ? (float)best_evl_ok : (float)evl_ok;
    final_best = final_best * 100.0f / (float)total_eval;
    printf("  H=%d  ens=%d  v_split=%d  h_split=%d  ep=%d  trn=%.1f%%  evl=%.1f%%  best=%.1f%%  lr=%.4f  time=%dms\n",
           H, ensembleN, splitVN, splitHN, aa.cfg_epochs, fin_trn, fin_evl, final_best,
           (double)aa.lr, elapsed_ms);

    /* REPORT uses best eval across all member evaluations.
     * PRF: report the REAL parallel width. With many members the outer loop
     * uses min(active, g_prf_width) threads; with FEW members (1) the inner
     * sample loops use g_prf_width cores (exclusive parallelism, 2026-08-09)
     * — the effective width is g_prf_width in both cases. */
    int report_evl_ok = (best_evl_ok > 0) ? best_evl_ok : final_evl_ok;
    int _rep_threads = g_prf_width > 0 ? g_prf_width : 1;
    ki_report_show(trn_ok, total_train, report_evl_ok, total_eval,
                   elapsed_ms, _rep_threads, fin_err, aa.lr, active_members, 0);

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

    }  /* if (!aa.sweep) — end of RESULT block */

    if (own_eval_data) { free(X_perm); free(y_perm); }
    /* Member CEX input buffers are owned by each member and freed via
     * ki_member_destroy (called per member before this point). */
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
