/*
 * lib/maj1.h — Exact bitwise majority (header only)
 * =================================================
 *
 * Replaces majority_tree3 (tree approximation) with true bitwise
 * majority: For each of the 32 bits, count across all N containers
 * whether the majority of bits is set.
 *
 * No approximation artifacts — bit-exact like the real DRAM chip.
 *
 * Usage:
 *   #include "maj1.h"
 *   uint32_t r = majority_tree1(match, 256, 128);
 *
 * match[]: uint32 array of length n (pre-XNOR'd values)
 * n:       number of containers (256 for CIFAR, 196 for MNIST)
 * return:  per bit: 1 if popcount(match[*], Bit b) > n/2
 */
#ifndef MAJ_TRUE_H
#define MAJ_TRUE_H

#include <stdint.h>

/**
 * ki_default_half — Default majority threshold for a given container count.
 *
 * Returns the pre-tuned threshold for known n values.
 * Default formula: n × 107/196 ≈ 54.6% (optimized for MNIST H=196).
 * Used when aa.maj1_thresh == -2 (auto per encoding).
 */
static inline int ki_default_half(int n) {
    switch (n) {
        case 196:  return 107;
        case 256:  return 135;
        case 512:  return 269;
        case 1024: return 540;
        default:   return n * 135 / 256;  // close to maj3
    }
}

/**
 * ki_compute_half — Centralized half-threshold computation.
 *
 * Computes the majority threshold (half) from aa.maj1_thresh and NC_slice.
 * Replaces the inline ternary repeated in 5+ places.
 *
 * @param NC_slice  number of containers per member slice
 * @return          computed half threshold
 */
static inline int ki_compute_half(int NC_slice) {
    return (aa.maj1_thresh == -2) ? ki_default_half(NC_slice) :
           (aa.maj1_thresh < 0)  ? NC_slice / 2 :
           aa.maj1_thresh;
}

/**
 * majority_tree1 — Exact bitwise majority over n uint32 values.
 *
 * For each of the 32 bits:
 *   output[b] = (popcount( match[0..n-1], Bit b ) > half)
 *
 * NOTE: half is NOT computed here — it is precomputed per member
 * (see ki_Member.half) and passed in directly.  This avoids repeated
 * multiplications in the inner loop.
 *
 * @param vals  match[] array (pre-XNOR'd)
 * @param n     number of containers
 * @param half  threshold count (bit set if popcount > half)
 */
static inline uint32_t majority_tree1(const uint32_t *vals, int n, int half) {
    if (n <= 0) return 0;
    int bits[32] = {0};

    /* Popcount per bit position across all n values.
     *
     * TESTED AND REJECTED ALTERNATIVES (all slower on Zen 4):
     *
     * 1. __builtin_ctz  (2026-07-23):  ~2.3× SLOWER (49s vs 21s @ H=2048).
     *    TZCNT has 3-5 cy latency and the data-dependent v &= v-1 loop
     *    prevents superscalar execution + compiler unrolling/vectorization.
     *    See plan-2026-07-23-ctz-slower.md.
     *
     * 2. Manual 32-way unrolled (e.g. bits[0] += (v>>0)&1; …): zero speed
     *    benefit — gcc -O3 unrolls the small loop identically.
     *
     * 3. AVX-512 SIMD majority_impl_avx512: BUGGY — counted per-lane
     *    (every 16th value) instead of per-bit-position.  Removed.
     *
     * 4. AVX2 SIMD majority_impl_avx2: SAME BUG (every 8th value).
     *    Result was always 0 for n > 8 because max per-lane count = n/8
     *    never exceeded threshold n/2.  Removed.
     *
     * 5. Inline assembly (2026-07-24): theoretically possible with 4-pass
     *    byte processing + 8 GPR counters, but gcc -O3 -march=native
     *    already auto-vectorizes this loop to AVX-512 (ZMM register
     *    counters, opmask for v==0 skip).  Measured overhead of
     *    majority_tree1 is <5% of total training time — no headroom.
     *
     * CONCLUSION: The simple 32-iteration loop is optimal.  gcc keeps
     * the bits[32] array in ZMM registers (no stack spill), auto-vectorizes
     * the inner loop, and the predictable branch pattern + superscalar
     * execution saturates the pipeline. */
    for (int i = 0; i < n; i++) {
        uint32_t v = vals[i];
        if (v == 0) continue;
        for (int b = 0; b < 32; b++)
            if (v & (1u << b)) bits[b]++;
    }

    uint32_t r = 0;
    for (int b = 0; b < 32; b++)
        if (bits[b] > half) r |= (1u << b);
    return r;
}

/**
 * majority_tree1_rowwise — Row-wise majority (32×32 grid)
 *
 * Splits the flat match[] array into KI_ROWS independent rows and
 * computes a per-bit majority for each row. Then the row-wise
 * majority is aggregated per-bit again.
 *
 * This respects the image structure: containers from different rows
 * are not compared directly.
 *
 * @param vals          match[] array (pre-XNOR'd)
 * @param n             number of containers (nc_local)
 * @param cont_per_row  containers per image row (KI_COLS * width / 32)
 * @param half          threshold count (bit set if popcount > half)
 * @return              32-bit majority result
 */
static inline uint32_t majority_tree1_rowwise(const uint32_t *vals, int n,
                                              int cont_per_row, int half) {
    /* ── Validation ──────────────────────────────────────────── */
    if (cont_per_row < 1) return majority_tree1(vals, n, half);
    int rows = n / cont_per_row;
    if (rows * cont_per_row != n || rows < 2)
        return majority_tree1(vals, n, half);   /* fallback: flat */
    if (rows > 256) rows = 256;                /* safety limit */

    /* ── Phase 1: per-row majority ────────────────────────────── */
    uint32_t row_maj[256];
    for (int r = 0; r < rows; r++)
        row_maj[r] = majority_tree1(vals + r * cont_per_row, cont_per_row, half);

    /* ── Phase 2: cross-row majority ──────────────────────────── */
    return majority_tree1(row_maj, rows, half);
}

/**
 * majority_tree1_pixel — Pixel-accurate majority
 *
 * Instead of per-container-bit majority, counts per bit position across ALL
 * pixels. KI_BIT_WIDTH (from dataset-specific ki-local.h) controls:
 *   KI_BIT_WIDTH=8:  4 px/cont, 8 bit-pos → 4 groups → 32 output bits
 *   KI_BIT_WIDTH=16: 2 px/cont, 16 bit-pos → 2 groups → 32 output bits
 *   KI_BIT_WIDTH=32: 1 px/cont, 32 bit-pos → 1 group → 32 output bits
 *
 * @param vals    match[] array (pre-XNOR'd), length n_cont
 * @param n_cont  number of containers (e.g. 8 for rows, 256 for flat)
 * @param half    threshold = total_pixel / 2
 * @return        KI_BIT_POS-bit result (in bits 0..KI_BIT_POS-1)
 */
#ifndef KI_PX_PER_CONT
#  error "KI_PX_PER_CONT not defined — include ki-common.h before maj1.h"
#endif

static inline uint32_t majority_tree1_pixel(const uint32_t *vals, int n_cont, int half) {
    if (n_cont <= 0) return 0;
    int bits[KI_BIT_POS];
    for (int b = 0; b < KI_BIT_POS; b++) bits[b] = 0;

    for (int c = 0; c < n_cont; c++) {
        uint32_t v = vals[c];
        if (v == 0) continue;
        /* NOTE: __builtin_ctz tested — ~2× slower than fixed 32-iteration
         * loop because TZCNT latency breaks superscalar execution and the
         * data-dependent loop prevents unrolling.  Keep simple loop. */
#if KI_BIT_WIDTH == 8
        uint8_t px0 = (uint8_t)(v >>  0);
        uint8_t px1 = (uint8_t)(v >>  8);
        uint8_t px2 = (uint8_t)(v >> 16);
        uint8_t px3 = (uint8_t)(v >> 24);
        for (int b = 0; b < 8; b++) {
            if (px0 & (1 << b)) bits[b]++;
            if (px1 & (1 << b)) bits[b]++;
            if (px2 & (1 << b)) bits[b]++;
            if (px3 & (1 << b)) bits[b]++;
        }
#elif KI_BIT_WIDTH == 16
        uint16_t px0 = (uint16_t)(v >>  0);
        uint16_t px1 = (uint16_t)(v >> 16);
        for (int b = 0; b < 16; b++) {
            if (px0 & (1 << b)) bits[b]++;
            if (px1 & (1 << b)) bits[b]++;
        }
#else /* KI_BIT_WIDTH == 32 */
        for (int b = 0; b < 32; b++)
            if (v & (1u << b)) bits[b]++;
#endif
    }

    uint32_t r = 0;
    for (int b = 0; b < KI_BIT_POS; b++)
        if (bits[b] > half) r |= (1u << b);
    return r;
}

#endif /* MAJ_TRUE_H */
