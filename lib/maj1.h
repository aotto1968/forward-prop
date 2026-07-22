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
 *   uint32_t r = majority_tree1(match, 256);
 *
 * match[]: uint32 array of length n (pre-XNOR'd values)
 * n:       number of containers (256 for CIFAR, 196 for MNIST)
 * return:  per bit: 1 if popcount(match[*], Bit b) > n/2
 */
#ifndef MAJ_TRUE_H
#define MAJ_TRUE_H

#include <stdint.h>

/**
 * majority_tree1 — Exact bitwise majority over n uint32 values.
 *
 * For each of the 32 bits:
 *   output[b] = (popcount( match[0..n-1], Bit b ) > n/2)
 *
 * That is: a bit is 1 when MORE THAN HALF of the match values
 * have that bit set.
 */
static inline uint32_t majority_tree1(const uint32_t *vals, int n) {
    if (n <= 0) return 0;
    int bits[32] = {0};
    int half = n / 2;

    /* Popcount per bit position across all n values */
    for (int i = 0; i < n; i++) {
        uint32_t v = vals[i];
        if (v == 0) continue;  /* fast skip for empty containers */
        /* Manual bit decomposition (no popcount intrinsic needed) */
        for (int b = 0; b < 32; b++)
            if (v & (1u << b)) bits[b]++;
    }

    /* Ergebnis: Bit b = 1 wenn bits[b] > half */
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
 * @return              32-bit majority result
 */
static inline uint32_t majority_tree1_rowwise(const uint32_t *vals, int n,
                                              int cont_per_row) {
    /* ── Validation ──────────────────────────────────────────── */
    if (cont_per_row < 1) return majority_tree1(vals, n);
    int rows = n / cont_per_row;
    if (rows * cont_per_row != n || rows < 2)
        return majority_tree1(vals, n);   /* fallback: flat */
    if (rows > 256) rows = 256;          /* safety limit */

    /* ── Phase 1: per-row majority ────────────────────────────── */
    uint32_t row_maj[256];
    for (int r = 0; r < rows; r++)
        row_maj[r] = majority_tree1(vals + r * cont_per_row, cont_per_row);

    /* ── Phase 2: cross-row majority ──────────────────────────── */
    return majority_tree1(row_maj, rows);
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
