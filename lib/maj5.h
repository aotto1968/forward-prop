/*
 * lib/maj5.h — Majority-of-5 Bit-Logic (header only)
 * ===================================================
 *
 * Provides:
 *   maj5(a,b,c,d,e)          — Majority of 5 uint32 (bit-parallel)
 *   majority_tree5(vals, n)  — Majority tree over n values using maj5
 *
 * Each maj5 node: output bit b = 1 if ≥3 of the 5 inputs have bit b = 1.
 * Compared to maj3 (2/3 ≈ 67% threshold), maj5 has 3/5 = 60% threshold —
 * slightly softer, but with 5 inputs per node the output is more
 * robust against noise.
 *
 * Tree structure: partitions into 5-groups, each reduced by maj5.
 * Uses the same fix for remainder cases as maj3.h (n%5 != 0).
 *
 * Usage:
 *   #include "maj5.h"
 *   uint32_t r = majority_tree5(vals, 256);
 *
 * BUF: buffer size (uint32 count), default 4096 = 16 KB stack
 *      Overridable via -DMAJ5_BUF=8192
 */
#ifndef MAJ5_H
#define MAJ5_H

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef MAJ5_BUF
#  define MAJ5_BUF  4096
#endif

/**
 * maj5(a,b,c,d,e) — Bitwise Majority of 5 uint32.
 *
 * For each bit: 1 if at least 4 of the 5 inputs are 1 (80% threshold).
 *   result = (a&b&c&d)|(a&b&c&e)|(a&b&d&e)|(a&c&d&e)|(b&c&d&e)
 */
static inline uint32_t maj5(uint32_t a, uint32_t b, uint32_t c,
                             uint32_t d, uint32_t e) {
    return (a & b & c & d) | (a & b & c & e)
         | (a & b & d & e) | (a & c & d & e)
         | (b & c & d & e);
}

/**
 * majority_tree5(vals, n) — Majority tree over n uint32 values using maj5.
 *
 * Reduces n uint32 values to 1 via majority tree of 5-input nodes.
 * Each level: partition into 5-groups → maj5 per group.
 *
 * @param vals  pointer to n uint32 values
 * @param n     number of values (> 0, ≤ MAJ5_BUF when using stack)
 * @return      Majority over all n values (per bit)
 */
static inline uint32_t majority_tree5(const uint32_t *vals, int n) {
    uint32_t buf[MAJ5_BUF];
    if (n <= 0) return 0;
    if (n > MAJ5_BUF) n = MAJ5_BUF;
    memcpy(buf, vals, (size_t)n * sizeof(uint32_t));

    while (n > 1) {
        int n_out = 0;

        if (n % 5 == 0) {
            /* Exact division: pure 5-groups */
            for (int i = 0; i < n; i += 5) {
                buf[n_out++] = maj5(buf[i], buf[i+1], buf[i+2], buf[i+3], buf[i+4]);
            }
        } else if (n < 5) {
            /* Small remainder: AND (4/5 threshold → all must agree for <5 inputs) */
            uint32_t r = buf[0];
            for (int i = 1; i < n; i++) r &= buf[i];
            buf[n_out++] = r;
        } else {
            /* n%5 == 1..4: process full 5-groups, handle remainder */
            int full = n / 5;
            for (int i = 0; i < full * 5; i += 5)
                buf[n_out++] = maj5(buf[i], buf[i+1], buf[i+2], buf[i+3], buf[i+4]);
            int rem = n - full * 5;
            /* Remainder: use AND (at 80% threshold, all remaining must agree) */
            uint32_t r = buf[full * 5];
            for (int j = 1; j < rem; j++)
                r &= buf[full * 5 + j];
            buf[n_out++] = r;
        }
        n = n_out;
    }
    return buf[0];
}

#endif /* MAJ5_H */
