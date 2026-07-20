/*
 * lib/maj4.h — Majority-of-4 Bit-Logic (header only)
 * ===================================================
 *
 * Provides:
 *   maj4(a,b,c,d)           — Majority of 4 (bit-parallel)
 *   majority_tree4(vals,n)  — Majority tree over n values using maj4
 *
 * maj4: output bit b = 1 if ≥2 of the 4 inputs have bit b = 1.
 * Threshold 2/4 = 50% — same as maj2 per-node, but the tree is
 * shallower (log₄n levels) so more structural diversity than maj2.
 *
 * Usage:
 *   #include "maj4.h"
 *   uint32_t r = majority_tree4(vals, 256);
 */
#ifndef MAJ4_H
#define MAJ4_H

#include <stdint.h>
#include <string.h>

#ifndef MAJ4_BUF
#  define MAJ4_BUF  4096
#endif

/**
 * maj4(a,b,c,d) — Majority of 4. At least 2 of 4 must be 1 (50%).
 *   result = (a&b)|(a&c)|(a&d)|(b&c)|(b&d)|(c&d)
 */
static inline uint32_t maj4(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    return (a & b) | (a & c) | (a & d)
         | (b & c) | (b & d)
         | (c & d);
}

/**
 * majority_tree4(vals, n) — Majority tree over n uint32 values using maj4.
 *
 * Each level: partition into 4-groups → maj4 per group.
 * Handles remainder: 1-3 remaining values use AND (all must agree).
 */
static inline uint32_t majority_tree4(const uint32_t *vals, int n) {
    uint32_t buf[MAJ4_BUF];
    if (n <= 0) return 0;
    if (n > MAJ4_BUF) n = MAJ4_BUF;
    memcpy(buf, vals, (size_t)n * sizeof(uint32_t));

    while (n > 1) {
        int n_out = 0;
        if (n % 4 == 0) {
            for (int i = 0; i < n; i += 4)
                buf[n_out++] = maj4(buf[i], buf[i+1], buf[i+2], buf[i+3]);
        } else {
            int full = n / 4;
            for (int i = 0; i < full * 4; i += 4)
                buf[n_out++] = maj4(buf[i], buf[i+1], buf[i+2], buf[i+3]);
            int rem = n - full * 4;
            if (rem == 1) {
                buf[n_out++] = buf[full * 4];
            } else {
                /* rem == 2 or 3: OR (at least 1 of rem must be 1 = 50%) */
                uint32_t r = buf[full * 4];
                for (int j = 1; j < rem; j++)
                    r |= buf[full * 4 + j];
                buf[n_out++] = r;
            }
        }
        n = n_out;
    }
    return buf[0];
}

#endif /* MAJ4_H */
