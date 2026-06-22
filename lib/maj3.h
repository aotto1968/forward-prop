/*
 * lib/maj3.h — Majority-of-3 Bit-Logic (header only)
 * ===================================================
 *
 * Provides:
 *   maj3(a,b,c)              — Majority of 3 uint32 (bit-parallel)
 *   majority_tree(vals, n)  — Majority tree over n values
 *
 * Fix 2026-06-17: Passthrough cascade at n = 3^k+1
 *   When n%3==1 at EVERY level, a single value without majority vote
 *   propagates through all levels → final level AND blocks result.
 *   Fix: n%3==1 → process n-4 as 3-groups + final 4-group.
 *
 * Usage:
 *   #include "maj3.h"
 *   uint32_t r = majority_tree(data, 196);
 *
 * BUF: buffer size (uint32 count), default 4096 = 16 KB stack
 *      Overridable via -DMAJ3_BUF=8192
 */
#ifndef MAJ3_H
#define MAJ3_H

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef MAJ3_BUF
#  define MAJ3_BUF  4096
#endif


/**
 * maj3(a,b,c) — Bitwise Majority of 3 uint32.
 *
 * For each bit: 1 if at least 2 of the 3 inputs are 1.
 *   result = (a & b) | (a & c) | (b & c)
 */
static inline uint32_t maj3(uint32_t a, uint32_t b, uint32_t c) {
    return (a & b) | (a & c) | (b & c);
}


/**
 * majority_tree(vals, n) — Majority tree over n uint32 values.
 *
 * Reduces n uint32 values to 1 uint32 via majority tree.
 * Each level: partition into 3-groups → maj3 per group.
 *
 * Fix for n%3==1 (2026-06-17):
 *   Original: last element as passthrough → at n=3^k+1 cascades
 *   → one value blocks final AND → result ≈ 0.
 *   New: process n-4 as 3-groups + final 4-group
 *   → majority of 4 = (a&b&c)|(a&b&d)|(a&c&d)|(b&c&d)
 *
 * @param vals  pointer to n uint32 values
 * @param n     number of values (> 0, ≤ MAJ3_BUF when using stack)
 * @return      Majority over all n values (per bit)
 */
static inline uint32_t majority_tree(const uint32_t *vals, int n) {
    uint32_t buf[MAJ3_BUF];
    if (n <= 0) return 0;
    if (n > MAJ3_BUF) n = MAJ3_BUF;
    memcpy(buf, vals, (size_t)n * sizeof(uint32_t));

    while (n > 1) {
        int n_out = 0;

        if (n % 3 == 1 && n > 4) {
            /* Fix for n%3==1: 4-group at end instead of passthrough cascade */
            int full = (n - 4) / 3;
            for (int i = 0; i < full * 3; i += 3)
                buf[n_out++] = maj3(buf[i], buf[i+1], buf[i+2]);
            int j = full * 3;
            uint32_t a = buf[j], b = buf[j+1], c = buf[j+2], d = buf[j+3];
            /* majority of 4: at least 3 of 4 must be 1 */
            buf[n_out++] = (a & b & c) | (a & b & d) | (a & c & d) | (b & c & d);

        } else if (n % 3 == 2 && n > 5) {
            /* Fix for n%3==2: 5-group at end instead of AND cascade */
            int full = (n - 5) / 3;
            for (int i = 0; i < full * 3; i += 3)
                buf[n_out++] = maj3(buf[i], buf[i+1], buf[i+2]);
            int j = full * 3;
            uint32_t a = buf[j], b = buf[j+1], c = buf[j+2], d = buf[j+3], e = buf[j+4];
            /* majority of 5: at least 3 of 5 must be 1 */
            buf[n_out++] = (a & b & c) | (a & b & d) | (a & b & e)
                         | (a & c & d) | (a & c & e) | (a & d & e)
                         | (b & c & d) | (b & c & e) | (b & d & e)
                         | (c & d & e);

        } else {
            /* Normal path: 3-groups + remainder */
            for (int i = 0; i < n; i += 3) {
                uint32_t a = buf[i];
                if (i + 2 < n)
                    buf[n_out++] = maj3(a, buf[i+1], buf[i+2]);
                else if (i + 1 < n)
                    buf[n_out++] = a & buf[i+1];
                else
                    buf[n_out++] = a;
            }
        }
        n = n_out;
    }
    return buf[0];
}

#endif /* MAJ3_H */
