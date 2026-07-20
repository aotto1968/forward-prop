/*
 * lib/maj2.h — Majority-of-2 Bit-Logic (header only)
 * ===================================================
 *
 * Provides:
 *   maj2(a,b)              — Majority of 2 = OR (bit-parallel)
 *   majority_tree2(vals,n) — Majority tree over n values using maj2
 *
 * maj2(a,b) = a | b  (1/2 = 50% threshold: at least 1 of 2)
 *
 * Tree: each level partitions into 2-groups, each reduced by OR.
 * Very deep tree (log₂n levels) → almost all bits survive → noisy.
 * Shows the baseline: what happens with minimal filtering.
 *
 * Usage:
 *   #include "maj2.h"
 *   uint32_t r = majority_tree2(vals, 256);
 */
#ifndef MAJ2_H
#define MAJ2_H

#include <stdint.h>
#include <string.h>

#ifndef MAJ2_BUF
#  define MAJ2_BUF  4096
#endif

/**
 * maj2(a,b) — Majority of 2 = OR (at least 1 of 2). Bit-parallel.
 * Threshold 1/2 = 50%: output[b] = 1 if a[b]=1 OR b[b]=1.
 */
static inline uint32_t maj2(uint32_t a, uint32_t b) {
    return a | b;
}

/**
 * majority_tree2(vals, n) — OR tree over n uint32 values.
 *
 * Each level: partition into 2-groups, OR per group.
 * With OR, almost all bits survive → minimal filtering.
 */
static inline uint32_t majority_tree2(const uint32_t *vals, int n) {
    uint32_t buf[MAJ2_BUF];
    if (n <= 0) return 0;
    if (n > MAJ2_BUF) n = MAJ2_BUF;
    memcpy(buf, vals, (size_t)n * sizeof(uint32_t));

    while (n > 1) {
        int n_out = 0;
        for (int i = 0; i < n; i += 2) {
            if (i + 1 < n)
                buf[n_out++] = maj2(buf[i], buf[i+1]);
            else
                buf[n_out++] = buf[i];  /* passthrough */
        }
        n = n_out;
    }
    return buf[0];
}

#endif /* MAJ2_H */
