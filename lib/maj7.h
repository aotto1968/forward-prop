/*
 * lib/maj7.h — Majority-of-7 Bit-Logic (header only)
 * ===================================================
 *
 * Provides:
 *   maj7(a,b,c,d,e,f,g)    — Majority of 7 (bit-parallel)
 *   majority_tree7(vals,n) — Majority tree over n values using maj7
 *
 * maj7: output bit b = 1 if ≥5 of the 7 inputs have bit b = 1.
 * Threshold 5/7 ≈ 71.4% — slightly above maj3 (67%).
 * 21 AND terms: C(7,5) = 21.
 *
 * Usage:
 *   #include "maj7.h"
 *   uint32_t r = majority_tree7(vals, 256);
 */
#ifndef MAJ7_H
#define MAJ7_H

#include <stdint.h>
#include <string.h>

#ifndef MAJ7_BUF
#  define MAJ7_BUF  4096
#endif

/**
 * maj7(a,b,c,d,e,f,g) — Majority of 7. At least 5 of 7 must be 1.
 */
static inline uint32_t maj7(uint32_t a, uint32_t b, uint32_t c,
                             uint32_t d, uint32_t e, uint32_t f,
                             uint32_t g) {
    return
    (a&b&c&d&e) | (a&b&c&d&f) | (a&b&c&d&g) | (a&b&c&e&f) | (a&b&c&e&g) |
    (a&b&c&f&g) | (a&b&d&e&f) | (a&b&d&e&g) | (a&b&d&f&g) | (a&b&e&f&g) |
    (a&c&d&e&f) | (a&c&d&e&g) | (a&c&d&f&g) | (a&c&e&f&g) | (a&d&e&f&g) |
    (b&c&d&e&f) | (b&c&d&e&g) | (b&c&d&f&g) | (b&c&e&f&g) | (b&d&e&f&g) |
    (c&d&e&f&g);
}

/**
 * majority_tree7(vals, n) — Majority tree over n uint32 values using maj7.
 *
 * Reduces n values to 1 via a tree of 7-input majority nodes.
 * Each level: partition into 7-groups → maj7 per group.
 *
 * Remainder handling (5/7 ≈ 71% threshold):
 *   R=1: passthrough       (1/1 = 100%)
 *   R=2: AND (a&b)         (2/2 = 100%)
 *   R=3: AND (all 3)       (3/3 = 100%)
 *   R=4: 3 of 4            (3/4 = 75% ≥ 71%)
 *   R=5: 4 of 5            (4/5 = 80% ≥ 71%)
 *   R=6: 5 of 6            (5/6 = 83% ≥ 71%)
 *
 * For large n with n%7 == 1, uses the same fix as maj3.h:
 * process n-7 as 7-groups + final 7-group at end.
 */
static inline uint32_t majority_tree7(const uint32_t *vals, int n) {
    uint32_t buf[MAJ7_BUF];
    if (n <= 0) return 0;
    if (n > MAJ7_BUF) n = MAJ7_BUF;
    memcpy(buf, vals, (size_t)n * sizeof(uint32_t));

    while (n > 1) {
        int n_out = 0;

        if (n % 7 == 1 && n > 7) {
            /* Fix for n%7==1: process n-7 as 7-groups + final 7-group */
            int full = (n - 7) / 7;
            for (int i = 0; i < full * 7; i += 7)
                buf[n_out++] = maj7(buf[i], buf[i+1], buf[i+2],
                                    buf[i+3], buf[i+4], buf[i+5], buf[i+6]);
            int j = full * 7;
            buf[n_out++] = maj7(buf[j], buf[j+1], buf[j+2],
                                buf[j+3], buf[j+4], buf[j+5], buf[j+6]);

        } else if (n % 7 == 0) {
            /* Pure 7-groups */
            for (int i = 0; i < n; i += 7)
                buf[n_out++] = maj7(buf[i], buf[i+1], buf[i+2],
                                    buf[i+3], buf[i+4], buf[i+5], buf[i+6]);
        } else {
            /* Remainder: process full 7-groups, handle remaining 1-6 values */
            int full = n / 7;
            for (int i = 0; i < full * 7; i += 7)
                buf[n_out++] = maj7(buf[i], buf[i+1], buf[i+2],
                                    buf[i+3], buf[i+4], buf[i+5], buf[i+6]);
            int rem = n - full * 7;
            uint32_t a = buf[full * 7];
            if (rem == 1) {
                buf[n_out++] = a;  /* passthrough */
            } else if (rem == 2) {
                uint32_t b = buf[full * 7 + 1];
                buf[n_out++] = a & b;  /* AND: 2/2 = 100% ≥ 71% */
            } else if (rem == 3) {
                uint32_t b = buf[full * 7 + 1], c = buf[full * 7 + 2];
                buf[n_out++] = a & b & c;  /* AND: 3/3 = 100% ≥ 71% */
            } else if (rem == 4) {
                uint32_t b = buf[full * 7 + 1], c = buf[full * 7 + 2], d = buf[full * 7 + 3];
                /* 3 of 4 = 75% ≥ 71% */
                buf[n_out++] = (a&b&c) | (a&b&d) | (a&c&d) | (b&c&d);
            } else if (rem == 5) {
                uint32_t b = buf[full * 7 + 1], c = buf[full * 7 + 2],
                         d = buf[full * 7 + 3], e = buf[full * 7 + 4];
                /* 4 of 5 = 80% ≥ 71% */
                buf[n_out++] = (a&b&c&d) | (a&b&c&e) | (a&b&d&e)
                             | (a&c&d&e) | (b&c&d&e);
            } else { /* rem == 6 */
                uint32_t b = buf[full * 7 + 1], c = buf[full * 7 + 2],
                         d = buf[full * 7 + 3], e = buf[full * 7 + 4],
                         f = buf[full * 7 + 5];
                /* 5 of 6 = 83% ≥ 71%: C(6,5) = 6 terms */
                buf[n_out++] = (a&b&c&d&e) | (a&b&c&d&f) | (a&b&c&e&f)
                             | (a&b&d&e&f) | (a&c&d&e&f) | (b&c&d&e&f);
            }
        }
        n = n_out;
    }
    return buf[0];
}

#endif /* MAJ7_H */
