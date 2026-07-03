/*
 * lib/w0_random.h — Full 32-bit randomness via splitmix64
 * =======================================================
 *
 * PROBLEM:  Standard glibc `rand()` is a Linear Congruential Generator (LCG)
 *           with poor lower-bit distribution, low period (2³¹), and serial
 *           correlation. After fixing the 31-bit bug (Bit 31 always 0), the
 *           `((uint32_t)rand()<<16) ^ (uint32_t)rand()` workaround still uses
 *           `rand()` underneath — mediocre quality costs ~3% accuracy.
 *
 * FIX:      Replace `rand()` with **splitmix64** (Sebastiano Vigna, 2015).
 *           - Passes all BigCrush tests (strongest Diehard family)
 *           - 64-bit state, simple arithmetic (no lookup tables)
 *           - Period 2⁶⁴ (practical infinity for our use)
 *           - Public domain code, see https://prng.di.unimi.it/splitmix64.c
 *
 * USAGE:
 *   #include "w0_random.h"
 *
 *   w0_srandom(0xSEED);              // Seed ONCE before generating
 *   uint32_t val = w0_random();      // 32-bit random value
 *
 *   // For parallel init with OpenMP:
 *   #pragma omp parallel
 *   {
 *       w0_srandom(seed + omp_get_thread_num());   // Per-thread seed
 *       #pragma omp for
 *       for (...) W0[i] = w0_random();             // Thread-local state!
 *   }
 *
 * ATTENTION:
 *   - Each translation unit (.c file) has its OWN static state.
 *     This is intentional: no global lock contention.
 *   - Always call w0_srandom() BEFORE w0_random() in each .c file.
 *
 * See also: plans/plan-2026-06-17-otto-score.md (W0-Bugfix section)
 *           docs/otto.md (Wichtige Erkenntnisse)
 */

#ifndef W0_RANDOM_H
#define W0_RANDOM_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * Internal splitmix64 state (one per translation unit).
 * Seeded via w0_srandom(), advanced by w0_random().
 */
static uint64_t w0_splitmix64_state;

/**
 * Optional random file handle.  If set, w0_random() reads from this
 * file instead of splitmix64.  The file cursor advances sequentially;
 * w0_srandom() is a no-op when a file is active.
 */
static FILE *w0_rf = NULL;

/**
 * w0_rand_set_file() — Switch w0_random() to read from a binary file.
 *
 * Each call to w0_random() reads the next uint32 from the file.
 * The file must contain at least as many uint32 as w0_random() calls.
 * Pass NULL or empty string to revert to splitmix64 PRNG.
 */
static inline void w0_rand_set_file(const char *path) {
    if (w0_rf) { fclose(w0_rf); w0_rf = NULL; }
    if (path && path[0]) {
        w0_rf = fopen(path, "rb");
        if (!w0_rf) {
            fprintf(stderr, "[FATAL] Cannot open random file: %s\n", path);
            exit(1);
        }
    }
}

/**
 * w0_srandom() — Seed the splitmix64 generator (or no-op in file mode).
 *
 * In PRNG mode: sets the splitmix64 state, call ONCE before w0_random().
 * In file mode: has no effect (file cursor is independent).
 * Any 64-bit value is valid (including 0 — splitmix64 handles it).
 */
static inline void w0_srandom(uint64_t seed) {
    w0_splitmix64_state = seed;
}

/**
 * w0_random() — Generate a 32-bit random value.
 *
 * FILE MODE (after w0_rand_set_file()):
 *   Reads next uint32 from the file.  Fast, no computation.
 *
 * PRNG MODE (default):
 *   Returns upper 32 bits of splitmix64 — passes BigCrush.
 *
 * @return uint32_t — 32-bit random value (0x00000000 .. 0xFFFFFFFF)
 */
static inline uint32_t w0_random(void) {
    if (w0_rf) {
        uint32_t v;
        if (fread(&v, sizeof(v), 1, w0_rf) == 1) return v;
        fprintf(stderr, "[FATAL] Random file EOF (need more uint32)\n");
        exit(1);
    }
    uint64_t z = (w0_splitmix64_state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return (uint32_t)((z ^ (z >> 31)) >> 32);
}

#endif /* W0_RANDOM_H */
