/*
 * lib/maj3.h — Majority-of-3 Bit-Logic (header only)
 * ===================================================
 *
 * Provides:
 *   maj3(a,b,c)              — Majority of 3 uint32 (bit-parallel)
 *   majority_tree3(vals, n)  — Majority tree over n values
 *
 * Fix 2026-06-17: Passthrough cascade at n = 3^k+1
 *   When n%3==1 at EVERY level, a single value without majority vote
 *   propagates through all levels → final level AND blocks result.
 *   Fix: n%3==1 → process n-4 as 3-groups + final 4-group.
 *
 * Usage:
 *   #include "maj3.h"
 *   uint32_t r = majority_tree3(data, 196);
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
/* PIXEL_TYPE, KI_PX_PER_CONT, KI_BIT_WIDTH, KI_BIT_POS defined in ki-common.h */

#ifndef MAJ3_BUF
#  define MAJ3_BUF  4096
#endif

/* PIXEL_TYPE-buffer size for majority_tree3_px.
 * MAJ3_BUF is in uint32_t units; convert to PIXEL_TYPE count.
 * PIXEL_TYPE is defined by ki-common.h (always included before this file). */
#ifndef MAJ3_BUF_PX
#  define MAJ3_BUF_PX (MAJ3_BUF * (int)(sizeof(uint32_t) / sizeof(PIXEL_TYPE)))
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
 * majority_tree3(vals, n) — Majority tree over n uint32 values.
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
static inline uint32_t majority_tree3(const uint32_t *vals, int n) {
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

/**
 * majority_tree3_step — Pixel-level step between majority triple members
 *
 * Standard majority_tree3 groups containers as (i, i+1, i+2). This variant
 * groups as (i, i+S, i+2S) where S = step in containers.
 * Only works when step >= 1 (container-level step).
 *
 * @param vals  match array (nc_local containers)
 * @param n     number of containers
 * @param step  container step (>= 1)
 */
static inline uint32_t majority_tree3_step(const uint32_t *vals, int n, int step) {
    if (step < 1) step = 1;
    uint32_t sub[MAJ3_BUF];
    int sn = 0;
    for (int i = 0; i < n && sn < MAJ3_BUF; i += step)
        sub[sn++] = vals[i];
    if (sn < 1) return 0;
    return majority_tree3(sub, sn);
}

/**
 * majority_tree3_px — Majority tree over PIXEL_TYPE values
 *
 * Same logic as majority_tree3 but operates on PIXEL_TYPE* instead of
 * uint32_t*. Groups of 3, with correct maj4/maj5 handling for remainders.
 * No packing into containers needed — each entry is one PIXEL_TYPE value.
 *
 * @param vals  array of n PIXEL_TYPE values (n > 0)
 * @param n     number of values
 * @return      bitwise majority over all n values (as PIXEL_TYPE)
 */
static inline PIXEL_TYPE majority_tree3_px(const PIXEL_TYPE *vals, int n) {
    if (n <= 0) return 0;
    if (n > MAJ3_BUF_PX) n = MAJ3_BUF_PX;
    PIXEL_TYPE buf[MAJ3_BUF_PX];
    memcpy(buf, vals, (size_t)n * sizeof(PIXEL_TYPE));

    while (n > 1) {
        int n_out = 0;
        if (n % 3 == 1 && n > 4) {
            int full = (n - 4) / 3;
            for (int i = 0; i < full * 3; i += 3) {
                PIXEL_TYPE a = buf[i], b = buf[i+1], c = buf[i+2];
                buf[n_out++] = (a & b) | (a & c) | (b & c);
            }
            int j = full * 3;
            PIXEL_TYPE a = buf[j], b = buf[j+1], c = buf[j+2], d = buf[j+3];
            buf[n_out++] = (a & b & c) | (a & b & d) | (a & c & d) | (b & c & d);
        } else if (n % 3 == 2 && n > 5) {
            int full = (n - 5) / 3;
            for (int i = 0; i < full * 3; i += 3) {
                PIXEL_TYPE a = buf[i], b = buf[i+1], c = buf[i+2];
                buf[n_out++] = (a & b) | (a & c) | (b & c);
            }
            int j = full * 3;
            PIXEL_TYPE a = buf[j], b = buf[j+1], c = buf[j+2], d = buf[j+3], e = buf[j+4];
            buf[n_out++] = (a & b & c) | (a & b & d) | (a & b & e)
                         | (a & c & d) | (a & c & e) | (a & d & e)
                         | (b & c & d) | (b & c & e) | (b & d & e)
                         | (c & d & e);
        } else {
            for (int i = 0; i < n; i += 3) {
                PIXEL_TYPE a = buf[i];
                if (i + 2 < n) {
                    PIXEL_TYPE b = buf[i+1], c = buf[i+2];
                    buf[n_out++] = (a & b) | (a & c) | (b & c);
                } else if (i + 1 < n) {
                    buf[n_out++] = a & buf[i+1];
                } else {
                    buf[n_out++] = a;
                }
            }
        }
        n = n_out;
    }
    return buf[0];
}

/**
 * majority_tree3_pixel_step — Pixel-level step majority
 *
 * Two paths depending on step size:
 *
 * Path 1 (step ≤ ppc): Uses the same container-level grouping as
 * majority_tree3 (3-container groups with MAJ4/MAJ5 for n%%3=1/2
 * remainders). Within each group, forms pixel triples with step=S
 * between members. For step = ppc, identical to majority_tree3.
 *
 * Path 2 (step > ppc): Per-slot approach without container grouping.
 * Triples at positions slot, slot+3S, slot+6S, … while i+2S < n_pixels.
 * Remaining pixels passed through. Correct for any step value.
 *
 * Each slot is reduced via majority_tree3_px and packed into uint32.
 *
 * @param vals    match array (n containers)
 * @param n       number of containers
 * @param px_step pixel step between triple members (>= 1)
 */
static inline uint32_t majority_tree3_pixel_step(const uint32_t *vals, int n,
                                                   int px_step) {
    if (px_step < 1) px_step = 1;
    int ppc = KI_PX_PER_CONT;
    int bpw = KI_BIT_WIDTH;

    const void *vptr = vals;
    const PIXEL_TYPE *px = (const PIXEL_TYPE *)vptr;
    int n_pixels = n * ppc;
    uint32_t result = 0;

    if (px_step > ppc) {
        /* ═══════════════════════════════════════════════════════════
         * PATH 2: step > ppc — flaches Packen, kein Slot-Konzept
         * ═══════════════════════════════════════════════════════════
         * Triples (px[i], px[i+S], px[i+2S]) werden der Reihe nach
         * in uint32_t-Container gepackt (ppc pro Container), dann
         * majority_tree3 reduziert sie. Der Output ist ein uint32_t
         * mit Tripel-Ergebnissen an beliebigen Slot-Positionen —
         * korrekt, weil jedes Bit unabhängig zählt. */
        int spacing = 3 * px_step;
        uint32_t pre[MAJ3_BUF];
        int pn = 0;
        uint32_t cont = 0;
        int slot = 0;
        for (int i = 0; i + 2 * px_step < n_pixels && pn < MAJ3_BUF;
             i += spacing) {
            PIXEL_TYPE v0 = px[i];
            PIXEL_TYPE v1 = px[i + px_step];
            PIXEL_TYPE v2 = px[i + 2 * px_step];
            PIXEL_TYPE pb = 0;
            for (int bit = 0; bit < bpw; bit++) {
                int cnt = ((v0>>bit)&1) + ((v1>>bit)&1) + ((v2>>bit)&1);
                if (cnt >= 2) pb |= (PIXEL_TYPE)(1u << bit);
            }
            cont |= (uint32_t)pb << (slot * bpw);
            slot++;
            if (slot == ppc) {
                pre[pn++] = cont;
                cont = 0;
                slot = 0;
            }
        }
        if (slot > 0) pre[pn++] = cont;
        if (pn < 1) return 0;
        return majority_tree3(pre, pn);
    } else {
        /* ═══════════════════════════════════════════════════════════
         * PATH 1: step ≤ ppc — container-level grouping
         * ═══════════════════════════════════════════════════════════ */
        int group3, rem = n % 3;
        if (rem == 1 && n > 4)
            group3 = (n - 4) / 3;
        else if (rem == 2 && n > 5)
            group3 = (n - 5) / 3;
        else
            group3 = n / 3;
        int group_px = 3 * ppc;

        for (int slot = 0; slot < ppc; slot++) {
            PIXEL_TYPE buf[MAJ3_BUF_PX];
            int pn = 0;

            /* Full 3-container groups as pixel triples (always within bounds) */
            for (int g = 0; g < group3 && pn < MAJ3_BUF_PX; g++) {
                int base = g * group_px;
                /* step ≤ ppc → slot+2*step ≤ slot+2*ppc < 3*ppc = group_px ✓ */
                PIXEL_TYPE v0 = px[base + slot];
                PIXEL_TYPE v1 = px[base + slot + px_step];
                PIXEL_TYPE v2 = px[base + slot + 2 * px_step];
                PIXEL_TYPE pb = 0;
                for (int bit = 0; bit < bpw; bit++) {
                    int cnt = ((v0>>bit)&1) + ((v1>>bit)&1) + ((v2>>bit)&1);
                    if (cnt >= 2) pb |= (PIXEL_TYPE)(1u << bit);
                }
                buf[pn++] = pb;
            }

            /* Remainder wie majority_tree3 */
            int last_cont = group3 * 3;
            int base_px = last_cont * ppc;

            if (rem == 1 && n > 4) {
                PIXEL_TYPE v0 = px[base_px + slot];
                PIXEL_TYPE v1 = px[base_px + slot + px_step];
                PIXEL_TYPE v2 = px[base_px + slot + 2 * px_step];
                PIXEL_TYPE v3 = px[base_px + slot + 3 * px_step];
                uint32_t pb = 0;
                for (int bit = 0; bit < bpw; bit++) {
                    uint32_t b0=(v0>>bit)&1,b1=(v1>>bit)&1,b2=(v2>>bit)&1,b3=(v3>>bit)&1;
                    if ((b0&b1&b2)|(b0&b1&b3)|(b0&b2&b3)|(b1&b2&b3)) pb |= (1u<<bit);
                }
                buf[pn++] = (PIXEL_TYPE)pb;
            } else if (rem == 2 && n > 5) {
                PIXEL_TYPE v0 = px[base_px + slot], v1 = px[base_px + slot + px_step];
                PIXEL_TYPE v2 = px[base_px + slot + 2*px_step];
                PIXEL_TYPE v3 = px[base_px + slot + 3*px_step];
                PIXEL_TYPE v4 = px[base_px + slot + 4*px_step];
                uint32_t pb = 0;
                for (int bit = 0; bit < bpw; bit++) {
                    int cnt = ((v0>>bit)&1)+((v1>>bit)&1)+((v2>>bit)&1)+((v3>>bit)&1)+((v4>>bit)&1);
                    if (cnt >= 3) pb |= (1u<<bit);
                }
                buf[pn++] = (PIXEL_TYPE)pb;
            } else if (rem == 1) {
                buf[pn++] = px[base_px + slot];
            } else if (rem == 2) {
                buf[pn++] = px[base_px + slot];
                buf[pn++] = px[base_px + ppc + slot];
            }

            if (pn > 0)
                result |= (uint32_t)majority_tree3_px(buf, pn) << (slot * bpw);
        }
    }
    return result;
}

#endif /* MAJ3_H */
