/*
 * lib/ki-train.h — Shared training loop (batch correct) for Otto + Bitvoting
 * ==========================================================================
 * 
 * Instead of duplicated code in seq.c and bitvoting.c: both call
 * ki_batch_correct() which works identically.
 *
 * For Otto:    gb_all = gb_buf (precomputed VN groups), H = hidden neurons
 * For Bitvote: gb_all = input_buf (encoded pixel containers), H = n_cont
 *
 * Both share the same target layout: [h × V × K] (h=H, V=32, K=10)
 * mit TGT_IDX(k, h, v, H, V) = h*V*K + v*K + k
 */
#ifndef KI_TRAIN_H
#define KI_TRAIN_H

#include "ki-common.h"

/* ── Target-Index: [H][V][KI_NCLASSES] — layout matching TGT_IDX in Otto ── */
#ifndef TGT_IDX
#define TGT_IDX(k, h, v, H, V) \
    ((size_t)(h) * (size_t)(V) * KI_NCLASSES + (size_t)(v) * KI_NCLASSES + (size_t)(k))
#endif

/* ── SCORE: like VN_SCORE_FROM_GB (reads from target, writes to sc) ── */
#define KT_SCORE(gb, h, H, NG, TGT, SC) do { \
    uint32_t _b = (gb); \
    while (_b) { int _v = __builtin_ctz(_b); \
        for (int _k = 0; _k < KI_NCLASSES; _k++) \
            (SC)[_k] += (TGT)[TGT_IDX(_k, (h), _v, H, NG)]; \
        _b &= _b - 1; } \
} while (0)

#define KT_CORRECT(gb, h, H, NG, DC, TK, PK, SI, BV) do { \
    uint32_t _b = (gb); \
    while (_b) { int _v = __builtin_ctz(_b); \
        size_t _ci = TGT_IDX((TK), (h), _v, H, NG); \
        (DC)[_ci].val += (SI); \
        if ((DC)[_ci].ver != BV) { (DC)[_ci].ver = BV; } \
        _ci = TGT_IDX((PK), (h), _v, H, NG); \
        (DC)[_ci].val -= (SI); \
        if ((DC)[_ci].ver != BV) { (DC)[_ci].ver = BV; } \
        _b &= _b - 1; } \
} while (0)

/* ═══════════════════════════════════════════════════════════════════════
 * ki_batch_correct — Shared Training Loop (Original Otto Logic)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * target:     int32[H × KI_NCLASSES × V] — learnable weights
 * H:          number of uint32 per sample (neurons for Otto, containers for Bitvote)
 * gb_all:     uint32 data per sample [N × H]
 * y:          labels
 * N:          number of samples
 * step:       base step size
 * tgt_sz:     size of target array (H × KI_NCLASSES × V)
 * filter_mask: class filter (0 = no filter)
 * stride:     distance between samples in gb_all (default: H)
 * no_gap:     1 = skip gap scaling (Bitvoting), 0 = original Otto behavior
 */
static inline int ki_batch_correct(int32_t *target, int H,
                                    const int64_t *class_offset,
                                    const uint32_t *gb_all,
                                    const uint8_t *y,
                                    int N, int step, size_t tgt_sz,
                                    int filter_mask,
                                    int stride, int no_gap) {
    if (stride < 1) stride = H;
    int n_threads = aa.threadN;
    if (n_threads < 1) n_threads = 1;
    int batch = aa.batchN > 0 ? aa.batchN : N;
    int corrections = 0;

    /* Thread-local accumulators (int32_t like original, NO OT_Entry!) */
    int32_t **dc = (int32_t **)malloc((size_t)n_threads * sizeof(int32_t *));
    for (int t = 0; t < n_threads; t++)
        dc[t] = (int32_t *)calloc(tgt_sz, sizeof(int32_t));

    for (int b_start = 0; b_start < N; b_start += batch) {
        int b_end = b_start + batch;
        if (b_end > N) b_end = N;
        int batch_corr = 0;
        int64_t sc[KI_NCLASSES];

        #pragma omp parallel for reduction(+:batch_corr) firstprivate(sc) schedule(static)
        for (int s = b_start; s < b_end; s++) {
            if (filter_mask && !((filter_mask >> (int)y[s]) & 1))
                continue;
            int tid = omp_get_thread_num();
            int true_k = (int)y[s];
            const uint32_t *gb_s = gb_all + (size_t)s * (size_t)stride;
            for (int k = 0; k < KI_NCLASSES; k++) sc[k] = class_offset[k];

            for (int h = 0; h < H; h++)
                KT_SCORE(gb_s[h], h, H, 32, target, sc);

            int pred = 0;
            for (int k = 1; k < KI_NCLASSES; k++)
                if (sc[k] > sc[pred]) pred = k;

            if (pred != true_k) {
                int64_t gap = sc[pred] - sc[true_k];
                if (no_gap) {
                    batch_corr++;
                    for (int h = 0; h < H; h++) {
                        uint32_t _b = gb_s[h];
                        while (_b) {
                            int _v = __builtin_ctz(_b);
                            dc[tid][TGT_IDX(true_k, h, _v, H, 32)] += step;
                            dc[tid][TGT_IDX(pred, h, _v, H, 32)] -= step;
                            _b &= _b - 1;
                        }
                    }
                } else if (gap > 0) {
                    int step_i = (gap < (int64_t)OT_F)
                        ? (int)(((int64_t)step * gap) >> OT_PRECISION)
                        : step;
                    if (step_i < 1) step_i = 1;
                    batch_corr++;
                    for (int h = 0; h < H; h++) {
                        uint32_t _b = gb_s[h];
                        while (_b) {
                            int _v = __builtin_ctz(_b);
                            dc[tid][TGT_IDX(true_k, h, _v, H, 32)] += step_i;
                            dc[tid][TGT_IDX(pred, h, _v, H, 32)] -= step_i;
                            _b &= _b - 1;
                        }
                    }
                }
            }
        }

        /* Apply: serial like original (only check != 0) */
        for (int t = 0; t < n_threads; t++) {
            int32_t *ct = dc[t];
            for (size_t i = 0; i < tgt_sz; i++) {
                int d = ct[i];
                if (d != 0) target[i] += d;
            }
            memset(ct, 0, tgt_sz * sizeof(int32_t));
        }
        corrections += batch_corr;
    }

    for (int t = 0; t < n_threads; t++) free(dc[t]);
    free(dc);
    return corrections;
}

#endif /* KI_TRAIN_H */