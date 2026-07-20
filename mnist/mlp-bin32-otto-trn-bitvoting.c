/*
 * mnist-1/mlp-bin32-otto-trn-bitvoting.c — Direct Pixel-Bit → Class Vote
 * ===============================================================
 *
 * Instead of W0 matrix + majority tree: every input bit votes directly on
 * the 10 classes. Kein H0, kein W0, kein majority_tree.
 *
 * Training:  for each set bit b in sample: target[b][label] += step
 *            for each set bit b in sample: target[b][pred]   -= step
 * Eval:      for each set bit b in sample: score[k] += target[b][k]
 *            argmax(score) = prediction
 *
 * Architecture: Members = active xforms, jeder Member hat eigenes Target.
 * Voting summiert Scores aller Member → argmax.
 */
#include "ki-load.h"           /* encoding-aware load_input */
#include "../lib/ki-encoding.h"
#include "../lib/ki-train.h"   /* shared batch_correct (ki_batch_correct) */
#include <inttypes.h>
#include <sys/stat.h>
#include <errno.h>

/* ── Global args ── */
ki_Args aa = {
    .hidden             = 64,
    .epochs             = 1,
    .batchN             = KI_DEFAULT_BATCH_N,
    .trainN             = 0,       /* auto */
    .evalN              = 0,       /* auto */
    .seed               = 42,
    .lr                 = KI_DEFAULT_LR,
    .threadN            = 8,
    .warmup_epochs      = 0,
    .step_power         = KI_DEFAULT_STEP_POWER,
    .step_mode          = KI_DEFAULT_STEP_MODE,
    .xforms             = (1 << KI_XFORM_ID),
    .ensembleN          = 1,
    .seed_splitmix      = 1,
    .enc_size           = KI_ENC_WIDTH_DEFAULT,
    .enc_default_width  = KI_ENC_WIDTH_DEFAULT,
    .channel            = KI_DEFAULT_COLOR,
    .splitVN            = 1,
    .splitHN            = 1,
};

/* ── Bit-Voting Model: pro Member (xform) ein Target ── */
/* load_input kommt aus ki-load.h (encoding-aware, shared mit Otto) */
#define BITVOTE_TOTAL_BITS (KI_NC_TOTAL * 32)  /* Container × 32 Bit */

typedef struct {
    int32_t *target;     /* [total_bits × KI_NCLASSES] — weights for each bit */
    int64_t  offset[KI_NCLASSES];  /* Klassen-Bias (Logit-Prior) */
    int      total_bits;
    float    trn_acc;    /* --member-threshold: training accuracy (0..100) */
    /* Debug-Info (wie Otto: encoding=xf) */
    int      enc_idx;    /* Index in aa.enc_array */
    int      xf_id;      /* xform ID (KI_XFORM_ID, ...) */
} BitVoteMember;

static BitVoteMember *bitvote_member_create(int total_bits) {
    BitVoteMember *m = (BitVoteMember *)calloc(1, sizeof(BitVoteMember));
    if (!m) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
    m->total_bits = total_bits;
    size_t sz = (size_t)total_bits * KI_NCLASSES;
    m->target = (int32_t *)calloc(sz, sizeof(int32_t));
    if (!m->target) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
    m->trn_acc = 100.0f;  /* initially: all members participate */
    return m;
}

static void bitvote_member_destroy(BitVoteMember *m) {
    if (m) { free(m->target); free(m); }
}

/* ── Datengetriebene Logit-Initialisierung ──
 * Wie Otto: target[b][k] = log(P(class=k | bit=1) / P(class=k))
 *          = log(P(bit=1|class=k)) - log(P(bit=1))
 * With Laplace smoothing over all classes.
 *
 * offset[k] = log(P(class=k)) × OT_F
 * target[b][k] = log(P(bit=1|class=k) / P(bit=1)) × OT_F
 */
static void bitvote_init_logit(BitVoteMember *m, const uint32_t *X,
                                const uint8_t *y, int N, int stride) {
    int total_bits = m->total_bits;
    int n_cont = total_bits / 32;  /* this member's containers */
    int klass_count[KI_NCLASSES] = {0};
    int64_t *bit_klass = (int64_t *)calloc((size_t)total_bits * KI_NCLASSES, sizeof(int64_t));
    if (!bit_klass) { fprintf(stderr, "[FATAL] bitvote_init OOM\n"); exit(1); }

    /* Phase 1: Count set bits per (bit, class) — only this member's containers */
    for (int s = 0; s < N; s++) {
        int label = (int)y[s];
        klass_count[label]++;
        const uint32_t *in = X + (size_t)s * (size_t)stride;
        for (int c = 0; c < n_cont; c++) {
            uint32_t v = in[c];
            while (v) {
                int b = __builtin_ctz(v);
                size_t idx = (size_t)(c * 32 + b) * KI_NCLASSES + (size_t)label;
                bit_klass[idx]++;
                v &= v - 1;
            }
        }
    }

    /* Phase 2: Likelihood-Ratio target[b][k] = log(P(k|b) / P(k)) */
    double total = (double)N;
    for (int k = 0; k < KI_NCLASSES; k++)
        m->offset[k] = (int64_t)(log((double)(klass_count[k] + 1) / (total + (double)KI_NCLASSES)) * (double)OT_F);

    /* Marginal bit probability P(bit=1) across all classes */
    for (int b = 0; b < total_bits; b++) {
        double bit_total = 0;
        for (int k = 0; k < KI_NCLASSES; k++)
            bit_total += (double)bit_klass[(size_t)b * KI_NCLASSES + (size_t)k];
        double p_bit = (bit_total + 1.0) / (total + (double)KI_NCLASSES);  /* Laplace */

        for (int k = 0; k < KI_NCLASSES; k++) {
            double nk = (double)klass_count[k];
            double cnt = (double)bit_klass[(size_t)b * KI_NCLASSES + (size_t)k];
            double p_k_given_bit = (cnt + 1.0) / (nk + 2.0);  /* P(bit=1 | class=k), Laplace */
            double ratio = p_k_given_bit / p_bit;              /* P(k|bit) / P(k) = P(bit|k) / P(bit) */
            double logit = log(ratio);
            m->target[(size_t)b * KI_NCLASSES + (size_t)k] = (int32_t)(logit * (double)OT_F);
        }
    }

    free(bit_klass);
}

/* ═══════════════════════════════════════════════════════════════
 * FORWARD: Compute scores from one member
 * ═══════════════════════════════════════════════════════════════ */
static void bitvote_forward(const BitVoteMember *m, const uint32_t *in,
                             int64_t *sc, int x_stride) {
    int n_cont = m->total_bits / 32;  /* containers for THIS member's encoding */
    for (int k = 0; k < KI_NCLASSES; k++) sc[k] = m->offset[k];
    for (int c = 0; c < n_cont; c++) {
        uint32_t v = in[c];
        while (v) {
            int b = __builtin_ctz(v);
            size_t base = (size_t)(c * 32 + b) * KI_NCLASSES;
            for (int k = 0; k < KI_NCLASSES; k++)
                sc[k] += m->target[base + (size_t)k];
            v &= v - 1;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * TRAINING: one epoch for one member
 * ═══════════════════════════════════════════════════════════════ */
static int bitvote_train_member(BitVoteMember *m, const uint32_t *X,
                                 const uint8_t *y, int N, int step, int x_stride) {
    /* Same code as Otto: ki_batch_correct from ki-train.h
     * Input (gb_all) = X, H = containers per encoding, target = m->target
     * stride = x_stride (distance between samples in X) */
    return ki_batch_correct(m->target, m->total_bits / 32,
                            m->offset, X, y, N, step,
                            (size_t)m->total_bits * KI_NCLASSES, aa.filter_mask,
                            x_stride, 0);  /* no_gap=0: Bayesian Gap-Scaling */
}

/* ═══════════════════════════════════════════════════════════════
 * EVALUATION: Alle Member voten, Scores summiert
 * ═══════════════════════════════════════════════════════════════ */
static int bitvote_eval(BitVoteMember **members, int n_members,
                         const uint32_t **x_buf, int stride,
                         const uint8_t *labels, int n_samp, uint8_t *pred_out) {
    int ok = 0;
    int64_t (*votes)[KI_NCLASSES] = (int64_t (*)[KI_NCLASSES])calloc(
        (size_t)n_samp, sizeof(int64_t[KI_NCLASSES]));
    if (!votes) { fprintf(stderr, "[FATAL] eval OOM\n"); exit(1); }

    int threshold = aa.member_threshold;
    for (int mi = 0; mi < n_members; mi++) {
        if (threshold > 0 && members[mi]->trn_acc < (float)threshold)
            continue;

        #pragma omp parallel for schedule(static)
        for (int s = 0; s < n_samp; s++) {
            int64_t sc[KI_NCLASSES];
            bitvote_forward(members[mi], x_buf[mi] + (size_t)s * (size_t)stride, sc, stride);
            for (int k = 0; k < KI_NCLASSES; k++)
                votes[s][k] += sc[k];
        }
    }

    for (int s = 0; s < n_samp; s++) {
        int pred = 0;
        for (int k = 1; k < KI_NCLASSES; k++)
            if (votes[s][k] > votes[s][pred]) pred = k;
        if (pred_out) pred_out[s] = (uint8_t)pred;
        if (pred == (int)labels[s]) ok++;
    }
    free(votes);
    return ok;
}

/* ═══════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════ */
int debug_epoch = 0;

int main(int argc, char *argv[]) {
    /* Filter out --debug-epoch before ki_parse_args (ki-common.h doesn't know it) */
    const char **av = (const char **)malloc((size_t)(argc + 1) * sizeof(char *));
    int ac = 0;
    av[ac++] = argv[0];
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--debug-epoch") == 0) { debug_epoch = 1; }
        else { av[ac++] = argv[i]; }
    }
    av[ac] = NULL;
    ki_parse_args(ac, (char **)av);
    free(av);
    omp_set_num_threads(aa.threadN);

    /* ── Load dataset (struct MUSS genullt sein, sonst dry_run=random!) ── */
    ki_Dataset data;
    memset(&data, 0, sizeof(data));
    if (ki_dataset_read(&data) != 0) return 1;
    if (aa.dry_run) { ki_dataset_free(&data); return 1; }

    int total_all = data.num_images;
    int total_train = (aa.trainN > 0) ? aa.trainN : data.n_train;
    int total_eval  = (aa.evalN  > 0) ? aa.evalN  : data.n_eval;
    if (total_train + total_eval > total_all)
        { total_train = total_all / 2; total_eval = total_all - total_train; }

    int epochs = aa.epochs;
    int step_init = (aa.lr > 0) ? (int)(aa.lr * (double)OT_F + 0.5) : 128;
    if (step_init < 1) step_init = 1;

    /* ── Encoding LUT ── */
    for (int _ei = 0; _ei < aa.enc_count && _ei < KI_ENC_MAX; _ei++)
        enc_lut_init_enc((int)aa.enc_array[_ei].type, (int)aa.enc_array[_ei].width);
    if (aa.enc_count == 0) {
        int def_enc = aa.debug_binarize ? KI_ENC_LIN7 : KI_ENC_RAW;
        enc_lut_init_enc(def_enc, KI_ENC_WIDTH_DEFAULT);
    }

    /* ── Determine encodings + xforms ── */
    int n_enc = aa.enc_count > 0 ? aa.enc_count : 1;
    int xf_id_list[KI_XFORM_COUNT], n_xforms = 0;
    for (int xf = 0; xf < KI_XFORM_COUNT; xf++)
        if (aa.xforms & (1 << xf)) xf_id_list[n_xforms++] = xf;
    if (n_xforms < 1) { xf_id_list[0] = KI_XFORM_ID; n_xforms = 1; }

    /* Container-Offset und -Stride pro Encoding (wie in load_input) */
    int enc_off[KI_ENC_MAX], enc_nc[KI_ENC_MAX], enc_stride_total = 0;
    for (int ei = 0; ei < n_enc && ei < KI_ENC_MAX; ei++) {
        int w = (ei < aa.enc_count) ? (int)aa.enc_array[ei].width : KI_ENC_WIDTH_DEFAULT;
        if (w < 1) w = KI_ENC_WIDTH_DEFAULT;
        enc_nc[ei] = KI_NC * w / 8;
        enc_off[ei] = enc_stride_total;
        enc_stride_total += enc_nc[ei];
    }

    int n_members = n_enc * n_xforms;

    /* ── Load input buffers (one per xform, cached) — BEFORE header print ── */
    size_t cache_stride = (size_t)enc_stride_total;
    uint32_t *X_xform[KI_XFORM_COUNT];
    for (int xi = 0; xi < n_xforms; xi++) {
        int xf = xf_id_list[xi];
        if (xf == KI_XFORM_ID) {
            X_xform[xi] = load_input_cached(data.X_raw, total_all, cache_stride);
        } else {
            uint8_t *xf_raw = (uint8_t *)malloc((size_t)total_all * (size_t)KI_PX);
            for (int s = 0; s < total_all; s++) {
                const uint8_t *src = data.X_raw + (size_t)s * (size_t)KI_PX;
                uint8_t *dst = xf_raw + (size_t)s * (size_t)KI_PX;
                ki_xform_raw(dst, src, data.cols, data.rows, KI_COLORS, xf);
            }
            X_xform[xi] = load_input_cached_xform(xf, xf_raw, total_all, cache_stride);
            free(xf_raw);
        }
    }

    /* ── Setup display ── */
    printf("══════════════════════════════════════════════════════════════════════\n");
    printf("══╡ BIT-VOTING ╞══  %s\n", KI_DATASET_NAME);
    int _raw_cont = KI_PX * KI_ENC_WIDTH_DEFAULT / 32;
    printf("  Input:       %d px → %d containers × 32 bits  (stride: %d)\n",
           KI_PX, _raw_cont, enc_stride_total);
    printf("  Classes:     %d\n", KI_NCLASSES);
    printf("  Members:     %d (%d enc × %d xform)\n", n_members, n_enc, n_xforms);
    printf("  Samples:     %d train / %d eval\n", total_train, total_eval);
    printf("  Training:    Perceptron-Update (+correct, -predicted)\n");
    printf("  Epochs:      %d  step=%d  lr=%.4f\n", epochs, step_init, (double)aa.lr);
    if (n_xforms > 1) {
        printf("  Xforms:      ");
        for (int xi = 0; xi < n_xforms; xi++) {
            if (xi > 0) printf(", ");
            printf("%s", ki_xform_name(xf_id_list[xi]));
        }
        printf("\n");
    }
    if (n_enc > 1 || aa.enc_count > 0) {
        printf("  Encodings:   ");
        for (int ei = 0; ei < n_enc; ei++) {
            if (ei > 0) printf(", ");
            int t = (ei < aa.enc_count) ? (int)aa.enc_array[ei].type : KI_ENC_RAW;
            int w = (ei < aa.enc_count) ? (int)aa.enc_array[ei].width : KI_ENC_WIDTH_DEFAULT;
            if (w < 1) w = KI_ENC_WIDTH_DEFAULT;
            const char *en = ki_enc_name_short(t);
            printf("%s%d(%d cont)", en, w, enc_nc[ei]);
        }
        printf("\n");
    }
    printf("\n");

    /* ── Create members: one per (encoding, xform) ── */
    BitVoteMember **members = (BitVoteMember **)malloc(
        (size_t)n_members * sizeof(BitVoteMember *));
    const uint32_t **X_tr = (const uint32_t **)malloc(
        (size_t)n_members * sizeof(uint32_t *));
    const uint32_t **X_te = (const uint32_t **)malloc(
        (size_t)n_members * sizeof(uint32_t *));

    for (int ei = 0; ei < n_enc; ei++) {
        int mem_stride = enc_nc[ei];
        int mem_bits = mem_stride * 32;
        for (int xi = 0; xi < n_xforms; xi++) {
            int m = ei * n_xforms + xi;
            members[m] = bitvote_member_create(mem_bits);
            members[m]->enc_idx = ei;
            members[m]->xf_id  = xf_id_list[xi];
            /* X pointer: first set, then init with data */
            X_tr[m] = X_xform[xi] + enc_off[ei];
            X_te[m] = X_xform[xi] + enc_off[ei]
                    + (size_t)total_train * (size_t)enc_stride_total;
            bitvote_init_logit(members[m], X_tr[m], data.y, total_train, enc_stride_total);
        }
    }

    /* ── Training: Member-aussen, Epoche-innen (wie Otto seq) ── */
    int best_ok = 0, best_trn = 0;
    int64_t (*acc_votes_tr)[KI_NCLASSES] = (int64_t (*)[KI_NCLASSES])calloc(
        (size_t)total_train, sizeof(int64_t[KI_NCLASSES]));
    int64_t (*acc_votes_te)[KI_NCLASSES] = total_eval > 0
        ? (int64_t (*)[KI_NCLASSES])calloc((size_t)total_eval, sizeof(int64_t[KI_NCLASSES]))
        : NULL;
    struct timeval tv0, tv1;
    gettimeofday(&tv0, NULL);

    /* ── TRAINING Header (wie Otto) ── */
    {
        const char *_mode_str = "cos-time";
        if (aa.step_mode == STEP_CONST) _mode_str = "const";
        else if (aa.step_mode == STEP_POW) _mode_str = "pow";
        printf("══╡ TRAINING ╞══  lr=%.4f  step=%d  mode=%s  F=%d",
               (double)aa.lr, step_init, _mode_str, OT_F);
        if (aa.warmup_epochs > 0)
            printf("  warmup=%d", aa.warmup_epochs);
        printf("  batch=%d", aa.batchN);
        /* Xforms */
        if (n_xforms > 1) {
            printf("  xform=");
            for (int xi = 0; xi < n_xforms; xi++) {
                if (xi > 0) printf(",");
                printf("%s", ki_xform_name(xf_id_list[xi]));
            }
        }
        printf("\n");
        fflush(stdout);
    }

    for (int m = 0; m < n_members; m++) {
        BitVoteMember *mem = members[m];

        /* ── All epochs for this member ── */
        for (int ep = 0; ep < epochs; ep++) {
            int step;
            if (aa.warmup_epochs > 0 && ep < aa.warmup_epochs) {
                float scale = (float)(ep + 1) / (float)aa.warmup_epochs;
                step = (int)((float)step_init * scale + 0.5f);
            } else if (aa.step_mode == STEP_COS_TIME) {
                int decay_ep = ep - aa.warmup_epochs;
                int decay_total = epochs - aa.warmup_epochs;
                if (decay_total < 1) decay_total = 1;
                float progress = (float)decay_ep / (float)decay_total;
                if (progress > 1.0f) progress = 1.0f;
                float cosine = (1.0f + cosf(progress * (float)3.14159265f)) / 2.0f;
                float lr_min_f = (aa.lr_min > 0.0f) ? aa.lr_min : 0.0f;
                step = (int)((float)step_init * (lr_min_f + (1.0f - lr_min_f) * cosine) + 0.5f);
            } else {
                step = step_init;
            }
            if (step < 1) step = 1;
            int _ep_err = bitvote_train_member(mem, X_tr[m], data.y, total_train, step, enc_stride_total);

            /* ── debug-epoch: pro Epoche anzeigen ── */
            if (debug_epoch) {
                int _ep_trn = total_train - _ep_err;
                int _ep_evl = 0;
                if (total_eval > 0) {
                    int64_t _sc[KI_NCLASSES];
                    #pragma omp parallel for firstprivate(_sc) reduction(+:_ep_evl) schedule(static)
                    for (int s = 0; s < total_eval; s++) {
                        bitvote_forward(mem, X_te[m] + (size_t)s * (size_t)enc_stride_total, _sc, enc_stride_total);
                        int _pred = 0;
                        for (int k = 1; k < KI_NCLASSES; k++)
                            if (_sc[k] > _sc[_pred]) _pred = k;
                        if (_pred == (int)data.y[total_train + s]) _ep_evl++;
                    }
                }
                printf("      [ep%3d/%d] trn=%5.1f%%  evl=%5.1f%%  err=%d  step=%d\n",
                       ep + 1, epochs,
                       (float)_ep_trn * 100.0f / (float)total_train,
                       (float)_ep_evl * 100.0f / (float)(total_eval > 0 ? total_eval : 1),
                       _ep_err, step);
                fflush(stdout);
            }
        }
        members[m]->trn_acc = 100.0f;  /* will be overwritten by eval */

        /* ── Add member votes to accumulator ── */
        #pragma omp parallel for schedule(static)
        for (int s = 0; s < total_train; s++) {
            int64_t sc[KI_NCLASSES];
            bitvote_forward(mem, X_tr[m] + (size_t)s * (size_t)enc_stride_total, sc, enc_stride_total);
            for (int k = 0; k < KI_NCLASSES; k++)
                acc_votes_tr[s][k] += sc[k];
        }
        if (acc_votes_te) {
            #pragma omp parallel for schedule(static)
            for (int s = 0; s < total_eval; s++) {
                int64_t sc[KI_NCLASSES];
                bitvote_forward(mem, X_te[m] + (size_t)s * (size_t)enc_stride_total, sc, enc_stride_total);
                for (int k = 0; k < KI_NCLASSES; k++)
                    acc_votes_te[s][k] += sc[k];
            }
        }

        /* ── Cumulative Accuracy ── */
        int _trn_ok = 0, _evl_ok = 0;
        for (int s = 0; s < total_train; s++) {
            int pred = 0;
            for (int k = 1; k < KI_NCLASSES; k++)
                if (acc_votes_tr[s][k] > acc_votes_tr[s][pred]) pred = k;
            if (pred == (int)data.y[s]) _trn_ok++;
        }
        for (int s = 0; s < total_eval; s++) {
            int pred = 0;
            for (int k = 1; k < KI_NCLASSES; k++)
                if (acc_votes_te[s][k] > acc_votes_te[s][pred]) pred = k;
            if (pred == (int)data.y[total_train + s]) _evl_ok++;
        }

        /* ── Per-Member accuracy (train + eval, wie Otto) ── */
        int _mem_trn = 0, _mem_evl = 0;
        #pragma omp parallel for reduction(+:_mem_trn) schedule(static)
        for (int s = 0; s < total_train; s++) {
            int64_t sc[KI_NCLASSES];
            bitvote_forward(mem, X_tr[m] + (size_t)s * (size_t)enc_stride_total, sc, enc_stride_total);
            int pred = 0;
            for (int k = 1; k < KI_NCLASSES; k++)
                if (sc[k] > sc[pred]) pred = k;
            if (pred == (int)data.y[s]) _mem_trn++;
        }
        if (total_eval > 0) {
            #pragma omp parallel for reduction(+:_mem_evl) schedule(static)
            for (int s = 0; s < total_eval; s++) {
                int64_t sc[KI_NCLASSES];
                bitvote_forward(mem, X_te[m] + (size_t)s * (size_t)enc_stride_total, sc, enc_stride_total);
                int pred = 0;
                for (int k = 1; k < KI_NCLASSES; k++)
                    if (sc[k] > sc[pred]) pred = k;
                if (pred == (int)data.y[total_train + s]) _mem_evl++;
            }
        }
        members[m]->trn_acc = (float)_mem_trn * 100.0f / (float)(total_train > 0 ? total_train : 1);

            if (_evl_ok > best_ok) { best_ok = _evl_ok; best_trn = _trn_ok; }

        /* ── Report interval: show aggregations if many members ── */
        int _report_int;
        if (n_members <= epochs || debug_epoch || aa.debug_member) {
            _report_int = 1;  /* show every member */
        } else if (epochs > 0) {
            _report_int = (n_members + epochs - 1) / epochs;  /* ~epochs lines */
        } else {
            _report_int = 1;
        }
        int _last_mb = (m == n_members - 1);
        if (_report_int > 0 && ((m + 1) % _report_int == 0 || _last_mb)) {
            gettimeofday(&tv1, NULL);
            int el = (int)((tv1.tv_sec - tv0.tv_sec) * 1000 +
                           (tv1.tv_usec - tv0.tv_usec) / 1000);
            int _filtered = (aa.member_threshold > 0 && members[m]->trn_acc < (float)aa.member_threshold);
            if (aa.debug_member || debug_epoch || (_report_int == 1 && n_members > 1)) {
                /* Verbose format with encoding/xform info */
                const char *_en = "?";
                int _w = KI_ENC_WIDTH_DEFAULT;
                if (mem->enc_idx >= 0 && mem->enc_idx < aa.enc_count) {
                    _en = ki_enc_name_short((int)aa.enc_array[mem->enc_idx].type);
                    _w  = (int)aa.enc_array[mem->enc_idx].width;
                }
                if (_w < 1) _w = KI_ENC_WIDTH_DEFAULT;
                const char *_xn = ki_xform_name(mem->xf_id);
                printf("  [%3d/%d] ens=%5.1f%%/%5.1f%%  mem=%5.1f%%/%5.1f%%  err=%d  time=%dms  %s=%s%d xf=%s%s\n",
                       m + 1, n_members,
                       (float)_trn_ok * 100.0f / (float)total_train,
                       (float)_evl_ok * 100.0f / (float)(total_eval > 0 ? total_eval : 1),
                       (float)_mem_trn * 100.0f / (float)total_train,
                       (float)_mem_evl * 100.0f / (float)(total_eval > 0 ? total_eval : 1),
                       total_train - _trn_ok, el,
                       KI_DATASET_NAME, _en, _w, _xn,
                       _filtered ? "  S" : "");
            } else {
                printf("  [%3d/%d] trn=%5.1f%%  evl=%5.1f%%  err=%d  time=%dms\n",
                       m + 1, n_members,
                       (float)_trn_ok * 100.0f / (float)total_train,
                       (float)_evl_ok * 100.0f / (float)(total_eval > 0 ? total_eval : 1),
                       total_train - _trn_ok, el);
            }
            fflush(stdout);
        }
    }

    /* Finale Accuracy aus Accumulator */
    int evl_ok = 0, trn_ok = 0;
    for (int s = 0; s < total_train; s++) {
        int pred = 0;
        for (int k = 1; k < KI_NCLASSES; k++)
            if (acc_votes_tr[s][k] > acc_votes_tr[s][pred]) pred = k;
        if (pred == (int)data.y[s]) trn_ok++;
    }
    for (int s = 0; s < total_eval; s++) {
        int pred = 0;
        for (int k = 1; k < KI_NCLASSES; k++)
            if (acc_votes_te[s][k] > acc_votes_te[s][pred]) pred = k;
        if (pred == (int)data.y[total_train + s]) evl_ok++;
    }

    gettimeofday(&tv1, NULL);
    int elapsed_ms = (int)((tv1.tv_sec - tv0.tv_sec) * 1000 +
                           (tv1.tv_usec - tv0.tv_usec) / 1000);
    /* REPORT: Train+Eval from BEST epoch (consistent!) */
    int report_trn = (best_ok > 0) ? best_trn : trn_ok;
    int report_evl = (best_ok > 0) ? best_ok : evl_ok;
    float fin_trn = (float)trn_ok * 100.0f / (float)total_train;
    float fin_evl = (float)evl_ok * 100.0f / (float)total_eval;
    float best_evl = (float)best_ok * 100.0f / (float)total_eval;

    printf("\n══╡ DONE ╞══════════════════════════════════════════════════\n");
    printf("  Members=%d  ep=%d  trn=%.1f%%  evl=%.1f%%  best=%.1f%%(trn=%.1f%%)  lr=%.4f  time=%dms\n",
           n_members, epochs, fin_trn, fin_evl, best_evl,
           (float)report_trn * 100.0f / (float)total_train,
           (double)aa.lr, elapsed_ms);

    printf("\n============================================================\n");
    printf("REPORT train=%.1f%% (%d) eval=%.1f%% (%d) err=%d lr=%.4f time=%dms threads=%d members=%d\n",
           (float)report_trn * 100.0f / (float)total_train, total_train,
           (float)report_evl * 100.0f / (float)total_eval, total_eval,
           total_train - report_trn, (double)aa.lr, elapsed_ms, aa.threadN, n_members);
    printf("============================================================\n");

    /* ── Cleanup ── */
    for (int m = 0; m < n_members; m++) bitvote_member_destroy(members[m]);
    free(members); free(X_tr); free(X_te);
    for (int xi = 0; xi < n_xforms; xi++) free(X_xform[xi]);
    ki_dataset_free(&data);
    return 0;
}
