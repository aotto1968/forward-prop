/*
 * otto-score-ifc/mlp-otto-score-ensemble.c — Otto Score Ensemble Trainer
 * =======================================================================
 *
 * Reference implementation of Otto Score with ensemble voting.
 * Train N independent W0s and combine via score averaging.
 *
 * --ensembleN N : number of W0s to train in parallel (default 1)
 *
 * Model format version 5 (compatible with mlp-otto-score-ifc):
 *   magic, version=5, mode, ensembleN, H, NC
 *   For each m in 0..ensembleN-1:
 *     W0[m]:    uint32[H * NC]
 *     Target[m]: int32[10 * H * 32]
 *     Offset[m]: int64[10]
 *
 * Original: ki-w2/mlp-otto-score-ensemble.c
 */
#include "ki-otto-common.h"
#include "w0_random.h"
#include "maj3.h"
#include <inttypes.h>

/* ── Konstanten ────────────────────────────────────────────────── */
#define NC        196
#define BITS       32
#define N_CLASSES KI_NCLASSES

/* Export file magic + version */
#define OTTO_MAGIC   0x4F54544FU   /* "OTTO" */
#define OTTO_VERSION 5U            /* v5 = ensemble (no precision field) */
#define OTTO_VERSION_V6 6U         /* v6 = ensemble + precision field */

/* Index: [10][H][32] — klasse × neuron × bit */


// === Zentraler Skalierungsfaktor OT_PRECISION ===
// Alle logit/log(p)-Werte werden mit F = (1<<OT_PRECISION) skaliert
// in int32/int64 gespeichert.  Der Korrektur-Step und die lr-Anzeige
// leiten sich daraus ab → eine Änderung wirkt auf alle abhängigen Stellen.
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#ifndef OT_PRECISION
#endif

// #pragma message(">>> OT_PRECISION = " TOSTRING(OT_PRECISION))


/* ═══════════════════════════════════════════════════════════════════
 * H0 MODE — XNOR (default) or XOR (via -DH0_XOR)
 * ═══════════════════════════════════════════════════════════════════ */
#ifdef H0_XOR
#  define H0_STR     "XOR"
#  define H0_MATCH(in, W0_row, c)  ((in)[c] ^ (W0_row)[c])
#  define H0_MODE_VAL 1U
#else
#  define H0_STR     "XNOR"
#  define H0_MATCH(in, W0_row, c)  (~((in)[c] ^ (W0_row)[c]))
#  define H0_MODE_VAL 0U
#endif


/* ═══════════════════════════════════════════════════════════════════
 * H0 FORWARD — MAJ3 pro Neuron
 * ═══════════════════════════════════════════════════════════════════ */
static uint32_t h0_neuron(const uint32_t *in, const uint32_t *W0_row) {
    uint32_t match[NC];
    for (int c = 0; c < NC; c++)
        match[c] = H0_MATCH(in, W0_row, c);
    return majority_tree(match, NC);
}


/* ═══════════════════════════════════════════════════════════════════
 * OTTO TARGET — class-dependent
 * ═══════════════════════════════════════════════════════════════════
 * target[h][k][b]++ ONLY when Sample class=k AND H0[h] bit b = 1
 */
static int32_t *build_target(const uint32_t *X, const uint8_t *Y, int N,
                              const uint32_t *W0, int H, int silent) {
    size_t sz = (size_t)H * 10 * 32;
    int32_t *target = (int32_t *)ki_xcalloc(sz, sizeof(int32_t));

    if (!silent) {
        printf("\n══╡ OTTO TARGET ╞═══════════════════════════════════════════════\n");
        printf("  Target[10][%d][32] = %zu KB\n", H, sz * sizeof(int32_t) / 1024);
        fflush(stdout);
    }

    #pragma omp parallel
    {
        int32_t *lt = (int32_t *)ki_xcalloc(sz, sizeof(int32_t));
        #pragma omp for schedule(static)
        for (int s = 0; s < N; s++) {
            int k = (int)Y[s];
            const uint32_t *in = X + (size_t)s * NC;
            for (int h = 0; h < H; h++) {
                uint32_t h0 = h0_neuron(in, W0 + (size_t)h * NC);
                for (int b = 0; b < 32; b++)
                    if (h0 & (1U << (unsigned)b))
                        lt[TGT_IDX(k, h, b, H)]++;
            }
        }
        #pragma omp critical
        { for (size_t i = 0; i < sz; i++) target[i] += lt[i]; }
        free(lt);
    }
    return target;
}


/* ═══════════════════════════════════════════════════════════════════
 * LOGIT CONVERT — Target-Counts → log-odds (in-place)
 * ═══════════════════════════════════════════════════════════════════
 *   target[k][h][b] = round( ln((t+1)/(N_k-t+1)) × F )
 *   mit F = (1<<OT_PRECISION) (default 10 → F=1024)
 *
 * Abhängigkeit: ot_precision() definiert die int32-Skalierung.
 */
static void logit_convert(int32_t *target, int H, const int class_counts[10]) {
    for (int k = 0; k < 10; k++) {
        int nk = class_counts[k];
        if (nk <= 0) continue;
        for (int h = 0; h < H; h++) {
            for (int b = 0; b < 32; b++) {
                size_t idx = TGT_IDX(k, h, b, H);
                int t = target[idx];
                double p = (double)(t + 1) / (double)(nk + 2);
                target[idx] = (int32_t)ot_precision(log(p / (1.0 - p)));
            }
        }
    }
}


/* ═══════════════════════════════════════════════════════════════════
 * CLASS OFFSET — Σ log(1-P_k) × F  (F = 1<<OT_PRECISION)
 * ═══════════════════════════════════════════════════════════════════
 * Muss VOR logit_convert berechnet werden (braucht raw counts).
 * Gemeinsame Skalierung: target und offset teilen F.
 *
 * Note: +-0.5 rounding removed because log(p1)*F is always negative
 * and the 0.5 correction (~0.001%) is below measurement noise (±0.3pp).
 * See: 2026-06-18 experiment — 3 runs with/without 0.5 gave identical
 * results within normal OpenMP scheduling noise.
 */
static void compute_class_offset(int64_t class_offset[10],
                                  const int32_t *target, int H,
                                  const int class_counts[10]) {
    for (int k = 0; k < 10; k++) {
        int64_t sum = 0;
        int nk = class_counts[k];
        if (nk <= 0) { class_offset[k] = 0; continue; }
        for (int h = 0; h < H; h++) {
            for (int b = 0; b < 32; b++) {
                int t = target[TGT_IDX(k, h, b, H)];
                double p1 = (double)(nk - t + 1) / (double)(nk + 2);
                sum += (int64_t)ot_precision(log(p1));
            }
        }
        class_offset[k] = sum;
    }
}
/* ═══════════════════════════════════════════════════════════════════
 * SCORE — Bayes log-Score
 */
static void scores_otto(const uint32_t *in, const uint32_t *W0, int H,
                         const int32_t *target,
                         const int64_t class_offset[10],
                         int64_t scores[10]) {
    for (int k = 0; k < N_CLASSES; k++)
        scores[k] = class_offset[k];

    for (int h = 0; h < H; h++) {
        uint32_t h0 = h0_neuron(in, W0 + (size_t)h * NC);
        for (int k = 0; k < N_CLASSES; k++) {
            for (int b = 0; b < 32; b++) {
                if (h0 & (1U << (unsigned)b))
                    scores[k] += target[TGT_IDX(k, h, b, H)];
            }
        }
    }
}


/* ═══════════════════════════════════════════════════════════════════
 * ACCURACY
 * ═══════════════════════════════════════════════════════════════════ */
static __attribute__((unused)) float accuracy_pct(const uint32_t *X, const uint8_t *Y, int N,
                           const uint32_t *W0, int H,
                           const int32_t *target,
                           const int64_t class_offset[10]) {
    int ok = 0;
    #pragma omp parallel for reduction(+:ok) schedule(static)
    for (int s = 0; s < N; s++) {
        int64_t scores[10];
        scores_otto(X + (size_t)s * NC, W0, H, target, class_offset, scores);
        int pred = 0;
        for (int k = 1; k < N_CLASSES; k++)
            if (scores[k] > scores[pred]) pred = k;
        if (pred == (int)Y[s]) ok++;
    }
    return (float)ok * 100.0f / (float)N;
}


/* ═══════════════════════════════════════════════════════════════════
 * EXPORT — W0 + Target + Offset als model.otto
 * ═══════════════════════════════════════════════════════════════════
 *
 * Format (Einzeldatei):
 *   Header: 20 Byte
 *     uint32 magic      = 0x4F544F54 ('OTTO')
 *     uint32 version    = 1
 *     uint32 h0_mode    = 0 (XNOR) / 1 (XOR)
 *     uint32 H          = hidden neurons
 *     uint32 NC         = containers per image
 *   W0:      uint32[H * NC]
 *   Target:  int32[10 * H * 32]
 *   Offset:  int64[10]
 */
static __attribute__((unused)) void export_model(const char *out_dir,
                           const uint32_t *W0, int H,
                           const int32_t *target,
                           const int64_t class_offset[10]) {
    /* Not used in ensemble mode — use export_ensemble instead */
    (void)out_dir; (void)H;
    (void)W0; (void)target; (void)class_offset;
    fprintf(stderr, "[ERROR] Use export_ensemble with --ensembleN\n");
}


/* ── Export ensemble: all N models into one file ──────────────── */
static void export_ensemble(const char *out_dir,
                             const uint32_t *W0_ens, int H, int ensembleN,
                             const int32_t *target_ens,
                             const int64_t *offset_ens) {
    char cmd[512], path[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", out_dir);
    if (system(cmd) != 0) return;
    snprintf(path, sizeof(path), "%s/model.otto", out_dir);

    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[ERROR] Cannot write %s\n", path); return; }

    uint32_t magic = OTTO_MAGIC, ver = OTTO_VERSION_V6, mode = H0_MODE_VAL;
    uint32_t ens = (uint32_t)ensembleN, hh = (uint32_t)H, ncc = (uint32_t)NC;
    uint32_t prec = (uint32_t)OT_PRECISION;
    fwrite(&magic,4,1,f); fwrite(&ver,4,1,f);
    fwrite(&mode,4,1,f); fwrite(&ens,4,1,f);
    fwrite(&hh,4,1,f); fwrite(&ncc,4,1,f);
    fwrite(&prec,4,1,f);   /* v6: precision field */

    size_t w0_bytes = (size_t)H * (size_t)NC * 4;
    size_t tgt_bytes = (size_t)H * 10 * 32 * 4;
    size_t off_bytes = 10 * 8;
    size_t total = 0;

    for (int m = 0; m < ensembleN; m++) {
        fwrite(W0_ens + (size_t)m * (size_t)H * (size_t)NC, 4, (size_t)H * (size_t)NC, f);
        fwrite(target_ens + (size_t)m * (size_t)10 * (size_t)H * (size_t)32, 4, (size_t)H * 10 * 32, f);
        fwrite(offset_ens + (size_t)m * 10, 8, 10, f);
        total += w0_bytes + tgt_bytes + off_bytes;
    }

    fclose(f);

    printf("\n══╡ EXPORT ╞═══════════════════════════════════════════════════\n");
    printf("  Model:  %s  (v%u, %d ensemble, F=%d, precision=%u)\n",
           path, OTTO_VERSION_V6, ensembleN, 1<<OT_PRECISION, prec);
    printf("  Total:  %zu KB  (%d × (W0=%zuKB + Tgt=%zuKB + Off=%zuB))\n",
           (24 + total) / 1024, ensembleN,
           w0_bytes / 1024, tgt_bytes / 1024, off_bytes);
    fflush(stdout);
}


/* ═══════════════════════════════════════════════════════════════════
 * SETUP
 * ═══════════════════════════════════════════════════════════════════ */
static void print_setup(int H, int epochs, int trainN, int evalN,
                         int threadN, unsigned int seed) {
    (void)trainN; (void)evalN; (void)threadN;
    size_t tgt_bytes = (size_t)H * 10 * 32 * sizeof(int32_t);
    printf("══════════════════════════════════════════════════════════════════════\n");
    printf("══╡ OTTO-SCORE ╞══  %s  H=%-4d  Ep=%-2d  NC=%-3d  seed=%-4u\n",
           H0_STR, H, epochs, NC, seed);
    printf("══╡ SETUP ╞══════════════════════════════════════════════════════════\n");
    printf("  W0:          random uint32[%d][%d]  (frozen, %zu KB)\n",
           H, NC, (size_t)H * (size_t)NC * sizeof(uint32_t) / 1024);
    printf("  Target:      int32[10][%d][32]      (class-dependent, %zu KB)\n",
           H, tgt_bytes / 1024);
    printf("  Score:       Σ_h Σ_b [ y×log(P_k) + (1-y)×log(1-P_k) ]\n");
    printf("  Predict:     argmax  (NO training, NO AdamW)\n");
    printf("  ───────────────────────────────────────────────────────────\n");
}


/* ═══════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[]) {
    ki_Args a = ki_args_defaults();
    a.lr = 0.05f;   /* Default step for iterative target-tuning */
    a.lr_step = (int)round(a.lr * (1<<OT_PRECISION));
    ki_parse_args(argc, argv, &a);
    omp_set_num_threads(a.threadN);

    int H = a.hidden;

    /* ── Dry run ─────────────────────────────────────────────── */
    if (a.dry_run) {
        print_setup(H, a.epochs, a.trainN, a.evalN, a.threadN, a.seed);
        return 0;
    }

    /* ── Load MNIST ──────────────────────────────────────────── */
    ki_MNISTData data;
    if (ki_mnist_read(&data) != 0) return 1;
    if (data.pixels != KI_PX) {
        fprintf(stderr, "[FATAL] Expected %d pixels, got %d\n", KI_PX, data.pixels);
        ki_mnist_free(&data); return 1;
    }

    int total_train = a.trainN;
    int total_eval  = a.evalN;
    int total_all   = total_train + total_eval;

    uint32_t *X_all = load_input(data.X_raw, total_all);
    uint32_t *X_tr  = X_all;
    uint32_t *X_te  = X_all + (size_t)total_train * NC;
    uint8_t  *y_tr  = data.y;
    uint8_t  *y_te  = data.y + total_train;
    int own_eval_data = 0;
    uint32_t *X_perm = NULL;  /* allocated by shuffle, freed in cleanup */
    uint8_t  *y_perm = NULL;

    /* ── Optional: shuffle indices before train/eval split ───────── */
    if (a.shuffle) {
        printf("  Shuffling %d samples before %d/%d split...\n",
               total_all, total_train, total_eval);
        int *idx = (int *)malloc((size_t)total_all * sizeof(int));
        for (int i = 0; i < total_all; i++) idx[i] = i;
        srand(a.seed);  /* reuse --seed for reproducibility */
        for (int i = total_all - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
        }

        X_perm = (uint32_t *)malloc((size_t)total_all * NC * sizeof(uint32_t));
        y_perm = (uint8_t *)malloc((size_t)total_all * sizeof(uint8_t));
        for (int i = 0; i < total_all; i++) {
            int src = idx[i];
            memcpy(X_perm + (size_t)i * NC, X_all + (size_t)src * NC,
                   (size_t)NC * sizeof(uint32_t));
            y_perm[i] = data.y[src];
        }

        X_tr = X_perm;
        X_te = X_perm + (size_t)total_train * NC;
        y_tr = y_perm;
        y_te = y_perm + total_train;
        own_eval_data = 1;
        free(idx);
    }

    int ensembleN = a.ensembleN;
    if (ensembleN < 1) ensembleN = 1;

    print_setup(H, a.epochs, total_train, total_eval, a.threadN, a.seed);
    ki_print_setup(&a, total_train, total_eval);
    printf("══════════════════════════════════════════════════════════════════════\n");
    const char *rng_src = a.random_file[0] ? "true random file" : "PRNG";
    if (a.ensemble_seed[0] && strcmp(a.ensemble_seed, "broken-31") == 0)
        rng_src = "broken-31 (glibc LCG)";
    else if (a.ensemble_seed[0] && strcmp(a.ensemble_seed, "fix-gnu") == 0)
        rng_src = "fix-gnu (rand XOR hack)";
    else if (a.ensemble_seed[0] && strcmp(a.ensemble_seed, "fix-splitmix") == 0)
        rng_src = "fix-splitmix (explicit)";
    printf("  Ensemble:  %d W0s (seed %u, %s", ensembleN, a.seed, rng_src);
    if (a.ensemble_seed[0]) printf(", --ensemble-seed %s", a.ensemble_seed);
    printf(")\n");
    fflush(stdout);

    /* ── W0: random uint32[ensembleN][H][NC] (frozen) ─────────── */
    size_t w0_sz = (size_t)ensembleN * (size_t)H * (size_t)NC;
    uint32_t *W0_ens = (uint32_t *)ki_xmalloc(w0_sz * sizeof(uint32_t));
    if (a.random_file[0]) {
        ki_rand_fill(W0_ens, w0_sz, a.random_file);
    } else if (a.ensemble_seed[0] && strcmp(a.ensemble_seed, "broken-31") == 0) {
        /* broken-31: glibc rand() — 31-bit (Bit 31 always 0), LCG */
        srand((unsigned int)a.seed);
        w0_srandom((unsigned int)a.seed);
        for (size_t i = 0; i < w0_sz; i++)
            W0_ens[i] = (uint32_t)rand();
    } else if (a.ensemble_seed[0] && strcmp(a.ensemble_seed, "fix-gnu") == 0) {
        /* fix-gnu: ((uint32_t)rand()<<16) ^ (uint32_t)rand() — 32-bit but still LCG */
        srand((unsigned int)a.seed);
        w0_srandom((unsigned int)a.seed);
        for (size_t i = 0; i < w0_sz; i++)
            W0_ens[i] = ((uint32_t)rand() << 16) ^ (uint32_t)rand();
    } else if (a.ensemble_seed[0] && strcmp(a.ensemble_seed, "incr") == 0) {
        /* incr: each member gets seed + m */
        for (int m = 0; m < ensembleN; m++) {
            w0_srandom((uint64_t)a.seed + (uint64_t)m);
            for (size_t i = 0; i < (size_t)H * (size_t)NC; i++)
                W0_ens[(size_t)m * (size_t)H * (size_t)NC + i] = w0_random();
        }
    } else if (a.ensemble_seed[0] && strcmp(a.ensemble_seed, "const") == 0) {
        /* const: same seed for every member → identical W0 */
        w0_srandom((unsigned int)a.seed);
        size_t row_sz = (size_t)H * (size_t)NC;
        uint32_t *row = (uint32_t *)ki_xmalloc(row_sz * sizeof(uint32_t));
        for (size_t i = 0; i < row_sz; i++) row[i] = w0_random();
        for (int m = 0; m < ensembleN; m++)
            memcpy(W0_ens + (size_t)m * row_sz, row, row_sz * sizeof(uint32_t));
        free(row);
    } else if (a.ensemble_seed[0] && strcmp(a.ensemble_seed, "fix-splitmix") == 0) {
        /* fix-splitmix: explicit alias for default — splitmix64 */
        w0_srandom((unsigned int)a.seed);
        for (size_t i = 0; i < w0_sz; i++)
            W0_ens[i] = w0_random();
    } else {
        /* default: seed once, fill all members sequentially */
        w0_srandom((unsigned int)a.seed);
        for (size_t i = 0; i < w0_sz; i++)
            W0_ens[i] = w0_random();
    }

    /* ── Target + Offset for each ensemble member ────────────────── */
    size_t tgt_sz_m = (size_t)H * 10 * 32;  /* per member */
    size_t tgt_sz   = (size_t)ensembleN * tgt_sz_m;
    int32_t *target_ens = (int32_t *)ki_xcalloc(tgt_sz, sizeof(int32_t));
    int64_t *offset_ens = (int64_t *)ki_xcalloc((size_t)ensembleN * 10, sizeof(int64_t));

    int class_counts[10] = {0};
    for (int s = 0; s < total_train; s++)
        class_counts[(int)y_tr[s]]++;

    /* Build target for each ensemble member */
    size_t tgt_kb = tgt_sz_m * sizeof(int32_t) / 1024;
    printf("\n══╡ ENSEMBLE ╞════════════════════════════════════════════════\n");
    for (int m = 0; m < ensembleN; m++) {
        int32_t *target_m = target_ens + (size_t)m * tgt_sz_m;
        const uint32_t *W0_m = W0_ens + (size_t)m * (size_t)H * (size_t)NC;

        /* Count (silent) */
        int32_t *tgt = build_target(X_tr, y_tr, total_train, W0_m, H, 1);
        memcpy(target_m, tgt, tgt_sz_m * sizeof(int32_t));
        free(tgt);

        /* Offset */
        int64_t off_m[10];
        compute_class_offset(off_m, target_m, H, class_counts);
        memcpy(offset_ens + (size_t)m * 10, off_m, 10 * sizeof(int64_t));

        /* Logit convert */
        logit_convert(target_m, H, class_counts);

        /* Marker: reseed for const mode to verify identical seed */
        if (a.ensemble_seed[0] && strcmp(a.ensemble_seed, "const") == 0)
            w0_srandom((unsigned int)a.seed);
        uint32_t rnd_marker = w0_random();

        printf("  Ensemble [%d/%d] - Target[10][%d][32] = %zu KB (seed=%u,#%u)\n",
               m + 1, ensembleN, H, tgt_kb, a.seed, rnd_marker);
    }
    printf("══╡ OTTO TARGET ╞═══════════════════════════════════════════════\n");

    /* ═══════════════════════════════════════════════════════════════
     * ITERATIVES TARGET-TUNING (per ensemble member)
     * ═══════════════════════════════════════════════════════════════
     * step = ot_precision(lr) = round(lr * F)  mit F = (1<<OT_PRECISION).
     * Dadurch sind Korrektur, Target und Offset in der gleichen Skala.
     *
     * Bei OT_PRECISION=10 (F=1024) und lr=0.05: step = round(0.05*1024) = 51.
     */
    int step_init = (a.lr > 0) ? (int)ot_precision(a.lr) : a.lr_step;
    int warmup = a.warmup_epochs;
    int epochs = (a.epochs < 1) ? 1 : a.epochs;

    /* Per-member best copies */
    int32_t *best_ens = (int32_t *)ki_xmalloc(tgt_sz * sizeof(int32_t));
    int64_t *best_off = (int64_t *)ki_xmalloc((size_t)ensembleN * 10 * sizeof(int64_t));
    float best_evl = 0.0f;

    memcpy(best_ens, target_ens, tgt_sz * sizeof(int32_t));
    memcpy(best_off, offset_ens, (size_t)ensembleN * 10 * sizeof(int64_t));

    struct timeval tv_start, tv_end;
    gettimeofday(&tv_start, NULL);

    for (int ep = 0; ep < epochs; ep++) {
        /* Per-epoch step: cosine decay + warmup (default) or constant */
        int current_step;
        if (a.step_const) {
            current_step = step_init;
        } else {
            float lr_scale = ki_lr_schedule(ep, epochs, warmup, 1.0f, 0.1f, 0);
            current_step = (int)((float)step_init * lr_scale);
            if (current_step < 1) current_step = 1;
        }

        /* Evaluate ensemble: sum scores across members */
        int trn_ok = 0, evl_ok = 0;
        #pragma omp parallel for reduction(+:trn_ok) schedule(static)
        for (int s = 0; s < total_train; s++) {
            int64_t total[10] = {0};
            for (int m = 0; m < ensembleN; m++) {
                int64_t sc[10];
                scores_otto(X_tr + (size_t)s * NC,
                           W0_ens + (size_t)m * (size_t)H * (size_t)NC, H,
                           target_ens + (size_t)m * tgt_sz_m,
                           offset_ens + (size_t)m * 10, sc);
                for (int k = 0; k < 10; k++) total[k] += sc[k];
            }
            int pred = 0;
            for (int k = 1; k < 10; k++)
                if (total[k] > total[pred]) pred = k;
            if (pred == (int)y_tr[s]) trn_ok++;
        }

        if (total_eval > 0) {
            #pragma omp parallel for reduction(+:evl_ok) schedule(static)
            for (int s = 0; s < total_eval; s++) {
                int64_t total[10] = {0};
                for (int m = 0; m < ensembleN; m++) {
                    int64_t sc[10];
                    scores_otto(X_te + (size_t)s * NC,
                               W0_ens + (size_t)m * (size_t)H * (size_t)NC, H,
                               target_ens + (size_t)m * tgt_sz_m,
                               offset_ens + (size_t)m * 10, sc);
                    for (int k = 0; k < 10; k++) total[k] += sc[k];
                }
                int pred = 0;
                for (int k = 1; k < 10; k++)
                    if (total[k] > total[pred]) pred = k;
                if (pred == (int)y_te[s]) evl_ok++;
            }
        }

        float trn_acc = (float)trn_ok * 100.0f / (float)total_train;
        float evl_acc = (total_eval > 0)
            ? (float)evl_ok * 100.0f / (float)total_eval : 0.0f;

        if (evl_acc > best_evl) {
            best_evl = evl_acc;
            memcpy(best_ens, target_ens, tgt_sz * sizeof(int32_t));
            memcpy(best_off, offset_ens, (size_t)ensembleN * 10 * sizeof(int64_t));
        }

        gettimeofday(&tv_end, NULL);
        int elapsed = (int)((tv_end.tv_sec - tv_start.tv_sec) * 1000
                          + (tv_end.tv_usec - tv_start.tv_usec) / 1000);

        printf("Ep %2d  trn=%.1f%%  evl=%.1f%%  best=%.1f%%  step=%d  time=%dms",
               ep + 1, trn_acc, evl_acc, best_evl, current_step, elapsed);

        if (ep + 1 >= epochs) { printf("\n"); break; }

        /* Print ensemble training error (misclassifications) before correction */
        int trn_err = total_train - trn_ok;
        printf("  err=%d", trn_err);

        /* Converged → no corrections needed, stop early */
        if (trn_err == 0) {
            printf("  ✓\n");
            break;
        }

        /* Correction pass for each ensemble member (mini-batch, H0-cached) */
        for (int m = 0; m < ensembleN; m++) {
            const uint32_t *W0_m = W0_ens + (size_t)m * (size_t)H * (size_t)NC;

            /* H0 für Member m vorberechnen */
            uint32_t *h0_m = (uint32_t *)ki_xmalloc((size_t)total_train * (size_t)H * sizeof(uint32_t));
            #pragma omp parallel for schedule(static)
            for (int s = 0; s < total_train; s++) {
                const uint32_t *in = X_tr + (size_t)s * NC;
                for (int h = 0; h < H; h++)
                    h0_m[(size_t)s * (size_t)H + (size_t)h] = h0_neuron(in, W0_m + (size_t)h * NC);
            }

            ki_batch_correct(
                target_ens + (size_t)m * tgt_sz_m, H,
                offset_ens + (size_t)m * 10,
                h0_m, y_tr, total_train,
                MAX(1, a.batchN), current_step, tgt_sz_m);

            free(h0_m);
        }
        if (epochs > 1) printf("\n");
        else printf("\n");
    }

    /* Restore best */
    memcpy(target_ens, best_ens, tgt_sz * sizeof(int32_t));
    memcpy(offset_ens, best_off, (size_t)ensembleN * 10 * sizeof(int64_t));
    free(best_ens); free(best_off);

    gettimeofday(&tv_end, NULL);
    int elapsed_ms = (int)((tv_end.tv_sec - tv_start.tv_sec) * 1000
                         + (tv_end.tv_usec - tv_start.tv_usec) / 1000);

    /* Final evaluation */
    int trn_ok = 0, evl_ok = 0;
    #pragma omp parallel for reduction(+:trn_ok) schedule(static)
    for (int s = 0; s < total_train; s++) {
        int64_t total[10] = {0};
        for (int m = 0; m < ensembleN; m++) {
            int64_t sc[10];
            scores_otto(X_tr + (size_t)s * NC,
                       W0_ens + (size_t)m * (size_t)H * (size_t)NC, H,
                       target_ens + (size_t)m * tgt_sz_m,
                       offset_ens + (size_t)m * 10, sc);
            for (int k = 0; k < 10; k++) total[k] += sc[k];
        }
        int pred = 0;
        for (int k = 1; k < 10; k++) if (total[k] > total[pred]) pred = k;
        if (pred == (int)y_tr[s]) trn_ok++;
    }
    if (total_eval > 0) {
        #pragma omp parallel for reduction(+:evl_ok) schedule(static)
        for (int s = 0; s < total_eval; s++) {
            int64_t total[10] = {0};
            for (int m = 0; m < ensembleN; m++) {
                int64_t sc[10];
                scores_otto(X_te + (size_t)s * NC,
                           W0_ens + (size_t)m * (size_t)H * (size_t)NC, H,
                           target_ens + (size_t)m * tgt_sz_m,
                           offset_ens + (size_t)m * 10, sc);
                for (int k = 0; k < 10; k++) total[k] += sc[k];
            }
            int pred = 0;
            for (int k = 1; k < 10; k++) if (total[k] > total[pred]) pred = k;
            if (pred == (int)y_te[s]) evl_ok++;
        }
    }

    float fin_trn = (float)trn_ok * 100.0f / (float)total_train;
    float fin_evl = (total_eval > 0)
        ? (float)evl_ok * 100.0f / (float)total_eval : 0.0f;

    printf("\n══╡ RESULT ╞══════════════════════════════════════════════════════\n");
    printf("  ENSEMBLE H=%d  N=%d  ep=%d  trn=%.1f%%  evl=%.1f%%  best=%.1f%%  lr=%.4f  time=%dms\n",
           H, ensembleN, epochs, fin_trn, fin_evl, best_evl,
           (double)a.lr, elapsed_ms);

    ki_report_show(trn_ok, total_train, evl_ok, total_eval, elapsed_ms, a.threadN);

    /* ── Export ensemble ───────────────────────────────────────── */
    if (a.out[0] != '\0')
        export_ensemble(a.out, W0_ens, H, ensembleN, target_ens, offset_ens);

    /* ── Cleanup ────────────────────────────────────────────── */
    if (own_eval_data) {
        free(X_perm);
        free(y_perm);
    }
    free(X_all);
    free(W0_ens); free(target_ens); free(offset_ens);
    ki_mnist_free(&data);
    return 0;
}
