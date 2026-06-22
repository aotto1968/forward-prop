/*
 * otto-score-ifc/mlp-flt32-trn-w1-adam.c — Float32 AdamW W1-Only (W0 Frozen)
 * =========================================================================
 *
 * Self-contained AdamW trainer (no #include "mlp.c", no #include "args.h").
 * Forward: matmul + LReLU(0.05) | Loss: MSE ±1 (all 10 classes)
 * Optimizer: AdamW(lr, wd=1e-4) | Schedule: warmup + cosine decay
 * Gradient clipping: max_norm=1.0
 * W0: kaiming_uniform init, FROZEN — W1: kaiming_uniform init, AdamW
 *
 * Interface: ki-common.h — same CLI parameters as all programs.
 *
 * Reference implementation for 2-layer float32 baseline.
 * See: ki-w1/mlp-flt32-trn-w1-adam.c (original)
 *      plans/plan-2026-06-04-flt32-w1-only.md (Resolved)
 */
#include "ki-common.h"

#define N_CLASSES KI_NCLASSES
#define ADAM_BETA1 0.9f
#define ADAM_BETA2 0.999f
#define ADAM_EPS   1e-8f
#define ADAM_WD    1e-4f

/* ═══════════════════════════════════════════════════════════════════════
 * FORWARD (Float: matmul + LReLU)
 * ═══════════════════════════════════════════════════════════════════════ */

static void forward(const ki_LinearLayer *l0, const ki_LinearLayer *l1,
                     const float *x, float *h0_out, float *output, int batchN) {
    ki_linear_forward(l0, x, h0_out, batchN);
    ki_leaky_relu(h0_out, batchN * l0->out_features);
    ki_linear_forward(l1, h0_out, output, batchN);
}

/* ═══════════════════════════════════════════════════════════════════════
 * ACCURACY
 * ═══════════════════════════════════════════════════════════════════════ */

static float accuracy_pct(const float *X, const uint8_t *Y, int N,
                           const ki_LinearLayer *l0, const ki_LinearLayer *l1,
                           int in_features) {
    int ok = 0;
    int H = l0->out_features;
    #pragma omp parallel reduction(+:ok)
    {
        float *h0_buf = (float *)malloc((size_t)H * sizeof(float));
        float *out_buf = (float *)malloc((size_t)N_CLASSES * sizeof(float));
        if (!h0_buf || !out_buf) { free(h0_buf); free(out_buf); }
        else {
            #pragma omp for schedule(static)
            for (int si = 0; si < N; si++) {
                const float *x = X + (size_t)si * (size_t)in_features;
                forward(l0, l1, x, h0_buf, out_buf, 1);
                int pred = 0;
                for (int k = 1; k < N_CLASSES; k++)
                    if (out_buf[k] > out_buf[pred]) pred = k;
                if (pred == (int)Y[si]) ok++;
            }
            free(h0_buf);
            free(out_buf);
        }
    }
    return (float)ok * 100.0f / (float)N;
}

/* ═══════════════════════════════════════════════════════════════════════
 * ADAMW OPTIMIZER STATE (for W1 only)
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    float *m;          /* first moment estimate */
    float *v;          /* second moment estimate (raw) */
    int    t;          /* time step */
    size_t n;          /* number of parameters */
    float  lr;         /* base learning rate */
    float  lr_min;     /* minimum LR (cosine floor) */
    float  beta1;
    float  beta2;
    float  eps;
    float  wd;         /* weight decay */
} ki_AdamWState;

static ki_AdamWState ki_adamw_create(size_t n, float lr, float lr_min) {
    ki_AdamWState s;
    s.n    = n;
    s.m    = (float *)ki_xcalloc(n, sizeof(float));
    s.v    = (float *)ki_xcalloc(n, sizeof(float));
    s.t    = 0;
    s.lr   = lr;
    s.lr_min = lr_min;
    s.beta1 = ADAM_BETA1;
    s.beta2 = ADAM_BETA2;
    s.eps   = ADAM_EPS;
    s.wd    = ADAM_WD;
    return s;
}

static void ki_adamw_free(ki_AdamWState *s) {
    free(s->m);
    free(s->v);
    s->m = NULL;
    s->v = NULL;
}

/* ── AdamW step: w -= lr_t * (m_hat / (sqrt(v_hat) + eps) + wd * w) ── */
static void ki_adamw_step(ki_AdamWState *s, float *w, const float *grad) {
    s->t++;
    float beta1 = s->beta1;
    float beta2 = s->beta2;
    float lr_t = s->lr;  /* external LR schedule applied via s->lr */
    float bc1 = 1.0f - powf(beta1, (float)s->t);  /* bias correction 1 */
    float bc2 = 1.0f - powf(beta2, (float)s->t);  /* bias correction 2 */

    #pragma omp parallel for schedule(static) if(s->n >= 64)
    for (size_t i = 0; i < s->n; i++) {
        float g = grad[i];

        /* Update biased moments */
        s->m[i] = beta1 * s->m[i] + (1.0f - beta1) * g;
        s->v[i] = beta2 * s->v[i] + (1.0f - beta2) * g * g;

        /* Bias correction */
        float m_hat = s->m[i] / bc1;
        float v_hat = s->v[i] / bc2;

        /* AdamW: decoupled weight decay + Adam update */
        w[i] -= lr_t * (s->wd * w[i] + m_hat / (sqrtf(v_hat) + s->eps));
    }
}

/* ── Gradient L2 norm ──────────────────────────────────────────── */
static float ki_grad_norm(const float *grad, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) sum += (double)grad[i] * (double)grad[i];
    return (float)sqrt(sum);
}

/* ── Gradient clipping (in-place, max_norm=1.0) ────────────────── */
static void ki_clip_grad(float *grad, size_t n) {
    float norm = ki_grad_norm(grad, n);
    if (norm > 1.0f) {
        float scale = 1.0f / norm;
        for (size_t i = 0; i < n; i++) grad[i] *= scale;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * WEIGHT EXPORT
 * ═══════════════════════════════════════════════════════════════════════ */

static int export_weights(const float *W0, const float *W1,
                           int hidden, int in_feat, const char *dir) {
    /* Create parent directory recursively */
    char parent[512];
    strncpy(parent, dir, sizeof(parent) - 1);
    parent[sizeof(parent) - 1] = '\0';
    char *slash = strrchr(parent, '/');
    if (slash) {
        *slash = '\0';
        if (parent[0] != '\0') {
            if (mkdir(parent, 0755) != 0 && errno != EEXIST) {
                fprintf(stderr, "[ERROR] Cannot create directory '%s': %s\n",
                        parent, strerror(errno));
                return -1;
            }
        }
    }

    if (mkdir(dir, 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "[ERROR] Cannot create directory '%s': %s\n",
                dir, strerror(errno));
        return -1;
    }

    char path[512];

    /* Meta */
    snprintf(path, sizeof(path), "%s/weights.meta", dir);
    FILE *f = fopen(path, "w");
    if (f) {
        fprintf(f, "2\n%d %d\n10 %d\n", hidden, in_feat, hidden);
        fclose(f);
    }

    /* W0: hidden × in_feat */
    size_t w0_size = (size_t)hidden * (size_t)in_feat * sizeof(float);
    snprintf(path, sizeof(path), "%s/W0.bin", dir);
    f = fopen(path, "wb");
    if (f) { fwrite(W0, 1, w0_size, f); fclose(f); }

    /* W1: 10 × hidden */
    size_t w1_size = (size_t)10 * (size_t)hidden * sizeof(float);
    snprintf(path, sizeof(path), "%s/W1.bin", dir);
    f = fopen(path, "wb");
    if (f) { fwrite(W1, 1, w1_size, f); fclose(f); }

    size_t total_kb = (w0_size + w1_size) / 1024;
    printf("══╡ EXPORT ╞═════════════════════════════════════════════════\n");
    printf("  Saved 2 layers to %s/  (%zu KB) [W0=frozen, W1=AdamW]\n",
           dir, total_kb);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════ */

int main(int argc, char *argv[]) {
    ki_Args a = ki_args_defaults();
    ki_parse_args(argc, argv, &a);

    /* Set OpenMP threads */
    omp_set_num_threads(a.threadN);

    int nc = KI_NC;  /* packed: 196 containers per image */
    int h0 = a.hidden;

    /* ── Dry run ─────────────────────────────────────────────── */
    if (a.dry_run) {
        ki_SetupInfo si = {
            .title = "Float AdamW W1-Only",
            .H = h0, .epochs = a.epochs,
            .bits_per_cont = KI_BITS_PER_CONT,
            .pixel_bits = 8,
            .seed = a.seed, .N = a.trainN, .ne = a.evalN,
            .n_threads = a.threadN, .px = KI_PX,
            .sizeof_bn = (int)sizeof(float),
            .nc = nc, .C = nc,
            .input_bit  = (size_t)nc * KI_BITS_PER_CONT,
            .hidden_bit = (size_t)h0 * KI_BITS_PER_CONT,
            .output_bit = (size_t)KI_NCLASSES * KI_BITS_PER_CONT,
            .w0_bit     = (size_t)h0 * (size_t)nc * KI_BITS_PER_CONT,
            .w1_bit     = (size_t)KI_NCLASSES * (size_t)h0 * KI_BITS_PER_CONT,
        };
        printf("══╡ Float AdamW W1-Only  W0=frozen  w1=AdamW ╞══  H=%-4d  Ep=%-2d  NC=%-3d\n",
               h0, a.epochs, nc);
        ki_dry_run_show(&si);
        printf("  %-18s %-12s  %s\n",  "mode",     "W1-only",   "W0 frozen (random), W1 AdamW");
        printf("  %-18s %-12d  %s\n",  "W1 params", KI_NCLASSES * h0, "trainable (AdamW)");
        printf("  %-18s %-12d  %s\n",  "W0 params", nc * h0,    "frozen (random kaiming)");
        printf("  %-18s %-12d  %s\n",  "NC",       nc,          "packed containers (4px/cont)");
        printf("  %-18s %-12s  %s\n",  "optimizer", "AdamW",     "beta1=0.9 beta2=0.999 wd=1e-4");
        printf("  %-18s %-12s  %s\n",  "schedule",  "warmup+cos","linear warmup + cosine decay");
        printf("  %-18s %-12s  %s\n",  "clip",      "max_norm=1","gradient clipping");
        return 1;
    }

    /* ── Load MNIST ──────────────────────────────────────────── */
    ki_MNISTData data;
    if (ki_mnist_read(&data) != 0) return 1;
    if (data.pixels != KI_PX) {
        fprintf(stderr, "[FATAL] Expected %d pixels, got %d\n", KI_PX, data.pixels);
        ki_mnist_free(&data);
        return 1;
    }

    /* ── Pack input: 784 px → 196 packed containers (4px/cont, averaged) ──
     *   Same 4-pixel grouping as mlp-bin32-trn-w1-hebbian.c.
     *   Instead of uint32, 4 pixels are averaged as float [-1,+1].
     */
    int total_train = a.trainN;
    int total_eval  = a.evalN;
    float *X_all = ki_pack_packed_float(data.X_raw, total_train + total_eval);
    float *X_tr  = X_all;
    float *X_te  = X_all + (size_t)total_train * (size_t)nc;
    uint8_t *y_tr = data.y;
    uint8_t *y_te = data.y + total_train;

    /* ── Architecture display ────────────────────────────────── */
    ki_SetupInfo si = {
        .title = "Float AdamW W1-Only",
        .H = h0, .epochs = a.epochs,
        .bits_per_cont = KI_BITS_PER_CONT,
        .pixel_bits = 8,
        .seed = a.seed, .N = total_train, .ne = total_eval,
        .n_threads = a.threadN, .px = KI_PX,
        .sizeof_bn = (int)sizeof(float),
        .nc = nc, .C = nc,
        .input_bit  = (size_t)nc * KI_BITS_PER_CONT,
        .hidden_bit = (size_t)h0 * KI_BITS_PER_CONT,
        .output_bit = (size_t)KI_NCLASSES * KI_BITS_PER_CONT,
        .w0_bit     = (size_t)h0 * (size_t)nc * KI_BITS_PER_CONT,
        .w1_bit     = (size_t)KI_NCLASSES * (size_t)h0 * KI_BITS_PER_CONT,
    };
    printf("══╡ Float AdamW W1-Only  W0=frozen  w1=AdamW ╞══  H=%-4d  Ep=%-2d  NC=%-3d\n",
           h0, a.epochs, nc);
    ki_setup_show(&si);

    int eff_warmup = a.warmup_epochs;
    float lr_min_f = ki_lr_uint_to_float(a.lr_min_uint);
    printf("══╡ TRAINING ╞══  lr=%.6f  lr-min=%.6f  decay=%s  batch=%d  warmup=%d  packed=4px/cont\n",
           (double)a.lr, (double)lr_min_f,
           a.no_decay ? "const" : "cosine",
           a.batchN, eff_warmup);

    /* ── Create model ────────────────────────────────────────── */
    ki_LinearLayer l0, l1;

    l0.in_features = nc;
    l0.out_features = h0;
    l0.W = (float *)ki_xcalloc((size_t)h0 * (size_t)nc, sizeof(float));

    l1.in_features = h0;
    l1.out_features = KI_NCLASSES;
    l1.W = (float *)ki_xcalloc((size_t)KI_NCLASSES * (size_t)h0, sizeof(float));

    /* Init with same seed for deterministic comparison */
    srand(a.seed);
    ki_init_kaiming(&l0, a.seed);
    srand(a.seed);
    ki_init_kaiming(&l1, a.seed);

    /* ── AdamW state for W1 (W0 is frozen) ───────────────────── */
    size_t w1_n = (size_t)KI_NCLASSES * (size_t)h0;
    ki_AdamWState adamw = ki_adamw_create(w1_n, a.lr, lr_min_f);

    /* Best-weight snapshots (for export) */
    float *best_W0 = (float *)ki_xmalloc((size_t)h0 * (size_t)nc * sizeof(float));
    float *best_W1 = (float *)ki_xmalloc(w1_n * sizeof(float));
    float best_eval_acc = 0.0f;

    /* Batch buffers */
    int *idx = (int *)ki_xmalloc((size_t)total_train * sizeof(int));
    float *grad_w1 = (float *)ki_xmalloc(w1_n * sizeof(float));
    float *h0_buf = (float *)ki_xmalloc((size_t)a.batchN * (size_t)h0 * sizeof(float));
    float *out_buf = (float *)ki_xmalloc((size_t)a.batchN * (size_t)KI_NCLASSES * sizeof(float));

    /* ── Training loop ───────────────────────────────────────── */
    double best_train = 0.0, final_eval = 0.0;
    struct timeval tv_start;
    gettimeofday(&tv_start, NULL);

    for (int ep = 0; ep < a.epochs; ep++) {

        /* LR schedule */
        float lr_ep = ki_lr_schedule(ep, a.epochs, eff_warmup,
                                      a.lr, lr_min_f, a.no_decay);
        adamw.lr = lr_ep;

        /* Shuffle */
        for (int i = 0; i < total_train; i++) idx[i] = i;
        ki_shuffle(idx, total_train);

        /* Batch loop */
        for (int start = 0; start < total_train; start += a.batchN) {
            int actual = a.batchN;
            if (start + actual > total_train) actual = total_train - start;

            /* Forward for all samples (parallel, each sample independent) */
            #pragma omp parallel for schedule(static)
            for (int bi = 0; bi < actual; bi++) {
                int sidx = idx[start + bi];
                const float *x = X_tr + (size_t)sidx * (size_t)nc;
                forward(&l0, &l1, x,
                        h0_buf + (size_t)bi * (size_t)h0,
                        out_buf + (size_t)bi * (size_t)KI_NCLASSES,
                        1);
            }

            /* Accumulate MSE gradient for W1 (parallel over class×neuron) */
            #pragma omp parallel for collapse(2) schedule(static)
            for (int k = 0; k < KI_NCLASSES; k++) {
                for (int h_id = 0; h_id < h0; h_id++) {
                    float acc = 0.0f;
                    for (int bi = 0; bi < actual; bi++) {
                        int sidx = idx[start + bi];
                        float tgt = (y_tr[sidx] == k) ? 1.0f : -1.0f;
                        float d_out = out_buf[bi * KI_NCLASSES + k] - tgt;
                        float h0_val = h0_buf[bi * h0 + h_id];
                        acc += d_out * h0_val;
                    }
                    grad_w1[k * h0 + h_id] = acc;
                }
            }

            /* Mean gradient */
            float inv_act = 1.0f / (float)actual;
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < w1_n; i++)
                grad_w1[i] *= inv_act;

            /* Gradient clipping (max_norm=1.0) */
            ki_clip_grad(grad_w1, w1_n);

            /* AdamW step */
            ki_adamw_step(&adamw, l1.W, grad_w1);
        }

        /* Evaluate */
        float train_acc = accuracy_pct(X_tr, y_tr, total_train, &l0, &l1, nc);
        float eval_acc = (total_eval > 0)
            ? accuracy_pct(X_te, y_te, total_eval, &l0, &l1, nc)
            : 0.0f;

        if (train_acc > best_train) best_train = train_acc;
        if (eval_acc > best_eval_acc) {
            best_eval_acc = eval_acc;
            /* Snapshot best weights */
            memcpy(best_W0, l0.W, (size_t)h0 * (size_t)nc * sizeof(float));
            memcpy(best_W1, l1.W, w1_n * sizeof(float));
        }
        if (ep == a.epochs - 1) final_eval = eval_acc;

        printf("  [AdamW-W1] E %2d | train=%5.1f%%  eval=%5.1f%%  lr=%.6f\n",
               ep + 1, (double)train_acc, (double)eval_acc, (double)lr_ep);
        fflush(stdout);
    }

    struct timeval tv_end;
    gettimeofday(&tv_end, NULL);
    int elapsed_ms = (int)((tv_end.tv_sec - tv_start.tv_sec) * 1000
                         + (tv_end.tv_usec - tv_start.tv_usec) / 1000);

    /* ── TRAIN block ─────────────────────────────────────────── */
    printf("\n══╡ TRAIN ╞══════════════════════════════════════════════════\n");
    printf("  accuracy     %.1f%% (best)\n", (double)best_eval_acc);
    printf("  time         %dms  (total train+eval)\n", elapsed_ms);

    /* ── EVAL block ──────────────────────────────────────────── */
    if (total_eval > 0) {
        printf("══╡ EVAL ╞═══════════════════════════════════════════════════\n");
        printf("  eval=        %.1f%%\n", (double)final_eval);
    }

    /* ── EXPORT ──────────────────────────────────────────────── */
    char out_dir[256];
    if (a.out[0] != '\0') {
        strncpy(out_dir, a.out, sizeof(out_dir) - 1);
        out_dir[sizeof(out_dir) - 1] = '\0';
    } else {
        snprintf(out_dir, sizeof(out_dir), KI_MODEL_DIR "/flt32-w1-h%d", h0);
    }
    export_weights(best_W0, best_W1, h0, nc, out_dir);

    /* ── REPORT ──────────────────────────────────────────────── */
    int train_ok = (int)(best_train * (float)total_train / 100.0f + 0.5f);
    int eval_ok  = (int)(final_eval * (float)total_eval / 100.0f + 0.5f);
    ki_report_show(train_ok, total_train, eval_ok, total_eval, elapsed_ms, a.threadN);

    /* ── Cleanup ─────────────────────────────────────────────── */
    ki_adamw_free(&adamw);
    free(best_W0);
    free(best_W1);
    free(idx);
    free(grad_w1);
    free(h0_buf);
    free(out_buf);
    free(X_all);
    free(l0.W);
    free(l1.W);
    ki_mnist_free(&data);

    return 0;
}
