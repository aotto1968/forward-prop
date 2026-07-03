/*
 * cifar-1/mlp-flt32-w1-adam-trn.c — Float32 AdamW W1-Only (W0 Frozen)
 * ====================================================================
 *
 * CIFAR-10 version. Forward: matmul + LReLU(0.05) | Loss: MSE ±1
 * Optimizer: AdamW(lr, wd=1e-4) | Schedule: warmup + cosine decay
 * W0: kaiming_uniform init, FROZEN — W1: kaiming_uniform init, AdamW
 *
 * Derived from otto-score-ifc/mlp-flt32-w1-adam-trn.c
 * Changes: CIFAR-10 data loader, 3072 input → 768 packed containers
 */
/* ki-adamw.h includes the project's current ki-common.h via cifar-include/ */
#include "ki-adamw.h"

#define N_CLASSES KI_NCLASSES
#define ADAM_BETA1 0.9f
#define ADAM_BETA2 0.999f
#define ADAM_EPS   1e-8f
#define ADAM_WD    1e-4f

/* ── Forward (Float: matmul + LReLU) ───────────────────────────── */
static void forward(const ki_LinearLayer *l0, const ki_LinearLayer *l1,
                     const float *x, float *h0_out, float *output, int batchN) {
    ki_linear_forward(l0, x, h0_out, batchN);
    ki_leaky_relu(h0_out, batchN * l0->out_features);
    ki_linear_forward(l1, h0_out, output, batchN);
}

/* ── Accuracy ──────────────────────────────────────────────────── */
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
    return 100.0f * (float)ok / (float)N;
}

/* ── AdamW state ───────────────────────────────────────────────── */
typedef struct {
    float lr, lr_min;
    float *m, *v;
    int t;
} ki_AdamWState;

static inline ki_AdamWState ki_adamw_create(size_t n, float lr, float lr_min) {
    ki_AdamWState s;
    s.lr = lr; s.lr_min = lr_min; s.t = 0;
    s.m = (float *)ki_xcalloc(n, sizeof(float));
    s.v = (float *)ki_xcalloc(n, sizeof(float));
    return s;
}

static inline void ki_adamw_update(ki_AdamWState *s, float *w, const float *g, size_t n) {
    s->t++;
    float lr_t = s->lr * sqrtf(1.0f - powf(ADAM_BETA2, (float)s->t))
                           / (1.0f - powf(ADAM_BETA1, (float)s->t));
    float lr_eff = fmaxf(lr_t, s->lr_min);
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        s->m[i] = ADAM_BETA1 * s->m[i] + (1.0f - ADAM_BETA1) * g[i];
        s->v[i] = ADAM_BETA2 * s->v[i] + (1.0f - ADAM_BETA2) * g[i] * g[i];
        float m_hat = s->m[i] / (1.0f - powf(ADAM_BETA1, (float)s->t));
        float v_hat = s->v[i] / (1.0f - powf(ADAM_BETA2, (float)s->t));
        w[i] -= lr_eff * m_hat / (sqrtf(v_hat) + ADAM_EPS);
        /* AdamW weight decay */
        w[i] -= lr_eff * ADAM_WD * w[i];
    }
}

/* ── Export weights (W0.bin + W1.bin + weights.meta) ───────────── */
static int export_weights(const ki_LinearLayer *l0, const ki_LinearLayer *l1,
                           const char *dir) {
    /* Create directory */
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", dir);
    if (system(cmd) != 0) {
        fprintf(stderr, "[ERROR] Cannot create directory %s\n", dir);
        return -1;
    }

    /* Write metadata */
    char path[512];
    snprintf(path, sizeof(path), "%s/weights.meta", dir);
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "[ERROR] Cannot write %s\n", path); return -1; }
    fprintf(f, "%d\n%d %d\n%d %d\n", 2, l0->out_features, l0->in_features,
            l1->out_features, l1->in_features);
    fclose(f);

    /* Write W0.bin */
    snprintf(path, sizeof(path), "%s/W0.bin", dir);
    f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[ERROR] Cannot write %s\n", path); return -1; }
    fwrite(l0->W, sizeof(float), (size_t)l0->out_features * (size_t)l0->in_features, f);
    fclose(f);

    /* Write W1.bin */
    snprintf(path, sizeof(path), "%s/W1.bin", dir);
    f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[ERROR] Cannot write %s\n", path); return -1; }
    fwrite(l1->W, sizeof(float), (size_t)l1->out_features * (size_t)l1->in_features, f);
    fclose(f);

    printf("  Exported model to %s/\n", dir);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════ */

ki_Args aa = {
    .hidden      = 64,
    .epochs      = 1,
    .batchN      = 64,
    .trainN      = 50000,
    .evalN       = 10000,
    .seed        = 42,
    .lr          = ADAMW_DEFAULT_LR,/* AdamW default (0.005 vs Otto's 0.05) */
    .threadN     = 8,
    .warmup_epochs = 2,
    .step_power  = 0.7f,
    .ensembleN   = 1,
    .splitVN     = 1,
    .splitHN     = 1,
    .channel = KI_DEFAULT_COLOR,
    .packedB= 1,
};

int main(int argc, char *argv[]) {
    ki_parse_args(argc, argv);  /* --lr overrides */

    int nhidden = aa.hidden;
    int epochs  = aa.epochs;
    int nc;  /* set below: block count × KI_NC */
    /* Container count = popcount(debug_mask) × (256|1024) per block */
    int dbg_mask = (aa.channel >= 0) ? aa.channel : 0x07;  /* default: r+g+b */
    int dbg_packed = aa.packedB;  /* 1=4px/cont, 0=1px/cont */
    { int n = 0; for (int b = 0; b < 8; b++) if (dbg_mask & (1 << b)) n++; nc = n * (dbg_packed ? KI_NC : (KI_PX / 3)); }

    /* ── Dry-run ──────────────────────────────────────────────── */
    if (aa.dry_run) {
        printf("══╡ Float AdamW W1-Only (CIFAR-10) ╞══\n");
        printf("  %-18s %-12d  %s\n",  "input",      KI_PX,      "pixels (32×32×3)");
        printf("  %-18s %-12d  %s\n",  "hidden",     nhidden,    "neurons");
        printf("  %-18s %-12d  %s\n",  "output",     N_CLASSES,  "classes");
        printf("  %-18s %-12d  %s\n",  "W1 params",  N_CLASSES * nhidden, "trainable (AdamW)");
        printf("  %-18s %-12d  %s\n",  "W0 params",  nc * nhidden, "frozen (random kaiming)");
        {
            char dbuf[128] = "";
            for (int b = 0; b < 8; b++)
                if (dbg_mask & (1 << b)) {
                    char tag[8] = "?";
                    if (b==0) { snprintf(tag,8,"r"); }
                    else if (b==1) { snprintf(tag,8,"g"); }
                    else if (b==2) { snprintf(tag,8,"b"); }
                    else if (b==3) { snprintf(tag,8,"y"); }
                    else if (b==4) { snprintf(tag,8,"lum"); }
                    else if (b==5) { snprintf(tag,8,"rg"); }
                    else if (b==6) { snprintf(tag,8,"by"); }
                    else if (b==7) { snprintf(tag,8,"yl"); }
                    size_t sl = strlen(dbuf);
                    snprintf(dbuf+sl, sizeof(dbuf)-sl, "%s+", tag);
                }
            if (strlen(dbuf) > 0) dbuf[strlen(dbuf)-1] = '\0'; /* remove trailing + */
            printf("  %-18s %-12d  %s\n",  "NC",         nc, dbuf);
            printf("  %-18s %s\n",  "packing",
                   dbg_packed ? "4px/cont (packed)" : "1px/cont (full)");
        }
        printf("  %-18s %-12s  %s\n",  "optimizer",  "AdamW",     "beta1=0.9 beta2=0.999 wd=1e-4");
        printf("  %-18s %-12s  %s\n",  "schedule",   "warmup+cos","linear warmup + cosine decay");
        return 0;
    }

    /* ── Load CIFAR-10 ────────────────────────────────────────── */
    ki_ImageData data;
    if (ki_cifar_read(&data) != 0) return 1;
    if (data.pixels != KI_PX) {
        fprintf(stderr, "[FATAL] Expected %d pixels, got %d\n", KI_PX, data.pixels);
        ki_cifar_free(&data);
        return 1;
    }

    /* ── Pack input: 768 (3×256 color) or 1024 (luminance) ─────── */
    int total_train = aa.trainN;
    int total_eval  = aa.evalN;
    if (total_train + total_eval > data.num_images) {
        total_eval = data.num_images - total_train;
        if (total_eval < 0) { total_eval = 0; total_train = data.num_images; }
    }
    /* Pack input: blocks selected by --channels mask */
    float *X_all = ki_pack_blocks_float(data.X_raw, total_train + total_eval, dbg_mask, dbg_packed);
    float *X_tr  = X_all;
    float *X_te  = X_all + (size_t)total_train * (size_t)nc;
    uint8_t *y_tr = data.y;
    uint8_t *y_te = data.y + total_train;

    /* ── Architecture display ─────────────────────────────────── */
    ki_SetupInfo si = {
        .title = "Float AdamW W1-Only (CIFAR-10)",
        .H = nhidden, .epochs = epochs,
        .bits_per_cont = KI_BITS_PER_CONT,
        .pixel_bits = 8,
        .seed = aa.seed, .N = total_train, .ne = total_eval,
        .n_threads = aa.threadN, .px = KI_PX,
        .sizeof_bn = (int)sizeof(float),
        .nc = nc, .C = nc,
        .input_bit  = (size_t)nc * KI_BITS_PER_CONT,
        .hidden_bit = (size_t)nhidden * KI_BITS_PER_CONT,
        .output_bit = (size_t)N_CLASSES * KI_BITS_PER_CONT,
        .w0_bit     = (size_t)nhidden * (size_t)nc * KI_BITS_PER_CONT,
        .w1_bit     = (size_t)N_CLASSES * (size_t)nhidden * KI_BITS_PER_CONT,
    };
    printf("══╡ CIFAR-10 Float AdamW W1-Only  W0=frozen  w1=AdamW ╞══  H=%-4d  Ep=%-2d  NC=%-3d\n",
           nhidden, epochs, nc);
    ki_setup_show(&si);

    int eff_warmup = aa.warmup_epochs;
    float lr_min_f = aa.lr_min;
    printf("══╡ TRAINING ╞══  lr=%.6f  lr-min=%.6f  decay=%s  batch=%d  warmup=%d  NC=%d (1px/cont)\n",
           (double)aa.lr, (double)lr_min_f,
           (aa.step_mode == STEP_CONST) ? "const" : "cosine",
           aa.batchN, eff_warmup, nc);

    /* ── Create model ─────────────────────────────────────────── */
    ki_LinearLayer l0, l1;

    l0.in_features = nc;
    l0.out_features = nhidden;
    l0.W = (float *)ki_xcalloc((size_t)nhidden * (size_t)nc, sizeof(float));

    l1.in_features = nhidden;
    l1.out_features = N_CLASSES;
    l1.W = (float *)ki_xcalloc((size_t)N_CLASSES * (size_t)nhidden, sizeof(float));

    srand(aa.seed);
    ki_init_kaiming(&l0, aa.seed);
    srand(aa.seed);
    ki_init_kaiming(&l1, aa.seed);

    /* ── AdamW state for W1 (W0 is frozen) ────────────────────── */
    size_t w1_n = (size_t)N_CLASSES * (size_t)nhidden;
    ki_AdamWState adamw = ki_adamw_create(w1_n, aa.lr, lr_min_f);

    /* Best-weight snapshots */
    float *best_W0 = (float *)ki_xmalloc((size_t)nhidden * (size_t)nc * sizeof(float));
    float *best_W1 = (float *)ki_xmalloc(w1_n * sizeof(float));
    float best_eval_acc = 0.0f;

    /* Batch buffers */
    int *idx = (int *)ki_xmalloc((size_t)total_train * sizeof(int));
    float *grad_w1 = (float *)ki_xmalloc(w1_n * sizeof(float));
    float *h0_buf = (float *)ki_xmalloc((size_t)aa.batchN * (size_t)nhidden * sizeof(float));
    float *out_buf = (float *)ki_xmalloc((size_t)aa.batchN * (size_t)N_CLASSES * sizeof(float));

    /* ── Training loop ────────────────────────────────────────── */
    double best_train = 0.0;
    float last_lr = 0.0f;          /* see below: capture for REPORT */
    struct timeval tv_start;
    gettimeofday(&tv_start, NULL);

    for (int ep = 0; ep < epochs; ep++) {
        float lr_ep = ki_lr_schedule(ep, epochs, eff_warmup,
                                      aa.lr, lr_min_f, (aa.step_mode == STEP_CONST));
        adamw.lr = lr_ep;

        /* Shuffle indices */
        for (int i = 0; i < total_train; i++) idx[i] = i;
        ki_shuffle(idx, total_train);

        int n_batches = (total_train + aa.batchN - 1) / aa.batchN;
        double epoch_loss = 0.0;
        int epoch_correct = 0;

        for (int b = 0; b < n_batches; b++) {
            int start = b * aa.batchN;
            int bs = (start + aa.batchN <= total_train) ? aa.batchN : (total_train - start);
            if (bs <= 0) break;

            /* Build batch */
            float *xb = (float *)malloc((size_t)bs * (size_t)nc * sizeof(float));
            for (int s = 0; s < bs; s++) {
                int sample_idx = idx[start + s];
                memcpy(&xb[(size_t)s * (size_t)nc], &X_tr[(size_t)sample_idx * (size_t)nc], (size_t)nc * sizeof(float));
            }

            /* Forward */
            forward(&l0, &l1, xb, h0_buf, out_buf, bs);

            /* MSE loss + gradient */
            memset(grad_w1, 0, w1_n * sizeof(float));
            float batch_loss = 0.0f;
            int batch_ok = 0;

            for (int s = 0; s < bs; s++) {
                int si2 = idx[start + s];
                int pred = 0;
                for (int k = 1; k < N_CLASSES; k++)
                    if (out_buf[s * N_CLASSES + k] > out_buf[s * N_CLASSES + pred])
                        pred = k;
                if (pred == (int)y_tr[si2]) batch_ok++;

                for (int k = 0; k < N_CLASSES; k++) {
                    float target = (float)(k == y_tr[si2] ? 1 : -1);
                    float err = out_buf[s * N_CLASSES + k] - target;
                    batch_loss += err * err;
                    float grad_out = 2.0f * err;
                    /* Gradient w.r.t. W1[k][h] = grad_out * h0[h]
                     * (LReLU already applied before W1, so h0 is the input to W1) */
                    float *h0_s = h0_buf + (size_t)s * (size_t)nhidden;
                    for (int h = 0; h < nhidden; h++) {
                        float gw = grad_out * h0_s[h];
                        grad_w1[(size_t)k * (size_t)nhidden + (size_t)h] += gw / (float)bs;
                    }
                }
            }

            float avg_loss = batch_loss / (float)(bs * N_CLASSES);
            epoch_loss += avg_loss;
            epoch_correct += batch_ok;

            /* Gradient clipping (max_norm=1.0) */
            float gnorm = 0.0f;
            for (size_t i = 0; i < w1_n; i++) gnorm += grad_w1[i] * grad_w1[i];
            gnorm = sqrtf(gnorm);
            float scale = (gnorm > 1.0f) ? 1.0f / gnorm : 1.0f;
            for (size_t i = 0; i < w1_n; i++) grad_w1[i] *= scale;

            /* AdamW update (W1 only) */
            ki_adamw_update(&adamw, l1.W, grad_w1, w1_n);

            free(xb);
        }

        if (total_eval > 0) {
            float eval_acc = accuracy_pct(X_te, y_te, total_eval, &l0, &l1, nc);
            if (eval_acc > best_eval_acc) {
                best_eval_acc = eval_acc;
                memcpy(best_W0, l0.W, (size_t)nhidden * (size_t)nc * sizeof(float));
                memcpy(best_W1, l1.W, w1_n * sizeof(float));
            }
            double elapsed = 0.0;
            struct timeval tv_now;
            gettimeofday(&tv_now, NULL);
            elapsed = (double)(tv_now.tv_sec - tv_start.tv_sec)
                    + (double)(tv_now.tv_usec - tv_start.tv_usec) / 1e6;
            last_lr = lr_ep;
            printf("  Ep %2d/%d  loss=%.4f  train=%.1f%%  eval=%.1f%%  lr=%.6f  time=%.0fs\n",
                   ep + 1, epochs, epoch_loss / (double)n_batches,
                   100.0 * (double)epoch_correct / (double)total_train,
                   (double)eval_acc, (double)lr_ep, elapsed);
            best_train = 100.0 * (double)epoch_correct / (double)total_train;
        } else {
            printf("  Ep %2d/%d  loss=%.4f  train=%.1f%%\n",
                   ep + 1, epochs, epoch_loss / (double)n_batches,
                   100.0 * (double)epoch_correct / (double)total_train);
        }
    }

    /* ── Export best model ─────────────────────────────────────── */
    memcpy(l0.W, best_W0, (size_t)nhidden * (size_t)nc * sizeof(float));
    memcpy(l1.W, best_W1, w1_n * sizeof(float));

    char export_dir[256];
    if (aa.out[0] != '\0') {
        snprintf(export_dir, sizeof(export_dir), "%s", aa.out);
    } else {
        /* Convention: adam1-h{hiddenN}-b{batchN}-e{epochsN} */
        snprintf(export_dir, sizeof(export_dir), "%s/adam1-h%d-b%d-e%d",
                 KI_MODEL_DIR, nhidden, aa.batchN, epochs);
    }
    export_weights(&l0, &l1, export_dir);

    /* ── Final report ─────────────────────────────────────────── */
    struct timeval tv_end;
    gettimeofday(&tv_end, NULL);
    int elapsed = (int)((tv_end.tv_sec - tv_start.tv_sec) * 1000
                      + (tv_end.tv_usec - tv_start.tv_usec) / 1000);

    float final_acc = accuracy_pct(X_te, y_te, total_eval, &l0, &l1, nc);
    printf("\n══╡ RESULT ╞══════════════════════════════════════════════════════\n");
    printf("  Best eval: %.1f%%  Final eval: %.1f%%\n", (double)best_eval_acc, (double)final_acc);

    ki_report_show((int)(best_train * (float)total_train / 100.0f + 0.5f),
                   total_train,
                   (int)(final_acc * (float)total_eval / 100.0f + 0.5f),
                   total_eval,
                   elapsed, aa.threadN, 0, last_lr);

    /* ── Cleanup ──────────────────────────────────────────────── */
    free(X_all);
    free(best_W0); free(best_W1);
    free(idx); free(grad_w1); free(h0_buf); free(out_buf);
    free(l0.W); free(l1.W);
    free(adamw.m); free(adamw.v);
    ki_cifar_free(&data);
    return 0;
}
