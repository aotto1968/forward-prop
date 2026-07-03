/*
 * cifar-1/mlp-flt32-trn-2layer-adam.c — Float32 2-Layer AdamW (CIFAR-10)
 * ======================================================================
 *
 * Two TRAINABLE layers (W0 + W1), both AdamW. No frozen weights.
 * Forward:  matmul(W0, x) → LReLU(0.05) → matmul(W1, h0) → output
 * Loss:     MSE ±1 over 10 classes
 * Optimizer: AdamW(lr, wd=1e-4) for BOTH layers
 * Schedule:  warmup + cosine decay
 *
 * This is the proper float32 baseline for CIFAR-10 MLP.
 * W1-only training plateaued at ~45% — both layers needed.
 */
#include "ki-common.h"
#include "ki-adamw.h"

#define N_CLASSES KI_NCLASSES
#define ADAM_BETA1 0.9f
#define ADAM_BETA2 0.999f
#define ADAM_EPS   1e-8f
#define ADAM_WD    1e-4f

/* ── AdamW state (per layer) ───────────────────────────────────── */
typedef struct {
    float lr, lr_min;
    float *m, *v;
    int t;
    size_t n;
} ki_AdamWLayer;

static inline ki_AdamWLayer ki_adamw_create(size_t n, float lr, float lr_min) {
    ki_AdamWLayer s;
    s.lr = lr; s.lr_min = lr_min; s.t = 0; s.n = n;
    s.m = (float *)ki_xcalloc(n, sizeof(float));
    s.v = (float *)ki_xcalloc(n, sizeof(float));
    return s;
}

static inline void ki_adamw_update(ki_AdamWLayer *s, float *w, const float *g) {
    s->t++;
    float lr_t = s->lr * sqrtf(1.0f - powf(ADAM_BETA2, (float)s->t))
                           / (1.0f - powf(ADAM_BETA1, (float)s->t));
    float lr_eff = fmaxf(lr_t, s->lr_min);
    #pragma omp parallel for
    for (size_t i = 0; i < s->n; i++) {
        s->m[i] = ADAM_BETA1 * s->m[i] + (1.0f - ADAM_BETA1) * g[i];
        s->v[i] = ADAM_BETA2 * s->v[i] + (1.0f - ADAM_BETA2) * g[i] * g[i];
        float m_hat = s->m[i] / (1.0f - powf(ADAM_BETA1, (float)s->t));
        float v_hat = s->v[i] / (1.0f - powf(ADAM_BETA2, (float)s->t));
        w[i] -= lr_eff * m_hat / (sqrtf(v_hat) + ADAM_EPS);
        w[i] -= lr_eff * ADAM_WD * w[i];
    }
}

/* ── Forward ──────────────────────────────────────────────────────
 *  h0_pre = W0 @ x (stored for backprop derivative)
 *  h0     = LReLU(h0_pre)
 *  out    = W1 @ h0
 *
 *  h0_pre and h0 share the same buffer (pre+act stored interleaved).
 *  Layout: [b][h] = h0_pre[b*H + h], h0[b*H + h] = LReLU(h0_pre)
 *  We store both to avoid recomputing W0 @ x in backward.
 */
static void forward(const float *x, int batchN, int nc, int H,
                     const float *W0, const float *W1,
                     float *h0_pre, float *h0_act, float *out_buf) {
    /* h0_pre = W0 @ x */
    #pragma omp parallel for collapse(2) if(batchN * H >= 64)
    for (int b = 0; b < batchN; b++) {
        for (int h = 0; h < H; h++) {
            float acc = 0.0f;
            for (int c = 0; c < nc; c++)
                acc += x[b * nc + c] * W0[(size_t)h * (size_t)nc + (size_t)c];
            h0_pre[b * H + h] = acc;
        }
    }

    /* LReLU: compute activation, store separately */
    #pragma omp parallel for if(batchN * H >= 256)
    for (int i = 0; i < batchN * H; i++) {
        float v = h0_pre[i];
        h0_act[i] = (v > 0.0f) ? v : v * 0.05f;
    }

    /* out = W1 @ h0_act */
    #pragma omp parallel for collapse(2) if(batchN * N_CLASSES >= 64)
    for (int b = 0; b < batchN; b++) {
        for (int k = 0; k < N_CLASSES; k++) {
            float acc = 0.0f;
            for (int h = 0; h < H; h++)
                acc += h0_act[b * H + h] * W1[(size_t)k * (size_t)H + (size_t)h];
            out_buf[b * N_CLASSES + k] = acc;
        }
    }
}

/* ── Accuracy ──────────────────────────────────────────────────── */
static float accuracy_pct(const float *X, const uint8_t *Y, int N,
                           int nc, int H,
                           const float *W0, const float *W1) {
    int ok = 0;
    #pragma omp parallel reduction(+:ok)
    {
        float *h0_pre = (float *)malloc((size_t)H * sizeof(float));
        float *h0_act = (float *)malloc((size_t)H * sizeof(float));
        float *out = (float *)malloc((size_t)N_CLASSES * sizeof(float));
        if (!h0_pre || !h0_act || !out) { free(h0_pre); free(h0_act); free(out); }
        else {
            #pragma omp for schedule(static)
            for (int si = 0; si < N; si++) {
                const float *x = X + (size_t)si * (size_t)nc;
                forward(x, 1, nc, H, W0, W1, h0_pre, h0_act, out);
                int pred = 0;
                for (int k = 1; k < N_CLASSES; k++)
                    if (out[k] > out[pred]) pred = k;
                if (pred == (int)Y[si]) ok++;
            }
            free(h0_pre); free(h0_act); free(out);
        }
    }
    return 100.0f * (float)ok / (float)N;
}

/* ── Export weights ────────────────────────────────────────────── */
static int export_weights(int nc, int H, const float *W0, const float *W1,
                           const char *dir) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", dir);
    if (system(cmd) != 0) return -1;

    char path[512];
    snprintf(path, sizeof(path), "%s/weights.meta", dir);
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "%d\n%d %d\n%d %d\n", 2, H, nc, N_CLASSES, H);
    fclose(f);

    snprintf(path, sizeof(path), "%s/W0.bin", dir);
    f = fopen(path, "wb");
    if (!f) return -1;
    fwrite(W0, sizeof(float), (size_t)H * (size_t)nc, f);
    fclose(f);

    snprintf(path, sizeof(path), "%s/W1.bin", dir);
    f = fopen(path, "wb");
    if (!f) return -1;
    fwrite(W1, sizeof(float), (size_t)N_CLASSES * (size_t)H, f);
    fclose(f);

    printf("  Exported model -> %s/\n", dir);
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
        printf("══╡ Float32 2-Layer AdamW (CIFAR-10) ╞══\n");
        printf("  %-18s %-12d  %s\n",  "input",      KI_PX,      "pixels (32x32x3)");
        printf("  %-18s %-12d  %s\n",  "hidden",     nhidden,    "neurons");
        printf("  %-18s %-12d  %s\n",  "output",     N_CLASSES,  "classes");
        printf("  %-18s %-12d  %s\n",  "W0 params",  nc * nhidden, "trainable (AdamW)");
        printf("  %-18s %-12d  %s\n",  "W1 params",  N_CLASSES * nhidden, "trainable (AdamW)");
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
            if (strlen(dbuf) > 0) dbuf[strlen(dbuf)-1] = '\0';
            printf("  %-18s %-12d  %s\n",  "NC",         nc, dbuf);
            printf("  %-18s %s\n",  "packing",
                   dbg_packed ? "4px/cont (packed)" : "1px/cont (full)");
        }
        printf("  %-18s %-12s  %s\n",  "optimizer",  "AdamW",     "both layers, wd=1e-4");
        return 0;
    }

    /* ── Load CIFAR-10 ────────────────────────────────────────── */
    ki_ImageData data;
    if (ki_cifar_read(&data) != 0) return 1;
    if (data.pixels != KI_PX) {
        fprintf(stderr, "[FATAL] Expected %d pixels, got %d\n", KI_PX, data.pixels);
        ki_cifar_free(&data); return 1;
    }

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

    /* ── Architecture ─────────────────────────────────────────── */
    size_t w0_n = (size_t)nhidden * (size_t)nc;
    size_t w1_n = (size_t)N_CLASSES * (size_t)nhidden;

    printf("══╡ CIFAR-10 Float32 2-Layer AdamW ╞══  H=%-4d  Ep=%-2d  NC=%-3d  params=%.1fK\n",
           nhidden, epochs, nc,
           (double)(w0_n + w1_n) / 1000.0);

    ki_SetupInfo si = {
        .title = "Float32 2-Layer AdamW (CIFAR-10)",
        .H = nhidden, .epochs = epochs,
        .bits_per_cont = KI_BITS_PER_CONT,
        .pixel_bits = 8, .seed = aa.seed,
        .N = total_train, .ne = total_eval,
        .n_threads = aa.threadN, .px = KI_PX,
        .sizeof_bn = (int)sizeof(float),
        .nc = nc, .C = nc,
        .input_bit  = (size_t)nc * KI_BITS_PER_CONT,
        .hidden_bit = (size_t)nhidden * KI_BITS_PER_CONT,
        .output_bit = (size_t)N_CLASSES * KI_BITS_PER_CONT,
        .w0_bit     = w0_n * KI_BITS_PER_CONT,
        .w1_bit     = w1_n * KI_BITS_PER_CONT,
    };
    ki_setup_show(&si);

    int eff_warmup = aa.warmup_epochs;
    float lr_min_f = aa.lr_min;

    /* ── Init weights (kaiming_uniform) ────────────────────────── */
    float *W0 = (float *)ki_xmalloc(w0_n * sizeof(float));
    float *W1 = (float *)ki_xmalloc(w1_n * sizeof(float));
    srand(aa.seed);
    {
        float bound0 = 1.0f / sqrtf((float)nc);
        for (size_t i = 0; i < w0_n; i++)
            W0[i] = (float)rand() / (float)RAND_MAX * 2.0f * bound0 - bound0;
    }
    {
        float bound1 = 1.0f / sqrtf((float)nhidden);
        for (size_t i = 0; i < w1_n; i++)
            W1[i] = (float)rand() / (float)RAND_MAX * 2.0f * bound1 - bound1;
    }

    /* ── AdamW state for BOTH layers ──────────────────────────── */
    ki_AdamWLayer adamw0 = ki_adamw_create(w0_n, aa.lr, lr_min_f);
    ki_AdamWLayer adamw1 = ki_adamw_create(w1_n, aa.lr, lr_min_f);

    /* Best-weight snapshots */
    float *best_W0 = (float *)ki_xmalloc(w0_n * sizeof(float));
    float *best_W1 = (float *)ki_xmalloc(w1_n * sizeof(float));
    float best_eval_acc = 0.0f;
    float best_train_acc = 0.0f;

    /* Batch buffers */
    int *idx = (int *)ki_xmalloc((size_t)total_train * sizeof(int));
    float *grad_w0 = (float *)ki_xmalloc(w0_n * sizeof(float));
    float *grad_w1 = (float *)ki_xmalloc(w1_n * sizeof(float));
    size_t batch_hidden_sz = (size_t)aa.batchN * (size_t)nhidden;
    size_t batch_out_sz = (size_t)aa.batchN * (size_t)N_CLASSES;
    float *h0_buf  = (float *)ki_xmalloc(batch_hidden_sz * sizeof(float));
    float *out_buf = (float *)ki_xmalloc(batch_out_sz * sizeof(float));

    /* ── Training loop ────────────────────────────────────────── */
    float last_lr = 0.0f;
    struct timeval tv_start;
    gettimeofday(&tv_start, NULL);

    for (int ep = 0; ep < epochs; ep++) {
        float lr_ep = ki_lr_schedule(ep, epochs, eff_warmup,
                                       aa.lr, lr_min_f, (aa.step_mode == STEP_CONST));
        adamw0.lr = lr_ep;
        adamw1.lr = lr_ep;

        /* Shuffle */
        for (int i = 0; i < total_train; i++) idx[i] = i;
        ki_shuffle(idx, total_train);

        int n_batches = (total_train + aa.batchN - 1) / aa.batchN;
        double epoch_loss = 0.0;
        int epoch_correct = 0;

        for (int b = 0; b < n_batches; b++) {
            int start = b * aa.batchN;
            int bs = (start + aa.batchN <= total_train) ? aa.batchN : (total_train - start);
            if (bs <= 0) break;

            /* Build batch input */
            size_t nc_sz = (size_t)nc;
            size_t H_sz = (size_t)nhidden;
            float *xb = (float *)malloc((size_t)bs * nc_sz * sizeof(float));
            for (int s = 0; s < bs; s++) {
                int samp_idx = idx[start + s];
                memcpy(&xb[(size_t)s * nc_sz], &X_tr[(size_t)samp_idx * nc_sz],
                       nc_sz * sizeof(float));
            }

            /* ── Forward ─────────────────────────────────────── */
            /* Allocate separate buffers for h0_pre and h0_act */
            float *h0_pre_buf = (float *)malloc(batch_hidden_sz * sizeof(float));
            forward(xb, bs, nc, nhidden, W0, W1, h0_pre_buf, h0_buf, out_buf);

            /* ── Backward (OpenMP parallel) ───────────────────── */
            memset(grad_w0, 0, w0_n * sizeof(float));
            memset(grad_w1, 0, w1_n * sizeof(float));

            /* Per-thread gradient buffers for race-free accumulation */
            int n_threads = omp_get_max_threads();
            float **grad_w0_t = (float **)malloc((size_t)n_threads * sizeof(float *));
            float **grad_w1_t = (float **)malloc((size_t)n_threads * sizeof(float *));
            for (int t = 0; t < n_threads; t++) {
                grad_w0_t[t] = (float *)calloc(w0_n, sizeof(float));
                grad_w1_t[t] = (float *)calloc(w1_n, sizeof(float));
            }

            float batch_loss = 0.0f;
            int batch_ok = 0;

            #pragma omp parallel reduction(+:batch_loss) reduction(+:batch_ok)
            {
                int tid = omp_get_thread_num();

                #pragma omp for schedule(static)
                for (int s = 0; s < bs; s++) {
                    size_t s_sz = (size_t)s;
                    int label = y_tr[idx[start + s]];

                    /* Prediction */
                    int pred = 0;
                    for (int k = 1; k < N_CLASSES; k++)
                        if (out_buf[s * N_CLASSES + k] > out_buf[s * N_CLASSES + pred])
                            pred = k;
                    if (pred == label) batch_ok++;

                    /* MSE gradient: grad_out[k] = 2*(out - target)/bs */
                    float grad_out[N_CLASSES];
                    for (int k = 0; k < N_CLASSES; k++) {
                        float target = (k == label) ? 1.0f : -1.0f;
                        float err = out_buf[s * N_CLASSES + k] - target;
                        batch_loss += err * err;
                        grad_out[k] = 2.0f * err / (float)bs;
                    }

                    /* h0_act for this sample */
                    float *h0_s = h0_buf + s_sz * H_sz;

                    /* ── W1 gradient (thread-local) ─────────── */
                    float *gw1_t = grad_w1_t[tid];
                    for (int k = 0; k < N_CLASSES; k++) {
                        float go = grad_out[k];
                        size_t base = (size_t)k * H_sz;
                        for (int h = 0; h < nhidden; h++)
                            gw1_t[base + (size_t)h] += go * h0_s[h];
                    }

                    /* ── Backprop through W1 ─────────────────── */
                    float dh0_pre[4096];
                    for (int h = 0; h < nhidden; h++) {
                        float acc = 0.0f;
                        for (int k = 0; k < N_CLASSES; k++)
                            acc += W1[(size_t)k * H_sz + (size_t)h] * grad_out[k];
                        dh0_pre[h] = acc;
                    }

                    /* ── LReLU derivative (use cached h0_pre) ─ */
                    float *h0p_s = h0_pre_buf + s_sz * H_sz;
                    for (int h = 0; h < nhidden; h++)
                        dh0_pre[h] *= (h0p_s[h] > 0.0f) ? 1.0f : 0.05f;

                    /* ── W0 gradient (thread-local) ─────────── */
                    const float *x_s = xb + s_sz * nc_sz;
                    float *gw0_t = grad_w0_t[tid];
                    for (int h = 0; h < nhidden; h++) {
                        size_t base = (size_t)h * nc_sz;
                        float dh = dh0_pre[h];
                        for (int c = 0; c < nc; c++)
                            gw0_t[base + (size_t)c] += dh * x_s[(size_t)c];
                    }
                }
            }

            /* Merge per-thread gradients into global arrays */
            for (int t = 0; t < n_threads; t++) {
                for (size_t i = 0; i < w0_n; i++) grad_w0[i] += grad_w0_t[t][i];
                for (size_t i = 0; i < w1_n; i++) grad_w1[i] += grad_w1_t[t][i];
                free(grad_w0_t[t]);
                free(grad_w1_t[t]);
            }
            free(grad_w0_t);
            free(grad_w1_t);

            free(h0_pre_buf);

            float avg_loss = batch_loss / (float)(bs * N_CLASSES);
            epoch_loss += avg_loss;
            epoch_correct += batch_ok;

            /* Gradient clipping (max_norm=1.0) for both layers */
            float gnorm0 = 0.0f, gnorm1 = 0.0f;
            for (size_t i = 0; i < w0_n; i++) gnorm0 += grad_w0[i] * grad_w0[i];
            for (size_t i = 0; i < w1_n; i++) gnorm1 += grad_w1[i] * grad_w1[i];
            float gnorm = sqrtf(gnorm0 + gnorm1);
            float scale = (gnorm > 1.0f) ? 1.0f / gnorm : 1.0f;
            if (scale < 1.0f) {
                for (size_t i = 0; i < w0_n; i++) grad_w0[i] *= scale;
                for (size_t i = 0; i < w1_n; i++) grad_w1[i] *= scale;
            }

            /* AdamW update (BOTH layers) */
            ki_adamw_update(&adamw0, W0, grad_w0);
            ki_adamw_update(&adamw1, W1, grad_w1);

            free(xb);

        } /* end batch loop */

        /* ── Eval ─────────────────────────────────────────────── */
        float train_acc = 100.0f * (float)epoch_correct / (float)total_train;
        if (train_acc > best_train_acc) best_train_acc = train_acc;

        if (total_eval > 0) {
            float eval_acc = accuracy_pct(X_te, y_te, total_eval, nc, nhidden, W0, W1);
            last_lr = lr_ep;
            if (eval_acc > best_eval_acc) {
                best_eval_acc = eval_acc;
                memcpy(best_W0, W0, w0_n * sizeof(float));
                memcpy(best_W1, W1, w1_n * sizeof(float));
            }
            struct timeval tv_now;
            gettimeofday(&tv_now, NULL);
            double elapsed = (double)(tv_now.tv_sec - tv_start.tv_sec)
                           + (double)(tv_now.tv_usec - tv_start.tv_usec) / 1e6;
            printf("  Ep %2d/%-2d  loss=%.4f  train=%.1f%%  eval=%.1f%%  lr=%.6f  time=%ds\n",
                   ep + 1, epochs, epoch_loss / (double)n_batches,
                   (double)train_acc, (double)eval_acc, (double)lr_ep, (int)elapsed);
        }

    } /* end epoch loop */

    /* ── Export best model ─────────────────────────────────────── */
    memcpy(W0, best_W0, w0_n * sizeof(float));
    memcpy(W1, best_W1, w1_n * sizeof(float));

    char export_dir[256];
    if (aa.out[0] != '\0')
        snprintf(export_dir, sizeof(export_dir), "%s", aa.out);
    else
        /* Convention: adam2-h{hiddenN}-b{batchN}-e{epochsN} */
        snprintf(export_dir, sizeof(export_dir), "%s/adam2-h%d-b%d-e%d",
                 KI_MODEL_DIR, nhidden, aa.batchN, epochs);
    export_weights(nc, nhidden, W0, W1, export_dir);

    /* ── Final report ─────────────────────────────────────────── */
    struct timeval tv_end;
    gettimeofday(&tv_end, NULL);
    int elapsed_ms = (int)((tv_end.tv_sec - tv_start.tv_sec) * 1000
                         + (tv_end.tv_usec - tv_start.tv_usec) / 1000);

    float final_acc = accuracy_pct(X_te, y_te, total_eval, nc, nhidden, W0, W1);
    printf("\n══╡ RESULT ╞══════════════════════════════════════════════════════\n");
    printf("  Best eval: %.1f%%  Final eval: %.1f%%\n", (double)best_eval_acc, (double)final_acc);

    ki_report_show((int)(best_train_acc * (float)total_train / 100.0f + 0.5f),
                   total_train,
                   (int)(final_acc * (float)total_eval / 100.0f + 0.5f),
                   total_eval,
                   elapsed_ms, aa.threadN, 0, last_lr);

    /* ── Cleanup ──────────────────────────────────────────────── */
    free(X_all);
    free(W0); free(W1);
    free(best_W0); free(best_W1);
    free(idx); free(grad_w0); free(grad_w1);
    free(h0_buf); free(out_buf);
    free(adamw0.m); free(adamw0.v);
    free(adamw1.m); free(adamw1.v);
    ki_cifar_free(&data);
    return 0;
}
