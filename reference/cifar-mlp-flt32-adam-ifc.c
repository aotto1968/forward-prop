/*
 * cifar-1/mlp-flt32-adam-ifc.c — Float32 2-Layer AdamW Inference (CIFAR-10)
 * =====================================================================
 *
 * Loads a trained 2-layer float32 model (exported by mlp-flt32-w1-adam-trn)
 * and classifies CIFAR-10 test data.
 *
 * Derived from otto-score-ifc/mlp-flt32-adam-ifc.c
 */
/* ki-adamw.h includes the project's current ki-common.h via cifar-include/ */
#include <ki-adamw.h>
#include <inttypes.h>

#define N_CLASSES KI_NCLASSES

/* ── Model structure ───────────────────────────────────────────── */
typedef struct {
    int      H;           /* Hidden neurons */
    int      in_feat;     /* Input features (NC=768) */
    float   *W0;          /* [H × in_feat] frozen random projection */
    float   *W1;          /* [10 × H] AdamW-trained classifier */
} Flt32Model;

/* ── Model loader ──────────────────────────────────────────────── */
static Flt32Model *model_load(const char *dir) {
    char path[512];
    snprintf(path, sizeof(path), "%s/weights.meta", dir);

    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[FATAL] Cannot open %s\n", path);
        return NULL;
    }

    int n_layers, in_feat, out_feat, H;
    if (fscanf(f, "%d\n%d %d\n%d %d\n",
               &n_layers, &H, &in_feat, &out_feat, &H) != 5) {
        fprintf(stderr, "[FATAL] Bad format in %s (expected: 2-layer meta)\n", path);
        fclose(f);
        return NULL;
    }
    fclose(f);

    if (n_layers != 2 || out_feat != N_CLASSES) {
        fprintf(stderr, "[FATAL] Unexpected model shape: %d layers, %d outputs\n",
                n_layers, out_feat);
        return NULL;
    }

    Flt32Model *m = (Flt32Model *)malloc(sizeof(Flt32Model));
    if (!m) { fprintf(stderr, "[FATAL] Out of memory\n"); return NULL; }
    m->H = H;
    m->in_feat = in_feat;

    size_t n0 = (size_t)H * (size_t)in_feat;
    size_t n1 = (size_t)N_CLASSES * (size_t)H;

    m->W0 = (float *)malloc(n0 * sizeof(float));
    m->W1 = (float *)malloc(n1 * sizeof(float));
    if (!m->W0 || !m->W1) {
        free(m->W0); free(m->W1); free(m);
        fprintf(stderr, "[FATAL] Out of memory\n"); return NULL;
    }

    snprintf(path, sizeof(path), "%s/W0.bin", dir);
    f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[FATAL] Cannot open %s\n", path); free(m->W0); free(m->W1); free(m); return NULL; }
    if (fread(m->W0, sizeof(float), n0, f) != n0) { fprintf(stderr, "[FATAL] Short read %s\n", path); fclose(f); free(m->W0); free(m->W1); free(m); return NULL; }
    fclose(f);

    snprintf(path, sizeof(path), "%s/W1.bin", dir);
    f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[FATAL] Cannot open %s\n", path); free(m->W0); free(m->W1); free(m); return NULL; }
    if (fread(m->W1, sizeof(float), n1, f) != n1) { fprintf(stderr, "[FATAL] Short read %s\n", path); fclose(f); free(m->W0); free(m->W1); free(m); return NULL; }
    fclose(f);

    printf("  Model: H=%d  in_feat=%d  dir=%s\n", H, in_feat, dir);
    return m;
}

static void model_free(Flt32Model *m) {
    if (!m) return;
    free(m->W0); free(m->W1); free(m);
}

/* ── Accuracy (batch-wise, same as trainer) ─────────────────── */
static float eval_model(Flt32Model *m, const float *X, const uint8_t *Y, int N, int nc) {
    int ok = 0;
    int batchN = 128;
    int n_batches = (N + batchN - 1) / batchN;
    ki_LinearLayer l0 = {.in_features=nc, .out_features=m->H, .W=m->W0};
    ki_LinearLayer l1 = {.in_features=m->H, .out_features=N_CLASSES, .W=m->W1};
    for (int b = 0; b < n_batches; b++) {
        int start = b * batchN;
        int bs = (start + batchN <= N) ? batchN : (N - start);
        float *h0_buf = (float *)malloc((size_t)bs * (size_t)m->H * sizeof(float));
        float *out_buf = (float *)malloc((size_t)bs * (size_t)N_CLASSES * sizeof(float));
        if (!h0_buf || !out_buf) { free(h0_buf); free(out_buf); break; }
        ki_linear_forward(&l0, X + (size_t)start * (size_t)nc, h0_buf, bs);
        ki_leaky_relu(h0_buf, bs * m->H);
        ki_linear_forward(&l1, h0_buf, out_buf, bs);
        for (int s = 0; s < bs; s++) {
            int pred = 0;
            for (int k = 1; k < N_CLASSES; k++)
                if (out_buf[s * N_CLASSES + k] > out_buf[s * N_CLASSES + pred]) pred = k;
            if (pred == (int)Y[start + s]) ok++;
        }
        free(h0_buf); free(out_buf);
    }
    return 100.0f * (float)ok / (float)N;
}

/* ═══════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[]) {
    const char *model_dir = NULL;
    int evalN = 10000;
    int threadN = 8;
    int channel = -1;  /* default: 3-channel color */

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s --model DIR [options]\n", argv[0]);
            printf("  --model DIR   Model directory (with weights.meta + W0.bin + W1.bin)\n");
            printf("  --evalN N     CIFAR-10 eval samples (default: 10000)\n");
            printf("  --threadN N   OpenMP threads (default: 8)\n");
            printf("  --channels 601|709  Luminance input (default: 3-channel color)\n");
            return 0;
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "--evalN") == 0 && i + 1 < argc) {
            evalN = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threadN") == 0 && i + 1 < argc) {
            threadN = atoi(argv[++i]);
            if (threadN < 1) threadN = 1;
        } else if (strcmp(argv[i], "--channels") == 0 && i + 1 < argc) {
            const char *val = argv[++i];
            if (strcmp(val, "601") == 0) channel = 601;
            else if (strcmp(val, "709") == 0) channel = 709;
            else { fprintf(stderr, "[ERROR] --channels: expected 601 or 709, got '%s'\n", val); return 1; }
        } else {
            fprintf(stderr, "[ERROR] Unknown argument: %s\nTry --help\n", argv[i]);
            return 1;
        }
    }

    if (!model_dir) {
        fprintf(stderr, "[FATAL] --model DIR is required\n");
        return 1;
    }

    omp_set_num_threads(threadN);

    /* ── Load Model ────────────────────────────────────────────── */
    Flt32Model *model = model_load(model_dir);
    if (!model) return 1;

    int nc = model->in_feat;

    /* ── Load CIFAR-10 ─────────────────────────────────────────── */
    ki_ImageData data;
    if (ki_cifar_read(&data) != 0) { model_free(model); return 1; }
    if (data.pixels != KI_PX) {
        fprintf(stderr, "[FATAL] Expected %d pixels, got %d\n", KI_PX, data.pixels);
        ki_cifar_free(&data); model_free(model); return 1;
    }

    int total_eval = evalN;
    int offset = data.num_images - total_eval;
    if (offset < 0) { offset = 0; total_eval = data.num_images; }

    /* Pack input — CIFAR default: R+G+B (KI_DEFAULT_COLOR = 0x0E) */
    int dbg_mask = (channel >= 0) ? channel : KI_DEFAULT_COLOR;
    int dbg_packed = 1;  /* IFC knows packed from model */
    float *X_all = ki_pack_blocks_float(data.X_raw, data.num_images, dbg_mask, dbg_packed);
    float *X_te  = X_all + (size_t)offset * (size_t)nc;
    uint8_t *y_te = data.y + offset;

    printf("\n══╡ INFERENCE ╞══════════════════════════════════════════════════\n");
    printf("  Evaluating %d CIFAR-10 samples (offset=%d)\n", total_eval, offset);
    fflush(stdout);

    struct timeval tv0, tv1;
    gettimeofday(&tv0, NULL);

    float acc = eval_model(model, X_te, y_te, total_eval, nc);

    gettimeofday(&tv1, NULL);
    int elapsed = (int)((tv1.tv_sec - tv0.tv_sec) * 1000
                      + (tv1.tv_usec - tv0.tv_usec) / 1000);

    printf("\n══╡ RESULT ╞══════════════════════════════════════════════════════\n");
    printf("  Model:   H=%d  Float32 AdamW (CIFAR-10)\n", model->H);
    printf("  Eval:    %.1f%%  (%d/%d)\n",
           acc, (int)(acc * (float)total_eval / 100.0f + 0.5f), total_eval);
    printf("  Time:    %dms  (%.1f µs/sample)\n",
           elapsed, (double)elapsed * 1000.0 / (double)total_eval);

    ki_report_show(0, 0,
                   (int)(acc * (float)total_eval / 100.0f + 0.5f), total_eval,
                   elapsed, threadN, 0, 0.0f);

    free(X_all);
    ki_cifar_free(&data);
    model_free(model);
    return 0;
}
