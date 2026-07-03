/*
 * otto-score-ifc/mlp-flt32-adam-ifc.c — Float32 2-Layer AdamW Inference
 * =================================================================
 *
 * Loads a trained 2-layer float32 model (exported by mlp-flt32-w1-adam-trn)
 * and classifies MNIST test data or a single image.
 *
 * Model format (from export_weights):
 *   weights.meta   — "2\nH in_feat\n10 H\n"
 *   W0.bin         — float32[H × in_feat]  (W0, frozen random projection)
 *   W1.bin         — float32[10 × H]       (W1, AdamW-trained classifier)
 *
 * Forward: matmul(W0, x) → LReLU(0.05) → matmul(W1, h0) → argmax
 * No bitwise ops, no MAJ3 — pure float32 matmul baseline.
 *
 * Build:
 *   make mlp-flt32-adam-ifc-xnor.exe
 *
 * Usage:
 *   ./mlp-flt32-adam-ifc-xnor.exe --model DIR   [--evalN N]  [--image FILE]
 */
#include "ki-common.h"
#include <inttypes.h>

#define N_CLASSES KI_NCLASSES

/* ═══════════════════════════════════════════════════════════════════════
 * MODEL — 2-layer float32 weights
 * ═══════════════════════════════════════════════════════════════════════ */
typedef struct {
    int      H;           /* Hidden neurons */
    int      in_feat;     /* Input features (NC=196) */
    float   *W0;          /* [H × in_feat] frozen random projection */
    float   *W1;          /* [10 × H] AdamW-trained classifier */
} Flt32Model;


/* ═══════════════════════════════════════════════════════════════════════
 * MODEL LOADER — reads weights.meta + W0.bin + W1.bin
 * ═══════════════════════════════════════════════════════════════════════ */
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
        fprintf(stderr, "[FATAL] Unsupported model: layers=%d, out=%d (expect 2, %d)\n",
                n_layers, out_feat, N_CLASSES);
        return NULL;
    }

    Flt32Model *m = (Flt32Model *)malloc(sizeof(Flt32Model));
    if (!m) return NULL;
    m->H       = H;
    m->in_feat = in_feat;

    size_t w0_sz = (size_t)H * (size_t)in_feat;
    size_t w1_sz = (size_t)N_CLASSES * (size_t)H;

    m->W0 = (float *)malloc(w0_sz * sizeof(float));
    m->W1 = (float *)malloc(w1_sz * sizeof(float));
    if (!m->W0 || !m->W1) {
        free(m->W0); free(m->W1); free(m);
        return NULL;
    }

    snprintf(path, sizeof(path), "%s/W0.bin", dir);
    f = fopen(path, "rb");
    if (!f || fread(m->W0, sizeof(float), w0_sz, f) != w0_sz) {
        fprintf(stderr, "[FATAL] Cannot read %s\n", path);
        if (f) fclose(f);
        free(m->W0); free(m->W1); free(m);
        return NULL;
    }
    fclose(f);

    snprintf(path, sizeof(path), "%s/W1.bin", dir);
    f = fopen(path, "rb");
    if (!f || fread(m->W1, sizeof(float), w1_sz, f) != w1_sz) {
        fprintf(stderr, "[FATAL] Cannot read %s\n", path);
        if (f) fclose(f);
        free(m->W0); free(m->W1); free(m);
        return NULL;
    }
    fclose(f);

    printf("══╡ MODEL ╞═══════════════════════════════════════════════════════\n");
    printf("  Dir:   %s\n", dir);
    printf("  Arch:  2-layer float32  W0[%d×%d]  W1[%d×%d]\n",
           H, in_feat, N_CLASSES, H);
    printf("  Size:  W0=%zu KB + W1=%zu KB = %zu KB\n",
           w0_sz * sizeof(float) / 1024,
           w1_sz * sizeof(float) / 1024,
           (w0_sz + w1_sz) * sizeof(float) / 1024);
    fflush(stdout);
    return m;
}

static void model_free(Flt32Model *m) {
    if (m) { free(m->W0); free(m->W1); free(m); }
}


/* ═══════════════════════════════════════════════════════════════════════
 * FORWARD — matmul(W0, x) → LReLU(0.05) → matmul(W1, h0) → out[10]
 * ═══════════════════════════════════════════════════════════════════════
 * Same forward as the trainer.
 */
static void forward(const Flt32Model *m, const float *x, float *output) {
    /* h0 = matmul(W0, x) */
    float h0_buf[4096];   /* enough for H up to 4096 */
    int H = m->H;
    int in_feat = m->in_feat;
    float *h0 = (H <= 4096) ? h0_buf : (float *)malloc((size_t)H * sizeof(float));

    for (int o = 0; o < H; o++) {
        float acc = 0.0f;
        for (int i = 0; i < in_feat; i++)
            acc += x[i] * m->W0[(size_t)o * (size_t)in_feat + (size_t)i];
        h0[o] = acc;
    }

    /* LReLU(0.05) */
    for (int o = 0; o < H; o++)
        if (h0[o] < 0.0f) h0[o] *= 0.05f;

    /* output = matmul(W1, h0) */
    for (int k = 0; k < N_CLASSES; k++) {
        float acc = 0.0f;
        for (int o = 0; o < H; o++)
            acc += h0[o] * m->W1[(size_t)k * (size_t)H + (size_t)o];
        output[k] = acc;
    }

    if (H > 4096 && h0 != h0_buf) free(h0);
}


/* ═══════════════════════════════════════════════════════════════════════
 * ACCURACY
 * ═══════════════════════════════════════════════════════════════════════ */
static float accuracy_pct(const float *X, const uint8_t *Y, int N,
                           const Flt32Model *m) {
    int ok = 0;
    #pragma omp parallel reduction(+:ok)
    {
        float out[10];
        #pragma omp for schedule(static)
        for (int s = 0; s < N; s++) {
            forward(m, X + (size_t)s * (size_t)m->in_feat, out);
            int pred = 0;
            for (int k = 1; k < N_CLASSES; k++)
                if (out[k] > out[pred]) pred = k;
            if (pred == (int)Y[s]) ok++;
        }
    }
    return (float)ok * 100.0f / (float)N;
}


/* ═══════════════════════════════════════════════════════════════════════
 * SINGLE IMAGE CLASSIFICATION — PGM or raw 784 bytes
 * ═══════════════════════════════════════════════════════════════════════
 * Same pixel packing as ki_pack_packed_float: 4px → 1 float [-1,+1].
 * 0=white, 255=black (MNIST convention). PGM (P5) auto-detected.
 */
static int classify_image(const char *image_path, const Flt32Model *m) {
    FILE *f = fopen(image_path, "rb");
    if (!f) {
        fprintf(stderr, "[ERROR] Cannot open image: %s\n", image_path);
        return -1;
    }

    uint8_t raw_pixels[784];
    size_t nread;

    /* Detect PGM format (P5 magic) */
    uint8_t magic[3];
    nread = fread(magic, 1, 3, f);
    if (nread == 3 && magic[0] == 'P' && magic[1] == '5') {
        /* Skip remaining PGM header */
        int ch, newlines = 0;
        while (newlines < 2 && (ch = fgetc(f)) != EOF)
            if (ch == '\n') newlines++;
        nread = fread(raw_pixels, 1, 784, f);
        if (nread != 784) {
            fprintf(stderr, "[ERROR] PGM: expected 784 pixels, got %zu\n", nread);
            fclose(f); return -1;
        }
    } else {
        /* Assume raw 784 bytes */
        if (nread > 0) raw_pixels[0] = magic[0];
        if (nread > 1) raw_pixels[1] = magic[1];
        if (nread > 2) raw_pixels[2] = magic[2];
        size_t more = fread(raw_pixels + nread, 1, 784 - nread, f);
        nread += more;
        if (nread != 784) {
            fprintf(stderr, "[ERROR] Expected 784 bytes, got %zu\n", nread);
            fclose(f); return -1;
        }
    }
    fclose(f);

    /* Pack into float[NC]: same as ki_pack_packed_float */
    float packed[KI_NC];
    for (int c = 0; c < KI_NC; c++) {
        int sum = 0;
        for (int k = 0; k < 4; k++)
            sum += (int)raw_pixels[(size_t)c * 4 + (size_t)k];
        packed[c] = (float)sum / (4.0f * 127.5f) - 1.0f;
    }

    /* Forward */
    float out[10];
    forward(m, packed, out);

    int pred = 0;
    for (int k = 1; k < N_CLASSES; k++)
        if (out[k] > out[pred]) pred = k;

    double px_sum = 0.0;
    for (int i = 0; i < 784; i++) px_sum += (double)raw_pixels[i];

    printf("\n══╡ SINGLE IMAGE ╞════════════════════════════════════════════════\n");
    printf("  File:  %s\n", image_path);
    printf("  Pixel mean: %.1f  (0=white, 255=black, row-major 28×28)\n",
           px_sum / 784.0);
    fflush(stdout);

    printf("\n  Scores (float32, unscaled):\n");
    for (int k = 0; k < N_CLASSES; k++)
        printf("    %d: %8.4f%s\n", k, (double)out[k],
               (k == pred) ? "  ← PREDICTED" : "");

    printf("\n  >>> Predicted digit: %d <<<\n", pred);
    fflush(stdout);
    return pred;
}


/* ═══════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[]) {
    char model_dir[512] = "";
    char image_path[512] = "";
    int evalN = 10000;
    int threadN = 8;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s --model DIR [options]\n", argv[0]);
            printf("  --model DIR   Path to export directory (required)\n");
            printf("  --image FILE  Classify a single image (PGM or raw 28×28)\n");
            printf("  --evalN N     MNIST eval samples (default: 10000)\n");
            printf("  --threadN N   OpenMP threads (default: 8)\n");
            return 0;
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            strncpy(model_dir, argv[++i], sizeof(model_dir) - 1);
            model_dir[sizeof(model_dir) - 1] = '\0';
        } else if (strcmp(argv[i], "--image") == 0 && i + 1 < argc) {
            strncpy(image_path, argv[++i], sizeof(image_path) - 1);
            image_path[sizeof(image_path) - 1] = '\0';
        } else if (strcmp(argv[i], "--evalN") == 0 && i + 1 < argc) {
            evalN = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threadN") == 0 && i + 1 < argc) {
            threadN = atoi(argv[++i]);
            if (threadN < 1) threadN = 1;
        } else {
            fprintf(stderr, "[ERROR] Unknown argument: %s\nTry --help\n", argv[i]);
            return 1;
        }
    }

    if (model_dir[0] == '\0') {
        fprintf(stderr, "[FATAL] --model DIR is required\n");
        return 1;
    }

    omp_set_num_threads(threadN);

    /* Load model */
    Flt32Model *model = model_load(model_dir);
    if (!model) return 1;

    /* Single image mode */
    if (image_path[0] != '\0') {
        int pred = classify_image(image_path, model);
        model_free(model);
        return (pred >= 0) ? 0 : 1;
    }

    /* Load MNIST */
    ki_MNISTData data;
    if (ki_mnist_read(&data) != 0) { model_free(model); return 1; }
    if (data.pixels != KI_PX) {
        fprintf(stderr, "[FATAL] Expected %d pixels, got %d\n", KI_PX, data.pixels);
        ki_mnist_free(&data); model_free(model); return 1;
    }

    int total_eval = evalN;
    int offset = data.num_images - total_eval;
    if (offset < 0) { offset = 0; total_eval = data.num_images; }

    /* Pack input: 784 raw pixels → 196 floats (4px avg) */
    float *X_all = ki_pack_packed_float(data.X_raw, data.num_images);
    float *X_te  = X_all + (size_t)offset * (size_t)model->in_feat;
    uint8_t *y_te = data.y + offset;

    printf("\n══╡ INFERENCE ╞══════════════════════════════════════════════════\n");
    printf("  Evaluating %d samples (offset=%d)\n", total_eval, offset);
    fflush(stdout);

    struct timeval tv0, tv1;
    gettimeofday(&tv0, NULL);

    float acc = accuracy_pct(X_te, y_te, total_eval, model);

    gettimeofday(&tv1, NULL);
    int elapsed = (int)((tv1.tv_sec - tv0.tv_sec) * 1000
                      + (tv1.tv_usec - tv0.tv_usec) / 1000);

    printf("\n══╡ RESULT ╞══════════════════════════════════════════════════════\n");
    printf("  Model:   H=%d  2-layer float32 (AdamW)\n", model->H);
    printf("  Eval:    %.1f%%  (%d/%d)\n",
           acc, (int)(acc * (float)total_eval / 100.0f + 0.5f), total_eval);
    printf("  Time:    %dms  (%.1f µs/sample)\n",
           elapsed, (double)elapsed * 1000.0 / (double)total_eval);

    ki_report_show(0, 0, (int)(acc * (float)total_eval / 100.0f + 0.5f), total_eval,
                   elapsed, threadN, 0, 0.0f);

    /* Cleanup */
    free(X_all);
    ki_mnist_free(&data);
    model_free(model);
    return 0;
}
