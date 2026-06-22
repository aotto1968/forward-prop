/*
 * otto-score-ifc/mlp-bin32-ifc.c — Bin32 Hebbian Inference
 * =========================================================
 *
 * Loads a trained uint32 binary model (exported by mlp-bin32-trn-w1-hebbian)
 * and classifies MNIST test data or a single image.
 *
 * Model format (same as flt32, but uint32 weights):
 *   weights.meta   — "2\nH in_feat\n10 H\n"
 *   W0.bin         — uint32[H × in_feat]  frozen random projection
 *   W1.bin         — uint32[10 × H]       Hebbian-trained weights
 *
 * Forward: MAJ3(~(in ^ W0)) → h0 → XNOR + popcnt(W1, h0) → argmax
 * Pure &|~ + popcount — NO floating point, NO matmul.
 *
 * Build:
 *   make mlp-bin32-ifc-xnor.exe
 *
 * Usage:
 *   ./mlp-bin32-ifc-xnor.exe --model DIR  [--evalN N]  [--image FILE]
 */
#include "ki-common.h"
#include "maj3.h"
#include <inttypes.h>

#define N_CLASSES KI_NCLASSES
#define NC        196
#define BITS       32

/* ═══════════════════════════════════════════════════════════════════════
 * H0 MODE — compile-time switch via -DH0_XOR
 * ═══════════════════════════════════════════════════════════════════════
 * Default: XNOR (~(in ^ W0)). With -DH0_XOR: XOR (in ^ W0).
 */
#ifdef H0_XOR
#  define H0_ALWAYS_XOR 1
#else
#  define H0_ALWAYS_XOR 0
#endif


/* ═══════════════════════════════════════════════════════════════════════
 * MODEL — uint32 binary weights
 * ═══════════════════════════════════════════════════════════════════════ */
typedef struct {
    int      H;           /* Hidden neurons */
    int      in_feat;     /* Input features (NC=196) */
    uint32_t *W0;         /* [H × in_feat] frozen random projection */
    uint32_t *W1;         /* [10 × H] Hebbian-trained weights */
} Bin32Model;


/* ═══════════════════════════════════════════════════════════════════════
 * MODEL LOADER — reads weights.meta + W0.bin + W1.bin
 * ═══════════════════════════════════════════════════════════════════════ */
static Bin32Model *model_load(const char *dir) {
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
        fprintf(stderr, "[FATAL] Bad format in %s\n", path);
        fclose(f);
        return NULL;
    }
    fclose(f);

    if (n_layers != 2 || out_feat != N_CLASSES || in_feat != NC) {
        fprintf(stderr, "[FATAL] Unsupported: layers=%d, H=%d, in=%d, out=%d\n",
                n_layers, H, in_feat, out_feat);
        return NULL;
    }

    Bin32Model *m = (Bin32Model *)malloc(sizeof(Bin32Model));
    if (!m) return NULL;
    m->H       = H;
    m->in_feat = in_feat;

    size_t w0_sz = (size_t)H * (size_t)in_feat;
    size_t w1_sz = (size_t)N_CLASSES * (size_t)H;

    m->W0 = (uint32_t *)malloc(w0_sz * sizeof(uint32_t));
    m->W1 = (uint32_t *)malloc(w1_sz * sizeof(uint32_t));
    if (!m->W0 || !m->W1) {
        free(m->W0); free(m->W1); free(m);
        return NULL;
    }

    snprintf(path, sizeof(path), "%s/W0.bin", dir);
    f = fopen(path, "rb");
    if (!f || fread(m->W0, sizeof(uint32_t), w0_sz, f) != w0_sz) {
        fprintf(stderr, "[FATAL] Cannot read %s\n", path);
        if (f) fclose(f);
        free(m->W0); free(m->W1); free(m);
        return NULL;
    }
    fclose(f);

    snprintf(path, sizeof(path), "%s/W1.bin", dir);
    f = fopen(path, "rb");
    if (!f || fread(m->W1, sizeof(uint32_t), w1_sz, f) != w1_sz) {
        fprintf(stderr, "[FATAL] Cannot read %s\n", path);
        if (f) fclose(f);
        free(m->W0); free(m->W1); free(m);
        return NULL;
    }
    fclose(f);

    printf("══╡ MODEL ╞═══════════════════════════════════════════════════════\n");
    printf("  Dir:   %s\n", dir);
    printf("  Arch:  bin32 Hebbian  W0[%d×%d]  W1[%d×%d]  %s\n",
           H, in_feat, N_CLASSES, H, H0_ALWAYS_XOR ? "XOR" : "XNOR");
    printf("  Size:  W0=%zu KB + W1=%zu KB = %zu KB\n",
           w0_sz * sizeof(uint32_t) / 1024,
           w1_sz * sizeof(uint32_t) / 1024,
           (w0_sz + w1_sz) * sizeof(uint32_t) / 1024);
    fflush(stdout);
    return m;
}

static void model_free(Bin32Model *m) {
    if (m) { free(m->W0); free(m->W1); free(m); }
}


/* ═══════════════════════════════════════════════════════════════════════
 * H0 NEURON — MAJ3
 * ═══════════════════════════════════════════════════════════════════════ */
static uint32_t h0_neuron(const uint32_t *in, const uint32_t *W0_row) {
    uint32_t match[NC];
    if (H0_ALWAYS_XOR) {
        for (int c = 0; c < NC; c++)
            match[c] = in[c] ^ W0_row[c];
    } else {
        for (int c = 0; c < NC; c++)
            match[c] = ~(in[c] ^ W0_row[c]);
    }
    return majority_tree(match, NC);
}


/* ═══════════════════════════════════════════════════════════════════════
 * SCORE — XNOR + popcnt (same as Hebbian trainer)
 * ═══════════════════════════════════════════════════════════════════════
 * score[k] = Σ_h popcnt(~(W1[k][h] ^ h0[h]))
 *
 * Pure integer: no float, no division, no Bayes log-odds.
 * Argmax is invariant under affine transform, so we skip /32 -0.5.
 */
static void scores_bin32(const uint32_t *in, const Bin32Model *m,
                          int64_t scores[10]) {
    /* Compute H0 for all neurons */
    uint32_t h0[4096];  /* enough for H up to 4096 */
    int H = m->H;
    for (int h = 0; h < H; h++)
        h0[h] = h0_neuron(in, m->W0 + (size_t)h * (size_t)m->in_feat);

    /* Score each class via popcnt(XNOR(W1[k][h], h0[h])) */
    for (int k = 0; k < N_CLASSES; k++) {
        int64_t sum = 0;
        const uint32_t *w1_row = m->W1 + (size_t)k * (size_t)H;
        for (int h = 0; h < H; h++)
            sum += (int64_t)__builtin_popcount(~(w1_row[h] ^ h0[h]));
        scores[k] = sum;
    }
}


/* ═══════════════════════════════════════════════════════════════════════
 * ACCURACY
 * ═══════════════════════════════════════════════════════════════════════ */
static float accuracy_pct(const uint32_t *X, const uint8_t *Y, int N,
                           const Bin32Model *m) {
    int ok = 0;
    #pragma omp parallel reduction(+:ok)
    {
        int64_t scores[10];
        #pragma omp for schedule(static)
        for (int s = 0; s < N; s++) {
            scores_bin32(X + (size_t)s * (size_t)m->in_feat, m, scores);
            int pred = 0;
            for (int k = 1; k < N_CLASSES; k++)
                if (scores[k] > scores[pred]) pred = k;
            if (pred == (int)Y[s]) ok++;
        }
    }
    return (float)ok * 100.0f / (float)N;
}


/* ═══════════════════════════════════════════════════════════════════════
 * SINGLE IMAGE CLASSIFICATION
 * ═══════════════════════════════════════════════════════════════════════
 * Loads PGM or raw 784 bytes, packs into uint32[196], runs inference.
 */
static int classify_image(const char *image_path, const Bin32Model *m) {
    FILE *f = fopen(image_path, "rb");
    if (!f) {
        fprintf(stderr, "[ERROR] Cannot open image: %s\n", image_path);
        return -1;
    }

    uint8_t raw_pixels[784];
    size_t nread;

    uint8_t magic[3];
    nread = fread(magic, 1, 3, f);
    if (nread == 3 && magic[0] == 'P' && magic[1] == '5') {
        int ch, newlines = 0;
        while (newlines < 2 && (ch = fgetc(f)) != EOF)
            if (ch == '\n') newlines++;
        nread = fread(raw_pixels, 1, 784, f);
        if (nread != 784) {
            fprintf(stderr, "[ERROR] PGM: expected 784 pixels\n");
            fclose(f); return -1;
        }
    } else {
        if (nread > 0) raw_pixels[0] = magic[0];
        if (nread > 1) raw_pixels[1] = magic[1];
        if (nread > 2) raw_pixels[2] = magic[2];
        size_t more = fread(raw_pixels + nread, 1, 784 - nread, f);
        nread += more;
        if (nread != 784) {
            fprintf(stderr, "[ERROR] Expected 784 bytes\n");
            fclose(f); return -1;
        }
    }
    fclose(f);

    /* Pack into uint32[NC]: p0|p1<<8|p2<<16|p3<<24 */
    uint32_t packed[NC];
    for (int c = 0; c < NC; c++) {
        uint32_t val = 0;
        for (int k = 0; k < 4; k++)
            val |= ((uint32_t)raw_pixels[(size_t)c * 4 + (size_t)k] & 0xFFU) << (unsigned)(k * 8);
        packed[c] = val;
    }

    int64_t scores[10];
    scores_bin32(packed, m, scores);

    int pred = 0;
    for (int k = 1; k < N_CLASSES; k++)
        if (scores[k] > scores[pred]) pred = k;

    double px_mean = 0.0;
    for (int i = 0; i < 784; i++) px_mean += (double)raw_pixels[i];

    printf("\n══╡ SINGLE IMAGE ╞════════════════════════════════════════════════\n");
    printf("  File:  %s\n", image_path);
    printf("  Pixel mean: %.1f  (0=white, 255=black, row-major 28×28)\n",
           px_mean / 784.0);
    printf("\n  Scores (popcount raw):\n");
    for (int k = 0; k < N_CLASSES; k++)
        printf("    %d: %7" PRId64 "%s\n", k, scores[k],
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

    Bin32Model *model = model_load(model_dir);
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

    /* Pack input: 784 raw pixels → uint32[196] */
    uint32_t *X_all = load_input(data.X_raw, data.num_images);
    uint32_t *X_te  = X_all + (size_t)offset * (size_t)model->in_feat;
    uint8_t  *y_te  = data.y + offset;

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
    printf("  Model:   H=%d  bin32 Hebbian (%s)\n", model->H,
           H0_ALWAYS_XOR ? "XOR" : "XNOR");
    printf("  Eval:    %.1f%%  (%d/%d)\n",
           acc, (int)(acc * (float)total_eval / 100.0f + 0.5f), total_eval);
    printf("  Time:    %dms  (%.1f µs/sample)\n",
           elapsed, (double)elapsed * 1000.0 / (double)total_eval);

    ki_report_show(0, 0,
                   (int)(acc * (float)total_eval / 100.0f + 0.5f), total_eval,
                   elapsed, threadN);

    free(X_all);
    ki_mnist_free(&data);
    model_free(model);
    return 0;
}
