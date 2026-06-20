/*
 * mlp-otto-score-ifc.c — Otto Score Inference
 * =============================================
 *
 * Loads an exported Otto Score model and classifies MNIST test data.
 * Pure &|~ + int32 — no float, no multiply, no AdamW.
 *
 * Build:
 *   make all                   (XNOR + XOR)
 *   make xnor                  (XNOR only)
 *   make xor                   (XOR only)
 *
 * Usage:
 *   ./mlp-otto-score-ifc-xnor.exe --model models/model-xnor.otto
 *   ./mlp-otto-score-ifc-xor.exe  --model models/model-xor.otto
 *   ./mlp-otto-score-ifc-xnor.exe --model models/model-xnor.otto --image digit.raw
 *   ./mlp-otto-score-ifc-xnor.exe --model models/model-xnor.otto --evalN 10000
 */
#include "ki-common.h"
#include "maj3.h"
#include <inttypes.h>

/* Konstanten */
#define NC        196
#define BITS       32
#define N_CLASSES KI_NCLASSES

/* OT_PRECISION — muss zum Trainer passen, default 10 → F=1024 */
#ifndef OT_PRECISION
#define OT_PRECISION 10
#endif
#define OT_F  (1 << OT_PRECISION)   /* Skalierungsfaktor für target/offset */

/* Export file magic + version (must match trainer) */
#define OTTO_MAGIC   0x4F54544FU   /* "OTTO" */
#define OTTO_VERSION_SINGLE 1U    /* v1 = single model */
#define OTTO_VERSION_ENSEMBLE 5U  /* v5 = ensemble (N members, no precision) */
#define OTTO_VERSION_ENSEMBLE_V6 6U /* v6 = ensemble + precision field */

/* Index: [10][H][32] */
#define TGT_IDX(k, h, b, H) \
    ((size_t)(k) * (size_t)(H) * 32 + (size_t)(h) * 32 + (size_t)(b))


/* ═══════════════════════════════════════════════════════════════════
 * MODEL — geladenes Model (single oder ensemble)
 * ═══════════════════════════════════════════════════════════════════
 * Version 1 (single): ensemble_n=1, arrays[0] werden genutzt.
 * Version 5 (ensemble): ensemble_n=N, arrays[0..N-1] pro Member.
 */
typedef struct {
    uint32_t h0_mode;          /* 0=XNOR, 1=XOR */
    int      H;                /* Hidden neurons (pro Member) */
    int      nc;               /* Containers per image */
    int      ensemble_n;       /* Anzahl Ensemble-Member (1=Single) */
    int      precision;        /* OT_PRECISION (F = 1<<precision) */
    uint32_t *W0;              /* [ensemble_n][H][nc] */
    int32_t  *target;          /* [ensemble_n][10][H][32] */
    int64_t  *class_offset;    /* [ensemble_n][10] */
} OttoModel;


/* ═══════════════════════════════════════════════════════════════════
 * H0 MODE — compile-time switch via -DH0_XOR
 * ═══════════════════════════════════════════════════════════════════
 * When -DH0_XOR is defined, XOR mode is used (overrides model mode).
 * Default: use the mode stored in the model file.
 */
#ifdef H0_XOR
#  define H0_ALWAYS_XOR 1
#else
#  define H0_ALWAYS_XOR 0
#endif


/* ═══════════════════════════════════════════════════════════════════
 * MAJ3 — majority_tree aus lib/maj3.h (bereits includiert)
 * ═══════════════════════════════════════════════════════════════════ */


/* ═══════════════════════════════════════════════════════════════════
 * MODEL LOADER — loads model from .otto file
 * ═══════════════════════════════════════════════════════════════════
 */
static OttoModel *model_load_path(const char *path) {

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[FATAL] Cannot open %s\n", path);
        return NULL;
    }

    /* Header lesen: erst version, dann je nach Format den Rest */
    uint32_t magic, version, mode = 0, ensemble_n = 1, H = 0, ncc = 0;
    if (fread(&magic,   sizeof(magic),   1, f) != 1 ||
        fread(&version, sizeof(version), 1, f) != 1) {
        fprintf(stderr, "[FATAL] Cannot read header from %s\n", path);
        fclose(f); return NULL;
    }

    if (magic != OTTO_MAGIC) {
        fprintf(stderr, "[FATAL] Bad magic in %s: 0x%08X (expected 0x%08X)\n",
                path, magic, OTTO_MAGIC);
        fclose(f); return NULL;
    }

    if (!(version == OTTO_VERSION_SINGLE || 
          version == OTTO_VERSION_ENSEMBLE ||
          version == OTTO_VERSION_ENSEMBLE_V6)) {
        fprintf(stderr, "[FATAL] Unsupported version %u (expected %u, %u, or %u)\n",
                version, OTTO_VERSION_SINGLE, OTTO_VERSION_ENSEMBLE, 
                OTTO_VERSION_ENSEMBLE_V6);
        fclose(f); return NULL;
    }

    if (version == OTTO_VERSION_SINGLE) {
        /* v1: mode(4) + H(4) + NC(4) = 12 Bytes */
        if (fread(&mode, sizeof(mode), 1, f) != 1 ||
            fread(&H,    sizeof(H),    1, f) != 1 ||
            fread(&ncc,  sizeof(ncc),  1, f) != 1) {
            fprintf(stderr, "[FATAL] Cannot read v1 header\n");
            fclose(f); return NULL;
        }
        ensemble_n = 1;
    } else {
        /* v5/v6: mode(4) + ensemble_n(4) + H(4) + NC(4) = 16 Bytes */
        if (fread(&mode,       sizeof(mode),       1, f) != 1 ||
            fread(&ensemble_n, sizeof(ensemble_n), 1, f) != 1 ||
            fread(&H,          sizeof(H),          1, f) != 1 ||
            fread(&ncc,        sizeof(ncc),        1, f) != 1) {
            fprintf(stderr, "[FATAL] Cannot read v5 header\n");
            fclose(f); return NULL;
        }
    }

    if ((int)ncc != NC) {
        fprintf(stderr, "[FATAL] Model NC=%u != compile-time NC=%d\n", ncc, NC);
        fclose(f); return NULL;
    }

    OttoModel *m = (OttoModel *)malloc(sizeof(OttoModel));
    if (!m) { fclose(f); return NULL; }
    m->h0_mode     = mode;
    m->H           = (int)H;
    m->nc          = (int)ncc;
    m->ensemble_n  = (int)ensemble_n;

    /* Precision aus Header oder Default */
    if (version == OTTO_VERSION_ENSEMBLE_V6) {
        uint32_t prec = 0;
        if (fread(&prec, sizeof(prec), 1, f) != 1) {
            fprintf(stderr, "[FATAL] Cannot read precision from v6 header\n");
            free(m); fclose(f); return NULL;
        }
        m->precision = (int)prec;
        if (m->precision != OT_PRECISION)
            fprintf(stderr, "[WARN] Model precision=%d != ifc OT_PRECISION=%d\n"
                    "  Model: %s  (trained with different scaling!)\n",
                    m->precision, OT_PRECISION, path);
    } else if (version == OTTO_VERSION_ENSEMBLE) {
        m->precision = OT_PRECISION;  /* v5: assume current default */
    } else {
        m->precision = 17;            /* v1: legacy ×100000 ≈ 1<<17 */
    }

    /* Daten lesen: Datei-Format ist W0[m] + Tgt[m] + Off[m] pro Member,
     * nicht alle W0, dann alle Tgt, dann alle Off.  Vgl. export_ensemble(). */
    size_t w0_per_m = (size_t)H * (size_t)ncc;
    size_t tgt_per_m = (size_t)H * 10 * 32;
    size_t w0_sz   = (size_t)ensemble_n * w0_per_m;
    size_t tgt_sz  = (size_t)ensemble_n * tgt_per_m;
    size_t off_sz  = (size_t)ensemble_n * 10;

    m->W0          = (uint32_t *)malloc(w0_sz * sizeof(uint32_t));
    m->target      = (int32_t  *)malloc(tgt_sz * sizeof(int32_t));
    m->class_offset= (int64_t  *)malloc(off_sz * sizeof(int64_t));

    if (!m->W0 || !m->target || !m->class_offset) {
        free(m->W0); free(m->target); free(m->class_offset); free(m);
        fclose(f); return NULL;
    }

    /* Lese member-weise: W0[m], Tgt[m], Off[m] */
    for (size_t mem = 0; mem < (size_t)ensemble_n; mem++) {
        if (fread(m->W0 + mem * w0_per_m, sizeof(uint32_t), w0_per_m, f) != w0_per_m ||
            fread(m->target + mem * tgt_per_m, sizeof(int32_t), tgt_per_m, f) != tgt_per_m ||
            fread(m->class_offset + mem * 10, sizeof(int64_t), 10, f) != 10) {
            fprintf(stderr, "[FATAL] Cannot read member %zu data from %s\n",
                    mem, path);
            free(m->W0); free(m->target); free(m->class_offset); free(m);
            fclose(f); return NULL;
        }
    }

    fclose(f);

    printf("══╡ MODEL ╞═══════════════════════════════════════════════════════\n");
    printf("  File:  %s\n", path);
    printf("  Mode:  %s   Version: %s   F=%d\n",
           m->h0_mode == 0 ? "XNOR" : "XOR",
           version == OTTO_VERSION_SINGLE ? "v1 (single)" : "v5 (ensemble)",
           1 << m->precision);
    printf("  H:     %d   NC: %d   Ensemble: %d\n",
           m->H, m->nc, m->ensemble_n);
    printf("  W0:    %zu KB   Tgt: %zu KB   Off: %zu KB   F=%d\n",
           w0_sz * sizeof(uint32_t) / 1024,
           tgt_sz * sizeof(int32_t) / 1024,
           off_sz * sizeof(int64_t) / 1024,
           1 << m->precision);
    fflush(stdout);

    return m;
}

static void model_free(OttoModel *m) {
    if (m) { free(m->W0); free(m->target); free(m->class_offset); free(m); }
}


/* ═══════════════════════════════════════════════════════════════════
 * H0 NEURON — MAJ3 mit geladener Mode
 * ═══════════════════════════════════════════════════════════════════ */
static uint32_t h0_neuron(const uint32_t *in, const uint32_t *W0_row) {
    uint32_t match[NC];
    int use_xor = H0_ALWAYS_XOR;  /* compile-time override */
    if (use_xor) {
        for (int c = 0; c < NC; c++)
            match[c] = in[c] ^ W0_row[c];
    } else {
        for (int c = 0; c < NC; c++)
            match[c] = ~(in[c] ^ W0_row[c]);
    }
    return majority_tree(match, NC);
}


/* ═══════════════════════════════════════════════════════════════════
 * SCORE — Bayes log-Score (summed across ensemble members)
 * ═══════════════════════════════════════════════════════════════════
 * Bei Single-Modell (ensemble_n=1) identisch zu v1.
 * Bei Ensemble: scores[k] = Σ_m offset_m[k] + Σ_h Σ_b y × target_m[k][h][b]
 */
static void scores_otto(const uint32_t *in, const OttoModel *m,
                         int64_t scores[10]) {
    for (int k = 0; k < N_CLASSES; k++)
        scores[k] = 0;

    size_t w0_row_sz = (size_t)m->nc;
    size_t tgt_m_sz  = (size_t)m->H * 10 * 32;  /* per member */

    for (int mem = 0; mem < m->ensemble_n; mem++) {
        size_t w0_off  = (size_t)mem * (size_t)m->H * w0_row_sz;
        size_t tgt_off = (size_t)mem * tgt_m_sz;

        for (int k = 0; k < N_CLASSES; k++)
            scores[k] += m->class_offset[(size_t)mem * 10 + (size_t)k];

        for (int h = 0; h < m->H; h++) {
            uint32_t h0 = h0_neuron(in, m->W0 + w0_off + (size_t)h * w0_row_sz);
            for (int k = 0; k < N_CLASSES; k++) {
                for (int b = 0; b < 32; b++) {
                    if (h0 & (1U << (unsigned)b))
                        scores[k] += m->target[tgt_off + TGT_IDX(k, h, b, m->H)];
                }
            }
        }
    }
}


/* ═══════════════════════════════════════════════════════════════════
 * ACCURACY
 * ═══════════════════════════════════════════════════════════════════ */
static float accuracy_pct(const uint32_t *X, const uint8_t *Y, int N,
                           const OttoModel *m) {
    int ok = 0;
    #pragma omp parallel for reduction(+:ok) schedule(static)
    for (int s = 0; s < N; s++) {
        int64_t scores[10];
        scores_otto(X + (size_t)s * (size_t)m->nc, m, scores);
        int pred = 0;
        for (int k = 1; k < N_CLASSES; k++)
            if (scores[k] > scores[pred]) pred = k;
        if (pred == (int)Y[s]) ok++;
    }
    return (float)ok * 100.0f / (float)N;
}


/* ═══════════════════════════════════════════════════════════════════
 * SINGLE IMAGE CLASSIFICATION
 * ═══════════════════════════════════════════════════════════════════
 * Reads a 28×28 grayscale image in PGM format or raw 784 bytes.
 * PGM (P5) is auto-detected — standard image viewers can open it.
 * Same pixel packing as MNIST training data.
 * 0=white, 255=black (MNIST convention).
 */
static int classify_image(const char *image_path, const OttoModel *m) {
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
        /* Skip remaining PGM header (2 more lines: dimensions, maxval) */
        int ch;
        int newlines = 0;
        while (newlines < 2 && (ch = fgetc(f)) != EOF) {
            if (ch == '\n') newlines++;
        }
        nread = fread(raw_pixels, 1, 784, f);
        if (nread != 784) {
            fprintf(stderr, "[ERROR] PGM: expected 784 pixels after header, got %zu\n", nread);
            fclose(f); return -1;
        }
    } else {
        /* Assume raw 784 bytes (no header, MNIST-compatible) */
        if (nread > 0) {
            raw_pixels[0] = magic[0];
            if (nread > 1) raw_pixels[1] = magic[1];
            if (nread > 2) raw_pixels[2] = magic[2];
        }
        size_t more = fread(raw_pixels + nread, 1, 784 - nread, f);
        nread += more;
        if (nread != 784) {
            fprintf(stderr, "[ERROR] Expected 784 bytes, got %zu\n", nread);
            fclose(f); return -1;
        }
    }
    fclose(f);

    /* Pack into uint32[NC] (same as load_input) */
    uint32_t packed[NC];
    double sum = 0.0;
    for (int c = 0; c < NC; c++) {
        uint32_t val = 0;
        for (int k = 0; k < 4; k++) {
            uint8_t px = raw_pixels[(size_t)c * 4 + (size_t)k];
            val |= ((uint32_t)px & 0xFFU) << (unsigned)(k * 8);
            sum += (double)px;
        }
        packed[c] = val;
    }

    /* Compute scores via ensemble-aware function */
    int64_t scores[10];
    scores_otto(packed, m, scores);

    /* Find best class */
    int pred = 0;
    for (int k = 1; k < N_CLASSES; k++)
        if (scores[k] > scores[pred]) pred = k;

    printf("\n══╡ SINGLE IMAGE ╞════════════════════════════════════════════════\n");
    printf("  File:  %s\n", image_path);
    printf("  Pixel mean: %.1f  (0=white, 255=black, row-major 28x28)\n",
           (double)sum / 784.0);
    int scale = 1 << m->precision;
    printf("\n  Scores:\n");
    for (int k = 0; k < N_CLASSES; k++)
        printf("    %d: %7.2f%s\n", k, (double)scores[k] / (double)scale,
               (k == pred) ? "  ← PREDICTED" : "");

    printf("\n  >>> Predicted digit: %d <<<\n", pred);
    fflush(stdout);
    return pred;
}


/* ═══════════════════════════════════════════════════════════════════
 * MAIN — minimal arg parser
 * ═══════════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[]) {
    char model_path[512] = "";
    char image_path[512] = "";
    int evalN = 10000;
    int threadN = 8;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s --model <path> [options]\n", argv[0]);
            printf("  --model PATH  Path to .otto model file (required)\n");
            printf("  --image FILE  Classify a single image (PGM or raw 28x28)\n");
            printf("                PGM (P5) auto-detected; raw = 784 bytes uint8\n");
            printf("  --evalN N     MNIST eval samples (default: 10000)\n");
            printf("  --threadN N   OpenMP threads (default: 8)\n");
            return 0;
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            strncpy(model_path, argv[++i], sizeof(model_path) - 1);
            model_path[sizeof(model_path) - 1] = '\0';
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

    if (model_path[0] == '\0') {
        fprintf(stderr, "[FATAL] --model PATH is required\n");
        fprintf(stderr, "Usage: %s --model <path-to-model.otto>\n", argv[0]);
        return 1;
    }

    omp_set_num_threads(threadN);

    /* ── Load Model ──────────────────────────────────────────── */
    OttoModel *model = model_load_path(model_path);
    if (!model) return 1;

    /* ── Single image mode? ──────────────────────────────────── */
    if (image_path[0] != '\0') {
        int pred = classify_image(image_path, model);
        model_free(model);
        return (pred >= 0) ? 0 : 1;
    }

    /* ── Load MNIST ──────────────────────────────────────────── */
    ki_MNISTData data;
    if (ki_mnist_read(&data) != 0) { model_free(model); return 1; }
    if (data.pixels != KI_PX) {
        fprintf(stderr, "[FATAL] Expected %d pixels, got %d\n", KI_PX, data.pixels);
        ki_mnist_free(&data); model_free(model); return 1;
    }

    int total_eval = evalN;
    /* Use only EVAL samples (last N of MNIST) */
    int offset = data.num_images - total_eval;
    if (offset < 0) { offset = 0; total_eval = data.num_images; }

    uint32_t *X_all = load_input(data.X_raw, data.num_images);
    uint32_t *X_te  = X_all + (size_t)offset * (size_t)model->nc;
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

    int scale = 1 << model->precision;
    printf("\n══╡ RESULT ╞══════════════════════════════════════════════════════\n");
    printf("  Model:   H=%d  %s  F=%d\n", model->H,
           model->h0_mode == 0 ? "XNOR" : "XOR", scale);
    printf("  Eval:    %.1f%%  (%d/%d)\n",
           acc, (int)(acc * (float)total_eval / 100.0f + 0.5f), total_eval);
    printf("  Time:    %dms  (%.1f µs/sample)\n",
           elapsed, (double)elapsed * 1000.0 / (double)total_eval);

    ki_report_show(0, 0,
                   (int)(acc * (float)total_eval / 100.0f + 0.5f), total_eval,
                   elapsed, threadN);

    /* ── Cleanup ────────────────────────────────────────────── */
    free(X_all);
    ki_mnist_free(&data);
    model_free(model);
    return 0;
}
