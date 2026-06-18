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
 */
#include "ki-common.h"
#include "maj3.h"
#include <inttypes.h>

/* ── Konstanten ────────────────────────────────────────────────── */
#define NC        196
#define BITS       32
#define N_CLASSES KI_NCLASSES

/* Export file magic + version (must match trainer) */
#define OTTO_MAGIC   0x4F54544FU   /* "OTTO" */
#define OTTO_VERSION 1U

/* Index: [10][H][32] */
#define TGT_IDX(k, h, b, H) \
    ((size_t)(k) * (size_t)(H) * 32 + (size_t)(h) * 32 + (size_t)(b))


/* ═══════════════════════════════════════════════════════════════════
 * MODEL — gepacktes geladenes Model
 * ═══════════════════════════════════════════════════════════════════ */
typedef struct {
    uint32_t h0_mode;          /* 0=XNOR, 1=XOR */
    int      H;                /* Hidden neurons */
    int      nc;               /* Containers per image (muss NC passen) */
    uint32_t *W0;              /* [H][nc] */
    int32_t  *target;          /* [10][H][32] (log-odds) */
    int64_t  class_offset[10]; /* per-class offset */
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

    /* Header lesen */
    uint32_t magic, version, mode, H, ncc;
    if (fread(&magic,   sizeof(magic),   1, f) != 1 ||
        fread(&version, sizeof(version), 1, f) != 1 ||
        fread(&mode,    sizeof(mode),    1, f) != 1 ||
        fread(&H,       sizeof(H),       1, f) != 1 ||
        fread(&ncc,     sizeof(ncc),     1, f) != 1) {
        fprintf(stderr, "[FATAL] Cannot read header from %s\n", path);
        fclose(f); return NULL;
    }

    if (magic != OTTO_MAGIC) {
        fprintf(stderr, "[FATAL] Bad magic in %s: 0x%08X (expected 0x%08X)\n",
                path, magic, OTTO_MAGIC);
        fclose(f); return NULL;
    }
    if (version != OTTO_VERSION) {
        fprintf(stderr, "[FATAL] Unsupported version %u\n", version);
        fclose(f); return NULL;
    }

    OttoModel *m = (OttoModel *)malloc(sizeof(OttoModel));
    if (!m) { fclose(f); return NULL; }
    m->h0_mode = mode;
    m->H       = (int)H;
    m->nc      = (int)ncc;

    /* W0 laden */
    size_t w0_count = (size_t)H * (size_t)ncc;
    m->W0 = (uint32_t *)malloc(w0_count * sizeof(uint32_t));
    if (!m->W0) { free(m); fclose(f); return NULL; }
    if (fread(m->W0, sizeof(uint32_t), w0_count, f) != w0_count) {
        fprintf(stderr, "[FATAL] Cannot read W0\n");
        free(m->W0); free(m); fclose(f); return NULL;
    }

    /* Target laden */
    size_t tgt_count = (size_t)H * 10 * 32;
    m->target = (int32_t *)malloc(tgt_count * sizeof(int32_t));
    if (!m->target) { free(m->W0); free(m); fclose(f); return NULL; }
    if (fread(m->target, sizeof(int32_t), tgt_count, f) != tgt_count) {
        fprintf(stderr, "[FATAL] Cannot read target\n");
        free(m->target); free(m->W0); free(m); fclose(f); return NULL;
    }

    /* class_offset laden */
    if (fread(m->class_offset, sizeof(int64_t), 10, f) != 10) {
        fprintf(stderr, "[FATAL] Cannot read class_offset\n");
        free(m->target); free(m->W0); free(m); fclose(f); return NULL;
    }

    fclose(f);

    printf("══╡ MODEL ╞═══════════════════════════════════════════════════════\n");
    printf("  File:  %s\n", path);
    printf("  Mode:  %s\n", m->h0_mode == 0 ? "XNOR" : "XOR");
    printf("  H:     %d\n", m->H);
    printf("  NC:    %d\n", m->nc);
    printf("  W0:    %zu KB\n", w0_count * sizeof(uint32_t) / 1024);
    printf("  Tgt:   %zu KB\n", tgt_count * sizeof(int32_t) / 1024);
    printf("  Off:   %zu B\n", 10 * sizeof(int64_t));
    fflush(stdout);

    return m;
}

static void model_free(OttoModel *m) {
    if (m) { free(m->W0); free(m->target); free(m); }
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
 * SCORE — Bayes log-Score
 * ═══════════════════════════════════════════════════════════════════ */
static void scores_otto(const uint32_t *in, const OttoModel *m,
                         int64_t scores[10]) {
    for (int k = 0; k < N_CLASSES; k++)
        scores[k] = m->class_offset[k];

    for (int h = 0; h < m->H; h++) {
        uint32_t h0 = h0_neuron(in, m->W0 + (size_t)h * (size_t)m->nc);
        for (int k = 0; k < N_CLASSES; k++) {
            for (int b = 0; b < 32; b++) {
                if (h0 & (1U << (unsigned)b))
                    scores[k] += m->target[TGT_IDX(k, h, b, m->H)];
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
 * MAIN — minimal arg parser
 * ═══════════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[]) {
    char model_path[512] = "";
    int evalN = 10000;
    int threadN = 8;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s --model <path> [options]\n", argv[0]);
            printf("  --model PATH  Path to .otto model file (required)\n");
            printf("  --evalN N     Eval samples (default: 10000)\n");
            printf("  --threadN N   OpenMP threads (default: 8)\n");
            return 0;
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            strncpy(model_path, argv[++i], sizeof(model_path) - 1);
            model_path[sizeof(model_path) - 1] = '\0';
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

    printf("\n══╡ RESULT ╞══════════════════════════════════════════════════════\n");
    printf("  Model:   H=%d  %s\n", model->H,
           model->h0_mode == 0 ? "XNOR" : "XOR");
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
