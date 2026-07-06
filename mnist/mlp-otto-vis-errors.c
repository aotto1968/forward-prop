/*
 * mnist-1/mlp-otto-vis-errors.c — Error Visualizer (MNIST + CIFAR-10)
 * ==============================================================
 *
 * Lädt ein trainiertes Modell (predictions-Datei), berechnet für
 * einen Index-Bereich alle Sample-Bilder und exportiert sie als
 * PNG (grayscale für MNIST, color RGB für CIFAR-10) + index.html
 * mit Fehler-Markierung.
 *
 * PNG writer ist dataset-spezifisch: ki_write_png() aus ki-local.h.
 *
 * Usage:
 *   ./tool --predictions FILE --export vis/
 *   → vis/index.html  (alle Samples, nach Klasse sortiert, Fehler rot)
 *
 * Predictions-Format (vom Trainer via --predictions):
 *   uint32 magic = 'PRED' (0x44455250)
 *   uint32 N     = Anzahl Samples
 *   uint8[N] preds = vorhergesagte Klassen
 */
#include "ki-common.h"
#include "maj3.h"
#include <inttypes.h>
#include <zlib.h>

/* ── Load predictions file ──────────────────────────────────────── */
static uint8_t *load_predictions(const char *path, int *n_out, int *offset_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[ERROR] Cannot open %s\n", path); return NULL; }

    uint32_t magic, n, off = 0;
    if (fread(&magic, sizeof(magic), 1, f) != 1 ||
        fread(&n, sizeof(n), 1, f) != 1) {
        fprintf(stderr, "[ERROR] Invalid predictions file (header)\n");
        fclose(f); return NULL;
    }
    if (magic != 0x44455250) {  /* 'PRED' */
        fprintf(stderr, "[ERROR] Bad magic: 0x%08X (expected 'PRED')\n", magic);
        fclose(f); return NULL;
    }
    /* Optional offset field (v2): default 0 if not present */
    if (fread(&off, sizeof(off), 1, f) != 1) off = 0;

    uint8_t *preds = (uint8_t *)malloc((size_t)n);
    if (!preds) { fclose(f); return NULL; }
    if (fread(preds, 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "[ERROR] Truncated predictions file\n");
        free(preds); fclose(f); return NULL;
    }
    fclose(f);
    *n_out = (int)n;
    if (offset_out) *offset_out = (int)off;
    return preds;
}

/* ── Class names (auto-selected via KI_PX) ─────────────────────── */
#define N_CLASS 10
static const char *class_names[N_CLASS] = {
#if KI_PX == 784
    "0","1","2","3","4","5","6","7","8","9"
#else
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
#endif
};
static const char *class_colors[N_CLASS] = {
    "#e6194b","#3cb44b","#ffe119","#4363d8","#f58231",
    "#911eb4","#42d4f4","#f032e6","#bfef45","#469990"
};

/* ═══════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[]) {
    const char *pred_path = NULL;
    const char *out_dir = "vis";
    int max_samples = 0;  /* 0 = all */
    int numH = 10;        /* images per row */

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--predictions") == 0 && i + 1 < argc) {
            pred_path = argv[++i];
        } else if (strcmp(argv[i], "--export") == 0 && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (strcmp(argv[i], "--max") == 0 && i + 1 < argc) {
            max_samples = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--numH") == 0 && i + 1 < argc) {
            numH = atoi(argv[++i]);
            if (numH < 1) numH = 1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s --predictions FILE [--export DIR] [--max N] [--numH N]\n", argv[0]);
            printf("  Export PNG + index.html (error-highlighted, sorted by class).\n");
            printf("  --max N:  process only first N samples (default: all)\n");
            printf("  --numH N: images per row (default: 10)\n");
            return 0;
        } else {
            fprintf(stderr, "[ERROR] Unknown: %s\n", argv[i]);
            return 1;
        }
    }

    if (!pred_path) {
        fprintf(stderr, "[ERROR] Need --predictions FILE\n");
        return 1;
    }

    /* Load predictions */
    int total_preds, start_offset = 0;
    uint8_t *preds = load_predictions(pred_path, &total_preds, &start_offset);
    if (!preds) return 1;

    /* Load dataset */
    ki_Dataset data = {0};
    if (ki_dataset_read(&data) != 0) { free(preds); return 1; }
    if (data.pixels != KI_PX) {
        fprintf(stderr, "[FATAL] Expected %d pixels, got %d\n", KI_PX, data.pixels);
        ki_dataset_free(&data); free(preds); return 1;
    }

    /* Use min of predictions, dataset, and --max */
    int N = total_preds;
    if (N > data.num_images) N = data.num_images;
    if (max_samples > 0 && max_samples < N) N = max_samples;

    int img_w = (KI_PX == 784) ? 28 : 32;
    int img_h = img_w;  /* square images */
    int is_mnist = (KI_PX == 784);

    printf("══╡ VIS ERRORS ╞════════════════════════════════════════════\n");
    printf("  Predictions: %s  (%d/%d samples)  Output: %s/\n",
           pred_path, N, total_preds, out_dir);

    /* Create output dir */
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", out_dir);
    if (system(cmd) != 0) { ki_dataset_free(&data); free(preds); return 1; }

    printf("══╡ VIS ERRORS ╞════════════════════════════════════════════\n");
    printf("  Predictions: %s  (%d samples)  Output: %s/\n",
           pred_path, N, out_dir);

    /* Allocate samples-by-class index */
    typedef struct { int idx; int pred; } SampleInfo;
    SampleInfo *by_class[N_CLASS];
    int by_class_n[N_CLASS];
    for (int c = 0; c < N_CLASS; c++) {
        by_class[c] = NULL;
        by_class_n[c] = 0;
    }

    /* Confusion matrix [true][pred] */
    int confusion[N_CLASS][N_CLASS];
    memset(confusion, 0, sizeof(confusion));

    /* Collect ALL samples, group by true class + fill confusion matrix */
    for (int s = 0; s < N; s++) {
        int ds = s + start_offset;  /* actual dataset index */
        int true_k = (int)data.y[(size_t)ds];
        if (true_k < 0 || true_k >= N_CLASS) continue;
        int pred_k = (int)preds[s];  /* always valid: s < total_preds = N */
        confusion[true_k][pred_k]++;
        int old_n = by_class_n[true_k];
        by_class[true_k] = (SampleInfo *)realloc(by_class[true_k],
                            (size_t)(old_n + 1) * sizeof(SampleInfo));
        by_class[true_k][old_n].idx = ds;  /* store actual dataset index */
        by_class[true_k][old_n].pred = pred_k;
        by_class_n[true_k]++;
    }

    /* ── Image dimension constants ─────────────────────────────── */
    int px_per_plane = is_mnist ? img_w * img_h : img_w * img_h;  /* 784 or 1024 */

    /* Generate PNGs per sample */
    for (int c = 0; c < N_CLASS; c++) {
        for (int i = 0; i < by_class_n[c]; i++) {
            int s = by_class[c][i].idx;
            int pred_k = by_class[c][i].pred;
            int is_err = (pred_k >= 0 && pred_k != c);

            /* Sample subdir — larger buffer for snprintf safety */
            char samp_dir[320];
            snprintf(samp_dir, sizeof(samp_dir), "%s/%d", out_dir, s);
            snprintf(cmd, sizeof(cmd), "mkdir -p %s", samp_dir);
            if (system(cmd) != 0) { ki_dataset_free(&data); free(preds); return 1; }

            char img_path[512];

            if (is_mnist) {
                /* ── MNIST: extract grayscale pixels, write img.png ── */
                uint8_t *px = (uint8_t *)malloc((size_t)img_w * (size_t)img_h);
                for (int p = 0; p < KI_PX; p++) {
                    px[p] = data.X_raw[(size_t)s * (size_t)KI_PX + (size_t)p];
                }
                snprintf(img_path, sizeof(img_path), "%s/img.png", samp_dir);
                ki_write_png(img_path, px, NULL, NULL, img_w, img_h);
                free(px);
            } else {
                /* ── CIFAR: extract RGB planes, write ORIG.png ── */
                size_t base = (size_t)s * (size_t)KI_PX;
                uint8_t *r = (uint8_t *)malloc((size_t)px_per_plane);
                uint8_t *g = (uint8_t *)malloc((size_t)px_per_plane);
                uint8_t *b = (uint8_t *)malloc((size_t)px_per_plane);
                for (int p = 0; p < px_per_plane; p++) {
                    r[p] = data.X_raw[base + (size_t)p];
                    g[p] = data.X_raw[base + (size_t)px_per_plane + (size_t)p];
                    b[p] = data.X_raw[base + 2 * (size_t)px_per_plane + (size_t)p];
                }
                snprintf(img_path, sizeof(img_path), "%s/ORIG.png", samp_dir);
                ki_write_png(img_path, r, g, b, img_w, img_h);
                free(r); free(g); free(b);
            }

            printf("  Sample %d  true=%s  pred=%s  %s\n",
                   s, class_names[c],
                   pred_k >= 0 ? class_names[pred_k] : "?",
                   is_err ? "❌" : "✓");
        }
    }

    /* ── Build index.html (grid layout) ──────────────────────────── */
    char html_path[512];
    snprintf(html_path, sizeof(html_path), "%s/index.html", out_dir);
    FILE *f = fopen(html_path, "w");
    if (!f) { fprintf(stderr, "[ERROR] Cannot write %s\n", html_path); return 1; }

    fprintf(f, "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n");
    fprintf(f, "<meta charset=\"UTF-8\">\n");
    fprintf(f, "<title>Predictions (all %d samples)</title>\n", N);
    fprintf(f, "<style>\n");
    fprintf(f, "  body { font-family: -apple-system,sans-serif; margin: 20px; background: #fafafa; }\n");
    fprintf(f, "  h1 { font-size: 20px; }\n");
    fprintf(f, "  .summary { font-size: 13px; color: #666; margin-bottom: 16px; }\n");
    fprintf(f, "  .class-header { font-size: 16px; font-weight: bold; margin: 20px 0 8px; padding: 6px 12px;"
               " border-radius: 4px; display: inline-block; }\n");
    /* Grid: images in rows, wrapping */
    fprintf(f, "  .grid { display: flex; flex-wrap: wrap; gap: 2px; margin-bottom: 24px; }\n");
    fprintf(f, "  .cell { width: %d%%%%; text-align: center; position: relative; }\n",
            100 / numH);
    fprintf(f, "  img { width: 100%%%%; max-width: %dpx; height: auto;"
               " image-rendering: pixelated; border-radius: 2px;"
               " border: 2px solid #e0e0e0; display: block; margin: 0 auto; }\n",
            is_mnist ? 56 : 64);
    fprintf(f, "  img.err { border-color: #e53935; }\n");
    fprintf(f, "  .idx { font-size: 9px; color: #999; line-height: 1; }\n");
    fprintf(f, "  .badge-err { position: absolute; top: 0; right: 0;"
               " background: #e53935; color: #fff; font-size: 9px;"
               " padding: 1px 4px; border-radius: 0 2px 0 2px; }\n");
    fprintf(f, "</style>\n</head>\n<body>\n");
    fprintf(f, "<h1>%s (%d samples, %d/row)</h1>\n",
            is_mnist ? "MNIST Predictions" : "CIFAR-10 Predictions", N, numH);
    fprintf(f, "<p class=\"summary\">Sorted by true class. Red border = error.</p>\n");

    /* ── Confusion matrix (heatmap) ─────────────────────────────── */
    {   int max_conf = 0;
        for (int r = 0; r < N_CLASS; r++)
            for (int cc = 0; cc < N_CLASS; cc++)
                if (confusion[r][cc] > max_conf) max_conf = confusion[r][cc];

        fprintf(f, "<h2>Confusion Matrix</h2>\n");
        fprintf(f, "<table style=\"border-collapse:collapse;font-size:12px;margin-bottom:20px;"
                   "table-layout:fixed;\">\n");
        fprintf(f, "<colgroup><col style=\"width:64px;\">");
        for (int cc = 0; cc < N_CLASS; cc++)
            fprintf(f, "<col style=\"width:44px;\">");
        fprintf(f, "<col style=\"width:44px;\"></colgroup>\n");
        fprintf(f, "<tr><th style=\"padding:2px 4px;text-align:center;\"><span style=\"display:block;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;max-width:64px;\">true \\ pred</span></th>");
        for (int cc = 0; cc < N_CLASS; cc++)
            fprintf(f, "<th style=\"padding:2px 4px;text-align:center;\">"
                       "<span style=\"display:block;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;max-width:44px;\">%s</span></th>",
                    class_names[cc]);
        fprintf(f, "<th style=\"padding:2px 4px;text-align:center;\">err%%</th></tr>\n");

        for (int r = 0; r < N_CLASS; r++) {
            int row_total = 0, row_err = 0;
            for (int cc = 0; cc < N_CLASS; cc++) {
                row_total += confusion[r][cc];
                if (cc != r) row_err += confusion[r][cc];
            }
            float err_pct = row_total > 0 ? (float)row_err * 100.0f / (float)row_total : 0.0f;
            /* Row label */
            fprintf(f, "<tr><td style=\"padding:2px 4px;text-align:center;font-weight:bold;\">"
                       "<span style=\"display:block;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;max-width:64px;\">%s</span></td>",
                    class_names[r]);
            for (int cc = 0; cc < N_CLASS; cc++) {
                int v = confusion[r][cc];
                /* Heat: white (0) → red (max). Diagonal in green tones. */
                int intensity = max_conf > 0 ? (v * 255) / max_conf : 0;
                int rv, gv, bv;
                if (cc == r) {
                    /* Diagonal: green intensity */
                    rv = 220 - intensity * 200 / 255;
                    gv = 245;
                    bv = 220 - intensity * 200 / 255;
                } else {
                    /* Off-diagonal: red intensity */
                    rv = 245;
                    gv = 245 - intensity * 200 / 255;
                    bv = 245 - intensity * 200 / 255;
                }
                if (rv < 0) { rv = 0; } if (gv < 0) { gv = 0; } if (bv < 0) { bv = 0; }
                fprintf(f, "<td style=\"padding:2px 4px;text-align:center;"
                           "background:rgb(%d,%d,%d);\">%d</td>",
                        rv, gv, bv, v);
            }
            /* Error percentage */
            fprintf(f, "<td style=\"padding:2px 4px;text-align:center;"
                       "color:%s;\">%.0f%%</td></tr>\n",
                    err_pct > 50.0f ? "#e53935" : (err_pct > 20.0f ? "#ef6c00" : "#555"),
                    err_pct);
        }
        fprintf(f, "</table>\n");
    }

    for (int c = 0; c < N_CLASS; c++) {
        if (by_class_n[c] == 0) continue;
        int n_ok = 0, n_err = 0;
        for (int i = 0; i < by_class_n[c]; i++) {
            if (by_class[c][i].pred == c) n_ok++; else n_err++;
        }
        float pct = (float)n_ok * 100.0f / (float)(n_ok + n_err);
        fprintf(f, "<div class=\"class-header\" style=\"background:%s20;border-left:4px solid %s;\">"
                   "Class %d: %s &mdash; %d/%d correct (%.0f%%)</div>\n",
                class_colors[c], class_colors[c], c, class_names[c], n_ok, n_ok + n_err, pct);
        fprintf(f, "<div class=\"grid\">\n");

        for (int i = 0; i < by_class_n[c]; i++) {
            int s = by_class[c][i].idx;
            int pred_k = by_class[c][i].pred;
            int is_err = (pred_k >= 0 && pred_k != c);
            char img_name[32];
            snprintf(img_name, sizeof(img_name), "%s.png",
                     is_mnist ? "img" : "ORIG");
            fprintf(f, "  <div class=\"cell\">\n");
            fprintf(f, "    <a href=\"%d/%s\">"
                       "<img src=\"%d/%s\" alt=\"s%d\" class=\"%s\"></a>\n",
                    s, img_name, s, img_name, s, is_err ? "err" : "");
            fprintf(f, "    <div class=\"idx\">#%d", s);
            if (is_err)
                fprintf(f, " <span style=\"color:#e53935;font-weight:bold;\">"
                           "&#10007;%s</span>", class_names[pred_k]);
            else
                fprintf(f, " &#10003;");
            fprintf(f, "</div>\n");
            fprintf(f, "  </div>\n");
        }
        fprintf(f, "</div>\n");
    }

    fprintf(f, "</body>\n</html>\n");
    fclose(f);
    printf("  -> %s\n", html_path);
    printf("\n══╡ DONE ╞════════════════════════════════════════════════════\n");

    for (int c = 0; c < N_CLASS; c++) free(by_class[c]);
    ki_dataset_free(&data);
    free(preds);
    return 0;
}
