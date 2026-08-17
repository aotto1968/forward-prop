/*
 * mnist-1/mlp-otto-xform-samples.c — Xform Sample Viewer
 * ========================================================
 *
 * Lädt MNIST/CIFAR/Fashion, wendet alle aktiven --xform auf einen
 * Sample-Index an und exportiert als PNG + index.html.
 *
 * Usage:
 *   ./mnist-mlp-otto-xform-samples.exe --idx 42 --xform all --export out/xf
 *   → out/xf/42/index.html  (alle 20 xforms, browser-ready)
 *
 * Symlink for CIFAR/Fashion:
 *   ln -s ../mnist-1/mlp-otto-xform-samples.c cifar-1/
 *   make cifar-mlp-otto-xform-samples.exe
 *   ./cifar-mlp-otto-xform-samples.exe --idx 0 --xform performance
 */
#include "ki-config.h"   /* Konvention (wie Trainer): ki-config VOR ki-common —
                            * definiert KI_SWEEP_PERFORMANCE_XFORM bevor
                            * ki-encoding.h den Fallback setzt (sonst -Werror
                            * redefined, make all-all). */
/* libtprint BEFORE ki-common.h: print_confusion_debug() in ki-common.h uses
 * TPrint (2026-08-14). */
#include "../lib/tprint.h"
#include "ki-common.h"
#include <inttypes.h>
#include <zlib.h>

/* ── PNG writer (grayscale + RGB, uses zlib for IDAT) ─────────── */

static void write_be32(FILE *f, uint32_t v) {
    uint8_t buf[4] = {
        (uint8_t)((v >> 24) & 0xFF),
        (uint8_t)((v >> 16) & 0xFF),
        (uint8_t)((v >>  8) & 0xFF),
        (uint8_t)( v        & 0xFF)
    };
    fwrite(buf, 1, 4, f);
}

static void png_chunk(FILE *f, const char type[4],
                       const void *data, size_t len) {
    unsigned long crc = crc32(0L, NULL, 0);
    write_be32(f, (uint32_t)len);
    crc = crc32(crc, (const unsigned char *)type, 4);
    fwrite(type, 1, 4, f);
    if (len > 0 && data) {
        crc = crc32(crc, (const unsigned char *)data, (unsigned int)len);
        fwrite(data, 1, len, f);
    }
    write_be32(f, (uint32_t)(crc & 0xFFFFFFFFUL));
}

static void write_png_gray(const char *path, const uint8_t *pixels,
                            int w, int h) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[ERROR] Cannot write %s\n", path); return; }

    uint8_t sig[8] = {137,80,78,71,13,10,26,10};
    fwrite(sig, 1, 8, f);

    uint8_t ihdr[13];
    for (int i = 0; i < 4; i++) {
        ihdr[3-i] = (uint8_t)((uint32_t)w >> (i*8));
        ihdr[7-i] = (uint8_t)((uint32_t)h >> (i*8));
    }
    ihdr[8]  = 8;  ihdr[9]  = 0;  /* grayscale */
    ihdr[10] = 0;  ihdr[11] = 0;  ihdr[12] = 0;
    png_chunk(f, "IHDR", ihdr, 13);

    size_t row_bytes = (size_t)w;
    size_t raw_size = (size_t)h * (1 + row_bytes);
    uint8_t *raw = (uint8_t *)malloc(raw_size);
    for (int y = 0; y < h; y++) {
        raw[y * (1 + row_bytes)] = 0;
        memcpy(raw + y * (1 + row_bytes) + 1, pixels + (size_t)y * w, row_bytes);
    }

    uLongf comp_len = compressBound(raw_size);
    uint8_t *comp = (uint8_t *)malloc(comp_len);
    if (compress(comp, &comp_len, raw, raw_size) != Z_OK) {
        fprintf(stderr, "[ERROR] PNG compression: %s\n", path);
        free(raw); free(comp); fclose(f); return;
    }
    png_chunk(f, "IDAT", comp, comp_len);
    png_chunk(f, "IEND", NULL, 0);
    fclose(f); free(raw); free(comp);
}

static void write_png_rgb(const char *path,
                           const uint8_t *r, const uint8_t *g,
                           const uint8_t *b, int w, int h) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[ERROR] Cannot write %s\n", path); return; }

    uint8_t sig[8] = {137,80,78,71,13,10,26,10};
    fwrite(sig, 1, 8, f);

    uint8_t ihdr[13];
    for (int i = 0; i < 4; i++) {
        ihdr[3-i] = (uint8_t)((uint32_t)w >> (i*8));
        ihdr[7-i] = (uint8_t)((uint32_t)h >> (i*8));
    }
    ihdr[8]  = 8;  ihdr[9]  = 2;  /* RGB */
    ihdr[10] = 0;  ihdr[11] = 0;  ihdr[12] = 0;
    png_chunk(f, "IHDR", ihdr, 13);

    size_t row_bytes = (size_t)w * 3;
    size_t raw_size = (size_t)h * (1 + row_bytes);
    uint8_t *raw = (uint8_t *)malloc(raw_size);
    for (int y = 0; y < h; y++) {
        raw[y * (1 + row_bytes)] = 0;
        for (int x = 0; x < w; x++) {
            size_t off = (size_t)y * w + (size_t)x;
            size_t dst = y * (1 + row_bytes) + 1 + (size_t)x * 3;
            raw[dst + 0] = r[off];
            raw[dst + 1] = g[off];
            raw[dst + 2] = b[off];
        }
    }

    uLongf comp_len = compressBound(raw_size);
    uint8_t *comp = (uint8_t *)malloc(comp_len);
    if (compress(comp, &comp_len, raw, raw_size) != Z_OK) {
        fprintf(stderr, "[ERROR] PNG compression: %s\n", path);
        free(raw); free(comp); fclose(f); return;
    }
    png_chunk(f, "IDAT", comp, comp_len);
    png_chunk(f, "IEND", NULL, 0);
    fclose(f); free(raw); free(comp);
}

/* ── Global args (needed by ki_parse_args) ──────────────────── */

ki_Args aa = {
    .enc_size          = KI_ENC_WIDTH_DEFAULT,
    .seed_splitmix     = 1,
    .rows_mode         = 0,
    .member_threshold  = 0,
    .xforms            = (1ull << KI_XFORM_COUNT) - 1ull,  /* default: all xforms */
};

/* ── CLI ─────────────────────────────────────────────────────── */

/* ── Render ONE sample: apply all active xforms, write PNGs only ──
 * PNGs land in "<out_dir>/<subdir>/<xform>.png" (subdir = sample id).
 * Returns the number of xform images written (0 on error). */
static int render_sample_pngs(const ki_Dataset *data, int idx,
                              const int *xf_list, int n_xf,
                              const char *out_dir, const char *subdir) {
    int w = data->cols, h = data->rows, ch = KI_COLORS;
    int plane_sz = w * h;                /* pixels per color plane */
    int img_sz   = plane_sz * ch;        /* total pixels */
    const uint8_t *raw = data->X_raw + (size_t)idx * (size_t)img_sz;

    /* ── Create output dir ────────────────────────────────────── */
    char base_dir[256];
    snprintf(base_dir, sizeof(base_dir), "%s/%s", out_dir, subdir);
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", base_dir);
    if (system(cmd) != 0) return 0;

    /* ── Allocate buffer for one transformed image ──────────────── */
    uint8_t *xf_buf = (uint8_t *)malloc((size_t)img_sz);
    if (!xf_buf) return 0;

    int written = 0;
    for (int xi = 0; xi < n_xf; xi++) {
        int xf = xf_list[xi];
        const char *xname = ki_xform_name(xf);

        ki_xform_raw(xf_buf, raw, w, h, ch, xf);

        char png_path[512];
        snprintf(png_path, sizeof(png_path), "%s/%s.png", base_dir, xname);
        if (ch == 1 || KI_COLORS == 1) {
            write_png_gray(png_path, xf_buf, w, h);
        } else {
            write_png_rgb(png_path, xf_buf, xf_buf + plane_sz,
                          xf_buf + 2 * plane_sz, w, h);
        }
        written++;
    }
    free(xf_buf);
    return written;
}

/* ── Write ONE gallery index.html ──────────────────────────────────
 * Rows = samples, columns = xforms. Each sample-id is a subdirectory
 * holding "<xform>.png". Also writes the sample list into out_dir so a
 * second run appends (same layout: sample id as subdir, xform pngs). */
static int write_gallery_html(const char *out_dir, const char *title,
                              const char **sample_ids, int n_samples,
                              const int *xf_list, int n_xf,
                              const char *subtitle) {
    char path[512];
    snprintf(path, sizeof(path), "%s/index.html", out_dir);
    FILE *f = fopen(path, "w");
    if (!f) return 0;

    fprintf(f, "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n");
    fprintf(f, "<meta charset=\"UTF-8\">\n");
    fprintf(f, "<title>%s — %s</title>\n", KI_DATASET_NAME, title);
    fprintf(f, "<style>\n");
    fprintf(f, "  body { font-family: -apple-system,BlinkMacSystemFont,sans-serif; "
               "margin: 20px; background: #fafafa; }\n");
    fprintf(f, "  h1 { font-size: 20px; }\n");
    fprintf(f, "  .subtitle { font-size: 13px; color: #666; margin-bottom: 14px; }\n");
    fprintf(f, "  table { border-collapse: collapse; margin-bottom: 30px; }\n");
    fprintf(f, "  th, td { padding: 4px 6px; text-align: center; "
               "vertical-align: middle; }\n");
    fprintf(f, "  th { background: #333; color: #fff; position: sticky; "
               "top: 0; font-size: 12px; }\n");
    fprintf(f, "  th:first-child, td:first-child { position: sticky; left: 0; "
               "background: #fff; }\n");
    fprintf(f, "  th:first-child { background: #444; }\n");
    fprintf(f, "  td:first-child { font-family: monospace; font-size: 12px; "
               "white-space: nowrap; }\n");
    fprintf(f, "  img { width: 96px; height: 96px; image-rendering: pixelated; "
               "border: 1px solid #ccc; border-radius: 2px; display: block; }\n");
    fprintf(f, "  tr:nth-child(even) { background: #fff; }\n");
    fprintf(f, "  tr:nth-child(odd)  { background: #f0f0f0; }\n");
    fprintf(f, "  tr:hover { background: #e0e8f5; }\n");
    fprintf(f, "</style>\n</head>\n<body>\n");
    fprintf(f, "<h1>%s — %s</h1>\n", KI_DATASET_NAME, title);
    fprintf(f, "<p class=\"subtitle\">%s &mdash; %d sample(s), %d xform(s) "
               "per row</p>\n", subtitle, n_samples, n_xf);

    /* Header: first cell = sample, then one column per xform */
    fprintf(f, "<table>\n<tr><th>sample</th>");
    for (int xi = 0; xi < n_xf; xi++)
        fprintf(f, "<th>%s</th>", ki_xform_name(xf_list[xi]));
    fprintf(f, "</tr>\n");

    /* One row per sample: sample-id label, then its xform images */
    for (int si = 0; si < n_samples; si++) {
        fprintf(f, "<tr>\n");
        fprintf(f, "  <td>%s</td>\n", sample_ids[si]);
        for (int xi = 0; xi < n_xf; xi++) {
            const char *xname = ki_xform_name(xf_list[xi]);
            fprintf(f, "  <td><a href=\"%s/%s.png\">"
                       "<img src=\"%s/%s.png\" alt=\"%s-%s\"></a></td>\n",
                       sample_ids[si], xname,
                       sample_ids[si], xname,
                       sample_ids[si], xname);
        }
        fprintf(f, "</tr>\n");
    }

    fprintf(f, "</table>\n");
    fprintf(f, "</body>\n</html>\n");
    fclose(f);
    printf("  -> %s\n", path);
    return 1;
}

int main(int argc, char *argv[]) {
    int idx = -1;
    int target = -1;
    int max_count = 0;                   /* --max N: cap class samples (0 = all) */
    const char *out_dir = "out";

    /* Parse --idx/--target/--export FIRST (before ki_parse_args) */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--idx") == 0 && i + 1 < argc) {
            idx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--target") == 0 && i + 1 < argc) {
            target = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max") == 0 && i + 1 < argc) {
            max_count = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--export") == 0 && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [--idx N | --target K [--max M]] "
                   "[--xform X] [--export DIR]\n", argv[0]);
            printf("  --idx N       Sample index to visualize\n");
            printf("  --target K    Visualize ALL samples of class K "
                   "(e.g. --target 6 = Shirt)\n");
            printf("  --max M       With --target: cap at M samples (default: all)\n");
            printf("  --xform X     Xform set (default: all)\n");
            printf("  --export DIR  Output directory (default: out)\n");
            return 0;
        }
    }

    /* Now let ki_parse_args handle the rest (--xform, etc.) */
    /* Filter out --idx/--target/--max/--export before passing to ki_parse_args */
    const char **av = (const char **)malloc((size_t)argc * sizeof(char *));
    int ac = 0;
    av[ac++] = argv[0];
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "--idx") == 0 || strcmp(argv[i], "--target") == 0 ||
             strcmp(argv[i], "--max") == 0) && i + 1 < argc) {
            i++; /* skip --idx/--target/--max N */
        } else if (strcmp(argv[i], "--export") == 0 && i + 1 < argc) {
            i++; /* skip --export DIR */
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            /* already handled above */
        } else {
            av[ac++] = argv[i];
        }
    }
    /* Temporarily replace argv for ki_parse_args */
    ki_parse_args(ac, (char **)av);
    free(av);

    if (idx < 0 && target < 0) {
        fprintf(stderr, "[ERROR] Use --idx N or --target K\n");
        return 1;
    }
    if (target >= KI_NCLASSES) {
        fprintf(stderr, "[ERROR] --target %d out of range (0..%d)\n",
                target, KI_NCLASSES - 1);
        return 1;
    }

    /* ── Load dataset ─────────────────────────────────────────── */
    /* Zero-init first: ki_dataset_read() reads out->dry_run BEFORE the
     * memset inside — an uninitialized stack struct trips -Werror=
     * maybe-uninitialized (GCC-15, fashion ki-local.h). */
    ki_Dataset data;
    memset(&data, 0, sizeof(data));
    if (ki_dataset_read(&data) != 0) return 1;
    if (idx >= data.num_images) {
        fprintf(stderr, "[ERROR] idx=%d out of range (0..%d)\n",
                idx, data.num_images - 1);
        ki_dataset_free(&data); return 1;
    }

    /* ── Determine which xforms are active ────────────────────── */
    int xf_list[KI_XFORM_COUNT], n_xf = 0;
    for (int xf = 0; xf < KI_XFORM_COUNT; xf++)
        if (aa.xforms & (1ull << xf)) xf_list[n_xf++] = xf;
    if (n_xf < 1) { xf_list[0] = KI_XFORM_ID; n_xf = 1; }

    printf("══╡ XFORM SAMPLES ╞════════════════════════════════════════\n");
    printf("  Dataset: %s  %dx%dx%d\n", KI_DATASET_NAME,
           data.cols, data.rows, KI_COLORS);
    printf("  Xforms:  %d active\n", n_xf);
    printf("  Output:  %s/\n", out_dir);

    if (target >= 0) {
        /* ── --target K: gallery of class-K samples ────────────── */
        const char *cname = (target < KI_NCLASSES) ? ki_class_names[target] : "?";
        printf("  Target:  class %d (%s)\n", target, cname);

        /* First pass: count class samples to size the id list */
        int n_total = 0;
        for (int s = 0; s < data.num_images; s++)
            if ((int)data.y[s] == target) n_total++;
        if (max_count > 0 && n_total > max_count) n_total = max_count;

        const char **sample_ids = (const char **)calloc((size_t)n_total,
                                                        sizeof(char *));
        char (*id_buf)[64] = (char (*)[64])calloc((size_t)n_total, sizeof(char[64]));
        if (!sample_ids || !id_buf) {
            fprintf(stderr, "[ERROR] OOM\n");
            free(sample_ids); free(id_buf);
            ki_dataset_free(&data); return 1;
        }

        int n_done = 0;
        for (int s = 0; s < data.num_images && (max_count == 0 || n_done < max_count); s++) {
            if ((int)data.y[s] != target) continue;
            snprintf(id_buf[n_done], 64, "%d_%d", target, s);
            sample_ids[n_done] = id_buf[n_done];
            render_sample_pngs(&data, s, xf_list, n_xf, out_dir, id_buf[n_done]);
            n_done++;
        }

        char title[128], subtitle[256];
        snprintf(title, sizeof(title), "Class %d (%s) — xform gallery",
                 target, cname);
        snprintf(subtitle, sizeof(subtitle), "label=%d (%s), all samples",
                 target, cname);
        write_gallery_html(out_dir, title, sample_ids, n_done,
                           xf_list, n_xf, subtitle);
        printf("  Rendered %d class-%d samples\n", n_done, target);
        free(sample_ids); free(id_buf);
    } else {
        /* ── --idx N: single sample (one-row gallery) ──────────── */
        char subdir[64];
        snprintf(subdir, sizeof(subdir), "%d", idx);
        render_sample_pngs(&data, idx, xf_list, n_xf, out_dir, subdir);

        const char *ids[1] = { subdir };
        char title[128], subtitle[256];
        snprintf(title, sizeof(title), "Sample %d — xform gallery", idx);
        snprintf(subtitle, sizeof(subtitle), "label=%d", (int)data.y[idx]);
        write_gallery_html(out_dir, title, ids, 1, xf_list, n_xf, subtitle);
    }

    ki_dataset_free(&data);
    printf("\n══╡ DONE ╞════════════════════════════════════════════════\n");
    return 0;
}
