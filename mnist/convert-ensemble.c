/*
 * otto-score-ifc/mnist/convert-ensemble.c -- Convert old .ens to v4 format
 * ========================================================================
 *
 * Reads .ens archives (any version v1..v3), computes the ensemble eval
 * accuracy from the stored scores + labels, and rewrites them as v4
 * files with embedded timestamp and ensemble_eval in the header.
 *
 * Usage:
 *   convert-ensemble SRC_DIR [DST_DIR]
 *
 *   SRC_DIR    Directory containing .ens files to convert
 *   DST_DIR    Output directory (default: SRC_DIR, in-place)
 *
 * New filename format:
 *   H..._SD{seed}_F4_TS{timestamp}.ens
 *
 * Build:
 *   gcc -O3 -o convert-ensemble convert-ensemble.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <errno.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <sys/stat.h>
#include <time.h>

#ifndef KI_NCLASSES
#define KI_NCLASSES 10
#endif

/* -- Read one .ens, compute ensemble eval, write v4 ------------ */
static int convert_file(const char *src_path, const char *dst_dir) {
    FILE *f = fopen(src_path, "rb");
    if (!f) { fprintf(stderr, "  [ERROR] Cannot open %s\n", src_path); return -1; }

    /* Get source file basename and mtime */
    const char *base = strrchr(src_path, '/');
    base = base ? base + 1 : src_path;

    struct stat st;
    time_t file_mtime = 0;
    if (stat(src_path, &st) == 0) file_mtime = st.st_mtime;

    /* -- Read header field-by-field ------------------------- */
    uint32_t hdr_base[11];
    if (fread(hdr_base, sizeof(uint32_t), 11, f) != 11) { fclose(f); return -1; }

    uint32_t magic     = hdr_base[0];
    uint32_t ver       = hdr_base[1];
    uint32_t n_test    = hdr_base[2];
    uint32_t n_classes = hdr_base[3];
    uint32_t n_members = hdr_base[4];
    uint32_t hidden    = hdr_base[5];
    uint32_t epochs    = hdr_base[6];
    uint32_t split_vn  = hdr_base[7];
    uint32_t split_hn  = hdr_base[8];
    float    target_err = 0.0f;
    memcpy(&target_err, &hdr_base[9], sizeof(float));
    uint32_t seed      = hdr_base[10];

    if (magic != 0x454E534D) { fclose(f); return -1; }
    if (ver < 1 || ver > 4)  { fprintf(stderr, "  [SKIP] %s: version %u unsupported\n", base, ver); fclose(f); return -1; }

    /* v3+: read timestamp */
    int64_t timestamp = 0;
    if (ver >= 3) {
        if (fread(&timestamp, 8, 1, f) != 1) { fclose(f); return -1; }
    }
    if (timestamp == 0) timestamp = (int64_t)file_mtime;

    /* v4: skip ensemble_eval (will recompute) */
    float old_eval = 0.0f;
    if (ver >= 4) {
        if (fread(&old_eval, 4, 1, f) != 1) { fclose(f); return -1; }
    }

    /* If already v4 with eval and correct filename, skip */
    if (ver >= 4 && old_eval > 0.0f) {
        fclose(f);
        return 1;  /* already converted */
    }

    /* -- Read per-member metadata (v2/v3/v4: color,enc_type,enc_width,pad) - */
    int is_v2 = (ver >= 2);
    uint8_t *meta_color    = NULL;
    uint8_t *meta_enc_type = NULL;
    uint8_t *meta_enc_wid  = NULL;
    if (is_v2) {
        meta_color    = (uint8_t *)malloc((size_t)n_members);
        meta_enc_type = (uint8_t *)malloc((size_t)n_members);
        meta_enc_wid  = (uint8_t *)malloc((size_t)n_members);
        if (!meta_color || !meta_enc_type || !meta_enc_wid) {
            fprintf(stderr, "[FATAL] OOM\n"); exit(1);
        }
        for (uint32_t m = 0; m < n_members; m++) {
            uint8_t col, typ, wid, pad;
            if (fread(&col, 1, 1, f) != 1 || fread(&typ, 1, 1, f) != 1 ||
                fread(&wid, 1, 1, f) != 1 || fread(&pad, 1, 1, f) != 1) {
                fprintf(stderr, "  [ERROR] %s: short metadata\n", base);
                free(meta_color); free(meta_enc_type); free(meta_enc_wid);
                fclose(f); return -1;
            }
            meta_color[m]    = col;
            meta_enc_type[m] = typ;
            meta_enc_wid[m]  = wid;
        }
    }

    /* -- Read all member scores -------------------------------- */
    size_t score_sz = (size_t)n_test * (size_t)n_classes;
    int64_t *all_scores = (int64_t *)calloc(score_sz, sizeof(int64_t));
    if (!all_scores) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }

    for (uint32_t m = 0; m < n_members; m++) {
        int64_t *member_sc = (int64_t *)malloc(score_sz * sizeof(int64_t));
        if (!member_sc) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
        if (fread(member_sc, sizeof(int64_t), score_sz, f) != score_sz) {
            fprintf(stderr, "  [ERROR] %s: short read member %u\n", base, m);
            free(member_sc); free(all_scores);
            fclose(f); return -1;
        }
        /* Accumulate into ensemble scores */
        for (size_t i = 0; i < score_sz; i++)
            all_scores[i] += member_sc[i];
        free(member_sc);
    }

    /* -- Read ground truth labels ------------------------------ */
    uint8_t *labels = (uint8_t *)malloc((size_t)n_test);
    if (!labels) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
    if (fread(labels, 1, (size_t)n_test, f) != (size_t)n_test) {
        fprintf(stderr, "  [ERROR] %s: no labels found\n", base);
        free(labels); free(all_scores); fclose(f); return -1;
    }
    fclose(f);

    /* -- Compute ensemble eval accuracy ------------------------ */
    int correct = 0;
    for (uint32_t s = 0; s < n_test; s++) {
        const int64_t *row = all_scores + (size_t)s * (size_t)n_classes;
        int pred = 0;
        for (uint32_t k = 1; k < n_classes; k++)
            if (row[k] > row[pred]) pred = (int)k;
        if (pred == (int)labels[s]) correct++;
    }
    float ensemble_eval = (float)correct * 100.0f / (float)n_test;

    /* -- Build v4 filename ------------------------------------- */
    int te_int = (int)(target_err * 100.0f + 0.5f);
    char new_name[512];
    snprintf(new_name, sizeof(new_name),
             "H%u_EP%u_VN%u_HN%u_TE%d_SD%u_F4_TS%" PRId64 ".ens",
             hidden, epochs, split_vn, split_hn, te_int, seed, timestamp);

    char dst_path[1024];
    snprintf(dst_path, sizeof(dst_path), "%s/%s", dst_dir, new_name);

    /* -- Write v4 file ----------------------------------------- */
    FILE *out = fopen(dst_path, "wb");
    if (!out) { fprintf(stderr, "  [ERROR] Cannot write %s\n", dst_path); return -1; }

    /* Header */
    uint32_t v4_magic = 0x454E534D;
    uint32_t v4_ver   = 4;
    uint32_t v4_n_test    = n_test;
    uint32_t v4_n_classes = n_classes;
    uint32_t v4_n_members = n_members;
    uint32_t v4_hidden    = hidden;
    uint32_t v4_epochs    = epochs;
    uint32_t v4_split_vn  = split_vn;
    uint32_t v4_split_hn  = split_hn;
    float    v4_tgt_err   = target_err;
    uint32_t v4_seed      = seed;
    int64_t  v4_stamp     = timestamp;
    float    v4_eval      = ensemble_eval;

    fwrite(&v4_magic, 4, 1, out); fwrite(&v4_ver, 4, 1, out);
    fwrite(&v4_n_test, 4, 1, out); fwrite(&v4_n_classes, 4, 1, out);
    fwrite(&v4_n_members, 4, 1, out);
    fwrite(&v4_hidden, 4, 1, out); fwrite(&v4_epochs, 4, 1, out);
    fwrite(&v4_split_vn, 4, 1, out); fwrite(&v4_split_hn, 4, 1, out);
    fwrite(&v4_tgt_err, 4, 1, out); fwrite(&v4_seed, 4, 1, out);
    fwrite(&v4_stamp, 8, 1, out);
    fwrite(&v4_eval, 4, 1, out);

    /* Per-member metadata */
    if (is_v2) {
        for (uint32_t m = 0; m < n_members; m++) {
            uint8_t pad = 0;
            fwrite(&meta_color[m], 1, 1, out);
            fwrite(&meta_enc_type[m], 1, 1, out);
            fwrite(&meta_enc_wid[m], 1, 1, out);
            fwrite(&pad, 1, 1, out);
        }
    } else {
        /* v1: no metadata, write zeroed */
        for (uint32_t m = 0; m < n_members; m++) {
            uint8_t zero = 0;
            fwrite(&zero, 1, 4, out);
        }
    }

    /* Re-read and write per-member scores (need a second pass) */
    /* We already have all_scores but not per-member scores; re-open source */
    f = fopen(src_path, "rb");
    if (!f) { fprintf(stderr, "[ERROR] Cannot re-open %s\n", src_path); fclose(out); return -1; }

    /* Skip to score data: header + metadata */
    size_t hdr_size = 44;  /* base 11 u */
    if (ver >= 3) hdr_size += 8;
    if (ver >= 4) hdr_size += 4;
    if (is_v2) hdr_size += (size_t)n_members * 4;
    if (fseek(f, (long)hdr_size, SEEK_SET) != 0) { fclose(f); fclose(out); return -1; }

    for (uint32_t m = 0; m < n_members; m++) {
        int64_t *member_sc = (int64_t *)malloc(score_sz * sizeof(int64_t));
        if (!member_sc) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
        if (fread(member_sc, sizeof(int64_t), score_sz, f) != score_sz) {
            fprintf(stderr, "  [ERROR] %s: short re-read member %u\n", base, m);
            free(member_sc); fclose(f); fclose(out); return -1;
        }
        fwrite(member_sc, sizeof(int64_t), score_sz, out);
        free(member_sc);
    }
    fclose(f);

    /* Labels */
    fwrite(labels, 1, (size_t)n_test, out);

    fclose(out);

    printf("  [OK] %s  eval=%.1f%%  -> %s\n", base, ensemble_eval, new_name);

    free(all_scores); free(labels);
    free(meta_color); free(meta_enc_type); free(meta_enc_wid);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        printf("Usage: %s SRC_DIR [DST_DIR]\n", argv[0]);
        printf("\n");
        printf("Convert .ens archives (v1..v3) to v4 format with embedded timestamp + eval.\n");
        printf("\n");
        printf("  SRC_DIR   Directory with .ens files to convert\n");
        printf("  DST_DIR   Output directory (default: SRC_DIR, in-place)\n");
        printf("\n");
        printf("Files already in v4 format with eval > 0 are skipped.\n");
        return 0;
    }

    const char *src_dir = argv[1];
    const char *dst_dir = (argc >= 3) ? argv[2] : argv[1];

    /* Create dst dir if different from src */
    if (strcmp(src_dir, dst_dir) != 0) {
        char cmd[512];
        snprintf(cmd, sizeof(cmd), "mkdir -p %s", dst_dir);
        if (system(cmd) != 0) { fprintf(stderr, "[ERROR] Cannot create %s\n", dst_dir); return 1; }
    }

    DIR *d = opendir(src_dir);
    if (!d) { fprintf(stderr, "[ERROR] Cannot open %s\n", src_dir); return 1; }

    int n_total = 0, n_conv = 0, n_skip = 0;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        const char *ext = strrchr(de->d_name, '.');
        if (!ext || strcmp(ext, ".ens") != 0) continue;
        n_total++;

        char src_path[1024];
        snprintf(src_path, sizeof(src_path), "%s/%s", src_dir, de->d_name);

        int rc = convert_file(src_path, dst_dir);
        if (rc == 0) n_conv++;
        else if (rc == 1) n_skip++;
    }
    closedir(d);

    printf("\nDone: %d files, %d converted, %d already v4, %d in %s\n",
           n_total, n_conv, n_skip, n_conv, dst_dir);
    return 0;
}
