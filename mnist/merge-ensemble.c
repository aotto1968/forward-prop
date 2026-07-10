/*
 * otto-score-ifc/mnist/merge-ensemble.c -- Merge score archives to EN curve
 * ========================================================================
 *
 * Usage:
 *   merge-ensemble DIR [options]
 *
 * Reads all .ens files from DIR, accumulates scores per seed+member,
 * and computes accuracy for EN=1..total_members.
 *
 * Archive format (produced by --save-scores):
 *   Header: magic(4) ver(4) n_test(4) n_classes(4) n_members(4)
 *           hidden(4) epochs(4) split_vn(4) split_hn(4)
 *           target_err(4) seed(4)
 *   Data:   n_members . int64[n_test . n_classes]
 *           uint8[n_test]  (ground truth labels)
 *
 * Options:
 *   DIR         Directory containing .ens score archives
 *   --num N     Only combine the first N members (default: all)
 *   --save FILE Save cumulative accuracy to FILE (default: DIR/merge.dat)
 *   -h, --help  Show this help text
 *
 * Build:
 *   gcc -O3 -o merge-ensemble merge-ensemble.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <dirent.h>
#include <errno.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <sys/stat.h>

#ifndef KI_NCLASSES
#define KI_NCLASSES 10
#endif

/* -- Archive header (matches mlp-bin32-otto-trn.c) ------------ */
/* Fields are read individually (size differs between versions) */
typedef struct {
    uint32_t magic;       /* 0x454E534D ('ENSM') */
    uint32_t version;     /* 1, 2, 3, or 4 */
    uint32_t n_test;      /* eval samples */
    uint32_t n_classes;   /* KI_NCLASSES */
    uint32_t n_members;   /* members in this file */
    uint32_t hidden;      /* H */
    uint32_t epochs;      /* epochs */
    uint32_t split_vn;
    uint32_t split_hn;
    float    target_err;
    uint32_t seed;
    int64_t  timestamp;   /* v3+: embedded creation time (0 for v1/v2) */
    float    ensemble_eval; /* v4+: final ensemble eval accuracy % (0 for v1/v2/v3) */
} ScoreHeader;

/* -- Per-member score block ----------------------------------- */
typedef struct {
    int      seed;
    int      member_idx;
    int64_t *scores;      /* [n_test x n_classes] */
    /* Encoding info (v2 archives) */
    int      color;       /* channel index (0=R, 1=G, 2=B, etc.) */
    int      enc_type;    /* encoding type (KI_ENC_RAW=0, LIN7=1, etc.) */
    int      enc_width;   /* encoding width (8, 16, 32) */
    time_t   file_time;   /* file mtime (ctime for sorting) */
    int      total_members; /* total members in this file (for #/# separator) */
    float    file_eval;     /* ensemble eval accuracy % (v4+, 0 for older) */
    char     source_file[128]; /* basename of .ens archive */
} ScoreBlock;

/* -- Global state -------------------------------------------- */
static int    n_blocks = 0;
static int    n_blocks_cap = 0;
static ScoreBlock *blocks = NULL;
static int    g_n_test = 0;
static int    g_n_classes = 0;
static int    g_hidden = 0;
static int    g_epochs = 0;
static int    g_split_vn = 0;
static int    g_split_hn = 0;
static int    g_target_err_x100 = 0;
static uint8_t *g_labels = NULL;    /* ground truth, loaded once from first archive */
static int    g_sort_mode = 1;      /* 0=seed, 1=ctime (default) */
static int    g_filter_count = 0;   /* number of active filter patterns */
static char   g_filter_pat[64][32]; /* member labels to exclude (case-insensitive) */
static int    g_eval_active = 0;    /* 0=disabled, 1=eval threshold active */
static int    g_eval_cmp = 0;       /* 0:>, 1:>=, 2:<, 3:<=, 4:= */
static char   g_eval_op = '>';     /* operator char for display */
static float  g_eval_thresh = 0.0f; /* threshold value */

/* -- Comparators --------------------------------------------- */
static int cmp_by_seed(const void *a, const void *b) {
    const ScoreBlock *ba = (const ScoreBlock *)a;
    const ScoreBlock *bb = (const ScoreBlock *)b;
    if (ba->seed != bb->seed) return ba->seed - bb->seed;
    return ba->member_idx - bb->member_idx;
}

static int cmp_by_ctime(const void *a, const void *b) {
    const ScoreBlock *ba = (const ScoreBlock *)a;
    const ScoreBlock *bb = (const ScoreBlock *)b;
    if (ba->file_time != bb->file_time) 
        return (ba->file_time < bb->file_time) ? -1 : 1;
    if (ba->seed != bb->seed) return ba->seed - bb->seed;
    return ba->member_idx - bb->member_idx;
}

/* -- Encoding name (self-contained, matches ki-encoding.h) --- */
static const char *enc_name(int type) {
    switch (type) {
        case 0:  return "raw";
        case 1:  return "lin7";
        case 2:  return "lin";
        case 3:  return "down";
        case 4:  return "up";
        case 5:  return "mid";
        case 6:  return "log";
        case 7:  return "exp";
        case 8:  return "sig";
        default: return "?";
    }
}

/* -- Color name (matches ki-local.h) ------------------------- */
static const char *color_name(int c) {
    switch (c) {
        case 0:  return "M";    /* mnist grayscale */
        case 1:  return "R";
        case 2:  return "G";
        case 3:  return "B";
        case 4:  return "Y";
        case 5:  return "YL";
        case 6:  return "AL";
        case 7:  return "AM";
        case 8:  return "AP";
        case 9:  return "RG";
        case 10: return "RB";
        case 11: return "GB";
        case 12: return "BL";
        case 13: return "BM";
        case 14: return "BP";
        case 15: return "H";
        case 16: return "S";
        case 17: return "C";
        case 18: return "CL";
        case 19: return "CM";
        case 20: return "CP";
        case 21: return "edge";
        case 22: return "bin";
        default: return "?";
    }
}

/* -- Build member label: "color=encW" ------------------------- */
static void member_label(const ScoreBlock *b, char *buf, size_t sz) {
    snprintf(buf, sz, "%s=%s%d",
             color_name(b->color), enc_name(b->enc_type), b->enc_width);
}

/* -- Check if member matches any filter pattern -------------- */
static int member_is_filtered(const ScoreBlock *b) {
    if (g_filter_count == 0) return 0;
    char label[32];
    member_label(b, label, sizeof(label));
    for (int i = 0; i < g_filter_count; i++) {
        /* Exact match (case-insensitive) */
        if (strcasecmp(label, g_filter_pat[i]) == 0)
            return 1;
        /* Substring match (case-insensitive) — e.g. "sig8" matches "BM=sig8" */
        if (strcasestr(label, g_filter_pat[i]) != NULL)
            return 1;
    }
    return 0;
}

/* -- Load one .ens file, append blocks ----------------------- */
static int load_archive(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "  [WARN] Cannot open %s: %s\n", path, strerror(errno)); return -1; }

    /* Extract basename for display */
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;

    /* Get file modification time (fallback for v1/v2) */
    struct stat st;
    time_t file_mtime = 0;
    if (stat(path, &st) == 0) file_mtime = st.st_mtime;

    /* ── Read header field-by-field (version-dependent size) ─ */
    ScoreHeader hdr;
    memset(&hdr, 0, sizeof(hdr));

    /* All versions share: magic, version, n_test, n_classes, n_members,
       hidden, epochs, split_vn, split_hn, target_err, seed = 11 x uint32 */
    uint32_t hdr_base[11];
    if (fread(hdr_base, sizeof(uint32_t), 11, f) != 11) { fclose(f); return -1; }
    hdr.magic     = hdr_base[0];
    hdr.version   = hdr_base[1];
    hdr.n_test    = hdr_base[2];
    hdr.n_classes = hdr_base[3];
    hdr.n_members = hdr_base[4];
    hdr.hidden    = hdr_base[5];
    hdr.epochs    = hdr_base[6];
    hdr.split_vn  = hdr_base[7];
    hdr.split_hn  = hdr_base[8];
    hdr.target_err = 0.0f;
    memcpy(&hdr.target_err, &hdr_base[9], sizeof(float));
    hdr.seed      = hdr_base[10];

    if (hdr.magic != 0x454E534D) { fclose(f); return -1; }
    if (hdr.version < 1 || hdr.version > 4) {
        fprintf(stderr, "  [WARN] %s: unknown version %u\n", path, hdr.version);
        fclose(f); return -1;
    }

    /* v3+: read embedded timestamp (int64) */
    hdr.timestamp = 0;
    if (hdr.version >= 3) {
        if (fread(&hdr.timestamp, 8, 1, f) != 1) { fclose(f); return -1; }
    }

    /* v4+: read ensemble eval accuracy (float) */
    hdr.ensemble_eval = 0.0f;
    if (hdr.version >= 4) {
        if (fread(&hdr.ensemble_eval, 4, 1, f) != 1) { fclose(f); return -1; }
    }

    /* Apply file-level eval filter: if ensemble_eval doesn't match, skip entire file */
    if (g_eval_active && hdr.version >= 4) {
        int pass = 0;
        switch (g_eval_cmp) {
            case 0: pass = (hdr.ensemble_eval >  g_eval_thresh); break;
            case 1: pass = (hdr.ensemble_eval >= g_eval_thresh); break;
            case 2: pass = (hdr.ensemble_eval <  g_eval_thresh); break;
            case 3: pass = (hdr.ensemble_eval <= g_eval_thresh); break;
            case 4: pass = (fabsf(hdr.ensemble_eval - g_eval_thresh) < 0.001f); break;
        }
        if (!pass) {
            const char *op_str = "gt";
            switch (g_eval_cmp) {
                case 0: op_str = "gt"; break;
                case 1: op_str = "ge"; break;
                case 2: op_str = "lt"; break;
                case 3: op_str = "le"; break;
                case 4: op_str = "eq"; break;
            }
            printf("  [SKIP] %s  eval=%.1f%%  (filter: eval %s %.1f)\n",
                   path, hdr.ensemble_eval, op_str, g_eval_thresh);
            fclose(f); return -1;
        }
    }

    /* Validate global dims + config params */
    if (g_n_test == 0) {
        g_n_test    = (int)hdr.n_test;
        g_n_classes = (int)hdr.n_classes;
        g_hidden    = (int)hdr.hidden;
        g_epochs    = (int)hdr.epochs;
        g_split_vn  = (int)hdr.split_vn;
        g_split_hn  = (int)hdr.split_hn;
        g_target_err_x100 = (int)(hdr.target_err * 100.0f + 0.5f);
    } else {
        int ok = 1;
        if (g_n_test != (int)hdr.n_test || g_n_classes != (int)hdr.n_classes) {
            fprintf(stderr, "  [WARN] %s: dim mismatch (test=%u/%d, cls=%u/%d)\n",
                    path, hdr.n_test, g_n_test, hdr.n_classes, g_n_classes);
            ok = 0;
        }
        if (g_hidden    != (int)hdr.hidden      ||
            g_epochs    != (int)hdr.epochs      ||
            g_split_vn  != (int)hdr.split_vn    ||
            g_split_hn  != (int)hdr.split_hn    ||
            g_target_err_x100 != (int)(hdr.target_err * 100.0f + 0.5f)) {
            fprintf(stderr, "  [WARN] %s: config mismatch "
                    "(H=%u/%d EP=%u/%d VN=%u/%d HN=%u/%d TE=%.2f/%.2f)\n",
                    path, hdr.hidden, g_hidden, hdr.epochs, g_epochs,
                    hdr.split_vn, g_split_vn, hdr.split_hn, g_split_hn,
                    hdr.target_err, (float)g_target_err_x100 / 100.0f);
            ok = 0;
        }
        if (!ok) { fclose(f); return -1; }
    }

    int n_members = (int)hdr.n_members;
    int is_v2 = (hdr.version >= 2);
    size_t score_sz = (size_t)g_n_test * (size_t)g_n_classes;

    /* Read per-member metadata (v2+: color, enc_type, enc_width, pad) */
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
        for (int m = 0; m < n_members; m++) {
            uint8_t col, typ, wid, pad;
            if (fread(&col, 1, 1, f) != 1 || fread(&typ, 1, 1, f) != 1 ||
                fread(&wid, 1, 1, f) != 1 || fread(&pad, 1, 1, f) != 1) {
                fprintf(stderr, "  [WARN] %s: short metadata member %d\n", path, m);
                free(meta_color); free(meta_enc_type); free(meta_enc_wid);
                fclose(f); return -1;
            }
            meta_color[m]    = col;
            meta_enc_type[m] = typ;
            meta_enc_wid[m]  = wid;
        }
    }

    for (int m = 0; m < n_members; m++) {
        /* Read scores (always — advances file position) */
        int64_t *scores = (int64_t *)calloc(score_sz, sizeof(int64_t));
        if (!scores) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
        if (fread(scores, sizeof(int64_t), score_sz, f) != score_sz) {
            fprintf(stderr, "  [WARN] %s: short read member %d\n", path, m);
            free(scores); fclose(f); return -1;
        }

        /* Build temporary block to check filter */
        ScoreBlock tmp;
        tmp.seed       = (int)hdr.seed;
        tmp.member_idx = m;
        tmp.file_time  = (hdr.timestamp != 0) ? (time_t)hdr.timestamp : file_mtime;
        tmp.file_eval  = hdr.ensemble_eval;
        tmp.total_members = (int)hdr.n_members;
        tmp.scores     = scores;
        strncpy(tmp.source_file, base, sizeof(tmp.source_file) - 1);
        tmp.source_file[sizeof(tmp.source_file) - 1] = '\0';
        if (is_v2) {
            tmp.color     = (int)meta_color[m];
            tmp.enc_type  = (int)meta_enc_type[m];
            tmp.enc_width = (int)meta_enc_wid[m];
        } else {
            tmp.color     = -1;
            tmp.enc_type  = -1;
            tmp.enc_width = 0;
        }

        if (member_is_filtered(&tmp)) {
            free(scores);  /* skip this member */
            continue;
        }

        /* Add to blocks array */
        if (n_blocks >= n_blocks_cap) {
            int new_cap = n_blocks_cap ? n_blocks_cap * 2 : 4096;
            ScoreBlock *nb = (ScoreBlock *)realloc(blocks, (size_t)new_cap * sizeof(ScoreBlock));
            if (!nb) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
            blocks = nb;
            n_blocks_cap = new_cap;
        }
        memcpy(&blocks[n_blocks], &tmp, sizeof(ScoreBlock));
        n_blocks++;
    }

    free(meta_color); free(meta_enc_type); free(meta_enc_wid);

    /* Read ground truth labels (uint8[N] appended by trainer) */
    if (!g_labels) {
        g_labels = (uint8_t *)malloc((size_t)g_n_test);
        if (!g_labels) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
        if (fread(g_labels, 1, (size_t)g_n_test, f) != (size_t)g_n_test) {
            fprintf(stderr, "  [WARN] %s: no labels found (old format?)\n", path);
            free(g_labels); g_labels = NULL;
        }
    } else {
        /* Verify labels match */
        uint8_t *check = (uint8_t *)malloc((size_t)g_n_test);
        if (check) {
            if (fread(check, 1, (size_t)g_n_test, f) == (size_t)g_n_test) {
                for (int i = 0; i < g_n_test; i++) {
                    if (check[i] != g_labels[i]) {
                        fprintf(stderr, "  [WARN] %s: label mismatch at sample %d (%d vs %d)\n",
                                path, i, check[i], g_labels[i]);
                        break;
                    }
                }
            }
            free(check);
        }
    }
    fclose(f);
    return n_members;
}

/* -- Load all .ens files from directory ---------------------- */
static int load_directory(const char *dir) {
    DIR *d = opendir(dir);
    if (!d) { fprintf(stderr, "[ERROR] Cannot open directory %s\n", dir); return -1; }

    int n_files = 0;
    int n_skipped = 0;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        /* Only match .ens files */
        const char *ext = strrchr(de->d_name, '.');
        if (!ext || strcmp(ext, ".ens") != 0) continue;

        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dir, de->d_name);
        int loaded = load_archive(path);
        if (loaded > 0) { n_files++; }
        else if (loaded == -1) { n_skipped++; }  /* config mismatch or error */
    }
    closedir(d);
    if (n_files > 0 && n_skipped > 0)
        printf("  (%d file(s) skipped due to config mismatch)\n", n_skipped);
    return n_files;
}


/* -- Merge and evaluate -------------------------------------- */
static int merge_and_eval(const char *save_path, int max_en)
{
    if (n_blocks == 0) { printf("  No score blocks loaded.\n"); return 0; }
    if (max_en <= 0 || max_en > n_blocks) max_en = n_blocks;
    int has_labels = (g_labels != NULL);

    /* Sort by selected mode */
    int (*cmp)(const void*, const void*) = 
        g_sort_mode ? cmp_by_ctime : cmp_by_seed;
    qsort(blocks, (size_t)n_blocks, sizeof(ScoreBlock), cmp);

    size_t score_sz = (size_t)g_n_test * (size_t)g_n_classes;
    size_t row_sz   = (size_t)g_n_classes;

    int64_t *sum_scores = (int64_t *)calloc(score_sz, sizeof(int64_t));
    if (!sum_scores) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }

    printf("\n");
    printf("== MERGE ENSEMBLE ============================================\n");
    printf("  %d score blocks (%d test samples, %d classes)\n",
           n_blocks, g_n_test, g_n_classes);
    printf("  Config: H=%d  EP=%d  VN=%d  HN=%d  TE=%d\n",
           g_hidden, g_epochs, g_split_vn, g_split_hn, g_target_err_x100);
    printf("\n");
    printf("  %-4s  %-7s  %-11s  %-7s  %s\n", "EN",   "acc[%]",  "correct",     "gain[%]", "member");
    printf("  %-4s  %-7s  %-11s  %-7s  %s\n", "----", "-------", "-----------", "-------", "-------------------");

    FILE *save_f = NULL;
    if (save_path && save_path[0]) {
        save_f = fopen(save_path, "w");
        if (save_f) fprintf(save_f, "# EN  acc  correct  total  gain\n");
    }

    float prev_acc = 0.0f;
    const char *prev_file = "";
    int file_idx = 0;
    int files_merged = 0;
    for (int en = 1; en <= n_blocks; en++) {
        int m = en - 1;

        /* Print file separator when source file changes */
        if (strcmp(blocks[m].source_file, prev_file) != 0) {
            prev_file = blocks[m].source_file;
            file_idx++;
            files_merged++;
            /* --max/--num limits the number of archive files, not score blocks */
            if (max_en > 0 && files_merged > max_en) break;
            printf("  ─── #%d [%.1f%%] %s ───\n", file_idx,
                   blocks[m].file_eval, prev_file);
        }

        const int64_t *sc = blocks[m].scores;
        for (size_t i = 0; i < score_sz; i++)
            sum_scores[i] += sc[i];

        int correct = 0;
        for (int s = 0; s < g_n_test && has_labels; s++) {
            const int64_t *row = sum_scores + (size_t)s * row_sz;
            int pred = 0;
            for (int k = 1; k < g_n_classes; k++)
                if (row[k] > row[pred]) pred = k;
            if (pred == (int)g_labels[s]) correct++;
        }

        float acc = has_labels ? (float)correct * 100.0f / (float)g_n_test : 0.0f;
        char label[24] = {'x'};
        member_label(&blocks[m], label, sizeof(label));

        float gain = acc - prev_acc;
        if (has_labels) {
            printf("  %-4d  %-7.2f  %5d/%-5d  %-+7.2f  %-s\n",
                   en, acc, correct, g_n_test, gain, label);
        } else {
            printf("  %-4d  (no labels)\n", en);
        }
        if (save_f && has_labels)
            fprintf(save_f, "%d  %.4f  %d  %d  %+.4f\n",
                    en, acc / 100.0f, correct, g_n_test, gain / 100.0f);
        prev_acc = acc;
    }

    if (save_f) { fclose(save_f); printf("\n  Saved:  %s\n", save_path); }
    free(sum_scores);
    return g_n_test;
}

/* =============================================================
 * MAIN
 * ============================================================= */
static void show_help(const char *prog) {
    printf("Usage: %s DIR [options]\n", prog);
    printf("\n");
    printf("Merge score archives (.ens) to an ensemble accuracy curve (EN=1..N).\n");
    printf("\n");
    printf("Options:\n");
    printf("  DIR          Directory containing .ens score archives\n");
    printf("  --num N      Only combine the first N archive files (default: all)\n");
    printf("  --max N      Alias for --num\n");
    printf("  --save FILE  Save cumulative accuracy to FILE (default: DIR/merge.dat)\n");
    printf("  --sort MODE  Sort members by 'seed' or 'ctime' (default: ctime)\n");
    printf("  --filter L   Filter members by label or eval threshold\n");
    printf("               Label exclude : --filter GB=sig8,BL=down8 (case-insensitive, no spaces)\n");
    printf("               Eval threshold: --filter eval gt 58.1    (gt, lt, ge, le, eq, no quotes needed)\n");
    printf("  -h, --help   Show this help text\n");
    printf("\n");
    printf("Output:\n");
    printf("  Table: EN  acc[%%]  correct/total  gain\n");
    printf("  File:  merge.dat  (for plotting)\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s scores/\n", prog);
    printf("  %s scores/ --num 5\n", prog);
    printf("  %s scores/ --save /tmp/curve.dat --num 20\n", prog);
    printf("\n");
    printf("The .ens archives are created by the Otto Score trainer via --save-scores DIR.\n");
    printf("Each archive contains all ensemble members from one training run.\n");
}

int main(int argc, char **argv) {
    if (argc < 2) {
        show_help(argv[0]);
        return 1;
    }

    /* Check for -h/--help anywhere */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            show_help(argv[0]);
            return 0;
        }
    }

    /* DIR = first non-option, non-value token */
    const char *dir = NULL;
    const char *save_path = NULL;
    int max_en = 0;

    /* Parse all options in any order; DIR = first bare token */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
            save_path = argv[++i];
        } else if (strcmp(argv[i], "--num") == 0 && i + 1 < argc) {
            max_en = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max") == 0 && i + 1 < argc) {
            max_en = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--sort") == 0 && i + 1 < argc) {
            const char *mode = argv[++i];
            if (strcmp(mode, "seed") == 0) {
                g_sort_mode = 0;
            } else if (strcmp(mode, "ctime") == 0) {
                g_sort_mode = 1;
            } else {
                fprintf(stderr, "[ERROR] --sort: expected 'seed' or 'ctime', got '%s'\n", mode);
                return 1;
            }
        } else if (strcmp(argv[i], "--filter") == 0) {
            /* Consume all following arguments until next flag or end */
            char buf[1024] = "";
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                i++;
                if (buf[0]) strncat(buf, " ", sizeof(buf) - strlen(buf) - 1);
                strncat(buf, argv[i], sizeof(buf) - strlen(buf) - 1);
            }
            if (buf[0] == '\0') {
                fprintf(stderr, "[ERROR] --filter requires an expression\n");
                return 1;
            }
            g_eval_active = 0;
            g_filter_count = 0;
            for (char *tok = strtok(buf, ","); tok && (g_filter_count < 64 || g_eval_active); tok = strtok(NULL, ",")) {
                while (*tok == ' ' || *tok == '\t') tok++;
                if (*tok == '\0') continue;

                if (strncasecmp(tok, "eval", 4) == 0) {
                    const char *p = tok + 4;
                    while (*p == ' ' || *p == '\t') p++;
                    /* parse text operator: gt, lt, ge, le, eq */
                    int op_len = 0;
                    if      (strncasecmp(p, "gt", 2) == 0) { g_eval_cmp = 0; g_eval_op = '>'; op_len = 2; }
                    else if (strncasecmp(p, "ge", 2) == 0) { g_eval_cmp = 1; g_eval_op = 'g'; op_len = 2; }
                    else if (strncasecmp(p, "lt", 2) == 0) { g_eval_cmp = 2; g_eval_op = '<'; op_len = 2; }
                    else if (strncasecmp(p, "le", 2) == 0) { g_eval_cmp = 3; g_eval_op = 'l'; op_len = 2; }
                    else if (strncasecmp(p, "eq", 2) == 0) { g_eval_cmp = 4; g_eval_op = '='; op_len = 2; }
                    else {
                        fprintf(stderr, "[ERROR] Unknown eval operator in '%s' — use gt, lt, ge, le, eq\n", tok);
                        return 1;
                    }
                    p += op_len;
                    while (*p == ' ' || *p == '\t') p++;
                    g_eval_thresh = (float)atof(p);
                    g_eval_active = 1;
                    continue;
                }

                strncpy(g_filter_pat[g_filter_count], tok, sizeof(g_filter_pat[0]) - 1);
                g_filter_pat[g_filter_count][sizeof(g_filter_pat[0]) - 1] = '\0';
                g_filter_count++;
            }
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "[ERROR] Unknown option: %s\n", argv[i]);
            return 1;
        } else if (!dir) {
            dir = argv[i];
        } else {
            fprintf(stderr, "[WARN] Extra argument: %s (use --help)\n", argv[i]);
        }
    }

    if (!dir) {
        fprintf(stderr, "[ERROR] Missing DIR\n");
        return 1;
    }
    /* Default save path */
    if (!save_path) {
        char buf[1024];
        snprintf(buf, sizeof(buf), "%s/merge.dat", dir);
        save_path = buf;
    }

    /* Load score archives */
    int nf = load_directory(dir);
    if (nf == 0) {
        fprintf(stderr, "[ERROR] No .ens files found in %s\n", dir);
        return 1;
    }
    printf("Loaded %d archive files (%d total score blocks)\n", nf, n_blocks);
    printf("  Sort: %s\n", g_sort_mode ? "ctime" : "seed");
    if (g_filter_count > 0) {
        printf("  Filter:");
        for (int i = 0; i < g_filter_count; i++)
            printf(" %s", g_filter_pat[i]);
    }
    if (g_eval_active) {
        const char *op_str = "gt";
        switch (g_eval_cmp) {
            case 0: op_str = "gt"; break;
            case 1: op_str = "ge"; break;
            case 2: op_str = "lt"; break;
            case 3: op_str = "le"; break;
            case 4: op_str = "eq"; break;
        }
        printf("  eval %s %.1f", op_str, g_eval_thresh);
    }
    printf("\n");

    merge_and_eval(save_path, max_en);
    for (int i = 0; i < n_blocks; i++) free(blocks[i].scores);
    free(blocks); free(g_labels);
    return 0;
}
