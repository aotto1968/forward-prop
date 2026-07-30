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
 * Archive format (produced by --export-merge-scores):
 *   Header: magic(4) ver(4) n_test(4) n_classes(4) n_members(4)
 *           hidden(4) epochs(4) split_vn(4) split_hn(4)
 *           target_err(4) seed(4)
 *   Data:   n_members . int64[n_test . n_classes]
 *           uint8[n_test]  (ground truth labels)
 *
 * Options:
 *   DIR         Directory containing .ens score archives
 *   --max N     Stop optimize/beam at N members (default: unlimited)
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
#include <time.h>
#include <sys/time.h>
#include <omp.h>
/* Define ki_Args aa in this file (make KI_ARGS_EXTERN empty) */
#define KI_ARGS_EXTERN
#include "../lib/ki-encoding.h"
#include "ki-common.h"

/* ── Simple name-to-ID lookup for v7 archive reading (exact match, no aliases) ── */
static int color_id_by_name(const char *name) {
    for (int i = 0; i < (int)(sizeof(ki_color_table)/sizeof(ki_color_table[0])); i++)
        if (strcmp(ki_color_table[i].name, name) == 0) return ki_color_table[i].id;
    return -1;
}
static int enc_id_by_name(const char *name) {
    for (int i = 0; i < (int)(sizeof(ki_enc_table)/sizeof(ki_enc_table[0])); i++)
        if (strcmp(ki_enc_table[i].name, name) == 0) return ki_enc_table[i].id;
    return -1;
}
/* xform_id_by_name — uses ki_xform_parse_or_pipe() to handle both
 * simple xforms (e.g. "colswap-1-4") and pipe xforms (e.g. "rot22@colswap-1-4").
 * Does NOT register the pipe (it's already registered by the trainer that created the archive). */
static int xform_id_by_name(const char *name) {
    int id = ki_xform_parse_or_pipe(name);
    if (id < 0) {
        /* Fallback: try the table directly (for names with trailing digits etc.) */
        for (int i = 0; i < (int)(sizeof(ki_xform_table)/sizeof(ki_xform_table[0])); i++)
            if (strcmp(ki_xform_table[i].name, name) == 0) return ki_xform_table[i].id;
    }
    return id;
}

/* xform_name_safe — returns pipe name for pipe IDs, regular name otherwise.
 * Used instead of ki_xform_name() which returns "?" for virtual pipe IDs. */
static inline const char *xform_name_safe(int xf_id) {
    if (xf_id < 0) return "?";
    return ki_xform_str(xf_id);
}

/* ── CRC64 for score integrity checking ── */
static uint64_t score_crc64(const SCORE_TYPE *scores, size_t n) {
    if (n == 0) return 0;
    uint64_t h = 0;
    for (size_t i = 0; i < n; i++)
        h = h * 0x9E3779B97F4A7C15ULL ^ (uint64_t)scores[i];
    return h;
}

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
    uint32_t w0_marker;    /* v10+: W0[0] marker for W0 verification */
} ScoreHeader;

/* -- Per-member score block ----------------------------------- */
typedef struct {
    int      seed;
    int      member_idx;
    SCORE_TYPE *scores;  /* [n_test x n_classes] (float or int64 depending on MODE_INT32) */
    /* Encoding info (v2 archives) */
    int      color;       /* channel index (0=R, 1=G, 2=B, etc.) */
    int      enc_type;    /* encoding type (KI_ENC_RAW=0, LIN7=1, etc.) */
    int      enc_width;   /* encoding width (8, 16, 32) */
    time_t   file_time;   /* file mtime (ctime for sorting) */
    int      total_members; /* total members in this file (for #/# separator) */
    float    file_eval;     /* ensemble eval accuracy % (v4+, 0 for older) */
    int      xform_id;      /* which xform this member belongs to (inferred) */
    int      xform_real;    /* actual KI_XFORM_ID (v5+), -1 for older archives */
    uint32_t w0_marker;     /* v10+: W0[0] marker for W0 verification */
    SCORE_TYPE  score_min;  /* minimum score value */
    SCORE_TYPE  score_max;  /* maximum score value */
    int      err;           /* eval errors (g_n_test - correct_count) */
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
static int    g_filter_count = 0;   /* number of active filter patterns */
static char   g_filter_pat[64][32]; /* member labels to exclude (case-insensitive) */
static int    g_eval_active = 0;    /* 0=disabled, 1=eval threshold active */
static int    g_eval_cmp = 0;       /* 0:>, 1:>=, 2:<, 3:<=, 4:= */
static char   g_eval_op = '>';     /* operator char for display */
static float  g_eval_thresh = 0.0f; /* threshold value */
static char   g_member_out[1024] = "";  /* --member-out PATH: write optimal subset as member file */
static char   g_xform_spec[1024] = "";  /* --xform SPEC: expand xform list (e.g. "sweep@spiral") */
static char   g_member_seed_spec[256] = ""; /* --member-seed SPEC: pre-seed beam with this member spec */
static int    g_tries = 1;              /* --tries N: repeat beam search N times with shuffle, keep best */
static int    g_max_mode = 0;           /* --max: find best eval even if more members needed (0.01% threshold off) */
static int    g_max_members = 0;        /* --max N: stop optimize/beam at N members (0 = unlimited) */
static int    g_diversity = 0;          /* --diversity N: min pairwise diff between beam paths (0=off) */
static int    g_expansion_sort = 0;     /* --expansion-sort: 0=abs (default), 1=marginal */
static float  g_min_gain = 0.01f;       /* --min-gain F: min. improvement %% to keep adding (default: 0.01) */
static int    g_debug = 0;              /* --debug: verbose debug output */
static int    g_seed_sort = 0;         /* --seed-sort: print single-member eval table */
static float    g_best_eval = -1.0f;      /* global best across tries (beam) */
static int      g_best_n = 0;            /* member count for global best */
static uint8_t *g_best_used = NULL;     /* used mask for global best */
static ScoreBlock *g_best_blocks = NULL; /* blocks at their best order */
static int      g_best_order[2048];      /* member indices in beam selection order */
static int      g_best_order_n = 0;      /* number of members in g_best_order */

/* -- Comparators --------------------------------------------- */
static int cmp_by_ctime(const void *a, const void *b) {
    const ScoreBlock *ba = (const ScoreBlock *)a;
    const ScoreBlock *bb = (const ScoreBlock *)b;
    if (ba->file_time != bb->file_time) 
        return (ba->file_time < bb->file_time) ? -1 : 1;
    if (ba->seed != bb->seed) return ba->seed - bb->seed;
    if (ba->member_idx != bb->member_idx) return ba->member_idx - bb->member_idx;
    /* Tiebreaker: source_file ensures deterministic order even when
     * file_time/seed/member_idx are equal (single-member .ens files). */
    return strcmp(ba->source_file, bb->source_file);
}

/* -- Xform group label (from archive order, NOT from ki_xform_name)
 * We keep a simple numeric fallback since the archive doesn't store xform names. */
static const char *xform_group_label(int xf) {
    static char buf[16];
    snprintf(buf, sizeof(buf), "xf%d", xf);
    return buf;
}

/* -- Build member label: "chan:encN" ------------------------- */
static void member_label(const ScoreBlock *b, char *buf, size_t sz) {
    const char *cn = (b->color >= 0) ? ki_color_name(b->color) : "?";
    const char *en = (b->enc_type >= 0) ? ki_enc_name_short(b->enc_type) : "?";
    int ew = (b->enc_width > 0) ? b->enc_width : 8;
    snprintf(buf, sz, "%s:%s%d", cn, en, ew);
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

/* -- Find member by spec string "xf:chan:encW" ----------------
 * Returns block index or -1 if not found. Used by --member-seed.
 * First tries exact match (xf:chan:encW), then lenient match
 * allowing partial enc name without width (e.g. "exp" → "exp8"). */
static int find_member_by_spec(const char *spec) {
    char _spec_buf[256];
    for (int i = 0; i < n_blocks; i++) {
        const char *_xn = (blocks[i].xform_real >= 0)
                          ? xform_name_safe(blocks[i].xform_real)
                          : xform_group_label(blocks[i].xform_id);
        char _ml[64];
        member_label(&blocks[i], _ml, sizeof(_ml));
        snprintf(_spec_buf, sizeof(_spec_buf), "%s:%s", _xn, _ml);
        if (strcmp(_spec_buf, spec) == 0) return i;
        /* Lenient: try stripping width digits from spec (e.g. "exp" matches "exp8") */
        {   size_t _sl = strlen(spec);
            size_t _bl = strlen(_spec_buf);
            if (_sl > 0 && _bl > _sl && _spec_buf[_sl] >= '0' && _spec_buf[_sl] <= '9' &&
                strncmp(spec, _spec_buf, _sl) == 0)
                return i;
        }
    }
    return -1;
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
    if (hdr.version < 1 || hdr.version > 10) {
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

    /* v10+: read W0[0] marker (for W0 verification) */
    hdr.w0_marker = 0;
    if (hdr.version >= 10) {
        if (fread(&hdr.w0_marker, 4, 1, f) != 1) { fclose(f); return -1; }
    }

    /* v5: read xform list (n_xforms + xform_ids) — removed in v6, each member carries its own xform_id */
    uint32_t _nxf = 1;
    uint32_t _xform_ids[64] = {0};
    if (hdr.version == 5) {
        if (fread(&_nxf, 4, 1, f) != 1 || _nxf > 64 || _nxf == 0) {
            fprintf(stderr, "  [WARN] %s: invalid xform list\n", path);
            _nxf = 1;
        }
        for (uint32_t xi = 0; xi < _nxf; xi++) {
            if (fread(&_xform_ids[xi], 4, 1, f) != 1) {
                fprintf(stderr, "  [WARN] %s: short xform list\n", path);
                _nxf = 0; break;
            }
        }
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

    /* Read per-member metadata (v2+: color, enc_type, enc_width, pad/xf_idx)
     * v7+: string-based (length-prefixed), v2-v6: binary 4 bytes per member */
    uint8_t *meta_color    = NULL;
    uint8_t *meta_enc_type = NULL;
    uint8_t *meta_enc_wid  = NULL;
    uint8_t *meta_xf_idx   = NULL;  /* v5+: xform index; v6: direct xform_id; v7: unused */
    /* For v7, parse strings into temp arrays, then convert to ints via parsers */
    int *meta_color_v7 = NULL, *meta_enc_v7 = NULL, *meta_wid_v7 = NULL, *meta_xf_v7 = NULL;
    if (hdr.version >= 7) {
        /* v7: length-prefixed strings — parse via ki_color_parse/ki_enc_parse/ki_xform_parse */
        meta_color_v7 = (int *)calloc((size_t)n_members, sizeof(int));
        meta_enc_v7   = (int *)calloc((size_t)n_members, sizeof(int));
        meta_wid_v7   = (int *)calloc((size_t)n_members, sizeof(int));
        meta_xf_v7    = (int *)calloc((size_t)n_members, sizeof(int));
        if (!meta_color_v7 || !meta_enc_v7 || !meta_wid_v7 || !meta_xf_v7) {
            fprintf(stderr, "[FATAL] OOM\n"); exit(1);
        }
        for (int m = 0; m < n_members; m++) {
            char str_buf[128];
            /* Read 4 length-prefixed strings: color, encoding, width, xform */
            for (int field = 0; field < 4; field++) {
                uint8_t slen;
                if (fread(&slen, 1, 1, f) != 1 || slen >= sizeof(str_buf)) {
                    fprintf(stderr, "  [ERROR] %s: bad metadata (member %d, field %d)\n", path, m, field);
                    free(meta_color_v7); free(meta_enc_v7); free(meta_wid_v7); free(meta_xf_v7);
                    fclose(f); return -1;
                }
                if (fread(str_buf, 1, slen, f) != slen) {
                    fprintf(stderr, "  [ERROR] %s: short metadata (member %d, field %d)\n", path, m, field);
                    free(meta_color_v7); free(meta_enc_v7); free(meta_wid_v7); free(meta_xf_v7);
                    fclose(f); return -1;
                }
                str_buf[slen] = '\0';
                switch (field) {
                    case 0: /* color */
                        meta_color_v7[m] = color_id_by_name(str_buf);
                        if (meta_color_v7[m] < 0) {
                            fprintf(stderr, "  [ERROR] %s: unknown channel '%s' (member %d)\n", path, str_buf, m);
                            free(meta_color_v7); free(meta_enc_v7); free(meta_wid_v7); free(meta_xf_v7);
                            fclose(f); return -1;
                        }
                        break;
                    case 1: { /* encoding */
                        meta_enc_v7[m] = enc_id_by_name(str_buf);
                        if (meta_enc_v7[m] < 0) {
                            fprintf(stderr, "  [ERROR] %s: unknown encoding '%s' (member %d)\n", path, str_buf, m);
                            free(meta_color_v7); free(meta_enc_v7); free(meta_wid_v7); free(meta_xf_v7);
                            fclose(f); return -1;
                        }
                        break;
                    }
                    case 2: /* enc_width (string), store as int */
                        meta_wid_v7[m] = atoi(str_buf);
                        break;
                    case 3: /* xform */
                        meta_xf_v7[m] = xform_id_by_name(str_buf);
                        if (meta_xf_v7[m] < 0) {
                            fprintf(stderr, "  [ERROR] %s: unknown xform '%s' (member %d)\n", path, str_buf, m);
                            free(meta_color_v7); free(meta_enc_v7); free(meta_wid_v7); free(meta_xf_v7);
                            fclose(f); return -1;
                        }
                        break;
                }
            }
        }
    } else if (is_v2) {
        meta_color    = (uint8_t *)malloc((size_t)n_members);
        meta_enc_type = (uint8_t *)malloc((size_t)n_members);
        meta_enc_wid  = (uint8_t *)malloc((size_t)n_members);
        if (hdr.version >= 5)  /* v5: xform index; v6: direct xform_id */
            meta_xf_idx = (uint8_t *)calloc((size_t)n_members, 1);
        if (!meta_color || !meta_enc_type || !meta_enc_wid) {
            fprintf(stderr, "[FATAL] OOM\n"); exit(1);
        }
        for (int m = 0; m < n_members; m++) {
            uint8_t col, typ, wid, pad;
            if (fread(&col, 1, 1, f) != 1 || fread(&typ, 1, 1, f) != 1 ||
                fread(&wid, 1, 1, f) != 1 || fread(&pad, 1, 1, f) != 1) {
                fprintf(stderr, "  [WARN] %s: short metadata member %d\n", path, m);
    free(meta_color); free(meta_enc_type); free(meta_enc_wid); free(meta_xf_idx);
    free(meta_color_v7); free(meta_enc_v7); free(meta_wid_v7); free(meta_xf_v7);
                fclose(f); return -1;
            }
            meta_color[m]    = col;
            meta_enc_type[m] = typ;
            meta_enc_wid[m]  = wid;
            if (meta_xf_idx) meta_xf_idx[m] = pad;  /* v5: pad = xform index */
        }
    }

    /* Score detection: not needed — SCORE_TYPE determines how we read.
     * File version (v8=4byte int32, v9=4byte float, v1-7=8byte int64)
     * determines the binary format. */

    for (int m = 0; m < n_members; m++) {
        /* Declare tmp early so v9 raw float tracking can access it */
        ScoreBlock tmp;
        memset(&tmp, 0, sizeof(tmp));
        SCORE_TYPE *scores = (SCORE_TYPE *)calloc(score_sz, sizeof(SCORE_TYPE));
        if (!scores) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
        if (COUNTER_TYPE_IS_FLOAT) {
            /* Float mode: store scores as float (no scaling needed for v9).
             * v8 int32 and v1-v7 int64 are converted to float (lossy for >2^24). */
            if (hdr.version >= 9) {
                /* v9: read float scores directly */
                size_t nr = fread(scores, sizeof(float), score_sz, f);
                if (nr != score_sz) {
                    fprintf(stderr, "  [WARN] %s: short read member %d (4-byte float scores)\n", path, m);
                    free(scores);
                    free(meta_color_v7); free(meta_enc_v7); free(meta_wid_v7); free(meta_xf_v7);
                    fclose(f); return -1;
                }
            } else if (hdr.version >= 8) {
                /* v8: 4-byte int32 → float */
                int32_t *buf32 = (int32_t *)malloc((size_t)score_sz * sizeof(int32_t));
                if (!buf32) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
                size_t nr = fread(buf32, sizeof(int32_t), score_sz, f);
                if (nr != score_sz) {
                    fprintf(stderr, "  [WARN] %s: short read member %d (4-byte scores)\n", path, m);
                    free(buf32); free(scores);
                    free(meta_color_v7); free(meta_enc_v7); free(meta_wid_v7); free(meta_xf_v7);
                    fclose(f); return -1;
                }
                for (size_t i = 0; i < score_sz; i++) scores[i] = (SCORE_TYPE)buf32[i];
                free(buf32);
            } else {
                /* v1-v7: int64 → float */
                int64_t *buf64 = (int64_t *)malloc((size_t)score_sz * sizeof(int64_t));
                if (!buf64) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
                size_t nr = fread(buf64, sizeof(int64_t), score_sz, f);
                if (nr != score_sz) {
                    fprintf(stderr, "  [WARN] %s: short read member %d\n", path, m);
                    free(buf64); free(scores);
                    free(meta_color_v7); free(meta_enc_v7); free(meta_wid_v7); free(meta_xf_v7);
                    fclose(f); return -1;
                }
                for (size_t i = 0; i < score_sz; i++) scores[i] = (SCORE_TYPE)buf64[i];
                free(buf64);
            }
        } else {
            /* Int32/Int64 mode (MODE_INT32): store as int64 */
            if (hdr.version >= 9) {
                /* v9: float → int64 via scaling */
                float *buf32 = (float *)malloc((size_t)score_sz * sizeof(float));
                if (!buf32) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
                size_t nr = fread(buf32, sizeof(float), score_sz, f);
                if (nr != score_sz) {
                    fprintf(stderr, "  [WARN] %s: short read member %d (4-byte float scores)\n", path, m);
                    free(buf32); free(scores);
                    free(meta_color_v7); free(meta_enc_v7); free(meta_wid_v7); free(meta_xf_v7);
                    fclose(f); return -1;
                }
                for (size_t i = 0; i < score_sz; i++)
                    scores[i] = (SCORE_TYPE)(buf32[i] * 1048576.0f);
                free(buf32);
            } else if (hdr.version >= 8) {
                /* v8: 4-byte int32 */
                int32_t *buf32 = (int32_t *)malloc((size_t)score_sz * sizeof(int32_t));
                if (!buf32) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
                size_t nr = fread(buf32, sizeof(int32_t), score_sz, f);
                if (nr != score_sz) {
                    fprintf(stderr, "  [WARN] %s: short read member %d (4-byte scores)\n", path, m);
                    free(buf32); free(scores);
                    free(meta_color_v7); free(meta_enc_v7); free(meta_wid_v7); free(meta_xf_v7);
                    fclose(f); return -1;
                }
                for (size_t i = 0; i < score_sz; i++) scores[i] = (SCORE_TYPE)buf32[i];
                free(buf32);
            } else {
                /* v1-v7: int64 directly */
                if (fread(scores, sizeof(SCORE_TYPE), score_sz, f) != score_sz) {
                    fprintf(stderr, "  [WARN] %s: short read member %d\n", path, m);
                    free(scores);
                    free(meta_color_v7); free(meta_enc_v7); free(meta_wid_v7); free(meta_xf_v7);
                    fclose(f); return -1;
                }
            }
        }

        /* Build temporary block to check filter */
        tmp.seed       = (int)hdr.seed;
        tmp.member_idx = m;
        tmp.file_time  = (hdr.timestamp != 0) ? (time_t)hdr.timestamp : file_mtime;
        tmp.file_eval  = hdr.ensemble_eval;
        tmp.w0_marker  = hdr.w0_marker;
        tmp.total_members = (int)hdr.n_members;
        /* xform_id: group index (inferred for v1-4, from metadata for v5+) */
        int memb_per_xf = _nxf > 0 ? (int)hdr.n_members / (int)_nxf : 1;
        if (memb_per_xf < 1) memb_per_xf = 1;
        tmp.xform_id = m / memb_per_xf;
        if (tmp.xform_id < 0) tmp.xform_id = 0;
        /* xform_real/color/enc/width from metadata (v7: string-parsed, v2-6: binary) */
        tmp.xform_real = -1;
        if (hdr.version >= 7) {
            tmp.color     = meta_color_v7 ? meta_color_v7[m] : 0;
            tmp.enc_type  = meta_enc_v7   ? meta_enc_v7[m]   : 0;
            tmp.enc_width = meta_wid_v7   ? meta_wid_v7[m]   : 8;
            tmp.xform_real = meta_xf_v7   ? meta_xf_v7[m]    : -1;
        } else if (meta_xf_idx && m < n_members) {
            if (hdr.version >= 6) {
                tmp.xform_real = (int)meta_xf_idx[m];  /* v6: 4th byte = xform_id directly */
            } else if (hdr.version == 5) {
                int xi = (int)meta_xf_idx[m];
                if (xi >= 0 && xi < (int)_nxf)
                    tmp.xform_real = (int)_xform_ids[xi];
            }
        }
        tmp.scores    = scores;
        /* Compute min/max score and standalone err */
        tmp.err = 0;
#if COUNTER_TYPE_IS_FLOAT
        tmp.score_min = INFINITY; tmp.score_max = -INFINITY;
#else
        tmp.score_min = INT64_MAX; tmp.score_max = INT64_MIN;
#endif
        if (scores && g_n_test > 0) {
            size_t _ss = (size_t)g_n_test * (size_t)g_n_classes;
            for (size_t _si = 0; _si < _ss; _si++) {
                if (scores[_si] < tmp.score_min) tmp.score_min = scores[_si];
                if (scores[_si] > tmp.score_max) tmp.score_max = scores[_si];
            }
        }
        if (hdr.ensemble_eval > 0)
            tmp.err = g_n_test - (int)(hdr.ensemble_eval * (float)g_n_test / 100.0f + 0.5f);
        strncpy(tmp.source_file, base, sizeof(tmp.source_file) - 1);
        tmp.source_file[sizeof(tmp.source_file) - 1] = '\0';
        if (is_v2 && hdr.version < 7) {
            tmp.color     = (int)meta_color[m];
            tmp.enc_type  = (int)meta_enc_type[m];
            tmp.enc_width = (int)meta_enc_wid[m];
        } else if (hdr.version < 7) {
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

    free(meta_color); free(meta_enc_type); free(meta_enc_wid); free(meta_xf_idx);
    free(meta_color_v7); free(meta_enc_v7); free(meta_wid_v7); free(meta_xf_v7);

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

    /* Read .meta (directory identity) if present */
    {
        char meta_path[1024];
        snprintf(meta_path, sizeof(meta_path), "%s/.meta", dir);
        FILE *mf = fopen(meta_path, "r");
        if (mf) {
            int m_h = 0, m_ep = 0, m_vn = 0, m_hn = 0;
            char line[128];
            while (fgets(line, sizeof(line), mf)) {
                if (sscanf(line, "H=%d", &m_h) == 1) continue;
                if (sscanf(line, "EPOCHS=%d", &m_ep) == 1) continue;
                if (sscanf(line, "VN=%d", &m_vn) == 1) continue;
                if (sscanf(line, "HN=%d", &m_hn) == 1) continue;
            }
            fclose(mf);
            printf("  Config from .meta: H=%d  EP=%d  VN=%d  HN=%d\n",
                   m_h, m_ep, m_vn, m_hn);
        } else {
            printf("  (no .meta — legacy directory, config from first archive)\n");
        }
    }

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

    /* Sort by ctime for deterministic order */
    qsort(blocks, (size_t)n_blocks, sizeof(ScoreBlock), cmp_by_ctime);

    size_t score_sz = (size_t)g_n_test * (size_t)g_n_classes;
    size_t row_sz   = (size_t)g_n_classes;

    SCORE_TYPE *sum_scores = (SCORE_TYPE *)calloc(score_sz, sizeof(SCORE_TYPE));
    if (!sum_scores) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }

    printf("\n");
    printf("== MERGE ENSEMBLE ============================================\n");
    printf("  %d score blocks (%d test samples, %d classes)\n",
           n_blocks, g_n_test, g_n_classes);
    printf("  Config: H=%d  EP=%d  VN=%d  HN=%d  TE=%d\n",
           g_hidden, g_epochs, g_split_vn, g_split_hn, g_target_err_x100);
    printf("\n");
    printf("  %-4s  %-7s  %-11s  %-8s  %s\n", "EN",   "acc[%]",  "correct",     "gain[%]", "member");
    printf("  %-4s  %-7s  %-11s  %-8s  %s\n", "----", "-------", "-----------", "--------", "-------------------");

    FILE *save_f = NULL;
    if (save_path && save_path[0]) {
        save_f = fopen(save_path, "w");
        if (save_f) fprintf(save_f, "# EN  acc  correct  total  gain\n");
    }

    float prev_acc = 0.0f;
    float best_acc = 0.0f;
    int best_en = 0;
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
            /* file separators */
            if (max_en > 0 && files_merged > max_en) break;
            printf("  ─── #%d [%.1f%%] %s ───\n", file_idx,
                   blocks[m].file_eval, prev_file);
        }

        const SCORE_TYPE *sc = blocks[m].scores;
        for (size_t i = 0; i < score_sz; i++)
            sum_scores[i] += sc[i];

        int correct = 0;
        if (has_labels) {
          #pragma omp parallel for reduction(+:correct)
          for (int s = 0; s < g_n_test; s++) {
              const SCORE_TYPE *row = sum_scores + (size_t)s * row_sz;
              int pred = 0;
              for (int k = 1; k < g_n_classes; k++)
                  if (row[k] > row[pred]) pred = k;
              if (pred == (int)g_labels[s]) correct++;
          }
        }

        float acc = has_labels ? (float)correct * 100.0f / (float)g_n_test : 0.0f;
        char label[24] = {'x'};
        member_label(&blocks[m], label, sizeof(label));

        float gain = acc - prev_acc;
        if (acc > best_acc) {
            best_acc = acc;
            best_en = en;
        }
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

    printf("\n══╡ BEST PREFIX ╞═══════════════════════════════════════════\n");
    printf("  Best:  EN=%d  acc=%.2f%%  (%d/%d)\n",
           best_en, best_acc,
           (int)(best_acc * (float)g_n_test / 100.0f + 0.5f), g_n_test);
    printf("  Final: %d members  acc=%.2f%%  (%d/%d)\n",
           n_blocks, prev_acc,
           (int)(prev_acc * (float)g_n_test / 100.0f + 0.5f), g_n_test);

    int _bc = (int)(best_acc * (float)g_n_test / 100.0f + 0.5f);
    printf("\n══╡ REPORT ╞════════════════════════════════════════════════════════\n");
    printf("REPORT train=0.0%% (0) eval=%.2f%% (%d) err=%d lr=0.0000 time=0ms threads=1 members=%d\n",
           best_acc, _bc, g_n_test - _bc, best_en);
    if (save_f) { fclose(save_f); printf("  Saved:  %s\n", save_path); }
    free(sum_scores);
    return g_n_test;
}

/* =============================================================
 * MAIN
 * ============================================================= */
/* ═══════════════════════════════════════════════════════════════════════
 * OPTIMIZE — Greedy optimal subset search
 * ═══════════════════════════════════════════════════════════════════════
 * Starts with empty ensemble. In each step, tries every unused member,
 * picks the one that improves eval the most. Stops when no member helps.
 */
static int merge_and_optimize(const char *save_path)
{
    if (n_blocks == 0) { printf("  No score blocks loaded.\n"); return 0; }
    int has_labels = (g_labels != NULL);
    int n = n_blocks;

    size_t score_sz = (size_t)g_n_test * (size_t)g_n_classes;
    size_t row_sz   = (size_t)g_n_classes;

    /* Cumulative sum of selected members' scores */
    SCORE_TYPE *sum_scores = (SCORE_TYPE *)calloc(score_sz, sizeof(SCORE_TYPE));
    if (!sum_scores) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }

    /* Track used members */
    int *used = (int *)calloc((size_t)n, sizeof(int));
    if (!used) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }

    printf("\n");
    printf("== OPTIMIZE ENSEMBLE =========================================\n");
    printf("  %d score blocks (%d test samples, %d classes)\n",
           n, g_n_test, g_n_classes);
    printf("  Config: H=%d  EP=%d  VN=%d  HN=%d  TE=%d\n",
           g_hidden, g_epochs, g_split_vn, g_split_hn, g_target_err_x100);
    printf("  Algorithm: greedy (optimal subset, O(N²))\n");
    printf("  Threads:    %d\n", omp_get_max_threads());
    printf("\n");
    printf("  %-4s  %-7s  %-11s  %-7s  %s\n", "EN",   "acc[%]",  "correct",     "gain[%]", "member");
    printf("  %-4s  %-7s  %-11s  %-7s  %s\n", "----", "-------", "-----------", "-------", "-------------------");

    FILE *save_f = NULL;
    if (save_path && save_path[0]) {
        save_f = fopen(save_path, "w");
        if (save_f) fprintf(save_f, "# EN  acc  correct  total  gain  member\n");
    }

    float prev_acc = 0.0f;
    float best_acc = 0.0f;
    int best_en = 0;
    int order[1024]; /* selected member order */
    int order_n = 0;

    for (int step = 0; step < n; step++) {
        /* --max N: Stopp bei N Membern */
        if (g_max_members > 0 && order_n >= g_max_members) break;

        int best_m = -1;
        float best_step_acc = -1.0f;

        /* Try every unused member — parallel über Member */
        #pragma omp parallel for reduction(max:best_step_acc)
        for (int m = 0; m < n; m++) {
            if (used[m]) continue;

            const SCORE_TYPE *base = sum_scores;
            const SCORE_TYPE *sc = blocks[m].scores;
            int correct = 0;
            if (has_labels) {
                for (int s = 0; s < g_n_test; s++) {
                    size_t row_start = (size_t)s * row_sz;
                    int pred = 0;
                    SCORE_TYPE pred_val = base[row_start] + sc[row_start];
                    for (int k = 1; k < g_n_classes; k++) {
                        SCORE_TYPE v = base[row_start + k] + sc[row_start + k];
                        if (v > pred_val) { pred = k; pred_val = v; }
                    }
                    if (pred == (int)g_labels[s]) correct++;
                }
            }
            float acc = has_labels ? (float)correct * 100.0f / (float)g_n_test : 0.0f;

            if (acc > best_step_acc) {
                #pragma omp critical
                if (acc > best_step_acc) {
                    best_step_acc = acc;
                    best_m = m;
                }
            }
        }

        if (best_m < 0) break;  /* no candidate (should not happen) */

        /* Commit this member */
        used[best_m] = 1;
        order[order_n++] = best_m;
        const SCORE_TYPE *sc = blocks[best_m].scores;
        for (size_t i = 0; i < score_sz; i++)
            sum_scores[i] += sc[i];

        /* Evaluate final cumulative sum for this step */
        int correct = 0;
        for (int s = 0; s < g_n_test && has_labels; s++) {
            const SCORE_TYPE *row = sum_scores + (size_t)s * row_sz;
            int pred = 0;
            for (int k = 1; k < g_n_classes; k++)
                if (row[k] > row[pred]) pred = k;
            if (pred == (int)g_labels[s]) correct++;
        }
        float acc = has_labels ? (float)correct * 100.0f / (float)g_n_test : 0.0f;

                        char label[256];
        const char *_xn = xform_name_safe(blocks[best_m].xform_real);
        char _ml[256];
        member_label(&blocks[best_m], _ml, sizeof(_ml));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
        snprintf(label, sizeof(label), "%s:%s", _xn, _ml);
#pragma GCC diagnostic pop
        float gain = acc - prev_acc;

        if (has_labels) {
            printf("  %-4d  %-7.2f  %5d/%-5d  %-+7.2f  %-s\n",
                   step + 1, acc, correct, g_n_test, gain, label);
        }
        if (save_f && has_labels)
            fprintf(save_f, "%d  %.4f  %d  %d  %+.4f  %s\n",
                    step + 1, acc / 100.0f, correct, g_n_test, gain / 100.0f, label);

        if (acc > best_acc) {
            best_acc = acc;
            best_en = step + 1;
        }

        prev_acc = acc;
    }

    int best_corr = (int)(best_acc * (float)g_n_test / 100.0f + 0.5f);
    printf("\n══╡ OPTIMAL SUBSET ╞════════════════════════════════════════\n");
    printf("  Best:   EN=%d  acc=%.2f%%  (%d/%d)\n",
           best_en, best_acc, best_corr, g_n_test);
    printf("  Total:  %d members available\n", n);
    printf("  Used:   %d members in optimal subset\n", order_n);
    printf("\n  Optimal order (member indices):");
    for (int i = 0; i < order_n && i < 20; i++) {
        if (i % 10 == 0) printf("\n   ");
        printf("  #%d", order[i]);
    }
    if (order_n > 20) printf("  ... (%d more)", order_n - 20);
    printf("\n");
    printf("  Report: H=%d  EP=%d  VN=%d  HN=%d  TE=%d  eval=%.2f%%  err=%d  N=%d  opt_en=%d\n",
           g_hidden, g_epochs, g_split_vn, g_split_hn, g_target_err_x100,
           best_acc, g_n_test - best_corr, g_n_test, best_en);

    printf("\n══╡ REPORT ╞════════════════════════════════════════════════════════\n");
    printf("REPORT train=0.0%% (0) eval=%.2f%% (%d) err=%d lr=0.0000 time=0ms threads=1 members=%d\n",
           best_acc, best_corr, g_n_test - best_corr, best_en);
    if (save_f) { fclose(save_f); printf("  Saved:  %s\n", save_path); }
    free(sum_scores); free(used);
    return g_n_test;
}

/* ═══════════════════════════════════════════════════════════════════════
 * BEAM SEARCH — better than greedy, keeps W candidates at each step
 * ═══════════════════════════════════════════════════════════════════════
 * Algorithm: start with empty ensemble (1 candidate). For each step,
 * expand every candidate by trying all unused members. Keep top W.
 * W=1 = greedy. W≥10 finds near-optimal subsets. */
/* ── Dedup: (seed, member_idx, xform_real, color, enc_type, enc_width, source_file) ─ */
static void dedup_blocks(void) {
    int write = 0;
    for (int read = 0; read < n_blocks; read++) {
        int dup = 0;
        for (int j = 0; j < write; j++) {
            if (blocks[j].seed       == blocks[read].seed &&
                blocks[j].member_idx == blocks[read].member_idx &&
                blocks[j].xform_real == blocks[read].xform_real &&
                blocks[j].color      == blocks[read].color &&
                blocks[j].enc_type   == blocks[read].enc_type &&
                blocks[j].enc_width  == blocks[read].enc_width &&
                strcmp(blocks[j].source_file, blocks[read].source_file) == 0) {
                dup = 1;
                break;
            }
        }
        if (!dup) {
            if (write != read) blocks[write] = blocks[read];
            write++;
        } else {
            free(blocks[read].scores);
        }
    }
    if (write < n_blocks)
        printf("  [DEDUP] removed %d duplicate(s) (%d → %d)\n",
               n_blocks - write, n_blocks, write);
    n_blocks = write;
}

/* ── Beam/Expansion types (file scope für qsort comparator) ── */
typedef struct {
    SCORE_TYPE *sum;  /* [score_sz] cumulative */
    uint8_t *used;   /* [n] used mask */
    int n_used;
    float eval;
} BeamSlot;

typedef struct {
    int slot_idx;      /* which beam slot */
    int member_idx;    /* which member to add */
    float eval;
    SCORE_TYPE margin; /* sum(score[best] - score[second]) over all samples */
} Expansion;

/* ── Member file schreiben (best CHAN:ENC + optional Xform-Expansion) ── */
static void write_member_file_from_best(const char *path, int beam_width) {
    if (!path[0] || !g_best_blocks || !g_best_used) return;
    FILE *mf = fopen(path, "w");
    if (!mf) return;

    fprintf(mf, "# %s\n", g_xform_spec[0]
        ? "Expanded with --xform — XF:CHAN:ENC format"
        : "Optimal subset: XF:CHAN:ENC (original xforms)");
    fprintf(mf, "# Generated by merge-ensemble  H=%d  EP=%d  beam=%d\n",
            g_hidden, g_epochs, beam_width);

    /* FIX: iterate g_best_used (the ACTUAL best ensemble bitmask) instead of
     * g_best_order (members added at improvement steps). The beam search
     * explores many branches — the best ensemble may contain members that
     * were NEVER in best_real_members [2026-07-30]. */
    int n_best = 0, best_col[256], best_typ[256], best_wid[256];
    if (g_xform_spec[0]) {
        /* --xform mode: dedup by (color, enc_type, enc_width) from g_best_used */
        for (int bi = 0; bi < n_blocks && n_best < 256; bi++) {
            if (!g_best_used[bi]) continue;
            int dup = 0;
            for (int j = 0; j < n_best; j++)
                if (best_col[j] == g_best_blocks[bi].color &&
                    best_typ[j] == g_best_blocks[bi].enc_type &&
                    best_wid[j] == g_best_blocks[bi].enc_width) { dup = 1; break; }
            if (!dup) {
                best_col[n_best] = g_best_blocks[bi].color;
                best_typ[n_best] = g_best_blocks[bi].enc_type;
                best_wid[n_best] = g_best_blocks[bi].enc_width;
                n_best++;
            }
        }
        char _spec[1024];
        strncpy(_spec, g_xform_spec, sizeof(_spec) - 1);
        _spec[sizeof(_spec) - 1] = '\0';
        int xf_ids[64], n_xf = 0;
        for (char *tok = strtok(_spec, ","); tok && n_xf < 64; tok = strtok(NULL, ",")) {
            int id = ki_xform_parse_or_pipe(tok);
            if (id >= 0) xf_ids[n_xf++] = id;
        }
        for (int xi = 0; xi < n_xf; xi++) {
            const char *xn = ki_xform_str(xf_ids[xi]);
            for (int si = 0; si < n_best; si++)
                fprintf(mf, "%s:%s:%s%d\n", xn,
                    ki_color_name(best_col[si]),
                    ki_enc_name_short(best_typ[si]), best_wid[si]);
        }
        printf("  Member-file: %s  (%d best CHAN:ENC × %d xforms = %d specs)\n",
               path, n_best, n_xf, n_best * n_xf);
    } else {
        /* Original xforms: iterate g_best_used */
        int _n_members = 0;
        for (int bi = 0; bi < n_blocks; bi++) {
            if (!g_best_used[bi]) continue;
            char label[64];
            const char *_xn = (g_best_blocks[bi].xform_real >= 0)
                              ? xform_name_safe(g_best_blocks[bi].xform_real)
                              : xform_group_label(g_best_blocks[bi].xform_id);
            member_label(&g_best_blocks[bi], label, sizeof(label));
            fprintf(mf, "%s:%s  0x%08X\n", _xn, label,
                    g_best_blocks[bi].w0_marker);
            _n_members++;
        }
        printf("  Member-file: %s  (%d members)\n", path, _n_members);
    }
    fclose(mf);
}

/* Global beam pointer for qsort comparator */
static BeamSlot *g_beam = NULL;

/* ── Expansion comparator for qsort (abs + marginal modes) ── */
static int cmp_exp(const void *a, const void *b) {
    const Expansion *ea = (const Expansion *)a;
    const Expansion *eb = (const Expansion *)b;
    double va = g_expansion_sort ? (double)ea->eval - g_beam[ea->slot_idx].eval : (double)ea->eval;
    double vb = g_expansion_sort ? (double)eb->eval - g_beam[eb->slot_idx].eval : (double)eb->eval;
    if (vb > va + 0.0001) return 1;
    if (va > vb + 0.0001) return -1;
    if (eb->margin > ea->margin) return 1;
    if (ea->margin > eb->margin) return -1;
    return 0;
}

static int merge_and_beam(const char *save_path, int beam_width,
                           int initial_member, uint8_t *pool_exclude)
{
    if (n_blocks == 0) { printf("  No score blocks loaded.\n"); return 0; }
    /* Eigenen pool_exclude allozieren wenn keiner übergeben (single beam run).
     * _owned_exclude != NULL → single-try: pool_exclude darf Mitglieder NICHT
     * global sperren (sonst verbraucht breiterer Beam schneller den Pool
     * und findet schlechtere Lösungen als schmalerer Beam).
     * _owned_exclude == NULL → multi-try: pool_exclude vom Aufrufer, Mitglieder
     * müssen global gesperrt werden, damit Tries sich nicht überlappen. */
    uint8_t *_owned_exclude = NULL;
    if (!pool_exclude) {
        _owned_exclude = (uint8_t *)calloc((size_t)n_blocks, 1);
        pool_exclude = _owned_exclude;
    }
    int has_labels = (g_labels != NULL);
    int n = n_blocks;
    size_t score_sz = (size_t)g_n_test * (size_t)g_n_classes;
    size_t row_sz   = (size_t)g_n_classes;

    if (beam_width < 1) beam_width = 1;
    if (beam_width > 64) beam_width = 64;

    /* Beam: array of candidates, each with cumulative scores + used mask */
    BeamSlot *beam = (BeamSlot *)calloc((size_t)beam_width, sizeof(BeamSlot));
    g_beam = beam;  /* for qsort comparator */
    int n_beam = 1;  /* may be increased by seed-fill below */
    /* Initial candidate: empty or pre-seeded with initial_member */
    if (initial_member >= 0 && initial_member < n) {
        beam[0].sum = (SCORE_TYPE *)calloc(score_sz, sizeof(SCORE_TYPE));
        for (size_t i = 0; i < score_sz; i++)
            beam[0].sum[i] = blocks[initial_member].scores[i];
        beam[0].used = (uint8_t *)calloc((size_t)n, 1);
        beam[0].used[initial_member] = 1;
        beam[0].n_used = 1;
        int _c = 0;
        for (int s = 0; s < g_n_test && has_labels; s++) {
            const SCORE_TYPE *row = beam[0].sum + (size_t)s * row_sz;
            int pred = 0;
            for (int k = 1; k < g_n_classes; k++)
                if (row[k] > row[pred]) pred = k;
            if (pred == (int)g_labels[s]) _c++;
        }
        beam[0].eval = (float)_c * 100.0f / (float)g_n_test;
        /* Fill remaining beam_width-1 slots with best singles (≠ seed) so the
         * beam state matches the unseeded case (beam_width single-member slots).
         * Without this, seeded and unseeded runs diverge because the beam starts
         * with different slot depths (1 pair vs 2 singles for beam_width=2). */
        if (beam_width > 1 && has_labels && !pool_exclude) {
            /* Evaluate all members as singles */
            typedef struct { int idx; float eval; SCORE_TYPE margin; } _SingleSc;
            _SingleSc *_ss = (_SingleSc *)malloc((size_t)n * sizeof(_SingleSc));
            if (_ss) {
                #pragma omp parallel for schedule(static)
                for (int _mi = 0; _mi < n; _mi++) {
                    if (pool_exclude && pool_exclude[_mi]) { _ss[_mi].idx = _mi; _ss[_mi].eval = -1.0f; _ss[_mi].margin = 0; continue; }
                    const SCORE_TYPE *_sc = blocks[_mi].scores;
                    int _ok = 0; SCORE_TYPE _mg = 0;
                    for (int _s = 0; _s < g_n_test; _s++) {
                        const SCORE_TYPE *_row = _sc + (size_t)_s * row_sz;
                        int _bk = 0, _sk = -1;
                        for (int _k = 1; _k < g_n_classes; _k++) {
                            if (_row[_k] > _row[_bk]) { _sk = _bk; _bk = _k; }
                            else if (_sk < 0 || _row[_k] > _row[_sk]) _sk = _k;
                        }
                        if (_bk == (int)g_labels[_s]) _ok++;
                        if (_sk >= 0) _mg += _row[_bk] - _row[_sk];
                    }
                    _ss[_mi].idx = _mi;
                    _ss[_mi].eval = (float)_ok * 100.0f / (float)g_n_test;
                    _ss[_mi].margin = _mg;
                }
                /* Sort by eval desc with threshold-based ties (same as expansion sort) */
                float _gt = g_max_mode ? 0.0f : g_min_gain;
                for (int _i = 0; _i < n - 1; _i++)
                    for (int _j = _i + 1; _j < n; _j++)
                        if (_ss[_j].eval > _ss[_i].eval + _gt ||
                            (fabsf(_ss[_j].eval - _ss[_i].eval) <= _gt + 0.0001f && _ss[_j].margin > _ss[_i].margin)) {
                            _SingleSc _t = _ss[_i]; _ss[_i] = _ss[_j]; _ss[_j] = _t;
                        }
                /* Fill beam[1..beam_width-1] with best singles ≠ seed */
                int _fill = 0;
                for (int _si = 1; _si < beam_width && _fill < n; ) {
                    int _mi = _ss[_fill].idx;
                    if (_mi != initial_member && !(pool_exclude && pool_exclude[_mi])) {
                        beam[_si].sum = (SCORE_TYPE *)malloc(score_sz * sizeof(SCORE_TYPE));
                        memcpy(beam[_si].sum, blocks[_mi].scores, score_sz * sizeof(SCORE_TYPE));
                        beam[_si].used = (uint8_t *)calloc((size_t)n, 1);
                        beam[_si].used[_mi] = 1;
                        beam[_si].n_used = 1;
                        beam[_si].eval = _ss[_fill].eval;
                        _si++;
                    }
                    _fill++;
                }
                n_beam = beam_width;
                free(_ss);
            }
        }
    } else {
        beam[0].sum   = (SCORE_TYPE *)calloc(score_sz, sizeof(SCORE_TYPE));
        beam[0].used  = (uint8_t *)calloc((size_t)n, 1);
        beam[0].n_used = 0;
        beam[0].eval   = 0.0f;
    }

    /* DEBUG REMOVED */

    /* Temp: evaluate candidate from parent + one new member */
    SCORE_TYPE *trial = (SCORE_TYPE *)malloc(score_sz * sizeof(SCORE_TYPE));
    if (!trial) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }

    /* Results buffer for current expansion step */
    Expansion *exp = (Expansion *)malloc((size_t)n * (size_t)beam_width * sizeof(Expansion));

    float best_eval = 0.0f;
    float gain_threshold = g_max_mode ? 0.0f : g_min_gain;
    int best_n = 0;                 /* total members in best_used */
    int best_real_n = 0;            /* only those with ≥threshold gain */
    int best_real_members[2048];    /* member indices of each real improvement */
    uint8_t *best_used = (uint8_t *)calloc((size_t)n, 1);  /* which members form best ensemble */
    if (initial_member >= 0) best_used[initial_member] = 1;  /* seed immer tracken */
    int best_member_idx = -1;       /* which member was added at the best step */
    int no_improve_steps = 0;       /* early-stop counter */

    printf("\n");
    printf("== BEAM SEARCH ENSEMBLE =====================================\n");
    printf("  %d score blocks (%d test samples, %d classes)\n",
           n, g_n_test, g_n_classes);
     printf("  Config: H=%d  EP=%d  VN=%d  HN=%d  TE=%d\n",
            g_hidden, g_epochs, g_split_vn, g_split_hn, g_target_err_x100);
    printf("  Score Mode:  %s\n",
           COUNTER_TYPE_IS_FLOAT ? "float32 (IEEE 754)" : "int32 (fixed-point)");
    printf("  Beam width: %d\n", beam_width);
    printf("  Threads:    %d\n", omp_get_max_threads());
    printf("\n");
    printf("  %-4s  %-7s  %-11s  %-7s  %-6s  %-5s  %-10s  %-13s  %-13s  %-7s  %s\n",
           "EN",   "acc[%]",  "correct",     "gain[%]", "eval", "err", "W0[0]", "MIN-score", "MAX-score", "t[s]", "member");
    printf("  %-4s  %-7s  %-11s  %-7s  %-6s  %-5s  %-10s  %-13s  %-13s  %-7s  %s\n",
           "----", "-------", "-----------", "-------", "------", "-----", "----------", "-----------", "-----------", "------", "-------------------");

    FILE *save_f = NULL;
    if (save_path && save_path[0]) {
        save_f = fopen(save_path, "w");
        if (save_f) fprintf(save_f, "# EN  acc  correct  total  gain  member\n");
    }

    float prev_acc = 0.0f;
    int en_offset = 0;  /* EN numbering offset for pre-seeded initial member */
    struct timeval _step_start;
    gettimeofday(&_step_start, NULL);

    /* If beam was pre-seeded with initial_member, show it as EN=1 */
    if (initial_member >= 0 && initial_member < n) {
        char _label[256];
        const char *_xn = xform_name_safe(blocks[initial_member].xform_real);
        char _ml[256];
        member_label(&blocks[initial_member], _ml, sizeof(_ml));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
        snprintf(_label, sizeof(_label), "%s:%s", _xn, _ml);
#pragma GCC diagnostic pop
        int _c = (int)(beam[0].eval * (float)g_n_test / 100.0f + 0.5f);
        int _err0 = g_n_test - _c;
#if COUNTER_TYPE_IS_FLOAT
        {   struct timeval _now; gettimeofday(&_now, NULL);
            double _elapsed = (double)(_now.tv_sec - _step_start.tv_sec) + (double)(_now.tv_usec - _step_start.tv_usec) * 1e-6;
        printf("  %-4d  %-7.2f  %5d/%-5d  %-+7.2f  %-5.1f%%  E=%5d  0x%08X  %+4.0f  %+4.0f  %-7.2f  %s\n",
               1, beam[0].eval, _c, g_n_test, beam[0].eval,
               beam[0].eval, _err0,
               blocks[initial_member].w0_marker,
               (double)blocks[initial_member].score_min,
               (double)blocks[initial_member].score_max,
               _elapsed, _label);
        }
#else
        {   struct timeval _now; gettimeofday(&_now, NULL);
            double _elapsed = (double)(_now.tv_sec - _step_start.tv_sec) + (double)(_now.tv_usec - _step_start.tv_usec) * 1e-6;
        printf("  %-4d  %-7.2f  %5d/%-5d  %-+7.2f  %5.1f%%  E=%5d  0x%08X  %+13" PRId64 "  %+13" PRId64 "  %-7.2f  %s\n",
               1, beam[0].eval, _c, g_n_test, beam[0].eval,
               beam[0].eval, _err0,
               blocks[initial_member].w0_marker,
               (int64_t)blocks[initial_member].score_min,
               (int64_t)blocks[initial_member].score_max,
               _elapsed, _label);
        }
#endif
        prev_acc = beam[0].eval;
        best_eval = beam[0].eval;
        best_n = 1;
        best_real_members[0] = initial_member;
        best_real_n = 1;
        en_offset = 1;
    }

    for (int step = 0; step < n; step++) {
        /* --max N: Stopp bei N Membern */
        if (g_max_members > 0 && best_n >= g_max_members) break;
        int n_exp = 0;

        /* ── Parallel expand: beam × unused members ── *
          * KEIN memcpy, KEIN trial-Buffer: base+sc wird ON THE FLY
          * im argmax-Loop addiert. Spart 2×400KB Speicherzugriff pro
          * Expansion — bei 10530 Exp./Step × 221 Steps = ~1.9TB weniger
          * Traffic. */
        #pragma omp parallel
        {
            #pragma omp for collapse(2) schedule(guided) nowait
            for (int si = 0; si < n_beam; si++) {
                for (int mi = 0; mi < n; mi++) {
                    if (beam[si].used[mi]) continue;
                    if (pool_exclude && pool_exclude[mi]) continue;

                    const SCORE_TYPE *base = beam[si].sum;
                    const SCORE_TYPE *sc = blocks[mi].scores;
                    int correct = 0;
                    SCORE_TYPE margin = 0;
                    if (has_labels) {
                        for (int s = 0; s < g_n_test; s++) {
                            size_t row_start = (size_t)s * row_sz;
                            int best_k = 0, second_k = -1;
                            SCORE_TYPE best_val = base[row_start] + sc[row_start];
                            SCORE_TYPE second_val = 0;
                            for (int k = 1; k < g_n_classes; k++) {
                                SCORE_TYPE v = base[row_start + k] + sc[row_start + k];
                                if (v > best_val) {
                                    second_k = best_k; second_val = best_val;
                                    best_k = k; best_val = v;
                                } else if (second_k < 0 || v > second_val) {
                                    second_k = k; second_val = v;
                                }
                            }
                            if (best_k == (int)g_labels[s]) correct++;
                            if (second_k >= 0) margin += best_val - second_val;
                        }
                    }
                    float acc = has_labels ? (float)correct * 100.0f / (float)g_n_test : 0.0f;

                    int pos = __sync_fetch_and_add(&n_exp, 1);
                    exp[pos].slot_idx   = si;
                    exp[pos].member_idx = mi;
                    exp[pos].eval       = acc;
                    exp[pos].margin     = margin;
                }
            }
        }

        /* ── Sort expansions via qsort (O(N log N) statt O(N²)) ── */
        qsort(exp, (size_t)n_exp, sizeof(Expansion), cmp_exp);

        /* ── Debug: top 5 expansions at this step ── */
        if (g_debug) {
            int _nshow = n_exp < 5 ? n_exp : 5;
            printf("  -- step %d+: top %d expansions (n_exp=%d) --\n",
                   step + 1 + en_offset, _nshow, n_exp);
            for (int ei = 0; ei < _nshow; ei++) {
                int _si = exp[ei].slot_idx;
                int _mi = exp[ei].member_idx;
                char _ml[64];
                const char *_xn = (blocks[_mi].xform_real >= 0)
                                  ? xform_name_safe(blocks[_mi].xform_real)
                                  : xform_group_label(blocks[_mi].xform_id);
                member_label(&blocks[_mi], _ml, sizeof(_ml));
                size_t _sz = (size_t)g_n_test * (size_t)g_n_classes;
                uint64_t _crc = score_crc64(blocks[_mi].scores, _sz);
#if COUNTER_TYPE_IS_FLOAT
                printf("       slot=%d mi=%d  %s:%s  eval=%.2f  margin=%.0f  CRC=0x%016" PRIx64 "\n",
                       _si, _mi, _xn, _ml, exp[ei].eval, (double)exp[ei].margin, _crc);
#else
                printf("       slot=%d mi=%d  %s:%s  eval=%.2f  margin=%" PRId64 "  CRC=0x%016" PRIx64 "\n",
                       _si, _mi, _xn, _ml, exp[ei].eval, exp[ei].margin, _crc);
#endif
            }
            fflush(stdout);
        }

        /* Build new beam: jedes Member kann NUR EINMAL über ALLE Slots+Steps ausgewählt werden.
         * Ersetzt das alte seen_mi + pairwise-diversity durch globale pool_exclude.
         * Dadurch sind alle Beam-Pfade automatisch divers (kein --diversity nötig). */
        BeamSlot *next = (BeamSlot *)calloc((size_t)beam_width, sizeof(BeamSlot));
        int n_next = 0;
        uint8_t *seen_mask = (uint8_t *)calloc((size_t)n, 1); /* temp for dedup */

        for (int ei = 0; ei < n_exp && n_next < beam_width; ei++) {
            int si = exp[ei].slot_idx;
            int mi = exp[ei].member_idx;

            /* Global exclusion: jedes Member kann NUR EINMAL gewählt werden (über alle Schritte) */
            if (pool_exclude && pool_exclude[mi]) continue;

            /* Build candidate: copy parent, add mi */
            next[n_next].sum  = (SCORE_TYPE *)malloc(score_sz * sizeof(SCORE_TYPE));
            memcpy(next[n_next].sum, beam[si].sum, score_sz * sizeof(SCORE_TYPE));
            const SCORE_TYPE *sc = blocks[mi].scores;
            for (size_t i = 0; i < score_sz; i++)
                next[n_next].sum[i] += sc[i];

            next[n_next].used = (uint8_t *)malloc((size_t)n);
            memcpy(next[n_next].used, beam[si].used, (size_t)n);
            next[n_next].used[mi] = 1;
            next[n_next].n_used = beam[si].n_used + 1;
            next[n_next].eval   = exp[ei].eval;

            /* Dedup: check if this member set is already in next */
            int dup = 0;
            for (int c = 0; c < n_next; c++) {
                int same = (next[c].n_used == next[n_next].n_used);
                if (same) {
                    for (int mm = 0; mm < n && same; mm++)
                        if (next[c].used[mm] != next[n_next].used[mm])
                            same = 0;
                }
                if (same) { dup = 1; break; }
            }
            if (dup) {
                free(next[n_next].sum);
                free(next[n_next].used);
                continue;
            }

            /* Member global sperren (nur im multi-try Mode, nicht im single beam run).
             * Im single beam run würde dies breiteren Beam bestrafen: mehr Pfade →
             * mehr Member gesperrt → weniger Diversität → schlechtere Ergebnisse.
             * Duplikate in best_real_members werden am Ende entfernt. */
            if (pool_exclude && !_owned_exclude) pool_exclude[mi] = 1;

                if (exp[ei].eval > best_eval + gain_threshold) {
                    best_eval = exp[ei].eval;
                    best_member_idx = mi;
                    best_n = 0;
                    for (int _bi = 0; _bi < n; _bi++)
                        if (best_used[_bi]) best_n++;
                }

            n_next++;
        }
        free(seen_mask);

        /* Free old beam */
        for (int i = 0; i < n_beam; i++) { free(beam[i].sum); free(beam[i].used); }
        free(beam);
        beam = next;
        n_beam = n_next;

            /* Show best expansion from this step (exp[0], not stale global best_member_idx)
             * so we report the actual member selected, even if gain = 0.00% */
            {
                char _label[256];
                float _member_eval = 0.0f;
                int _m = best_member_idx;
                if (_m < 0 && n_exp > 0) _m = exp[0].member_idx;  /* fallback */
                if (_m >= 0) {
                    const char *_xn = xform_name_safe(blocks[_m].xform_real);
                    char _ml[256];
                    member_label(&blocks[_m], _ml, sizeof(_ml));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
                    snprintf(_label, sizeof(_label), "%s:%s", _xn, _ml);
#pragma GCC diagnostic pop
                    _member_eval = blocks[_m].file_eval;
                } else
                    snprintf(_label, sizeof(_label), "-");

                /* Track the GLOBAL best (may be from an earlier step) */
                float _gain = best_eval - prev_acc;
                int _corr = (int)(best_eval * (float)g_n_test / 100.0f + 0.5f);
                if (_gain > gain_threshold) {
                    struct timeval _now; gettimeofday(&_now, NULL);
                    double _elapsed = (double)(_now.tv_sec - _step_start.tv_sec) + (double)(_now.tv_usec - _step_start.tv_usec) * 1e-6;
#if COUNTER_TYPE_IS_FLOAT
        printf("  %-4d  %-7.2f  %5d/%-5d  %-+7.2f  %5.1f%%  E=%d  0x%08X  %-+13.0f  %-+13.0f  %-7.2f  %s\n",
               step + 1 + en_offset, best_eval, _corr, g_n_test, _gain,
               _member_eval,
               (_m >= 0) ? blocks[_m].err : 0,
               (_m >= 0) ? blocks[_m].w0_marker : 0,
               (double)((_m >= 0) ? blocks[_m].score_min : 0),
               (double)((_m >= 0) ? blocks[_m].score_max : 0),
               _elapsed, _label);
#else
        printf("  %-4d  %-7.2f  %5d/%-5d  %-+7.2f  %5.1f%%  E=%d  0x%08X  %-+13" PRId64 "  %-+13" PRId64 "  %-7.2f  %s\n",
               step + 1 + en_offset, best_eval, _corr, g_n_test, _gain,
               _member_eval,
               (_m >= 0) ? blocks[_m].err : 0,
               (_m >= 0) ? blocks[_m].w0_marker : 0,
               (int64_t)((_m >= 0) ? blocks[_m].score_min : 0),
               (int64_t)((_m >= 0) ? blocks[_m].score_max : 0),
               _elapsed, _label);
#endif
                if (save_f)
#if COUNTER_TYPE_IS_FLOAT
                    fprintf(save_f, "%d  %.4f  %d  %d  %+.4f  %d  0x%08X  %.0f  %.0f  %s\n",
                            step + 1 + en_offset, best_eval / 100.0f, _corr, g_n_test, _gain / 100.0f,
                            (_m >= 0) ? blocks[_m].err : 0,
                            (_m >= 0) ? blocks[_m].w0_marker : 0,
                            (double)((_m >= 0) ? blocks[_m].score_min : 0),
                            (double)((_m >= 0) ? blocks[_m].score_max : 0), _label);
#else
                    fprintf(save_f, "%d  %.4f  %d  %d  %+.4f  %d  0x%08X  %" PRId64 "  %" PRId64 "  %s\n",
                            step + 1 + en_offset, best_eval / 100.0f, _corr, g_n_test, _gain / 100.0f,
                            (_m >= 0) ? blocks[_m].err : 0,
                            (_m >= 0) ? blocks[_m].w0_marker : 0,
                            (_m >= 0) ? blocks[_m].score_min : 0,
                            (_m >= 0) ? blocks[_m].score_max : 0, _label);
#endif
                /* Track displayed member for member-out (only steps with
                 * gain ≥ gain_threshold). Copy FULL beam candidate set. */
                if (_m >= 0) {
                    best_real_members[best_real_n++] = _m;
                    memset(best_used, 0, (size_t)n);
                    memcpy(best_used, beam[0].used, (size_t)n);
                }
                no_improve_steps = 0;
                prev_acc = best_eval;
            } else {
                no_improve_steps++;
                if (no_improve_steps >= beam_width) {
                    printf("\n  [Early stop] no improvement after %d steps\n", no_improve_steps);
                    break;
                }
            }
        }
        fflush(stdout);

        if (n_beam == 0) break;
    }

    /* Count members in best_used (already set during beam loop) */
    best_n = 0;
    for (int _bi = 0; _bi < n; _bi++)
        if (best_used[_bi]) best_n++;
    int real_n = best_n;

    /* Update global best (for multi-try) */
    if (best_eval > g_best_eval || g_best_used == NULL) {
        g_best_eval = best_eval;
        g_best_n    = real_n;
        if (!g_best_used) g_best_used = (uint8_t *)calloc((size_t)n, 1);
        for (int i = 0; i < n; i++) g_best_used[i] = best_used[i];
        /* Save blocks + selection order for member-out */
        if (!g_best_blocks) g_best_blocks = (ScoreBlock *)malloc((size_t)n * sizeof(ScoreBlock));
        memcpy(g_best_blocks, blocks, (size_t)n * sizeof(ScoreBlock));
        g_best_order_n = best_real_n;
        for (int _ri = 0; _ri < best_real_n; _ri++)
            g_best_order[_ri] = best_real_members[_ri];
    }

    int best_corr = (int)(best_eval * (float)g_n_test / 100.0f + 0.5f);
    best_n = real_n;

    printf("\n══╡ BEAM BEST ╞════════════════════════════════════════════\n");
    printf("  Best:  %d members  acc=%.2f%%  (%d/%d)\n",
           best_n, best_eval, best_corr, g_n_test);
    printf("  Beam:  width=%d  blocks=%d%s\n",
           beam_width, n, g_max_mode ? "  --max mode" : "");

    /* ── Report line (compact, inside BEAM BEST header) ── */
    char _last_label[256] = "-";
    if (best_member_idx >= 0) {
        const char *_xn = xform_name_safe(blocks[best_member_idx].xform_real);
        char _ml[256];
        member_label(&blocks[best_member_idx], _ml, sizeof(_ml));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
        snprintf(_last_label, sizeof(_last_label), "%s:%s", _xn, _ml);
#pragma GCC diagnostic pop
    }
    printf("  Report:     H=%d  EP=%d  VN=%d  HN=%d  TE=%d  eval=%.2f%%  err=%d  N=%d  best_en=%d  beam=%d  last=%s\n",
           g_hidden, g_epochs, g_split_vn, g_split_hn, g_target_err_x100,
           best_eval, g_n_test - best_corr, g_n_test, best_n, beam_width,
           _last_label);
    /* Write member file if --member-out was specified (before free of best_used)
     * Skip when --xform is active — the caller writes the expanded version. */
    if (g_member_out[0] && !g_xform_spec[0]) {
        FILE *mf = fopen(g_member_out, "w");
        if (mf) {
            fprintf(mf, "# Optimal subset: %d members, %.2f%% eval (beam width %d)\n",
                    best_n, best_eval, beam_width);
            fprintf(mf, "# Generated by merge-ensemble\n");
            for (int i = 0; i < n; i++) {
                if (!best_used[i]) continue;
                char label[256];
                const char *_xn = (blocks[i].xform_real >= 0)
                                  ? xform_name_safe(blocks[i].xform_real)
                                  : xform_group_label(blocks[i].xform_id);
                member_label(&blocks[i], label, sizeof(label));
                fprintf(mf, "%s:%s\n", _xn, label);
            }
            fclose(mf);
            printf("  MemberFile: %s  (%d members)\n", g_member_out, best_n);
        } else {
            fprintf(stderr, "  [ERROR] Cannot write --member-out: %s\n", g_member_out);
        }
    }
    if (save_f) { fclose(save_f); printf("  Saved:      %s\n", save_path); }
    for (int i = 0; i < n_beam; i++) free(beam[i].sum);
    for (int i = 0; i < n_beam; i++) free(beam[i].used);
    free(beam); free(trial); free(exp); free(best_used); free(_owned_exclude);

    printf("\n══╡ REPORT ╞════════════════════════════════════════════════════════\n");
    printf("REPORT train=0.0%% (0) eval=%.2f%% (%d) err=%d lr=0.0000 time=0ms threads=1 members=%d\n",
           best_eval, best_corr, g_n_test - best_corr, best_n);

    return g_n_test;
}

static void show_help(const char *prog) {
    printf("Usage: %s DIR [options]\n", prog);
    printf("\n");
    printf("Merge score archives (.ens) to an ensemble accuracy curve (EN=1..N).\n");
    printf("\n");
    printf("Options:\n");
    printf("  DIR          Directory containing .ens score archives\n");
    printf("  --save FILE  Save cumulative accuracy to FILE (default: DIR/merge.dat)\n");
    printf("  --expansion-sort MODE  Sort expansions by 'abs' (accuracy) or 'marginal' (gain)  (default: abs)\n");
    printf("  --filter L   Filter members by label or eval threshold\n");
    printf("               Label exclude : --filter GB=sig8,BL=down8 (case-insensitive, no spaces)\n");
    printf("               Eval threshold: --filter eval gt 58.1    (gt, lt, ge, le, eq, no quotes needed)\n");
            printf("  --optimize   Greedy optimal subset (members sorted by contribution)\n");
            printf("  --beam N     Beam search with width N (N=10 recommended, better than greedy)\n");
            printf("  --tries N    Repeat beam search N times with shuffle (keeps best)\n");
            printf("  --max        Maximize eval%% (ignore 0.01%% threshold)\n");
            printf("  --max N      Stop optimize/beam at N members\n");
            printf("  --min-gain F Minimum gain [%%] to keep adding (default: %.2f)\n", g_min_gain);
            printf("  --diversity N Enforce min path diversity (N member diff) in beam (default: 0=off)\n");
            printf("  --seed-sort   Print all single-member eval sorted by strength (no search)\n");
            printf("  --member-seed SPEC Pre-seed beam with member spec (e.g. \"rot22@spiral:BP:sig8\")\n");
            printf("  --member-out FILE  Export optimal subset as --member-file format (use with --beam)\n");
    printf("  --check      Validate .ens files in DIR (header, size, scores, labels)\n");
    printf("  -h, --help   Show this help text\n");
    printf("\n");
    printf("Output:\n");
    printf("  Table: EN  acc[%%]  correct/total  gain\n");
    printf("  File:  merge.dat  (for plotting)\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s scores/\n", prog);
    printf("  %s scores/ --max 50\n", prog);
    printf("  %s scores/ --optimize\n", prog);
    printf("\n");
    printf("The .ens archives are created by the Otto Score trainer via --export-merge-scores DIR.\n");
    printf("Each archive contains all ensemble members from one training run.\n");
}

/* ═══════════════════════════════════════════════════════════════════════
 * CHECK — validate .ens file integrity
 * ═══════════════════════════════════════════════════════════════════════
 * Self-contained validator that reads one .ens file, checks header,
 * metadata, file size, scores (all-zero, outliers), and labels.
 * Does NOT modify global state (g_n_test, blocks, etc.).
 */
typedef struct { int ok; char msg[256]; } CheckResult;

static CheckResult check_archive(const char *path) {
    CheckResult res = {0, ""};
    FILE *f = fopen(path, "rb");
    if (!f) {
        res.ok = -1; snprintf(res.msg, sizeof(res.msg), "Cannot open: %s", strerror(errno));
        return res;
    }

    /* File size */
    struct stat st;
    if (stat(path, &st) != 0) {
        res.ok = -1; snprintf(res.msg, sizeof(res.msg), "stat: %s", strerror(errno));
        fclose(f); return res;
    }
    off_t file_size = st.st_size;

    /* ── Read header ── */
    uint32_t hdr_base[11];
    if (fread(hdr_base, sizeof(uint32_t), 11, f) != 11) {
        res.ok = -1; snprintf(res.msg, sizeof(res.msg), "Short header");
        fclose(f); return res;
    }
    uint32_t magic     = hdr_base[0];
    uint32_t version   = hdr_base[1];
    uint32_t n_test    = hdr_base[2];
    uint32_t n_classes = hdr_base[3];
    uint32_t n_members = hdr_base[4];
    uint32_t hidden    = hdr_base[5];
    uint32_t epochs    = hdr_base[6];
    uint32_t split_vn  = hdr_base[7];
    uint32_t split_hn  = hdr_base[8];

    if (magic != 0x454E534D) {
        res.ok = -1; snprintf(res.msg, sizeof(res.msg), "Bad magic 0x%08X", magic);
        fclose(f); return res;
    }
    if (version < 1 || version > 10) {
        res.ok = -1; snprintf(res.msg, sizeof(res.msg), "Unsupported version %u", version);
        fclose(f); return res;
    }
    if (n_test == 0 || n_test > 1000000) {
        res.ok = -1; snprintf(res.msg, sizeof(res.msg), "Suspicious n_test=%u", n_test);
        fclose(f); return res;
    }
    if (n_classes < 1 || n_classes > 1000) {
        res.ok = -1; snprintf(res.msg, sizeof(res.msg), "Suspicious n_classes=%u", n_classes);
        fclose(f); return res;
    }
    if (n_members < 1 || n_members > 10000) {
        res.ok = -1; snprintf(res.msg, sizeof(res.msg), "Suspicious n_members=%u", n_members);
        fclose(f); return res;
    }

    /* Compute expected file size (header + metadata + scores + labels) */
    off_t expected = 11 * 4;  /* base header */
    off_t after_header = 11 * 4;  /* file position after reading all header fields */

    /* v3+: timestamp */
    float ensemble_eval = 0.0f;
    uint32_t w0_check = 0;
    if (version >= 3) { expected += 8; after_header += 8; }
    /* v4+: ensemble_eval (seek back, read it, then continue) */
    if (version >= 4) {
        expected += 4; after_header += 4;
        long save_pos = ftell(f);
        fseek(f, 11 * 4 + (version >= 3 ? 8 : 0), SEEK_SET);
        if (fread(&ensemble_eval, 4, 1, f) != 1) ensemble_eval = 0.0f;
        /* v10+: read W0 marker after ensemble_eval */
        if (version >= 10) {
            expected += 4; after_header += 4;
            fread(&w0_check, 4, 1, f);
        }
        fseek(f, save_pos, SEEK_SET);
    }

    /* Seek past header to metadata position */
    fseek(f, after_header, SEEK_SET);

    /* v5: xform list (read here, position doesn't change for non-v5) */
    if (version == 5) {
        uint32_t nxf;
        if (fread(&nxf, 4, 1, f) != 1 || nxf > 64) {
            res.ok = -1; snprintf(res.msg, sizeof(res.msg), "Bad v5 xform list");
            fclose(f); return res;
        }
        expected += 4 + (off_t)nxf * 4;
        if ((off_t)nxf * 4 > file_size - expected) {
            res.ok = -1; snprintf(res.msg, sizeof(res.msg), "v5 xform list exceeds file");
            fclose(f); return res;
        }
        fseek(f, (long)nxf * 4, SEEK_CUR);
    }
    uint32_t meta_bytes = 0;
    if (version >= 2) {
        if (version >= 7) {
            for (uint32_t m = 0; m < n_members; m++) {
                for (int field = 0; field < 4; field++) {
                    uint8_t slen;
                    if (fread(&slen, 1, 1, f) != 1) {
                        res.ok = -1; snprintf(res.msg, sizeof(res.msg), "Bad meta (member %u)", m);
                        fclose(f); return res;
                    }
                    if (slen > 128) {
                        res.ok = -1; snprintf(res.msg, sizeof(res.msg), "Meta string too long (member %u, len=%u)", m, slen);
                        fclose(f); return res;
                    }
                    meta_bytes += 1 + slen;
                    fseek(f, slen, SEEK_CUR);
                }
            }
        } else {
            meta_bytes = n_members * 4;
            fseek(f, meta_bytes, SEEK_CUR);
        }
    }
    expected += meta_bytes;

    /* Scores: n_members × n_test × n_classes × 8 (v7 int64) or 4 (v8 int32) */
    size_t score_sz = (size_t)n_test * (size_t)n_classes;
    int score_elem_size = (version >= 8) ? 4 : 8;  /* v8=int32, v9=float, v1-v7=int64 */
    off_t score_bytes = (off_t)n_members * (off_t)score_sz * (off_t)score_elem_size;
    expected += score_bytes;

    /* Labels: n_test */
    expected += (off_t)n_test;

    /* ── Check file size ── */
    if (file_size != expected) {
        snprintf(res.msg, sizeof(res.msg), "Size mismatch: file=%lld expected=%lld (diff%+lld)",
                 (long long)file_size, (long long)expected, (long long)(file_size - expected));
        if (file_size < expected) { res.ok = -1; fclose(f); return res; }
        res.ok = 1;  /* warn but continue */
    }

    /* ── Read and validate scores (first member only, then sample rest) ── */
    int all_zero = 1, has_outliers = 0;
    int64_t vmin = INT64_MAX, vmax = INT64_MIN;

    for (uint32_t m = 0; m < n_members && m < 5; m++) {
        int64_t *sc = (int64_t *)malloc((size_t)score_sz * sizeof(int64_t));
        if (!sc) { res.ok = -1; snprintf(res.msg, sizeof(res.msg), "OOM"); fclose(f); return res; }
        if (version >= 8) {
            int32_t *buf32 = (int32_t *)malloc((size_t)score_sz * sizeof(int32_t));
            if (!buf32) { res.ok = -1; snprintf(res.msg, sizeof(res.msg), "OOM"); fclose(f); return res; }
            size_t nr = fread(buf32, sizeof(int32_t), score_sz, f);
            if (nr != score_sz) {
                snprintf(res.msg, sizeof(res.msg), "Short scores (member %u: read %zu of %zu)", m, nr, score_sz);
                free(buf32); free(sc); fclose(f); return res;
            }
            for (size_t i = 0; i < score_sz; i++) sc[i] = (int64_t)buf32[i];
            free(buf32);
        } else {
            size_t nr = fread(sc, sizeof(int64_t), score_sz, f);
            if (nr != score_sz) {
                snprintf(res.msg, sizeof(res.msg), "Short scores (member %u: read %zu of %zu)", m, nr, score_sz);
                free(sc); fclose(f); return res;
            }
        }
        for (size_t i = 0; i < score_sz; i++) {
            if (sc[i] != 0) all_zero = 0;
            if (sc[i] < vmin) vmin = sc[i];
            if (sc[i] > vmax) vmax = sc[i];
            /* Outlier: |value| > 2^40 ≈ 1e12 (reasonable for int64 accumulated scores) */
            if (sc[i] > ((int64_t)1 << 40) || sc[i] < -((int64_t)1 << 40)) has_outliers = 1;
        }
        free(sc);
    }
    /* Skip remaining members (not read) */
    if (n_members > 5)
        fseek(f, (long)(n_members - 5) * (long)score_sz * 8, SEEK_CUR);

    if (all_zero) {
        snprintf(res.msg + strlen(res.msg), sizeof(res.msg) - strlen(res.msg),
                 "%sall scores zero", res.msg[0] ? "; " : "");
        if (res.ok <= 0) res.ok = 1;
    }
    if (has_outliers) {
        snprintf(res.msg + strlen(res.msg), sizeof(res.msg) - strlen(res.msg),
                 "%sscore outliers >2^40", res.msg[0] ? "; " : "");
        if (res.ok <= 0) res.ok = 1;
    }

    /* ── Read labels ── */
    if ((off_t)ftell(f) + (off_t)n_test <= file_size) {
        uint8_t *labels = (uint8_t *)malloc(n_test);
        if (labels) {
            if (fread(labels, 1, n_test, f) == n_test) {
                int bad_label = 0;
                for (uint32_t i = 0; i < n_test; i++)
                    if (labels[i] >= n_classes) { bad_label = 1; break; }
                if (bad_label) {
                    snprintf(res.msg + strlen(res.msg), sizeof(res.msg) - strlen(res.msg),
                             "%slabels out of range", res.msg[0] ? "; " : "");
                    if (res.ok <= 0) res.ok = 1;
                }
            }
            free(labels);
        }
    }

    fclose(f);

    /* Build final status message */
    char dims[256];
    snprintf(dims, sizeof(dims), "v%u  %u×%u  H=%u  EP=%u  VN=%u  HN=%u  %u member%s  EVL=%.1f%%  W0=0x%08X  scores=[%+" PRId64 ",%+" PRId64 "]",
             version, n_test, n_classes, hidden, epochs, split_vn, split_hn,
             n_members, n_members == 1 ? "" : "s", ensemble_eval, w0_check, vmin, vmax);
    if (res.msg[0]) {
        char tmp[512];
        snprintf(tmp, sizeof(tmp), "%s  [%s]", dims, res.msg);
        strncpy(res.msg, tmp, sizeof(res.msg) - 1);
        res.msg[sizeof(res.msg) - 1] = '\0';
    } else {
        strncpy(res.msg, dims, sizeof(res.msg) - 1);
        res.msg[sizeof(res.msg) - 1] = '\0';
    }
    return res;
}

/* ── Check all .ens files in directory ── */
static int check_directory(const char *dir) {
    DIR *d = opendir(dir);
    if (!d) { fprintf(stderr, "[ERROR] Cannot open %s\n", dir); return 1; }

    int n_ok = 0, n_warn = 0, n_err = 0, n_files = 0;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        const char *ext = strrchr(de->d_name, '.');
        if (!ext || strcmp(ext, ".ens") != 0) continue;

        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dir, de->d_name);
        CheckResult r = check_archive(path);

        const char *tag;
        if (r.ok < 0)      { tag = "ERROR";  n_err++; }
        else if (r.ok > 0) { tag = "WARN ";  n_warn++; }
        else               { tag = "OK   ";  n_ok++; }
        printf("  %s  %-40s  %s\n", tag, de->d_name, r.msg);
        n_files++;
    }
    closedir(d);

    printf("\n══╡ CHECK SUMMARY ╞══════════════════════════════════════════════\n");
    printf("  %d .ens file(s):  %d OK,  %d WARN,  %d ERROR\n",
           n_files, n_ok, n_warn, n_err);
    return n_err > 0 ? 1 : 0;
}


int main(int argc, char **argv) {
    /* OMP: alle verfügbaren Kerne nutzen */
    omp_set_num_threads(omp_get_num_procs());
    if (getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(atoi(getenv("OMP_NUM_THREADS")));
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
    int optimize = 0;
    int beam_width = 0;
    int check_mode = 0;

    /* Parse all options in any order; DIR = first bare token */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--optimize") == 0) {
            optimize = 1;
        } else if (strcmp(argv[i], "--beam") == 0 && i + 1 < argc) {
            beam_width = atoi(argv[++i]);
            if (beam_width < 1) beam_width = 1;
            optimize = 1;  /* beam implies optimize mode */
        } else if (strcmp(argv[i], "--tries") == 0 && i + 1 < argc) {
            g_tries = atoi(argv[++i]);
            if (g_tries < 1) g_tries = 1;
        } else if (strcmp(argv[i], "--max") == 0) {
            /* --max allein = gain threshold aus; --max N = max member count */
            if (i + 1 < argc && argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9') {
                g_max_members = atoi(argv[++i]);
                if (g_max_members < 1) g_max_members = 1;
            } else {
                g_max_mode = 1;
            }
        } else if (strcmp(argv[i], "--min-gain") == 0 && i + 1 < argc) {
            g_min_gain = (float)atof(argv[++i]);
            if (g_min_gain < 0.0f) g_min_gain = 0.0f;
        } else if (strcmp(argv[i], "--debug") == 0) {
            g_debug = 1;
        } else if (strcmp(argv[i], "--diversity") == 0 && i + 1 < argc) {
            g_diversity = atoi(argv[++i]);
            if (g_diversity < 1) g_diversity = 1;
        } else if (strcmp(argv[i], "--seed-sort") == 0) {
            g_seed_sort = 1;
        } else if (strcmp(argv[i], "--member-seed") == 0 && i + 1 < argc) {
            strncpy(g_member_seed_spec, argv[++i], sizeof(g_member_seed_spec) - 1);
            g_member_seed_spec[sizeof(g_member_seed_spec) - 1] = '\0';
        } else if (strcmp(argv[i], "--member-out") == 0 && i + 1 < argc) {
            strncpy(g_member_out, argv[++i], sizeof(g_member_out) - 1);
            g_member_out[sizeof(g_member_out) - 1] = '\0';
        } else if (strcmp(argv[i], "--xform") == 0 && i + 1 < argc) {
            strncpy(g_xform_spec, argv[++i], sizeof(g_xform_spec) - 1);
            g_xform_spec[sizeof(g_xform_spec) - 1] = '\0';
        } else if (strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
            save_path = argv[++i];
        } else if (strcmp(argv[i], "--num") == 0 && i + 1 < argc) {
            fprintf(stderr, "[WARN] --num is obsolete (ignored)\n"); i++;
        } else if (strcmp(argv[i], "--expansion-sort") == 0 && i + 1 < argc) {
            const char *_mode = argv[++i];
            if (strcmp(_mode, "marg") == 0 || strcmp(_mode, "marginal") == 0) {
                g_expansion_sort = 1;
            } else if (strcmp(_mode, "abs") == 0) {
                g_expansion_sort = 0;
            } else {
                fprintf(stderr, "[ERROR] --expansion-sort: expected 'abs' or 'marg', got '%s'\n", _mode);
                return 1;
            }
        } else if (strcmp(argv[i], "--check") == 0) {
            check_mode = 1;
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

    /* Check or load */
    if (check_mode) {
        return check_directory(dir);
    }

    /* Load score archives */
    int nf = load_directory(dir);
    if (nf == 0) {
        fprintf(stderr, "[ERROR] No .ens files found in %s\n", dir);
        return 1;
    }
    printf("Loaded %d archive files (%d total score blocks)\n", nf, n_blocks);
    dedup_blocks();
    printf("  Expansion sort: %s\n", g_expansion_sort ? "marginal" : "abs");
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

    /* ── Debug: block CRC table after dedup+sort ── */
    if (g_debug) {
        printf("\n── Block CRC table (%d blocks) ──\n", n_blocks);
        for (int i = 0; i < n_blocks && i < 2048; i++) {
            char _ml[64];
            const char *_xn = (blocks[i].xform_real >= 0)
                              ? xform_name_safe(blocks[i].xform_real)
                              : xform_group_label(blocks[i].xform_id);
            member_label(&blocks[i], _ml, sizeof(_ml));
            size_t _sz = (size_t)g_n_test * (size_t)g_n_classes;
            uint64_t _crc = score_crc64(blocks[i].scores, _sz);
            printf("  [%3d] CRC=0x%016" PRIx64 "  %s:%s  (file_eval=%.2f%%)\n",
                   i, _crc, _xn, _ml, blocks[i].file_eval);
        }
        printf("── End CRC table ──\n\n");
        fflush(stdout);
    }

    /* Sort blocks deterministically before beam/optimize/eval.
     * Without this, the readdir() load order propagates through
     * dedup and into the search, causing non-deterministic results. */
    qsort(blocks, (size_t)n_blocks, sizeof(ScoreBlock), cmp_by_ctime);

    /* ── Debug: block order after sort ── */
    if (g_debug) {
        printf("── Sorted block order (%d blocks) ──\n", n_blocks);
        for (int i = 0; i < n_blocks && i < 2048; i++) {
            char _ml[64];
            const char *_xn = (blocks[i].xform_real >= 0)
                              ? xform_name_safe(blocks[i].xform_real)
                              : xform_group_label(blocks[i].xform_id);
            member_label(&blocks[i], _ml, sizeof(_ml));
            uint64_t _crc = score_crc64(blocks[i].scores, (size_t)g_n_test * (size_t)g_n_classes);
            printf("  [%3d] CRC=0x%016" PRIx64 "  %-30s  ft=%lld\n",
                   i, _crc, _xn, (long long)blocks[i].file_time);
        }
        printf("── End sorted order ──\n\n");
        fflush(stdout);
    }

    /* ── --seed-sort: single-member eval table ── */
    if (g_seed_sort) {
        typedef struct { int idx; float eval; SCORE_TYPE margin; } _SeedSc;
        _SeedSc *_ss2 = (_SeedSc *)malloc((size_t)n_blocks * sizeof(_SeedSc));
        size_t _rsz = (size_t)g_n_classes;
        if (_ss2 && g_labels) {
            for (int _mi = 0; _mi < n_blocks; _mi++) {
                const SCORE_TYPE *_sc = blocks[_mi].scores;
                int _ok = 0; SCORE_TYPE _mg = 0;
                for (int _s = 0; _s < g_n_test; _s++) {
                    const SCORE_TYPE *_row = _sc + (size_t)_s * _rsz;
                    int _bk = 0, _sk = -1;
                    for (int _k = 1; _k < g_n_classes; _k++) {
                        if (_row[_k] > _row[_bk]) { _sk = _bk; _bk = _k; }
                        else if (_sk < 0 || _row[_k] > _row[_sk]) _sk = _k;
                    }
                    if (_bk == (int)g_labels[_s]) _ok++;
                    if (_sk >= 0) _mg += _row[_bk] - _row[_sk];
                }
                _ss2[_mi].idx = _mi;
                _ss2[_mi].eval = (float)_ok * 100.0f / (float)g_n_test;
                _ss2[_mi].margin = _mg;
            }
            /* Sort by eval desc, margin desc (threshold-based tiebreak to avoid -Wfloat-equal) */
            for (int _i = 0; _i < n_blocks - 1; _i++)
                for (int _j = _i + 1; _j < n_blocks; _j++)
                    if (_ss2[_j].eval > _ss2[_i].eval + 0.0001f ||
                        (fabsf(_ss2[_j].eval - _ss2[_i].eval) <= 0.0001f && _ss2[_j].margin > _ss2[_i].margin)) {
                        _SeedSc _t = _ss2[_i]; _ss2[_i] = _ss2[_j]; _ss2[_j] = _t;
                    }
            printf("\n── SEED SORT (all %d singles by strength) ──\n", n_blocks);
            printf("  %-4s  %-7s  %15s  %s\n", "rank", "acc[%]", "block", "member");
            printf("  %-4s  %-7s  %15s  %s\n", "----", "-------", "-------------", "-------------------");
            for (int _i = 0; _i < n_blocks && _i < 2048; _i++) {
                int _mi = _ss2[_i].idx;
                char _ml[64];
                const char *_xn = (blocks[_mi].xform_real >= 0)
                    ? xform_name_safe(blocks[_mi].xform_real)
                    : xform_group_label(blocks[_mi].xform_id);
                member_label(&blocks[_mi], _ml, sizeof(_ml));
                printf("  %-4d  %-7.2f  %13d  %s:%s\n",
                       _i + 1, _ss2[_i].eval, _mi, _xn, _ml);
            }
            printf("── End seed sort ──\n\n");
        }
        free(_ss2);
        return 0;  /* seed-sort is inspection-only, no search */
    }

    if (beam_width > 0) {
        /* Resolve --member-seed if specified */
        int member_seed_idx = -1;
        if (g_member_seed_spec[0]) {
            member_seed_idx = find_member_by_spec(g_member_seed_spec);
            if (member_seed_idx < 0)
                fprintf(stderr, "  [WARN] --member-seed '%s' not found among %d blocks\n",
                        g_member_seed_spec, n_blocks);
            else {
                /* Compute actual single-member eval (file_eval may be wrong —
                 * it's the source archive's ensemble eval, not this member's) */
                float _seed_eval = 0.0f;
                if (g_labels) {
                    int _ok = 0;
                    size_t _rsz = (size_t)g_n_classes;
                    const SCORE_TYPE *_sc = blocks[member_seed_idx].scores;
                    for (int _s = 0; _s < g_n_test; _s++) {
                        const SCORE_TYPE *_row = _sc + (size_t)_s * _rsz;
                        int _bk = 0;
                        for (int _k = 1; _k < g_n_classes; _k++)
                            if (_row[_k] > _row[_bk]) _bk = _k;
                        if (_bk == (int)g_labels[_s]) _ok++;
                    }
                    _seed_eval = (float)_ok * 100.0f / (float)g_n_test;
                }
                printf("  MemberSeed: %s  (block %d, %.2f%%)\n",
                       g_member_seed_spec, member_seed_idx, _seed_eval);
            }
        }
        if (g_tries <= 1) {
            merge_and_beam(save_path, beam_width, member_seed_idx, NULL);
            if (g_member_out[0])
                write_member_file_from_best(g_member_out, beam_width);
        } else {
            /* Multi-try: jeder Try sucht den stärksten noch freien Member als Seed */
            char saved_member_out[1024];
            strncpy(saved_member_out, g_member_out, sizeof(saved_member_out));
            g_member_out[0] = '\0';

            int _n_tries = g_tries < n_blocks ? g_tries : n_blocks;
            if (_n_tries < 1) _n_tries = 1;
            printf("\n  Multi-try: %d tries\n", _n_tries);
            size_t _row_sz = (size_t)g_n_classes;
            {
                uint8_t *pool_exclude = (uint8_t *)calloc((size_t)n_blocks, 1);
                int n_active_tries = 0;
                float _global_best_eval = -1.0f;
                int   _global_best_n = 0;
                uint8_t *_global_best_used = NULL;

                /* If --member-seed specified, force it as first try's seed */
                int _seed_idx = -1;
                if (member_seed_idx >= 0) {
                    if (pool_exclude[member_seed_idx])
                        fprintf(stderr, "  [WARN] --member-seed %d is already excluded\n", member_seed_idx);
                    _seed_idx = member_seed_idx;
                }

                for (int ti = 0; ti < _n_tries; ti++) {
                    /* Seed finden: stärksten Member aus dem Pool der Noch-Nicht-Genutzten */
                    if (ti > 0 || member_seed_idx < 0) {
                        _seed_idx = -1;
                        float _best_seed_eval = -1.0f;
                        for (int mi = 0; mi < n_blocks; mi++) {
                            if (pool_exclude[mi]) continue;
                            const SCORE_TYPE *sc = blocks[mi].scores;
                            int _c = 0;
                            for (int s = 0; s < g_n_test; s++) {
                                const SCORE_TYPE *_r = sc + (size_t)s * _row_sz;
                                int pred = 0;
                                for (int k = 1; k < g_n_classes; k++)
                                    if (_r[k] > _r[pred]) pred = k;
                                if (pred == (int)g_labels[s]) _c++;
                            }
                            float _eval = (float)_c * 100.0f / (float)g_n_test;
                            if (_eval > _best_seed_eval) {
                                _best_seed_eval = _eval;
                                _seed_idx = mi;
                            }
                        }
                        if (_seed_idx < 0) break;
                    }

                    n_active_tries++;

                    /* Seed eval for display */
                    float _seed_eval = 0.0f;
                    {
                        const SCORE_TYPE *sc = blocks[_seed_idx].scores;
                        int _c = 0;
                        for (int s = 0; s < g_n_test; s++) {
                            const SCORE_TYPE *_r = sc + (size_t)s * _row_sz;
                            int pred = 0;
                            for (int k = 1; k < g_n_classes; k++)
                                if (_r[k] > _r[pred]) pred = k;
                            if (pred == (int)g_labels[s]) _c++;
                        }
                        _seed_eval = (float)_c * 100.0f / (float)g_n_test;
                    }
                    printf("\n══╡ TRY %d/%d ══ seed=%d (%.2f%%) ╞═══════════════════════════════\n",
                           ti + 1, _n_tries, _seed_idx, _seed_eval);

                    /* g_best_used zurücksetzen — merge_and_beam füllt es neu für diesen Try */
                    free(g_best_used); g_best_used = NULL;
                    g_best_eval = -1.0f;
                    merge_and_beam(NULL, beam_width, _seed_idx, pool_exclude);

                    /* Globalen Best über alle Trys aktualisieren */
                    if (g_best_eval > _global_best_eval && g_best_used) {
                        _global_best_eval = g_best_eval;
                        _global_best_n = g_best_n;
                        free(_global_best_used);
                        _global_best_used = (uint8_t *)malloc((size_t)n_blocks);
                        memcpy(_global_best_used, g_best_used, (size_t)n_blocks);
                    }

                    /* Member dieses Trys für alle folgenden Trys sperren:
                     * - immer den Seed selbst (auch wenn er nicht im Beam-Ergebnis landete)
                     * - alle Mitglieder aus g_best_used */
                    pool_exclude[_seed_idx] = 1;
                    if (g_best_used) {
                        for (int i = 0; i < n_blocks; i++)
                            if (g_best_used[i]) pool_exclude[i] = 1;
                    }
                }
                free(pool_exclude);

                /* g_best_used = global best (für --member-out + write_member_file_from_best) */
                free(g_best_used); g_best_used = _global_best_used;
                g_best_eval = _global_best_eval;
                g_best_n    = _global_best_n;

                /* Output global best */
                strncpy(g_member_out, saved_member_out, sizeof(g_member_out));
                int _corr = (int)(g_best_eval * (float)g_n_test / 100.0f + 0.5f);
                printf("\n══╡ GLOBAL BEST ╞════════════════════════════════════════\n");
                printf("  Best:  %d members  acc=%.2f%%  (%d/%d)  (best of %d tries)\n",
                       g_best_n, g_best_eval, _corr, g_n_test, n_active_tries);
                printf("  Report: H=%d  EP=%d  VN=%d  HN=%d  TE=%d  eval=%.2f%%  err=%d  N=%d  best_en=%d  beam=%d  tries=%d\n",
                       g_hidden, g_epochs, g_split_vn, g_split_hn, g_target_err_x100,
                       g_best_eval, g_n_test - _corr, g_n_test, g_best_n, beam_width, n_active_tries);
                if (g_member_out[0])
                    write_member_file_from_best(g_member_out, beam_width);
            }

            free(g_best_used); g_best_used = NULL; free(g_best_blocks); g_best_blocks = NULL;
            g_best_eval = -1.0f;
        }
    } else if (optimize) {
        merge_and_optimize(save_path);
    } else {
        merge_and_eval(save_path, 0);
    }
    for (int i = 0; i < n_blocks; i++) free(blocks[i].scores);
    free(blocks); free(g_labels);
    return 0;
}
