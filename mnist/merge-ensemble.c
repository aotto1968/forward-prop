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
 *   Data:   n_members . scores[n_test . n_classes] + uint8[n_test] labels
 *   Score type by version: v1-7 int64, v8 int32, v9-11 float,
 *                          v12 double, v13 int64 (2026-08-06: export follows
 *                          the internal SCORE_TYPE — no precision loss).
 *   v14 int64+precision, v15 double+precision (2026-08-12): the header adds
 *   ot_precision(4) bit_width(4) counter_type[24] — how the stored logits
 *   were computed, so --check can verify archives "fit together".
 *
 * Options:
 *   DIR         Directory containing .ens score archives
 *   --max N     Stop greedy/beam at N members (default: unlimited)
 *   --save FILE Save cumulative accuracy to FILE (default: DIR/merge.dat)
 *   --target N[,M,...]  Optimize the ensemble for the given class(es) only
 *                       (recall on target-class samples, 0-100%).
 *   --filter-sample N[,M,...]  Restrict the EVALUATION set to samples whose
 *                       label is in the list (sample-level filter).
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
#include <regex.h>
#include <omp.h>
#include <carquet/carquet.h> /* --sample-index: Parquet writer (2026-08-15,
                               replaces SQLite — see DESIGN comment below) */
/* Define ki_Args aa in this file (make KI_ARGS_EXTERN empty) */
#define KI_ARGS_EXTERN
#include "ki-config.h"
/* libtprint BEFORE ki-common.h: print_confusion_debug() in ki-common.h uses
 * TPrint for the confusion table (2026-08-12). */
#include "../lib/ki-encoding.h"
#include "../lib/ki-ens.h"   /* .ens format: reader (all version logic central) */
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

/* ── FNV-1a 64-bit — stable member ID for --sample-index (2026-08-14) ──
 * The member_id in the sample-index Parquet files is FNV-1a over the .ens
 * basename (the spec, e.g. "colswap-1-4@rot112:lbp:tri8"). Deterministic
 * from the name → the same member gets the same ID across corpus changes,
 * regardless of n_blocks order. Collision probability for ~25k names:
 * negligible (< 1 in 2^64/25k — practically zero); on collision the caller
 * appends a "_2" suffix and warns. */
static uint64_t member_id_fnv1a(const char *name) {
    uint64_t h = 1469598103934665603ULL;   /* FNV offset basis */
    for (const unsigned char *p = (const unsigned char *)name; *p; p++) {
        h ^= (uint64_t)*p;
        h *= 1099511628211ULL;             /* FNV prime */
    }
    return h;
}

#ifndef KI_NCLASSES
#define KI_NCLASSES 10
#endif

/* -- Archive header (matches mlp-bin32-otto-trn.c) ------------ */
/* Fields are read individually (size differs between versions) */
typedef struct {
    uint32_t magic;       /* 0x454E534D ('ENSM') */
    uint32_t version;     /* 1..15: 1-7 int64, 8 int32, 9-11 float, 12 double,
                           13 int64, 14 int64+precision, 15 double+precision */
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
    char     maj_token[8]; /* v11+: majority mode token ("1","1r","1p","1rp","3","7") */
    uint32_t maj1_thresh;  /* v11+: maj1 threshold (-2=auto, -1=n/2, >=0 exact) */
    /* v14+ precision block (2026-08-12) */
    int32_t  ot_precision; /* F = 1<<OT_PRECISION */
    int32_t  bit_width;    /* KI_BIT_WIDTH (8/16/32) */
    char     counter_type[24]; /* "int32_t" | "float" */
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
    char     maj_token[8];  /* v11+: majority mode token ("" for older archives) */
    int      maj1_thresh;   /* v11+: maj1 threshold (-999 if absent) */
    SCORE_TYPE  score_min;  /* minimum score value */
    SCORE_TYPE  score_max;  /* maximum score value */
    double      clarity;    /* mean per-sample std-dev of the K class scores —
                               "decision clarity": how distinctly this member
                               separates the classes (offset-invariant). High =
                               strong, unambiguous votes; low = flat scores that
                               can flip other members' decisions without being
                               confident themselves (plan-2026-08-08). */
    int      err;           /* eval errors (g_n_test - correct_count) */
    char     source_file[128]; /* basename of .ens archive */
} ScoreBlock;

/* Tentative declarations of the block-array globals — used by blocks_from_index
 * which sits before their real definitions (with initializers). Legal C: these
 * tentative definitions are completed by the later definitions. */
static int    n_blocks;
static int    n_blocks_cap;
static ScoreBlock *blocks;
static int member_is_filtered(const ScoreBlock *b);
static int    g_n_test;
static int    g_n_classes;
static int    g_hidden;
static int    g_epochs;
static int    g_split_vn;
static int    g_split_hn;
static int    g_target_err_x100;
/* Precision-consistency across archives (v14+, 2026-08-12) — tentative
 * decls (blocks_from_index sits before the definitions with initializers). */
static int    g_ens_otp;
static int    g_ens_bits;
static char   g_ens_ct[16];
/* W0-marker consistency: the FIRST non-placeholder archive decides whether
 * this directory holds Otto archives (w0_marker != 0) or Bit-Voting archives
 * (w0_marker == 0). Mixing both would corrupt the ensemble — mixing check
 * added 2026-08-02 (plans/plan-2026-08-02-ki-bitvoting-flag.md). */
static int    g_ens_w0_zero = -1;    /* -1 = unset, 0 = Otto, 1 = Bit-Voting */
static int    g_eval_active;
static int    g_eval_cmp;
static int    g_eval_pct;
static float  g_eval_thresh;
static char   g_ens_maj[8];
static int    g_ens_majt;
static int    g_meta_ok;
static int    g_meta_h;
static int    g_meta_ep;
static int    g_meta_vn;
static int    g_meta_hn;
static char   g_meta_maj[8];
static int    g_meta_majt;
static char   g_meta_ct[64];   /* .meta COUNTER_TYPE label ("" = absent) */
static char   g_meta_st[64];   /* .meta SCORE_TYPE label ("" = absent) */

/* ── .index cache entry (metadata only, no scores) ────────────
 * One IndexFile per .ens archive + one IndexMember per member — the per-member
 * metadata (channel/encoding/width/xform) differs WITHIN an archive, and all
 * filters, column extraction and the dedup/sort logic run on it. Stored as a
  * custom binary file "<dir>/.index" so that filtering/columns/--stdout never
  * have to open the .ens files themselves (100k+ files case). Validity is
  * checked by stat()ing every .ens file (ctime+size) — cheap; changed files
  * are re-read incrementally. No GDBM: access is purely sequential and a plain
  * file avoids an external dependency. */
typedef struct {
    int member_idx, color, enc_type, enc_width, xform_real;
    float eval;             /* REAL member accuracy % (argmax-vs-truth) */
    char xform_name[64];    /* v7+ xform string — pipe IDs are NOT process-stable
                             * (ki_xform_pipe_create registers them at runtime),
                             * so the NAME is cached and re-parsed on load. */
} IndexMember;

typedef struct {
    char     name[128];     /* basename of the .ens file (the spec) */
    int64_t  mtime, ctime, size; /* validity check against stat() (ctime!) */
    int64_t  file_time;     /* embedded timestamp (v3+) or mtime */
    float    file_eval;     /* header ensemble eval % (v4+, 0 for older) */
    int      version, seed;
    uint32_t w0_marker;
    char     maj_token[8];
    int      maj1_thresh;
    int      hidden, epochs, split_vn, split_hn, target_err_x100; /* .meta identity check */
    int      n_test, n_classes;   /* eval samples / classes (per archive) */
    int      n_members, n_xforms;
    /* v14+ precision (2026-08-12): 0/"" = absent (older archives) */
    int      ot_precision;
    int      bit_width;
    char     counter_type[24];
    IndexMember *members;   /* [n_members] */
} IndexFile;

#define KIDX_MAGIC   0x5844494Bu  /* "KIDX" */
#define KIDX_VERSION 3            /* v3: +ot_precision/bit_width/counter_type per file */

/* Binary layout of one IndexFile (see index_save): name[128], mtime i64,
 * ctime i64, size i64, file_time i64, file_eval f32, version i32, seed i32,
 * w0_marker u32, maj_token[8], maj1_thresh i32, hidden i32, epochs i32,
 * split_vn i32, split_hn i32, target_err_x100 i32, n_test i32, n_classes i32,
 * n_members i32, n_xforms i32,
 * v3: ot_precision i32, bit_width i32, counter_type[24]
 * → 128+8+8+8+8+4+4+4+4+8+4+4+4+4+4+4+4+4+4+4+4+4+24 = 256 B, followed by
 * n_members × (member_idx, color, enc_type, enc_width, xform_real, eval =
 * 6×i32 = 24 B + xform_name[64] = 88 B). */

/* Read the header + per-member metadata of ONE .ens file into an IndexFile —
 * the .index cache is filter-INDEPENDENT: this function never skips a file and
 * never touches global state. It also computes the REAL per-member eval
 * (argmax-vs-truth from scores+labels, or the header eval for n_members==1) —
 * the 1x work that makes --seed-sort + filters run purely off the index.
 * Unreadable/corrupt files yield a placeholder (version=0) so the index mirrors
  * every *.ens file. On success e->members is heap-allocated ([n_members]) —
  * caller frees. Always returns 1. */


static int read_ens_meta(const char *path, IndexFile *e) {
    struct stat st;
    memset(e, 0, sizeof(*e));
    const char *base = strrchr(path, '/'); base = base ? base + 1 : path;
    snprintf(e->name, sizeof(e->name), "%.*s", (int)sizeof(e->name) - 1, base);
    if (stat(path, &st) != 0) {
        /* Unstatable file — placeholder entry so the .index is a complete
         * mirror of *.ens (the incremental merge needs the count to match).
         * version=0 marks it; blocks_from_index skips it. */
        return 1;  /* placeholder: version=0, n_members=0 */
    }
    e->mtime = (int64_t)st.st_mtime;
    e->ctime = (int64_t)st.st_ctime;
    e->size = (int64_t)st.st_size;

    /* ── Header + metadata + scores via the shared reader (lib/ki-ens.h) ──
     * All version logic (v1-13: width, metadata layout, bounds) lives in the
     * library — a version missed in one path can no longer drift (bug
     * 2026-08-06: "0 blocks filled" SEGV). */
    EnsReader rd;
    if (ens_reader_open(&rd, path) != 0) return 1;   /* bad magic/version → placeholder */
    int version = (int)rd.version;
    e->version = version;
    e->seed = (int)rd.seed;
    e->hidden = (int)rd.hidden; e->epochs = (int)rd.epochs;
    e->split_vn = (int)rd.split_vn; e->split_hn = (int)rd.split_hn;
    e->target_err_x100 = (int)(rd.target_err * 100.0f + 0.5f);
    e->n_test = (int)rd.n_test; e->n_classes = (int)rd.n_classes;
    e->file_time = rd.file_time ? rd.file_time : (int64_t)st.st_mtime;
    e->file_eval = rd.ensemble_eval;
    e->w0_marker = rd.w0_marker;
    snprintf(e->maj_token, sizeof(e->maj_token), "%s", rd.maj_token);
    e->maj1_thresh = rd.maj1_thresh;   /* -999 for v<11 (reader default) */
    e->n_xforms = rd.n_xforms;
    e->n_members = (int)rd.n_members;
    /* v14+ precision block (2026-08-12) */
    e->ot_precision = (int)rd.ot_precision;
    e->bit_width    = (int)rd.bit_width;
    snprintf(e->counter_type, sizeof(e->counter_type), "%s", rd.counter_type);
    if (rd.n_members < 1 || rd.n_members > 100000) { ens_reader_close(&rd); return 1; }
    e->members = (IndexMember *)calloc((size_t)rd.n_members, sizeof(IndexMember));
    if (!e->members) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }

    /* per-member metadata (v7+: 4 length-prefixed strings; v2-6: 4 bytes) */
    if (version >= 7) {
        char fields[4][128]; uint8_t raw[4] = {0};
        for (uint32_t m = 0; m < rd.n_members; m++) {
            if (ens_reader_read_member_meta(&rd, fields, raw) != 0) { ens_reader_close(&rd); return 1; }
            e->members[m].member_idx = (int)m;
            e->members[m].color = color_id_by_name(fields[0]);
            e->members[m].enc_type = enc_id_by_name(fields[1]);
            e->members[m].enc_width = atoi(fields[2]);
            /* xform: cache the NAME (pipe IDs are runtime-registered,
             * not process-stable — see IndexMember.xform_name) */
            snprintf(e->members[m].xform_name,
                     sizeof(e->members[m].xform_name), "%.63s", fields[3]);
            e->members[m].xform_real = xform_id_by_name(fields[3]);
        }
    } else if (version >= 2) {
        /* INTENTIONAL: raw[] init {0} silences GCC-15 -Wmaybe-uninitialized.
         * ens_reader_read_member_meta() guards writes with `if (raw)`, which
         * the compiler cannot correlate with the non-NULL arg here (false
         * positive under -Werror; values are always overwritten on success). */
        char fields[4][128]; uint8_t raw[4] = {0};
        for (uint32_t m = 0; m < rd.n_members; m++) {
            if (ens_reader_read_member_meta(&rd, fields, raw) != 0) { ens_reader_close(&rd); return 1; }
            e->members[m].member_idx = (int)m;
            e->members[m].color = (int)raw[0];
            e->members[m].enc_type = (int)raw[1];
            e->members[m].enc_width = (int)raw[2];
            e->members[m].xform_real = -1;
            if (version == 5) {
                int xi = (int)raw[3];
                if (xi >= 0 && xi < rd.n_xforms) e->members[m].xform_real = (int)rd.xform_ids[xi];
            } else if (version == 6) {
                e->members[m].xform_real = (int)raw[3];   /* v6: 4th byte = xform_id */
            }
        }
    } else {  /* v1: no metadata — one member with unknown channel/enc */
        e->members[0].member_idx = 0;
        e->members[0].color = -1; e->members[0].enc_type = -1;
        e->members[0].enc_width = 0; e->members[0].xform_real = -1;
    }

    /* ── REAL member evals ──
     * n_members==1: header file_eval IS the member eval (no score read).
     * n_members>1: read all member scores + ground-truth labels (appended by
     * the trainer), compute argmax-vs-truth accuracy per member. This 1x work
     * happens only for actually-read files (incremental rebuild) and makes
     * --seed-sort + filters work purely off the index. */
    if (rd.n_members == 1) {
        e->members[0].eval = (version >= 4) ? rd.ensemble_eval : 0.0f;
    } else {
        float *member_sc = (float *)malloc(rd.score_sz * sizeof(float));
        uint8_t *preds = (uint8_t *)malloc((size_t)rd.n_test * (size_t)rd.n_members);
        uint8_t *labels = (uint8_t *)malloc((size_t)rd.n_test);
        if (!member_sc || !preds || !labels) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
        int ok = 1;
        for (uint32_t m = 0; m < rd.n_members && ok; m++) {
            if (ens_reader_read_scores_float(&rd, member_sc) != 0) { ok = 0; break; }
            uint8_t *pm = preds + (size_t)m * (size_t)rd.n_test;
            for (uint32_t s = 0; s < rd.n_test; s++) {
                const float *row = member_sc + (size_t)s * (size_t)rd.n_classes;
                int best = 0;
                for (uint32_t k = 1; k < rd.n_classes; k++)
                    if (row[k] > row[best]) best = (int)k;
                pm[s] = (uint8_t)best;
            }
        }
        if (ok && fread(labels, 1, (size_t)rd.n_test, rd.f) == (size_t)rd.n_test) {
            for (uint32_t m = 0; m < rd.n_members; m++) {
                const uint8_t *pm = preds + (size_t)m * (size_t)rd.n_test;
                int correct = 0;
                for (uint32_t s = 0; s < rd.n_test; s++)
                    if (pm[s] == labels[s]) correct++;
                e->members[m].eval = 100.0f * (float)correct / (float)rd.n_test;
            }
        } else {
            /* scores/labels unreadable (corrupt file) → fall back to file_eval */
            for (uint32_t m = 0; m < rd.n_members; m++)
                e->members[m].eval = (version >= 4) ? rd.ensemble_eval : 0.0f;
        }
        free(member_sc); free(preds); free(labels);
    }
    ens_reader_close(&rd);
    return 1;
}


/* ── .index save/load ────────────────────────────────────────
 * Header: magic u32, version u32, n_files u32, then directory identity
 * (n_test, n_classes, H, EP, VN, HN, TE_x100 u32 + maj_token[8] + maj1_thresh
 * i32) so a stale index from a different config is rejected. */
static int index_save(const char *dir, const IndexFile *files, int n_files) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/.index", dir);
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "  [WARN] Cannot write %s: %s\n", path, strerror(errno)); return -1; }
    uint32_t magic = KIDX_MAGIC, ver = KIDX_VERSION, nf = (uint32_t)n_files;
    fwrite(&magic, 4, 1, f); fwrite(&ver, 4, 1, f); fwrite(&nf, 4, 1, f);
    /* Directory identity comes from the FILES (first non-placeholder archive),
     * NOT from the globals — in the full-build path the globals are still 0
     * when index_save runs (blocks_from_index sets them afterwards). Storing
     * the globals would write a zero header → .meta mismatch on next load. */
    uint32_t id[7] = {0, 0, 0, 0, 0, 0, 0};
    char maj[8] = {0};
    int32_t majt = -999;
    for (int i = 0; i < n_files; i++) {
        if (files[i].version < 1) continue;
        if (id[0] == 0) {
            id[0] = (uint32_t)files[i].n_test;
            id[1] = (uint32_t)files[i].n_classes;
            id[2] = (uint32_t)files[i].hidden;
            id[3] = (uint32_t)files[i].epochs;
            id[4] = (uint32_t)files[i].split_vn;
            id[5] = (uint32_t)files[i].split_hn;
            id[6] = (uint32_t)files[i].target_err_x100;
        }
        if (files[i].version >= 11 && maj[0] == '\0') {
            snprintf(maj, sizeof(maj), "%s", files[i].maj_token);
            majt = (int32_t)files[i].maj1_thresh;
        }
    }
    fwrite(id, 4, 7, f);
    fwrite(maj, 1, 8, f);
    fwrite(&majt, 4, 1, f);
    for (int i = 0; i < n_files; i++) {
        const IndexFile *e = &files[i];
        fwrite(e->name, 1, 128, f);
        fwrite(&e->mtime, 8, 1, f); fwrite(&e->ctime, 8, 1, f);
        fwrite(&e->size, 8, 1, f);
        fwrite(&e->file_time, 8, 1, f);
        fwrite(&e->file_eval, 4, 1, f);
        int32_t v = e->version, s = e->seed;
        fwrite(&v, 4, 1, f); fwrite(&s, 4, 1, f);
        fwrite(&e->w0_marker, 4, 1, f);
        fwrite(e->maj_token, 1, 8, f);
        int32_t mt = e->maj1_thresh,
                h = e->hidden, ep = e->epochs, vn = e->split_vn, hn = e->split_hn,
                te = e->target_err_x100,
                nt = e->n_test, nc = e->n_classes,
                nm = e->n_members, nx = e->n_xforms,
                otp = e->ot_precision, bw = e->bit_width;
        fwrite(&mt, 4, 1, f); fwrite(&h, 4, 1, f); fwrite(&ep, 4, 1, f);
        fwrite(&vn, 4, 1, f); fwrite(&hn, 4, 1, f); fwrite(&te, 4, 1, f);
        fwrite(&nt, 4, 1, f); fwrite(&nc, 4, 1, f);
        fwrite(&nm, 4, 1, f); fwrite(&nx, 4, 1, f);
        fwrite(&otp, 4, 1, f); fwrite(&bw, 4, 1, f);
        fwrite(e->counter_type, 1, 24, f);
        for (int m = 0; m < e->n_members; m++) {
            int32_t mm = e->members[m].member_idx, c = e->members[m].color,
                    t = e->members[m].enc_type, w = e->members[m].enc_width,
                    x = e->members[m].xform_real;
            float ev = e->members[m].eval;
            fwrite(&mm, 4, 1, f); fwrite(&c, 4, 1, f); fwrite(&t, 4, 1, f);
            fwrite(&w, 4, 1, f); fwrite(&x, 4, 1, f); fwrite(&ev, 4, 1, f);
            fwrite(e->members[m].xform_name, 1, 64, f);
        }
    }
    fclose(f);
    return 0;
}

/* Load .index; returns n_files (>=0) or -1 on error / -2 on mismatch.
 * On success the global config (g_n_test, g_n_classes, H/EP/VN/HN/TE, maj)
 * is adopted from the index header — in the index path read_ens_meta is NOT
 * called (the whole point), so these globals must come from here. Identity is
 * verified against .meta (H/EP/VN/HN + MAJ/MAJ1_THRESH) when present. */
static int index_load(const char *dir, IndexFile **files_out) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/.index", dir);
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    uint32_t magic, ver, nf;
    if (fread(&magic, 4, 1, f) != 1 || fread(&ver, 4, 1, f) != 1 ||
        fread(&nf, 4, 1, f) != 1) { fclose(f); return -1; }
    if (magic != KIDX_MAGIC || ver != KIDX_VERSION || nf > 10000000) { fclose(f); return -1; }
    uint32_t id[7];
    char maj[8] = {0}; int32_t majt = -999;
    if (fread(id, 4, 7, f) != 7 || fread(maj, 1, 8, f) != 8 ||
        fread(&majt, 4, 1, f) != 1) { fclose(f); return -1; }
    /* .meta identity (when present): the index header must match it */
    if (g_meta_ok) {
        if ((int)id[2] != g_meta_h || (int)id[3] != g_meta_ep ||
            (int)id[4] != g_meta_vn || (int)id[5] != g_meta_hn) {
            fclose(f); return -2;   /* stale config */
        }
        /* maj="-1" = no-majority marker (Bit-Voting) — skip the checks. */
        if (maj[0] && strcmp(maj, "-1") != 0 && g_meta_maj[0] &&
            strcmp(g_meta_maj, "-1") != 0 && strcmp(maj, g_meta_maj) != 0) { fclose(f); return -2; }
        if (g_meta_majt != -999 && majt != -999 && strcmp(maj, "-1") != 0 &&
            majt != g_meta_majt) { fclose(f); return -2; }
    }
    /* Adopt global config from the index header */
    g_n_test = (int)id[0]; g_n_classes = (int)id[1];
    g_hidden = (int)id[2]; g_epochs = (int)id[3];
    g_split_vn = (int)id[4]; g_split_hn = (int)id[5];
    g_target_err_x100 = (int)id[6];
    if (maj[0] && g_ens_maj[0] == '\0') {
        snprintf(g_ens_maj, sizeof(g_ens_maj), "%s", maj);
        g_ens_majt = (int)majt;
    }
    IndexFile *files = (IndexFile *)calloc((size_t)nf, sizeof(IndexFile));
    if (!files) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
    for (uint32_t i = 0; i < nf; i++) {
        IndexFile *e = &files[i];
        if (fread(e->name, 1, 128, f) != 128 ||
            fread(&e->mtime, 8, 1, f) != 1 || fread(&e->ctime, 8, 1, f) != 1 ||
            fread(&e->size, 8, 1, f) != 1 ||
            fread(&e->file_time, 8, 1, f) != 1 ||
            fread(&e->file_eval, 4, 1, f) != 1) {
            fprintf(stderr, "  [WARN] %s: truncated index — rebuilding\n", path);
            for (uint32_t j = 0; j < i; j++) free(files[j].members);
            free(files); fclose(f); return -1;
        }
        int32_t v, s; uint32_t w0;
        if (fread(&v, 4, 1, f) != 1 || fread(&s, 4, 1, f) != 1 ||
            fread(&w0, 4, 1, f) != 1 || fread(e->maj_token, 1, 8, f) != 8) {
            fprintf(stderr, "  [WARN] %s: truncated index — rebuilding\n", path);
            for (uint32_t j = 0; j < i; j++) free(files[j].members);
            free(files); fclose(f); return -1;
        }
        int32_t mt, h, ep, vn, hn, te, nt, nc, nm, nx, otp, bw;
        if (fread(&mt, 4, 1, f) != 1 || fread(&h, 4, 1, f) != 1 ||
            fread(&ep, 4, 1, f) != 1 || fread(&vn, 4, 1, f) != 1 ||
            fread(&hn, 4, 1, f) != 1 || fread(&te, 4, 1, f) != 1 ||
            fread(&nt, 4, 1, f) != 1 || fread(&nc, 4, 1, f) != 1 ||
            fread(&nm, 4, 1, f) != 1 || fread(&nx, 4, 1, f) != 1 ||
            fread(&otp, 4, 1, f) != 1 || fread(&bw, 4, 1, f) != 1 ||
            fread(e->counter_type, 1, 24, f) != 24) {
            fprintf(stderr, "  [WARN] %s: truncated index — rebuilding\n", path);
            for (uint32_t j = 0; j < i; j++) free(files[j].members);
            free(files); fclose(f); return -1;
        }
        e->version = (int)v; e->seed = (int)s; e->w0_marker = w0;
        e->maj1_thresh = (int)mt; e->n_members = (int)nm; e->n_xforms = (int)nx;
        e->hidden = (int)h; e->epochs = (int)ep;
        e->split_vn = (int)vn; e->split_hn = (int)hn;
        e->target_err_x100 = (int)te;
        e->n_test = (int)nt; e->n_classes = (int)nc;
        e->ot_precision = (int)otp; e->bit_width = (int)bw;
        e->counter_type[23] = '\0';
        if (e->n_members < 0 || e->n_members > 100000) {
            for (uint32_t j = 0; j < i; j++) free(files[j].members);
            free(files); fclose(f); return -1;
        }
        if (e->n_members > 0) {
            e->members = (IndexMember *)calloc((size_t)e->n_members, sizeof(IndexMember));
            if (!e->members) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
        }
        for (int m = 0; m < e->n_members; m++) {
            int32_t mm, c, t, w, x; float ev;
            if (fread(&mm, 4, 1, f) != 1 || fread(&c, 4, 1, f) != 1 ||
                fread(&t, 4, 1, f) != 1 || fread(&w, 4, 1, f) != 1 ||
                fread(&x, 4, 1, f) != 1 || fread(&ev, 4, 1, f) != 1 ||
                fread(e->members[m].xform_name, 1, 64, f) != 64) {
                fprintf(stderr, "  [WARN] %s: truncated index — rebuilding\n", path);
                for (uint32_t j = 0; j < i + 1; j++) free(files[j].members);
                free(files); fclose(f); return -1;
            }
            e->members[m].member_idx = (int)mm; e->members[m].color = (int)c;
            e->members[m].enc_type = (int)t; e->members[m].enc_width = (int)w;
            e->members[m].xform_real = (int)x;
            e->members[m].eval = ev;
            e->members[m].xform_name[63] = '\0';
            /* Pipe xform IDs are runtime-registered → NOT process-stable.
             * Re-parse the cached NAME in this process so xform_name_safe()
             * prints the pipe name, not "pipe?". Non-pipe names resolve to the
             * same table ID (no-op). */
            if (e->members[m].xform_name[0]) {
                int _id = xform_id_by_name(e->members[m].xform_name);
                if (_id >= 0) e->members[m].xform_real = _id;
            }
        }
    }
    fclose(f);
    *files_out = files;
    return (int)nf;
}

/* stat()-based snapshot of the current .ens files (name, ctime, size) — the
 * incremental-rebuild decision criterion. ctime (not mtime) is used because it
 * also changes when a file is copied or its attributes change. O(n log n): one
 * stat per file, no file content is opened. Returns snapshot count (>=0). */
typedef struct {
    char name[128]; int64_t ctime, size;
} IndexSnap;

static int cmp_index_snap(const void *a, const void *b) {
    const IndexSnap *sa = (const IndexSnap *)a, *sb = (const IndexSnap *)b;
    return strcmp(sa->name, sb->name);
}

static int index_snapshot(const char *dir, IndexSnap **snap_out) {
    DIR *d = opendir(dir);
    if (!d) return -1;
    IndexSnap *snap = NULL;
    int n_snap = 0, cap = 0;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        const char *ext = strrchr(de->d_name, '.');
        if (!ext || strcmp(ext, ".ens") != 0) continue;
        if (n_snap >= cap) {
            cap = cap ? cap * 2 : 4096;
            snap = (IndexSnap *)realloc(snap, (size_t)cap * sizeof(*snap));
            if (!snap) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
        }
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dir, de->d_name);
        struct stat st;
        if (stat(path, &st) != 0) continue;
        snprintf(snap[n_snap].name, 128, "%.127s", de->d_name);
        snap[n_snap].ctime = (int64_t)st.st_ctime;
        snap[n_snap].size = (int64_t)st.st_size;
        n_snap++;
    }
    closedir(d);
    qsort(snap, (size_t)n_snap, sizeof(snap[0]), cmp_index_snap);
    *snap_out = snap;
    return n_snap;
}

/* Name comparator for IndexFile — the .index entries MUST be name-sorted so
 * that the load path and index_snapshot (name-sorted too) merge deterministically
 * regardless of readdir order. */
static int cmp_index_file(const void *a, const void *b) {
    const IndexFile *fa = (const IndexFile *)a, *fb = (const IndexFile *)b;
    return strcmp(fa->name, fb->name);
}


/* Build ScoreBlocks (metadata only, scores=NULL) from index files — the
 * index-backed replacement for load_archive's per-member loop. This is the
 * SINGLE filter point: the .index mirrors ALL *.ens files (including
 * placeholders and config-mismatched archives), and everything that would have
 * been skipped by load_archive is skipped HERE, from the cached fields, so the
 * result is identical on rebuild and on cache load. */
static void blocks_from_index(IndexFile *files, int n_files) {
    for (int i = 0; i < n_files; i++) {
        IndexFile *e = &files[i];
        /* Placeholder (unreadable archive): skip silently */
        if (e->version < 1) continue;
        /* maj consistency: the FIRST non-placeholder archive sets the expected
         * majority mode; later archives that differ are excluded (mixing e.g.
         * --maj 1 and --maj 3 members would corrupt the ensemble).
         * INTENTIONAL: maj_token == "-1" is the explicit "no majority" marker
         * written by Bit-Voting builds (identity W0, linear voter). Skip the
         * consistency check for it — Bit-Voting archives carry no real majority
         * config and may legitimately sit next to older BV archives (maj="1"
         * trainer default) in the same directory. */
        if (e->version >= 11 && e->maj_token[0] && strcmp(e->maj_token, "-1") != 0) {
            if (g_ens_maj[0] == '\0') {
                snprintf(g_ens_maj, sizeof(g_ens_maj), "%s", e->maj_token);
                g_ens_majt = e->maj1_thresh;
            } else if (strcmp(g_ens_maj, e->maj_token) != 0 ||
                       g_ens_majt != e->maj1_thresh) {
                fprintf(stderr, "  [WARN] %s: majority mismatch "
                        "(maj=%s/%s maj1-thresh=%d/%d) — skipping\n",
                        e->name, e->maj_token, g_ens_maj,
                        e->maj1_thresh, g_ens_majt);
                continue;
            }
        }
        /* W0-marker consistency: Otto archives (marker != 0) and Bit-Voting
         * archives (marker == 0, no W0 at all) must never mix — the ensemble
         * would combine incompatible members. The first archive decides.
         * Only version >= 10 archives carry the marker field; older v1-v9
         * (incl. int32 v8) are legacy formats without it and are skipped here. */
        if (e->version >= 10) {
            int _is_bitvoting = (e->w0_marker == 0) ? 1 : 0;
            if (g_ens_w0_zero < 0) {
                g_ens_w0_zero = _is_bitvoting;
            } else if (g_ens_w0_zero != _is_bitvoting) {
                fprintf(stderr, "  [WARN] %s: W0-marker mismatch — mixing Bit-Voting "
                        "(w0_marker=0) and Otto (w0_marker!=0) archives — skipping\n",
                        e->name);
                continue;
            }
        }
        /* Precision-consistency (v14+, 2026-08-12): the first v14+ archive
         * sets the expected ot_precision/bit_width/counter_type; every later
         * archive must match — mixing e.g. int32/OT_PRECISION=17 with
         * flt32/OT_PRECISION=16 archives would sum incompatible logits. */
        if (e->version >= 14 && e->ot_precision != 0) {
            if (g_ens_otp == 0 && g_ens_ct[0] == '\0') {
                g_ens_otp = e->ot_precision;
                g_ens_bits = e->bit_width;
                snprintf(g_ens_ct, sizeof(g_ens_ct), "%s", e->counter_type);
            } else if (g_ens_otp != e->ot_precision ||
                       g_ens_bits != e->bit_width ||
                       strcmp(g_ens_ct, e->counter_type) != 0) {
                fprintf(stderr, "  [WARN] %s: precision mismatch "
                        "(otp=%d/%d bits=%d/%d ct=%s/%s) — skipping\n",
                        e->name, e->ot_precision, g_ens_otp,
                        e->bit_width, g_ens_bits,
                        e->counter_type, g_ens_ct);
                continue;
            }
        }
        /* .meta identity: catch stale/mixed configs */
        if (g_meta_ok) {
            if (e->hidden != g_meta_h || e->epochs != g_meta_ep ||
                e->split_vn != g_meta_vn || e->split_hn != g_meta_hn) {
                fprintf(stderr, "  [WARN] %s: .meta config mismatch "
                        "(H=%d/%d EP=%d/%d VN=%d/%d HN=%d/%d) — skipping\n",
                        e->name, e->hidden, g_meta_h, e->epochs, g_meta_ep,
                        e->split_vn, g_meta_vn, e->split_hn, g_meta_hn);
                continue;
            }
            /* .meta maj checks: skip for the "-1" no-majority marker
             * (Bit-Voting) — see blocks_from_index maj-consistency note. */
            if (e->version >= 11 && g_meta_maj[0] && strcmp(g_meta_maj, "-1") != 0 &&
                strcmp(g_meta_maj, e->maj_token) != 0) {
                fprintf(stderr, "  [WARN] %s: .meta maj mismatch (maj=%s/%s) — skipping\n",
                        e->name, e->maj_token, g_meta_maj);
                continue;
            }
            if (e->version >= 11 && g_meta_majt != -999 &&
                strcmp(g_meta_maj, "-1") != 0 && g_meta_majt != e->maj1_thresh) {
                fprintf(stderr, "  [WARN] %s: .meta maj1-thresh mismatch — skipping\n", e->name);
                continue;
            }
            /* .meta score-type check (2026-08-06): the file version encodes
             * the stored score type — a double dir (v12) must not mix float
             * (v9-11) or int64 (v13) archives. Only enforced when the .meta
             * carries the SCORE_TYPE field (newer binaries). */
            if (g_meta_st[0]) {
                int ver_ok = 0;
                if (strcmp(g_meta_st, "IEEE 754 double") == 0)
                    ver_ok = (e->version == 12 || e->version == 15);
                else if (strcmp(g_meta_st, "int64_t") == 0)
                    ver_ok = (e->version == 13 || e->version == 14 || e->version == 8);
                else /* "IEEE 754 float32" */
                    ver_ok = (e->version >= 9 && e->version <= 11);
                if (!ver_ok) {
                    fprintf(stderr, "  [WARN] %s: .meta SCORE_TYPE=%s mismatch (version %d) — skipping\n",
                            e->name, g_meta_st, e->version);
                    continue;
                }
            }
        }
        /* global dims: set from the first passing archive (deterministic —
         * the index is name-sorted), check the rest */
        if (g_n_test == 0) {
            g_n_test = e->n_test; g_n_classes = e->n_classes;
            g_hidden = e->hidden; g_epochs = e->epochs;
            g_split_vn = e->split_vn; g_split_hn = e->split_hn;
            g_target_err_x100 = e->target_err_x100;
        } else if (g_n_test != e->n_test || g_n_classes != e->n_classes ||
                   g_hidden != e->hidden || g_epochs != e->epochs ||
                   g_split_vn != e->split_vn || g_split_hn != e->split_hn ||
                   g_target_err_x100 != e->target_err_x100) {
            fprintf(stderr, "  [WARN] %s: config mismatch (H/EP/VN/HN/TE) — skipping\n", e->name);
            continue;
        }
        int memb_per_xf = e->n_xforms > 0 ? e->n_members / e->n_xforms : 1;
        if (memb_per_xf < 1) memb_per_xf = 1;
        for (int m = 0; m < e->n_members; m++) {
            IndexMember *im = &e->members[m];
            ScoreBlock tmp;
            memset(&tmp, 0, sizeof(tmp));
            tmp.seed = e->seed;
            tmp.member_idx = im->member_idx;
            tmp.file_time = (time_t)e->file_time;
            tmp.file_eval = im->eval;   /* REAL member eval from the index */
            tmp.w0_marker = e->w0_marker;
            if (e->maj_token[0])
                snprintf(tmp.maj_token, sizeof(tmp.maj_token), "%s", e->maj_token);
            tmp.maj1_thresh = (e->version >= 11) ? e->maj1_thresh : -999;
            tmp.total_members = e->n_members;
            tmp.xform_id = m / memb_per_xf;
            if (tmp.xform_id < 0) tmp.xform_id = 0;
            tmp.color = im->color;
            tmp.enc_type = im->enc_type;
            tmp.enc_width = im->enc_width;
            tmp.xform_real = im->xform_real;
            tmp.scores = NULL;   /* scores loaded on demand (load_scores_directory) */
            tmp.err = 0;
            if (im->eval > 0)
                tmp.err = g_n_test - (int)(im->eval * (float)g_n_test / 100.0f + 0.5f);
            strncpy(tmp.source_file, e->name, sizeof(tmp.source_file) - 1);
            tmp.source_file[sizeof(tmp.source_file) - 1] = '\0';

            /* File-level eval filter (absolute only — percentiles run in main).
             * Uses the REAL member eval, so this works identically for M1 and
             * multi-member archives. */
            if (g_eval_active && !g_eval_pct && e->version >= 4) {
                int pass = 0;
                switch (g_eval_cmp) {
                    case 0: pass = (im->eval >  g_eval_thresh); break;
                    case 1: pass = (im->eval >= g_eval_thresh); break;
                    case 2: pass = (im->eval <  g_eval_thresh); break;
                    case 3: pass = (im->eval <= g_eval_thresh); break;
                    case 4: pass = (fabsf(im->eval - g_eval_thresh) < 0.001f); break;
                }
                if (!pass) {
                    /* INTENTIONAL (2026-08-14): the per-member [SKIP] line
                     * was removed — with a low absolute threshold ("eval gt
                     * 80") thousands of members are filtered and the log is
                     * flooded (1000+ lines). The percentile filter path
                     * prints a single summary line in main already; this
                     * absolute path now stays silent and only the final
                     * block count shows the filter effect. */
                    continue;
                }
            }

            if (member_is_filtered(&tmp)) continue;

            if (n_blocks >= n_blocks_cap) {
                int new_cap = n_blocks_cap ? n_blocks_cap * 2 : 4096;
                ScoreBlock *nb = (ScoreBlock *)realloc(blocks, (size_t)new_cap * sizeof(ScoreBlock));
                if (!nb) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
                blocks = nb;
                n_blocks_cap = new_cap;
            }
            blocks[n_blocks++] = tmp;
        }
    }
}

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
static char   g_filter_pat[64][32]; /* member labels (case-insensitive) */
static int    g_filter_pat_not[64]; /* 1 = '!' prefix → EXCLUDE match, 0 = INCLUDE match */
static int    g_filter_re_count = 0; /* number of active regex patterns */
static char   g_filter_re[64][128]; /* POSIX regex patterns vs. full XF:CHAN:ENC spec
                                       (--filter regex 'PATTERN') */
static int    g_filter_re_not[64];  /* 1 = '!' prefix → EXCLUDE match, 0 = INCLUDE match */
static int    g_eval_active = 0;    /* 0=disabled, 1=eval threshold active */
static int    g_eval_cmp = 0;       /* 0:>, 1:>=, 2:<, 3:<=, 4:= */
static int    g_eval_pct = 0;       /* 1 = eval threshold is a PERCENTILE of all
                                       member evals ("gt 50%" = top half), not an
                                       absolute value ("gt 50" = eval > 50) */
static char   g_eval_op = '>';     /* operator char for display */
static float  g_eval_thresh = 0.0f; /* threshold value */
static char   g_ens_maj[8] = "";    /* v11+: majority token of first loaded .ens */
static int    g_ens_majt = -999;    /* v11+: maj1_thresh of first loaded .ens */

/* ── Target-class optimization (2026-08-11) ─────────────────────────
 * --target 3,7        : optimize the ensemble as if only these classes
 *                       existed. correct counts ONLY when
 *                       label ∈ targets AND pred == label (exact hit);
 *                       the denominator is the count of target-class
 *                       samples (recall, 0-100% — not diluted by the
 *                       class distribution to 0-10%).
 * --filter-sample 3,7 : SAMPLE-level filter: only samples whose LABEL
 *                       is in the list enter the evaluation (correct AND
 *                       denominator). Members are still all loaded —
 *                       a member-level eval filter is pointless here,
 *                       every member contains all classes.
 * Both flags accept a comma-separated class list (0..KI_NCLASSES-1).
 * Default (neither flag): exact old behavior. */
static int    g_target_active = 0;      /* 1 = --target given */
static int    g_target_binary = 0;      /* --target-binary: k vs. all other
                                           classes as ONE binary decision.
                                           TP+TN = correct, FP+FN = error
                                           (2026-08-14, plan: specialist
                                           "is it k?" with FP penalty — the
                                           old target-only recall let a
                                           member say "everything is k"). */
static int    g_target_count = 0;       /* number of target classes */
static int    g_target_classes[10];     /* --target 3,7 → {3,7} */
static int    g_filter_sample_active = 0; /* 1 = --filter-sample given */
static int    g_filter_sample_count = 0;
static int    g_filter_sample_classes[10]; /* --filter-sample 0,9 */
static int    g_eval_denom = 0;         /* effective denominator (after label load) */
/* .meta (directory identity) values — used to validate every .ens against
 * the directory's recorded config. SEED is intentionally NOT a criterion:
 * --ensembleN (per-member seeds) and --sweep are mutually exclusive, and
 * W0 verification runs via the w0_marker of the first member only. */
static int    g_meta_ok = 0;        /* 1 = .meta exists with H/EP/VN/HN */
static int    g_meta_h = 0, g_meta_ep = 0, g_meta_vn = 0, g_meta_hn = 0;
static int    g_meta_bits = 0;      /* BITS= (container bit width, 0 = legacy absent) */
static int    g_meta_otp = 0;       /* OT_PRECISION (F=1<<OT_PRECISION logit scale, 0 = legacy absent) */
/* Precision-consistency across archives (2026-08-12): the FIRST v14+ archive
 * decides the expected ot_precision/bit_width/counter_type; later archives
 * must match (mixing int32/otp17 with flt32/otp17 archives would corrupt the
 * ensemble). 0 = unset (no v14+ archive seen yet). */
static int    g_ens_otp = 0;
static int    g_ens_bits = 0;
static char   g_ens_ct[16] = "";
static char   g_meta_maj[8] = "";   /* "" = field absent */
static int    g_meta_majt = -999;   /* -999 = field absent */
static char   g_member_out[1024] = "";  /* --member-out PATH: write optimal subset as member file */
static int    g_member_out_default = 0; /* --member-out-default: derive member-{DIR}.out
                                           from the corpus dir (2026-08-16) */
static char   g_xform_spec[1024] = "";  /* --xform SPEC: expand xform list (e.g. "sweep@spiral") */
static char   g_member_seed_spec[256] = ""; /* --member-seed SPEC: pre-seed beam with this member spec */
static char   g_start_member_file[1024] = ""; /* --member-start FILE: start the beam from a
                                           previously exported attractor (--member-out) instead of
                                           a single seed. The beam then only ADDS members — the
                                           pool extension becomes monotone: adding @shuffle etc.
                                           can no longer redirect the path below the saved attractor
                                           (Finding 11 fix, 2026-08-09). */
static int    g_tries = 1;              /* --tries N: repeat beam search N times with shuffle, keep best */
static int    g_tries_no_lock = 0;      /* --tries-no-lock: each try searches the FULL pool
                                           (no locking of previously found members — the beam
                                           gets a fresh empty pool_exclude per try, so later
                                           tries can re-use members found by earlier tries and
                                           reach DIFFERENT attractors, not disjoint partitions;
                                           plan-2026-08-08 Finding 1). */
static int    g_tries_seed = 0;         /* --tries-top-seed (0, default) / --tries-random-seed (1):
                                           how the per-try seed member is chosen. top = k-th
                                           strongest unused member (deterministic, reproducible);
                                           random = splitmix64 pick (broader diversity). */
static int    g_beam_bestN = 1;         /* --beam-bestN N (default 1): start the per-level beam
                                           selection at the N-th best candidate instead of the
                                           1st-best. Breaks the fixation on the 1st-best member
                                           (a once-added member is never re-evaluated as wrong —
                                           add-only beam). Maximum diversity across --tries: try k
                                           uses bestN=k. Reproduced via lr= in the REPORT (lr is
                                           unused by merge-ensemble, so it carries bestN). */
static int    g_bestN_winner = 1;       /* bestN of the winning try (for the REPORT lr= field) */
static int    g_max_mode = 0;           /* --max: cumulative acceptance (2026-08-07)
                                           — per-step acceptance threshold 0.0,
                                           block-commit when the accumulated gain
                                           over the last beam_width steps reaches
                                           min-gain; strict (per-member) is default */
static int    g_max_members = 0;        /* --max N: stop greedy/beam at N members (0 = unlimited) */
static int    g_expansion_sort = 0;     /* --expansion-sort: 0=abs (default), 1=marginal,
                                           2=clarity (member's own class-score std-dev —
                                           prefer unambiguous voters that disturb other
                                           members the least, plan-2026-08-08) */
static float  g_min_gain = 0.01f;       /* --min-gain F: min. improvement %% to keep adding (default: 0.01) */
static int    g_optimal = 0;            /* --optimal: 2-opt exchange on the beam result
                                           (removal allowed — breaks the add-only
                                           attractor, plan-2026-08-08 Stage 2) */
/* --optimal mode (2026-08-12): 0 = best-improvement (default, deterministic,
 * applies the GLOBAL best swap per pass), 1 = first-improvement (applies the
 * FIRST swap that improves — explores a different local optimum, faster per
 * pass). Both stay monotonic (err strictly decreases). */
static int    g_optimal_mode = 0;       /* 0 = best (default), 1 = first */
static int    g_optimal_passes = 0;     /* --optimal-passes N: cap the fixpoint
                                           search at N passes (0 = unlimited,
                                           run until no improvement) */
static int    g_greedy_clarity = 0;     /* --greedy-clarity: among candidates that do NOT
                                           worsen the current ensemble (acc >= prev_acc),
                                           pick the one with the highest member clarity
                                           (own class-score std-dev) instead of the best
                                           acc — prefer unambiguous voters (plan-2026-08-08) */
static int    g_debug = 0;              /* --debug: verbose debug output */
static int    g_debug_member = 0;      /* --debug-member: detailed list of the SELECTED
                                          members after the beam, in the trainer's
                                          --debug-member format (2026-08-12) */
static int    g_debug_confusion = 0;   /* --debug-confusion: print the final ensemble's
                                          confusion matrix (tprint table, same as the
                                          trainer's --debug-confusion) after the beam /
                                          greedy / multi-try search (2026-08-14) */
static int    g_seed_sort = 0;         /* --seed-sort: print single-member eval table */
static int    g_seed_top = 0;          /* --seed-sort N: show only top N rows (0 = all) */
static char   g_sample_index[1024] = ""; /* --sample-index FILE: write member↔sample
                                           recognition pairs as Parquet
                                           (2026-08-15, 3rd gen) — full pair table,
                                           ALL assignments (argmax pred) + margin */
static int    g_stdout = 0;            /* --stdout: pure filter-export — print the filtered
                                          XF:CHAN:ENC specs to stdout, suppress normal output */
static int    g_stdout_field = 0;      /* --stdout column extraction: 0=full spec,
                                          1=xform, 2=channel, 3=encoding (--xform/--channel/--encoding) */
static float    g_best_eval = -1.0f;      /* global best across tries (beam) */
static int      g_best_n = 0;            /* member count for global best */
static uint8_t *g_best_used = NULL;     /* used mask for global best */
/* Additionsreihenfolge des global besten Ensembles (order[] aus dem Beam),
 * in der Reihenfolge, in der die Member dem Ensemble hinzugefügt wurden.
 * --debug-member iteriert darüber, damit acc[%] kumulativ monoton steigt
 * und die Tabelle exakt die Member des member-out files zeigt (gleiche
 * Menge, andere Sortierung). 2026-08-14. */
static int      g_best_order_n = 0;      /* Anzahl Member in g_best_order */
static int     *g_best_order = NULL;     /* block-Indizes in Additionsreihenfolge */

/* ═══════════════════════════════════════════════════════════════════════
 * COMPLETION TABLE — for --completion flag (bash auto-completion)
 * ═══════════════════════════════════════════════════════════════════════
 * Same scheme as the trainer's table in ki-common.h: type is one of
 * "none", "file", "dir", "num", "float", "token"; values are the token
 * list for "token" type ("@color"/"@enc"/"@xform" markers are resolved
 * dynamically by comp_tokens_build from ki-common.h). */
static const struct _comp_entry merge_comp_table[] = {
    {"--check",            "none",  NULL},
    {"--filter",           "token", "eval regexp regex ! :"},
    {"--filter-t1",        "none",  NULL},
    {"--target",           "token", "0 1 2 3 4 5 6 7 8 9"},
    {"--target-binary",    "none",  NULL},
    {"--filter-sample",    "token", "0 1 2 3 4 5 6 7 8 9"},
    {"--seed-sort",        "num",   NULL},
    {"--sample-index",     "file",  NULL},
    {"--greedy",           "none",  NULL},
    {"--greedy-clarity",   "none",  NULL},
    {"--beam",             "num",   NULL},
    {"--tries",            "num",   NULL},
    {"--tries-no-lock",    "none",  NULL},
    {"--tries-top-seed",   "none",  NULL},
    {"--tries-random-seed","none",  NULL},
    {"--beam-bestN",      "num",   NULL},
    {"--max",              "num",   NULL},
    {"--min-gain",         "float", NULL},
    {"--member-seed",      "token", NULL},
    {"--member-start","file",  NULL},
    {"--eff-lambda",       "float", NULL},
    {"--expansion-sort",   "token", "abs marg marginal clarity clr"},
    {"--optimal",          "token", "best first"},
    {"--optimal-passes",   "num",   NULL},
    {"--debug",            "none",  NULL},
    {"--debug-member",     "none",  NULL},
    {"--debug-confusion",  "none",  NULL},
    {"--stdout",           "none",  NULL},
    {"--encoding",         "token", "@enc"},
    {"--channel",          "token", "@color"},
    {"--xform",            "token", "@xform"},
    {"--member-out",       "file",  NULL},
    {"--member-out-default", "none", NULL},
    {"--save",             "file",  NULL},
    {"--num",              "num",   NULL},
    {"--help",             "none",  NULL},
    {NULL, NULL, NULL}
};

/* --completion [--<flag>] — merge-ensemble's OWN table (the trainer's
 * ki_completion_print would answer from ITS _comp_table). Reuses
 * comp_tokens_build() from ki-common.h for the "@" dynamic markers. */
static void merge_completion_print(const char *target) {
    const struct _comp_entry *e;
    for (e = merge_comp_table; e->flag; e++) {
        if (strcmp(e->flag, target) != 0) continue;
        if (e->values && e->values[0] == '@') {
            char _tb[2048];
            comp_tokens_build(e->values, _tb, sizeof(_tb));
            printf("%s %s\n", e->type, _tb);
        } else if (e->values) {
            printf("%s %s\n", e->type, e->values);
        } else {
            printf("%s\n", e->type);
        }
        return;
    }
    fprintf(stderr, "[ERROR] --completion: unknown flag '%s'\n", target);
}

static void merge_completion_dispatch(int argc, char **argv, int *pi) {
    /* argv[*pi] == "--completion"; an optional next arg is the target flag */
    if (*pi + 1 < argc && argv[*pi + 1][0] == '-' && argv[*pi + 1][1] == '-') {
        merge_completion_print(argv[++(*pi)]);
        exit(0);
    }
    /* --completion alone → all valid flag names */
    for (const struct _comp_entry *e = merge_comp_table; e->flag; e++)
        puts(e->flag);
    exit(0);
}

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

/* float comparator for percentile threshold lookup */
static int cmp_float(const void *a, const void *b) {
    float fa = *(const float *)a, fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
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

/* -- Match a full member spec against ONE compiled regex pattern ------
 * Patterns are compiled once (REG_EXTENDED|REG_ICASE) on first use. */
static int filter_regex_match_idx(int idx, const char *spec) {
    static regex_t re[64];
    static int ok[64];
    static int compiled = 0;
    if (!compiled) {
        for (int i = 0; i < g_filter_re_count; i++) {
            ok[i] = (regcomp(&re[i], g_filter_re[i], REG_EXTENDED | REG_ICASE) == 0);
            if (!ok[i])
                fprintf(stderr, "  [WARN] invalid regex (ignored): %s\n", g_filter_re[i]);
        }
        compiled = 1;
    }
    if (idx < 0 || idx >= g_filter_re_count || !ok[idx]) return 0;
    return (regexec(&re[idx], spec, 0, NULL, 0) == 0);
}

/* -- Check if member matches any filter pattern --------------
 * SEMANTICS (unified INCLUDE/EXCLUDE):
 *   - Every filter pattern (label substring or regex) has a direction:
 *       plain pattern        → INCLUDE: keep member if it matches
 *       '!' prefixed pattern → EXCLUDE: drop member if it matches
 *   - All filter kinds AND together. Within one kind:
 *       INCLUDE patterns OR  (match any → keep)
 *       EXCLUDE patterns OR  (match any → drop)
 *   - If a kind has only EXCLUDE patterns, non-matching members pass
 *     (no implicit INCLUDE). Example: '!colswap' drops all colswap
 *     members and keeps everything else.
 *   - Regex ('regex PATTERN') is matched against the full XF:CHAN:ENC
 *     spec; '!' goes after the 'regex ' keyword: --filter regex '!pat'.
 */
static int member_is_filtered(const ScoreBlock *b) {
    if (g_filter_count == 0 && g_filter_re_count == 0) return 0;
    char label[32];
    member_label(b, label, sizeof(label));
    /* Full spec "XF:CHAN:ENC" — label filters match against THIS (substring),
     * so e.g. '!colswap' excludes members whose xform contains "colswap",
     * and 'sig8' matches any encoding/color containing "sig8". */
    char spec[128];
    const char *_xn = (b->xform_real >= 0)
                      ? xform_name_safe(b->xform_real)
                      : xform_group_label(b->xform_id);
    snprintf(spec, sizeof(spec), "%s:%s", _xn, label);

    /* ── Label patterns (substring match on full XF:CHAN:ENC, case-insens.) ── */
    int label_have_inc = 0, label_inc_ok = 0, label_exc_ok = 0;
    for (int i = 0; i < g_filter_count; i++) {
        int m = (strcasecmp(spec, g_filter_pat[i]) == 0) ||
                (strcasestr(spec, g_filter_pat[i]) != NULL);
        if (g_filter_pat_not[i]) { if (m) label_exc_ok = 1; }
        else                      { label_have_inc = 1; if (m) label_inc_ok = 1; }
    }
    if (label_exc_ok) return 1;               /* any '!' label matched → drop */
    if (label_have_inc && !label_inc_ok) return 1;  /* INCLUDE labels, none matched → drop */

    /* ── Regex patterns (full XF:CHAN:ENC spec) ── */
    if (g_filter_re_count > 0) {
        int re_have_inc = 0, re_inc_ok = 0, re_exc_ok = 0;
        for (int i = 0; i < g_filter_re_count; i++) {
            int m = filter_regex_match_idx(i, spec);
            if (g_filter_re_not[i]) { if (m) re_exc_ok = 1; }
            else                     { re_have_inc = 1; if (m) re_inc_ok = 1; }
        }
        if (re_exc_ok) return 1;              /* any '!' regex matched → drop */
        if (re_have_inc && !re_inc_ok) return 1;  /* INCLUDE regexes, none matched → drop */
    }
    return 0;
}

/* ── Target-class / filter-sample evaluation helpers (2026-08-11) ──
 * These centralize the "is sample s correct" decision that used to be
 * duplicated at every correct++ site. Semantics:
 *   --target T        : label ∈ T AND pred == label  (exact hit of the
 *                       target class; denominator = target-class count)
 *   --filter-sample L : only samples with label ∈ L enter the eval
 *                       (both correct AND denominator); members are all
 *                       loaded — this filters the SAMPLE set, not members.
 * Default (neither): pred == label, denominator = n_test (old behavior). */
static int in_class_list(const int *list, int n, int c) {
    for (int i = 0; i < n; i++)
        if (list[i] == c) return 1;
    return 0;
}

/* Returns 1 if a sample with true label `label` and PREDICTED class
 * `pred` counts as "correct" under the active target/filter-sample mode.
 * This is the pure membership decision (no argmax) — callers that already
 * computed the argmax (candidate evaluation with base+sc sums) use this
 * directly; eval_sample_correct() wraps it with the argmax. */
static int eval_sample_correct_pred(int pred, uint8_t label) {
    /* --filter-sample: exclude samples whose label is not in the list */
    if (g_filter_sample_active &&
        !in_class_list(g_filter_sample_classes, g_filter_sample_count, (int)label))
        return 0;
    if (g_target_active) {
        if (g_target_binary) {
            /* BINARY: --target k vs. all other classes as ONE decision.
             * correct = (label==k && pred==k)  [TP]
             *         | (label!=k && pred!=k)  [TN]
             * FP (label!=k, pred==k) and FN (label==k, pred!=k) are errors.
             * This penalizes the degenerate "everything is k" specialist
             * that a pure target-recall objective tolerates. */
            int is_k   = in_class_list(g_target_classes, g_target_count, (int)label);
            int pred_k = in_class_list(g_target_classes, g_target_count, pred);
            return (is_k == pred_k) ? 1 : 0;
        }
        /* Variante A: only target-class samples count, and only when the
         * prediction EXACTLY hits the target class. */
        if (!in_class_list(g_target_classes, g_target_count, (int)label)) return 0;
        return (pred == (int)label) ? 1 : 0;
    }
    return (pred == (int)label) ? 1 : 0;
}

/* Row-based wrapper: computes argmax over the K class scores, then applies
 * the membership decision. Used at the commit/print sites where only the
 * committed sum_scores row is available. */
static int eval_sample_correct(const SCORE_TYPE *row, uint8_t label) {
    int pred = 0;
    for (int k = 1; k < g_n_classes; k++)
        if (row[k] > row[pred]) pred = k;
    return eval_sample_correct_pred(pred, label);
}

/* Effective denominator for accuracy display: target/filter-sample mode
 * (computed once after label load), n_test otherwise. Never 0. */
static int eval_denom(void) {
    return (g_eval_denom > 0) ? g_eval_denom : g_n_test;
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
/* -- Load scores for all blocks still missing them (scores==NULL) ---------
 * The .index cache path loads metadata ONLY (blocks_from_index) so that pure
 * metadata operations (filters, columns, --stdout, dedup, sort) never open the
 * .ens files. Scores are fetched on demand here — after ALL filters ran —
 * grouped by source_file so each archive is opened exactly once. Labels are
 * loaded from the first archive that carries them and verified against the
 * rest (same as load_archive did). Returns number of blocks filled. */
static int cmp_block_source(const void *a, const void *b) {
    int ia = *(const int *)a, ib = *(const int *)b;
    return strcmp(blocks[ia].source_file, blocks[ib].source_file);
}

static int load_scores_directory(const char *dir) {
    /* Collect indices of blocks that still need scores */
    int need = 0;
    for (int i = 0; i < n_blocks; i++)
        if (blocks[i].scores == NULL) need++;
    if (need == 0) return 0;

    int *_order = (int *)malloc((size_t)need * sizeof(int));
    if (!_order) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
    int k = 0;
    for (int i = 0; i < n_blocks; i++)
        if (blocks[i].scores == NULL) _order[k++] = i;
    qsort(_order, (size_t)need, sizeof(int), cmp_block_source);

    /* ── Group ranges: all members of one source_file are contiguous in
     * _order (cmp_block_source). Precompute [grp_begin, grp_end) per file so
     * the OMP loop can iterate FILES (each opened exactly once) instead of
     * blocks — one worker per file, disjoint blocks per worker. (2026-08-14) */
    int n_groups = 0;
    for (int o = 0; o < need; ) {
        int oe = o;
        while (oe < need &&
               strcmp(blocks[_order[oe]].source_file, blocks[_order[o]].source_file) == 0)
            oe++;
        n_groups++;
        o = oe;
    }
    int *grp_begin = (int *)malloc((size_t)n_groups * sizeof(int));
    int *grp_end   = (int *)malloc((size_t)n_groups * sizeof(int));
    if (!grp_begin || !grp_end) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
    {
        int gi = 0;
        for (int o = 0; o < need; ) {
            int oe = o;
            while (oe < need &&
                   strcmp(blocks[_order[oe]].source_file, blocks[_order[o]].source_file) == 0)
                oe++;
            grp_begin[gi] = o; grp_end[gi] = oe; gi++;
            o = oe;
        }
    }

    int filled = 0;
    int files_processed = 0;
    /* ── Progress indicator (2026-08-03): ONE line of dots ──
     * The score load reads every .ens file (~19 GB for 48k archives) and
     * takes 30-120 s — without this the run looks frozen between the SETUP
     * output and the search header. One dot per dot_interval files keeps it
     * to a single line (~48 dots) instead of one line per 1000 files. */
    int dot_interval = need / 48;
    if (dot_interval < 1) dot_interval = 1;
    if (!g_stdout) {
        printf("  Loading scores");
        fflush(stdout);
    }

    /* ── Labels + eval denominator ONCE, before the parallel loop ──
     * Every archive carries the SAME test labels (the .index/.meta already
     * validated the corpus identity), so they are loaded from the first
     * readable file here and then only READ by the parallel workers below.
     * INTENTIONAL: the label MISMATCH warning previously checked every
     * archive; with OMP that would race on stderr + duplicate the read, and
     * corpus identity is already guaranteed by the .index consistency check.
     * A missing-labels archive still emits the classic warning. */
    for (int gr = 0; gr < n_groups && !g_labels; gr++) {
        const char *fname = blocks[_order[grp_begin[gr]]].source_file;
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dir, fname);
        EnsReader rd;
        if (ens_reader_open(&rd, path) != 0) continue;
        if (rd.n_members < 1 || rd.n_members > 100000) { ens_reader_close(&rd); continue; }
        if (!COUNTER_TYPE_IS_FLOAT) rd.fp_scale = 1;
        if (ens_reader_skip_all_meta(&rd) != 0) { ens_reader_close(&rd); continue; }
        int _ok = 1;
        for (uint32_t m = 0; m < rd.n_members; m++)
            if (ens_reader_skip_scores(&rd) != 0) { _ok = 0; break; }
        if (!_ok) { ens_reader_close(&rd); continue; }
        g_labels = (uint8_t *)malloc((size_t)g_n_test);
        if (!g_labels) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
        if (fread(g_labels, 1, (size_t)g_n_test, rd.f) != (size_t)g_n_test) {
            fprintf(stderr, "  [WARN] %s: no labels found (old format?)\n", path);
            free(g_labels); g_labels = NULL;
        }
        ens_reader_close(&rd);
    }
    /* Effective denominator for the target/filter-sample modes: computed
     * ONCE here (identical to the old first-label-load logic). */
    if (g_labels && g_eval_denom == 0) {
        if (g_target_binary) {
            g_eval_denom = g_n_test;   /* binary accuracy, not recall */
        } else {
            int d = 0;
            for (int i = 0; i < g_n_test; i++) {
                int lb = (int)g_labels[i];
                if (g_filter_sample_active &&
                    !in_class_list(g_filter_sample_classes,
                                   g_filter_sample_count, lb)) continue;
                if (g_target_active &&
                    !in_class_list(g_target_classes, g_target_count, lb)) continue;
                d++;
            }
            g_eval_denom = d > 0 ? d : g_n_test;   /* guard div-by-zero */
        }
    }

    /* ── PARALLEL score load (2026-08-14) ──
     * One OMP worker per source_file group. Every worker opens its OWN
     * EnsReader (stack-local → thread-safe) and writes into DISJOINT blocks
     * (scores / score_min / score_max / clarity are per-member, never shared
     * between workers). Labels + g_eval_denom are read-only here (pre-loaded
     * above) — no lock needed. Thread count = the global OMP setting the
     * merge already uses (omp_get_num_procs / OMP_NUM_THREADS, see main);
     * set OMP_NUM_THREADS to control it. schedule(dynamic) balances the
     * (homogeneous ~810 KB) files across workers. */
    int files_done = 0;
    #pragma omp parallel for schedule(dynamic) reduction(+:filled)
    for (int gr = 0; gr < n_groups; gr++) {
        int o = grp_begin[gr], oe = grp_end[gr];
        const char *fname = blocks[_order[o]].source_file;

        /* one dot per dot_interval files, printed by thread 0 only
         * (omp master is not allowed inside a worksharing loop) */
        #pragma omp atomic
        files_done++;
        if (!g_stdout && omp_get_thread_num() == 0 &&
            (files_done % dot_interval == 0)) {
            printf(".");
            fflush(stdout);
        }

        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dir, fname);

        /* ── Open + parse via the shared reader (lib/ki-ens.h) — all version
         * logic (bounds, width, metadata layout) is central; a version missed
         * in one path can no longer drift (bug 2026-08-06: "0 blocks filled"
         * SEGV). */
        EnsReader rd;
        if (ens_reader_open(&rd, path) != 0) continue;
        if (rd.n_members < 1 || rd.n_members > 100000) { ens_reader_close(&rd); continue; }
        /* int32-merge fixed-point scale on v9-11 float scores (backward
         * compatible with the int32 merge semantics; argmax-invariant). */
        if (!COUNTER_TYPE_IS_FLOAT) rd.fp_scale = 1;
        if (ens_reader_skip_all_meta(&rd) != 0) { ens_reader_close(&rd); continue; }
        uint32_t n_members = rd.n_members;
        size_t score_sz = rd.score_sz;
        int bad = 0;

        for (uint32_t m = 0; m < n_members; m++) {
            /* Find matching block within [o, oe) — member_idx m of this file */
            int match = -1;
            for (int q = o; q < oe; q++)
                if (blocks[_order[q]].member_idx == (int)m) { match = _order[q]; break; }
            if (match < 0) {  /* member filtered out — skip its scores */
                if (ens_reader_skip_scores(&rd) != 0) { bad = 1; break; }
                continue;
            }
            SCORE_TYPE *scores = (SCORE_TYPE *)calloc(score_sz, sizeof(SCORE_TYPE));
            if (!scores) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
            if (ens_reader_read_scores(&rd, scores) != 0) {
                fprintf(stderr, "  [WARN] %s: short read member %u — skipping\n", path, m);
                free(scores); bad = 1; break;
            }
            blocks[match].scores = scores;
            /* min/max score (same as load_archive) — type-generic limits:
             * float/double → ±INFINITY, int64 → INT64 extremes. (Was split by
             * COUNTER_TYPE_IS_FLOAT, which breaks mixed builds like
             * SCORE_TYPE=double with MODE_INT32 or int64 with float counters,
             * 2026-08-06.) */
            blocks[match].score_min = _Generic((SCORE_TYPE)0,
                int64_t: (SCORE_TYPE)INT64_MAX,
                default: (SCORE_TYPE)INFINITY);
            blocks[match].score_max = _Generic((SCORE_TYPE)0,
                int64_t: (SCORE_TYPE)INT64_MIN,
                default: (SCORE_TYPE)-INFINITY);
            if (g_n_test > 0) {
                for (size_t _si = 0; _si < score_sz; _si++) {
                    if (scores[_si] < blocks[match].score_min) blocks[match].score_min = scores[_si];
                    if (scores[_si] > blocks[match].score_max) blocks[match].score_max = scores[_si];
                }
            }
            /* ── clarity: mean per-sample std-dev of the K class scores ──
             * Computed in double (int64 squares would overflow), offset-invariant
             * (the mean-subtraction cancels the .ens target+offset). O(N·K) once
             * per member; used by --expansion-sort clarity / --greedy-clarity.
             * plan-2026-08-08: members with high clarity vote unambiguously and
             * disturb other members' decisions the least. */
            {
                double _cl_sum = 0.0;
                int _cl_n = 0;
                for (int _s = 0; _s < g_n_test; _s++) {
                    const SCORE_TYPE *_row = scores + (size_t)_s * (size_t)g_n_classes;
                    double _mean = 0.0;
                    for (int _k = 0; _k < g_n_classes; _k++)
                        _mean += (double)_row[_k];
                    _mean /= (double)g_n_classes;
                    double _var = 0.0;
                    for (int _k = 0; _k < g_n_classes; _k++) {
                        double _d = (double)_row[_k] - _mean;
                        _var += _d * _d;
                    }
                    _cl_sum += sqrt(_var / (double)g_n_classes);
                    _cl_n++;
                }
                blocks[match].clarity = (_cl_n > 0) ? _cl_sum / (double)_cl_n : 0.0;
            }
            filled++;
        }
        (void)bad;   /* INTENTIONAL: label-read skipped in parallel workers
                      * (done once in the pre-loop above); scores are the only
                      * payload read here. */
        ens_reader_close(&rd);
    }
    files_processed = files_done;
    /* Close the dot line and report the summary. */
    if (!g_stdout && files_processed > 0)
        printf(" done (%d files, %d blocks filled)\n", files_processed, filled);
    free(grp_begin); free(grp_end);
    free(_order);
    return filled;
}

/* ── Load <dir>/.meta into the g_meta_* globals (directory identity) ──
 * Used by load_directory AND fix_directory (--check) — the latter needs
 * g_meta_bits/g_meta_otp/g_meta_ct to validate v14+ precision against the
 * archives (2026-08-12). Returns 1 when a .meta was read, 0 otherwise. */
static int meta_load_globals(const char *dir) {
    char meta_path[1024];
    snprintf(meta_path, sizeof(meta_path), "%s/.meta", dir);
    FILE *mf = fopen(meta_path, "r");
    if (!mf) return 0;
    int m_h = 0, m_ep = 0, m_vn = 0, m_hn = 0, m_seed = 0;
    int m_bits = 0;         /* 0 = field absent (legacy .meta) */
    int m_otp  = 0;         /* OT_PRECISION (0 = field absent) */
    char m_maj[8] = "";     /* empty = field absent */
    int  m_majt = -999;     /* -999 = field absent */
    char m_ct[64] = "";     /* COUNTER_TYPE label (empty = absent) */
    char m_st[64] = "";     /* SCORE_TYPE label (empty = absent) */
    char line[128];
    while (fgets(line, sizeof(line), mf)) {
        if (sscanf(line, "H=%d", &m_h) == 1) continue;
        if (sscanf(line, "EPOCHS=%d", &m_ep) == 1) continue;
        if (sscanf(line, "VN=%d", &m_vn) == 1) continue;
        if (sscanf(line, "HN=%d", &m_hn) == 1) continue;
        if (sscanf(line, "SEED=%d", &m_seed) == 1) continue;
        if (sscanf(line, "MAJ=%7s", m_maj) == 1) continue;
        if (sscanf(line, "MAJ1_THRESH=%d", &m_majt) == 1) continue;
        if (sscanf(line, "BITS=%d", &m_bits) == 1) continue;
        if (sscanf(line, "OT_PRECISION=%d", &m_otp) == 1) continue;
        /* labels contain spaces ("IEEE 754 double") → rest of line */
        if (sscanf(line, "COUNTER_TYPE=%63[^\n]", m_ct) == 1) continue;
        if (sscanf(line, "SCORE_TYPE=%63[^\n]", m_st) == 1) continue;
    }
    fclose(mf);
    g_meta_ok = 1;
    g_meta_h = m_h; g_meta_ep = m_ep; g_meta_vn = m_vn; g_meta_hn = m_hn;
    g_meta_bits = m_bits;
    g_meta_otp = m_otp;
    g_meta_maj[0] = '\0';
    if (m_maj[0]) snprintf(g_meta_maj, sizeof(g_meta_maj), "%s", m_maj);
    g_meta_majt = m_majt;
    g_meta_ct[0] = '\0';
    if (m_ct[0]) snprintf(g_meta_ct, sizeof(g_meta_ct), "%s", m_ct);
    g_meta_st[0] = '\0';
    if (m_st[0]) snprintf(g_meta_st, sizeof(g_meta_st), "%s", m_st);
    return 1;
}

/* -- Load all .ens files from directory ---------------------- */
static int load_directory(const char *dir) {
    if (!dir || !dir[0]) return -1;

    /* Read .meta (directory identity) if present */
    if (meta_load_globals(dir)) {
        int m_h = g_meta_h, m_ep = g_meta_ep, m_vn = g_meta_vn, m_hn = g_meta_hn;
        int m_seed = 0;   /* SEED is not stored in g_meta_* (not a merge criterion) */
        int m_bits = g_meta_bits, m_otp = g_meta_otp;
        char m_maj[8]; snprintf(m_maj, sizeof(m_maj), "%s", g_meta_maj);
        int  m_majt = g_meta_majt;
        char m_ct[64]; snprintf(m_ct, sizeof(m_ct), "%s", g_meta_ct);
        char m_st[64]; snprintf(m_st, sizeof(m_st), "%s", g_meta_st);
        if (!g_stdout) {
                printf("  Config from .meta: H=%d  EP=%d  VN=%d  HN=%d",
                       m_h, m_ep, m_vn, m_hn);
                if (m_seed != 0) printf("  SEED=%d", m_seed);
                /* MAJ=-1 = explicit "no majority" marker (Bit-Voting builds);
                 * show it as such instead of a misleading maj=1 default. */
                if (m_maj[0] && strcmp(m_maj, "-1") != 0) {
                    printf("  MAJ=%s", m_maj);
                    if (m_majt != -999) printf("  MAJ1_THRESH=%d", m_majt);
                } else if (m_maj[0] && strcmp(m_maj, "-1") == 0) {
                    printf("  MAJ=-1 (Bit-Voting: no majority)");
                }
                if (m_ct[0]) printf("  COUNTER_TYPE=%s", m_ct);
                if (m_st[0]) printf("  SCORE_TYPE=%s", m_st);
                if (m_bits != 0) printf("  BITS=%d", m_bits);
                if (m_otp != 0)  printf("  OT_PRECISION=%d", m_otp);
                printf("\n");
            }
        } else {
            if (!g_stdout) printf("  (no .meta — legacy directory, config from first archive)\n");
        }

    int n_files = 0;
    int n_rebuilt = 0;       /* files actually re-read (new/changed) */
    int n_orphaned = 0;      /* index entries whose file disappeared */
    IndexFile *idx = NULL;
    int n_idx = index_load(dir, &idx);
    if (n_idx == -2) {
        /* Config mismatch: stale index from a different training run — drop
         * it and rebuild everything. */
        if (!g_stdout) printf("  .index: config mismatch — full rebuild\n");
        n_idx = -1;
    }

    /* stat() snapshot of ALL current .ens files (name-sorted, ctime+size) —
     * the incremental-rebuild decision basis. */
    IndexSnap *snap = NULL;
    int n_snap = index_snapshot(dir, &snap);

    if (n_idx >= 0 && n_snap < 0) {
        /* Index present but dir unreadable — fall back to the cache alone */
        if (!g_stdout) printf("  .index: %d files (dir unreadable — cache only)\n", n_idx);
        blocks_from_index(idx, n_idx);
        n_files = n_idx;
        for (int i = 0; i < n_idx; i++) free(idx[i].members);
        free(idx); free(snap);
        return n_files;
    }

    if (n_idx < 0) {
        /* No/stale index → full build: read EVERY .ens file. Names come from
         * the (sorted) snapshot so processing order is deterministic — this
         * fixes the global state (maj token, dims) from a stable first file. */
        IndexFile *build = (IndexFile *)calloc((size_t)(n_snap > 0 ? n_snap : 1),
                                               sizeof(IndexFile));
        if (!build) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
        int n_build = 0;
        for (int i = 0; i < n_snap; i++) {
            char path[1024];
            snprintf(path, sizeof(path), "%s/%s", dir, snap[i].name);
            IndexFile tmp;
            read_ens_meta(path, &tmp);
            build[n_build++] = tmp;
            n_rebuilt++;
        }
        qsort(build, (size_t)n_build, sizeof(IndexFile), cmp_index_file);
        if (!g_stdout)
            printf("  .index: built %d file(s) (full build)\n", n_build);
        index_save(dir, build, n_build);
        blocks_from_index(build, n_build);
        n_files = n_build;
        for (int i = 0; i < n_build; i++) free(build[i].members);
        free(build);
    } else {
        /* Incremental merge of the cache (idx, name-sorted) with the current
         * snapshot (snap, name-sorted). Entries whose (ctime,size) match are
         * kept WITHOUT opening the .ens file; only new/changed files are
         * re-read. Orphaned entries (file deleted) are dropped. */
        IndexFile *merged = (IndexFile *)calloc((size_t)n_snap, sizeof(IndexFile));
        if (!merged) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
        int n_merged = 0;
        int i = 0, j = 0;
        while (i < n_snap) {
            int cmp = (j < n_idx) ? strcmp(snap[i].name, idx[j].name) : -1;
            if (cmp == 0) {
                /* Same name: keep cached entry iff ctime+size unchanged */
                if (snap[i].ctime == idx[j].ctime && snap[i].size == idx[j].size) {
                    merged[n_merged++] = idx[j];   /* take ownership, no re-read */
                    idx[j].members = NULL;         /* prevent double free */
                } else {
                    char path[1024];
                    snprintf(path, sizeof(path), "%s/%s", dir, snap[i].name);
                    /* FIX (2026-08-10) double-free: read_ens_meta allocates a
                     * NEW members buffer (no aliasing with idx[j].members),
                     * so the old cache allocation must be released BEFORE the
                     * re-read — and set to NULL so the final cleanup loop
                     * (free(idx[ii].members)) skips it instead of freeing a
                     * stale pointer a second time. */
                    free(idx[j].members);
                    idx[j].members = NULL;
                    read_ens_meta(path, &merged[n_merged++]);
                    n_rebuilt++;
                }
                i++; j++;
            } else if (cmp < 0) {
                /* New file (not in cache) */
                char path[1024];
                snprintf(path, sizeof(path), "%s/%s", dir, snap[i].name);
                read_ens_meta(path, &merged[n_merged++]);
                n_rebuilt++;
                i++;
            } else {
                /* Cache entry with no matching file → orphan */
                /* FIX (2026-08-10): NULL after free — the final cleanup loop
                 * (free(idx[ii].members)) must skip this entry instead of
                 * double-freeing it (crash: "free(): double free detected"
                 * after deleting 208 @shuffle files, 2026-08-10). */
                free(idx[j].members);
                idx[j].members = NULL;
                n_orphaned++;
                j++;
            }
        }
        /* FIX (2026-08-10): trailing cache entries whose file is gone.
         * The main loop runs over the SNAPSHOT (i < n_snap) — any remaining
         * idx entries (j < n_idx) have no matching file and must be dropped
         * as orphans. Before this, deleting files (e.g. the 208 @shuffle
         * .ens) left them unnoticed: "1885 files (cache up to date)" while
         * the index still held 1886 — and worse, the stale entry's members
         * were double-freed later (crash 2026-08-10). */
        while (j < n_idx) {
            free(idx[j].members);
            idx[j].members = NULL;
            n_orphaned++;
            j++;
        }
        free(snap);
        if (!g_stdout) {
            if (n_rebuilt == 0 && n_orphaned == 0)
                printf("  .index: %d files (cache up to date)\n", n_merged);
            else
                printf("  .index: %d files (%d re-read, %d orphaned)\n",
                       n_merged, n_rebuilt, n_orphaned);
        }
        if (n_rebuilt > 0 || n_orphaned > 0)
            index_save(dir, merged, n_merged);
        blocks_from_index(merged, n_merged);
        n_files = n_merged;
        for (int ii = 0; ii < n_merged; ii++) free(merged[ii].members);
        free(merged);
        for (int ii = 0; ii < n_idx; ii++) free(idx[ii].members);
        free(idx);
    }
    return n_files;
}


/* Forward decl — export_member_file is defined with the beam writers below. */
static void export_member_file(const char *path, const uint8_t *set,
                               int n_members, float eval, int beam_width);
static void backup_member_to_dir(const char *dir, const char *path);
/* Forward decl — defined with the --debug-member block below (greedy calls
 * it before the definition site). 2026-08-14. */
static void print_ensemble_confusion(const uint8_t *set, int n_blk);

/* -- Merge and evaluate -------------------------------------- */
static int merge_and_eval(const char *save_path, int max_en)
{
    if (n_blocks == 0) { printf("  No score blocks loaded.\n"); return 0; }
    if (max_en <= 0 || max_en > n_blocks) max_en = n_blocks;
    int has_labels = (g_labels != NULL);
    struct timeval _t0; gettimeofday(&_t0, NULL);

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
    /* Target-class / filter-sample mode display */
    if (g_target_active || g_filter_sample_active) {
        printf("  Mode:");
        if (g_target_active) {
            printf("  target={");
            for (int _t = 0; _t < g_target_count; _t++)
                printf("%s%d", _t ? "," : "", g_target_classes[_t]);
            printf("}");
        }
        if (g_filter_sample_active) {
            printf("  filter-sample={");
            for (int _f = 0; _f < g_filter_sample_count; _f++)
                printf("%s%d", _f ? "," : "", g_filter_sample_classes[_f]);
            printf("}");
        }
        printf("  (denom=%d/%d)\n", g_eval_denom, g_n_test);
    }
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
            if (max_en > 0 && files_merged > max_en) break;
        }

        const SCORE_TYPE *sc = blocks[m].scores;
        for (size_t i = 0; i < score_sz; i++)
            sum_scores[i] += sc[i];

        int correct = 0;
        if (has_labels) {
          #pragma omp parallel for reduction(+:correct)
          for (int s = 0; s < g_n_test; s++) {
              const SCORE_TYPE *row = sum_scores + (size_t)s * row_sz;
              if (eval_sample_correct(row, g_labels[s])) correct++;
          }
        }

        float acc = has_labels ? (float)correct * 100.0f / (float)(eval_denom()) : 0.0f;
        /* Full label "XF:CHAN:ENC" in ONE line (no separate file separator
         * line — one member = one row, see 2026-08-01) */
        char _full[96];
        {
            char label[24] = {'x'};
            member_label(&blocks[m], label, sizeof(label));
            const char *_xn = (blocks[m].xform_real >= 0)
                              ? xform_name_safe(blocks[m].xform_real)
                              : xform_group_label(blocks[m].xform_id);
            snprintf(_full, sizeof(_full), "%s:%s", _xn, label);
        }

        float gain = acc - prev_acc;
        if (acc > best_acc) {
            best_acc = acc;
            best_en = en;
        }
        if (has_labels) {
            printf("  %-4d  %-7.2f  %5d/%-5d  %-+7.2f  %s\n",
                   en, acc, correct, eval_denom(), gain, _full);
        } else {
            printf("  %-4d  (no labels)\n", en);
        }
        if (save_f && has_labels)
            fprintf(save_f, "%d  %.4f  %d  %d  %+.4f\n",
                    en, acc / 100.0f, correct, eval_denom(), gain / 100.0f);
        prev_acc = acc;
    }

    printf("\n══╡ BEST PREFIX ╞═══════════════════════════════════════════\n");
    printf("  Best:  EN=%d  acc=%.2f%%  (%d/%d)\n",
           best_en, best_acc,
           (int)(best_acc * (float)eval_denom() / 100.0f + 0.5f), eval_denom());
    printf("  Final: %d members  acc=%.2f%%  (%d/%d)\n",
           n_blocks, prev_acc,
           (int)(prev_acc * (float)eval_denom() / 100.0f + 0.5f), eval_denom());
    if (save_f) { fclose(save_f); printf("  Saved:  %s\n", save_path); }
    /* --member-out in merge mode: write ALL loaded (post-filter) members as
     * XF:CHAN:ENC specs. Combined with --filter eval this halves the training
     * set — TRN --member-file recomputes only the relevant members. */
    if (g_member_out[0])
        export_member_file(g_member_out, NULL, n_blocks, prev_acc, 0);

    int _bc = (int)(best_acc * (float)eval_denom() / 100.0f + 0.5f);
    struct timeval _t1; gettimeofday(&_t1, NULL);
    double _ms = (double)(_t1.tv_sec - _t0.tv_sec) * 1000.0
               + (double)(_t1.tv_usec - _t0.tv_usec) / 1000.0;
    ki_report_show(0, 0, _bc, g_n_test, (int)_ms, omp_get_max_threads(),
                   g_n_test - _bc, 0.0f, best_en, n_blocks);

    free(sum_scores);
    return g_n_test;
}

/* =============================================================
 * MAIN
 * ============================================================= */
/* ═══════════════════════════════════════════════════════════════════════
 * GREEDY — Greedy optimal subset search
 * ═══════════════════════════════════════════════════════════════════════
 * Starts with empty ensemble. In each step, tries every unused member,
 * picks the one that improves eval the most. Stops when no member helps.
 */
static int merge_and_greedy(const char *save_path, int initial_member, int beam_width)
{
    if (n_blocks == 0) { printf("  No score blocks loaded.\n"); return 0; }
    int has_labels = (g_labels != NULL);
    int n = n_blocks;
    struct timeval _t0; gettimeofday(&_t0, NULL);

    /* --max cumulative window: beam_width is the window length. When the
     * greedy runs without --beam the caller passes 0 → fall back to the
     * beam default (10), matching the beam's window semantics. */
    if (beam_width < 1) beam_width = 10;

    size_t score_sz = (size_t)g_n_test * (size_t)g_n_classes;
    size_t row_sz   = (size_t)g_n_classes;

    /* Cumulative sum of selected members' scores */
    SCORE_TYPE *sum_scores = (SCORE_TYPE *)calloc(score_sz, sizeof(SCORE_TYPE));
    if (!sum_scores) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }

    /* Track used members */
    int *used = (int *)calloc((size_t)n, sizeof(int));
    if (!used) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }

    /* --member-seed: pre-seed the chain with this member (same semantics as
     * the beam). It is committed FIRST (EN=1) — the greedy scan then starts
     * from its accumulated scores. Added 2026-08-03 (user request). */
    if (initial_member >= 0 && initial_member < n) {
        used[initial_member] = 1;
        const SCORE_TYPE *sc0 = blocks[initial_member].scores;
        for (size_t i = 0; i < score_sz; i++)
            sum_scores[i] += sc0[i];
    }

    printf("\n");
    printf("== GREEDY ENSEMBLE ============================================\n");
    printf("  %d score blocks (%d test samples, %d classes)\n",
           n, g_n_test, g_n_classes);
    printf("  Config: H=%d  EP=%d  VN=%d  HN=%d  TE=%d\n",
           g_hidden, g_epochs, g_split_vn, g_split_hn, g_target_err_x100);
    printf("  Algorithm: greedy (optimal subset, O(N²))\n");
    printf("  min-gain:   %.2f%%%s\n",
           g_min_gain,
           g_max_mode ? "  (--max: cumulative min-gain)" : "");
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
    /* INTENTIONAL: dynamic allocation — greedy can select up to n members;
     * the old fixed order[1024] overflowed the stack for pools > 1024
     * (SIGSEGV after the final report, see bug report 2026-08-03). */
    int *order = (int *)malloc((size_t)n * sizeof(int));
    if (!order) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
    int order_n = 0;

    /* --max cumulative acceptance (2026-08-07): sub-threshold members are
     * buffered here and committed as ONE block once the accumulated gain
     * since the last commit reaches min-gain. Size = the window (max
     * beam_width pending steps). Members stay in pending[] (not order[])
     * until the block commits; if the window closes without a commit, the
     * pending buffer is discarded. */
    int *pending = (int *)malloc((size_t)(g_max_mode ? (beam_width > 0 ? beam_width : 1) : 1) * sizeof(int));
    if (!pending) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
    int pending_n = 0;
    int no_improve_steps = 0;  /* --max window counter: steps since last commit */

    /* Seed member as EN=1 (--member-seed): committed first, shown as the
     * initial row. prev_acc/best_acc start from its single-member eval. */
    if (initial_member >= 0 && initial_member < n) {
        int _c = 0;
        for (int s = 0; s < g_n_test && has_labels; s++) {
            const SCORE_TYPE *row = sum_scores + (size_t)s * row_sz;
            if (eval_sample_correct(row, g_labels[s])) _c++;
        }
        float sev = has_labels ? (float)_c * 100.0f / (float)eval_denom() : 0.0f;
        order[order_n++] = initial_member;
        char _lbl[256];
        const char *_xn = xform_name_safe(blocks[initial_member].xform_real);
        char _ml[256];
        member_label(&blocks[initial_member], _ml, sizeof(_ml));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
        snprintf(_lbl, sizeof(_lbl), "%s:%s", _xn, _ml);
#pragma GCC diagnostic pop
        if (has_labels) {
            printf("  %-4d  %-7.2f  %5d/%-5d  %-+7.2f  %s\n",
                   1, sev, _c, eval_denom(), sev, _lbl);
        }
        prev_acc = sev;
        best_acc = sev;
        best_en = 1;
    }

    for (int step = 0; step < n; step++) {
        /* --max N: Stopp bei N Membern */
        if (g_max_members > 0 && order_n >= g_max_members) break;

        /* Try every unused member — parallel über Member.
         * INTENTIONAL: per-thread local best + ONE global critical compare
         * instead of reduction(max)+critical mixing. The old pattern
         *   #pragma omp parallel for reduction(max:best_step_acc)
         *   ... if (acc > best_step_acc) { critical { best_step_acc=acc; best_m=m; } }
         * had a race: best_step_acc is a reduction variable (per-thread private
         * copy), so two threads could BOTH enter the critical block and the
         * LAST writer won — best_m did not match the maximum acc. Result:
         * --greedy picked the wrong first member (39.34% instead of 40.20%
         * on M1.add, bug report 2026-08-03). With local bests the global
         * critical compare sees the true maximum → best_m is always the index
         * of the highest acc. */
        float best_step_acc = -1.0f;
        double best_clr = -1.0;   /* --greedy-clarity: best clarity found */
        int best_m = -1;
        /* --greedy-clarity: among candidates that do NOT worsen the current
         * ensemble (acc >= prev_acc), pick the one with the highest member
         * clarity instead of the best acc — prefer unambiguous voters that
         * disturb other members the least (plan-2026-08-08). prev_acc is
         * firstprivate (read-only inside the parallel region). */
        float _cl_thresh = g_greedy_clarity ? prev_acc : -1.0f;
        #pragma omp parallel
        {
            float local_best = -1.0f;
            double local_clr = -1.0;
            int local_m = -1;
            #pragma omp for schedule(static)
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
                        /* target/filter-sample: membership decision on the
                         * CANDIDATE prediction (base+sc), not committed. */
                        if (eval_sample_correct_pred(pred, g_labels[s])) correct++;
                    }
                }
                float acc = has_labels ? (float)correct * 100.0f / (float)(eval_denom()) : 0.0f;

                if (g_greedy_clarity) {
                    /* no-worsen constraint + max clarity */
                    if (acc >= _cl_thresh - 0.0001f &&
                        blocks[m].clarity > local_clr) {
                        local_clr = blocks[m].clarity;
                        local_best = acc;
                        local_m = m;
                    }
                } else if (acc > local_best) { local_best = acc; local_m = m; }
            }
            #pragma omp critical
            {
                if (g_greedy_clarity) {
                    if (local_clr > best_clr) {
                        best_clr = local_clr;
                        best_step_acc = local_best;  /* keep the acc for min-gain */
                        best_m = local_m;
                    }
                } else if (local_best > best_step_acc) {
                    best_step_acc = local_best;
                    best_m = local_m;
                }
            }
        }

        if (best_m < 0) break;  /* no candidate (should not happen) */

        /* ── min-gain as COMMIT FILTER with stop ───────────────────────
         * The user decision (2026-08-03): a member is only COMMITTED
         * (counted into order[]/sum_scores) when it improves the current
         * ensemble by more than min-gain. best_m is the BEST remaining
         * candidate — if it does not beat min-gain, no other remaining
         * member can either, so the search stops right here (no wasted
         * scans of the rest). This drops harmful members from the result.
         *
         * --max (cumulative acceptance, 2026-08-07): the per-member commit
         * threshold is 0.0 — EVERY improvement buffers the member in
         * pending[] and advances sum_scores; the whole pending block is
         * committed to order[] once the ACCUMULATED gain since the last
         * commit (best_step_acc - prev_acc) reaches min-gain. Stop only
         * after beam_width consecutive steps whose accumulated sum stayed
         * below min-gain. Greedy (single path) gets the same cumulative
         * semantics as the beam (plan 2026-08-07). */
        float gain_threshold = g_min_gain;
        float gain = best_step_acc - prev_acc;

        if (g_max_mode && gain > 0.0f) {
            /* --max: buffer the member; commit the block when the sum of
             * the buffered gains (gain = best_step_acc - prev_acc is
             * ALREADY cumulative, prev_acc only updates at a commit)
             * STRICTLY EXCEEDS min-gain (consistency with the beam and the
             * strict mode: `>` not `>=`, 2026-08-08). */
            used[best_m] = 1;
            pending[pending_n++] = best_m;
            const SCORE_TYPE *sc = blocks[best_m].scores;
            for (size_t i = 0; i < score_sz; i++)
                sum_scores[i] += sc[i];

            if (gain > gain_threshold) {
                /* ── COMMIT the pending block ──
                 * --max N (FIX 2026-08-10): cap the block at the remaining
                 * slots — the whole block must not overshoot g_max_members.
                 * The overshoot happened because the outer loop checks
                 * order_n >= g_max_members only BETWEEN steps, while a
                 * cumulative block can commit many members at once.
                 * Members beyond the cap are rolled back from sum_scores
                 * (they stay used[] — consumed, not retried). */
                int _commit_n = pending_n;
                if (g_max_members > 0 && order_n + _commit_n > g_max_members) {
                    _commit_n = g_max_members - order_n;
                    for (int _p = _commit_n; _p < pending_n; _p++) {
                        const SCORE_TYPE *_sc = blocks[pending[_p]].scores;
                        for (size_t _i = 0; _i < score_sz; _i++)
                            sum_scores[_i] -= _sc[_i];
                    }
                }
                for (int _p = 0; _p < _commit_n; _p++)
                    order[order_n++] = pending[_p];

                /* Evaluate the accumulated sum (whole block) */
                int correct = 0;
                for (int s = 0; s < g_n_test && has_labels; s++) {
                    const SCORE_TYPE *row = sum_scores + (size_t)s * row_sz;
                    if (eval_sample_correct(row, g_labels[s])) correct++;
                }
                float acc = has_labels ? (float)correct * 100.0f / (float)(eval_denom()) : 0.0f;

                /* HONEST BLOCK LABEL (2026-08-12): a --max commit is a BLOCK
                 * of _commit_n members, not a single one. Showing only the
                 * last member (old behavior) made blocks with N>1 look like
                 * a single member — and could show the SAME name twice when
                 * two blocks ended on the same member (EN=28/29 artefact).
                 * Now the label states the block size explicitly. */
                char label[256];
                int _blk_last = pending[pending_n - 1];
                const char *_xn = xform_name_safe(blocks[_blk_last].xform_real);
                char _ml[256];
                member_label(&blocks[_blk_last], _ml, sizeof(_ml));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
                if (_commit_n > 1)
                    snprintf(label, sizeof(label), "(block +%d, last: %s:%s)",
                             _commit_n, _xn, _ml);
                else
                    snprintf(label, sizeof(label), "%s:%s", _xn, _ml);
#pragma GCC diagnostic pop
                float gain2 = acc - prev_acc;   /* accumulated gain of the block */

                if (has_labels) {
                    printf("  %-4d  %-7.2f  %5d/%-5d  %-+7.2f  %s\n",
                           order_n, acc, correct, eval_denom(), gain2, label);
                }
                if (save_f && has_labels)
                    fprintf(save_f, "%d  %.4f  %d  %d  %+.4f  %s\n",
                            order_n, acc / 100.0f, correct, eval_denom(), gain2 / 100.0f, label);

                if (acc > best_acc) {
                    best_acc = acc;
                    best_en = order_n;   /* count of COMMITTED members */
                }

                prev_acc = acc;
                pending_n = 0;
                no_improve_steps = 0;
            } else {
                /* Accumulated gain still below min-gain — keep buffering.
                 * Window counter counts EVERY step since the last commit;
                 * stop only when beam_width steps stayed below min-gain. */
                no_improve_steps++;
                if (no_improve_steps >= beam_width) {
                    printf("\n  [Early stop] accumulated gain %.2f%% stayed below "
                           "min-gain %.2f%% for %d steps — stop\n",
                           gain, gain_threshold, no_improve_steps);
                    break;
                }
            }
        } else if (gain > gain_threshold) {
        /* Commit this member (strict mode, unchanged) */
        used[best_m] = 1;
        order[order_n++] = best_m;
        const SCORE_TYPE *sc = blocks[best_m].scores;
        for (size_t i = 0; i < score_sz; i++)
            sum_scores[i] += sc[i];

        /* Evaluate final cumulative sum for this step */
        int correct = 0;
        for (int s = 0; s < g_n_test && has_labels; s++) {
            const SCORE_TYPE *row = sum_scores + (size_t)s * row_sz;
            if (eval_sample_correct(row, g_labels[s])) correct++;
        }
        float acc = has_labels ? (float)correct * 100.0f / (float)(eval_denom()) : 0.0f;

                        char label[256];
        const char *_xn = xform_name_safe(blocks[best_m].xform_real);
        char _ml[256];
        member_label(&blocks[best_m], _ml, sizeof(_ml));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
        snprintf(label, sizeof(label), "%s:%s", _xn, _ml);
#pragma GCC diagnostic pop
        float gain2 = acc - prev_acc;

        if (has_labels) {
            printf("  %-4d  %-7.2f  %5d/%-5d  %-+7.2f  %-s\n",
                   order_n + 1, acc, correct, eval_denom(), gain2, label);
        }
        if (save_f && has_labels)
            fprintf(save_f, "%d  %.4f  %d  %d  %+.4f  %s\n",
                    order_n + 1, acc / 100.0f, correct, eval_denom(), gain2 / 100.0f, label);

        if (acc > best_acc) {
            best_acc = acc;
            best_en = order_n;   /* count of COMMITTED members (steps that
                                    were discarded by min-gain don't count) */
        }

        prev_acc = acc;
        } else if (g_max_mode) {
            /* --max: gain <= 0 (no candidate improves the accumulated sum).
             * Tolerate dry steps like the beam — only stop after beam_width
             * consecutive steps below min-gain (consistency fix 2026-08-08:
             * the old code hit the `else break` immediately, cutting off the
             * --max tolerance window). */
            no_improve_steps++;
            if (no_improve_steps >= beam_width) {
                printf("\n  [Early stop] accumulated gain %.2f%% stayed below "
                       "min-gain %.2f%% for %d steps — stop\n",
                       gain, gain_threshold, no_improve_steps);
                break;
            }
        } else {
            /* INTENTIONAL: stop on first non-commit. best_m is the BEST
             * remaining member — if it fails to beat min-gain, every other
             * remaining member is worse and can never be committed either.
             * Scanning the rest would be pure waste (48.6k × full scan ≈
             * hours for zero result; user decision 2026-08-03). Result is
             * identical to the old discard-and-scan: only members with
             * gain > min-gain are committed. */
            break;
        }
    }

    int best_corr = (int)(best_acc * (float)eval_denom() / 100.0f + 0.5f);
    printf("\n══╡ OPTIMAL SUBSET ╞════════════════════════════════════════\n");
    printf("  Best:   EN=%d  acc=%.2f%%  (%d/%d)\n",
           best_en, best_acc, best_corr, eval_denom());
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
           best_acc, eval_denom() - best_corr, eval_denom(), best_en);

    struct timeval _t1; gettimeofday(&_t1, NULL);
    double _ms = (double)(_t1.tv_sec - _t0.tv_sec) * 1000.0
               + (double)(_t1.tv_usec - _t0.tv_usec) / 1000.0;
    /* lr= carries --beam-bestN (unused by merge-ensemble) for reproducibility */
    ki_report_show(0, 0, best_corr, eval_denom(), (int)_ms, omp_get_max_threads(),
                   eval_denom() - best_corr, (float)g_beam_bestN, best_en, n_blocks);
    if (save_f) { fclose(save_f); printf("  Saved:  %s\n", save_path); }
    /* --member-out in greedy mode: export the BEST prefix only (order[0..
     * best_en-1]) — same semantics as beam, which exports g_best_used up to
     * the best eval. Greedy walks the full pool (order_n == n), but the
     * winning subset is the first best_en members; exporting all would make
     * member.out == the whole pool and break the REPORT==retrain guarantee. */
    if (g_member_out[0] && best_en > 0) {
        uint8_t *greedy_set = (uint8_t *)calloc((size_t)n, 1);
        if (!greedy_set) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
        for (int i = 0; i < best_en; i++)
            greedy_set[order[i]] = 1;
        export_member_file(g_member_out, greedy_set, best_en, best_acc, 0);
        free(greedy_set);
    }
    /* --debug-confusion: confusion matrix of the winning greedy subset
     * (order[0..best_en-1] — same set as member-out/REPORT). */
    if (g_debug_confusion && best_en > 0) {
        uint8_t *greedy_cm = (uint8_t *)calloc((size_t)n, 1);
        if (!greedy_cm) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
        for (int i = 0; i < best_en; i++)
            greedy_cm[order[i]] = 1;
        print_ensemble_confusion(greedy_cm, n);
        free(greedy_cm);
    }
    free(sum_scores); free(used); free(order); free(pending);
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
    if (write < n_blocks && !g_stdout)
        printf("  [DEDUP] removed %d duplicate(s) (%d → %d)\n",
               n_blocks - write, n_blocks, write);
    n_blocks = write;
}

/* ── Beam/Expansion types (file scope für qsort comparator) ── */
typedef struct {
    SCORE_TYPE *sum; /* [score_sz] cumulative — SCORE_TYPE is double in FLT
                        mode (order-independent accumulation, bug 2026-08-02)
                        and int64_t in INT32 mode. */
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
/* ═══════════════════════════════════════════════════════════════════
 * backup_member_to_dir — timestamped copy of a member file into DIR/.member/
 * ═══════════════════════════════════════════════════════════════════
 * Every --member-out result is additionally stored as DIR/.member/YYYY-MM-DD_HH-MM-SS.member
 * so old runs can be reproduced later (e.g. the 2026-07-28 69.6% run whose
 * member file was not kept). The backup lives in the SCORES directory (.member
 * subdir), independent of where --member-out points. Uses mkdir() + byte copy
 * (no system()), no-op when path is empty or the source cannot be opened. */
static void backup_member_to_dir(const char *dir, const char *path) {
    if (!dir || !dir[0] || !path || !path[0]) return;
    FILE *src = fopen(path, "rb");
    if (!src) return;
    /* Ensure DIR/.member exists */
    char subdir[1024];
    snprintf(subdir, sizeof(subdir), "%s/.member", dir);
    if (mkdir(subdir, 0755) != 0 && errno != EEXIST) {
        fclose(src);
        fprintf(stderr, "  [WARN] cannot create %s: %s\n", subdir, strerror(errno));
        return;
    }
    /* Timestamp: YYYY-MM-DD_HH-MM-SS */
    struct timeval tv; gettimeofday(&tv, NULL);
    struct tm tmv; localtime_r(&tv.tv_sec, &tmv);
    char stamp[64];
    snprintf(stamp, sizeof(stamp), "%04d-%02d-%02d_%02d-%02d-%02d",
             tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday,
             tmv.tm_hour, tmv.tm_min, tmv.tm_sec);
    char dst[1100];
    snprintf(dst, sizeof(dst), "%s/%s.member", subdir, stamp);
    FILE *out = fopen(dst, "wb");
    if (!out) {
        fclose(src);
        fprintf(stderr, "  [WARN] cannot write backup %s: %s\n", dst, strerror(errno));
        return;
    }
    char buf[65536]; size_t r;
    while ((r = fread(buf, 1, sizeof(buf), src)) > 0)
        fwrite(buf, 1, r, out);
    fclose(out); fclose(src);
    printf("  [BACKUP] %s\n", dst);
}

/* ═══════════════════════════════════════════════════════════════════
 * build_sample_index — --sample-index FILE (2026-08-14, 3rd gen 08-15)
 *
 * Writes the member↔sample prediction table of ALL loaded (post-filter)
 * blocks as THREE Parquet files (columnar, compressed, queryable
 * directly by DuckDB / pandas / pyarrow — no DB engine, no import):
 *
 *   FILE.samples.parquet  (sample_id INT32, label INT32)
 *     ground-truth labels of all test samples.
 *
 *   FILE.members.parquet  (member_id INT64, name VARCHAR)
 *     member_id = FNV-1a 64-bit over the .ens basename (stable, name-bound)
 *     name      = the spec ("colswap-1-4@rot112:lbp:tri8")
 *
 *   FILE.pairs.parquet    (member_id INT64, sample_id INT32,
 *                          pred INT32, margin INT64)
 *     EVERY assignment member→sample, with the member's argmax CLASS (pred)
 *     and the decision margin (best−second). The pred column is the key
 *     for the Shirt (class 6) analysis: it stores WRONG predictions too,
 *     so "which members see Shirt-sample X as T-shirt/Pullover/Coat" is
 *     directly answerable (the confusion question) — a correct-only table
 *     cannot.
 *
 * Example queries (DuckDB):
 *   1) which members recognize sample X (and how)?
 *        SELECT m.name, p.pred, p.margin FROM read_parquet('FILE.pairs.parquet') p
 *        JOIN read_parquet('FILE.members.parquet') m ON m.member_id=p.member_id
 *        WHERE p.sample_id=X ORDER BY p.margin DESC;
 *   2) which samples does member Y see as class k?
 *        SELECT sample_id, margin FROM read_parquet('FILE.pairs.parquet')
 *        WHERE member_id=Y AND pred=k;
 *   3) which members see Shirt-samples as Shirt (the specialists)?
 *        SELECT m.name, COUNT(*) FROM read_parquet('FILE.pairs.parquet') p
 *        JOIN read_parquet('FILE.members.parquet') m ON m.member_id=p.member_id
 *        JOIN read_parquet('FILE.samples.parquet') s ON s.sample_id=p.sample_id
 *        WHERE s.label=6 AND p.pred=6 GROUP BY p.member_id ORDER BY 2 DESC;
 *   4) the confusion: which members see Shirt-samples as T-shirt?
 *        ... WHERE s.label=6 AND p.pred=0 ...;
 *
 * DESIGN (3rd gen, 2026-08-15): full pair table (variant B), ALL
 * assignments (not only correct), pred + margin stored.
 *
 * PERFORMANCE (2026-08-15): the original SQLite implementation was
 * structurally slow — the pairs table was WITHOUT ROWID with
 * PK(member_id,sample_id), so EVERY insert went through the PK B-Tree
 * (random page jumps with FNV-1a member_ids), and the two deferred
 * CREATE INDEX passes (idx_pairs_sample, idx_pairs_pred) were
 * single-threaded full scans. Three serial single-thread passes over
 * 368M rows. Parquet removes all of it:
 *   1. member_ids[] is precomputed once and the block indices are sorted
 *      by member_id → pairs stream out in (member_id, sample_id) order.
 *   2. carquet (pure C11 Parquet writer, built locally under 3rdparty/)
 *      writes the sorted columns with zstd compression — a single
 *      sequential columnar pass, no B-Tree, no index build.
 *   3. Each block's scores are freed after its chunk is written (scores
 *      are read once only) → peak RAM stays below the score-load peak.
 *   4. Member-id collision check is one sort+scan (O(n log n) instead of
 *      the old O(n²) re-hashing; collisions are practically impossible
 *      at 2^64).
 * The argmax computation is chunk-parallel (OMP, implicit barrier per
 * chunk); the carquet writer is called serially by the main thread.
 * NOTE: carquet API uses CARQUET_WARN_UNUSED_RESULT — every call's
 * return value is checked (build has -Werror). */

/* qsort helper for build_sample_index: sort (member_id, block-index) pairs
 * by member_id (stable tiebreak on block index). This gives the blocks in
 * member_id order, so the pairs stream out in (member_id, sample_id) order
 * — sequential B-Tree append into the WITHOUT ROWID PK (2026-08-15). */
typedef struct { uint64_t id; int block; } MidSortEnt;
static int cmp_mid_sort(const void *a, const void *b) {
    const MidSortEnt *ea = (const MidSortEnt *)a, *eb = (const MidSortEnt *)b;
    if (ea->id < eb->id) return -1;
    if (ea->id > eb->id) return 1;
    return ea->block - eb->block;   /* stable tiebreak (same member_id) */
}

static int build_sample_index(const char *path) {
    if (!path || !path[0] || !g_labels || g_n_test <= 0 || g_n_classes <= 0)
        return 0;

    /* ── Member-IDs einmal vorberechnen + Blocks nach member_id sortieren ──
     * (2026-08-15, 3rd gen) The pairs Parquet is written in
     * (member_id, sample_id) order — sorting the block indices by
     * member_id ONCE here makes Pass 2 stream the pairs in exactly that
     * order (ordered columnar pages, good compression). The member-id
     * collision check (O(n²) in the old SQLite members pass) becomes one
     * sort+scan (O(n log n)); collisions are practically impossible at
     * 2^64. */
    MidSortEnt *mid_sorted = (MidSortEnt *)malloc((size_t)n_blocks * sizeof(MidSortEnt));
    uint64_t *member_ids = (uint64_t *)malloc((size_t)n_blocks * sizeof(uint64_t));
    int *block_order = (int *)malloc((size_t)n_blocks * sizeof(int));
    if (!mid_sorted || !member_ids || !block_order) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
    for (int i = 0; i < n_blocks; i++) {
        mid_sorted[i].id = member_id_fnv1a(blocks[i].source_file);
        mid_sorted[i].block = i;
    }
    qsort(mid_sorted, (size_t)n_blocks, sizeof(MidSortEnt), cmp_mid_sort);
    /* Kollisionen auflösen: benachbarte gleiche IDs (sortiert → einmal scannen).
     * Praktisch nie, aber der PK erzwingt Eindeutigkeit — prüfen statt raten.
     * Suffix "_2"/"_2_"... wird angehängt, bis der Hash frei ist. */
    for (int i = 1; i < n_blocks; i++) {
        if (mid_sorted[i].id != mid_sorted[i - 1].id) continue;
        const char *name = blocks[mid_sorted[i].block].source_file;
        char buf[192];
        snprintf(buf, sizeof(buf), "%s_2", name);
        while (1) {
            uint64_t id2 = member_id_fnv1a(buf);
            int dup = 0;
            for (int j = 0; j < i; j++)
                if (mid_sorted[j].id == id2 || strcmp(buf, blocks[mid_sorted[j].block].source_file) == 0)
                    { dup = 1; break; }
            if (!dup) { mid_sorted[i].id = id2; break; }
            if (strlen(buf) >= sizeof(buf) - 2) break;   /* Überlaufschutz */
            strncat(buf, "_", sizeof(buf) - strlen(buf) - 1);
        }
        fprintf(stderr, "  [WARN] --sample-index: member-id collision for '%s' — used suffix\n", name);
    }
    for (int i = 0; i < n_blocks; i++) {
        block_order[i] = mid_sorted[i].block;
        member_ids[mid_sorted[i].block] = mid_sorted[i].id;
    }
    free(mid_sorted);

    /* ── Parquet-Writer-Setup (2026-08-15, 3rd gen) ──
     * Three files derived from the --sample-index path:
     *   FILE.samples.parquet, FILE.members.parquet, FILE.pairs.parquet
     * carquet (pure C11, local build in 3rdparty/) writes columnar,
     * zstd-compressed Parquet — one sequential pass, no B-Tree, no
     * index build (SQLite was 3 serial single-thread passes; see DESIGN
     * comment above). */
    char p_samples[1100], p_members[1100], p_pairs[1100];
    snprintf(p_samples, sizeof(p_samples), "%s.samples.parquet", path);
    snprintf(p_members, sizeof(p_members), "%s.members.parquet", path);
    snprintf(p_pairs,   sizeof(p_pairs),   "%s.pairs.parquet",   path);
    /* Remove stale artifacts from a previous run (rebuildable). */
    unlink(p_samples); unlink(p_members); unlink(p_pairs);

    carquet_logical_type_t string_type = { .id = CARQUET_LOGICAL_STRING };
    carquet_error_t cerr = CARQUET_ERROR_INIT;
    carquet_status_t st;
    int rc = 0;

    /* ── Pass 0: samples.parquet (sample_id INT32, label INT32) ── */
    {
        carquet_schema_t *sc = carquet_schema_create(&cerr);
        if (!sc) { fprintf(stderr, "  [ERROR] --sample-index: samples schema: %s\n", cerr.message); return 1; }
        st = carquet_schema_add_column(sc, "sample_id", CARQUET_PHYSICAL_INT32, NULL,
                                       CARQUET_REPETITION_REQUIRED, 0, 0);
        st = carquet_schema_add_column(sc, "label", CARQUET_PHYSICAL_INT32, NULL,
                                       CARQUET_REPETITION_REQUIRED, 0, 0);
        if (st != CARQUET_OK) { fprintf(stderr, "  [ERROR] --sample-index: samples schema col: %s\n", cerr.message); carquet_schema_free(sc); return 1; }
        carquet_writer_options_t opts;
        carquet_writer_options_init(&opts);
        opts.compression = CARQUET_COMPRESSION_ZSTD;
        carquet_writer_t *w = carquet_writer_create(p_samples, sc, &opts, &cerr);
        if (!w) { fprintf(stderr, "  [ERROR] --sample-index: create %s: %s\n", p_samples, cerr.message); carquet_schema_free(sc); return 1; }
        int32_t *sid = (int32_t *)malloc((size_t)g_n_test * sizeof(int32_t));
        int32_t *lbl = (int32_t *)malloc((size_t)g_n_test * sizeof(int32_t));
        if (!sid || !lbl) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
        for (int s = 0; s < g_n_test; s++) { sid[s] = s; lbl[s] = (int32_t)g_labels[s]; }
        st = carquet_writer_write_batch(w, 0, sid, (int64_t)g_n_test, NULL, NULL);
        st = carquet_writer_write_batch(w, 1, lbl, (int64_t)g_n_test, NULL, NULL);
        if (st != CARQUET_OK) fprintf(stderr, "  [ERROR] --sample-index: samples write: %s\n", cerr.message);
        st = carquet_writer_close(w);
        if (st != CARQUET_OK) { fprintf(stderr, "  [ERROR] --sample-index: samples close: %s\n", cerr.message); rc = 1; }
        free(sid); free(lbl);
        carquet_schema_free(sc);
        if (rc) return 1;
        printf("  [sample-index] samples: %d labels written\n", g_n_test);
    }

    /* ── Pass 1: members.parquet (member_id INT64, name VARCHAR) ──
     * IDs come from the precomputed member_ids[] (collision resolution
     * already ran once, O(n log n)); order follows block_order for
     * determinism — same IDs as in Pass 2. */
    {
        carquet_schema_t *sc = carquet_schema_create(&cerr);
        if (!sc) { fprintf(stderr, "  [ERROR] --sample-index: members schema: %s\n", cerr.message); return 1; }
        st = carquet_schema_add_column(sc, "member_id", CARQUET_PHYSICAL_INT64, NULL,
                                       CARQUET_REPETITION_REQUIRED, 0, 0);
        st = carquet_schema_add_column(sc, "name", CARQUET_PHYSICAL_BYTE_ARRAY, &string_type,
                                       CARQUET_REPETITION_REQUIRED, 0, 0);
        if (st != CARQUET_OK) { fprintf(stderr, "  [ERROR] --sample-index: members schema col: %s\n", cerr.message); carquet_schema_free(sc); return 1; }
        carquet_writer_options_t opts;
        carquet_writer_options_init(&opts);
        opts.compression = CARQUET_COMPRESSION_ZSTD;
        carquet_writer_t *w = carquet_writer_create(p_members, sc, &opts, &cerr);
        if (!w) { fprintf(stderr, "  [ERROR] --sample-index: create %s: %s\n", p_members, cerr.message); carquet_schema_free(sc); return 1; }
        int64_t *mid = (int64_t *)malloc((size_t)n_blocks * sizeof(int64_t));
        carquet_byte_array_t *names = (carquet_byte_array_t *)calloc((size_t)n_blocks, sizeof(carquet_byte_array_t));
        if (!mid || !names) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
        for (int i = 0; i < n_blocks; i++) {
            int b = block_order[i];
            mid[i] = (int64_t)member_ids[b];
            names[i].data = (uint8_t *)(uintptr_t)blocks[b].source_file;
            names[i].length = (int32_t)strlen(blocks[b].source_file);
        }
        st = carquet_writer_write_batch(w, 0, mid, (int64_t)n_blocks, NULL, NULL);
        st = carquet_writer_write_batch(w, 1, names, (int64_t)n_blocks, NULL, NULL);
        if (st != CARQUET_OK) fprintf(stderr, "  [ERROR] --sample-index: members write: %s\n", cerr.message);
        st = carquet_writer_close(w);
        if (st != CARQUET_OK) { fprintf(stderr, "  [ERROR] --sample-index: members close: %s\n", cerr.message); rc = 1; }
        free(mid); free(names);
        carquet_schema_free(sc);
        if (rc) return 1;
        printf("  [sample-index] members: %d names written\n", n_blocks);
    }

    /* ── Pass 2: pairs.parquet — parallel over members ──
     * IMPORTANT (2026-08-15): schedule(static) — OpenMP assigns the
     * iterations in ascending thread-number order in consecutive blocks
     * (thread 0 = smallest member_ids). Since the blocks run through
     * block_order[] (sorted by member_id) and the thread buffers are
     * written in thread order, the pairs stream out globally sorted by
     * (member_id, sample_id) → perfectly ordered columnar pages.
     * schedule(dynamic) would destroy the ordering (random assignment). */
    size_t row_sz = (size_t)g_n_classes;
    int n_threads = omp_get_max_threads();
    if (n_threads < 1) n_threads = 1;
    if (n_threads > 64) n_threads = 64;
    long total_rows = 0;
    int done_blocks = 0;
    int dot_interval = n_blocks / 40;
    if (dot_interval < 1) dot_interval = 1;
    int fail = 0;   /* shared error flag (omp atomic writes) */
    printf("  [sample-index] pairs (OMP %d threads): ", n_threads);
    fflush(stdout);

    /* Column buffers per thread slot: each thread fills slot [tid]
     * (disjoint member ranges), then the main thread writes the slots
     * in thread order via carquet (single writer — carquet is not
     * thread-safe for writing). */
    typedef struct { int64_t mid; int32_t sid; int32_t pred; int64_t mg; } _Trip;
    /* PERFORMANCE (2026-08-15): SI_CHUNK=32 → 1152 chunks with per-chunk
     * overhead; SI_CHUNK=512 → 72 chunks, more work per omp region. */
    enum { SI_CHUNK = 512 };
    size_t slot_sz = (size_t)SI_CHUNK * (size_t)g_n_test;
    _Trip *chunk_bufs = (_Trip *)malloc((size_t)n_threads * slot_sz * sizeof(_Trip));
    int   *chunk_cnt  = (int *)calloc((size_t)n_threads, sizeof(int));
    if (!chunk_bufs || !chunk_cnt) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }

    /* ── pairs writer ── */
    carquet_schema_t *p_sc = carquet_schema_create(&cerr);
    if (!p_sc) { fprintf(stderr, "  [ERROR] --sample-index: pairs schema: %s\n", cerr.message); return 1; }
    st = carquet_schema_add_column(p_sc, "member_id", CARQUET_PHYSICAL_INT64, NULL,
                                   CARQUET_REPETITION_REQUIRED, 0, 0);
    st = carquet_schema_add_column(p_sc, "sample_id", CARQUET_PHYSICAL_INT32, NULL,
                                   CARQUET_REPETITION_REQUIRED, 0, 0);
    st = carquet_schema_add_column(p_sc, "pred", CARQUET_PHYSICAL_INT32, NULL,
                                   CARQUET_REPETITION_REQUIRED, 0, 0);
    st = carquet_schema_add_column(p_sc, "margin", CARQUET_PHYSICAL_INT64, NULL,
                                   CARQUET_REPETITION_REQUIRED, 0, 0);
    if (st != CARQUET_OK) { fprintf(stderr, "  [ERROR] --sample-index: pairs schema col: %s\n", cerr.message); carquet_schema_free(p_sc); return 1; }
    carquet_writer_options_t p_opts;
    carquet_writer_options_init(&p_opts);
    p_opts.compression = CARQUET_COMPRESSION_ZSTD;
    carquet_writer_t *p_w = carquet_writer_create(p_pairs, p_sc, &p_opts, &cerr);
    if (!p_w) { fprintf(stderr, "  [ERROR] --sample-index: create %s: %s\n", p_pairs, cerr.message); carquet_schema_free(p_sc); return 1; }

    for (int cstart = 0; cstart < n_blocks; cstart += SI_CHUNK) {
        int cend = cstart + SI_CHUNK;
        if (cend > n_blocks) cend = n_blocks;
        memset(chunk_cnt, 0, (size_t)n_threads * sizeof(int));
        #pragma omp parallel for num_threads(n_threads) schedule(static)
        for (int m = cstart; m < cend; m++) {
            int tid = omp_get_thread_num();
            _Trip *mybuf = chunk_bufs + (size_t)tid * slot_sz;
            int b = block_order[m];   /* nach member_id sortiert (2026-08-15) */
            const SCORE_TYPE *sc = blocks[b].scores;
            if (!sc) continue;
            uint64_t mid = member_ids[b];   /* vorberechnet, kein Re-Hash */
            int n = chunk_cnt[tid];   /* thread-lokal fortsetzen */
            for (int s = 0; s < g_n_test; s++) {
                const SCORE_TYPE *row = sc + (size_t)s * row_sz;
                int best = 0, second = -1;
                SCORE_TYPE best_v = row[0], second_v = 0;
                for (int k = 1; k < g_n_classes; k++) {
                    if (row[k] > best_v) { second = best; second_v = best_v; best = k; best_v = row[k]; }
                    else if (second < 0 || row[k] > second_v) { second = k; second_v = row[k]; }
                }
                /* 2nd gen (2026-08-15): store EVERY assignment with the
                 * member's argmax CLASS (pred). The Shirt analysis needs the
                 * WRONG predictions too: "which members see Shirt-sample X
                 * as T-shirt/Pullover/Coat" is the confusion question — a
                 * correct-only table cannot answer it. margin = best−second. */
                mybuf[n].mid  = (int64_t)mid;
                mybuf[n].sid  = (int32_t)s;
                mybuf[n].pred = (int32_t)best;
                mybuf[n].mg   = (int64_t)(best_v - second_v);
                n++;
            }
            chunk_cnt[tid] = n;
        }
        /* seriell (Haupt-Thread): Chunk-Puffer aller Threads in
         * Thread-Reihenfolge via carquet schreiben (Spalte für Spalte) —
         * carquet wants one contiguous array per column, so the trip
         * buffers are transposed into four column arrays first. */
        {
            int64_t chunk_n = 0;
            for (int t = 0; t < n_threads; t++) chunk_n += (int64_t)chunk_cnt[t];
            if (chunk_n > 0) {
                int64_t *cm = (int64_t *)malloc((size_t)chunk_n * sizeof(int64_t));
                int32_t *cs = (int32_t *)malloc((size_t)chunk_n * sizeof(int32_t));
                int32_t *cp = (int32_t *)malloc((size_t)chunk_n * sizeof(int32_t));
                int64_t *cg = (int64_t *)malloc((size_t)chunk_n * sizeof(int64_t));
                if (!cm || !cs || !cp || !cg) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
                int64_t o = 0;
                for (int t = 0; t < n_threads; t++) {
                    _Trip *tb = chunk_bufs + (size_t)t * slot_sz;
                    for (int b = 0; b < chunk_cnt[t]; b++) {
                        cm[o] = tb[b].mid; cs[o] = tb[b].sid;
                        cp[o] = tb[b].pred; cg[o] = tb[b].mg;
                        o++;
                    }
                }
                st = carquet_writer_write_batch(p_w, 0, cm, chunk_n, NULL, NULL);
                st = carquet_writer_write_batch(p_w, 1, cs, chunk_n, NULL, NULL);
                st = carquet_writer_write_batch(p_w, 2, cp, chunk_n, NULL, NULL);
                st = carquet_writer_write_batch(p_w, 3, cg, chunk_n, NULL, NULL);
                if (st != CARQUET_OK) { fprintf(stderr, "  [WARN] --sample-index: pairs write: %s\n", cerr.message); fail = 1; }
                free(cm); free(cs); free(cp); free(cg);
            }
        }
        for (int t = 0; t < n_threads; t++)
            total_rows += chunk_cnt[t];
        done_blocks += (cend - cstart);
        /* Progressive Score-Freigabe (2026-08-15): jeder Block wird nur
         * EINMAL gelesen (Pass 2 ist der letzte Nutzer). Freigeben direkt
         * nach dem Chunk → Peak-RAM = Score-Load-Peak (zu Beginn) bzw.
         * Dateigröße (am Ende), nie die Summe beider. */
        for (int m = cstart; m < cend; m++) {
            int b = block_order[m];
            free(blocks[b].scores);
            blocks[b].scores = NULL;
        }
        if (done_blocks % dot_interval == 0) {
            printf(".");
            fflush(stdout);
        }
    }
    st = carquet_writer_close(p_w);
    if (st != CARQUET_OK) { fprintf(stderr, "  [ERROR] --sample-index: pairs close: %s\n", cerr.message); fail = 1; }
    carquet_schema_free(p_sc);
    free(chunk_bufs);
    free(chunk_cnt);
    printf(" done (%ld pairs)\n", total_rows);
    printf("  [sample-index] %s  (members=%d pairs=%ld samples=%d)\n",
           p_pairs, n_blocks, total_rows, g_n_test);

    free(block_order);
    free(member_ids);
    return (fail || rc) ? 1 : 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * print_ensemble_confusion — --debug-confusion: confusion matrix of the
 * FINAL ensemble over the full test split (2026-08-14).
 *
 * Sums the scores of all set-members (same semantics as the final
 * re-evaluation in the beam), takes argmax per sample and feeds the
 * resulting y_pred into print_confusion_debug() from ki-common.h — the
 * SAME tprint table the trainer's --debug-confusion prints, so MERGE and
 * TRN outputs can be compared 1:1. The full 10×10 matrix is shown even
 * under --target k / --filter-sample (the raw class drift — e.g. where
 * Shirt/T-shirt/Pullover samples end up — is exactly the question for the
 * specialist-orchestrator research).
 *
 * set:  member bitmask (g_best_used / greedy_set) of the winning ensemble.
 * n_blk: total blocks. */
static void print_ensemble_confusion(const uint8_t *set, int n_blk) {
    if (!set || !g_labels || g_n_test <= 0 || g_n_classes <= 0) return;
    size_t row_sz = (size_t)g_n_classes;
    size_t score_sz = (size_t)g_n_test * row_sz;
    SCORE_TYPE *sum = (SCORE_TYPE *)calloc(score_sz, sizeof(SCORE_TYPE));
    if (!sum) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
    for (int i = 0; i < n_blk; i++) {
        if (!set[i]) continue;
        const SCORE_TYPE *sc = blocks[i].scores;
        if (!sc) continue;
        for (size_t j = 0; j < score_sz; j++) sum[j] += sc[j];
    }
    uint8_t *y_pred = (uint8_t *)malloc((size_t)g_n_test);
    if (!y_pred) { free(sum); fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
    for (int s = 0; s < g_n_test; s++) {
        const SCORE_TYPE *row = sum + (size_t)s * row_sz;
        int best = 0;
        for (int k = 1; k < g_n_classes; k++)
            if (row[k] > row[best]) best = k;
        y_pred[s] = (uint8_t)best;
    }
    print_confusion_debug(g_labels, y_pred, g_n_test, 0, 1);
    free(y_pred);
    free(sum);
}

/* ═══════════════════════════════════════════════════════════════════
 * print_debug_members — --debug-member: detailed list of the SELECTED
 * members after the beam search, in the trainer's --debug-member format
 * so MERGE and TRN outputs can be compared 1:1 (2026-08-12).
 *
 * The trainer line is:
 *   [  3/3] mem=88.6%/82.0%/E=1801  time=2833ms  xf:chan:enc  W0=..  MIN=..  MAX=..  pxz=..
 * The merge has NO training phase → trn%/E/time/pxz do not exist here and
 * are shown as "-". The REAL per-member evl% is computed from the loaded
 * .ens scores + ground truth (argmax-vs-truth over g_n_test samples) —
 * this is the merge-side number to compare against the trainer's evl%.
 * set:    member bitmask (g_best_used) of the winning ensemble.
 * n_used: number of set members (= g_best_n). */
/* print_debug_members — --debug-member: detailed list of the SELECTED
 * members after the beam search, in the trainer's --debug-member format
 * so MERGE and TRN outputs can be compared 1:1 (2026-08-12).
 *
 * TABLE FORMAT (2026-08-14, user design):
 *   EN  acc[%]  eval[%]  evl-all[%]  err  member  W0  MIN  MAX
 * - acc[%]     = kumulative Ensemble-Accuracy in ADDITIONSREIHENFOLGE
 *                (order[] aus dem Beam, monoton steigend, endet bei ens-total)
 * - eval[%]    = Target-Eval des Members (eval_sample_correct(), Nenner
 *                eval_denom() — unter --target k nur die Target-Samples,
 *                identisch zur Search-Tabelle; ohne --target == evl-all[%])
 * - evl-all[%] = volle Member-Eval (roher argmax über g_n_test, wie die
 *                Trainer-Evl% — zeigt, ob ein Member die Nicht-Target-
 *                Klassen zerstört, während er die 0624-Familie optimiert)
 * - err        = eval_denom() - correct (Target-Fehler wie die Search)
 * Die Menge der Member ist EXAKT die des member-out files (set ==
 * g_best_used) — nur die Sortierung folgt der Additionsreihenfolge statt
 * der Block-Reihenfolge. "time", "pxz", "trn%" (mem=-.-%) entfallen.
 *
 * set:    member bitmask (g_best_used) of the winning ensemble.
 * order:  Additionsreihenfolge (g_best_order); NULL → set in Block-Reihenfolge.
 * order_n: Anzahl Einträge in order.
 * n_used: number of set members (= g_best_n).
 * n_blk:  total blocks.
 * best_eval: final ensemble eval (%). */
static void print_debug_members(const uint8_t *set, const int *order,
                                int order_n, int n_used, int n_blk,
                                float best_eval) {
    if (!set || n_used <= 0) return;
    printf("\n══╡ MEMBERS (--debug-member) ╞═  %d members  ens-eval=%.2f%%  ══════════\n",
           n_used, best_eval);

    /* ── Anzeige-Reihenfolge bauen: 1) order[]-Einträge die in set sind
     * (Additionsreihenfolge), 2) restliche set-Member in Block-Reihenfolge
     * (nur im --max-Sonderfall relevant, wo best_set_best von order[]
     * abweichen kann). Menge bleibt == set == member-out file. */
    int *disp = (int *)malloc((size_t)n_used * sizeof(int));
    if (!disp) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
    int disp_n = 0;
    if (order && order_n > 0) {
        for (int oi = 0; oi < order_n && disp_n < n_used; oi++) {
            int bi = order[oi];
            if (bi >= 0 && bi < n_blk && set[bi])
                disp[disp_n++] = bi;
        }
    }
    for (int bi = 0; bi < n_blk && disp_n < n_used; bi++)
        if (set[bi]) {
            int already = 0;
            for (int di = 0; di < disp_n; di++)
                if (disp[di] == bi) { already = 1; break; }
            if (!already) disp[disp_n++] = bi;
        }
    if (disp_n == 0) { free(disp); return; }

    /* Tabellenkopf + Zeilen via libtprint (2026-08-14) — dieselbe
     * Tabellen-Library wie die Trainer (tprint_create/…), damit die
     * Spaltenbreiten automatisch zur längsten Zelle passen (hand-rolled
     * %-7s brach bei langen Specs wie "colswap-1-4@rot112:lbp:tri8"). */
    TPrint *tp = tprint_create(stdout, TRUE, TRUE, 2, 2);
    tprint_set_double_fmt(tp, "%7.2f");
    tprint_column_add(tp, "EN",       TPAlign_center, TPAlign_right);
    tprint_column_add(tp, "acc[%]",   TPAlign_center, TPAlign_right);
    tprint_column_add(tp, "eval[%]",  TPAlign_center, TPAlign_right);
    tprint_column_add(tp, "evl-all[%]",TPAlign_center, TPAlign_right);
    tprint_column_add(tp, "err",      TPAlign_center, TPAlign_right);
    tprint_column_add(tp, "member",   TPAlign_center, TPAlign_left);
    tprint_column_add(tp, "W0",       TPAlign_center, TPAlign_left);
    tprint_column_add(tp, "MIN",      TPAlign_center, TPAlign_right);
    tprint_column_add(tp, "MAX",      TPAlign_center, TPAlign_right);

    /* Kumulative Ensemble-Summe: pro Schritt die Scores des Members addieren
     * und argmax-vs-truth über g_n_test (wie die FINAL-RE-EVALUATION im
     * Beam). Am Ende == ens-total. */
    size_t row_sz = (size_t)g_n_classes;
    size_t score_sz = (size_t)g_n_test * row_sz;
    SCORE_TYPE *acc_sum = (SCORE_TYPE *)calloc(score_sz, sizeof(SCORE_TYPE));
    if (!acc_sum) { free(disp); fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
    int e_ok = 0;
    int e_ok_all = 0;   /* volle Ensemble-Eval (roher argmax, alle Samples) */

    for (int di = 0; di < disp_n; di++) {
        int i = disp[di];
        /* ZWEI Member-Evals (2026-08-14, user design):
         * - m_correct_all : volle Eval über ALLE g_n_test Samples (roher
         *   argmax) — "evl-all[%]", konsistent zur Trainer-Evl%.
         * - m_correct_tgt : Target-Eval über eval_sample_correct() —
         *   "eval[%]", identisch zur Search-Tabelle (unter --target k
         *   zählt nur die Target-Samples). Ohne --target sind beide gleich. */
        int m_correct_all = 0;
        int m_correct_tgt = 0;
        if (g_labels && blocks[i].scores) {
            for (int s = 0; s < g_n_test; s++) {
                const SCORE_TYPE *row = blocks[i].scores +
                                        (size_t)s * (size_t)g_n_classes;
                int best = 0;
                for (int k = 1; k < g_n_classes; k++)
                    if (row[k] > row[best]) best = k;
                if (best == (int)g_labels[s]) m_correct_all++;
                if (eval_sample_correct(row, g_labels[s])) m_correct_tgt++;
            }
        }
        float evl_tgt = (eval_denom() > 0)
                        ? 100.0f * (float)m_correct_tgt / (float)eval_denom() : 0.0f;
        float evl_all = (g_n_test > 0)
                        ? 100.0f * (float)m_correct_all / (float)g_n_test : 0.0f;
        /* kumulative Ensemble-Summe aktualisieren */
        const SCORE_TYPE *sc = blocks[i].scores;
        if (sc) {
            for (size_t j = 0; j < score_sz; j++) acc_sum[j] += sc[j];
            e_ok = 0;
            e_ok_all = 0;
            for (int s = 0; s < g_n_test && g_labels; s++) {
                const SCORE_TYPE *row = acc_sum + (size_t)s * row_sz;
                if (eval_sample_correct(row, g_labels[s])) e_ok++;
                /* volle Ensemble-Eval: roher argmax über alle Samples */
                int best = 0;
                for (int k = 1; k < g_n_classes; k++)
                    if (row[k] > row[best]) best = k;
                if (best == (int)g_labels[s]) e_ok_all++;
            }
        }
        float acc = (eval_denom() > 0)
                    ? 100.0f * (float)e_ok / (float)eval_denom() : 0.0f;
        const char *_xn = (blocks[i].xform_real >= 0)
                          ? xform_name_safe(blocks[i].xform_real)
                          : xform_group_label(blocks[i].xform_id);
        char _ml[64];
        member_label(&blocks[i], _ml, sizeof(_ml));
        char _spec[128];
        snprintf(_spec, sizeof(_spec), "%s:%s", _xn, _ml);
        char _w0[16];
        snprintf(_w0, sizeof(_w0), "0x%08X", blocks[i].w0_marker);
        char _min[32], _max[32];
#if COUNTER_TYPE_IS_FLOAT
        snprintf(_min, sizeof(_min), "%.0f", (double)blocks[i].score_min);
        snprintf(_max, sizeof(_max), "%.0f", (double)blocks[i].score_max);
#else
        snprintf(_min, sizeof(_min), "%" PRId64, (int64_t)blocks[i].score_min);
        snprintf(_max, sizeof(_max), "%" PRId64, (int64_t)blocks[i].score_max);
#endif
        tprint_data_add_int32(tp, 0, di + 1);
        tprint_data_add_double(tp, 1, (double)acc);
        tprint_data_add_double(tp, 2, (double)evl_tgt);
        tprint_data_add_double(tp, 3, (double)evl_all);
        tprint_data_add_int32(tp, 4, eval_denom() - m_correct_tgt);
        tprint_data_add_str(tp, 5, _spec);
        tprint_data_add_str(tp, 6, _w0);
        tprint_data_add_str(tp, 7, _min);
        tprint_data_add_str(tp, 8, _max);
    }
    /* ens-total: finale Ensemble-Eval (acc_sum der kompletten Menge).
     * INTENTIONAL (2026-08-14): eval[%] (Spalte 2) nutzt eval_denom() —
     * identisch zur Search-Tabelle, die unter --target k nur die
     * Target-Samples zählt. evl-all[%] (Spalte 3) zeigt die volle
     * Ensemble-Eval über alle Samples. Vorher gab es nur den g_n_test-
     * Nenner: unter --target 0,2,4,6 zeigte ens-total 33.90% statt der
     * Search-84.75% (gleiche correct-Zahl, falscher Nenner). */
    {
        float e_eval = (eval_denom() > 0)
                       ? 100.0f * (float)e_ok / (float)eval_denom() : 0.0f;
        float e_eval_all = (g_n_test > 0)
                           ? 100.0f * (float)e_ok_all / (float)g_n_test : 0.0f;
        char _et[128];
        snprintf(_et, sizeof(_et), "ens-total  (correct=%d/%d)",
                 e_ok, eval_denom());
        tprint_data_add_str(tp, 0, "");
        tprint_data_add_str(tp, 1, "");
        tprint_data_add_double(tp, 2, (double)e_eval);
        tprint_data_add_double(tp, 3, (double)e_eval_all);
        tprint_data_add_int32(tp, 4, eval_denom() - e_ok);
        tprint_data_add_str(tp, 5, _et);
        tprint_data_add_str(tp, 6, "");
        tprint_data_add_str(tp, 7, "");
        tprint_data_add_str(tp, 8, "");
    }
    tprint_print(tp);
    tprint_free(tp);
    free(acc_sum);
    free(disp);
    printf("\n");
    fflush(stdout);
}

/* ═══════════════════════════════════════════════════════════════════
 * export_member_file — SINGLE API for --member-out (all modes)
 * ═══════════════════════════════════════════════════════════════════
 * set:    member bitmask to export (NULL = ALL loaded blocks; used by the
 *         plain merge-eval mode), or best_used/g_best_used (beam/greedy).
 * n_members, eval, beam_width: metadata for the header line.
 * Preserves the functional behavior of the three former writers:
 *   - --xform mode: dedup by (color, enc_type, enc_width), expand xforms
 *   - otherwise: write XF:CHAN:ENC + W0-marker per selected member
 * One function = one header/set semantics → no more display/export drift. */
static void export_member_file(const char *path, const uint8_t *set,
                               int n_members, float eval, int beam_width) {
    if (!path[0]) return;
    FILE *mf = fopen(path, "w");
    if (!mf) { fprintf(stderr, "  [ERROR] Cannot write %s\n", path); return; }

    if (g_xform_spec[0]) {
        fprintf(mf, "# Expanded with --xform — XF:CHAN:ENC format\n");
    } else {
        fprintf(mf, "# Optimal subset: XF:CHAN:ENC (original xforms)\n");
    }
    fprintf(mf, "# Generated by merge-ensemble  H=%d  EP=%d  beam=%d  eval=%.2f%%  members=%d\n",
            g_hidden, g_epochs, beam_width, eval, n_members);
    /* META header — the TRN reads this BEFORE the CLI options and adopts
     * the values as DEFAULTS (user CLI options still win). The values come
     * from the ensemble .meta (g_meta_*) with fallback to the .ens/index
     * adopted config (g_hidden/g_epochs/g_split_vn/g_split_hn/g_ens_maj/
     * g_ens_majt). Roundtrip: MERGE --member-out → TRN --member-file →
     * IFC --import (feature 2026-08-10). */
    {
        int _H = g_meta_ok ? g_meta_h   : g_hidden;
        int _EP = g_meta_ok ? g_meta_ep  : g_epochs;
        int _VN = g_meta_ok ? g_meta_vn  : g_split_vn;
        int _HN = g_meta_ok ? g_meta_hn  : g_split_hn;
        const char *_MAJ = g_meta_maj[0] ? g_meta_maj : (g_ens_maj[0] ? g_ens_maj : "1");
        int _MAJT = (g_meta_majt != -999) ? g_meta_majt
                  : (g_ens_majt != -999 ? g_ens_majt : -2);
        /* TARGET= is appended when --target was given: the TRN's
         * --debug-class-voting uses it to mark the specialist's target
         * column (feature 2026-08-12, "one specialist = one question").
         * No --target → no field (old member files stay compatible). */
        char target_spec[64] = "";
        if (g_target_active && g_target_count > 0) {
            int off = 0;
            for (int ti = 0; ti < g_target_count; ti++)
                off += snprintf(target_spec + off, sizeof(target_spec) - (size_t)off,
                                "%s%d", ti ? "," : "", g_target_classes[ti]);
        }
        fprintf(mf, "# META: H=%d  EP=%d  VN=%d  HN=%d  MAJ=%s  MAJ1_THRESH=%d%s%s\n",
                _H, _EP, _VN, _HN, _MAJ, _MAJT,
                g_target_active ? "  TARGET=" : "", target_spec);
    }

    if (g_xform_spec[0]) {
        /* --xform mode: dedup by (color, enc_type, enc_width) from the set */
        int n_best = 0, best_col[256], best_typ[256], best_wid[256];
        for (int bi = 0; bi < n_blocks && n_best < 256; bi++) {
            if (set && !set[bi]) continue;
            int dup = 0;
            for (int j = 0; j < n_best; j++)
                if (best_col[j] == blocks[bi].color &&
                    best_typ[j] == blocks[bi].enc_type &&
                    best_wid[j] == blocks[bi].enc_width) { dup = 1; break; }
            if (!dup) {
                best_col[n_best] = blocks[bi].color;
                best_typ[n_best] = blocks[bi].enc_type;
                best_wid[n_best] = blocks[bi].enc_width;
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
        /* Original xforms: iterate the set (NULL = all blocks) */
        int _n_members = 0;
        for (int bi = 0; bi < n_blocks; bi++) {
            if (set && !set[bi]) continue;
            char label[64];
            const char *_xn = (blocks[bi].xform_real >= 0)
                              ? xform_name_safe(blocks[bi].xform_real)
                              : xform_group_label(blocks[bi].xform_id);
            member_label(&blocks[bi], label, sizeof(label));
            fprintf(mf, "%s:%s  0x%08X\n", _xn, label, blocks[bi].w0_marker);
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
    double va, vb;
    if (g_expansion_sort == 2) {
        /* clarity: sort by the MEMBER's own class-score std-dev (path-
         * independent). Prefer unambiguous voters — they disturb other
         * members' decisions the least (plan-2026-08-08). */
        va = blocks[ea->member_idx].clarity;
        vb = blocks[eb->member_idx].clarity;
    } else {
        va = g_expansion_sort ? (double)ea->eval - g_beam[ea->slot_idx].eval : (double)ea->eval;
        vb = g_expansion_sort ? (double)eb->eval - g_beam[eb->slot_idx].eval : (double)eb->eval;
    }
    if (vb > va + 0.0001) return 1;
    if (va > vb + 0.0001) return -1;
    if (eb->margin > ea->margin) return 1;
    if (ea->margin > eb->margin) return -1;
    return 0;
}

/* ── Load a previously exported attractor (--member-out format) into a
 * member mask. Returns the number of members loaded (0 = none/failed).
 * The beam can then start from THIS SET instead of a single seed — adding
 * new pool members (e.g. @shuffle pipelines) can no longer redirect the
 * beam path below the saved attractor (monotone extension, Finding 11 fix,
 * 2026-08-09). Format: one "XF:CHAN:ENC" per line, '#' = comment. */
static int load_start_member_set(const char *path, uint8_t *mask, int n) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "  [WARN] --member-start: cannot open %s\n", path); return 0; }
    char line[256];
    int loaded = 0, missing = 0;
    while (fgets(line, sizeof(line), f)) {
        char *nl = strchr(line, '\n'); if (nl) *nl = '\0';
        char *sp = strchr(line, '#');  if (sp) *sp = '\0';
        /* trim */
        char *s = line;
        while (*s == ' ' || *s == '\t') s++;
        size_t ln = strlen(s);
        while (ln > 0 && (s[ln-1] == ' ' || s[ln-1] == '\t')) s[--ln] = '\0';
        if (!*s) continue;
        /* the member file format is "SPEC  W0-marker" — cut the spec at the
         * first whitespace (the marker is not part of the lookup key). */
        for (char *_w = s; *_w; _w++) {
            if (*_w == ' ' || *_w == '\t') { *_w = '\0'; break; }
        }
        int idx = find_member_by_spec(s);
        if (idx < 0) { missing++; continue; }
        mask[idx] = 1;
        loaded++;
    }
    fclose(f);
    if (missing > 0)
        fprintf(stderr, "  [WARN] --member-start: %d spec(s) not found in pool (skipped)\n", missing);
    printf("  StartSet:  %d members loaded from %s\n", loaded, path);
    return loaded;
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
    /* Unseeded runs auto-seed with the BEST single member and start with ONLY
     * that one slot (no fill) — exactly like multi-try TRY 1, whose pair-based
     * step-1 expansion reaches better attractors than the 10-single-slot start
     * (bug 2026-07-31: M3 without --tries stopped at 67.70% while --tries 5
     * TRY 1 found 69.36% — the divergence was the beam slot depth, 1 seeded
     * slot vs beam_width single slots, not the seed itself). */
    int auto_seed = (initial_member < 0 || initial_member >= n);
    /* --member-start: start the beam from a previously exported attractor
     * (monotone pool extension — the saved set is the FLOOR; new pool members
     * can only add, never redirect below it). Overrides the single seed.
     * Finding 11 fix (2026-08-09): @shuffle pipelines can no longer worsen
     * the result. */
    uint8_t *start_mask = (uint8_t *)calloc((size_t)n, 1);
    int n_start = 0;
    if (g_start_member_file[0]) {
        n_start = load_start_member_set(g_start_member_file, start_mask, n);
        if (n_start > 0) {
            initial_member = -1;         /* start-set replaces the seed */
            auto_seed = 1;               /* but do NOT overwrite below: handled here */
            /* keep the set for the init block */
        } else {
            fprintf(stderr, "  [WARN] --member-start: no valid members — falling back to seed\n");
        }
    }
    if (auto_seed && n_start == 0) {
        float best_se = -1.0f;
        for (int mi = 0; mi < n; mi++) {
            const SCORE_TYPE *sc = blocks[mi].scores;
            int ok = 0;
            for (int s = 0; s < g_n_test && has_labels; s++) {
                const SCORE_TYPE *r = sc + (size_t)s * row_sz;
                int p = 0;
                for (int k = 1; k < g_n_classes; k++)
                    if (r[k] > r[p]) p = k;
                if (p == (int)g_labels[s]) ok++;
            }
            float ev = has_labels ? (float)ok * 100.0f / (float)g_n_test : 0.0f;
            if (ev > best_se) { best_se = ev; initial_member = mi; }
        }
    }
    /* Initial candidate: empty, pre-seeded with initial_member, or pre-loaded
     * with the full start-member set (--member-start). */
    if (n_start > 0) {
        /* Sum ALL start-set members into beam[0] — the saved attractor is the
         * floor the beam extends from. */
        beam[0].sum = (SCORE_TYPE *)calloc(score_sz, sizeof(SCORE_TYPE));
        beam[0].used = (uint8_t *)calloc((size_t)n, 1);
        int _cnt = 0;
        for (int _mi = 0; _mi < n; _mi++) {
            if (!start_mask[_mi]) continue;
            const SCORE_TYPE *_sc = blocks[_mi].scores;
            for (size_t _i = 0; _i < score_sz; _i++)
                beam[0].sum[_i] += _sc[_i];
            beam[0].used[_mi] = 1;
            _cnt++;
        }
        beam[0].n_used = _cnt;
        int _c = 0;
        for (int s = 0; s < g_n_test && has_labels; s++) {
            const SCORE_TYPE *row = beam[0].sum + (size_t)s * row_sz;
            if (eval_sample_correct(row, g_labels[s])) _c++;
        }
        beam[0].eval = (float)_c * 100.0f / (float)eval_denom();
        /* floor display: shown as EN=0 row right after the table header
         * (see the fmt block below — beam[0] holds the start-set sum). */
    } else if (initial_member >= 0 && initial_member < n) {
        beam[0].sum = (SCORE_TYPE *)calloc(score_sz, sizeof(SCORE_TYPE));
        for (size_t i = 0; i < score_sz; i++)
            beam[0].sum[i] = (SCORE_TYPE)blocks[initial_member].scores[i];
        beam[0].used = (uint8_t *)calloc((size_t)n, 1);
        beam[0].used[initial_member] = 1;
        beam[0].n_used = 1;
        int _c = 0;
        for (int s = 0; s < g_n_test && has_labels; s++) {
            const SCORE_TYPE *row = beam[0].sum + (size_t)s * row_sz;
            if (eval_sample_correct(row, g_labels[s])) _c++;
        }
        beam[0].eval = (float)_c * 100.0f / (float)eval_denom();
        /* Fill remaining beam_width-1 slots with best singles (≠ seed) so the
         * beam state matches the unseeded case (beam_width single-member slots).
         * Only for EXPLICIT seeds (--member-seed): auto-seeded unseeded runs
         * start with the single seed slot (like multi-try TRY 1) — the pair
         * based step-1 expansion reaches better attractors (67.70 → 69.36).
         * Without this, seeded and unseeded runs diverge because the beam starts
         * with different slot depths (1 pair vs 2 singles for beam_width=2). */
        if (beam_width > 1 && has_labels && !pool_exclude && !auto_seed) {
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
                float _gt = g_min_gain;
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
                        for (size_t _i2 = 0; _i2 < score_sz; _i2++)
                            beam[_si].sum[_i2] = (SCORE_TYPE)blocks[_mi].scores[_i2];
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

    float best_eval = (n_start > 0) ? beam[0].eval : 0.0f;  /* floor = saved attractor */
    float gain_threshold = g_min_gain;
    int best_n = 0;                 /* total members in best_used */
    uint8_t *best_used = (uint8_t *)calloc((size_t)n, 1);  /* which members form best ensemble */
    /* BEST-SET TRACKING (FIX 2026-08-10): best_used is overwritten at every
     * commit, so the final re-evaluation rated the LAST set — which can be
     * WORSE than an earlier one (12 members: 88.62% vs 11: 88.72%, a member
     * that lowers the ensemble acc). Keep the highest-eval set: on each
     * commit, re-evaluate the set and remember the best one. */
    uint8_t *best_set_best = (uint8_t *)calloc((size_t)n, 1);
    int      best_set_n = 0;
    float    best_set_eval = -1.0f;
    uint8_t *best_cand_used = (uint8_t *)calloc((size_t)n, 1); /* used-mask of the ACCEPTED candidate
                                                                  (beam[si].used | mi) — NOT beam[0].used
                                                                  after pruning; see acceptance block */
    /* --max cumulative acceptance (2026-08-07): members accepted between two
     * commits are buffered here — the block commit must add ALL of them to
     * best_used (best_cand_used only holds the LAST accepted candidate's
     * path, which may not include earlier buffered members of the block). */
    int *pending = (int *)malloc((size_t)(g_max_mode ? (size_t)n : 1) * sizeof(int));
    if (!pending) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
    int pending_n = 0;
    /* Additionsreihenfolge des akzeptierten Ensembles (2026-08-14): jeder
     * akzeptierte Member wird hier in der Reihenfolge notiert, in der der
     * Beam ihn annahm. --debug-member zeigt damit die kumulative acc[%]
     * monoton steigend (identisch zur EN-Tabelle). Menge == best_used. */
    int *accepted = (int *)malloc((size_t)n * sizeof(int));
    if (!accepted) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
    int accepted_n = 0;
    if (initial_member >= 0) {
        best_used[initial_member] = 1;  /* seed immer tracken */
        /* INTENTIONAL (2026-08-14): der Seed gehört an den ANFANG der
         * Additionsreihenfolge — vorher fehlte er in accepted[] und die
         * --debug-member Tabelle startete mit EN=2 statt EN=1 (die kumulative
         * acc[%] begann beim falschen Member). */
        if (accepted_n < n) accepted[accepted_n++] = initial_member;
    }
    /* --member-start: the saved attractor is the floor — best_eval starts
     * there and only IMPROVEMENTS are accepted (monotone extension). */
    if (n_start > 0) {
        for (int _mi = 0; _mi < n; _mi++)
            if (start_mask[_mi]) best_used[_mi] = 1;
        best_n = n_start;
    }
    int best_member_idx = -1;       /* which member was added at the best step */
    int best_slot_idx = -1;         /* beam slot (path) of the accepted candidate */

    printf("\n");
    printf("══╡ BEAM SEARCH ENSEMBLE ╞═════════════════════════════════\n");
    printf("  %d score blocks (%d test samples, %d classes)\n",
           n, g_n_test, g_n_classes);
    printf("  Config: H=%d  EP=%d  VN=%d  HN=%d  TE=%d\n",
            g_hidden, g_epochs, g_split_vn, g_split_hn, g_target_err_x100);
    if (g_target_active || g_filter_sample_active) {
        printf("  Mode:");
        if (g_target_active) {
            printf("  target={");
            for (int _t = 0; _t < g_target_count; _t++)
                printf("%s%d", _t ? "," : "", g_target_classes[_t]);
            printf("}");
        }
        if (g_filter_sample_active) {
            printf("  filter-sample={");
            for (int _f = 0; _f < g_filter_sample_count; _f++)
                printf("%s%d", _f ? "," : "", g_filter_sample_classes[_f]);
            printf("}");
        }
        printf("  (denom=%d/%d)\n", g_eval_denom, g_n_test);
    }
    printf("  Score Mode:  %s\n", ki_score_type_str());
    printf("  Beam width: %d\n", beam_width);
    printf("  Scan:       %s\n",
           g_max_mode ? "cumulative min-gain (--max: block-commit when the sum "
                       "of the last beam_width steps reaches min-gain)"
                      : "strict per-member min-gain (immediate stop)");
    printf("  Threads:    %d\n", omp_get_max_threads());
    printf("\n");
    printf("  %-4s  %-7s  %-11s  %-7s  %-7s  %-5s  %-10s  %-13s  %-13s  %-7s  %-4s  %s\n",
            "EN", "acc[%]",  "correct",  "gain[%]", "eval[%]", "err", "W0[0]", "MIN-score", "MAX-score", "t[s]", "slot", "member");
    printf("  %-4s  %-7s  %-11s  %-7s  %-7s  %-5s  %-10s  %-13s  %-13s  %-7s  %-4s  %s\n",
           "----", "-------", "-----------", "-------", "-------", "-----", "----------", "-----------", "-----------", "------", "----", "-------------------");

#if COUNTER_TYPE_IS_FLOAT
    const char fmt[]="  %-4d  %-7.2f  %5d/%-5d  %-+7.2f  %-7.1f  %-5d  0x%08X  %-+13.0f  %-+13.0f  %-7.2f  %-4d  %s\n";
#else
    const char fmt[]="  %-4d  %-7.2f  %5d/%-5d  %-+7.2f  %-7.1f  %-5d  0x%08X  %-+13" PRId64 "  %-+13" PRId64 "  %-7.2f  %-4d  %s\n";
#endif

    /* ── EN=0 row: the --member-start floor (2026-08-12) ──
     * When a start-set was loaded, beam[0] already holds its summed scores
     * and eval. Show it as EN=0 so the table reads "0 = start-set (fixed),
     * 1..N = added members" — the final member count is start+N. */
    if (n_start > 0 && has_labels) {
        int _c0 = (int)(beam[0].eval * (float)eval_denom() / 100.0f + 0.5f);
        char _lbl0[96];
        snprintf(_lbl0, sizeof(_lbl0), "(Start-Set: %d members)", beam[0].n_used);
#if COUNTER_TYPE_IS_FLOAT
        printf(fmt, 0, beam[0].eval, _c0, eval_denom(), 0.0f, beam[0].eval,
               eval_denom() - _c0, 0u,
               (double)blocks[0].score_min, (double)blocks[0].score_max,
               0.0, -1, _lbl0);
#else
        printf(fmt, 0, beam[0].eval, _c0, eval_denom(), 0.0f, beam[0].eval,
               eval_denom() - _c0, 0u,
               (int64_t)blocks[0].score_min, (int64_t)blocks[0].score_max,
               0.0, -1, _lbl0);
#endif
    }

    FILE *save_f = NULL;
    if (save_path && save_path[0]) {
        save_f = fopen(save_path, "w");
        if (save_f) fprintf(save_f, "# EN  acc  correct  total  gain  member\n");
    }

    float prev_acc = (n_start > 0) ? beam[0].eval : 0.0f;  /* floor for gain calc */
    int no_improve_steps = 0;  /* --max (cumulative acceptance): window counter —
                                  counts EVERY step since the last commit; reset
                                  only at a commit; stop at beam_width (plan
                                  2026-08-07) */
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
        int _c = (int)(beam[0].eval * (float)eval_denom() / 100.0f + 0.5f);
        int _err0 = eval_denom() - _c;
#if COUNTER_TYPE_IS_FLOAT
        {   struct timeval _now; gettimeofday(&_now, NULL);
            double _elapsed = (double)(_now.tv_sec - _step_start.tv_sec) + (double)(_now.tv_usec - _step_start.tv_usec) * 1e-6;
        printf(fmt, 1, beam[0].eval, _c, eval_denom(), beam[0].eval,
               beam[0].eval, _err0,
               blocks[initial_member].w0_marker,
               (double)blocks[initial_member].score_min,
               (double)blocks[initial_member].score_max,
               _elapsed, -1, _label);
        }
#else
        {   struct timeval _now; gettimeofday(&_now, NULL);
            double _elapsed = (double)(_now.tv_sec - _step_start.tv_sec) + (double)(_now.tv_usec - _step_start.tv_usec) * 1e-6;
        printf(fmt, 1, beam[0].eval, _c, eval_denom(), beam[0].eval,
               beam[0].eval, _err0,
               blocks[initial_member].w0_marker,
               (int64_t)blocks[initial_member].score_min,
               (int64_t)blocks[initial_member].score_max,
               _elapsed, -1, _label);
        }
#endif
        prev_acc = beam[0].eval;
        best_eval = beam[0].eval;
        best_n = 1;
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
                    /* --max N (FIX 2026-08-10): never expand a path that is
                     * already at the member cap — the acceptance can then not
                     * overshoot (the old code only capped at the block commit,
                     * but the accepted candidate's PATH itself grew past N). */
                    if (g_max_members > 0 && beam[si].n_used >= g_max_members) continue;

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
                            if (eval_sample_correct_pred(best_k, g_labels[s])) correct++;
                            if (second_k >= 0) margin += best_val - second_val;
                        }
                    }
                    float acc = has_labels ? (float)correct * 100.0f / (float)(eval_denom()) : 0.0f;

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
                printf("       slot=%d mi=%d  %s:%s  eval=%.2f  margin=%.0f  clarity=%.1f  CRC=0x%016" PRIx64 "\n",
                       _si, _mi, _xn, _ml, exp[ei].eval, (double)exp[ei].margin,
                       blocks[_mi].clarity, _crc);
#else
                printf("       slot=%d mi=%d  %s:%s  eval=%.2f  margin=%" PRId64 "  clarity=%.1f  CRC=0x%016" PRIx64 "\n",
                       _si, _mi, _xn, _ml, exp[ei].eval, exp[ei].margin,
                       blocks[_mi].clarity, _crc);
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

        /* --beam-bestN: start the per-level selection at the N-th best candidate
         * instead of the 1st-best. The 1st-best member is often the "obvious but
         * path-fixing" choice — skipping it forces a different beam path (maximum
         * diversity across --tries, where try k uses bestN=k). No fallback: a
         * level with < N candidates simply yields fewer next-slots. */
        int _start = (g_beam_bestN > 1) ? g_beam_bestN - 1 : 0;
        for (int ei = _start; ei < n_exp && n_next < beam_width; ei++) {
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

            /* INTENTIONAL: NO global pool_exclude write here (was
             * `pool_exclude[mi] = 1`). In multi-try mode this polluted the
             * caller-owned pool with EVERY member ever placed into any beam
             * candidate — the beam scans up to beam_width candidates per step
             * over many steps, so TRY 1 alone locked ~150 of 221 members and
             * the next try found no seed (bug 2026-08-02: --tries 5 ran only
             * 2 tries, "best of 2 tries"). Intra-beam diversity is already
             * enforced by beam[si].used (a member can appear only once per
             * path); cross-try exclusivity is applied correctly AFTER the try
             * via g_best_used in main (pool_exclude[seed] + g_best_used). */

                /* Acceptance (Semantics B, 2026-08-02; extended 2026-08-07):
                 * best_eval advances ONLY when a candidate beats the current
                 * best by more than the acceptance threshold. In strict mode
                 * (default) that threshold is min-gain — the minimum
                 * contribution per added member. In --max mode (cumulative
                 * acceptance) the threshold is 0.0: ANY improvement advances
                 * best_eval/best_cand_used, and the COMMIT (best_used update
                 * in the display block) happens only when the accumulated
                 * gain since the last commit (best_eval - prev_acc) reaches
                 * min-gain — sub-threshold steps are buffered in
                 * best_cand_used and committed as one block (bit-mass
                 * accumulation, plan 2026-08-07).
                 * ATOMIC SET/EVAL COUPLING (bug 2026-08-02, 3rd recurrence of
                 * 07-30): best_eval = exp[ei].eval is the eval of THIS
                 * candidate, which is computed over beam[si].used + mi (the
                 * candidate's full PATH — see the expansion eval at line
                 * base[row]+sc[row] where base=beam[si].sum). Therefore
                 * best_cand_used MUST be beam[si].used | mi — the exact set
                 * that was evaluated. Using "accepted sequence" (best_used+mi)
                 * decoupled the two: reported 97.40% (path) vs member.out
                 * 96.97% (sequence) — the same class of bug as 07-30. */
                if (exp[ei].eval > best_eval + (g_max_mode ? 0.0f : gain_threshold)) {
                    best_eval = exp[ei].eval;
                    best_member_idx = mi;
                    best_slot_idx = si;   /* path (beam slot) that produced this candidate */
                    /* best_used = FULL CANDIDATE PATH (beam[si].used | mi),
                     * atomically with best_eval — NOT the accepted sequence.
                     * --max: best_used is set ONLY at the block commit (display
                     * block below); accepted members are buffered in pending[]
                     * instead, so ALL block members (not just the last
                     * candidate's path) reach best_used at the commit. */
                    if (!g_max_mode) {
                        if (!best_used[mi] && accepted_n < n)
                            accepted[accepted_n++] = mi;   /* nur neue Member */
                        best_used[mi] = 1;
                    }
                    else if (pending_n < n) pending[pending_n++] = mi;
                    if (best_cand_used) {
                        memset(best_cand_used, 0, (size_t)n);
                        memcpy(best_cand_used, beam[si].used, (size_t)n);
                        best_cand_used[mi] = 1;
                    }
                }

            n_next++;
        }
        free(seen_mask);

        /* Free old beam */
        for (int i = 0; i < n_beam; i++) { free(beam[i].sum); free(beam[i].used); }
        free(beam);
        beam = next;
        n_beam = n_next;

            /* Show the ACCEPTED CANDIDATE — the member column shows the newly
             * added member (best_member_idx), the path column shows which beam
             * slot (path) produced it. Step 1 (the seed) has no predecessor
             * path → blank. The beam mixes slots, so a member can appear from
             * a different path than the previous step — the path column makes
             * those switches visible. */
            {
                char _label[256];
                float _member_eval = 0.0f;
                int _m = best_member_idx;
                if (_m < 0 && n_exp > 0) _m = exp[0].member_idx;  /* fallback */
                if (_m >= 0) {
                    const char *_xn = xform_name_safe(blocks[_m].xform_real);
                    char _ml[64];
                    member_label(&blocks[_m], _ml, sizeof(_ml));
                    /* HONEST LABEL (2026-08-12): in --max mode the shown member
                     * is the LAST member of the BEST PATH (best_slot_idx), NOT
                     * necessarily the member added in THIS step — the beam can
                     * keep the same path-end member across steps while the
                     * path composition changes (EN=28/29 artefact: same name,
                     * different slot). Mark re-used path ends explicitly.
                     * Only in --max: there best_used is set at the BLOCK
                     * COMMIT (after this print), so a member already in
                     * best_used here is genuinely a re-used path end. In
                     * strict mode best_used[mi]=1 is set at ACCEPT (before
                     * the print) — every member would look "re-used", so the
                     * marker is meaningless there and must stay off. */
                    int _reused = g_max_mode && best_used && best_used[_m];
                    if (_reused)
                        snprintf(_label, sizeof(_label), "%s:%s (path-end, slot %d)",
                                 _xn, _ml, best_slot_idx);
                    else
                        snprintf(_label, sizeof(_label), "%s:%s", _xn, _ml);
                    _member_eval = blocks[_m].file_eval;
                } else
                    snprintf(_label, sizeof(_label), "-");

                /* Track the GLOBAL best.
                 * Semantics (2026-08-02, user design decision): --min-gain is
                 * the MINIMUM CONTRIBUTION per added member. best_eval only
                 * advances when a candidate beats it by > the acceptance
                 * threshold (see the acceptance block above).
                 *
                 * Commit condition (2026-08-07, --max cumulative acceptance):
                 * - strict (default): _gain > 0 here means THIS step was
                 *   accepted and must be shown + added to best_used. The
                 *   first step without acceptance (_gain == 0) stops the
                 *   search: 8 members for ~0.3% is inefficient — min-gain 0.2
                 *   stops at the 3rd member (97.05%) instead of accumulating
                 *   11 for 97.41%. Member-count efficiency is the design goal.
                 * - --max: _gain = best_eval - prev_acc is the ACCUMULATED
                 *   gain of all steps since the last commit (prev_acc is only
                 *   updated at a commit). A commit happens only when that
                 *   accumulated sum is a REAL improvement AND STRICTLY
                 *   exceeds gain_threshold — sub-threshold steps are buffered
                 *   in best_cand_used and committed as one block (bit-mass
                 *   accumulation, plan 2026-08-07).
                 *   CONSISTENCY (2026-08-08): `>` not `>=` — the strict mode
                 *   rejects a member whose gain equals min-gain exactly
                 *   (`exp[ei].eval > best_eval + gain_threshold`); a --max
                 *   block whose summed gain equals min-gain exactly must be
                 *   rejected the same way (only STRICTLY exceeding counts,
                 *   "the sum of the steps EXCEEDS min-gain").
                 *   FIX 2026-08-08: `_gain > 0.0f` is REQUIRED in --max mode
                 *   too. With --min-gain 0 the old `_gain >= gain_threshold`
                 *   was `_gain >= 0.0` — ALWAYS true, so a step with no
                 *   improvement (_gain == 0) committed anyway, never
                 *   incrementing no_improve_steps → the search never stopped
                 *   (same member re-committed forever, 50..74 in the log). */
                float _gain = best_eval - prev_acc;
                int _corr = (int)(best_eval * (float)eval_denom() / 100.0f + 0.5f);
                if (_gain > 0.0f && (g_max_mode ? _gain > gain_threshold : 1)) {
                    struct timeval _now; gettimeofday(&_now, NULL);
                    double _elapsed = (double)(_now.tv_sec - _step_start.tv_sec) + (double)(_now.tv_usec - _step_start.tv_usec) * 1e-6;
#if COUNTER_TYPE_IS_FLOAT
        printf(fmt, step + 1 + en_offset, best_eval, _corr, eval_denom(), _gain,
               _member_eval,
               (_m >= 0) ? blocks[_m].err : 0,
               (_m >= 0) ? blocks[_m].w0_marker : 0,
               (double)((_m >= 0) ? blocks[_m].score_min : 0),
               (double)((_m >= 0) ? blocks[_m].score_max : 0),
               _elapsed, best_slot_idx, _label);
#else
        printf(fmt, step + 1 + en_offset, best_eval, _corr, eval_denom(), _gain,
               _member_eval,
               (_m >= 0) ? blocks[_m].err : 0,
               (_m >= 0) ? blocks[_m].w0_marker : 0,
               (int64_t)((_m >= 0) ? blocks[_m].score_min : 0),
               (int64_t)((_m >= 0) ? blocks[_m].score_max : 0),
               _elapsed, best_slot_idx, _label);
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
                /* Track accepted member for member-out (only gains > min-gain).
                 * Copy the ACCEPTED CANDIDATE's used-mask (recorded in the
                 * acceptance block) — copying beam[0].used after pruning mixed
                 * in members of a different beam path (07-30 bug recurrence). */
                if (_m >= 0) {
                    if (best_cand_used) {
                        memset(best_used, 0, (size_t)n);
                        memcpy(best_used, best_cand_used, (size_t)n);
                    }
                    /* --max block commit: best_cand_used holds only the LAST
                     * accepted candidate's path — add ALL members buffered in
                     * pending[] since the previous commit, so the whole block
                     * lands in best_used (bit-mass, plan 2026-08-07).
                     * --max N (FIX 2026-08-10): cap the block at the remaining
                     * slots — otherwise a block commit overshoots the cap in
                     * one step (the outer loop checks best_n >= g_max_members
                     * only BETWEEN steps; a cumulative block adds many members
                     * at once, 12 → 18). */
                    if (g_max_mode) {
                        int _cur_n = 0;
                        for (int _bi = 0; _bi < n; _bi++)
                            if (best_used[_bi]) _cur_n++;
                        for (int _p = 0; _p < pending_n; _p++) {
                            if (g_max_members > 0 && _cur_n >= g_max_members) break;
                            if (!best_used[pending[_p]]) {
                                best_used[pending[_p]] = 1;
                                if (accepted_n < n) accepted[accepted_n++] = pending[_p];
                                _cur_n++;
                            }
                        }
                        pending_n = 0;
                    }
                }
                /* Re-count best_used AFTER the commit — the acceptance block
                 * does not know the final set (the commit replaces best_used
                 * with the full candidate path + pending block), so counting
                 * there lagged behind and --max N stopped one member late
                 * (2026-08-07, 4 → 5 members). */
                best_n = 0;
                for (int _bi = 0; _bi < n; _bi++)
                    if (best_used[_bi]) best_n++;
                /* BEST-SET TRACKING (FIX 2026-08-10): re-evaluate the set
                 * that was just committed and keep the HIGHEST-eval set.
                 * Without this the final REPORT rated the LAST commit, which
                 * can be worse than an earlier one (BV --max 12: 11 members
                 * 88.72% vs 12 members 88.62% — the 12th member hurt). */
                if (g_max_mode && best_n > 0) {
                    SCORE_TYPE *_set_sum = (SCORE_TYPE *)calloc(score_sz, sizeof(SCORE_TYPE));
                    if (_set_sum) {
                        for (int _bi = 0; _bi < n; _bi++) {
                            if (!best_used[_bi]) continue;
                            const SCORE_TYPE *_sc = blocks[_bi].scores;
                            for (size_t _j = 0; _j < score_sz; _j++)
                                _set_sum[_j] += _sc[_j];
                        }
                        int _sok = 0;
                        for (int _s = 0; _s < g_n_test; _s++) {
                            const SCORE_TYPE *_row = _set_sum + (size_t)_s * row_sz;
                            int _pred = 0;
                            for (int _k = 1; _k < g_n_classes; _k++)
                                if (_row[_k] > _row[_pred]) _pred = _k;
                            /* FIX (2026-08-12): target-aware — same semantics
                             * as eval_sample_correct_pred(). Previously plain
                             * _pred==label counted over ALL 10000 samples but
                             * the caller divided by eval_denom()=1000 → the
                             * --max "BEAM BEST" reported e.g. 878/1000 that
                             * was actually 878 global hits over 10000, not the
                             * target recall (real target recall 69.0%). */
                            if (eval_sample_correct_pred(_pred, g_labels[_s])) _sok++;
                        }
                        float _set_eval = (float)_sok * 100.0f / (float)eval_denom();
                        if (_set_eval > best_set_eval) {
                            best_set_eval = _set_eval;
                            best_set_n = best_n;
                            memset(best_set_best, 0, (size_t)n);
                            memcpy(best_set_best, best_used, (size_t)n);
                        }
                        free(_set_sum);
                    }
                }
                prev_acc = best_eval;
                no_improve_steps = 0;
            } else if (g_max_mode) {
                /* --max: cumulative acceptance window (2026-08-07).
                 * The accumulated gain (best_eval - prev_acc) has NOT reached
                 * gain_threshold yet — either the step accepted a sub-
                 * threshold member (buffered in best_cand_used) or accepted
                 * nothing. The window counter counts EVERY step since the
                 * last commit (positive or dry) and resets only at a commit;
                 * stop only after beam_width steps whose accumulated sum
                 * stayed below min-gain. This keeps the bit-mass gains that
                 * the strict per-step stop would cut off, without letting
                 * pure noise through (plan 2026-08-07). */
                no_improve_steps++;
                if (no_improve_steps >= beam_width) {
                    printf("\n  [Early stop] accumulated gain %.2f%% stayed below "
                           "min-gain %.2f%% for %d steps — stop\n",
                           _gain, gain_threshold, no_improve_steps);
                    break;
                }
            } else {
                /* No candidate beat best_eval by > min-gain → immediate stop. */
                printf("\n  [Early stop] gain %.2f%% <= min-gain %.2f%% — stop (member %d adds nothing\n",
                       _gain, gain_threshold, step + 1 + en_offset);
                break;
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

    /* ── FINAL RE-EVALUATION over the ACCEPTED SET (best_used) ─────────
     * best_eval so far holds the eval of the LAST ACCEPTED BEAM CANDIDATE
     * (exp[ei].eval = sum over beam[si].sum — the candidate's PATH). The
     * beam mixes slots, so that path may contain members that were never in
     * the accepted sequence (best_used), which is what member.out is written
     * from. The reported eval then disagrees with member.out (e.g. 97.40%
     * reported vs 97.02% from the written members — bug 2026-08-02, THIRD
     * recurrence of bug-2026-07-30 "best_used incomplete"). The structural
     * fix: best_used is the SINGLE source of truth — re-evaluate the sum of
     * exactly those members so REPORT == member.out always. The beam stays
     * the search algorithm (finds the set); the eval comes from the set.
     * BEST-SET (FIX 2026-08-10): in --max mode use the HIGHEST-eval set that
     * was committed (best_set_best) — the LAST commit can be worse (BV
     * --max 12: 11 members 88.72% vs 12 members 88.62%). */
    if (g_max_mode && best_set_eval > 0.0f) {
        memcpy(best_used, best_set_best, (size_t)n);
        best_n = best_set_n;
        real_n = best_n;
        best_eval = best_set_eval;
    } else if (best_n > 0) {
        SCORE_TYPE *final_sum = (SCORE_TYPE *)calloc(score_sz, sizeof(SCORE_TYPE));
        if (final_sum) {
            for (int i = 0; i < n; i++) {
                if (!best_used[i]) continue;
                const SCORE_TYPE *sc = blocks[i].scores;
                for (size_t j = 0; j < score_sz; j++) final_sum[j] += sc[j];
            }
            int _fok = 0;
            for (int s = 0; s < g_n_test; s++) {
                const SCORE_TYPE *row = final_sum + (size_t)s * row_sz;
                if (eval_sample_correct(row, g_labels[s])) _fok++;
            }
            best_eval = (float)_fok * 100.0f / (float)eval_denom();
            free(final_sum);
        }
    }

    /* Update global best (for multi-try) */
    if (best_eval > g_best_eval || g_best_used == NULL) {
        g_best_eval = best_eval;
        g_best_n    = real_n;
        if (!g_best_used) g_best_used = (uint8_t *)calloc((size_t)n, 1);
        for (int i = 0; i < n; i++) g_best_used[i] = best_used[i];
        /* INTENTIONAL (2026-08-14): Additionsreihenfolge des besten Ensembles
         * sichern — --debug-member iteriert darüber, damit acc[%] kumulativ
         * in der Beam-Additionsreihenfolge steigt. Menge == g_best_used
         * (accepted[] enthält genau die akzeptierten Member), nur die
         * Sortierung unterscheidet sich von der Block-Reihenfolge des
         * member-out files. Im --max-Modus kann die Best-Set-Kopie
         * (best_set_best) von accepted[] abweichen — dann zeigt die Tabelle
         * die accepted[]-Reihenfolge der finalen best_used-Menge
         * (best_set_best ist bereits in best_used kopiert; accepted[] enthält
         * die Additionsreihenfolge der Kandidaten). */
        free(g_best_order);
        g_best_order = NULL;
        if (accepted_n > 0) {
            g_best_order = (int *)malloc((size_t)accepted_n * sizeof(int));
            if (g_best_order) {
                memcpy(g_best_order, accepted, (size_t)accepted_n * sizeof(int));
                g_best_order_n = accepted_n;
            } else {
                g_best_order_n = 0;
            }
        } else {
            g_best_order_n = 0;
        }
    }

    int best_corr = (int)(best_eval * (float)eval_denom() / 100.0f + 0.5f);
    best_n = real_n;

    if (g_tries > 1) {
        /* Multi-try: the caller (main) prints the final GLOBAL BEST and the
         * machine-readable REPORT (global best over ALL tries). Print only a
         * compact per-try line here — otherwise the REPORT of the LAST try
         * shadows the global best in run-grep (bug 2026-07-31: --tries 5
         * reported the last try's eval instead of the best). */
        printf("  ── TRY best: %d members  acc=%.2f%%  (%d/%d) ──\n",
               best_n, best_eval, best_corr, eval_denom());
    } else {
        printf("\n══╡ BEAM BEST ╞════════════════════════════════════════════\n");
        printf("  Best:       %d members  acc=%.2f%%  (%d/%d)  [slot %d]\n",
               best_n, best_eval, best_corr, eval_denom(), best_slot_idx);
        printf("  Beam:       width=%d  blocks=%d%s\n",
                beam_width, n, g_max_mode ? "  --max (cumulative min-gain)" : "");

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
               best_eval, eval_denom() - best_corr, eval_denom(), best_n, beam_width,
               _last_label);
        if (save_f) { fclose(save_f); printf("  Saved:      %s\n", save_path); }
    }
    for (int i = 0; i < n_beam; i++) free(beam[i].sum);
    for (int i = 0; i < n_beam; i++) free(beam[i].used);
    free(beam); free(trial); free(exp); free(best_used); free(best_cand_used); free(pending); free(_owned_exclude);
    free(accepted);   /* Additionsreihenfolge (2026-08-14) */
    /* FIX (2026-08-10): free the best-set-tracking buffer (was leaked —
     * ASan: 1885 bytes in merge_and_beam, Z.2376). */
    free(best_set_best);
    free(start_mask);

    if (g_tries <= 1) {
        struct timeval _now; gettimeofday(&_now, NULL);
        double _ms = (double)(_now.tv_sec - _step_start.tv_sec) * 1000.0
                   + (double)(_now.tv_usec - _step_start.tv_usec) / 1000.0;
        /* lr= carries --beam-bestN (unused by merge-ensemble) for reproducibility */
        ki_report_show(0, 0, best_corr, eval_denom(), (int)_ms, omp_get_max_threads(),
                       eval_denom() - best_corr, (float)g_beam_bestN, best_n, n_blocks);
    }

    return g_n_test;
}

/* ── --optimal: 2-opt exchange on the beam result (plan-2026-08-08 Stage 2) ──
 * Starts from g_best_used (the beam's add-only attractor) and allows REMOVAL:
 * for each (m ∈ S, m' ∉ S) test err(S − m + m') < err(S); apply the best
 * improvement; repeat until a full pass yields no improvement (fixpoint).
 * Monotonically convergent (err strictly decreases), deterministic — a real
 * formalism, not Monte-Carlo sampling. Updates g_best_used/g_best_n/g_best_eval
 * in place so the caller's member-out/REPORT reflect the exchanged set. */
static void merge_and_exchange(int beam_width)
{
    if (n_blocks == 0) return;
    if (!g_best_used || g_best_n == 0) {
        fprintf(stderr, "  [WARN] --optimal: no beam result to exchange (run --beam first)\n");
        return;
    }
    int has_labels = (g_labels != NULL);
    int n = n_blocks;
    size_t score_sz = (size_t)g_n_test * (size_t)g_n_classes;
    size_t row_sz   = (size_t)g_n_classes;

    /* ── Build cumulative score sum of the current solution S ── */
    SCORE_TYPE *sum = (SCORE_TYPE *)calloc(score_sz, sizeof(SCORE_TYPE));
    if (!sum) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
    int cur_n = 0;
    for (int m = 0; m < n; m++) {
        if (!g_best_used[m]) continue;
        const SCORE_TYPE *sc = blocks[m].scores;
        for (size_t i = 0; i < score_sz; i++) sum[i] += sc[i];
        cur_n++;
    }

    /* ── Current err(S) ──
     * FIX (2026-08-12): count errors with the ACTIVE evaluation semantics
     * (--target/--filter-sample: only the target-class samples count, and
     * only exact pred==label hits are correct). Previously this used plain
     * argmax over g_n_test — --optimal was target-blind, producing
     * nonsensical evals (881.30% / -18.90%) when combined with --target.
     * FIX 2 (2026-08-12): the FIRST fix still counted a cur_err++ for EVERY
     * sample where eval_sample_correct() returns 0 — but under --target k
     * that is ALL non-target samples (9000 of 10000). cur_err started at
     * ~9299 while the swap scorer (eval_denom() - ok, max 1000) compared
     * e < cur_err → almost every swap was accepted, degrading the real
     * target-err (88% → 70% in the --target 6 run). Now cur_err counts ONLY
     * target-class errors, exactly like the swap scorer: start at
     * eval_denom() (all target samples wrong) and subtract each correct hit. */
    int _den0 = eval_denom();
    int cur_err = _den0;
    for (int s = 0; s < g_n_test && has_labels; s++) {
        const SCORE_TYPE *row = sum + (size_t)s * row_sz;
        if (eval_sample_correct(row, g_labels[s])) cur_err--;
    }
    if (!has_labels) cur_err = _den0; /* no labels → cannot improve */

    struct timeval _t0; gettimeofday(&_t0, NULL);
    printf("\n");
    printf("══╡ OPTIMAL EXCHANGE (2-opt, removal allowed) ╞════════════════════\n");
    printf("  Start:   %d members  err=%d  (beam attractor)\n", cur_n, cur_err);
    printf("  Pass  members  err     gain   time[s]\n");
    printf("  ----  -------  ------- -----  --------\n");

    int pass = 0;
    /* Pre-sort the member indices once (S / not-S). collapse(2) over the two
     * sorted arrays keeps the parallel loop balanced and spec-conformant —
     * `continue` inside a collapsed loop is only legal in the innermost
     * level, so the `if (!g_best_used[m]) continue;` guards from the first
     * version were rewritten into plain array indices (2026-08-09). */
    int *in_S  = (int *)malloc((size_t)cur_n * sizeof(int));
    int *out_S = (int *)malloc((size_t)(n - cur_n) * sizeof(int));
    if (!in_S || !out_S) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
    int n_in = 0, n_out = 0;
    for (int m = 0; m < n; m++) {
        if (g_best_used[m]) in_S[n_in++] = m;
        else                out_S[n_out++] = m;
    }
    for (;;) {
        pass++;
        /* Passes cap (--optimal-passes N, 2026-08-12): stop after N passes
         * even if a further improvement might exist — runtime control for
         * very large ensembles. The fixpoint check below still applies. */
        if (g_optimal_passes > 0 && pass > g_optimal_passes) {
            printf("  [stop] pass limit %d reached (fixpoint not proven)\n",
                   g_optimal_passes);
            break;
        }
        /* Best-improvement search: for each (m ∈ S, m' ∉ S) compute
         * err(S − m + m') = err((sum − sc_m) + sc_m') on the fly — no trial
         * buffer, same trick as the beam expansion (base+sc in the argmax
         * loop). Parallel over (m,m') with per-thread local bests + ONE
         * critical compare (same race-safe pattern as merge_and_greedy). */
        int best_m = -1, best_mp = -1;
        int best_err = cur_err;
        /* FIRST-improvement (2026-08-12): a shared flag lets all threads stop
         * scanning once ANY improving swap is found. The first thread to find
         * one claims it (critical guard best_m<0). This is intentionally
         * thread-order-dependent (unlike best-improvement, which is fully
         * deterministic) — it explores a different local optimum faster.
         * `continue` is legal here: collapse(2) makes the oo-loop the
         * innermost, and the comment at the pre-sort explains the rule. */
        volatile int _found = 0;
        #pragma omp parallel
        {
            int l_m = -1, l_mp = -1, l_err = cur_err;
            /* INTENTIONAL: NO `nowait` here — the implicit barrier after the
             * loop is REQUIRED so all threads have written their local bests
             * before the critical compare. With nowait, the fastest thread
             * reached the critical first, read best_err=cur_err and reported
             * best_m=-1 — the 2-opt silently found NO improvement at all
             * (fixed 2026-08-09). */
            #pragma omp for collapse(2) schedule(guided)
            for (int ii = 0; ii < n_in; ii++) {
                for (int oo = 0; oo < n_out; oo++) {
                    if (g_optimal_mode == 1 && _found) continue;  /* first-mode: stop when any found */
                    int m  = in_S[ii];
                    int mp = out_S[oo];
                    const SCORE_TYPE *scm = blocks[m].scores;
                    const SCORE_TYPE *scp = blocks[mp].scores;
                    int ok = 0;
                    for (int s = 0; s < g_n_test && has_labels; s++) {
                        size_t rs = (size_t)s * row_sz;
                        int bk = 0;
                        SCORE_TYPE bv = sum[rs] - scm[rs] + scp[rs];
                        for (int k = 1; k < g_n_classes; k++) {
                            SCORE_TYPE v = sum[rs + k] - scm[rs + k] + scp[rs + k];
                            if (v > bv) { bv = v; bk = k; }
                        }
                        /* FIX (2026-08-12): target-aware — the candidate is
                         * correct only under the active eval semantics
                         * (--target k: label==k && pred==k). Previously plain
                         * bk==label made --optimal target-blind. */
                        if (eval_sample_correct_pred(bk, g_labels[s])) ok++;
                    }
                    int e = has_labels ? eval_denom() - ok : cur_err;
                    if (g_optimal_mode == 1) {
                        /* FIRST-improvement: claim the FIRST strict improvement
                         * (thread-order-dependent, see the _found comment). */
                        if (e < cur_err) {
                            #pragma omp critical
                            {
                                if (best_m < 0) { best_m = m; best_mp = mp; best_err = e; }
                            }
                            _found = 1;   /* signal all threads to stop */
                        }
                    } else {
                        if (e < l_err) { l_err = e; l_m = m; l_mp = mp; }
                    }
                }
            }
            if (g_optimal_mode != 1) {
                #pragma omp critical
                {
                    if (l_err < best_err) { best_err = l_err; best_m = l_m; best_mp = l_mp; }
                }
            }
        }

        if (best_m < 0 || best_err >= cur_err) {
            /* DEBUG: show the best non-improving swap to prove the search ran */
            if (best_m >= 0 && g_debug) {
                char _ml1[64], _ml2[64];
                const char *_xn1 = (blocks[best_m].xform_real >= 0)
                                   ? xform_name_safe(blocks[best_m].xform_real)
                                   : xform_group_label(blocks[best_m].xform_id);
                member_label(&blocks[best_m], _ml1, sizeof(_ml1));
                member_label(&blocks[best_mp], _ml2, sizeof(_ml2));
                printf("  [debug] best swap would be %s:%s (err %d->%d), no improvement\n",
                       _xn1, _ml1, cur_err, best_err);
            }
            break;   /* fixpoint */
        }

        /* Apply the swap: sum = sum − sc_m + sc_mp; update used mask */
        {
            const SCORE_TYPE *scm = blocks[best_m].scores;
            const SCORE_TYPE *scp = blocks[best_mp].scores;
            for (size_t i = 0; i < score_sz; i++) sum[i] += scp[i] - scm[i];
            g_best_used[best_m] = 0;
            g_best_used[best_mp] = 1;
        }
        int gain = cur_err - best_err;
        cur_err = best_err;
        struct timeval _now; gettimeofday(&_now, NULL);
        double _el = (double)(_now.tv_sec - _t0.tv_sec)
                   + (double)(_now.tv_usec - _t0.tv_usec) * 1e-6;
        printf("  %-4d  %-7d  %-7d  %+4d  %.1f\n", pass, cur_n, cur_err, gain, _el);
        fflush(stdout);
        /* Safety: never loop more than the pool size (each swap changes S) */
        if (pass >= n) break;
    }

    /* ── Update globals so member-out/REPORT show the exchanged set ──
     * FIX (2026-08-12): use eval_denom() (the target/filter-sample
     * denominator), NOT g_n_test — the 2-opt err counts only the samples
     * in the active evaluation set (--target k → 1000, not 10000).
     * With g_n_test the REPORT showed eval=881.30% (8813/1000) after
     * --optimal in --target mode. */
    g_best_n = cur_n;
    int _den = eval_denom();
    g_best_eval = has_labels ? (float)(_den - cur_err) * 100.0f / (float)_den
                             : 0.0f;
    int _corr = _den - cur_err;
    struct timeval _now; gettimeofday(&_now, NULL);
    double _ms = (double)(_now.tv_sec - _t0.tv_sec) * 1000.0
               + (double)(_now.tv_usec - _t0.tv_usec) / 1000.0;
    printf("\n══╡ OPTIMAL BEST ╞════════════════════════════════════════════\n");
    printf("  Best:   %d members  acc=%.2f%%  (%d/%d)  [2-opt fixpoint]\n",
           cur_n, g_best_eval, _corr, eval_denom());
    printf("  Report: H=%d  EP=%d  VN=%d  HN=%d  TE=%d  eval=%.2f%%  err=%d  N=%d  best_en=%d  beam=%d  optimal=1\n",
           g_hidden, g_epochs, g_split_vn, g_split_hn, g_target_err_x100,
           g_best_eval, cur_err, eval_denom(), cur_n, beam_width);
    /* lr= carries --beam-bestN (unused by merge-ensemble) for reproducibility */
    ki_report_show(0, 0, _corr, eval_denom(), (int)_ms, omp_get_max_threads(),
                   cur_err, (float)g_beam_bestN, cur_n, n_blocks);
    free(in_S); free(out_S);
    free(sum);
}

static void show_help(const char *prog) {
    printf("Usage: %s DIR [options]\n", prog);
    printf("\n");
    printf("Merge score archives (.ens) to an ensemble accuracy curve (EN=1..N).\n");
    printf("\n");
    printf("Options:\n");
    printf("  DIR               Directory containing .ens score archives\n");
    printf("\n  MAIN ACTION -------------------------------------------------------------------------------------\n");
    printf("  --check           Validate .ens files in DIR (header, size, scores, labels),\n");
    printf("                    detect broken header evals (|header-computed| > 1.0%%), write\n");
    printf("                    a missing .meta, and append missing MAJ fields.\n");
    printf("  --filter L        Filter members (INCLUDE by default, 'not ' = EXCLUDE)\n");
    printf("                    INCLUDE label: --filter sig8,GB=down8  (substring match on\n");
    printf("                    the full XF:CHAN:ENC spec, case-insensitive, comma-separated)\n");
    printf("                    EXCLUDE label: --filter 'not colswap'  ('not ' prefix drops matches;\n");
    printf("                    '!' is an alias for 'not ' — --filter !colswap)\n");
    printf("                    Eval threshold: --filter eval gt 58.1    (gt, lt, ge, le, eq)\n");
    printf("                    Percentile    : --filter eval gt 50%%   ('%%' = N-th percentile of ALL\n");
    printf("                    member evals — 'gt 50%%' keeps the top half, halves the input set)\n");
    printf("                    Regex INCLUDE : --filter regex ':(cbrt8|exp8|...)$'  ('regex' or\n");
    printf("                    'regexp' + POSIX pattern vs the full XF:CHAN:ENC spec)\n");
    printf("                    Regex EXCLUDE : --filter regex '!:(otto|hermann)$'  ('!' or 'not '\n");
    printf("                    after the keyword flips to EXCLUDE; multiple --filter calls AND together)\n");
    printf("  --filter-t1      Named filter: short for\n");
    printf("                    --filter regex '\\<rot[0-9]+@(id|spiral|avg2|avg3|avg4):'\n");
    printf("                    (keep rot-chain members whose pipeline exits via\n");
    printf("                     id/spiral/avg2/avg3/avg4 — 2026-08-15)\n");
    printf("  --target N[,M,...] Optimize the ensemble for the given class(es) only:\n");
    printf("                    correct counts ONLY when label is one of the target\n");
    printf("                    classes AND the prediction hits it exactly. The\n");
    printf("                    denominator becomes the number of target-class\n");
    printf("                    samples (recall, 0-100%% — not diluted by the class\n");
    printf("                    distribution). --member-out exports the subset that\n");
    printf("                    is optimal FOR THESE CLASSES. Examples: --target 3,\n");
    printf("                    --target 3,7. Default (no flag): all classes.\n");
    printf("  --target-binary   With --target k: k vs. ALL other classes as ONE binary\n");
    printf("                    decision (TP+TN = correct, FP+FN = error). Prevents the\n");
    printf("                    \"everything is k\" cheat of plain target recall. 2026-08-14\n");
    printf("  --filter-sample N[,M,...]  SAMPLE-level filter: only test samples whose\n");
    printf("                    LABEL is in the list enter the evaluation (both the\n");
    printf("                    correct count and the denominator). Members are still\n");
    printf("                    ALL loaded — this restricts the evaluation set, not\n");
    printf("                    the member pool. Combinable with --target\n");
    printf("                    (--target 3,7 --filter-sample 3,7 = full recall mode\n");
    printf("                    on the target classes).\n");
    printf("  --seed-sort [N]   Print single-member eval sorted by strength (no search, N=top rows, default: all)\n");
    printf("  --sample-index FILE  Write member↔sample recognition pairs as Parquet\n");
    printf("                    (no search; writes FILE.samples/.members/.pairs.parquet —\n");
    printf("                     queryable via DuckDB read_parquet() or sample-query.py;\n");
    printf("                     pairs = ALL assignments + argmax pred + margin)\n");
    printf("  --greedy          Greedy optimal subset (members sorted by contribution)\n");
    printf("    --greedy-clarity    Greedy variant: among candidates that do NOT worsen\n");
    printf("                        the current ensemble, pick the one with the highest\n");
    printf("                        member clarity (own class-score std-dev) instead of\n");
    printf("                        the best accuracy — prefer unambiguous voters that\n");
    printf("                        disturb other members the least (plan-2026-08-08)\n");
    printf("  --beam N          Beam search with width N (N=10 recommended, better than greedy)\n");
    printf("    --tries N           Repeat beam search N times with shuffle (keeps best)\n");
    printf("      --tries-no-lock     Do NOT lock members found by earlier tries: each\n");
    printf("                          try searches the FULL pool (fresh empty pool per\n");
    printf("                          try) → different attractors in the same space,\n");
    printf("                          not disjoint partitions. Seed variation still\n");
    printf("                          applies (see seed flags below).\n");
    printf("      --tries-top-seed    Seed choice for tries: k-th strongest free member\n");
    printf("                          (Try 1 = strongest, Try 2 = 2nd, ...). Deterministic.\n");
    printf("                          Default.\n");
    printf("      --tries-random-seed Seed choice for tries: splitmix64 random pick\n");
    printf("                          among the free members. Broader diversity, still\n");
    printf("                          reproducible (fixed RNG base per corpus/config).\n");
    printf("    --beam-bestN N      Start the per-level beam selection at the N-th best\n");
    printf("                        candidate instead of the 1st-best (default 1). Breaks\n");
    printf("                        the fixation on the 1st-best member (add-only beam never\n");
    printf("                        re-evaluates an added member as wrong). With --tries N,\n");
    printf("                        try k uses bestN=k (max diversity); the winning bestN is\n");
    printf("                        reported in the unused lr= field (reproducibility).\n");
    printf("    --max               Cumulative acceptance: per-step acceptance\n");
    printf("                        threshold 0.0; the accumulated gain of the\n");
    printf("                        steps since the last commit is committed as\n");
    printf("                        ONE block once it reaches min-gain. Stop only\n");
    printf("                        after beam_width steps whose sum stayed below\n");
    printf("                        min-gain (strict per-member stop is default).\n");
    printf("    --max N             Stop greedy/beam at N members\n");
    printf("    --min-gain F        Minimum gain [%%] to keep adding (default: %.2f)\n", g_min_gain);
    printf("    --member-seed SPEC  Pre-seed beam with member spec (e.g. \"rot22@spiral:BP:sig8\")\n");
    printf("    --member-start FILE  Start the beam from a previously exported\n");
    printf("                        attractor (--member-out format) instead of a single\n");
    printf("                        seed. The saved set is the FLOOR: the beam only ADDS\n");
    printf("                        members, so extending the pool (e.g. new @shuffle\n");
    printf("                        pipelines) can no longer redirect the path below it\n");
    printf("                        (monotone pool extension, Finding 11 fix 2026-08-09).\n");
    printf("\n  RESEARCH ACTION ---------------------------------------------------------------------------------\n");
    printf("  --eff-lambda F    Member penalty for the eff= score in the REPORT\n");
    printf("                    (eff = eval - lambda*(members-1); default 0.02).\n");
    printf("  --expansion-sort MODE  Sort beam expansions (default: abs)\n");
    printf("                    abs : by candidate eval (best absolute accuracy)\n");
    printf("                    marg: by incremental gain of the new member on its\n");
    printf("                          path (eval_with - eval_without) — different heuristic,\n");
    printf("                          may yield a smaller/different ensemble\n");
    printf("                    clarity: by the member's own class-score std-dev\n");
    printf("                          (--seed-sort shows it) — prefer unambiguous\n");
    printf("                          voters that disturb other members the least\n");
    printf("  --optimal [best|first]  2-opt exchange on the beam result: allows REMOVAL\n");
    printf("                    (for each m in S, m' not in S: if err(S-m+m') <\n");
    printf("                    err(S), swap). Breaks the add-only attractor,\n");
    printf("                    monotonically convergent to a fixpoint. Only with\n");
    printf("                    --beam. See: plans/plan-2026-08-08\n");
    printf("                    best  (default): apply the GLOBAL best swap per pass\n");
    printf("                          (deterministic, 88.69->88.73 on BV32/29 members)\n");
    printf("                    first : apply the FIRST improving swap (thread-order-\n");
    printf("                          dependent, faster per pass, may land on a\n");
    printf("                          different local optimum than best)\n");
    printf("    --optimal-passes N  Cap the 2-opt fixpoint search at N passes\n");
    printf("                    (0 = unlimited, run until no improvement; the\n");
    printf("                    fixpoint is then not proven, \"[stop] pass limit\")\n");
    printf("\n  EXPORT ACTION -----------------------------------------------------------------------------------\n");
    printf("  --debug-member    After the beam search, print the detailed list of the\n");
    printf("                    SELECTED members in the trainer's --debug-member format\n");
    printf("                    ([n/N] mem=trn/evl/E= time= xf:chan:enc W0= MIN= MAX= pxz=).\n");
    printf("                    trn%%/E/time/pxz are '-' here (merge has no training phase);\n");
    printf("                    evl%% is the REAL per-member accuracy computed from the\n");
    printf("                    .ens scores vs. ground truth — compare 1:1 with the TRN.\n");
    printf("  --debug-confusion Print the confusion matrix of the FINAL ensemble\n");
    printf("                    (summed member scores → argmax per sample, the same\n");
    printf("                    tprint table as the trainer's --debug-confusion —\n");
    printf("                    merge and TRN outputs compare 1:1, 2026-08-14).\n");
    printf("  --stdout          Pure filter-export: print the filtered member specs\n");
    printf("                    (XF:CHAN:ENC, one per line) to stdout and exit. Combine\n");
    printf("                    with --filter, e.g. '--filter eval gt 50%%' for the top\n");
    printf("                    half — pipe into cut/sort. Suppresses normal output.\n");
    printf("  --encoding / --channel / --xform  With --stdout: print only the ENC,\n");
    printf("                    CHAN or XF column instead of the full spec (e.g.\n");
    printf("                    '--stdout --encoding | run-sort-unique'). --xform with a\n");
    printf("                    value keeps its member-file expansion meaning.\n");
    printf("  --member-out FILE With --check: write the broken members' XF:CHAN:ENC specs\n");
    printf("                    to FILE for TRN --member-file (retrain, no patching). With\n");
    printf("                    --beam: export the optimal subset instead.\n");
    printf("  --member-out-default  Derive member-{DIR}.out from the corpus dir\n");
    printf("                    (e.g. scores-H196-E10-BV8-FLT64 → member-H196-E10-BV8-FLT64.out,\n");
    printf("                    'scores-' prefix stripped). Explicit --member-out wins. 2026-08-16\n");
    printf("  --save FILE       Save cumulative accuracy to FILE (default: DIR/merge.dat)\n");
    printf("\n  USAGE ACTION ------------------------------------------------------------------------------------\n");
    printf("  -h, --help        Show this help text\n");
    printf("\n");
    printf("Output:\n");
    printf("  Table: EN  acc[%%]  correct/total  gain\n");
    printf("  File:  merge.dat  (for plotting)\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s scores/\n", prog);
    printf("  %s scores/ --max 50\n", prog);
    printf("  %s scores/ --greedy\n", prog);
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
typedef struct { int ok; char msg[256];
    /* Cross-file identity (2026-08-12): filled by check_archive so
     * check_directory can detect duplicate internal specs / dimension
     * mismatches across the corpus. */
    char internal_spec[256]; /* "xf:color:enc<width>" from v7 meta ("" if n/a) */
    uint32_t n_test, hidden, epochs, w0_marker;
} CheckResult;

static CheckResult check_archive(const char *path) {
    CheckResult res;
    memset(&res, 0, sizeof(res));

    /* File size */
    struct stat st;
    if (stat(path, &st) != 0) {
        res.ok = -1; snprintf(res.msg, sizeof(res.msg), "stat: %s", strerror(errno));
        return res;
    }
    off_t file_size = st.st_size;

    /* ── Parse via the shared reader (lib/ki-ens.h) — ONE version logic,
     * no inline version dispatch can drift (bug 2026-08-06). ── */
    EnsReader rd;
    if (ens_reader_open(&rd, path) != 0) {
        res.ok = -1; snprintf(res.msg, sizeof(res.msg), "Header unreadable");
        return res;
    }
    uint32_t version   = rd.version;
    uint32_t n_test    = rd.n_test;
    uint32_t n_classes = rd.n_classes;
    uint32_t n_members = rd.n_members;
    uint32_t hidden    = rd.hidden;
    uint32_t epochs    = rd.epochs;
    uint32_t split_vn  = rd.split_vn;
    uint32_t split_hn  = rd.split_hn;
    float    ensemble_eval = rd.ensemble_eval;
    uint32_t w0_check  = rd.w0_marker;
    /* Cross-file identity for check_directory (2026-08-12) */
    res.n_test = n_test; res.hidden = hidden; res.epochs = epochs;
    res.w0_marker = w0_check;

    /* v14+ precision vs .meta (2026-08-12): a MISMATCH is an ERROR — the
     * archive's logits would not sum correctly with the other members.
     * Old archives (v<14) are valid but carry no precision block: reported
     * as OK + a [WARN: ...] text suffix (they have "höheren Zweifel", not
     * a defect — the user decision 2026-08-12). */
    int prec_verdict = 0;    /* 0 = none/ok, 1 = v<14 (warn-text), -1 = mismatch (error) */
    if (version >= 14 && (g_meta_otp != 0 || g_meta_bits != 0 || g_meta_ct[0])) {
        if ((g_meta_otp != 0 && rd.ot_precision != g_meta_otp) ||
            (g_meta_bits != 0 && rd.bit_width    != g_meta_bits) ||
            (g_meta_ct[0] && strcmp(rd.counter_type, g_meta_ct) != 0))
            prec_verdict = -1;
    } else if (version < 14 && (g_meta_otp != 0 || g_meta_bits != 0 || g_meta_ct[0])) {
        prec_verdict = 1;
    }

    if (!ens_version_valid((int)version)) {
        res.ok = -1; snprintf(res.msg, sizeof(res.msg), "Unsupported version %u", version);
        ens_reader_close(&rd); return res;
    }
    if (n_test == 0 || n_test > 1000000) {
        res.ok = -1; snprintf(res.msg, sizeof(res.msg), "Suspicious n_test=%u", n_test);
        ens_reader_close(&rd); return res;
    }
    if (n_classes < 1 || n_classes > 1000) {
        res.ok = -1; snprintf(res.msg, sizeof(res.msg), "Suspicious n_classes=%u", n_classes);
        ens_reader_close(&rd); return res;
    }
    if (n_members < 1 || n_members > 10000) {
        res.ok = -1; snprintf(res.msg, sizeof(res.msg), "Suspicious n_members=%u", n_members);
        ens_reader_close(&rd); return res;
    }

    /* ── Read per-member metadata (2026-08-12: no longer skipped) ──
     * v7+ carries 4 length-prefixed strings: color, enc, width, xform.
     * Validate them AND compare the internal spec against the filename —
     * a mismatch means the archive was written with a racy pipe id (see
     * bugs/bug-2026-08-12-trainer-pipe-registry-race.md) and the merge
     * would load the wrong member via find_member_by_spec(). */
    char fields[4][128]; uint8_t raw[4];
    for (uint32_t m = 0; m < n_members; m++) {
        if (ens_reader_read_member_meta(&rd, fields, raw) != 0) {
            res.ok = -1; snprintf(res.msg, sizeof(res.msg), "Bad metadata");
            ens_reader_close(&rd); return res;
        }
        if (version >= 7) {
            /* A2/A3: unknown xform / color / encoding names */
            if (xform_id_by_name(fields[3]) < 0) {
                snprintf(res.msg + strlen(res.msg), sizeof(res.msg) - strlen(res.msg),
                         "%sunknown xform '%s'", res.msg[0] ? "; " : "", fields[3]);
                if (res.ok <= 0) res.ok = 1;
            }
            if (color_id_by_name(fields[0]) < 0) {
                snprintf(res.msg + strlen(res.msg), sizeof(res.msg) - strlen(res.msg),
                         "%sunknown color '%s'", res.msg[0] ? "; " : "", fields[0]);
                if (res.ok <= 0) res.ok = 1;
            }
            if (enc_id_by_name(fields[1]) < 0) {
                snprintf(res.msg + strlen(res.msg), sizeof(res.msg) - strlen(res.msg),
                         "%sunknown encoding '%s'", res.msg[0] ? "; " : "", fields[1]);
                if (res.ok <= 0) res.ok = 1;
            }
            /* A4: implausible encoding width */
            int ew = atoi(fields[2]);
            if (ew != 8 && ew != 16 && ew != 32) {
                snprintf(res.msg + strlen(res.msg), sizeof(res.msg) - strlen(res.msg),
                         "%simplausible enc_width '%s'", res.msg[0] ? "; " : "", fields[2]);
                if (res.ok <= 0) res.ok = 1;
            }
            /* A1: filename vs internal spec — the pipe-race signature.
             * filename: "xf:color:enc<width>" without the ".ens" suffix. */
            const char *base = strrchr(path, '/');
            base = base ? base + 1 : path;
            char file_spec[256], int_spec[256];
            {
                size_t blen = strlen(base);
                size_t cut = (blen >= 4 && strcmp(base + blen - 4, ".ens") == 0) ? 4 : 0;
                size_t cp = blen - cut;
                if (cp >= sizeof(file_spec)) cp = sizeof(file_spec) - 1;
                memcpy(file_spec, base, cp);
                file_spec[cp] = '\0';
            }
            snprintf(int_spec, sizeof(int_spec), "%.63s:%.63s:%.63s%.63s",
                     fields[3], fields[0], fields[1], fields[2]);
            /* Cross-file identity for the duplicate-spec check */
            snprintf(res.internal_spec, sizeof(res.internal_spec), "%.250s", int_spec);
            if (strcmp(file_spec, int_spec) != 0) {
                snprintf(res.msg + strlen(res.msg), sizeof(res.msg) - strlen(res.msg),
                         "%sname mismatch (file '%s' vs internal '%s')",
                         res.msg[0] ? "; " : "", file_spec, int_spec);
                if (res.ok <= 0) res.ok = 1;
            }
        }
    }

    /* ── Expected file size (all accounting from the reader) ── */
    off_t expected = rd.hdr_bytes + rd.meta_bytes + rd.scores_bytes + rd.labels_bytes;
    if (file_size != expected) {
        snprintf(res.msg, sizeof(res.msg), "Size mismatch: file=%lld expected=%lld (diff%+lld)",
                 (long long)file_size, (long long)expected, (long long)(file_size - expected));
        if (file_size < expected) { res.ok = -1; ens_reader_close(&rd); return res; }
        res.ok = 1;  /* warn but continue */
    }

    /* ── Read and validate scores (first member only, then sample rest) ── */
    int all_zero = 1, has_outliers = 0;
    int64_t vmin = INT64_MAX, vmax = INT64_MIN;
    size_t score_sz = rd.score_sz;

    float *fsc = (float *)malloc(score_sz * sizeof(float));
    if (!fsc) { res.ok = -1; snprintf(res.msg, sizeof(res.msg), "OOM"); ens_reader_close(&rd); return res; }
    for (uint32_t m = 0; m < n_members && m < 5; m++) {
        if (ens_reader_read_scores_float(&rd, fsc) != 0) {
            snprintf(res.msg, sizeof(res.msg), "Short scores (member %u)", m);
            free(fsc); res.ok = -1; ens_reader_close(&rd); return res;
        }
        for (size_t i = 0; i < score_sz; i++) {
            int64_t v = (int64_t)fsc[i];
            if (v != 0) all_zero = 0;
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
            /* Outlier: |value| > 2^40 ≈ 1e12 (reasonable for accumulated scores) */
            if (v > ((int64_t)1 << 40) || v < -((int64_t)1 << 40)) has_outliers = 1;
        }
    }
    free(fsc);
    /* Skip remaining members (not read) — elem-aware per-member skip */
    for (uint32_t m = 5; m < n_members; m++)
        if (ens_reader_skip_scores(&rd) != 0) {
            snprintf(res.msg, sizeof(res.msg), "Short scores (member %u)", m);
            res.ok = -1; ens_reader_close(&rd); return res;
        }

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
    if ((off_t)ftell(rd.f) + (off_t)n_test <= file_size) {
        uint8_t *labels = (uint8_t *)malloc(n_test);
        if (labels) {
            if (fread(labels, 1, n_test, rd.f) == n_test) {
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

    ens_reader_close(&rd);

    /* Build final status message */
    char dims[256];
    if (version >= 14) {
        snprintf(dims, sizeof(dims), "v%u  %u×%u  H=%u  EP=%u  VN=%u  HN=%u  %u member%s  EVL=%.1f%%  W0=0x%08X  otp=%d bits=%d ct=%.15s  scores=[%+" PRId64 ",%+" PRId64 "]",
                 version, n_test, n_classes, hidden, epochs, split_vn, split_hn,
                 n_members, n_members == 1 ? "" : "s", ensemble_eval, w0_check,
                 rd.ot_precision, rd.bit_width, rd.counter_type, vmin, vmax);
    } else {
        snprintf(dims, sizeof(dims), "v%u  %u×%u  H=%u  EP=%u  VN=%u  HN=%u  %u member%s  EVL=%.1f%%  W0=0x%08X  scores=[%+" PRId64 ",%+" PRId64 "]",
                 version, n_test, n_classes, hidden, epochs, split_vn, split_hn,
                 n_members, n_members == 1 ? "" : "s", ensemble_eval, w0_check, vmin, vmax);
    }
    /* Precision verdict: mismatch → ERROR; v<14 → OK + [WARN: ...] text */
    if (prec_verdict == -1) {
        snprintf(res.msg + strlen(res.msg), sizeof(res.msg) - strlen(res.msg),
                 "%sprecision mismatch vs .meta (otp=%d bits=%d ct=%.15s)",
                 res.msg[0] ? "; " : "",
                 rd.ot_precision, rd.bit_width, rd.counter_type);
        res.ok = -1;
    } else if (prec_verdict == 1) {
        snprintf(res.msg + strlen(res.msg), sizeof(res.msg) - strlen(res.msg),
                 "%sold format (v<14), no precision block — cannot verify vs .meta "
                 "BITS/OT_PRECISION (archive stays valid, retrain to get v14)",
                 res.msg[0] ? "; " : "");
        /* res.ok stays 0 → tag OK, the WARN is text-only */
    }
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
    /* Cross-file identity tables (2026-08-12) — bounded by the corpus size */
    enum { MAX_FILES = 8192 };
    char (*specs)[256] = (char (*)[256])calloc((size_t)MAX_FILES, sizeof(char[256]));
    uint32_t *dims = (uint32_t *)calloc((size_t)MAX_FILES, 4 * sizeof(uint32_t));
    char (*fnames)[256] = (char (*)[256])calloc((size_t)MAX_FILES, sizeof(char[256]));
    int n_specs = 0;
    if (!specs || !dims || !fnames) {
        fprintf(stderr, "[ERROR] check_directory OOM\n");
        free(specs); free(dims); free(fnames); closedir(d); return 1;
    }

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

        /* Collect identity for the cross-file checks */
        if (r.internal_spec[0] && n_specs < MAX_FILES) {
            snprintf(specs[n_specs], 256, "%s", r.internal_spec);
            dims[n_specs * 4 + 0] = r.n_test;
            dims[n_specs * 4 + 1] = r.hidden;
            dims[n_specs * 4 + 2] = r.epochs;
            dims[n_specs * 4 + 3] = r.w0_marker;
            snprintf(fnames[n_specs], 256, "%s", de->d_name);
            n_specs++;
        }
    }
    closedir(d);

    /* ── B1: duplicate internal specs (the pipe-race signature) ── */
    int dup_errors = 0;
    for (int i = 0; i < n_specs; i++) {
        for (int j = i + 1; j < n_specs; j++) {
            if (strcmp(specs[i], specs[j]) == 0) {
                printf("  ERROR  %-40s  duplicate internal spec '%s' (also %s) — "
                       "merge loads the wrong member\n",
                       fnames[i], specs[i], fnames[j]);
                dup_errors++;
                n_err++;
            }
        }
    }

    /* ── B2: dimension inconsistency (n_test/H/EP) across the corpus ── */
    if (n_specs > 0) {
        uint32_t b_nt = dims[0], b_h = dims[1], b_ep = dims[2];
        for (int i = 1; i < n_specs; i++) {
            if (dims[i*4+0] != b_nt || dims[i*4+1] != b_h || dims[i*4+2] != b_ep) {
                printf("  ERROR  %-40s  dimension mismatch: n_test=%u H=%u EP=%u "
                       "vs corpus %u/%u/%u\n",
                       fnames[i], dims[i*4+0], dims[i*4+1], dims[i*4+2],
                       b_nt, b_h, b_ep);
                n_err++;
            }
        }
        /* ── B3: majority/W0 mixing (Otto w0!=0 vs Bit-Voting w0==0) ── */
        uint32_t b_w0 = dims[3];
        for (int i = 1; i < n_specs; i++) {
            if ((dims[i*4+3] == 0) != (b_w0 == 0)) {
                printf("  ERROR  %-40s  W0-marker mixing: file w0=0x%08X vs corpus 0x%08X "
                       "(Otto + Bit-Voting in one corpus)\n",
                       fnames[i], dims[i*4+3], b_w0);
                n_err++;
            }
        }
    }

    free(specs); free(dims); free(fnames);

    printf("\n══╡ CHECK SUMMARY ╞══════════════════════════════════════════════\n");
    printf("  %d .ens file(s):  %d OK,  %d WARN,  %d ERROR%s%s\n",
           n_files, n_ok, n_warn, n_err,
           dup_errors ? "  (+duplicate specs)" : "",
           n_specs > 1 ? "" : "  (cross-file checks need >=2 files)");
    return n_err > 0 ? 1 : 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * CHECK REPAIRS (--check) — broken-eval detection + .meta + member-out
 * ═══════════════════════════════════════════════════════════════════════
 * Problem (2026-07-31): some v7 archives written by the old trainer carry
 * ensemble_eval = 100.0 in the header (train/eval mix-up). The BEAM table
 * displays this header value as "eval[%]" (via ScoreBlock.file_eval), which
 * is misleading — e.g. colswap-2-4@spiral:BP:sig8 shows 100.0 in the beam
 * but the true single-member accuracy (scores vs. labels) is only 40.54%.
 *
 * Strategy (user decision 2026-07-31): do NOT patch the .ens binaries.
 * Instead, detect the broken members (|header eval - computed eval| > 1.0%)
 * and write their XF:CHAN:ENC specs to the --member-out file. The trainer
 * (TRN) re-reads that file via --member-file and RETRAINS the members
 * (with --force it overwrites the broken archives in place). Recomputing
 * is faster and absolutely safe — no binary is modified by this tool.
 *
 * Also writes DIR/.meta from the first archive's header if it is missing.
 * The MAJ/MAJ1_THRESH fields are ALWAYS written (the trainer's .meta format
 * requires them): from the v11 header if present, otherwise the trainer
 * defaults (maj=1, thresh=-2) — the archives are v10 or older and carry no
 * majority info, but the current trainer default is maj 1, so this is the
 * safe assumption (mixing a maj3 member into this dir would then WARN).
  * Also appends missing MAJ fields to an existing .meta written by an older
  * binary (no MAJ line present).
  * ─────────────────────────────────────────────────────────────────────── */

/* Detect a broken header eval: open the archive once, parse the header,
 * read all member scores, compute the true eval (argmax vs labels).
 * Returns 1 when |header - computed| > 1.0% (broken), 0 when ok, -1 on error.
 * *header_eval_out / *computed_out receive the values (may be NULL).
 * PRECISION CHECK (v14+, 2026-08-12): when the archive carries the precision
 * block and .meta specifies BITS/OT_PRECISION/COUNTER_TYPE, a mismatch sets
 * *prec_mismatch_out = 1 (caller reports it as broken — the scores would not
 * sum correctly with the other archives). *prec_mismatch_out may be NULL. */
static int archive_detect_broken(const char *path, const uint8_t *labels,
                                 int n_test, int n_classes,
                                 float *header_eval_out, float *computed_out,
                                 int *prec_mismatch_out) {
    if (prec_mismatch_out) *prec_mismatch_out = 0;
    EnsReader rd;
    if (ens_reader_open(&rd, path) != 0) return -1;
    int version = (int)rd.version;
    float header_eval = rd.ensemble_eval;
    /* v14+ precision vs .meta (both must carry the fields for a check) */
    if (prec_mismatch_out && version >= 14 &&
        (g_meta_otp != 0 || g_meta_bits != 0 || g_meta_ct[0])) {
        if ((g_meta_otp != 0 && rd.ot_precision != g_meta_otp) ||
            (g_meta_bits != 0 && rd.bit_width    != g_meta_bits) ||
            (g_meta_ct[0] && strcmp(rd.counter_type, g_meta_ct) != 0)) {
            *prec_mismatch_out = 1;
        }
    } else if (prec_mismatch_out && version < 14 &&
               (g_meta_otp != 0 || g_meta_bits != 0 || g_meta_ct[0])) {
        /* OLD archive (v<14) in a directory whose .meta carries precision:
         * the file has no precision block — cannot verify it fits. Report as
         * a WARN via *prec_mismatch_out = 2 (NOT broken — the eval is still
         * valid; only the precision provenance is unknown). */
        *prec_mismatch_out = 2;
    }
    if (rd.n_test != (uint32_t)n_test || rd.n_classes != (uint32_t)n_classes) {
        ens_reader_close(&rd); return -1;
    }
    size_t score_sz = rd.score_sz;
    float *sum = (float *)calloc(score_sz, sizeof(float));
    if (!sum) { ens_reader_close(&rd); return -1; }
    if (ens_reader_skip_all_meta(&rd) != 0) { free(sum); ens_reader_close(&rd); return -1; }
    float *buf = (float *)malloc(score_sz * sizeof(float));
    if (!buf) { free(sum); ens_reader_close(&rd); return -1; }
    for (uint32_t m = 0; m < rd.n_members; m++) {
        if (ens_reader_read_scores_float(&rd, buf) != 0) {
            free(buf); free(sum); ens_reader_close(&rd); return -1;
        }
        for (size_t i = 0; i < score_sz; i++) sum[i] += buf[i];
    }
    free(buf);
    ens_reader_close(&rd);

    int correct = 0;
    for (int s = 0; s < n_test; s++) {
        const float *row = sum + (size_t)s * (size_t)n_classes;
        int pred = 0;
        for (int k = 1; k < n_classes; k++)
            if (row[k] > row[pred]) pred = k;
        if (pred == (int)labels[s]) correct++;
    }
    free(sum);
    float computed = (float)correct * 100.0f / (float)n_test;
    if (header_eval_out) *header_eval_out = header_eval;
    if (computed_out)    *computed_out    = computed;
    return (version >= 4 && fabsf(header_eval - computed) > 1.0f) ? 1 : 0;
}

/* Extract ground-truth labels from an archive that embeds them (v8+ files
 * are header + scores + n_test label bytes; v7 files end after the scores).
 * Returns 1 and allocates *labels_out on success. */
static int archive_get_labels(const char *path, uint8_t **labels_out,
                              int *n_test_out, int *n_classes_out) {
    EnsReader rd;
    if (ens_reader_open(&rd, path) != 0) return 0;
    if (ens_reader_skip_all_meta(&rd) != 0) { ens_reader_close(&rd); return 0; }
    for (uint32_t m = 0; m < rd.n_members; m++)
        if (ens_reader_skip_scores(&rd) != 0) { ens_reader_close(&rd); return 0; }
    uint8_t *lab = (uint8_t *)malloc((size_t)rd.n_test);
    if (!lab) { ens_reader_close(&rd); return 0; }
    if (fread(lab, 1, (size_t)rd.n_test, rd.f) != (size_t)rd.n_test) {
        free(lab); ens_reader_close(&rd); return 0;
    }
    uint32_t n_test = rd.n_test, n_classes = rd.n_classes;
    ens_reader_close(&rd);
    *labels_out = lab;
    *n_test_out = (int)n_test;
    *n_classes_out = (int)n_classes;
    return 1;
}

/* Write DIR/.meta from the first valid archive's header, if missing.
 * The MAJ/MAJ1_THRESH fields are ALWAYS written (the trainer's .meta format
 * requires them): from the v11 header if present, otherwise the trainer
 * defaults (maj=1, thresh=-2) — the archives are v10 or older and carry no
 * majority info, but the current trainer default is maj 1, so this is the
 * safe assumption (mixing a maj3 member into this dir would then WARN).
 * Also appends missing MAJ fields to an existing .meta that was written by
 * an older binary (no MAJ line present).
 * Returns 1 if .meta was created or modified, 0 otherwise. */
static int fix_write_meta(const char *dir) {
    char meta_path[1024];
    snprintf(meta_path, sizeof(meta_path), "%s/.meta", dir);

    /* First archive's header values (MAJ only known for v11+) */
    uint32_t m_h = 0, m_ep = 0, m_vn = 0, m_hn = 0, m_seed = 0;
    char     m_maj[8] = "";      /* "" = no v11 archive seen */
    uint32_t m_majt = 0;
    DIR *d = opendir(dir);
    if (!d) return 0;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        const char *ext = strrchr(de->d_name, '.');
        if (!ext || strcmp(ext, ".ens") != 0) continue;
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dir, de->d_name);
        EnsReader rd;
        if (ens_reader_open(&rd, path) == 0) {
            m_h = rd.hidden; m_ep = rd.epochs; m_vn = rd.split_vn;
            m_hn = rd.split_hn; m_seed = rd.seed;
            if (rd.maj_token[0]) {
                snprintf(m_maj, sizeof(m_maj), "%s", rd.maj_token);
                m_majt = (uint32_t)rd.maj1_thresh;
            }
            ens_reader_close(&rd);
            break;
        }
    }
    closedir(d);
    if (m_h == 0) return 0;   /* no valid archive found */

    /* MAJ values: v11 header if available, else trainer defaults.
     * KI_MAJ_1 / maj1_thresh=-2 (see mlp-bin32-otto-trn-seq.c config). */
    char maj_line[64], majt_line[64];
    if (m_maj[0]) {
        snprintf(maj_line,  sizeof(maj_line),  "MAJ=%s", m_maj);
        snprintf(majt_line, sizeof(majt_line), "MAJ1_THRESH=%u", m_majt);
    } else {
        snprintf(maj_line,  sizeof(maj_line),  "MAJ=1");
        snprintf(majt_line, sizeof(majt_line), "MAJ1_THRESH=-2");
    }

    FILE *mf = fopen(meta_path, "r");
    if (mf) {
        /* Existing .meta: append MAJ fields if missing (older binary) */
        char line[128];
        int has_maj = 0;
        while (fgets(line, sizeof(line), mf))
            if (strncmp(line, "MAJ=", 4) == 0) { has_maj = 1; break; }
        fclose(mf);
        if (has_maj) return 0;   /* complete already */

        mf = fopen(meta_path, "a");
        if (!mf) return 0;
        fprintf(mf, "%s\n%s\n", maj_line, majt_line);
        fclose(mf);
        printf("  [META] appended %s %s to %s%s\n", maj_line, majt_line,
               meta_path, m_maj[0] ? "" : "  (assumed defaults: archives v10 or older)");
        return 1;
    }

    /* Create fresh .meta */
    mf = fopen(meta_path, "w");
    if (!mf) return 0;
    fprintf(mf, "H=%u\nEPOCHS=%u\nVN=%u\nHN=%u\nSEED=%u\n%s\n%s\n",
            m_h, m_ep, m_vn, m_hn, m_seed, maj_line, majt_line);
    fclose(mf);
    printf("  [META] created %s (H=%u EP=%u VN=%u HN=%u SEED=%u %s %s%s)\n",
           meta_path, m_h, m_ep, m_vn, m_hn, m_seed, maj_line, majt_line,
           m_maj[0] ? "" : "  (assumed defaults: archives v10 or older)");
    return 1;
}

/* ── CHECK REPAIRS entry point (--check) ── */
static int fix_directory(const char *dir) {
    printf("\n══╡ CHECK — broken member detection ╞════════════════════════════\n");
    /* Load .meta identity (incl. BITS/OT_PRECISION/COUNTER_TYPE) so the
     * v14+ precision check in archive_detect_broken can validate against it. */
    meta_load_globals(dir);
    fix_write_meta(dir);

    /* Collect labels from the first archive that embeds them (v8+) */
    uint8_t *labels = NULL;
    int n_test = 0, n_classes = 0;
    DIR *d = opendir(dir);
    if (!d) { printf("  [CHECK] cannot open %s\n", dir); return 0; }
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        const char *ext = strrchr(de->d_name, '.');
        if (!ext || strcmp(ext, ".ens") != 0) continue;
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dir, de->d_name);
        if (archive_get_labels(path, &labels, &n_test, &n_classes)) break;
    }
    closedir(d);
    if (!labels) {
        printf("  [CHECK] no archive contains labels — cannot detect broken evals\n");
        return 0;
    }

    /* Safety: verify label consistency against a second archive that also
     * embeds labels. If they differ, the test sets are not identical and
     * recomputed evals would be garbage — abort without writing member-out. */
    int verified = 0;
    d = opendir(dir);
    int first_match = 1;
    while ((de = readdir(d)) != NULL) {
        const char *ext = strrchr(de->d_name, '.');
        if (!ext || strcmp(ext, ".ens") != 0) continue;
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dir, de->d_name);
        uint8_t *lab2 = NULL;
        int nt2 = 0, nc2 = 0;
        if (archive_get_labels(path, &lab2, &nt2, &nc2)) {
            if (nt2 == n_test && nc2 == n_classes && first_match) {
                /* this is the archive we already took labels from */
                first_match = 0;
                free(lab2);
                continue;
            }
            if (nt2 == n_test && memcmp(lab2, labels, (size_t)n_test) == 0)
                verified = 1;
            free(lab2);
            if (verified) break;
        }
    }
    closedir(d);
    if (!verified) {
        printf("  [CHECK] ABORT: labels of two label-carrying archives differ — "
               "test sets are not identical, not writing member-out.\n");
        free(labels);
        return 0;
    }
    printf("  [CHECK] labels (%d samples) verified against a second archive\n", n_test);

    /* Open --member-out for the broken member specs (XF:CHAN:ENC, the format
     * TRN reads via --member-file; '#' comment lines are skipped there). */
    FILE *mo = NULL;
    if (g_member_out[0]) {
        mo = fopen(g_member_out, "w");
        if (mo) {
            fprintf(mo, "# members to recompute via TRN --member-file (2026-07-31):\n");
            fprintf(mo, "#   header=stored header eval  computed=scores+labels eval\n");
            fprintf(mo, "#   PRECISION-MISMATCH / OLD FORMAT (v<14) — retrain to get v14/v15\n");
        } else {
            fprintf(stderr, "  [WARN] cannot open --member-out %s\n", g_member_out);
        }
    }

    int n_broken = 0, n_ok = 0, n_err = 0;
    d = opendir(dir);
    while ((de = readdir(d)) != NULL) {
        const char *ext = strrchr(de->d_name, '.');
        if (!ext || strcmp(ext, ".ens") != 0) continue;
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dir, de->d_name);
        float he = 0.0f, ce = 0.0f;
        int pm = 0;
        int br = archive_detect_broken(path, labels, n_test, n_classes, &he, &ce, &pm);
        if (br < 0) { n_err++; continue; }
        if (pm == 2) {
            /* OLD archive (v<14) in a precision-carrying .meta dir: the file
             * has no precision block, so its provenance cannot be verified.
             * The file stays VALID (shown as OK+[WARN:] by check_archive) —
             * but to close the loop (next check reports nothing), write the
             * spec to --member-out so a TRN retrain upgrades it to v14. */
            if (mo) {
                char spec[256];
                snprintf(spec, sizeof(spec), "%s", de->d_name);
                size_t sl = strlen(spec);
                if (sl > 4 && strcmp(spec + sl - 4, ".ens") == 0) spec[sl - 4] = '\0';
                fprintf(mo, "# %s  OLD FORMAT (v<14) — no precision block, retrain to get v14\n%s\n",
                        spec, spec);
            }
            n_ok++;
            continue;
        }
        if (br || pm) {
            n_broken++;
            if (mo) {
                /* Spec = filename without ".ens" (the .ens files ARE named
                 * XF:CHAN:ENC — no reconstruction needed). */
                char spec[256];
                snprintf(spec, sizeof(spec), "%s", de->d_name);
                size_t sl = strlen(spec);
                if (sl > 4 && strcmp(spec + sl - 4, ".ens") == 0) spec[sl - 4] = '\0';
                if (pm) {
                    fprintf(mo, "# %s  PRECISION-MISMATCH (otp/bits/counter_type vs .meta)\n%s\n",
                            spec, spec);
                    printf("  [BROKEN] %-40s PRECISION-MISMATCH vs .meta (otp/bits/counter_type)\n",
                           spec);
                } else {
                    fprintf(mo, "# %s  header=%.2f%%  computed=%.2f%%\n%s\n",
                            spec, he, ce, spec);
                    printf("  [BROKEN] %-40s header=%.2f%%  computed=%.2f%%\n",
                           spec, he, ce);
                }
            }
        } else n_ok++;
    }
    closedir(d);
    if (mo) { fclose(mo); mo = NULL; }
    free(labels);
    if (n_err > 0)
        printf("  [CHECK] %d file(s) could not be parsed\n", n_err);
    if (g_member_out[0])
        printf("  [CHECK] broken evals: %d of %d  (specs written to %s)\n",
               n_broken, n_broken + n_ok, g_member_out);
    else
        printf("  [CHECK] broken evals: %d of %d  (use --member-out FILE to "
               "write the specs for TRN --member-file)\n", n_broken, n_broken + n_ok);
    return 0;
}


/* ── Strict numeric-argument guard (2026-08-11) ─────────────────────
 * The num/float flags (--beam, --tries, --beam-bestN, --min-gain,
 * --eff-lambda) REQUIRE a value. Before the fix they consumed the next
 * token unconditionally via atoi(argv[++i]) — even when it was another
 * flag (e.g. "--beam --filter eval gt 30%" swallowed "--filter" as the
 * beam width, atoi("--filter")=0, silently disabling the filter).
 * Now: the next token must be a numeric literal, otherwise the run
 * aborts with an error. is_num_arg() accepts:
 *   ints:    leading digit (0-9)
 *   floats:  optional sign, then digit or '.'
 * (a value of "-1" or "-0.5" is numeric here; range checks downstream
 *  still reject negatives where they are not allowed). */
static int is_num_arg(const char *s, int allow_float) {
    if (!s || !*s) return 0;
    if (*s == '+' || *s == '-') s++;
    if (!*s) return 0;
    if (allow_float) {
        return (*s >= '0' && *s <= '9') || *s == '.';
    }
    return (*s >= '0' && *s <= '9');
}

int main(int argc, char **argv) {
    /* merge-ensemble defines aa zero-initialized (no struct initializer) —
     * set the eff-lambda default explicitly (--eff-lambda can override). */
    aa.eff_lambda = 0.02f;
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
    /* --completion [--<flag>]: bash auto-completion (merge table) */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--completion") == 0) {
            merge_completion_dispatch(argc, argv, &i);
        }
    }

    /* DIR = first non-option, non-value token */
    const char *dir = NULL;
    const char *save_path = NULL;
    int greedy = 0;
    int beam_width = 0;
    int check_mode = 0;

    /* Parse all options in any order; DIR = first bare token */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--greedy") == 0) {
            greedy = 1;
        } else if (strcmp(argv[i], "--greedy-clarity") == 0) {
            greedy = 1;
            g_greedy_clarity = 1;
        } else if (strcmp(argv[i], "--beam") == 0) {
            /* STRICT (2026-08-11): --beam REQUIRES a numeric value. Before,
             * "--beam --filter ..." swallowed the next flag as the width. */
            if (i + 1 < argc && is_num_arg(argv[i + 1], 0)) {
                beam_width = atoi(argv[++i]);
                if (beam_width < 1) beam_width = 1;
            } else {
                fprintf(stderr, "[ERROR] --beam requires a numeric argument\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--tries") == 0) {
            if (i + 1 < argc && is_num_arg(argv[i + 1], 0)) {
                g_tries = atoi(argv[++i]);
                if (g_tries < 1) g_tries = 1;
            } else {
                fprintf(stderr, "[ERROR] --tries requires a numeric argument\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--tries-no-lock") == 0) {
            g_tries_no_lock = 1;
        } else if (strcmp(argv[i], "--tries-top-seed") == 0) {
            g_tries_seed = 0;
        } else if (strcmp(argv[i], "--tries-random-seed") == 0) {
            g_tries_seed = 1;
        } else if (strcmp(argv[i], "--beam-bestN") == 0) {
            if (i + 1 < argc && is_num_arg(argv[i + 1], 0)) {
                g_beam_bestN = atoi(argv[++i]);
                if (g_beam_bestN < 1) g_beam_bestN = 1;
            } else {
                fprintf(stderr, "[ERROR] --beam-bestN requires a numeric argument\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--max") == 0) {
            /* --max allein = Beam-as-a-Whole Scan (alter Formalismus);
             * --max N = max member count.
             * FIX (2026-08-10): --max N must ALSO enable the cumulative
             * acceptance mode (g_max_mode) — before this, --max 12 only set
             * the cap and ran in STRICT mode (each member must beat min-gain
             * individually), which stopped early (6 members instead of 12)
             * while --max alone gave 26. The two forms must share the same
             * acceptance algorithm; N only caps the result. */
            if (i + 1 < argc && argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9') {
                g_max_members = atoi(argv[++i]);
                if (g_max_members < 1) g_max_members = 1;
                g_max_mode = 1;
            } else {
                g_max_mode = 1;
            }
        } else if (strcmp(argv[i], "--min-gain") == 0) {
            /* STRICT (2026-08-11): value required — see is_num_arg() */
            if (i + 1 < argc && is_num_arg(argv[i + 1], 1)) {
                g_min_gain = (float)atof(argv[++i]);
                if (g_min_gain < 0.0f) g_min_gain = 0.0f;
            } else {
                fprintf(stderr, "[ERROR] --min-gain requires a numeric argument\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--target") == 0) {
            /* Target classes (comma list, e.g. "3,7"): optimize the
             * ensemble as if only these classes existed. Value required. */
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                const char *spec = argv[++i];
                g_target_count = 0;
                char buf[64];
                snprintf(buf, sizeof(buf), "%.63s", spec);
                for (char *tok = strtok(buf, ","); tok && g_target_count < 10;
                     tok = strtok(NULL, ",")) {
                    int c = atoi(tok);
                    /* NOTE: g_n_classes is 0 during CLI parsing (set later
                     * at archive load) — validate against the compile-time
                     * constant KI_NCLASSES here. */
                    if (c < 0 || c >= KI_NCLASSES) {
                        fprintf(stderr, "[ERROR] --target: class '%s' out of range 0..%d\n",
                                tok, KI_NCLASSES - 1);
                        return 1;
                    }
                    g_target_classes[g_target_count++] = c;
                }
                if (g_target_count == 0) {
                    fprintf(stderr, "[ERROR] --target requires a class list (e.g. 3,7)\n");
                    return 1;
                }
                g_target_active = 1;
            } else {
                fprintf(stderr, "[ERROR] --target requires a class list (e.g. 3,7)\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--target-binary") == 0) {
            /* BINARY target mode: --target k becomes a k-vs-all decision.
             * TP (label==k, pred==k) and TN (label!=k, pred!=k) are both
             * correct; FP (label!=k, pred==k) is now an ERROR — prevents
             * the "everything is k" degenerate specialist (2026-08-14). */
            g_target_binary = 1;
            if (!g_target_active) {
                fprintf(stderr, "[ERROR] --target-binary requires --target (e.g. --target 6 --target-binary)\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--filter-sample") == 0) {
            /* Sample-level filter: only samples whose LABEL is in the
             * list enter the evaluation (correct AND denominator).
             * Value required. */
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                const char *spec = argv[++i];
                g_filter_sample_count = 0;
                char buf[64];
                snprintf(buf, sizeof(buf), "%.63s", spec);
                for (char *tok = strtok(buf, ","); tok && g_filter_sample_count < 10;
                     tok = strtok(NULL, ",")) {
                    int c = atoi(tok);
                    /* NOTE: g_n_classes is 0 during CLI parsing (set later
                     * at archive load) — validate against the compile-time
                     * constant KI_NCLASSES here. */
                    if (c < 0 || c >= KI_NCLASSES) {
                        fprintf(stderr, "[ERROR] --filter-sample: class '%s' out of range 0..%d\n",
                                tok, KI_NCLASSES - 1);
                        return 1;
                    }
                    g_filter_sample_classes[g_filter_sample_count++] = c;
                }
                if (g_filter_sample_count == 0) {
                    fprintf(stderr, "[ERROR] --filter-sample requires a class list (e.g. 0,9)\n");
                    return 1;
                }
                g_filter_sample_active = 1;
            } else {
                fprintf(stderr, "[ERROR] --filter-sample requires a class list (e.g. 0,9)\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--eff-lambda") == 0) {
            if (i + 1 < argc && is_num_arg(argv[i + 1], 1)) {
                aa.eff_lambda = (float)atof(argv[++i]);
                if (aa.eff_lambda < 0.0f) {
                    fprintf(stderr, "[ERROR] --eff-lambda must be >= 0\n");
                    return 1;
                }
            } else {
                fprintf(stderr, "[ERROR] --eff-lambda requires a numeric argument\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--debug") == 0) {
            g_debug = 1;
        } else if (strcmp(argv[i], "--debug-member") == 0) {
            g_debug_member = 1;
        } else if (strcmp(argv[i], "--debug-confusion") == 0) {
            g_debug_confusion = 1;
        } else if (strcmp(argv[i], "--seed-sort") == 0) {
            g_seed_sort = 1;
            /* Optional limit: "--seed-sort N" (top N rows). Without N → show all. */
            if (i + 1 < argc && argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9') {
                g_seed_top = atoi(argv[++i]);
                if (g_seed_top < 1) g_seed_top = 1;
            }
        } else if (strcmp(argv[i], "--stdout") == 0) {
            g_stdout = 1;
        } else if (strcmp(argv[i], "--sample-index") == 0 && i + 1 < argc) {
            strncpy(g_sample_index, argv[++i], sizeof(g_sample_index) - 1);
            g_sample_index[sizeof(g_sample_index) - 1] = '\0';
        } else if (strcmp(argv[i], "--member-seed") == 0 && i + 1 < argc) {
            strncpy(g_member_seed_spec, argv[++i], sizeof(g_member_seed_spec) - 1);
            g_member_seed_spec[sizeof(g_member_seed_spec) - 1] = '\0';
        } else if (strcmp(argv[i], "--member-start") == 0 && i + 1 < argc) {
            strncpy(g_start_member_file, argv[++i], sizeof(g_start_member_file) - 1);
            g_start_member_file[sizeof(g_start_member_file) - 1] = '\0';
        } else if (strcmp(argv[i], "--member-out") == 0 && i + 1 < argc) {
            strncpy(g_member_out, argv[++i], sizeof(g_member_out) - 1);
            g_member_out[sizeof(g_member_out) - 1] = '\0';
        } else if (strcmp(argv[i], "--member-out-default") == 0) {
            /* Derive member-{DIR}.out from the corpus dir (2026-08-16):
             * "scores-H196-E10-BV8-FLT64" → "member-H196-E10-BV8-FLT64.out"
             * (strips a leading "scores-" prefix). Resolved after parsing
             * (needs `dir`). Overridden by an explicit --member-out. */
            g_member_out_default = 1;
        } else if (strcmp(argv[i], "--xform") == 0) {
            /* --xform SPEC (value) = member-file expansion (existing); 
             * --xform (no value, next arg is a flag) = stdout XF column */
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                strncpy(g_xform_spec, argv[++i], sizeof(g_xform_spec) - 1);
                g_xform_spec[sizeof(g_xform_spec) - 1] = '\0';
            } else {
                g_stdout_field = 1;
            }
        } else if (strcmp(argv[i], "--encoding") == 0) {
            g_stdout_field = 3;   /* ENC column in --stdout output */
        } else if (strcmp(argv[i], "--channel") == 0) {
            g_stdout_field = 2;   /* CHAN column in --stdout output */
        } else if (strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
            save_path = argv[++i];
        } else if (strcmp(argv[i], "--num") == 0 && i + 1 < argc) {
            fprintf(stderr, "[WARN] --num is obsolete (ignored)\n"); i++;
        } else if (strcmp(argv[i], "--expansion-sort") == 0 && i + 1 < argc) {
            const char *_mode = argv[++i];
            if (strcmp(_mode, "marg") == 0 || strcmp(_mode, "marginal") == 0) {
                g_expansion_sort = 1;
            } else if (strcmp(_mode, "clarity") == 0 || strcmp(_mode, "clr") == 0) {
                g_expansion_sort = 2;
            } else if (strcmp(_mode, "abs") == 0) {
                g_expansion_sort = 0;
            } else {
                fprintf(stderr, "[ERROR] --expansion-sort: expected 'abs', 'marg' or 'clarity', got '%s'\n", _mode);
                return 1;
            }
        } else if (strcmp(argv[i], "--optimal") == 0) {
            /* 2-opt exchange (2026-08-12): optional mode token —
             *   --optimal         = best-improvement (default, deterministic)
             *   --optimal best    = same
             *   --optimal first   = first-improvement (different local optimum)
             * A following non-flag token is consumed as the mode; a flag or
             * end-of-args leaves the default (best). */
            g_optimal = 1;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                const char *_om = argv[++i];
                if (strcmp(_om, "first") == 0)      g_optimal_mode = 1;
                else if (strcmp(_om, "best") == 0)  g_optimal_mode = 0;
                else {
                    fprintf(stderr, "[ERROR] --optimal: expected 'best' or 'first', got '%s'\n", _om);
                    return 1;
                }
            }
        } else if (strcmp(argv[i], "--optimal-passes") == 0) {
            if (i + 1 < argc && is_num_arg(argv[i + 1], 0)) {
                g_optimal_passes = atoi(argv[++i]);
                if (g_optimal_passes < 1) g_optimal_passes = 1;
            } else {
                fprintf(stderr, "[ERROR] --optimal-passes requires a numeric argument\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--check") == 0) {
            check_mode = 1;
        } else if (strcmp(argv[i], "--filter-t1") == 0) {
            /* Benannter Filter (2026-08-15): Kurzform für
             *   --filter regex '\<rot[0-9]+@(id|spiral|avg2|avg3|avg4):'
             * Behält nur Members, deren XF:CHAN:ENC-Spec mit einer rot-Kette
             * beginnt, deren Pipeline-Ausgang id/spiral/avg2/avg3/avg4 ist.
             * Nutzt denselben g_filter_re[]-Pfad wie --filter regex — die
             * Filter-Semantik (member_is_filtered) ist identisch. Additiv zu
             * weiteren --filter-Aufrufen (kein Reset). */
            if (g_filter_re_count >= 64) {
                fprintf(stderr, "[ERROR] too many regex filters\n");
                return 1;
            }
            g_filter_re_not[g_filter_re_count] = 0;
            snprintf(g_filter_re[g_filter_re_count],
                     sizeof(g_filter_re[0]),
                     "\\<rot[0-9]+@(id|spiral|avg2|avg3|avg4):");
            g_filter_re_count++;
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
            /* ADDITIVE: multiple --filter calls combine (AND across the filter
             * kinds: eval threshold, label exclude, regex include). Do NOT reset
             * g_eval_active/g_filter_count here — a second --filter (e.g. regex)
             * would otherwise wipe the eval filter of the first one. */
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
                    /* '%'-Suffix after the value → PERCENTILE filter: compare
                     * each member's eval against the N-th percentile of ALL
                     * loaded member evals (e.g. "gt 50%" = top half, "lt 25%"
                     * = weakest quarter). Without '%' the threshold is absolute
                     * ("gt 50" = eval > 50). */
                    g_eval_pct = 0;
                    {   const char *_pe = p;
                        while (*_pe && ((*_pe >= '0' && *_pe <= '9') ||
                                        *_pe == '.' || *_pe == ' ' || *_pe == '\t')) _pe++;
                        if (*_pe == '%') g_eval_pct = 1;
                    }
                    g_eval_active = 1;
                    continue;
                }

                /* --filter regex 'PATTERN': POSIX regex on the full
                 * XF:CHAN:ENC spec ("regex " word + space-separated pattern).
                 * 'regexp' accepted as an alias for 'regex'. A leading '!'
                 * after the keyword flips to EXCLUDE:
                 *   regex ':(cbrt8|log8)$'   → INCLUDE matches (default)
                 *   regex '!:(otto|hermann)$' → EXCLUDE matches
                 * 'not ' (space-separated) is an alias for '!'
                 *   regex not ':(otto)$'      → EXCLUDE matches (2026-08-09) */
                if ((strncasecmp(tok, "regex ", 6) == 0 ||
                     strncasecmp(tok, "regexp ", 7) == 0) && g_filter_re_count < 64) {
                    const char *pat = tok + ((strncasecmp(tok, "regexp ", 7) == 0) ? 7 : 6);
                    int _not = 0;
                    if (*pat == '!') { _not = 1; pat++; }
                    else if (strncasecmp(pat, "not ", 4) == 0) { _not = 1; pat += 4; }
                    while (*pat == ' ' || *pat == '\t') pat++;
                    if (*pat) {
                        strncpy(g_filter_re[g_filter_re_count], pat,
                                sizeof(g_filter_re[0]) - 1);
                        g_filter_re[g_filter_re_count][sizeof(g_filter_re[0]) - 1] = '\0';
                        g_filter_re_not[g_filter_re_count] = _not;
                        g_filter_re_count++;
                    }
                    continue;
                }

                /* Label pattern (substring match, case-insensitive).
                 * Default INCLUDE; '!' prefix flips to EXCLUDE:
                 *   'colswap'   → INCLUDE: keep only colswap members
                 *   '!colswap'  → EXCLUDE: drop all colswap members
                 * 'not ' (space-separated) is an alias for '!':
                 *   'not shuffle' → EXCLUDE: drop all shuffle members
                 * (2026-08-09, user request: `--filter not shuffle` instead
                 * of `--filter '!shuffle'`.) */
                {
                    const char *pt = tok;
                    int _not = 0;
                    if (*pt == '!') { _not = 1; pt++; }
                    else if (strncasecmp(pt, "not ", 4) == 0) { _not = 1; pt += 4; }
                    while (*pt == ' ' || *pt == '\t') pt++;
                    if (*pt) {
                        strncpy(g_filter_pat[g_filter_count], pt,
                                sizeof(g_filter_pat[0]) - 1);
                        g_filter_pat[g_filter_count][sizeof(g_filter_pat[0]) - 1] = '\0';
                        g_filter_pat_not[g_filter_count] = _not;
                        g_filter_count++;
                    }
                }
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

    /* --member-out-default: derive member-{DIR}.out from the corpus dir
     * (2026-08-16). "scores-H196-E10-BV8-FLT64" → "member-H196-E10-BV8-FLT64.out"
     * (a leading "scores-" prefix is stripped). An explicit --member-out
     * always wins (checked here, AFTER parsing). */
    if (g_member_out_default && !g_member_out[0]) {
        const char *_base = strrchr(dir, '/');
        _base = _base ? _base + 1 : dir;
        const char *_tag = _base;
        if (strncmp(_base, "scores-", 7) == 0) _tag = _base + 7;
        snprintf(g_member_out, sizeof(g_member_out), "member-%s.out", _tag);
        printf("  Member-out:  %s  (--member-out-default)\n", g_member_out);
    }

#if 0
    /* Default save path — static buffer so save_path stays valid for the
     * whole run (stack-use-after-scope: was a block-local char[] that
     * dangled when merge_and_beam later fopen()'d it — ASan, 2026-08-02). */
    static char _default_save[1024];
    if (!save_path) {
        snprintf(_default_save, sizeof(_default_save), "%s/merge.dat", dir);
        save_path = _default_save;
    }
#endif

    /* Check or load */
    if (check_mode) {
        /* Load .meta identity first — both check_directory (precision
         * verdict in check_archive) and fix_directory need it. */
        meta_load_globals(dir);
        int rc = check_directory(dir);
        fix_directory(dir);
        return rc;
    }

    if (!g_stdout) printf("══╡ SETUP MERGE ╞══════════════════════════════════════════\n");
    /* Load score archives */
    int nf = load_directory(dir);
    if (nf == 0) {
        fprintf(stderr, "[ERROR] No .ens files found in %s\n", dir);
        return 1;
    }
    if (!g_stdout) printf("  Loaded %d archive files (%d total score blocks)\n", nf, n_blocks);
    if (!g_stdout) {
        /* Bit-Voting (w0_marker==0) never applies a majority — the maj=
         * token in the headers is the trainer default, not a real config.
         * Detect it via w0_marker (works for old and new BV archives). */
        if (g_ens_w0_zero == 1) {
            printf("  Majority:    n/a — Bit-Voting (w0_marker=0, linear, no majority)\n");
        } else if (g_ens_maj[0]) {
            printf("  Majority:    maj=%s  maj1-thresh=%d  (from .ens v11 headers)\n",
                   g_ens_maj, g_ens_majt);
        } else if (nf > 0) {
            printf("  Majority:    (not recorded — archives are v10 or older)\n");
        }
    }
    dedup_blocks();

    /* ── Percentile eval filter: "--filter eval gt 50%" etc. ──
     * Applied AFTER loading, when all member evals are known. Keeps the
     * members whose file_eval passes the operator against the N-th percentile
     * threshold ("gt 50%" = top half → halves the input set for TRN
     * --member-file). Absolute thresholds ("gt 50") were applied during load. */
    if (g_eval_active && g_eval_pct && n_blocks > 0) {
        int _k = n_blocks;
        float *_ev = (float *)malloc((size_t)_k * sizeof(float));
        for (int i = 0; i < _k; i++) _ev[i] = blocks[i].file_eval;
        qsort(_ev, (size_t)_k, sizeof(float), cmp_float);
        int _idx = (int)((double)_k * (double)g_eval_thresh / 100.0);
        if (_idx < 0) _idx = 0;
        if (_idx >= _k) _idx = _k - 1;
        float _thr = _ev[_idx];
        free(_ev);
        const char *op_str = "gt";
        switch (g_eval_cmp) {
            case 0: op_str = "gt"; break;
            case 1: op_str = "ge"; break;
            case 2: op_str = "lt"; break;
            case 3: op_str = "le"; break;
            case 4: op_str = "eq"; break;
        }
        int _w = 0;
        for (int i = 0; i < _k; i++) {
            int pass = 0;
            switch (g_eval_cmp) {
                case 0: pass = (blocks[i].file_eval >  _thr); break;
                case 1: pass = (blocks[i].file_eval >= _thr); break;
                case 2: pass = (blocks[i].file_eval <  _thr); break;
                case 3: pass = (blocks[i].file_eval <= _thr); break;
                case 4: pass = (fabsf(blocks[i].file_eval - _thr) < 0.001f); break;
            }
            if (pass) { if (_w != i) blocks[_w] = blocks[i]; _w++; }
            else free(blocks[i].scores);
        }
        if (!g_stdout)
            printf("  [FILTER] eval %s %.0f%% (percentile of %d): keeping %d members (thresh=%.1f%%)\n",
                   op_str, g_eval_thresh, _k, _w, _thr);
        n_blocks = _w;
    }
    if (!g_stdout) {
        printf("  Expansion sort: %s\n",
               g_expansion_sort == 2 ? "clarity (member class-score std-dev)"
               : g_expansion_sort ? "marginal (new-member gain)"
                                  : "abs (candidate eval)");
        bool done=false;
        if (g_filter_count > 0) {
            done=true;
            printf("  Filter:");
            for (int i = 0; i < g_filter_count; i++)
                printf(" %s%s", g_filter_pat_not[i] ? "not " : "", g_filter_pat[i]);
        }
        if (g_filter_re_count > 0) {
            done=true;
            printf("  Filter:");
            for (int i = 0; i < g_filter_re_count; i++)
                printf(" %sregex %s", g_filter_re_not[i] ? "!" : "", g_filter_re[i]);
        }
        if (g_eval_active) {
            done=true;
            const char *op_str = "gt";
            switch (g_eval_cmp) {
                case 0: op_str = "gt"; break;
                case 1: op_str = "ge"; break;
                case 2: op_str = "lt"; break;
                case 3: op_str = "le"; break;
                case 4: op_str = "eq"; break;
            }
            printf("  eval %s %.1f%s", op_str, g_eval_thresh, g_eval_pct ? "% (percentile)" : "");
        }
        if (done) printf("\n");
    }

    /* ── --stdout: pure filter-export mode ──
     * Print the filtered member specs (XF:CHAN:ENC, one per line, NO comments)
     * to stdout and exit — usable as a pipe source (e.g. cut/sort). All normal
     * output was suppressed above (2026-08-01). */
    if (g_stdout) {
        for (int i = 0; i < n_blocks; i++) {
            char _lbl[64];
            const char *_xn = (blocks[i].xform_real >= 0)
                              ? xform_name_safe(blocks[i].xform_real)
                              : xform_group_label(blocks[i].xform_id);
            member_label(&blocks[i], _lbl, sizeof(_lbl));
            if (g_stdout_field == 0) {
                printf("%s:%s\n", _xn, _lbl);   /* full XF:CHAN:ENC spec */
            } else {
                /* Column extraction from "XF:CHAN:ENC" */
                char _spec[128];
                snprintf(_spec, sizeof(_spec), "%s:%s", _xn, _lbl);
                char *_c1 = strchr(_spec, ':');
                char *_c2 = _c1 ? strchr(_c1 + 1, ':') : NULL;
                if (g_stdout_field == 1 && _c1) {          /* xform */
                    *_c1 = '\0';
                    printf("%s\n", _spec);
                } else if (g_stdout_field == 2 && _c1 && _c2) {  /* channel */
                    *_c2 = '\0';
                    printf("%s\n", _c1 + 1);
                } else if (g_stdout_field == 3 && _c2) {  /* encoding */
                    printf("%s\n", _c2 + 1);
                }
            }
        }
        return 0;
    }

    if (g_seed_sort) {
        /* Seed-sort runs PURELY on the .index cache: blocks[].file_eval now
         * holds the REAL member eval (computed once during the rebuild, see
         * read_ens_meta). No scores / no g_labels are needed — this works even
         * before load_scores_directory() runs (2026-08-01). */
        typedef struct { int idx; float eval; } _SeedSc;
        _SeedSc *_ss2 = (_SeedSc *)malloc((size_t)n_blocks * sizeof(_SeedSc));
        if (_ss2) {
            for (int _mi = 0; _mi < n_blocks; _mi++) {
                _ss2[_mi].idx = _mi;
                _ss2[_mi].eval = blocks[_mi].file_eval;
            }
            /* Sort by eval desc (threshold-based tiebreak to avoid -Wfloat-equal) */
            for (int _i = 0; _i < n_blocks - 1; _i++)
                for (int _j = _i + 1; _j < n_blocks; _j++)
                    if (_ss2[_j].eval > _ss2[_i].eval + 0.0001f) {
                        _SeedSc _t = _ss2[_i]; _ss2[_i] = _ss2[_j]; _ss2[_j] = _t;
                    }
            printf("\n── SEED SORT (%d singles by strength, real evals from .index)%s ──\n",
                   n_blocks, g_seed_top > 0 ? ", top rows shown below" : "");
            printf("  %-4s  %-7s  %-15s  %s\n", "rank", "acc[%]", "block", "member");
            printf("  %-4s  %-7s  %-15s  %s\n", "----", "-------", "-------------", "-------------------");
            int _shown = 0;
            for (int _i = 0; _i < n_blocks; _i++) {
                if (g_seed_top > 0 && _shown >= g_seed_top) break;
                int _mi = _ss2[_i].idx;
                char _ml[64];
                const char *_xn = (blocks[_mi].xform_real >= 0)
                    ? xform_name_safe(blocks[_mi].xform_real)
                    : xform_group_label(blocks[_mi].xform_id);
                member_label(&blocks[_mi], _ml, sizeof(_ml));
                printf("  %-4d  %-7.2f  %-15d  %s:%s\n",
                       _i + 1, _ss2[_i].eval, _mi, _xn, _ml);
                _shown++;
            }
            printf("── End seed sort (%d rows shown) ──\n\n", _shown);
        }
        free(_ss2);
        return 0;  /* seed-sort is inspection-only, no search */
    }
    /* ── Scores on demand ──
     * The .index path delivered metadata only (scores==NULL) so the filters
     * above never touched the .ens files. Remaining modes — debug CRC, beam,
     * greedy, merge-eval — need the real scores: load them now, grouped by
     * source_file (each archive opened exactly once). --seed-sort returned
     * above (real evals come from the index). Progress is shown inside
     * load_scores_directory as a single line of dots. */
    load_scores_directory(dir);

    /* ── --sample-index: eigenständiger Check-Modus (2026-08-15) ──
     * Schreibt die Member↔Sample-Zuordnung als 3 Parquet-Dateien und
     * beendet sich danach — kein Beam/Greedy. Nutzt die geladenen Scores
     * (post-filter) und g_labels. Reihenfolge ist egal (Member-IDs sind
     * Namens-Hashes; der Parquet-Writer sortiert via block_order). */
    if (g_sample_index[0]) {
        int rc = build_sample_index(g_sample_index);
        return rc;
    }

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

    /* Sort blocks deterministically before beam/greedy/eval.
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
            /* Single try: pass an internal pool_exclude so the beam enforces
             * the global one-use-per-member diversity rule exactly like a
             * multi-try run (TRY 1). On large pools this finds better
             * attractors — M3: 67.70% without the rule vs 69.36% with it
             * (bug 2026-07-31). Auto-seed (best single) + no fill = TRY 1
             * start state. */
            uint8_t *_pe = (uint8_t *)calloc((size_t)n_blocks, 1);
            merge_and_beam(save_path, beam_width, member_seed_idx, _pe);
            free(_pe);
            if (g_optimal) merge_and_exchange(beam_width);
            if (g_debug_member)
                print_debug_members(g_best_used, g_best_order, g_best_order_n, g_best_n, n_blocks, g_best_eval);
            if (g_debug_confusion)
                print_ensemble_confusion(g_best_used, n_blocks);
            if (g_member_out[0]) {
                export_member_file(g_member_out, g_best_used, g_best_n, g_best_eval, beam_width);
                backup_member_to_dir(dir, g_member_out);
            }
        } else {
            /* Multi-try: jeder Try sucht den stärksten noch freien Member als Seed */
            char saved_member_out[1024];
            strncpy(saved_member_out, g_member_out, sizeof(saved_member_out));
            g_member_out[0] = '\0';

            int _n_tries = g_tries < n_blocks ? g_tries : n_blocks;
            if (_n_tries < 1) _n_tries = 1;
            printf("\n  Multi-try: %d tries%s%s\n", _n_tries,
                   g_tries_no_lock ? "  (no-lock: full pool per try)" : "",
                   g_tries_seed ? "  (random-seed)" : "  (top-seed)");
            struct timeval _mt0; gettimeofday(&_mt0, NULL);
            size_t _row_sz = (size_t)g_n_classes;
            /* --tries-random-seed: deterministic RNG base so a given corpus +
             * config always reproduces the same seed sequence (splitmix64). */
            if (g_tries_seed)
                w0_srandom(0x9E3779B97F4A7C15ULL ^ (uint64_t)(uint32_t)g_hidden
                           * 2654435761ULL ^ (uint64_t)(uint32_t)g_epochs
                           * 40503ULL ^ (uint64_t)(uint32_t)g_tries);
            {
                uint8_t *pool_exclude = (uint8_t *)calloc((size_t)n_blocks, 1);
                int n_active_tries = 0;
                float _global_best_eval = -1.0f;
                int   _global_best_n = 0;
                uint8_t *_global_best_used = NULL;
                int   _global_best_order_n = 0;   /* Additionsreihenfolge (2026-08-14) */
                int  *_global_best_order = NULL;

                /* If --member-seed specified, force it as first try's seed */
                int _seed_idx = -1;
                if (member_seed_idx >= 0) {
                    if (pool_exclude[member_seed_idx])
                        fprintf(stderr, "  [WARN] --member-seed %d is already excluded\n", member_seed_idx);
                    _seed_idx = member_seed_idx;
                }

                for (int ti = 0; ti < _n_tries; ti++) {
                    /* --beam-bestN per try: try k uses bestN=k (1..N) — maximum
                     * diversity across tries. The seed still varies (top/random),
                     * but the per-level candidate offset adds a second, orthogonal
                     * diversity source. The winning bestN is recorded for the
                     * REPORT lr= field (reproducibility). */
                    g_beam_bestN = (ti % _n_tries) + 1;
                    /* Seed finden: stärksten Member aus dem Pool der Noch-Nicht-Genutzten.
                     * --tries-no-lock: pool_exclude hält NUR die bereits als Seed
                     * benutzten Member (Locking-Block unten markiert g_best_used NICHT)
                     * → top-seed = k-ter stärkster freier Member (Try 1 = stärkster,
                     * Try 2 = 2.-stärkster, ...), deterministisch. */
                    if (ti > 0 || member_seed_idx < 0) {
                        _seed_idx = -1;
                        if (g_tries_seed) {
                            /* random-seed: splitmix64 pick among the free members */
                            int _free_count = 0;
                            for (int mi = 0; mi < n_blocks; mi++)
                                if (!pool_exclude[mi]) _free_count++;
                            if (_free_count > 0) {
                                uint32_t _r = w0_random() % (uint32_t)_free_count;
                                int _c = 0;
                                for (int mi = 0; mi < n_blocks; mi++) {
                                    if (pool_exclude[mi]) continue;
                                    if (_c++ == (int)_r) { _seed_idx = mi; break; }
                                }
                            }
                        } else {
                            float _best_seed_eval = -1.0f;
                            for (int mi = 0; mi < n_blocks; mi++) {
                                if (pool_exclude[mi]) continue;
                                const SCORE_TYPE *sc = blocks[mi].scores;
                                if (!sc) continue;
                                int _c = 0;
                                for (int s = 0; s < g_n_test; s++) {
                                    const SCORE_TYPE *_r = sc + (size_t)s * _row_sz;
                                    if (eval_sample_correct(_r, g_labels[s])) _c++;
                                }
                                float _eval = (float)_c * 100.0f / (float)eval_denom();
                                if (_eval > _best_seed_eval) {
                                    _best_seed_eval = _eval;
                                    _seed_idx = mi;
                                }
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
                            if (eval_sample_correct(_r, g_labels[s])) _c++;
                        }
                        _seed_eval = (float)_c * 100.0f / (float)eval_denom();
                    }
                    printf("\n══╡ TRY %d/%d ══ seed=%d (%.2f%%) bestN=%d ╞═══════════════════════════════\n",
                           ti + 1, _n_tries, _seed_idx, _seed_eval, g_beam_bestN);

                    /* g_best_used zurücksetzen — merge_and_beam füllt es neu für diesen Try */
                    free(g_best_used); g_best_used = NULL;
                    free(g_best_order); g_best_order = NULL; g_best_order_n = 0;
                    g_best_eval = -1.0f;
                    /* --tries-no-lock: the beam gets a FRESH empty pool so it can
                     * re-use members found by earlier tries (full-space search →
                     * different attractors). Without it, pool_exclude accumulates
                     * the found members and later tries only see the disjoint
                     * remainder (pool partitioning, plan-2026-08-08 Finding 1). */
                    uint8_t *_fresh = g_tries_no_lock ? (uint8_t *)calloc((size_t)n_blocks, 1) : NULL;
                    merge_and_beam(NULL, beam_width, _seed_idx,
                                   _fresh ? _fresh : pool_exclude);
                    free(_fresh);

                    /* Globalen Best über alle Trys aktualisieren */
                    if (g_best_eval > _global_best_eval && g_best_used) {
                        _global_best_eval = g_best_eval;
                        _global_best_n = g_best_n;
                        g_bestN_winner = g_beam_bestN;   /* reproduce the winning try */
                        free(_global_best_used);
                        _global_best_used = (uint8_t *)malloc((size_t)n_blocks);
                        memcpy(_global_best_used, g_best_used, (size_t)n_blocks);
                        /* INTENTIONAL (2026-08-14): Additionsreihenfolge des
                         * besten Trys mitkopieren, damit --debug-member nach
                         * dem Multi-Try-Merge weiter die kumulative acc[%] in
                         * der richtigen Reihenfolge zeigen kann. */
                        free(_global_best_order);
                        _global_best_order = NULL;
                        _global_best_order_n = 0;
                        if (g_best_order_n > 0) {
                            _global_best_order = (int *)malloc((size_t)g_best_order_n * sizeof(int));
                            if (_global_best_order) {
                                memcpy(_global_best_order, g_best_order,
                                       (size_t)g_best_order_n * sizeof(int));
                                _global_best_order_n = g_best_order_n;
                            }
                        }
                    }

                    /* Member dieses Trys für alle folgenden Trys sperren:
                     * - immer den Seed selbst (auch wenn er nicht im Beam-Ergebnis landete)
                     * - alle Mitglieder aus g_best_used
                     * --tries-no-lock: ONLY the seed is marked (for the next seed
                     * choice); g_best_used members stay free so later tries can
                     * combine them with other seeds (diversity, not partition). */
                    pool_exclude[_seed_idx] = 1;
                    if (!g_tries_no_lock && g_best_used) {
                        for (int i = 0; i < n_blocks; i++)
                            if (g_best_used[i]) pool_exclude[i] = 1;
                    }
                }
                free(pool_exclude);

                /* g_best_used = global best (für --member-out + write_member_file_from_best) */
                free(g_best_used); g_best_used = _global_best_used;
                free(g_best_order); g_best_order = _global_best_order;
                g_best_order_n = _global_best_order_n;
                g_best_eval = _global_best_eval;
                g_best_n    = _global_best_n;

                /* Output global best */
                strncpy(g_member_out, saved_member_out, sizeof(g_member_out));
                int _corr = (int)(g_best_eval * (float)eval_denom() / 100.0f + 0.5f);
                printf("\n══╡ GLOBAL BEST ╞════════════════════════════════════════\n");
                printf("  Best:  %d members  acc=%.2f%%  (%d/%d)  (best of %d tries)\n",
                       g_best_n, g_best_eval, _corr, eval_denom(), n_active_tries);
                printf("  Report: H=%d  EP=%d  VN=%d  HN=%d  TE=%d  eval=%.2f%%  err=%d  N=%d  best_en=%d  beam=%d  tries=%d\n",
                       g_hidden, g_epochs, g_split_vn, g_split_hn, g_target_err_x100,
                       g_best_eval, eval_denom() - _corr, eval_denom(), g_best_n, beam_width, n_active_tries);
                 /* --optimal: exchange AFTER the beam result, BEFORE the
                  * member-out export so the exported set is the fixpoint. */
                 if (g_optimal) merge_and_exchange(beam_width);
                 if (g_debug_member)
                     print_debug_members(g_best_used, g_best_order, g_best_order_n, g_best_n, n_blocks, g_best_eval);
                 if (g_debug_confusion)
                     print_ensemble_confusion(g_best_used, n_blocks);
                 if (g_member_out[0]) {
                     export_member_file(g_member_out, g_best_used, g_best_n, g_best_eval, beam_width);
                     backup_member_to_dir(dir, g_member_out);
                 }

                /* Machine-readable REPORT with the GLOBAL best over all tries.
                 * merge_and_beam suppresses its own REPORT in multi-try mode,
                 * so this is the ONLY REPORT line — run-grep parses the best
                 * result, not the last try (bug 2026-07-31).
                 * lr= carries the WINNING try's --beam-bestN (reproducibility:
                 * try k uses bestN=k, so lr=N tells which try produced the best). */
                struct timeval _mt1; gettimeofday(&_mt1, NULL);
                double _ms = (double)(_mt1.tv_sec - _mt0.tv_sec) * 1000.0
                           + (double)(_mt1.tv_usec - _mt0.tv_usec) / 1000.0;
                ki_report_show(0, 0, _corr, g_n_test, (int)_ms, omp_get_max_threads(),
                               g_n_test - _corr, (float)g_bestN_winner, g_best_n, n_blocks);
            }
            free(g_best_used); g_best_used = NULL;
            free(g_best_order); g_best_order = NULL; g_best_order_n = 0;
            g_best_eval = -1.0f;
        }
    } else if (greedy) {
        /* --member-seed for greedy: resolve the spec (same lookup as the
         * beam). NULL/absent → plain greedy (auto first member). */
        int _gseed = -1;
        if (g_member_seed_spec[0]) {
            _gseed = find_member_by_spec(g_member_seed_spec);
            if (_gseed < 0)
                fprintf(stderr, "  [WARN] --member-seed '%s' not found among %d blocks\n",
                        g_member_seed_spec, n_blocks);
        }
        merge_and_greedy(save_path, _gseed, beam_width);
        /* Greedy exports its member-out internally — back it up here. */
        if (g_member_out[0])
            backup_member_to_dir(dir, g_member_out);
    } else {
        merge_and_eval(save_path, 0);
        /* merge-eval exports ALL members as member-out — back it up too. */
        if (g_member_out[0])
            backup_member_to_dir(dir, g_member_out);
    }
    for (int i = 0; i < n_blocks; i++) free(blocks[i].scores);
    free(blocks); free(g_labels);
    return 0;
}
