/*
 * lib/ki-ens.h — .ens score archive format: constants, reader, writer, verify
 * ============================================================================
 *
 * The .ens format (written by the trainer's --export-merge-scores, read by
 * merge-ensemble) evolved over versions. EVERYTHING version-specific lives
 * here — a single mapping from version to {metadata layout, score width,
 * score type} — so the trainer writer and the merge readers can never drift
 * apart again (bug 2026-08-06: a version bound missed in one reader path
 * caused "0 blocks filled" + SEGV).
 *
 * Archive layout (header, then per-member metadata, then scores + labels):
 *   magic(4) ver(4) n_test(4) n_classes(4) n_members(4)
 *   hidden(4) epochs(4) split_vn(4) split_hn(4) target_err(4) seed(4)
 *   v>=3:  timestamp i64
 *   v>=4:  ensemble_eval f32
 *   v>=10: w0_marker u32 (W0[0])
 *   v>=11: maj_token char[8] + maj1_thresh i32
 *   v==5:  n_xforms u32 + xform ids u32[]
 *   per member (v>=7: 4 length-prefixed strings; v2-6: 4 bytes; v1: none)
 *   scores: n_members × (n_test × n_classes) elements
 *   labels: n_test bytes
 *
 * Score type by version:  v1-7 int64(8), v8 int32(4), v9-11 float(4),
 *                         v12 double(8), v13 int64(8).
 * The writer picks the version from the internal SCORE_TYPE (decision
 * 2026-08-06: the export follows the internal format — no precision loss).
 *
 * Header-only (static inline) like the other lib/ modules; the per-build
 * SCORE_TYPE is baked in at include time (after ki-common.h).
 */
#ifndef KI_ENS_H
#define KI_ENS_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ki-common.h"   /* SCORE_TYPE / COUNTER_TYPE (include-order safe) */

#define ENS_MAGIC 0x454E534D  /* 'ENSM' */
#define ENS_VER_MIN 1
#define ENS_VER_MAX 13

/* ═══════════════════════════════════════════════════════════════════════
 * PART A — version → format mapping (the single source of truth)
 * ═══════════════════════════════════════════════════════════════════════ */

static inline int ens_version_valid(int ver) {
    return ver >= ENS_VER_MIN && ver <= ENS_VER_MAX;
}

/* Element width of one stored score (v1-7 int64, v8 int32, v9-11 float,
 * v12 double, v13 int64). */
static inline int ens_score_bytes(int ver) {
    return (ver >= 8 && ver <= 11) ? 4 : 8;
}

/* Human label of the STORED score type (for headers/verify messages). */
static inline const char *ens_score_type_str(int ver) {
    if (ver >= 13) return "int64";
    if (ver == 12) return "double";
    if (ver >= 9)  return "float32";
    if (ver == 8)  return "int32";
    return "int64";   /* v1-7 */
}

/* Metadata layout by version */
static inline int ens_has_ts(int ver)            { return ver >= 3; }   /* header timestamp */
static inline int ens_has_eval(int ver)          { return ver >= 4; }   /* header ensemble_eval */
static inline int ens_has_xform_list(int ver)    { return ver == 5; }   /* v5 xform id list */
static inline int ens_has_member_strings(int ver){ return ver >= 7; }   /* 4 length-prefixed strings */
static inline int ens_has_w0(int ver)            { return ver >= 10; }  /* W0[0] marker */
static inline int ens_has_maj(int ver)           { return ver >= 11; }  /* maj_token + maj1_thresh */

/* Version for the internal SCORE_TYPE (writer) — the export follows the
 * internal format (decision 2026-08-06). */
static inline int ens_version_for_score_type(void) {
    return _Generic((SCORE_TYPE)0,
        float:   11,
        double:  12,
        int64_t: 13,
        default: 8);
}

/* ═══════════════════════════════════════════════════════════════════════
 * PART B — writer config + write + verify
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    uint32_t n_test, n_classes, n_members;
    uint32_t hidden, epochs, split_vn, split_hn, seed;
    float    target_err;      /* header target_err (0 normally) */
    float    member_eval;     /* v>=4: ensemble eval % */
    int64_t  file_time;       /* 0 → time(NULL) */
    uint32_t w0_marker;       /* v>=10: W0[0] */
    char     maj_token[8];    /* v>=11: majority token */
    int32_t  maj1_thresh;     /* v>=11: maj1 threshold */
    const char *member_fields[4]; /* v>=7: color, enc, enc-width, xform strings */
} EnsWriteCfg;

/* Write ONE .ens file. version from SCORE_TYPE; scores/labels stored
 * natively (width matches the version). Returns 0 / -1 (removes the file
 * on error). */
static inline int ens_write(const char *path, const EnsWriteCfg *c,
                            const uint8_t *y, int N,
                            const SCORE_TYPE *scores, size_t score_sz) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int ver = ens_version_for_score_type();
    int rc = -1;
    uint32_t magic = ENS_MAGIC, v = (uint32_t)ver;
    uint32_t n_test = c->n_test, n_classes = c->n_classes, n_mem = c->n_members;
    uint32_t hidden = c->hidden, epochs = c->epochs, svn = c->split_vn, shn = c->split_hn;
    float tgt_err = c->target_err;
    uint32_t seed = c->seed;
    int64_t stamp = c->file_time ? c->file_time : (int64_t)time(NULL);

    if (fwrite(&magic, 4, 1, f) != 1 || fwrite(&v, 4, 1, f) != 1 ||
        fwrite(&n_test, 4, 1, f) != 1 || fwrite(&n_classes, 4, 1, f) != 1 ||
        fwrite(&n_mem, 4, 1, f) != 1 ||
        fwrite(&hidden, 4, 1, f) != 1 || fwrite(&epochs, 4, 1, f) != 1 ||
        fwrite(&svn, 4, 1, f) != 1 || fwrite(&shn, 4, 1, f) != 1 ||
        fwrite(&tgt_err, 4, 1, f) != 1 || fwrite(&seed, 4, 1, f) != 1)
        goto done;
    if (ens_has_ts(ver) && fwrite(&stamp, 8, 1, f) != 1) goto done;
    if (ens_has_eval(ver) && fwrite(&c->member_eval, 4, 1, f) != 1) goto done;
    if (ens_has_w0(ver) && fwrite(&c->w0_marker, 4, 1, f) != 1) goto done;
    if (ens_has_maj(ver)) {
        if (fwrite(c->maj_token, 1, 8, f) != 8) goto done;
        if (fwrite(&c->maj1_thresh, 4, 1, f) != 1) goto done;
    }
    if (ens_has_member_strings(ver)) {
        for (int i = 0; i < 4; i++) {
            const char *s = c->member_fields[i] ? c->member_fields[i] : "";
            uint8_t l = (uint8_t)strlen(s);
            if (fwrite(&l, 1, 1, f) != 1 || fwrite(s, 1, l, f) != l) goto done;
        }
    }
    if (fwrite(scores, sizeof(SCORE_TYPE), score_sz, f) != score_sz) goto done;
    if (N > 0 && fwrite(y, 1, (size_t)N, f) != (size_t)N) goto done;
    rc = 0;
done:
    fclose(f);
    if (rc != 0) remove(path);
    return rc;
}

/* ═══════════════════════════════════════════════════════════════════════
 * PART C — reader (EnsReader): open → metadata → scores/skip → close
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    FILE       *f;
    uint32_t    version, n_test, n_classes, n_members;
    uint32_t    hidden, epochs, split_vn, split_hn, seed;
    float       target_err, ensemble_eval;
    uint32_t    w0_marker;
    char        maj_token[8];
    int32_t     maj1_thresh;
    int64_t     file_time;     /* 0 = absent (v1-2) */
    int         n_xforms;      /* v5: list count; else 1 */
    uint32_t    xform_ids[64]; /* v5 only */
    size_t      score_sz;      /* n_test × n_classes */
    int         elem;          /* ens_score_bytes(version) */
    int         fp_scale;      /* 1 = scale v9-11 float scores ×1048576 into
                                  int64 (int32-merge fixed-point, argmax-
                                  invariant; set by the int32 merge only) */
    off_t       hdr_bytes;     /* bytes consumed by the header (incl. v5 list) */
    off_t       meta_bytes;    /* per-member metadata bytes (accumulated) */
    off_t       scores_bytes;  /* n_members × score_sz × elem */
    off_t       labels_bytes;  /* n_test */
    unsigned char *tmp;        /* reusable raw-read buffer */
    size_t      tmp_cap;
} EnsReader;

/* Open + parse the full header. File positioned at the per-member
 * metadata (v7+ strings / v2-6 4 bytes) or the scores (v1). */
static inline int ens_reader_open(EnsReader *rd, const char *path) {
    memset(rd, 0, sizeof(*rd));
    rd->maj1_thresh = -999;
    rd->f = fopen(path, "rb");
    if (!rd->f) return -1;
    uint32_t hdr[11];
    if (fread(hdr, 4, 11, rd->f) != 11) { fclose(rd->f); rd->f = NULL; return -1; }
    if (hdr[0] != ENS_MAGIC || !ens_version_valid((int)hdr[1])) {
        fclose(rd->f); rd->f = NULL; return -1;
    }
    rd->version  = hdr[1];
    rd->n_test   = hdr[2]; rd->n_classes = hdr[3]; rd->n_members = hdr[4];
    rd->hidden   = hdr[5]; rd->epochs = hdr[6];
    rd->split_vn = hdr[7]; rd->split_hn = hdr[8];
    memcpy(&rd->target_err, &hdr[9], 4);
    rd->seed = hdr[10];
    rd->score_sz = (size_t)rd->n_test * (size_t)rd->n_classes;
    rd->elem = ens_score_bytes((int)rd->version);
    rd->hdr_bytes = 11 * 4;   /* base header (11×uint32) */
    if (ens_has_ts((int)rd->version)) {
        if (fread(&rd->file_time, 8, 1, rd->f) != 1) { fclose(rd->f); rd->f = NULL; return -1; }
        rd->hdr_bytes += 8;
    }
    if (ens_has_eval((int)rd->version)) {
        if (fread(&rd->ensemble_eval, 4, 1, rd->f) != 1) { fclose(rd->f); rd->f = NULL; return -1; }
        rd->hdr_bytes += 4;
    }
    if (ens_has_w0((int)rd->version)) {
        if (fread(&rd->w0_marker, 4, 1, rd->f) != 1) { fclose(rd->f); rd->f = NULL; return -1; }
        rd->hdr_bytes += 4;
    }
    if (ens_has_maj((int)rd->version)) {
        if (fread(rd->maj_token, 1, 8, rd->f) != 8) { fclose(rd->f); rd->f = NULL; return -1; }
        if (fread(&rd->maj1_thresh, 4, 1, rd->f) != 1) { fclose(rd->f); rd->f = NULL; return -1; }
        rd->hdr_bytes += 8 + 4;
    }
    rd->n_xforms = 1;
    if (ens_has_xform_list((int)rd->version)) {
        uint32_t nxf;
        if (fread(&nxf, 4, 1, rd->f) != 1) { fclose(rd->f); rd->f = NULL; return -1; }
        if (nxf > 64 || nxf == 0) nxf = 1;
        rd->n_xforms = (int)nxf;
        if (fread(rd->xform_ids, 4, nxf, rd->f) != nxf) { fclose(rd->f); rd->f = NULL; return -1; }
        rd->hdr_bytes += 4 + (off_t)nxf * 4;
    }
    rd->scores_bytes = (off_t)rd->n_members * (off_t)rd->score_sz * (off_t)rd->elem;
    rd->labels_bytes = (off_t)rd->n_test;
    return 0;
}

/* Read the NEXT member's metadata. v7+: 4 length-prefixed strings into
 * fields[4][128]; v2-6: 4 raw bytes into raw[4] (fields untouched);
 * v1: returns -1 (no metadata). Advances the file position. */
static inline int ens_reader_read_member_meta(EnsReader *rd,
                                              char fields[4][128],
                                              uint8_t raw[4]) {
    int ver = (int)rd->version;
    if (!ens_has_member_strings(ver)) {
        if (ver < 2) return -1;
        uint8_t b[4];
        if (fread(b, 1, 4, rd->f) != 4) return -1;
        if (raw) { raw[0] = b[0]; raw[1] = b[1]; raw[2] = b[2]; raw[3] = b[3]; }
        rd->meta_bytes += 4;
        return 0;
    }
    for (int i = 0; i < 4; i++) {
        uint8_t slen;
        if (fread(&slen, 1, 1, rd->f) != 1 || slen >= 128) return -1;
        if (fread(fields[i], 1, slen, rd->f) != slen) return -1;
        fields[i][slen] = '\0';
        rd->meta_bytes += 1 + (off_t)slen;
    }
    return 0;
}

/* Skip all per-member metadata of this archive (position → first scores). */
static inline int ens_reader_skip_all_meta(EnsReader *rd) {
    char fields[4][128]; uint8_t raw[4];
    for (uint32_t m = 0; m < rd->n_members; m++)
        if (ens_reader_read_member_meta(rd, fields, raw) != 0) return -1;
    return 0;
}

/* Raw read of one member's scores (score_sz elements × elem bytes) into
 * rd->tmp (grown as needed). Returns 0 / -1. */
static inline int ens_reader_read_raw(EnsReader *rd) {
    size_t need = rd->score_sz * (size_t)rd->elem;
    if (rd->tmp_cap < need) {
        unsigned char *nb = (unsigned char *)realloc(rd->tmp, need);
        if (!nb) return -1;
        rd->tmp = nb; rd->tmp_cap = need;
    }
    if (fread(rd->tmp, (size_t)rd->elem, rd->score_sz, rd->f) != rd->score_sz) return -1;
    return 0;
}

/* Read the NEXT member's scores into out[] (converted to SCORE_TYPE).
 * Returns 0 / -1. */
static inline int ens_reader_read_scores(EnsReader *rd, SCORE_TYPE *out) {
    if (ens_reader_read_raw(rd) != 0) return -1;
    size_t n = rd->score_sz;
    int ver = (int)rd->version;
    if (ver >= 13) { for (size_t i = 0; i < n; i++) out[i] = (SCORE_TYPE)((int64_t *)rd->tmp)[i]; }
    else if (ver == 12) { for (size_t i = 0; i < n; i++) out[i] = (SCORE_TYPE)((double *)rd->tmp)[i]; }
    else if (ver >= 9)  { if (rd->fp_scale) { for (size_t i = 0; i < n; i++) out[i] = (SCORE_TYPE)(((float *)rd->tmp)[i] * 1048576.0f); }
                          else { for (size_t i = 0; i < n; i++) out[i] = (SCORE_TYPE)((float *)rd->tmp)[i]; } }
    else if (ver >= 8)  { for (size_t i = 0; i < n; i++) out[i] = (SCORE_TYPE)((int32_t *)rd->tmp)[i]; }
    else                { for (size_t i = 0; i < n; i++) out[i] = (SCORE_TYPE)((int64_t *)rd->tmp)[i]; }
    return 0;
}

/* Read the NEXT member's scores into out[] as float (for eval computation,
 * independent of the merge SCORE_TYPE). Returns 0 / -1. */
static inline int ens_reader_read_scores_float(EnsReader *rd, float *out) {
    if (ens_reader_read_raw(rd) != 0) return -1;
    size_t n = rd->score_sz;
    int ver = (int)rd->version;
    if (ver >= 13) { for (size_t i = 0; i < n; i++) out[i] = (float)((int64_t *)rd->tmp)[i]; }
    else if (ver == 12) { for (size_t i = 0; i < n; i++) out[i] = (float)((double *)rd->tmp)[i]; }
    else if (ver >= 9)  { for (size_t i = 0; i < n; i++) out[i] = (float)((float *)rd->tmp)[i]; }
    else if (ver >= 8)  { for (size_t i = 0; i < n; i++) out[i] = (float)((int32_t *)rd->tmp)[i]; }
    else                { for (size_t i = 0; i < n; i++) out[i] = (float)((int64_t *)rd->tmp)[i]; }
    return 0;
}

/* Skip the NEXT member's scores (no read). Returns 0 / -1. */
static inline int ens_reader_skip_scores(EnsReader *rd) {
    return (fseek(rd->f, (long)(rd->score_sz * (size_t)rd->elem), SEEK_CUR) == 0) ? 0 : -1;
}

static inline void ens_reader_close(EnsReader *rd) {
    if (rd->f) { fclose(rd->f); rd->f = NULL; }
    free(rd->tmp); rd->tmp = NULL; rd->tmp_cap = 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * PART D — roundtrip verify (called right after ens_write, per member)
 * ═══════════════════════════════════════════════════════════════════════
 * Reads the file back and checks: version ↔ SCORE_TYPE, header fields,
 * member metadata, scores bit-exact, labels. Catches the write-format bug
 * class (version/width mismatch) at write time, not at merge time.
 * Returns 0 = ok, -1 = mismatch (message printed). */
static inline int ens_verify(const char *path, const EnsWriteCfg *c,
                             const uint8_t *y, int N,
                             const SCORE_TYPE *scores, size_t score_sz) {
    EnsReader rd;
    if (ens_reader_open(&rd, path) != 0) {
        fprintf(stderr, "[ens-verify] %s: cannot open/parse\n", path);
        return -1;
    }
    int rc = 0;
    int want = ens_version_for_score_type();
    if ((int)rd.version != want) {
        fprintf(stderr, "[ens-verify] %s: version %u != SCORE_TYPE version %d\n",
                path, rd.version, want);
        rc = -1;
    }
    if (rd.n_test != c->n_test || rd.n_classes != c->n_classes ||
        rd.n_members != c->n_members ||
        rd.hidden != c->hidden || rd.epochs != c->epochs ||
        rd.split_vn != c->split_vn || rd.split_hn != c->split_hn ||
        rd.seed != c->seed) {
        fprintf(stderr, "[ens-verify] %s: header config mismatch\n", path);
        rc = -1;
    }
    if (ens_has_w0((int)rd.version) && rd.w0_marker != c->w0_marker) {
        fprintf(stderr, "[ens-verify] %s: W0 marker mismatch (0x%08X/0x%08X)\n",
                path, rd.w0_marker, c->w0_marker);
        rc = -1;
    }
    if (ens_has_maj((int)rd.version) &&
        (strncmp(rd.maj_token, c->maj_token, 8) != 0 ||
         rd.maj1_thresh != c->maj1_thresh)) {
        fprintf(stderr, "[ens-verify] %s: maj token/threshold mismatch\n", path);
        rc = -1;
    }
    /* member metadata */
    if (ens_has_member_strings((int)rd.version)) {
        for (uint32_t m = 0; m < rd.n_members; m++) {
            char fields[4][128]; uint8_t raw[4];
            if (ens_reader_read_member_meta(&rd, fields, raw) != 0) {
                fprintf(stderr, "[ens-verify] %s: member %u metadata unreadable\n", path, m);
                rc = -1; break;
            }
            for (int j = 0; j < 4; j++) {
                const char *expect = c->member_fields[j] ? c->member_fields[j] : "";
                if (strcmp(fields[j], expect) != 0) {
                    fprintf(stderr, "[ens-verify] %s: member %u field %d mismatch (%s/%s)\n",
                            path, m, j, fields[j], expect);
                    rc = -1;
                }
            }
        }
    } else if (ens_reader_skip_all_meta(&rd) != 0) {
        fprintf(stderr, "[ens-verify] %s: metadata skip failed\n", path);
        rc = -1;
    }
    /* scores (bit-exact) */
    if (rd.score_sz != score_sz) {
        fprintf(stderr, "[ens-verify] %s: score size mismatch (%zu/%zu)\n",
                path, rd.score_sz, score_sz);
        rc = -1;
    } else {
        SCORE_TYPE *back = (SCORE_TYPE *)malloc(score_sz * sizeof(SCORE_TYPE));
        if (!back) { rc = -1; }
        else {
            if (ens_reader_read_scores(&rd, back) != 0) {
                fprintf(stderr, "[ens-verify] %s: short scores\n", path);
                rc = -1;
            } else if (memcmp(back, scores, score_sz * sizeof(SCORE_TYPE)) != 0) {
                fprintf(stderr, "[ens-verify] %s: score mismatch\n", path);
                rc = -1;
            }
            free(back);
        }
    }
    /* labels */
    if (N > 0) {
        uint8_t *lbl = (uint8_t *)malloc((size_t)N);
        if (!lbl) { rc = -1; }
        else {
            if (fread(lbl, 1, (size_t)N, rd.f) != (size_t)N) {
                fprintf(stderr, "[ens-verify] %s: short labels\n", path);
                rc = -1;
            } else if (memcmp(lbl, y, (size_t)N) != 0) {
                fprintf(stderr, "[ens-verify] %s: label mismatch\n", path);
                rc = -1;
            }
            free(lbl);
        }
    }
    ens_reader_close(&rd);
    return rc;
}

#endif /* KI_ENS_H */
