/*
 * mnist-1/ki-common.h — Shared infrastructure for Otto Score
 * ===========================================================
 *
 * Dataset-independent parts (CLI, batch correction, report, helpers).
 * Dataset-specific constants + loader come from ki-local.h.
 *
 * Symlinked from cifar-1/ki-common.h → ../mnist-1/ki-common.h
 */
#ifndef KI_COMMON_H
#define KI_COMMON_H

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <strings.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <zlib.h>
#include <omp.h>

/* ═══════════════════════════════════════════════════════════════════════
 * w0_random.h — splitmix64 PRNG
 * ═══════════════════════════════════════════════════════════════════════ */
#include "w0_random.h"

/* ═══════════════════════════════════════════════════════════════════════
 * STEP MODE — eindeutige Identifikation des Step-Algorithmus
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Jeder Member bekommt seinen eigenen Step pro Epoche.
 * Der Step-Mechanismus wird über --step-err / --step-const
 * gesetzt und im struct ki_Args.step_mode gespeichert.
 *
 *   STEP_POW:       step = step_init × (err/total)^step_power  (DEFAULT)
 *                   Fällt sanft mit Fehler → kein Overfitting.
 *   STEP_COS_TIME:  step = step_init × cosine(ep/epochs)
 *                   Zeitbasierter Cosine-Decay.
 *   STEP_COS_ERR:   step = step_init × cosine(1 - err/total)
 *                   Fehlerbasierter Cosine-Decay.
 *   STEP_CONST:     step = stepN (wenn stepN>0) sonst step_init
 *                   Konstanter Step.
 */
enum step_mode {
    STEP_POW        = 0,
    STEP_COS_TIME   = 1,
    STEP_COS_ERR    = 2,
    STEP_CONST      = 3,
};

/* ── Enum → lesbarer String ─────────────────────────────────── */
__attribute__((unused))
static const char *step_mode_name[] = {
    [STEP_POW]      = "pow",
    [STEP_COS_TIME] = "cos-time",
    [STEP_COS_ERR]  = "cos-err",
    [STEP_CONST]    = "const",
};

/* Forward declaration: mode_str(), color_str(), enc_str() sind weiter unten
 * definiert (nach OT_F), werden aber von ki_parse_args() im --help
 * bereits vor OT_F verwendet. */
__attribute__((unused))
static const char *mode_str(void);
__attribute__((unused))
static const char *color_str(void);
__attribute__((unused))
static const char *enc_str(void);

/* ═══════════════════════════════════════════════════════════════════════
 * ENCODING — Pixel-zu-Thermometer-Transformationen
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Jeder aktive Farb-Block kann eine eigene Encoding-Funktion haben.
 * Steuerung über --encoding r=lin8,g=up,b=down (per Block) oder
 * --encoding lin8 (Default für alle aktiven Blöcke).
 */

/* ═══════════════════════════════════════════════════════════════════════
 * ENCODING — Thermometer-Bitmasken für binäre Eingabe
 * ═══════════════════════════════════════════════════════════════════════
 * Encoding-Enum, Parser, Apply, LUT und Farbdefinitionen
 * sind in den gemeinsamen Header ausgelagert: */
#include "../lib/ki-encoding.h"

/* ── Parser: String → Bit-Position (für --channels) ────────────
 * Returns bit index or -1 if not a color name.
 * Handles: mnist,r,g,b,y,601,lum,l,rg,by,yl,709,auge,packed,full
 */
static inline int ki_color_parse(const char *tok) {
    /* Datengetrieben: {name, index} — Alias-Mehrfacheinträge erlaubt.
     * strcasecmp → case-insensitive. Sonderfälle: auge=-2, grey=-3,
     * rgb=-4, diff=-5. */
    static const struct { const char *name; int idx; } _lut[] = {
        {"mnist", COLOR_MNIST},
        {"r",     COLOR_R},
        {"g",     COLOR_G},
        {"b",     COLOR_B},

        {"y",     COLOR_Y},
        {"601",   COLOR_Y},
        {"yl",    COLOR_YL},
        {"709",   COLOR_YL},

        {"lum",   COLOR_AL},
        {"al",    COLOR_AL},
        {"am",    COLOR_AM},
        {"ap",    COLOR_AP},
        {"by",    COLOR_AP},

        {"bl",    COLOR_BL},
        {"bm",    COLOR_BM},
        {"bp",    COLOR_BP},

        {"rg",    COLOR_RG},
        {"rb",    COLOR_RB},
        {"gb",    COLOR_GB},

        {"h",     COLOR_H},
        {"s",     COLOR_S},
        {"c",     COLOR_C},

        {"cl",    COLOR_CL},
        {"cm",    COLOR_CM},
        {"cp",    COLOR_CP},

        /* Specials */
        {"auge",  -2},
        {"grey",  -3},
        {"rgb",   -4},
        {"diff",  -5},
    };
    for (size_t i = 0; i < sizeof(_lut) / sizeof(_lut[0]); i++) {
        if (strcasecmp(tok, _lut[i].name) == 0)
            return _lut[i].idx;
    }
    return -1;  /* kein Farbname */
}

/* ═══════════════════════════════════════════════════════════════════════
 * ki-local.h — dataset-specific constants + data loader
 * (MNIST in mnist-1/, CIFAR-10 in cifar-1/)
 * ═══════════════════════════════════════════════════════════════════════ */
#include "ki-local.h"

#ifndef MAX
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * VIRTUAL NEURONS (VN) — Bit-Grouping per Container
 * ═══════════════════════════════════════════════════════════════════════
 *
 * --splitVN (1,2,4,8,16,32) gruppiert Bits eines Containers zu
 * virtuellen Neuronen.  Jedes virtuelle Neuron feuert wenn die
 * Popcount-Mehrheit seiner Bits gesetzt ist.
 *
 *   VN_BITS(G)  = G       Bits pro virtuellem Neuron (= splitVN)
 *   VN_GROUPS(G) = 32 / G Virtuelle Neuronen pro Container
 *   VN_THRESH(G) = G / 2  Popcount-Schwelle (>=majority)
 */
#define VN_BITS(G)   (G)
#define VN_GROUPS(G) (32 / (G))
#define VN_THRESH(G) ((G) / 2)

#define VN_BITS_  VN_BITS(aa.splitVN)
#define VN_GROUPS_ VN_GROUPS(aa.splitVN)
#define VN_THRESH_ VN_THRESH(aa.splitVN)

/* Target-Index: [H][V][KI_NCLASSES] — neuron × virtual-neuron × class
 * V = VN_GROUPS = 32 / splitVN = compile-time constant per variant.
 * k-last layout: scores[k] += target[...k] iterates k=0..9 →
 * KI_NCLASSES contiguous int32_t = 40 bytes = 1 cache line.
 * Was [KI_NCLASSES][H][32] — caused 55% D1 cache misses. */
#ifndef TGT_IDX
#define TGT_IDX(k, h, v, H, V) \
    ((size_t)(h) * (size_t)(V) * KI_NCLASSES + (size_t)(v) * KI_NCLASSES + (size_t)(k))
#endif

/* ── VN-Dispatch-Makro: erzeugt switch über alle 6 splitVN-Werte ──
 * Jeder Zweig ruft func_vn<N> mit den restlichen args auf.
 * func muss als _vn1, _vn2, _vn4, _vn8, _vn16, _vn32 existieren. */
#define VN_DISPATCH(func, G, args...) do {                          \
    switch (G) {                                                    \
        case 1:  func ## _vn1(args); break;                         \
        case 2:  func ## _vn2(args); break;                         \
        case 4:  func ## _vn4(args); break;                         \
        case 8:  func ## _vn8(args); break;                         \
        case 16: func ## _vn16(args); break;                        \
        case 32: func ## _vn32(args); break;                        \
        default: fprintf(stderr, "[FATAL] invalid --splitVN %d\n", G); exit(1); \
    }                                                               \
} while (0)
#define VN_DISPATCH_R(func, G, args...) ({                          \
    __typeof__(func ## _vn1 args) _r;                               \
    switch (G) {                                                    \
        case 1:  _r = func ## _vn1(args); break;                    \
        case 2:  _r = func ## _vn2(args); break;                    \
        case 4:  _r = func ## _vn4(args); break;                    \
        case 8:  _r = func ## _vn8(args); break;                    \
        case 16: _r = func ## _vn16(args); break;                   \
        case 32: _r = func ## _vn32(args); break;                   \
        default: fprintf(stderr, "[FATAL] invalid --splitVN %d\n", G); exit(1); \
    }                                                               \
    _r;                                                             \
})

/* ═══════════════════════════════════════════════════════════════════════
 * ARGS — CLI Parameters (Otto Score only)
 * ═══════════════════════════════════════════════════════════════════════ */

 #define KI_ENC_MAX 16
 #define KI_DEFAULT_TARGET_NORM 0  /* --optional target-norm: default ON */

typedef struct {
    int8_t type;   /* KI_ENC_RAW..KI_ENC_SIG */
    int8_t width;  /* 8, 16, 32 */
    int8_t color;  /* COLOR_BIT or -1 (default/all) */
} ki_EncSlot;

typedef struct {
    int    hidden;          /* Hidden neurons (--hiddenN, default: 64) */
    int    epochs;          /* Iterations (--epochsN, default: 1) */
    int    batchN;          /* Mini-batch size (--batchN, default: 64) */
    int    trainN;          /* Training samples (--trainN, default: 50000) */
    int    evalN;           /* Eval samples (--evalN, default: 10000) */
    int    dry_run;         /* --dry-run: print arch and exit */
    int    debug;           /* --debug: verbose output */
    unsigned int seed;      /* Random seed (--seed, default: 42) */
    char   out[256];        /* --export DIR: export directory */
    char   predictions[256]; /* --predictions FILE: export per-sample predictions (for vis-errors) */
    float  lr;              /* Step size (--lr, default: 0.05) */
    float  lr_min;          /* Min LR fraction (--lr-min, default: 0.1) */
    int    lr_step;         /* round(aa.lr * (1<<OT_PRECISION)) */
    int    threadN;         /* OpenMP threads (--threadN, default: 8) */
    int    debug_h0;        /* --debug-h0: per-neuron debug */
    int    shuffle;         /* --shuffle: randomize train/eval split */
    int    warmup_epochs;   /* --warmup N: linear warmup epochs (default: 2) */
    int    step_mode;       /* enum step_mode: Algorithmus (siehe oben) */
    int    stepN;           /* --step-const N: const step value (0=use lr, default: 0) */
    float  step_power;      /* --step-power F: exponent für pow/cos (default: 0.7) */
    float  target_err;      /* --target-err F: training error target (0.0=off). Step→0 when err≤target */
    int    err_rollback;    /* --err-rollback: rollback targets when err increases (default: 0) */
    int    ensembleN;       /* --ensembleN N: independent W0 copies (default: 1) */
    int    splitVN;         /* --splitVN N: vertical H split (default: 1) */
    int    splitHN;         /* --splitHN N: horizontal NC split (default: 1) */
    int    channel;        /* --channels bitmask of selected blocks */
    int    channel_explicit; /* 1=--channels was set explicitly */
    int    packedB;    /* 1=4px/cont (256/blk), 0=1px/cont (1024/blk) */
    int    debug_flat;      /* 1=all selected blocks in one flat array, 0=separate members */
    int    debug_binarize;  /* 1=threshold block values at 128 → 0x00/0xFF per pixel */
    int    hebbian_pct;     /* --hebbian-pct: flip threshold % (reference, default 50) */
    int8_t enc[COLOR_NB];         /* --encoding: per-block encoding type (-1=not set, derived from enc_array) */
    int8_t enc_width[COLOR_NB];  /* --encoding: per-block width (derived from enc_array) */
    int8_t enc_default_type;      /* --encoding: fallback type for blocks without specific setting */
    int8_t enc_default_width;     /* --encoding: fallback width (8, 16, 32) */
    ki_EncSlot enc_array[KI_ENC_MAX];  /* Alle aktiven Encodings als (type,width)-Paare */
    int         enc_count;             /* Anzahl Einträge in enc_array */
    int    opt_target_norm;    /* --optional target-norm: vote normalisierung aktivieren */
    char   seed_file[256]; /* --seed-file PATH: true random source */
    char   model[512];    /* --import DIR: load model for inference */
    int    seed_splitmix;  /* --seed-splitmix: ignore seed_file, use splitmix64 PRNG */
    int    multi_correct;  /* --multi-correct: alle über true_k bestrafen (default: 1) */
    int    ensemble_seed;    /* ENS_SEED_ONCE|CONST|INCR (default: ONCE) */
    int    debug_class_voting; /* --debug-class-voting: per-member per-class accuracy table */
} ki_Args;

/* ── Global args (defined in each main .c file) ────────────── */
extern ki_Args aa;

/* ── Ensemble seeding strategy ──────────────────────────────── */
typedef enum {
    ENS_SEED_ONCE  = 0,  /* default: one seed, sequential fill */
    ENS_SEED_CONST,      /* const: share W0 within ensemble */
    ENS_SEED_INCR,       /* incr: each member gets seed + m */
} ki_EnsembleSeed;

static inline const char *ensemble_seed_str() {
    switch (aa.ensemble_seed) {
        case ENS_SEED_CONST: return "const";
        case ENS_SEED_INCR:  return "incr";
        default:             return "once";
    }
}

/* ── Parse CLI ─────────────────────────────────────────────────── */
static inline void ki_parse_args(int argc, char *argv[]) {
    /* enc[] initialisieren: -1 = "nicht gesetzt" (Default wird später aufgelöst) */
    for (int i = 0; i < COLOR_NB; i++) aa.enc[i] = -1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --hiddenN N       Hidden neurons                                                (default: %d)\n", aa.hidden);
            printf("  --epochsN N       Iterations                                                    (default: %d)\n", aa.epochs);
            printf("  --ensembleN N     Independent W0 copies                                         (default: %d, total bitmass N×(H×NC))\n", aa.ensembleN);
            printf("  --splitVN 1|2|4|8|16|32  Bit-Grouping pro Neuron                                (default: %d, 1=no grouping)\n", aa.splitVN);
            printf("  --splitHN N       Horizontal NC-split                                           (default: %d, was old --sliceN)\n", aa.splitHN);
            printf("  --batchN N        Mini-batch size                                               (default: %d)\n", aa.batchN);
            printf("  --trainN N        Training samples                                              (default: %d)\n", aa.trainN);
            printf("  --evalN N         Eval samples                                                  (default: %d)\n", aa.evalN);
            printf("  --threadN N       OpenMP threads                                                (default: %d)\n", aa.threadN);
            printf("  ---------------------------------------------------------------------------------------------\n");
            printf("  --lr FLOAT        Step size                                                     (default: %.4f)\n", (double)aa.lr);
            printf("  --lr-min FLOAT    Min lr fraction for cosine decay                              (default: %.1f, stop at step=1)\n", (double)aa.lr_min);
            printf("  --step-err cos-time|cos-err|pow|pow=NUM|const|const=NUM                         (default: %s)\n", mode_str());
            printf("                    Step mode: N=err-proportional, auto=compute,\n");
            printf("                    cos-time  : time-based cosine\n");
            printf("                    cos-err   : error cosine\n");
            printf("                    pow       : step_init×(err/total)^power\n");
            printf("                    const     : step_init (const=NUM: fixed step NUM)\n");
            printf("  --step-const N    alias for --step-err const=####                               (default: %d)\n", aa.stepN);
            printf("  --step-power F    alias for --step-err pow=####                                 (default: %.1f, 1.0=linear)\n", (double)aa.step_power);
            printf("  --target-err F    Training error target (0.0=off). Step→0 when err≤target       (default: %.2f)\n", (double)aa.target_err);
            printf("                    Soft scaling: step *= (err - target) / target\n");
            printf("  --err-rollback    Rollback targets when training err increases                  (default: off)\n");
            printf("  --warmup N        Linear warmup epochs                                          (default: %d, 0=off)\n", aa.warmup_epochs);
            printf("  ---------------------------------------------------------------------------------------------\n");
            printf("  --seed N          Random seed                                                   (default: %d)\n", aa.seed);
            printf("  --seed-file PATH-TO-RANDOM-FILE                                                 (default: none)\n");
            printf("                    Use true random data from file instead of PRNG\n");
            printf("  --seed-splitmix   Ignore seed file, use splitmix64 PRNG explicitly              (default: off)\n");
            printf("  --seed-member const|incr|once  W0 seeding mode.                                 (default: %s)\n", ensemble_seed_str());
            printf("                    once    : one seed, all sequential.\n");
            printf("                    const   : same W0 per ensemble (tests color diversity).\n");
            printf("                    incr    : unique seed per member.\n");
            printf("  ---------------------------------------------------------------------------------------------\n");
            printf("  --channels [packed|full,][flat,]...                                             (default: %s)\n", color_str());
            printf("                    Channel selection (comma-sep).  Encoding via --encoding.\n");
            printf("                    packed/full   : 4px/cont or 1px/cont,\n");
            printf("                    flat          : all selected blocks in one wide W0,\n");
            printf("                    auge          : lum|al=(R+G)/2, rg|am=R-G, by|ap=B-(R+G)/2,\n");
            printf("                                  :     bl=(R+B)/2,    bm=R-B,    bp=G-(R+B)/2,\n");
            printf("                    diff          : rg=R-G, rb=R-B, gb=G-B (color opponent),\n");
            printf("                    rgb           : r=R, g=G, b=B,\n");
            printf("                    grey          : y|601=ITU-601, yl|709=ITU-709\n");
            printf("                    h             : hue (Farbwinkel, atan2-basiert)\n");
            printf("                    s             : saturation (Farbsättigung, max-min)\n");
            printf("                    c             : contrast (Sobel-Kanten auf LUM)\n");
            printf("                    mnist         : single grayscale.\n");
            printf("  --encoding [raw|lin7|lin8|down|up|mid|log|exp|sig]                              (default: %s)\n", enc_str());
            printf("                    OR <color>=<enc>[width] per-block: r=exp16,g=lin8   \n");
            printf("                    Pixel-Encoding pro Farb-Block.\n");
            printf("                    Optionaler Width-Suffix: exp16=16-bit, lin32=32-bit\n");
            printf("                    8-bit  : 4 px/cont, 8 Stufen (default, exp=0.3)\n");
            printf("                    16-bit : 2 px/cont, 16 Stufen (exp=0.5, 2× Breite)\n");
            printf("                    32-bit : 1 px/cont, 32 Stufen (exp=0.7, 4× Breite)\n");
            printf("                    lin7   7-level thermometer (old bin),\n");
            printf("                    lin8   linear (pv*width/256),\n");
            printf("                    down   shadow emphasis,\n");
            printf("                    up     highlight emphasis,\n");
            printf("                    mid    midtone emphasis,\n");
            printf("                    log    logarithmic,\n");
            printf("                    exp    exponential,\n");
            printf("                    sig    sigmoid.\n");
            printf("                    raw    no encoding (raw 8-bit bytes).\n");
            printf("  --export DIR      Export directory                                              (default: none)\n");
            printf("  --import DIR      Load model for inference                                      (default: none)\n");
            printf("  --optional target-norm  Vote normalisierung (equal voting power)                (default: off)\n");
            printf("  --?no-?multi-correct  Only punish argmax, not all over true_k                   (default: multi-correct)\n");
            printf("  ---------------------------------------------------------------------------------------------\n");
            printf("  --dry-run         Print architecture and exit                                   (default: off)\n");
            printf("  --quick           5000 train / 2000 eval\n");
            printf("  --qq              5000 train / 2000 eval / 3 epochs\n");
            printf("  --debug           Verbose output                                                (default: off)\n");
            printf("  --debug-h0        Per-neuron debug                                              (default: off)\n");
            printf("  --debug-class-voting  Member × Class accuracy table (trainN only)               (default: off)\n");
            printf("  --shuffle         Shuffle data before train/eval split                          (default: off)\n");
            exit(0);
        } else if (strcmp(argv[i], "--dry-run") == 0) {
            aa.dry_run = 1;
            aa.epochs  = 0;
        } else if (strcmp(argv[i], "--debug") == 0) {
            aa.debug = 1;
        } else if (strcmp(argv[i], "--quick") == 0) {
            aa.trainN = 5000; aa.evalN = 2000;
        } else if (strcmp(argv[i], "--qq") == 0) {
            aa.trainN = 5000; aa.evalN = 2000; aa.epochs = 3;
        } else if (strcmp(argv[i], "--hiddenN") == 0 && i + 1 < argc) {
            aa.hidden = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--epochsN") == 0 && i + 1 < argc) {
            aa.epochs = atoi(argv[++i]);
            if (aa.epochs == 0) aa.dry_run = 1;
        } else if (strcmp(argv[i], "--batchN") == 0 && i + 1 < argc) {
            aa.batchN = atoi(argv[++i]);
            if (aa.batchN < 1) aa.batchN = 1;
        } else if (strcmp(argv[i], "--trainN") == 0 && i + 1 < argc) {
            aa.trainN = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--evalN") == 0 && i + 1 < argc) {
            aa.evalN = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            aa.lr = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--threadN") == 0 && i + 1 < argc) {
            aa.threadN = atoi(argv[++i]);
            if (aa.threadN < 1) aa.threadN = 1;
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            aa.warmup_epochs = atoi(argv[++i]);
            if (aa.warmup_epochs < 0) aa.warmup_epochs = 0;
        } else if (strcmp(argv[i], "--step-err") == 0 && i + 1 < argc) {
            const char *val = argv[++i];
            if (strcmp(val, "cos-err") == 0) {
                aa.step_mode  = STEP_COS_ERR;
            } else if (strcmp(val, "cos-time") == 0) {
                aa.step_mode  = STEP_COS_TIME;
            } else if (strncmp(val, "const", 5) == 0) {
                aa.step_mode  = STEP_CONST;
                if (val[5] == '=')               /* --step-err const=1000 → fester Step 1000 */
                    aa.stepN = atoi(val + 6);
                if (aa.stepN < 0) aa.stepN = 0;
            } else if (strncmp(val, "pow", 3) == 0) {
                aa.step_mode  = STEP_POW;
                if (val[3] == '=')               /* --step-err pow=0.5 → setzt step_power */
                    aa.step_power = (float)atof(val + 4);
            } else {
                fprintf(stderr, "[ERROR] --step-err: unknown mode '%s'. "
                        "Valid: cos-time, cos-err, pow[=NUM], const[=NUM]\n", val);
                exit(1);
            }
        } else if (strcmp(argv[i], "--step-const") == 0 && i + 1 < argc) {
            aa.stepN = atoi(argv[++i]);   /* legacy alias for --step-err const=N */
            if (aa.stepN < 0) aa.stepN = 0;
            aa.step_mode = STEP_CONST;
        } else if (strcmp(argv[i], "--step-power") == 0 && i + 1 < argc) {
            aa.step_power = (float)atof(argv[++i]);
            if (aa.step_power < 0.0f) aa.step_power = 0.0f;
            aa.step_mode = STEP_POW;
        } else if (strcmp(argv[i], "--target-err") == 0 && i + 1 < argc) {
            aa.target_err = (float)atof(argv[++i]);
            if (aa.target_err < 0.0f) aa.target_err = 0.0f;
            if (aa.target_err > 1.0f) aa.target_err = 1.0f;
        } else if (strcmp(argv[i], "--err-rollback") == 0) {
            aa.err_rollback = 1;
        } else if (strcmp(argv[i], "--iter") == 0 && i + 1 < argc) {
            i++;  /* ignored (BV32 compatibility) */
        } else if (strcmp(argv[i], "--lr-min") == 0 && i + 1 < argc) {
            aa.lr_min = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            aa.seed = (unsigned int)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--export") == 0 || strcmp(argv[i], "--out") == 0) {
            if (i + 1 >= argc) { fprintf(stderr, "[ERROR] --export DIR\n"); exit(1); }
            strncpy(aa.out, argv[++i], sizeof(aa.out) - 1);
            aa.out[sizeof(aa.out) - 1] = '\0';
        } else if (strcmp(argv[i], "--predictions") == 0 && i + 1 < argc) {
            strncpy(aa.predictions, argv[++i], sizeof(aa.predictions) - 1);
            aa.predictions[sizeof(aa.predictions) - 1] = '\0';
        } else if (strcmp(argv[i], "--import") == 0 || strcmp(argv[i], "--model") == 0) {
            if (i + 1 >= argc) { fprintf(stderr, "[ERROR] --import DIR\n"); exit(1); }
            strncpy(aa.model, argv[++i], sizeof(aa.model) - 1);
            aa.model[sizeof(aa.model) - 1] = '\0';
        } else if (strcmp(argv[i], "--debug-h0") == 0) {
            aa.debug_h0 = 1;
        } else if (strcmp(argv[i], "--debug-class-voting") == 0) {
            aa.debug_class_voting = 1;
        } else if (strcmp(argv[i], "--shuffle") == 0) {
            aa.shuffle = 1;
        } else if (strcmp(argv[i], "--ensembleN") == 0 && i + 1 < argc) {
            aa.ensembleN = atoi(argv[++i]);
            if (aa.ensembleN < 1) aa.ensembleN = 1;
        } else if (strcmp(argv[i], "--splitVN") == 0 && i + 1 < argc) {
            aa.splitVN = atoi(argv[++i]);
            if (aa.splitVN != 1 && aa.splitVN != 2 && aa.splitVN != 4
                && aa.splitVN != 8 && aa.splitVN != 16 && aa.splitVN != 32) {
                fprintf(stderr, "[ERROR] --splitVN: expected 1,2,4,8,16,32, got %d\n", aa.splitVN);
                exit(1);
            }
        } else if (strcmp(argv[i], "--splitHN") == 0 && i + 1 < argc) {
            aa.splitHN = atoi(argv[++i]);
            if (aa.splitHN < 1) aa.splitHN = 1;
        } else if (strcmp(argv[i], "--sliceN") == 0 && i + 1 < argc) {
            aa.splitHN = atoi(argv[++i]);  /* alias for --splitHN */
            if (aa.splitHN < 1) aa.splitHN = 1;
        } else if (strcmp(argv[i], "--seed-file") == 0 && i + 1 < argc) {
            strncpy(aa.seed_file, argv[++i], sizeof(aa.seed_file) - 1);
            aa.seed_file[sizeof(aa.seed_file) - 1] = '\0';
        } else if (strcmp(argv[i], "--random-file") == 0 && i + 1 < argc) {
            /* Legacy alias for --seed-file */
            strncpy(aa.seed_file, argv[++i], sizeof(aa.seed_file) - 1);
            aa.seed_file[sizeof(aa.seed_file) - 1] = '\0';
        } else if (strcmp(argv[i], "--seed-splitmix") == 0) {
            aa.seed_splitmix = 1;
            aa.seed_file[0] = '\0';  /* deaktiviert seed_file */
        } else if (strcmp(argv[i], "--multi-correct") == 0) {
            aa.multi_correct = 1;
        } else if (strcmp(argv[i], "--no-multi-correct") == 0) {
            aa.multi_correct = 0;
        } else if (strcmp(argv[i], "--channels") == 0 && i + 1 < argc) {
            const char *val = argv[++i];
            int mask = 0;
            char buf[128];
            strncpy(buf, val, sizeof(buf) - 1);
            buf[sizeof(buf) - 1] = '\0';
            const char *delim = ",";
            for (char *tok = strtok(buf, delim); tok; tok = strtok(NULL, delim)) {
                while (*tok == ' ' || *tok == '\t') tok++;
                for (char *p = tok; *p; p++) *p = (char)tolower((unsigned char)*p);
                if (strcmp(tok, "packed") == 0) { aa.packedB = 1; continue; }
                else if (strcmp(tok, "full") == 0) { aa.packedB = 0; continue; }
                else if (strcmp(tok, "flat") == 0) { aa.debug_flat = 1; continue; }
                /* NOTE: 'bin' removed. Encoding via --encoding. */
                int bit = ki_color_parse(tok);
                if (bit == -2) { mask |= (1<<COLOR_AL)|(1<<COLOR_AM)|(1<<COLOR_AP); continue; }
                if (bit == -3) { mask |= (1<<COLOR_Y); continue; }
                if (bit == -4) { mask |= (1<<COLOR_R)|(1<<COLOR_G)|(1<<COLOR_B); continue; }
                if (bit == -5) { mask |= (1<<COLOR_RG)|(1<<COLOR_RB)|(1<<COLOR_GB); continue; }
                if (bit >= 0) { mask |= (1 << bit); continue; }
                fprintf(stderr, "[ERROR] --channels: unknown '%s'. "
                        "Valid: r,g,b,y,lum,l,rg,by,yl,h,s,c,mnist,601,709,packed,full,flat\n", tok);
                exit(1);
            }
            if (mask == 0) {
                fprintf(stderr, "[ERROR] --channels: at least one channel required\n");
                exit(1);
            }
            aa.channel = mask;
            aa.channel_explicit = 1;
        } else if (strcmp(argv[i], "--encoding") == 0 && i + 1 < argc) {
            const char *val = argv[++i];
            /* --encoding impliziert debug_binarize (thermometer mode).
             * Explizites --encoding raw deaktiviert es. */
            int has_raw = 0;
            char buf[256];
            strncpy(buf, val, sizeof(buf) - 1);
            buf[sizeof(buf) - 1] = '\0';

            /* ── Encoding-Alias-Expansion ─────────────────────────────
             * Erlaubt --encoding latest statt langer Komma-Listen.
             * Neue Aliases hier eintragen (beide Arrays同步 halten).
             * Expandiert rekursiv bis 5 Tiefe: zunaechst voller String,
             * dann pro Komma-Token. */
            #define KI_ENC_ALIAS_N 8
            static const char * const _alias_name[KI_ENC_ALIAS_N] = {
                "ey-a",
                "ey-b",
                "ey-c",
                "ey-h",
                "best-mnist",
                "top-rgb",
                "latest",
                "latest-2"
            };
            static const char * const _alias_val[KI_ENC_ALIAS_N] = {
                "b=up,al=down,am=sig,ap=sig",
                "g=up,bl=down,bm=sig,bp=sig",
                "r=up,cl=down,cm=sig,cp=sig",
                "h=down,c=exp,gb=sig",
                "exp,log,log",
                "r=down,g=down,b=down",
                "ey-b,ey-a,ey-h",
                "g=lin8,bl=lin8,bm=sig,bp=sig,ey-a,ey-h",
            };
            for (int _iter = 0; _iter < 5; _iter++) {
                /* Phase 1: voller String match */
                int _any = 0;
                for (int _a = 0; _a < KI_ENC_ALIAS_N; _a++) {
                    if (strcasecmp(buf, _alias_name[_a]) == 0) {
                        strncpy(buf, _alias_val[_a], sizeof(buf) - 1);
                        buf[sizeof(buf) - 1] = '\0';
                        _any = 1; break;
                    }
                }
                if (_any) continue;
                /* Phase 2: pro Token expandieren */
                char _tmp[256], _new[256] = "";
                strncpy(_tmp, buf, sizeof(_tmp) - 1);
                _tmp[sizeof(_tmp) - 1] = '\0';
                char *_t = strtok(_tmp, ",");
                while (_t) {
                    int _hit = 0;
                    for (int _a = 0; _a < KI_ENC_ALIAS_N; _a++) {
                        if (strcasecmp(_t, _alias_name[_a]) == 0) {
                            if (_new[0]) strncat(_new, ",", sizeof(_new) - 1);
                            strncat(_new, _alias_val[_a], sizeof(_new) - strlen(_new) - 1);
                            _hit = 1; _any = 1; break;
                        }
                    }
                    if (!_hit) {
                        if (_new[0]) strncat(_new, ",", sizeof(_new) - 1);
                        strncat(_new, _t, sizeof(_new) - strlen(_new) - 1);
                    }
                    _t = strtok(NULL, ",");
                }
                if (!_any) break; /* keine weiteren Expansionen */
                strncpy(buf, _new, sizeof(buf) - 1);
                buf[sizeof(buf) - 1] = '\0';
            }
            #undef KI_ENC_ALIAS_N

            for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
                while (*tok == ' ' || *tok == '\t') tok++;
                const char *eq = strchr(tok, '=');
                int enc, w = KI_ENC_WIDTH_DEFAULT;
                if (eq) {
                    /* Per-block encoding: r=exp16 */
                    char col_buf[32];
                    size_t col_len = (size_t)(eq - tok);
                    if (col_len >= sizeof(col_buf)) col_len = sizeof(col_buf) - 1;
                    memcpy(col_buf, tok, col_len);
                    col_buf[col_len] = '\0';
                    int bit = ki_color_parse(col_buf);
                    if (bit < 0) {
                        fprintf(stderr, "[ERROR] --encoding: unknown color '%s' in '%s'. "
                                "Valid: r,g,b,rg,rb,gb,lum,by,y,yl\n", col_buf, tok);
                        exit(1);
                    }
                    enc = ki_enc_parse(eq + 1, &w);
                    if (enc < 0) {
                        fprintf(stderr, "[ERROR] --encoding: unknown encoding '%s' in '%s'. "
                                "Valid: raw,lin7,lin8,down,up,mid,log,exp,sig\n", eq + 1, tok);
                        exit(1);
                    }
                    /* In enc_array eintragen — der einzige Encoding-Pfad */
                    #define ADD_ENC(COL, E, W) do { \
                        if (aa.enc_count < KI_ENC_MAX) { \
                            aa.enc_array[aa.enc_count].type  = (int8_t)(E); \
                            aa.enc_array[aa.enc_count].width = (int8_t)(W); \
                            aa.enc_array[aa.enc_count].color = (int8_t)(COL); \
                            aa.enc_count++; \
                        } } while(0)
                    if (bit == -2) { /* auge = lum+rg+by */
                        ADD_ENC(COLOR_AL, enc, w); ADD_ENC(COLOR_AM, enc, w); ADD_ENC(COLOR_AP, enc, w);
                    } else if (bit == -3) { /* grey = y */
                        ADD_ENC(COLOR_Y, enc, w);
                    } else if (bit == -4) { /* color = r+g+b */
                        ADD_ENC(COLOR_R, enc, w); ADD_ENC(COLOR_G, enc, w); ADD_ENC(COLOR_B, enc, w);
                    } else if (bit == -5) { /* diff = rg+rb+gb */
                        ADD_ENC(COLOR_RG, enc, w); ADD_ENC(COLOR_RB, enc, w); ADD_ENC(COLOR_GB, enc, w);
                    } else {
                        ADD_ENC((int)bit, enc, w);
                    }
                    #undef ADD_ENC
                    if (enc == KI_ENC_RAW) has_raw = 1;
                } else {
                    /* Single token: encoding name ± width suffix.
                     * Jeder Token → EIN Eintrag in enc_array[]. */
                    enc = ki_enc_parse(tok, &w);
                    if (enc < 0) {
                        fprintf(stderr, "[ERROR] --encoding: unknown '%s'. "
                                "Valid: raw,lin7,lin8,down,up,mid,log,exp,sig\n", tok);
                        exit(1);
                    }
                    if (aa.enc_count < KI_ENC_MAX) {
                        aa.enc_array[aa.enc_count].type  = (int8_t)enc;
                        aa.enc_array[aa.enc_count].width = (int8_t)w;
                        aa.enc_array[aa.enc_count].color = -1;  /* default/all */
                        aa.enc_count++;
                    }
                    /* Erster Token setzt auch enc_default (Backward compat) */
                    if (aa.enc_count == 1) {
                        aa.enc_default_type  = (int8_t)enc;
                        aa.enc_default_width = (int8_t)w;
                    }
                    if (enc == KI_ENC_RAW) has_raw = 1;
                }
            }
            /* --encoding raw deaktiviert binarize */
            if (has_raw) aa.debug_binarize = 0;
            else         aa.debug_binarize = 1;
        } else if (strcmp(argv[i], "--seed-member") == 0 && i + 1 < argc) {
            const char *val = argv[++i];
            if (strcmp(val, "const") == 0)
                aa.ensemble_seed = ENS_SEED_CONST;
            else if (strcmp(val, "incr") == 0)
                aa.ensemble_seed = ENS_SEED_INCR;
            else if (strcmp(val, "once") == 0)
                aa.ensemble_seed = ENS_SEED_ONCE;
            else {
                fprintf(stderr, "[ERROR] --seed-member: expected 'const', 'incr', or 'once', got '%s'\n", val);
                exit(1);
            }
        } else if (strcmp(argv[i], "--ensemble-seed") == 0 && i + 1 < argc) {
            /* Legacy: accept const/incr + existing file (random file path) */
            const char *val = argv[++i];
            if (strcmp(val, "const") == 0)
                aa.ensemble_seed = ENS_SEED_CONST;
            else if (strcmp(val, "incr") == 0)
                aa.ensemble_seed = ENS_SEED_INCR;
            else if (strcmp(val, "once") == 0)
                aa.ensemble_seed = ENS_SEED_ONCE;
            else if (access(val, R_OK) == 0) {
                strncpy(aa.seed_file, val, sizeof(aa.seed_file) - 1);
                aa.seed_file[sizeof(aa.seed_file) - 1] = '\0';
            } else {
                fprintf(stderr, "[ERROR] --ensemble-seed: '%s' is not "
                        "'const', 'incr', or an existing file\n", val);
                exit(1);
            }
        } else if (strcmp(argv[i], "--optional") == 0 && i + 1 < argc) {
            const char *val = argv[++i];
            if (strcmp(val, "target-norm") == 0) {
                aa.opt_target_norm = !KI_DEFAULT_TARGET_NORM;
            } else {
                fprintf(stderr, "[ERROR] --optional: unknown option '%s'. Valid: target-norm\n", val);
                exit(1);
            }
        } else {
            fprintf(stderr, "[ERROR] Unknown argument: %s\nTry --help\n", argv[i]);
            exit(1);
        }
    }
    /* ── Post-processing: enc_array füllen/expandieren ──────────────
     * 1. Wenn enc_array leer: Defaults pro aktivem Kanal eintragen
     * 2. Wenn enc_array Einträge mit color=-1 (bare tokens): expandieren
     *    auf alle aktiven Kanäle.
     * 3. Channel-Maske aus enc_array ableiten (falls nicht explizit). */
    {   int first_bare = -1;  /* index of first color=-1 entry */
        for (int i = 0; i < aa.enc_count && i < KI_ENC_MAX; i++)
            if (aa.enc_array[i].color < 0) { first_bare = i; break; }

        if (aa.enc_count == 0) {
            /* Kein --encoding: Defaults pro aktivem Kanal */
            int def_type = aa.debug_binarize ? KI_ENC_LIN7 : KI_ENC_RAW;
            for (int b = 0; b < COLOR_NB; b++) {
                if (!(aa.channel & (1 << b))) continue;
                if (aa.enc_count < KI_ENC_MAX) {
                    aa.enc_array[aa.enc_count].type  = (int8_t)def_type;
                    aa.enc_array[aa.enc_count].width = KI_ENC_WIDTH_DEFAULT;
                    aa.enc_array[aa.enc_count].color = (int8_t)b;
                    aa.enc_count++;
                }
            }
        } else if (first_bare >= 0) {
            /* Mindestens ein color=-1 Eintrag → expandieren */
            int old_count = aa.enc_count;
            for (int i = 0; i < old_count; i++) {
                if (aa.enc_array[i].color >= 0) continue;  /* bereits per-block */
                int typ = (int)aa.enc_array[i].type;
                int w   = (int)aa.enc_array[i].width;
                int first = 1;
                for (int b = 0; b < COLOR_NB; b++) {
                    if (!(aa.channel & (1 << b))) continue;
                    if (first) {
                        aa.enc_array[i].color = (int8_t)b;  /* replace in-place */
                        first = 0;
                    } else {
                        if (aa.enc_count < KI_ENC_MAX) {
                            aa.enc_array[aa.enc_count].type  = (int8_t)typ;
                            aa.enc_array[aa.enc_count].width = (int8_t)w;
                            aa.enc_array[aa.enc_count].color = (int8_t)b;
                            aa.enc_count++;
                        }
                    }
                }
            }
        }
        /* Channel-Maske aus enc_array ableiten */
        if (!aa.channel_explicit) {
            int enc_mask = 0;
            for (int i = 0; i < aa.enc_count; i++)
                if (aa.enc_array[i].color >= 0)
                    enc_mask |= (1 << aa.enc_array[i].color);
            if (enc_mask) aa.channel = enc_mask;
        }
    }
    /* ── dry_run override: --dry-run = epochs=0 regardless of --epochsN ── */
    if (aa.dry_run) aa.epochs = 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * LR SCHEDULE — Cosine Decay + Linear Warmup
 * ═══════════════════════════════════════════════════════════════════════
 * Returns a float multiplier (0..1) for the base LR.
 *
 *   warmup > 0 : linear increase 0→1 over first `warmup` epochs
 *   after warmup: cosine decay from base_lr down to lr_min
 *   no_decay=1: always returns base_lr (identity)
 */
static inline float ki_lr_schedule(int epoch, int total_epochs, int warmup,
                                    float base_lr, float lr_min, int no_decay) {
    if (no_decay) return base_lr;
    if (epoch < warmup)
        return base_lr * (float)(epoch + 1) / (float)warmup;
    int decay_epochs = total_epochs - warmup;
    if (decay_epochs <= 0) return base_lr;
    float progress = (float)(epoch - warmup) / (float)decay_epochs;
    float cosine = (float)(1.0 + cos(progress * 3.14159265358979323846)) / 2.0f;
    return lr_min + (base_lr - lr_min) * cosine;
}

/* ═══════════════════════════════════════════════════════════════════════
 * OT_PRECISION — Skalierungshilfe: in × F + 0.5-Rounding
 * ═══════════════════════════════════════════════════════════════════════
 * F = (1<<OT_PRECISION).  Alle logit/log(p)-Werte werden mit F skaliert
 * in int32/int64 gespeichert.  ot_precision() rundet kaufmännisch.
 */
#define OT_F (1 << OT_PRECISION)
static inline double ot_precision(double in) {
    return in * (double)OT_F + (in >= 0 ? 0.5 : -0.5);
}

/* ── Mode-String mit Parameter (für TRAINING Header und --help) ── */
/* Gibt "pow(2.4)", "const(155)", "cos-time" etc. zurück.
 * Nutzt static buffer für snprintf-Modi.
 * Das ist eine reine Anzeigefunktion — der eigentliche step-Wert
 * wird im TRAINING-Header separat ausgegeben. */
__attribute__((unused))
static const char *mode_str(void) {
    static char _mode_buf[64];
    switch (aa.step_mode) {
      case STEP_POW:
          snprintf(_mode_buf, sizeof(_mode_buf), "pow(%.3g)", (double)aa.step_power);
          return _mode_buf;
      case STEP_COS_TIME: 
          snprintf(_mode_buf, sizeof(_mode_buf), "cos-time(%d)", aa.warmup_epochs);
          return _mode_buf;
      case STEP_COS_ERR:  return "cos-err";
      case STEP_CONST: {
          int cstep = (aa.stepN > 0) ? aa.stepN : (int)(aa.lr * (double)OT_F + 0.5);
          snprintf(_mode_buf, sizeof(_mode_buf), "const(%d)", cstep);
          return _mode_buf;
      }
      default:            return "?";
    }
}

/* ── Color-String (für --help und SETUP Header) ────────────────── */
/* Gibt "R,G,B", "packed:MNIST", "LUM,RG,BY" etc. aus der
 * aa.channel Maske + aa.packedB Flag.
 * Nutzt static buffer für snprintf, analog zu mode_str(). */
__attribute__((unused))
static const char *color_str(void) {
    static char _color_buf[64];
    int pos = 0;
    if (aa.debug_flat)
        pos += snprintf(_color_buf, sizeof(_color_buf), "flat,");
    if (aa.packedB)
        pos += snprintf(_color_buf + pos, sizeof(_color_buf) - (size_t)pos, "packed:");
    for (int b = 0; b < COLOR_NB && pos < (int)sizeof(_color_buf) - 3; b++) {
        if (aa.channel & (1 << b)) {
            if (pos > 0 && _color_buf[pos-1] != ':' && _color_buf[pos-1] != ',') {
                _color_buf[pos++] = ',';
            }
            const char *n = ki_color_name(b);
            while (*n && pos < (int)sizeof(_color_buf) - 2) {
                _color_buf[pos++] = *n++;
            }
        }
    }
    if (pos == 0) { _color_buf[pos++] = '?'; }
    _color_buf[pos] = '\0';
    return _color_buf;
}

/* ── Encoding-String (für --help und TRAINING Header) ──────────── */
/* Gibt "R=exp8,G=lin8,B=sig8" etc. aus — immer aus enc_array[].
 * Bei MNIST (single channel) ohne Color-Präfix: "exp8,sig8".
 * Nutzt static buffer, analog zu color_str(). */
__attribute__((unused))
static const char *enc_str(void) {
    static char _enc_buf[128];
    int pos = 0;

    int show_color = (KI_COLORS > 1);  /* CIFAR: show color=enc, MNIST: enc only */
    for (int i = 0; i < aa.enc_count && i < KI_ENC_MAX; i++) {
        if (pos > 0) _enc_buf[pos++] = ',';
        int col = (int)aa.enc_array[i].color;
        int typ = (int)aa.enc_array[i].type;
        int w   = (int)aa.enc_array[i].width;
        if (show_color && col >= 0 && col < COLOR_NB) {
            const char *cn = ki_color_name(col);
            while (*cn && pos < (int)sizeof(_enc_buf) - 2) _enc_buf[pos++] = *cn++;
            _enc_buf[pos++] = '=';
        }
        const char *en = ki_enc_name_short((int8_t)typ);
        while (*en && pos < (int)sizeof(_enc_buf) - 8) _enc_buf[pos++] = *en++;
        int written = snprintf(_enc_buf + pos, (size_t)(sizeof(_enc_buf) - (size_t)pos), "%d", w);
        if (written > 0) pos += written;
    }
    if (pos == 0) { _enc_buf[pos++] = '?'; }
    _enc_buf[pos] = '\0';
    return _enc_buf;
}

/* ═══════════════════════════════════════════════════════════════════════
 * CORRECTION — atomare Target-Updates (für Einzelsample-Korrektur)
 * ═══════════════════════════════════════════════════════════════════════
 * Für ein fehlklassifiziertes Sample (true_k ≠ pred):
 *   target[true_k][h][v] += step   für jedes aktive virtuelle Neuron
 *   target[pred][h][v]   -= step
 *
 * Schreibt per `#pragma omp atomic` direkt auf shared target
 * (im Gegensatz zu ki_batch_correct, das Thread-Caches nutzt).
 */
static inline void ki_correct_target(int32_t *target, const uint32_t *h0_s,
                                      int H, int true_k, int pred, int step) {
    int V = VN_GROUPS_, G = aa.splitVN, TH = VN_THRESH_;
    for (int h = 0; h < H; h++) {
        uint32_t h0 = h0_s[h];
        uint32_t gbits;
        if (G == 1) {
            gbits = h0;
        } else {
            gbits = 0;
            for (int v = 0; v < V; v++) {
                uint32_t slice = (h0 >> (v * G)) & ((1u << G) - 1u);
                if (__builtin_popcount(slice) > TH) gbits |= (1u << v);
            }
        }
        while (gbits) {
            int v = __builtin_ctz(gbits);
            #pragma omp atomic
            target[TGT_IDX(true_k, h, v, H, V)] += step;
            #pragma omp atomic
            target[TGT_IDX(pred, h, v, H, V)] -= step;
            gbits &= gbits - 1;
        }
    }
}


/* ═══════════════════════════════════════════════════════════════════════
 * MEMORY HELPERS
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void *ki_xmalloc(size_t size) {
    if (size == 0) return NULL;
    void *ptr = malloc(size);
    if (!ptr) { fprintf(stderr, "[FATAL] ki_xmalloc(%zu) failed\n", size); exit(1); }
    return ptr;
}

static inline void *ki_xcalloc(size_t nmemb, size_t size) {
    if (nmemb == 0 || size == 0) return NULL;
    void *ptr = calloc(nmemb, size);
    if (!ptr) { fprintf(stderr, "[FATAL] ki_xcalloc(%zu, %zu) failed\n", nmemb, size); exit(1); }
    return ptr;
}

/* ═══════════════════════════════════════════════════════════════════════
 * VIRTUAL NEURON (VN) KERNELS — compile-time-optimized per splitVN
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Jeder splitVN-Wert (1,2,4,8,16,32) hat eigene Konstanten G, V, TH
 * zur Compile-Zeit → Loop-Unrolling, konstanter Mask-Shift, kein
 * Switch innerhalb der Sample/Neuron-Loops.
 *
 * Verwendet von: ki_batch_correct, scores_otto, ki_build_target
 *
 *   VN_SCORE(h0, h, H, target, scores):
 *       Accumuliert target[][h][v][] in scores[] für ein Neuron.
 *   VN_CORRECT(h0, h, H, dc, true_k, pred_k, step_i):
 *       Wendet step_i auf dc[true_k] und dc[pred_k] an.
 */

#define VN_SCORE_1(h0, h, H, TGT, SC) do { \
    uint32_t _b = (h0); \
    while (_b) { int _v = __builtin_ctz(_b); \
        for (int _k = 0; _k < KI_NCLASSES; _k++) \
            (SC)[_k] += (TGT)[TGT_IDX(_k, (h), _v, H, 32)]; \
        _b &= _b - 1; } \
} while (0)

#define VN_CORRECT_1(h0, h, H, DC, TK, PK, SI) do { \
    uint32_t _b = (h0); \
    while (_b) { int _v = __builtin_ctz(_b); \
        (DC)[TGT_IDX((TK), (h), _v, H, 32)] += (SI); \
        (DC)[TGT_IDX((PK), (h), _v, H, 32)] -= (SI); \
        _b &= _b - 1; } \
} while (0)

/* ── VN=2: 16 groups of 2 bits, TH=1 ───────────────────────── */
#define VN_SCORE_2(h0, h, H, TGT, SC) do { \
    uint32_t _gb = 0; \
    _gb |= (__builtin_popcount((h0) & 0x3u) > 1) ? 0x0001u : 0; \
    _gb |= (__builtin_popcount(((h0)>>2) & 0x3u) > 1) ? 0x0002u : 0; \
    _gb |= (__builtin_popcount(((h0)>>4) & 0x3u) > 1) ? 0x0004u : 0; \
    _gb |= (__builtin_popcount(((h0)>>6) & 0x3u) > 1) ? 0x0008u : 0; \
    _gb |= (__builtin_popcount(((h0)>>8) & 0x3u) > 1) ? 0x0010u : 0; \
    _gb |= (__builtin_popcount(((h0)>>10) & 0x3u) > 1) ? 0x0020u : 0; \
    _gb |= (__builtin_popcount(((h0)>>12) & 0x3u) > 1) ? 0x0040u : 0; \
    _gb |= (__builtin_popcount(((h0)>>14) & 0x3u) > 1) ? 0x0080u : 0; \
    _gb |= (__builtin_popcount(((h0)>>16) & 0x3u) > 1) ? 0x0100u : 0; \
    _gb |= (__builtin_popcount(((h0)>>18) & 0x3u) > 1) ? 0x0200u : 0; \
    _gb |= (__builtin_popcount(((h0)>>20) & 0x3u) > 1) ? 0x0400u : 0; \
    _gb |= (__builtin_popcount(((h0)>>22) & 0x3u) > 1) ? 0x0800u : 0; \
    _gb |= (__builtin_popcount(((h0)>>24) & 0x3u) > 1) ? 0x1000u : 0; \
    _gb |= (__builtin_popcount(((h0)>>26) & 0x3u) > 1) ? 0x2000u : 0; \
    _gb |= (__builtin_popcount(((h0)>>28) & 0x3u) > 1) ? 0x4000u : 0; \
    _gb |= (__builtin_popcount(((h0)>>30) & 0x3u) > 1) ? 0x8000u : 0; \
    while (_gb) { int _v = __builtin_ctz(_gb); \
        for (int _k = 0; _k < KI_NCLASSES; _k++) \
            (SC)[_k] += (TGT)[TGT_IDX(_k, (h), _v, H, 16)]; \
        _gb &= _gb - 1; } \
} while (0)

#define VN_CORRECT_2(h0, h, H, DC, TK, PK, SI) do { \
    uint32_t _gb = 0; \
    _gb |= (__builtin_popcount((h0) & 0x3u) > 1) ? 0x0001u : 0; \
    _gb |= (__builtin_popcount(((h0)>>2) & 0x3u) > 1) ? 0x0002u : 0; \
    _gb |= (__builtin_popcount(((h0)>>4) & 0x3u) > 1) ? 0x0004u : 0; \
    _gb |= (__builtin_popcount(((h0)>>6) & 0x3u) > 1) ? 0x0008u : 0; \
    _gb |= (__builtin_popcount(((h0)>>8) & 0x3u) > 1) ? 0x0010u : 0; \
    _gb |= (__builtin_popcount(((h0)>>10) & 0x3u) > 1) ? 0x0020u : 0; \
    _gb |= (__builtin_popcount(((h0)>>12) & 0x3u) > 1) ? 0x0040u : 0; \
    _gb |= (__builtin_popcount(((h0)>>14) & 0x3u) > 1) ? 0x0080u : 0; \
    _gb |= (__builtin_popcount(((h0)>>16) & 0x3u) > 1) ? 0x0100u : 0; \
    _gb |= (__builtin_popcount(((h0)>>18) & 0x3u) > 1) ? 0x0200u : 0; \
    _gb |= (__builtin_popcount(((h0)>>20) & 0x3u) > 1) ? 0x0400u : 0; \
    _gb |= (__builtin_popcount(((h0)>>22) & 0x3u) > 1) ? 0x0800u : 0; \
    _gb |= (__builtin_popcount(((h0)>>24) & 0x3u) > 1) ? 0x1000u : 0; \
    _gb |= (__builtin_popcount(((h0)>>26) & 0x3u) > 1) ? 0x2000u : 0; \
    _gb |= (__builtin_popcount(((h0)>>28) & 0x3u) > 1) ? 0x4000u : 0; \
    _gb |= (__builtin_popcount(((h0)>>30) & 0x3u) > 1) ? 0x8000u : 0; \
    while (_gb) { int _v = __builtin_ctz(_gb); \
        (DC)[TGT_IDX((TK), (h), _v, H, 16)] += (SI); \
        (DC)[TGT_IDX((PK), (h), _v, H, 16)] -= (SI); \
        _gb &= _gb - 1; } \
} while (0)

/* ── VN=4: 8 groups of 4 bits, TH=2 ────────────────────────── */
#define VN_SCORE_4(h0, h, H, TGT, SC) do { \
    uint32_t _gb = 0; \
    _gb |= (__builtin_popcount((h0) & 0xFu) > 2) ? 1u<<0 : 0; \
    _gb |= (__builtin_popcount(((h0)>>4) & 0xFu) > 2) ? 1u<<1 : 0; \
    _gb |= (__builtin_popcount(((h0)>>8) & 0xFu) > 2) ? 1u<<2 : 0; \
    _gb |= (__builtin_popcount(((h0)>>12) & 0xFu) > 2) ? 1u<<3 : 0; \
    _gb |= (__builtin_popcount(((h0)>>16) & 0xFu) > 2) ? 1u<<4 : 0; \
    _gb |= (__builtin_popcount(((h0)>>20) & 0xFu) > 2) ? 1u<<5 : 0; \
    _gb |= (__builtin_popcount(((h0)>>24) & 0xFu) > 2) ? 1u<<6 : 0; \
    _gb |= (__builtin_popcount(((h0)>>28) & 0xFu) > 2) ? 1u<<7 : 0; \
    while (_gb) { int _v = __builtin_ctz(_gb); \
        for (int _k = 0; _k < KI_NCLASSES; _k++) \
            (SC)[_k] += (TGT)[TGT_IDX(_k, (h), _v, H, 8)]; \
        _gb &= _gb - 1; } \
} while (0)

#define VN_CORRECT_4(h0, h, H, DC, TK, PK, SI) do { \
    uint32_t _gb = 0; \
    _gb |= (__builtin_popcount((h0) & 0xFu) > 2) ? 1u<<0 : 0; \
    _gb |= (__builtin_popcount(((h0)>>4) & 0xFu) > 2) ? 1u<<1 : 0; \
    _gb |= (__builtin_popcount(((h0)>>8) & 0xFu) > 2) ? 1u<<2 : 0; \
    _gb |= (__builtin_popcount(((h0)>>12) & 0xFu) > 2) ? 1u<<3 : 0; \
    _gb |= (__builtin_popcount(((h0)>>16) & 0xFu) > 2) ? 1u<<4 : 0; \
    _gb |= (__builtin_popcount(((h0)>>20) & 0xFu) > 2) ? 1u<<5 : 0; \
    _gb |= (__builtin_popcount(((h0)>>24) & 0xFu) > 2) ? 1u<<6 : 0; \
    _gb |= (__builtin_popcount(((h0)>>28) & 0xFu) > 2) ? 1u<<7 : 0; \
    while (_gb) { int _v = __builtin_ctz(_gb); \
        (DC)[TGT_IDX((TK), (h), _v, H, 8)] += (SI); \
        (DC)[TGT_IDX((PK), (h), _v, H, 8)] -= (SI); \
        _gb &= _gb - 1; } \
} while (0)

/* ── VN=8: 4 groups of 8 bits, TH=4 ────────────────────────── */
#define VN_SCORE_8(h0, h, H, TGT, SC) do { \
    uint32_t _gb = 0; \
    _gb |= (__builtin_popcount((h0) & 0xFFu) > 4) ? 1u<<0 : 0; \
    _gb |= (__builtin_popcount(((h0)>>8) & 0xFFu) > 4) ? 1u<<1 : 0; \
    _gb |= (__builtin_popcount(((h0)>>16) & 0xFFu) > 4) ? 1u<<2 : 0; \
    _gb |= (__builtin_popcount(((h0)>>24) & 0xFFu) > 4) ? 1u<<3 : 0; \
    while (_gb) { int _v = __builtin_ctz(_gb); \
        for (int _k = 0; _k < KI_NCLASSES; _k++) \
            (SC)[_k] += (TGT)[TGT_IDX(_k, (h), _v, H, 4)]; \
        _gb &= _gb - 1; } \
} while (0)

#define VN_CORRECT_8(h0, h, H, DC, TK, PK, SI) do { \
    uint32_t _gb = 0; \
    _gb |= (__builtin_popcount((h0) & 0xFFu) > 4) ? 1u<<0 : 0; \
    _gb |= (__builtin_popcount(((h0)>>8) & 0xFFu) > 4) ? 1u<<1 : 0; \
    _gb |= (__builtin_popcount(((h0)>>16) & 0xFFu) > 4) ? 1u<<2 : 0; \
    _gb |= (__builtin_popcount(((h0)>>24) & 0xFFu) > 4) ? 1u<<3 : 0; \
    while (_gb) { int _v = __builtin_ctz(_gb); \
        (DC)[TGT_IDX((TK), (h), _v, H, 4)] += (SI); \
        (DC)[TGT_IDX((PK), (h), _v, H, 4)] -= (SI); \
        _gb &= _gb - 1; } \
} while (0)

/* ── VN=16: 2 groups of 16 bits, TH=8 ──────────────────────── */
#define VN_SCORE_16(h0, h, H, TGT, SC) do { \
    uint32_t _gb = 0; \
    if (__builtin_popcount((h0) & 0xFFFFu) > 8) _gb |= 1u<<0; \
    if (__builtin_popcount((h0) >> 16) > 8) _gb |= 1u<<1; \
    while (_gb) { int _v = __builtin_ctz(_gb); \
        for (int _k = 0; _k < KI_NCLASSES; _k++) \
            (SC)[_k] += (TGT)[TGT_IDX(_k, (h), _v, H, 2)]; \
        _gb &= _gb - 1; } \
} while (0)

#define VN_CORRECT_16(h0, h, H, DC, TK, PK, SI) do { \
    uint32_t _gb = 0; \
    if (__builtin_popcount((h0) & 0xFFFFu) > 8) _gb |= 1u<<0; \
    if (__builtin_popcount((h0) >> 16) > 8) _gb |= 1u<<1; \
    while (_gb) { int _v = __builtin_ctz(_gb); \
        (DC)[TGT_IDX((TK), (h), _v, H, 2)] += (SI); \
        (DC)[TGT_IDX((PK), (h), _v, H, 2)] -= (SI); \
        _gb &= _gb - 1; } \
} while (0)

/* ── VN=32: 1 group of 32 bits, TH=16 ──────────────────────── */
#define VN_SCORE_32(h0, h, H, TGT, SC) do { \
    if (__builtin_popcount(h0) > 16) { \
        for (int _k = 0; _k < KI_NCLASSES; _k++) \
            (SC)[_k] += (TGT)[TGT_IDX(_k, (h), 0, H, 1)]; \
    } \
} while (0)

#define VN_CORRECT_32(h0, h, H, DC, TK, PK, SI) do { \
    if (__builtin_popcount(h0) > 16) { \
        (DC)[TGT_IDX((TK), (h), 0, H, 1)] += (SI); \
        (DC)[TGT_IDX((PK), (h), 0, H, 1)] -= (SI); \
    } \
} while (0)
static inline int ki_omp_nthreads(void) {
    int n = 1;
    #pragma omp parallel
    #pragma omp single
    n = omp_get_num_threads();
    return n;
}

static inline int32_t **ki_cache_alloc(int n_threads, size_t tgt_sz) {
    int32_t **cache = (int32_t **)malloc((size_t)n_threads * sizeof(int32_t *));
    if (!cache) { fprintf(stderr, "[FATAL] ki_cache_alloc(%d) failed\n", n_threads); exit(1); }
    for (int t = 0; t < n_threads; t++)
        cache[t] = (int32_t *)ki_xcalloc(tgt_sz, sizeof(int32_t));
    return cache;
}

static inline void ki_cache_apply_free(int32_t **cache, int n_threads,
                                        int32_t *target, size_t tgt_sz) {
    for (int t = 0; t < n_threads; t++) {
        int32_t *ct = cache[t];
        for (size_t i = 0; i < tgt_sz; i++)
            target[i] += ct[i];
        free(ct);
    }
    free(cache);
}

/* ═══════════════════════════════════════════════════════════════════════
 * BATCH CORRECTION — parallel + deterministisch via Mini-Batches
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Phasen pro Batch:
 *   1. Parallel: Scores aus target berechnen, Deltas in Thread-Cache
 *   2. Sequentiell: Deltas auf target anwenden → nächster Batch sieht Änderung
 *
 * Schrittgrösse pro Sample:
 *   gap = sc[pred] - sc[true_k]   (Member-eigener Score-Abstand)
 *   gap > 0 → Korrektur, step skaliert proportional: step × gap / F
 *   gap ≥ F → vollen Schritt
 *   gap ≤ 0 → kein Update (Member lag richtig)
 *
 * target:     Ziel-Target (mit Offset für Ensemble)
 * H:          Anzahl Neuronen
 * class_offset: Offset pro Klasse
 * h0_all:     Vorberechnete H0-Werte [N × H]
 * y:          Labels
 * N:          Anzahl Trainings-Samples
 * step:       Basis-Schritt (Obergrenze, Member skaliert via gap)
 * tgt_sz:     Grösse des Target-Arrays (H × KI_NCLASSES × V)
 *
 * Returns:    Anzahl Korrekturen
 */
static inline int ki_batch_correct(int32_t *target, int H,
                                    const int64_t *class_offset,
                                    const uint32_t *h0_all,
                                    const uint8_t *y,
                                    int N, int step, size_t tgt_sz) {
    int n_threads = ki_omp_nthreads();
    int32_t **dc = ki_cache_alloc(n_threads, tgt_sz);
    int corrections = 0;

    for (int b_start = 0; b_start < N; b_start += aa.batchN) {
        int b_end = b_start + aa.batchN;
        if (b_end > N) b_end = N;
        int batch_corr = 0;
        int batch_multi = 0;  /* Samples mit >1 Gegner (nur multi-correct) */

        #pragma omp parallel for reduction(+:batch_corr,batch_multi) schedule(static)
        for (int s = b_start; s < b_end; s++) {
            int tid = omp_get_thread_num();
            int true_k = (int)y[s];
            const uint32_t *h0_s = h0_all + (size_t)s * (size_t)H;
            int64_t sc[KI_NCLASSES];
            for (int k = 0; k < KI_NCLASSES; k++) sc[k] = class_offset[k];

            /* ── Score mit VN-Grouping ──────────────────────────── */
            for (int h = 0; h < H; h++) {
                uint32_t h0 = h0_s[h];
                switch (aa.splitVN) {
                    case 1:  VN_SCORE_1(h0, h, H, target, sc); break;
                    case 2:  VN_SCORE_2(h0, h, H, target, sc); break;
                    case 4:  VN_SCORE_4(h0, h, H, target, sc); break;
                    case 8:  VN_SCORE_8(h0, h, H, target, sc); break;
                    case 16: VN_SCORE_16(h0, h, H, target, sc); break;
                    case 32: VN_SCORE_32(h0, h, H, target, sc); break;
                }
            }

            int pred = 0;
            for (int k = 1; k < KI_NCLASSES; k++)
                if (sc[k] > sc[pred]) pred = k;

            if (pred != true_k) {
                int64_t gap = sc[pred] - sc[true_k];
                if (gap > 0) {
                    int step_i;
                    if (gap < (int64_t)OT_F) {
                        int64_t scaled = ((int64_t)step * gap) >> OT_PRECISION;
                        step_i = (int)scaled;
                        if (step_i < 1) step_i = 1;
                    } else {
                        step_i = step;
                    }
                    if (aa.multi_correct) {
                        /* ── Multi-Class Target Correction ─────────────
                         * Alle Klassen mit sc[k] > sc[true_k] bestrafen.
                         * Winner (pred) kriegt vollen step_i,
                         * Secondary Opponents teilen sich den Rest
                         * proportional zu ihrem gap.
                         * true_k += step, pred -= step_w, 2nd -= step_2nd */
                        int n_over = 0;
                        int over_k[KI_NCLASSES];
                        int64_t over_gap[KI_NCLASSES];
                        int64_t total_gap = 0;

                        for (int k = 0; k < KI_NCLASSES; k++) {
                            if (k == true_k) continue;
                            int64_t g = sc[k] - sc[true_k];
                            if (g > 0) {
                                over_k[n_over] = k;
                                over_gap[n_over] = g;
                                total_gap += g;
                                n_over++;
                            }
                        }

                        if (n_over > 0) {
                            batch_corr++;
                            if (n_over > 1) {
                                batch_multi++;
                                /* total_gap > winner_gap → größeres Budget
                                 * nötig.  step_i neu auf total_gap basieren. */
                                if (total_gap < (int64_t)OT_F) {
                                    int64_t scaled = ((int64_t)step * total_gap) >> OT_PRECISION;
                                    if (scaled > (int64_t)step) scaled = step;
                                    step_i = (int)scaled;
                                    if (step_i < 1) step_i = 1;
                                } else {
                                    step_i = step;
                                }
                            }
                            /* Step proportional auf ALLE Gegner verteilen
                             * (inkl. pred).  true_k += step (summiert über
                             * alle VN_CORRECT-Aufrufe), jeder Gegner −step_k.
                             * Letzter Gegner kriegt Rest (Rundungsausgleich). */
                            int64_t assigned = 0;
                            for (int oi = 0; oi < n_over; oi++) {
                                int k = over_k[oi];
                                int64_t step_k = (oi == n_over - 1)
                                    ? (int64_t)step_i - assigned
                                    : ((int64_t)step_i * over_gap[oi]) / total_gap;
                                if (step_k < 1) step_k = 1;
                                if (assigned + step_k > step_i)
                                    step_k = step_i - assigned;
                                assigned += step_k;
                                if (step_k > 0) {
                                    for (int h = 0; h < H; h++) {
                                        uint32_t h0 = h0_s[h];
                                        switch (aa.splitVN) {
                                            case 1:  VN_CORRECT_1(h0, h, H, dc[tid], true_k, k, (int)step_k); break;
                                            case 2:  VN_CORRECT_2(h0, h, H, dc[tid], true_k, k, (int)step_k); break;
                                            case 4:  VN_CORRECT_4(h0, h, H, dc[tid], true_k, k, (int)step_k); break;
                                            case 8:  VN_CORRECT_8(h0, h, H, dc[tid], true_k, k, (int)step_k); break;
                                            case 16: VN_CORRECT_16(h0, h, H, dc[tid], true_k, k, (int)step_k); break;
                                            case 32: VN_CORRECT_32(h0, h, H, dc[tid], true_k, k, (int)step_k); break;
    }
}
                                }
                            }
                        }
                    } else {
                        /* ── Original Single-Class Correction ──────────
                         * Nur den Argmax-Gegner bestrafen (wie vor 2026-06-30). */
                        batch_corr++;
                        for (int h = 0; h < H; h++) {
                            uint32_t h0 = h0_s[h];
                            switch (aa.splitVN) {
                                case 1:  VN_CORRECT_1(h0, h, H, dc[tid], true_k, pred, step_i); break;
                                case 2:  VN_CORRECT_2(h0, h, H, dc[tid], true_k, pred, step_i); break;
                                case 4:  VN_CORRECT_4(h0, h, H, dc[tid], true_k, pred, step_i); break;
                                case 8:  VN_CORRECT_8(h0, h, H, dc[tid], true_k, pred, step_i); break;
                                case 16: VN_CORRECT_16(h0, h, H, dc[tid], true_k, pred, step_i); break;
                                case 32: VN_CORRECT_32(h0, h, H, dc[tid], true_k, pred, step_i); break;
                            }
                        }
                    }
                }
            }
        }

        /* Apply + clear cache */
        for (int t = 0; t < n_threads; t++) {
            int32_t *ct = dc[t];
            for (size_t i = 0; i < tgt_sz; i++) {
                int d = ct[i];
                if (d != 0) target[i] += d;
            }
            memset(ct, 0, tgt_sz * sizeof(int32_t));
        }
        corrections += batch_corr;
/*
        if (aa.debug && batch_multi > 1)
            printf("  [MULTI] batch=%d corr=%d multi=%d\n",
                   (int)((b_end - b_start)), batch_corr, batch_multi);
*/
    }

    for (int t = 0; t < n_threads; t++) free(dc[t]);
    free(dc);
    return corrections;
}


/* ═══════════════════════════════════════════════════════════════════════
 * INPUT LOADING — pack uint8 pixels into uint32 containers
 * ═══════════════════════════════════════════════════════════════════════
 * KI_PACK=4 (KI_NC=196): 4 px/cont, p0|p1<<8|p2<<16|p3<<24
 * KI_PACK=1 (KI_NC=784): 1 px/cont, byte-repeat (*0x01010101)
 *
 * Guard KI_COMMON_LOAD_INPUT: überschreibe in eigener Datei
 *   #define KI_COMMON_LOAD_INPUT
 *   #include "ki-common.h"
 *   // eigener load_input
 */
#ifndef KI_COMMON_LOAD_INPUT
static __attribute__((unused)) uint32_t *load_input(const uint8_t *X_raw, int n_samples) {
    uint32_t *Xb = ki_xmalloc((size_t)n_samples * (size_t)KI_NC * sizeof(uint32_t));
#if KI_PACK == 4
    for (int s = 0; s < n_samples; s++) {
        uint32_t *row = Xb + (size_t)s * KI_NC;
        for (int c = 0; c < KI_NC; c++) {
            uint32_t val = 0;
            for (int k = 0; k < 4; k++) {
                size_t p = (size_t)s * (size_t)KI_PX + (size_t)c * 4 + (size_t)k;
                val |= ((uint32_t)X_raw[p] & 0xFFU) << (unsigned)(k * 8);
            }
            row[c] = val;
        }
    }
#elif KI_PACK == 3
    /* 1 px/cont: R|G|B|0 — 32 Bit: 3 Byte Farbe + 1 Byte Null
     * INTENTIONAL: CIFAR-10 stores RGB planar, not interleaved.
     * Each container = one pixel position (R[c], G[c], B[c] from 3 planes). */
    for (int s = 0; s < n_samples; s++) {
        uint32_t *row = Xb + (size_t)s * KI_NC;
        size_t base = (size_t)s * (size_t)KI_PX;
        for (int c = 0; c < KI_NC; c++) {
            uint32_t r = (uint32_t)X_raw[base + (size_t)c];           /* R plane */
            uint32_t g = (uint32_t)X_raw[base + 1024 + (size_t)c];    /* G plane */
            uint32_t b = (uint32_t)X_raw[base + 2048 + (size_t)c];    /* B plane */
            row[c] = r | (g << 8) | (b << 16);
        }
    }
#elif KI_PACK == 1
    for (int s = 0; s < n_samples; s++) {
        uint32_t *row = Xb + (size_t)s * KI_NC;
        for (size_t p = 0; p < KI_PX; p++) {
            size_t off = (size_t)s * (size_t)KI_PX + p;
            row[p] = ((uint32_t)X_raw[off] & 0xFFU) * 0x01010101U;
        }
    }
#else
#  error "load_input: KI_PACK must be 3 (1024), 4 (196) or 1 (784)"
#endif
    return Xb;
}
#endif /* KI_COMMON_LOAD_INPUT */


/* ═══════════════════════════════════════════════════════════════════════
 * SHUFFLE — Fisher-Yates (für Adam/Ref-Trainer)
 * ═══════════════════════════════════════════════════════════════════════ */
static inline void ki_shuffle(int *indices, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int t = indices[i]; indices[i] = indices[j]; indices[j] = t;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * LR CONVERSION — uint32↔float (für alte Adam-Trainer)
 * ═══════════════════════════════════════════════════════════════════════
 * Alte ki_Args-Version hatte uint32 LR Felder. Die Konverter
 * sind für Kompatibilität mit mlp-flt32-w1-adam-trn etc.
 */
static inline float ki_lr_uint_to_float(uint32_t lr_uint) {
    return (float)lr_uint / (float)UINT32_MAX;
}
static inline uint32_t ki_float_to_lr_uint(float lr) {
    if (lr <= 0.0f) return 0;
    if (lr >= 1.0f) return UINT32_MAX;
    return (uint32_t)(lr * (float)UINT32_MAX + 0.5f);
}

/* ═══════════════════════════════════════════════════════════════════════
 * REPORT — Machine-parseable result line
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void ki_report_show(int train_ok, int train_n,
                                   int eval_ok,  int eval_n,
                                   int elapsed_ms, int threadN,
                                   int err, float lr) {
    float tp  = (float)train_ok * 100.0f / (float)train_n;
    float ep  = (float)eval_ok  * 100.0f / (float)eval_n;
    printf("\n============================================================\n");
    printf("REPORT train=%.1f%% (%d) eval=%.1f%% (%d)"
           " err=%d lr=%.4f time=%dms threads=%d\n",
           tp, train_n, ep, eval_n,
           err, (double)lr, elapsed_ms, threadN);
    printf("============================================================\n");
}

#endif /* KI_COMMON_H */
