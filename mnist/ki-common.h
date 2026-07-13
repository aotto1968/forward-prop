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
 * STEP MODE — unique identification des Step-Algorithmus
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Each member gets its own step per epoch.
 * The step mechanism is set via --step-err / --step-const
 * set in struct ki_Args.step_mode gespeichert.
 *
 *   STEP_POW:       step = step_init × (err/total)^step_power  (DEFAULT)
 *                   Decays smoothly with error → kein Overfitting.
 *   STEP_COS_TIME:  step = step_init × cosine(ep/epochs)
 *                   Time-based cosine decay.
 *   STEP_COS_ERR:   step = step_init × cosine(1 - err/total)
 *                   Error-based cosine decay.
 *   STEP_CONST:     step = stepN (wenn stepN>0) otherwise step_init
 *                   Constant step.
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

/* Forward declaration: mode_str(), color_str(), enc_str() are defined below
 * defined (after OT_F), but overwritten by ki_parse_args() in --help
 * already used before OT_F. */
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
 * Each has its own Encoding-Funktion haben.
 * Controlled via --encoding r=lin8,g=up,b=down (per Block) oder
 * --encoding lin8 (default for all active blocks).
 */

/* ═══════════════════════════════════════════════════════════════════════
 * ENCODING — thermometer bitmasks for binary input
 * ═══════════════════════════════════════════════════════════════════════
 * Encoding-Enum, Parser, Apply, LUT und Farbdefinitionen
 * are in the shared header ausgelagert: */
#include "../lib/ki-encoding.h"

/* ── Parser: string → bit position (for --channels) ────────────
 * Returns bit index or -1 if not a color name.
 * Handles: mnist,r,g,b,y,601,lum,l,rg,by,yl,709,auge,packed,full
 */
static inline int ki_color_parse(const char *tok) {
    /* Data-driven: multiple aliases allowed.
     * strcasecmp → case-insensitive. Sonderfaelle: auge=-2, grey=-3,
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

        {"edge",  COLOR_EDGE},
        {"bin",   COLOR_BIN},

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

/* ── Encoding alias lookup (dataset-specific, defined in ki-local.h) ──
 * Each dataset provides its own ki_alias_lookup() via KI_COMMON_ALIAS_LOOKUP
 * guard.  Fallback returns NULL (no aliases). */
#ifndef KI_COMMON_ALIAS_LOOKUP
static const char *ki_alias_lookup(const char *name) {
    (void)name;
    return NULL;
}
#endif

#ifndef MAX
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * VIRTUAL NEURONS (VN) — Bit-Grouping per Container
 * ═══════════════════════════════════════════════════════════════════════
 *
 * --splitVN groups bits of a container into
 * virtual neurons. Each virtual neuron fires when the
 * popcount majority of its bits are set.
 *
 *   VN_BITS(G)  = G       Bits pro virtuellem Neuron (= splitVN)
 *   VN_GROUPS(G) = 32 / G Virtuelle Neuronen per container
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
    int    hidden;          			/* Hidden neurons (--hiddenN, default: 64) */
    int    epochs;          			/* Iterations (--epochsN, default: 1) */
    int    batchN;          			/* Mini-batch size (--batchN, default: 64) */
    int    trainN;          			/* Training samples (--trainN, default: 50000) */
    int    evalN;           			/* Eval samples (--evalN, default: 10000) */
    int    dry_run;         			/* --dry-run: print arch and exit */
    int    debug;           			/* --debug: verbose output */
    unsigned int seed;      			/* Random seed (--seed, default: 42) */
    char   exportD[256];    			/* --export DIR: export directory */
    char   predictions[256]; 			/* --predictions FILE: export per-sample predictions (for vis-errors) */
    char   export_merge_scores[256];   		/* --export-merge-scores DIR: save per-member scores to archive files */
    char   export_scores[256]; 			/* --export-scores FILE: save per-sample scores (10×int64+uint8) */
    char   export_neurons[256]; 		/* --export-neurons FILE: save gb_buf+Target+Offset für Adam-on-neurons */
    float  lr;              			/* Step size (--lr, default: 0.05) */
    float  lr_min;          			/* Min LR fraction (--lr-min, default: 0.1) */
    int    lr_step;         			/* round(aa.lr * (1<<OT_PRECISION)) */
    int    threadN;         			/* OpenMP threads (--threadN, default: 8) */
    int    debug_h0;        			/* --debug-h0: per-neuron debug */
    int    shuffle;         			/* --shuffle: randomize train/eval split */
    int    warmup_epochs;   			/* --warmup N: linear warmup epochs (default: 2) */
    int    step_mode;       			/* enum step_mode: Algorithmus (siehe oben) */
    int    stepN;           			/* --step-const N: const step value (0=use lr, default: 0) */
    float  step_power;      			/* --step-power F: exponent für pow/cos (default: 0.7) */
    float  gap_k;           			/* --gap-k K: exp(-K×gap) step damping when train/eval gap widens (default: 0.0=off) */
    int    err_rollback;    			/* --err-rollback: rollback targets when err increases (default: 0) */
    int    ensembleN;       			/* --ensembleN N: independent W0 copies (default: 1) */
    int    splitVN;         			/* --splitVN N: vertical H split (default: 1) */
    int    splitHN;         			/* --splitHN N: horizontal NC split (default: 1) */
    int    channel;        			/* --channels bitmask of selected blocks */
    int    channel_explicit; 			/* 1=--channels was set explicitly */
    int    packedB;    			        /* 1=4px/cont (256/blk), 0=1px/cont (1024/blk) */
    int    debug_flat;      			/* 1=all selected blocks in one flat array, 0=separate members */
    int    debug_binarize;  			/* 1=threshold block values at 128 → 0x00/0xFF per pixel */
    int    hebbian_pct;     			/* --hebbian-pct: flip threshold % (reference, default 50) */
    int8_t enc[COLOR_NB];         		/* --encoding: per-block encoding type (-1=not set, derived from enc_array) */
    int8_t enc_width[COLOR_NB];  		/* --encoding: per-block width (derived from enc_array) */
    int8_t enc_default_type;      		/* --encoding: fallback type for blocks without specific setting */
    int8_t enc_default_width;     		/* --encoding: fallback width (8, 16, 32) */
    ki_EncSlot enc_array[KI_ENC_MAX];  		/* Alle aktiven Encodings als (type,width)-Paare */
    int         enc_count;             		/* number of Eintraege in enc_array */
    int    opt_target_norm;    			/* --optional target-norm: vote normalisierung aktivieren */
    char   seed_file[256]; 			/* --seed-file PATH: true random source */
    char   importD[512];    			/* --import DIR: load model for inference */
    int    seed_splitmix;  			/* --seed-splitmix: ignore seed_file, use splitmix64 PRNG */
    int    multi_correct;  			/* --multi-correct: punish all over true_k (default: 1) */
    int    target_random_init; 			/* --target-random-init: random targets statt counting (default: 0) */
    int    ensemble_seed;    			/* ENS_SEED_ONCE|CONST|INCR (default: ONCE) */
    int    debug_class_voting; 			/* --debug-class-voting: Member × Class accuracy (end only) */
    int    debug_class_voting_all; 		/* --debug-class-voting-all: every epoch */
    int    debug_confusion;    			/* --debug-confusion-matrix: confusion matrix (end only) */
    int    debug_confusion_all; 		/* --debug-confusion-matrix-all: every epoch */
    char   filter_str[128];    			/* --filter "0,1,airplan,cat": raw string */
    int    filter_mask;        			/* computed bitmask from filter_str (0 = no filter) */
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
    /* enc[] initialisieren: -1 = "nicht gesetzt" (default resolved later) */
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
            printf("  --gap-k F         Exp(-K × gap) step damping when overfitting gap widens          (default: %.1f)\n", (double)aa.gap_k);
            printf("                    gap = train_err%% - eval_err%%  |  step *= exp(-K × gap)\n");
            printf("  --err-rollback    Rollback targets when training err increases                  (default: off)\n");
            printf("  --warmup N        Linear warmup epochs                                          (default: %d, 0=off)\n", aa.warmup_epochs);
            printf("  ---------------------------------------------------------------------------------------------\n");
            printf("  --seed N          Random seed                                                   (default: %d)\n", aa.seed);
            printf("  --seed-file PATH-TO-RANDOM-FILE                                                 (default: none)\n");
            printf("                    Use true random data from file instead of PRNG\n");
            printf("  --seed-splitmix   Use splitmix64 PRNG (default: on)                                      (default: off)\n");
            printf("  --seed-member const|incr|once  W0 seeding mode.                                 (default: %s)\n", ensemble_seed_str());
            printf("                    once    : one seed, all sequential.\n");
            printf("                    const   : same W0 per ensemble (tests color diversity).\n");
            printf("                    incr    : unique seed per member.\n");
            printf("  ---------------------------------------------------------------------------------------------\n");
            printf("  --channels [packed|full,][flat,]...                                             (default: %s)\n", color_str());
#if KI_COLORS > 1
            printf("                    Channel selection (comma-sep).  Encoding via --encoding.\n");
            printf("                    packed/full   : 4px/cont or 1px/cont,\n");
            printf("                    flat          : all selected blocks in one wide W0,\n");
            printf("                    auge          : lum|al=(R+G)/2, rg|am=R-G, by|ap=B-(R+G)/2,\n");
            printf("                                  :     bl=(R+B)/2,    bm=R-B,    bp=G-(R+B)/2,\n");
            printf("                    diff          : rg=R-G, rb=R-B, gb=G-B (color opponent),\n");
            printf("                    rgb           : r=R, g=G, b=B,\n");
            printf("                    grey          : y|601=ITU-601, yl|709=ITU-709\n");
            printf("                    h             : hue (Farbwinkel, atan2-basiert)\n");
            printf("                    s             : saturation (Farbsaettigung, max-min)\n");
            printf("                    c             : contrast (Sobel-Kanten auf LUM)\n");
            printf("                    edge          : edges via Sobel on Y luminance\n");
            printf("                    bin           : Otsu-binarized Y luminance (filled black/white regions)\n");
#else
            printf("                    mnist         : single grayscale block (only available channel)\n");
#endif
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
            printf("  --export-merge-scores DIR  Save per-member scores to archive files for merge    (default: none)\n");
            printf("  --export-scores FILE  Save per-sample ensemble scores (10×int64+uint8)          (default: none)\n");
            printf("  --export-neurons FILE  Save gb_buf+Target+Offset per member (v3) for Adam..     (default: none)\n");
            printf("  --export DIR      Export directory                                              (default: none)\n");
            printf("  --import DIR      Load model for inference                                      (default: none)\n");
            printf("  --predictions FILE                                                              (default: none)\n");
            printf("                    Export per-sample predictions (for vis-errors, eval only)\n");
            printf("  --optional target-norm  Vote normalisierung (equal voting power)                (default: off)\n");
            printf("  --?no-?multi-correct  Only punish argmax, not all over true_k                   (default: multi-correct)\n");
            printf("  --?no-?target-random-init  Random targets statt counting (correction baut auf)  (default: off)\n");
            printf("  ---------------------------------------------------------------------------------------------\n");
            printf("  --dry-run         Print architecture and exit                                   (default: off)\n");
            printf("  --quick           5000 train / 2000 eval\n");
            printf("  --qq              5000 train / 2000 eval / 3 epochs\n");
            printf("  --debug           Verbose output                                                (default: off)\n");
            printf("  --debug-h0        Per-neuron debug                                              (default: off)\n");
            printf("  --debug-class-voting?-all?  Member × Class accuracy table (end only)            (default: off)\n");
            printf("  --debug-confusion-matrix?-all?  Confusion matrix table (end only)               (default: off)\n");
            printf("  --filter #,#,... or name,name,...  Restrict to specific classes only            (default: none)\n");
            printf("                    Examples: --filter 0,1 or --filter name1,name2\n");
            printf("                    Available:");
            for (int _k = 0; _k < KI_NCLASSES; _k++)
                printf(" %s(%d)", ki_class_names[_k], _k);
            printf("\n");
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
            /* NOTE: epochs=0 bedeutet "counting-only + eval, kein Training".
             * Do not couple with dry-run — --dry-run must be explicit. */
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
        } else if (strcmp(argv[i], "--gap-k") == 0 && i + 1 < argc) {
            aa.gap_k = (float)atof(argv[++i]);
            if (aa.gap_k < 0.0f) aa.gap_k = 0.0f;
        } else if (strcmp(argv[i], "--err-rollback") == 0) {
            aa.err_rollback = 1;
        } else if (strcmp(argv[i], "--iter") == 0 && i + 1 < argc) {
            i++;  /* ignored (BV32 compatibility) */
        } else if (strcmp(argv[i], "--lr-min") == 0 && i + 1 < argc) {
            aa.lr_min = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            aa.seed = (unsigned int)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--export") == 0 || strcmp(argv[i], "--export") == 0) {
            if (i + 1 >= argc) { fprintf(stderr, "[ERROR] --export DIR\n"); exit(1); }
            strncpy(aa.exportD, argv[++i], sizeof(aa.exportD) - 1);
            aa.exportD[sizeof(aa.exportD) - 1] = '\0';
        } else if (strcmp(argv[i], "--predictions") == 0 && i + 1 < argc) {
            strncpy(aa.predictions, argv[++i], sizeof(aa.predictions) - 1);
            aa.predictions[sizeof(aa.predictions) - 1] = '\0';
        } else if (strcmp(argv[i], "--export-merge-scores") == 0 && i + 1 < argc) {
            if (i + 1 >= argc) { fprintf(stderr, "[ERROR] --export-merge-scores DIR\n"); exit(1); }
            strncpy(aa.export_merge_scores, argv[++i], sizeof(aa.export_merge_scores) - 1);
            aa.export_merge_scores[sizeof(aa.export_merge_scores) - 1] = '\0';
        } else if (strcmp(argv[i], "--export-scores") == 0 && i + 1 < argc) {
            strncpy(aa.export_scores, argv[++i], sizeof(aa.export_scores) - 1);
            aa.export_scores[sizeof(aa.export_scores) - 1] = '\0';
        } else if (strcmp(argv[i], "--export-neurons") == 0 && i + 1 < argc) {
            strncpy(aa.export_neurons, argv[++i], sizeof(aa.export_neurons) - 1);
            aa.export_neurons[sizeof(aa.export_neurons) - 1] = '\0';
        } else if (strcmp(argv[i], "--import") == 0 || strcmp(argv[i], "--import") == 0) {
            if (i + 1 >= argc) { fprintf(stderr, "[ERROR] --import DIR\n"); exit(1); }
            strncpy(aa.importD, argv[++i], sizeof(aa.importD) - 1);
            aa.importD[sizeof(aa.importD) - 1] = '\0';
        } else if (strcmp(argv[i], "--debug-h0") == 0) {
            aa.debug_h0 = 1;
        } else if (strcmp(argv[i], "--debug-class-voting") == 0) {
            aa.debug_class_voting = 1;
        } else if (strcmp(argv[i], "--debug-class-voting-all") == 0) {
            aa.debug_class_voting_all = 1;
        } else if (strcmp(argv[i], "--debug-confusion-matrix") == 0) {
            aa.debug_confusion = 1;
        } else if (strcmp(argv[i], "--debug-confusion-matrix-all") == 0) {
            aa.debug_confusion_all = 1;
        } else if (strcmp(argv[i], "--filter") == 0 && i + 1 < argc) {
            const char *val = argv[++i];
            strncpy(aa.filter_str, val, sizeof(aa.filter_str) - 1);
            aa.filter_str[sizeof(aa.filter_str) - 1] = '\0';
            if (aa.filter_str[0] == '\0') {
                fprintf(stderr, "[ERROR] --filter: empty string\n");
                exit(1);
            }
            /* Parse immediately into bitmask — needs ki_class_names[] (available
             * since ki-local.h is included before ki_parse_args). */
            aa.filter_mask = 0;
            char fbuf[128];
            strncpy(fbuf, aa.filter_str, sizeof(fbuf) - 1);
            fbuf[sizeof(fbuf) - 1] = '\0';
            for (char *tok = strtok(fbuf, ","); tok; tok = strtok(NULL, ",")) {
                while (*tok == ' ' || *tok == '\t') tok++;
                if (*tok == '\0') continue;
                char *end = NULL;
                long c = strtol(tok, &end, 10);
                if (end != tok && *end == '\0') {
                    if (c < 0 || c >= KI_NCLASSES) {
                        fprintf(stderr, "[ERROR] --filter: invalid class index %ld (0..%d)\n",
                                c, KI_NCLASSES - 1);
                        exit(1);
                    }
                    aa.filter_mask |= (1 << (int)c);
                    continue;
                }
                int found = 0;
                for (int _k = 0; _k < KI_NCLASSES; _k++) {
                    if (strcasecmp(tok, ki_class_names[_k]) == 0) {
                        aa.filter_mask |= (1 << _k);
                        found = 1; break;
                    }
                }
                if (!found) {
                    fprintf(stderr, "[ERROR] --filter: unknown class '%s'.\n", tok);
                    exit(1);
                }
            }
            if (aa.filter_mask == 0) {
                fprintf(stderr, "[ERROR] --filter: at least one class required\n");
                exit(1);
            }
        } else if (strcmp(argv[i], "--shuffle") == 0) {
            aa.shuffle = 1;
        } else if (strcmp(argv[i], "--ensembleN") == 0 && i + 1 < argc) {
            aa.ensembleN = atoi(argv[++i]);
            if (aa.ensembleN < 1) aa.ensembleN = 1;
        } else if (strcmp(argv[i], "--splitVN") == 0 && i + 1 < argc) {
            aa.splitVN = atoi(argv[++i]);
            if (aa.splitVN != 1 && aa.splitVN != 2 && aa.splitVN != 3 && aa.splitVN != 4
                && aa.splitVN != 8 && aa.splitVN != 16 && aa.splitVN != 32) {
                fprintf(stderr, "[ERROR] --splitVN: expected 1,2,3,4,8,16,32, got %d\n", aa.splitVN);
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
        } else if (strcmp(argv[i], "--target-random-init") == 0) {
            aa.target_random_init = 1;
        } else if (strcmp(argv[i], "--no-target-random-init") == 0) {
            aa.target_random_init = 0;
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
             * Dataset-specific aliases defined in ki-local.h
             * via ki_alias_lookup().  Iterative 5-pass expansion:
             * Phase 1: full-string match, Phase 2: per-token. */
            for (int _iter = 0; _iter < 5; _iter++) {
                /* Phase 1: full string match */
                const char *_full = ki_alias_lookup(buf);
                if (_full) {
                    strncpy(buf, _full, sizeof(buf) - 1);
                    buf[sizeof(buf) - 1] = '\0';
                    continue;
                }
                /* Phase 2: per-token expand */
                char _tmp[256], _new[256] = "";
                strncpy(_tmp, buf, sizeof(_tmp) - 1);
                _tmp[sizeof(_tmp) - 1] = '\0';
                char *_t = strtok(_tmp, ",");
                int _any = 0;
                while (_t) {
                    const char *_val = ki_alias_lookup(_t);
                    if (_val) {
                        if (_new[0]) strncat(_new, ",", sizeof(_new) - 1);
                        strncat(_new, _val, sizeof(_new) - strlen(_new) - 1);
                        _any = 1;
                    } else {
                        if (_new[0]) strncat(_new, ",", sizeof(_new) - 1);
                        strncat(_new, _t, sizeof(_new) - strlen(_new) - 1);
                    }
                    _t = strtok(NULL, ",");
                }
                if (!_any) break; /* no further expansions */
                strncpy(buf, _new, sizeof(buf) - 1);
                buf[sizeof(buf) - 1] = '\0';
            }

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
                    /* In enc_array eintragen — the only encoding path */
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
                     * Each token means ONE entry in enc_array[]. */
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
                    /* First token also sets enc_default (Backward compat) */
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
    /* ── Post-processing: fill.expand enc_array ──────────────
     * 1. Wenn enc_array leer: Defaults pro aktivem Kanal eintragen
     * 2. If enc_array entries have color=-1 (bare tokens: expand
     *    to all active channels.
     * 3. Channel-Maske aus enc_array ableiten (if not explicit). */
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
 * F = (1<<OT_PRECISION).  All logit values are scaled by F
 * in int32/int64 gespeichert.  ot_precision() rundet kaufmaennisch.
 */
#define OT_F (1 << OT_PRECISION)
static inline double ot_precision(double in) {
    return in * (double)OT_F + (in >= 0 ? 0.5 : -0.5);
}

/* ── Mode string with parameter (for TRAINING header and --help) ── */
/* Returns "pow()", "const()", "cos-time" etc..
 * Uses static buffer for snprintf-Modi.
 * This is purely for display — the actual step value
 * is output separately in the TRAINING header. */
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
 * Uses static buffer for snprintf, similar to mode_str(). */
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
 * For MNIST without color prefix: "exp8,sig8".
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
 * CORRECTION — atomic target updates (for single-sample correction)
 * ═══════════════════════════════════════════════════════════════════════
 * For a misclassified sample (true_k ≠ pred):
 *   target[true_k][h][v] += step   for each active virtual neuron
 *   target[pred][h][v]   -= step
 *
 * Writes via #pragma omp atomic directly to shared target
 * (unlike ki_batch_correct, which uses thread caches).
 */



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
 * (macros defined in mlp-bin32-otto-trn.c)
 */

/* ═══════════════════════════════════════════════════════════════════════
 * INPUT LOADING — pack uint8 pixels into uint32 containers
 * ═══════════════════════════════════════════════════════════════════════
 * KI_PACK=4 (KI_NC=196): 4 px/cont, p0|p1<<8|p2<<16|p3<<24
 * KI_PACK=1 (KI_NC=784): 1 px/cont, byte-repeat (*0x01010101)
 *
 * Guard KI_COMMON_LOAD_INPUT: override in own file
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
 * SHUFFLE — Fisher-Yates (for Adam/Ref trainers)
 * ═══════════════════════════════════════════════════════════════════════ */
static inline void ki_shuffle(int *indices, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int t = indices[i]; indices[i] = indices[j]; indices[j] = t;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * LR CONVERSION — uint32↔float (for old Adam trainers)
 * ═══════════════════════════════════════════════════════════════════════
 * Old ki_Args version had uint32 LR fields. Die Konverter
 * are for compatibility mit mlp-flt32-w1-adam-trn etc.
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
    float tp  = (train_n > 0) ? (float)train_ok * 100.0f / (float)train_n : 0.0f;
    float ep  = (eval_n  > 0) ? (float)eval_ok  * 100.0f / (float)eval_n  : 0.0f;
    printf("\n============================================================\n");
    printf("REPORT train=%.1f%% (%d) eval=%.1f%% (%d)"
           " err=%d lr=%.4f time=%dms threads=%d\n",
           tp, train_n, ep, eval_n,
           err, (double)lr, elapsed_ms, threadN);
     printf("============================================================\n");
}

/* ═══════════════════════════════════════════════════════════════════
 * CONFUSION MATRIX — [true][pred] Tabelle (K×K), generisch
 * ═══════════════════════════════════════════════════════════════════
 * Can be used by any trainer.
 * is_final: 1 = Endausgabe, 0 = per-epoch
 */
__attribute__((unused))
static void print_confusion_debug(const uint8_t *y_true, const uint8_t *y_pred,
                                   int N, int ep, int is_final) {
    if (N <= 0) return;
    int cm[KI_NCLASSES][KI_NCLASSES];
    memset(cm, 0, sizeof(cm));
    for (int s = 0; s < N; s++) {
        int t = (int)y_true[s];
        int p = (int)y_pred[s];
        if (t >= 0 && t < KI_NCLASSES && p >= 0 && p < KI_NCLASSES)
            cm[t][p]++;
    }

    /* Only classes with samples anzeigen (important with --filter) */
    int active_cols[KI_NCLASSES], n_active = 0;
    for (int k = 0; k < KI_NCLASSES; k++) {
        int has = 0;
        for (int r = 0; r < KI_NCLASSES && !has; r++)
            if (cm[r][k] > 0 || cm[k][r] > 0) has = 1;
        if (has) active_cols[n_active++] = k;
    }
    if (n_active == 0) return;

    printf("\n  ── Confusion Matrix %s ─────────────────────────────\n", is_final ? "(final)" : "(per epoch)");
    if (!is_final)
        printf("  Ep %d\n", ep + 1);
    else
        (void)ep;
    printf("  %-12s", "true \\ pred");
    for (int ai = 0; ai < n_active; ai++)
        printf("  %-7s", ki_class_names[active_cols[ai]]);
    printf("  %-7s\n", "err%");

    printf("  %-12s", "────────────");
    for (int ai = 0; ai < n_active; ai++)
        printf("  %-7s", "───────");
    printf("  %-7s\n", "───────");

    for (int ri = 0; ri < n_active; ri++) {
        int r = active_cols[ri];
        int row_tot = 0, row_err = 0;
        for (int ci = 0; ci < n_active; ci++) {
            int cc = active_cols[ci];
            row_tot += cm[r][cc];
            if (cc != r) row_err += cm[r][cc];
        }
        if (row_tot == 0) continue;  /* only show rows with samples */
        float err_pct = (float)row_err * 100.0f / (float)row_tot;
        printf("  %-12s", ki_class_names[r]);
        for (int ci = 0; ci < n_active; ci++) {
            int cc = active_cols[ci];
            float col_pct = (float)cm[r][cc] * 100.0f / (float)row_tot;
            printf("  %6.1f%%", (double)col_pct);
        }
        printf("  %6.1f%%\n", (double)err_pct);
    }
    printf("  ─────────────────────────────────────────────────────\n\n");
    fflush(stdout);
}

/* ═══════════════════════════════════════════════════════════════════════
 * FILTER DATASET — Restrict to specific classes only
 * ═══════════════════════════════════════════════════════════════════════
 * Called AFTER ki_dataset_read(). Compacts data->y[], data->X_raw[],
 * and data->X[] (if present) to contain only samples whose labels
 * are in aa.filter_mask. Updates data->num_images and clamps
 * aa.trainN/aa.evalN to the filtered count.
 */
__attribute__((unused))
static inline void ki_filter_dataset(ki_Dataset *data) {
    if (aa.filter_str[0] == '\0') return;  /* no filter → no-op */
    if (aa.dry_run) return;           /* dry-run: no pixel data loaded yet */
    if (!data->y || data->num_images <= 0) return;  /* safety: no labels loaded */

    /* ── Parse aa.filter_str into a bitmask ─────────────────────
     * Accepts both numeric indices (0,1,2) and class names
     * (airplan,cat,deer). Case-insensitive for names. */
    int mask = 0;
    char buf[128];
    strncpy(buf, aa.filter_str, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        while (*tok == ' ' || *tok == '\t') tok++;
        if (*tok == '\0') continue;

        /* Try numeric first (atoi with full-string validation) */
        char *end = NULL;
        long c = strtol(tok, &end, 10);
        if (end != tok && *end == '\0') {
            if (c < 0 || c >= KI_NCLASSES) {
                fprintf(stderr, "[ERROR] --filter: invalid class index %ld (valid: 0..%d)\n",
                        c, KI_NCLASSES - 1);
                exit(1);
            }
            mask |= (1 << (int)c);
            continue;
        }

        /* Try class name match (case-insensitive) */
        int found = 0;
        for (int k = 0; k < KI_NCLASSES; k++) {
            if (strcasecmp(tok, ki_class_names[k]) == 0) {
                mask |= (1 << k);
                found = 1;
                break;
            }
        }
        if (!found) {
            fprintf(stderr, "[ERROR] --filter: unknown class '%s'. Valid:", tok);
            for (int k = 0; k < KI_NCLASSES; k++)
                fprintf(stderr, " %s(%d)", ki_class_names[k], k);
            fprintf(stderr, "\n");
            exit(1);
        }
    }

    if (mask == 0) {
        fprintf(stderr, "[ERROR] --filter: at least one class required\n");
        exit(1);
    }
    aa.filter_mask = mask;  /* store computed mask for print_setup etc. */

    int total = data->num_images;
    int *idx = (int *)malloc((size_t)total * sizeof(int));
    if (!idx) { fprintf(stderr, "[FATAL] ki_filter_dataset OOM\n"); exit(1); }
    int nf = 0;
    for (int i = 0; i < total; i++) {
        int lbl = (int)data->y[i];
        if (lbl >= 0 && lbl < KI_NCLASSES && ((mask >> lbl) & 1))
            idx[nf++] = i;
    }

    if (nf == 0) {
        fprintf(stderr, "[ERROR] --filter: no samples found for classes in mask 0x%02X\n", mask);
        free(idx); exit(1);
    }

    /* Compact labels */
    uint8_t *new_y = (uint8_t *)malloc((size_t)nf);
    if (!new_y) { fprintf(stderr, "[FATAL] ki_filter_dataset OOM\n"); free(idx); exit(1); }
    for (int i = 0; i < nf; i++) new_y[i] = data->y[idx[i]];
    free(data->y);
    data->y = new_y;

    /* Compact X_raw (always present after ki_dataset_read) */
    size_t px = (size_t)data->pixels;
    uint8_t *new_Xr = (uint8_t *)malloc((size_t)nf * px);
    if (!new_Xr) { fprintf(stderr, "[FATAL] ki_filter_dataset OOM\n"); free(idx); exit(1); }
    for (int i = 0; i < nf; i++)
        memcpy(new_Xr + (size_t)i * px, data->X_raw + (size_t)idx[i] * px, px);
    free(data->X_raw);
    data->X_raw = new_Xr;

    /* Compact X (float, only present for some trainers like Adam) */
    if (data->X) {
        float *new_Xf = (float *)malloc((size_t)nf * px * sizeof(float));
        if (!new_Xf) { fprintf(stderr, "[FATAL] ki_filter_dataset OOM\n"); free(idx); exit(1); }
        for (int i = 0; i < nf; i++)
            memcpy(new_Xf + (size_t)i * px, data->X + (size_t)idx[i] * px, px * sizeof(float));
        free(data->X);
        data->X = new_Xf;
    }

    data->num_images = nf;
    free(idx);

    /* ── Preserve original eval ratio when clamping ─────────────
     * If original split was 50000/10000 (17% eval), after filter
     * with 5923 samples, eval should be 5923×10000/60000 = 987
     * (NOT 5923-50000 = 0). */
    int total_desired = aa.trainN + aa.evalN;
    if (total_desired > nf) {
        if (total_desired <= 0) {
            aa.trainN = nf;
            aa.evalN = 0;
        } else {
            int new_eval = (int)((long long)aa.evalN * (long long)nf / (long long)total_desired);
            if (new_eval > aa.evalN) new_eval = aa.evalN;  /* never exceed original evalN */
            if (new_eval < 1 && nf >= 10) new_eval = 1;    /* at least 1 eval if enough samples */
            int new_train = nf - new_eval;
            if (new_train < 1) { new_train = 1; new_eval = nf - 1; }
            if (new_eval < 0) new_eval = 0;
            aa.trainN = new_train;
            aa.evalN = new_eval;
        }
    }

    /* Print class names from filter mask */
    printf("  [FILTER] Classes:");
    for (int k = 0; k < KI_NCLASSES; k++) {
        if ((mask >> k) & 1)
            printf(" %s", ki_class_names[k]);
    }
    printf("  → %d samples  split=%d/%d\n", nf, aa.trainN, aa.evalN);
    fflush(stdout);
}

#endif /* KI_COMMON_H */
