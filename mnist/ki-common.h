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

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

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
 * libtprint embedded (2026-08-17) — print_confusion_debug() below uses
 * TPrint, so the tprint implementation is compiled into EVERY translation
 * unit that includes ki-common.h. All tprint.c functions are static, so
 * each TU gets its own copy — no linker symbol conflicts, no per-Makefile
 * $(TPRINT_SRC) needed. Requires glib-2.0 (pkg-config --cflags/lib).
 * ═══════════════════════════════════════════════════════════════════════ */
#include "../lib/tprint.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "../lib/tprint.c"
#pragma GCC diagnostic pop

#define printC(_nme) printf("%s[%d] : " #_nme "=%s\n",__func__,__LINE__,_nme)

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
    STEP_POW_EVAL   = 4,  /* pow using EVAL error: step = step_init × (eval_err/total)^power */
};

/* ── Enum → lesbarer String ─────────────────────────────────── */
__attribute__((unused))
static const char *step_mode_name[] = {
    [STEP_POW]      = "pow",
    [STEP_COS_TIME] = "cos-time",
    [STEP_COS_ERR]  = "cos-err",
    [STEP_CONST]    = "const",
    [STEP_POW_EVAL] = "pow-eval",
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
__attribute__((unused))
static const char *target_init_str(void);

/* ═══════════════════════════════════════════════════════════════════════
 * ENCODING — Pixel-zu-Thermometer-Transformationen
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Each has its own Encoding-Funktion haben.
 * Controlled via --encoding r:lin8,g:up,b:down (per Block) oder
 * --encoding lin8 (default for all active blocks).
 */

/* ═══════════════════════════════════════════════════════════════════════
 * ENCODING — thermometer bitmasks for binary input
 * ═══════════════════════════════════════════════════════════════════════
 * Encoding-Enum, Parser, Apply, LUT und Farbdefinitionen
 * are in the shared header ausgelagert: */
#include "../lib/ki-encoding.h"

/* ── Parser: string → bit position (for --channel) ────────────
 * Returns bit index or -1 if not a color name.
 * Handles: mnist,r,g,b,y,601,lum,l,rg,by,yl,709,auge,packed,full
 */
static inline int ki_color_parse(const char *tok) {
    for (int i = 0; i < (int)(sizeof(ki_color_table)/sizeof(ki_color_table[0])); i++)
        if (strcasecmp(tok, ki_color_table[i].name) == 0)
            return ki_color_table[i].id;
    return -1;  /* no color name — aliases are expanded beforehand */
}

/* ═══════════════════════════════════════════════════════════════════════
 * MODE_FLT32 / MODE_FLT64 / MODE_INT32 — computation-mode switches
 * ═══════════════════════════════════════════════════════════════════════
 * Define -DMODE_FLT32 to enable IEEE 754 float counters/scores/types.
 * Define -DMODE_FLT64 to enable IEEE 754 DOUBLE counters/scores — full
 * precision, no fixed-point quantization (2026-08-16: flt64 = the
 * quantization-free control for the 92%-wall investigation; each rounding
 * loses information, float64 removes it).
 * Derived defines (COUNTER_TYPE, COUNTER_TYPE_IS_FLOAT, SCORE_TYPE) are
 * all set centrally from these switches.
 *
 * Default (no mode switch): float counters, double scores.
 *
 * NOTE: Individual -D overrides (COUNTER_TYPE, SCORE_TYPE) still work.
 */
#ifdef MODE_FLT32
/* Legacy alias — float is now the default, this is a no-op. */
#endif

#ifdef MODE_FLT64
#  ifndef COUNTER_TYPE
#    define COUNTER_TYPE double
#  endif
#  ifndef COUNTER_TYPE_IS_FLOAT
#    define COUNTER_TYPE_IS_FLOAT 1
#  endif
#  ifndef SCORE_TYPE
#    define SCORE_TYPE double
#  endif
#endif

#ifdef MODE_INT32
#  ifndef COUNTER_TYPE
#    define COUNTER_TYPE int32_t
#  endif
#  ifndef COUNTER_TYPE_IS_FLOAT
#    define COUNTER_TYPE_IS_FLOAT 0
#  endif
#  ifndef SCORE_TYPE
#    define SCORE_TYPE int64_t
#  endif
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * ki-local.h — dataset-specific constants + data loader
 * (MNIST in mnist-1/, CIFAR-10 in cifar-1/)
 * ═══════════════════════════════════════════════════════════════════════ */
#include "ki-local.h"

/* ── Bit-width dependent types & constants (defined after ki-local.h sets KI_BIT_WIDTH) ── */
#if KI_BIT_WIDTH == 4
#  define KI_PX_PER_CONT 8
#  define KI_BIT_POS     4
#  define KI_PIXEL_GROUPS 8
#  define PIXEL_TYPE     uint8_t
#elif KI_BIT_WIDTH == 8
#  define KI_PX_PER_CONT 4
#  define KI_BIT_POS     8
#  define KI_PIXEL_GROUPS 4
#  define PIXEL_TYPE     uint8_t
#elif KI_BIT_WIDTH == 12
#  define KI_PX_PER_CONT 2
#  define KI_BIT_POS     12
#  define KI_PIXEL_GROUPS 2
#  define PIXEL_TYPE     uint16_t
#elif KI_BIT_WIDTH == 16
#  define KI_PX_PER_CONT 2
#  define KI_BIT_POS     16
#  define KI_PIXEL_GROUPS 2
#  define PIXEL_TYPE     uint16_t
#elif KI_BIT_WIDTH == 32
#  define KI_PX_PER_CONT 1
#  define KI_BIT_POS     32
#  define KI_PIXEL_GROUPS 1
#  define PIXEL_TYPE     uint32_t
#else
#  error "KI_BIT_WIDTH must be 4, 8, 12, 16, or 32"
#endif

/* ── Encoding alias lookup (dataset-specific, defined in ki-local.h) ──
 * Each dataset provides its own ki_encoding_alias_lookup() via KI_COMMON_ALIAS_LOOKUP
 * guard.  Fallback returns NULL (no aliases). */
#ifndef KI_COMMON_ALIAS_LOOKUP
static inline const char *ki_encoding_alias_lookup(const char *name) {
    return ki_encoding_alias_expand(name);
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

 #define KI_ENC_MAX 512
 #define KI_MEMBER_MAX 32768    /* max member specs (was KI_ENC_MAX*8=4096, now 64×) */
 #define KI_DEFAULT_TARGET_NORM 0  /* --optional target-norm: default ON */

 /* ── Target initialisation modes (--target-init) ──────────── */
 enum ki_TargetInit {
     KI_TARGET_COUNT   = 0,  /* counting (default, proven) */
     KI_TARGET_RANDOM  = 1,  /* pure random */
     KI_TARGET_INVERSE = 2,  /* inverse count: +1 for all classes except true */
     KI_TARGET_UNIFORM = 3,  /* all targets = 1 (constant, class prior only) */
     KI_TARGET_PRIOR   = 4,  /* per-class constant = class count (no per-neuron var) */
     KI_TARGET_LAPLACE = 5,  /* count +1 per entry (additive smoothing, clamped) */
     KI_TARGET_DAMPEN  = 6,  /* count >> 1 (shape preserved, amplitude halved) */
 };

typedef struct {
    int8_t type;   /* KI_ENC_RAW..KI_ENC_SIG */
    int8_t width;  /* 8, 16, 32 */
    int8_t color;  /* COLOR_BIT or -1 (default/all) */
} ki_EncSlot;

/* ── Explicit member spec (from --member-file, --member, or product gen) ─── */
typedef struct {
    int xform_id;    /* KI_XFORM_ID, KI_XFORM_HFLIP, ... */
    int color;       /* COLOR_R, COLOR_G, COLOR_B, COLOR_LBP_RG, ... */
    int enc_type;    /* KI_ENC_RAW, KI_ENC_EXP, ... */
    int enc_width;   /* 8, 16, 32 */
    int seed;        /* random seed for W0 (0 = default), from --member-out */
    int source_idx;  /* member index within the seed's archive */
    int hn_idx;      /* splitHN index (0 .. splitHN-1), for slice offset */
} ki_MemberSpec;

typedef struct {
    int    hidden;          			/* Hidden neurons (--hiddenN, default: 64) */
    int    epochs;          			/* Iterations (--epochsN, default: 1) */
    /* cfg_epochs (FIX 2026-08-10): the CONFIGURED epoch count for DISPLAY.
     * --dry-run forces aa.epochs = 0 (no training), which made the Args:/
     * RESULT header show Ep=0 even when --epochsN 10 was given — the user
     * could not verify the intended config in dry-run output. cfg_epochs
     * keeps the user's --epochsN value untouched for the display while
     * aa.epochs stays 0 for the execution. */
    int    cfg_epochs;         		/* configured epochs for display (dry-run safe) */
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
    char   export_neurons[256]; 		/* --export-neurons FILE: save gb_buf+Target+Offset for Adam-on-neurons */
    int    export_gb;           		/* --export-gb: persist gb_buf to data/gb/ cache */
    int    import_gb;           		/* --import-gb: load gb_buf from data/gb/ cache on startup */
    int    sweep;              			/* --sweep: member sweep mode (skip if .ens exists) */
    int    force;              			/* --force: rebuild even if .ens exists (with --sweep) */
    char   member_file[1024];  		/* --member-file PATH: explicit member list (bypasses --xform/--encoding) */
    char   member_str[4096];  		/* --member CHAN:ENC:XF,... explicit member list (inline) */
    ki_MemberSpec member_spec[KI_MEMBER_MAX];  /* parsed member specs from file */
    int    member_spec_count;
    float  lr;              			/* Step size (--lr, default: 0.05) */
    float  lr_min;          			/* Min LR fraction (--lr-min, default: 0.1) */
    float  eff_lambda;      			/* --eff-lambda F: member-count penalty for the eff= score
                                           (eff = eval - lambda*(members-1), default 0.02) */
    int    lr_step;         			/* round(aa.lr * (1<<OT_PRECISION)) */
    int    threadN;         			/* OpenMP threads (--threadN, default: 8) */
    int    debug_h0;        			/* --debug-h0: per-neuron debug */
    int    shuffle;         			/* --shuffle: randomize train/eval split */
    int    warmup_epochs;   			/* --warmup N: linear warmup epochs (default: 2) */
    int    step_mode;       			/* enum step_mode: Algorithmus (siehe oben) */
    int    stepN;           			/* --step-const N: const step value (0=use lr, default: 0) */
    float  step_power;      			/* --step-power F: exponent for pow/cos (default: 0.7) */
    float  gap_k;           			/* --gap-k K: exp(-K×gap) step damping when train/eval gap widens (default: 0.0=off) */
    int    err_rollback;    			/* --err-rollback: rollback targets when err increases (default: 0) */
    int    maj_mode;         			/* --maj 3|true: majority mode (3=tree, true=exact per-bit) */
    int    maj1_thresh;      			/* --maj1-thresh N: -2=auto(n*135/256), -1=n/2, >=0 exact (default: -2) */
    int    maj_step;          			/* --maj-step N: pixel step between majority triples (default: KI_PX_PER_CONT) */
    int    debug_maj;         			/* --debug-maj auto|container|pixel: force majority path (default: auto) */
    int    rows_mode;         			/* --rows-mode flat|rows: 0=flat, 1=per-row members */
    int    no_precompute;   			/* --no-precompute: skip h0/gb caches, compute on-the-fly per batch (default: 0) */
    int    ensembleN;       			/* --ensembleN N: independent W0 copies (default: 1) */
    int    splitVN;        			/* --splitVN N: vertical H split (default: 1) */
    int    splitHN;  			        /* --splitHN N: horizontal NC split (default: 1) */
    int    channel;        			/* --channel bitmask of selected blocks */
    int    channel_explicit; 			/* 1=--channel was set explicitly */
    int    packedB;    			        /* 1=4px/cont (256/blk), 0=1px/cont (1024/blk) */
    int    debug_flat;      			/* 1=all selected blocks in one flat array, 0=separate members */
    int    debug_binarize;  			/* 1=threshold block values at 128 → 0x00/0xFF per pixel */
    int    hebbian_pct;     			/* --hebbian-pct: flip threshold % (reference, default 50) */
    int8_t enc[COLOR_NB];         		/* --encoding: per-block encoding type (-1=not set, derived from enc_array) */
    int8_t enc_width[COLOR_NB];  		/* --encoding: per-block width (derived from enc_array) */
    int8_t enc_default_type;      		/* --encoding: fallback type for blocks without specific setting */
    int8_t enc_default_width;     		/* --encoding: fallback width (8, 16, 32) */
    int    enc_size;             		/* --encoding-sizeN: global encoding bit-width (8,16,24,32) */
    ki_EncSlot enc_array[KI_ENC_MAX];  		/* Alle aktiven Encodings als (type,width)-Paare */
    int         enc_count;             		/* number of Eintraege in enc_array */
    int         member_threshold;    		/* --member-threshold: ignore members below N%% training acc */
    int    opt_target_norm;    			/* --optional target-norm: vote normalisierung aktivieren */
    char   seed_file[256]; 			/* --seed-file PATH: true random source */
    char   importD[512];    			/* --import DIR: load model for inference */
    int    seed_splitmix;  			/* --seed-splitmix: ignore seed_file, use splitmix64 PRNG */
    int    multi_correct;  			/* --multi-correct: punish all over true_k (default: 1) */
    int    target_init_mode; 			/* enum ki_TargetInit (default: KI_TARGET_COUNT) */
    int    ensemble_seed;    			/* ENS_SEED_ONCE|CONST|INCR (default: CONST) */
    int    debug_class_voting; 			/* --debug-class-voting: Member × Class accuracy (end only) */
    int    debug_class_voting_all; 		/* --debug-class-voting-all: every epoch */
    int    member_target_mask; 			/* from member-file "# META: ... TARGET=3,7":
                                           bitmask of the specialist's target class(es)
                                           (0 = no target info). Drives the target-column
                                           marker in --debug-class-voting (2026-08-12). */
    int    debug_confusion;    			/* --debug-confusion-matrix: confusion matrix (end only) */
    int    debug_confusion_all; 		/* --debug-confusion-matrix-all: every epoch */
    int    debug_member;        		/* --debug-member: verbose member-by-member output (seq only) */
    int    debug_member_stats;  		/* --debug-member-stats: per-member quality table at end */
    int    debug_cache;         		/* --debug-cache: log EVERY cache create/use (xform cache, cex *.pre, gb cache) */
    int    no_ens_cache;        		/* --no-ens-cache: skip the cex_*.pre FILE cache —
                                           real-live: compute inputs from raw data every run
                                           (the in-memory xform cache stays — it builds from
                                           the run's data, 2026-08-06) */
    int    xform_cache_level;   		/* --xform-cache-level 1|2: 1=transform only (level 1), 2=+block cache (level 2, default) */
    char   member_scores_path[512]; 	/* --debug-member-stats PATH: scores file (default: member-scores.bin) */
    char   filter_str[128];    			/* --filter "0,1,airplan,cat": raw string */
    int    filter_mask;        			/* computed bitmask from filter_str (0 = no filter) */
    uint64_t xforms;             		/* bitmask of active transforms (--xform, default: 1<<KI_XFORM_ID, 64-bit for 34+ xforms) */
#define KI_XFORM_LIST_MAX 512           /* xform_list[] holds base IDs AND pipe IDs
                                           (>= KI_XFORM_COUNT); 128 overflowed with
                                           pipeline-heavy corpora (2026-08-09) */
    int    xform_list[KI_XFORM_LIST_MAX];/* xform IDs as entered (preserves duplicates) */
    int    xform_list_count;            /* number of entries in xform_list */
} ki_Args;
 
/* ── Majority mode: 1=container flat, 1r=container row, 1p=pixel flat, 1rp=pixel+row, 3=tree (default), 7=7-tree ── */
enum ki_MajMode {
    KI_MAJ_1   = 0,  /* majority_tree1() — container-level flat */
    KI_MAJ_3   = 1,  /* majority_tree3() — 3-group tree approximation (default) */
    KI_MAJ_7   = 2,  /* majority_tree7() — 7-group tree approximation */
    KI_MAJ_1R  = 3,  /* majority_tree1_rowwise() — container-level row-wise */
    KI_MAJ_1P  = 4,  /* majority_tree1_pixel() — pixel-accurate flat */
    KI_MAJ_1RP = 5,  /* pixel-accurate row-wise: per-row pixel then cross-row */
};
/* ── Global args (defined in each main .c file) ────────────── */
#ifndef KI_ARGS_EXTERN
#define KI_ARGS_EXTERN extern
#endif
KI_ARGS_EXTERN ki_Args aa;

/* ── Cache debug logging gate ──────────────────────────────────
 * --debug-cache (2026-08-05) logs EVERY cache event: the in-memory
 * xform cache (ki-load.h), the cex_*.pre file cache (disk load/save) and
 * the gb cache (data/gb). Replaced the old --debug-gb flag (removed
 * 2026-08-05). */
static inline int ki_debug_cache(void) { return aa.debug_cache; }

/* ── Unified encoding display (reads aa.enc_array directly) ───────── */
static inline void ki_print_encodings(void) {
    printf("  Structure: ");
    int n = aa.enc_count;
    if (n <= 0) { printf("(none)\n"); return; }
    for (int i = 0; i < n && i < KI_ENC_MAX; i++) {
        if (i > 0) printf(", ");
        printf("M%d(", i);
        if (aa.enc_array[i].color >= 0)
            printf("%s", ki_color_name(aa.enc_array[i].color));
        printf("=%s%d)", ki_enc_name_short(aa.enc_array[i].type),
               (int)aa.enc_array[i].width);
    }
    if (aa.ensembleN > 1) printf("  × EN=%d", aa.ensembleN);
    printf("\n");
}

/* ── Ensemble seeding strategy ──────────────────────────────── */
typedef enum {
    ENS_SEED_CONST = 0,  /* default: one W0 shared across ALL members */
    ENS_SEED_ONCE  = 1,  /* once: one seed, sequential fill (legacy) */
    ENS_SEED_INCR  = 2,  /* incr: each member gets seed + m */
} ki_EnsembleSeed;

static inline const char *ensemble_seed_str() {
    switch (aa.ensemble_seed) {
        case ENS_SEED_CONST: return "const";
        case ENS_SEED_ONCE:  return "once";
        case ENS_SEED_INCR:  return "incr";
        default:             return "?";
    }
}

/* ── Rebuild enc_array from member_spec (single source of truth) ─── *
 * Called after --member-file / --member / --encoding parsing so that
 * load_input() and all enc_array consumers see the correct encoding set.
 * enc_array is only a DERIVED cache of member_spec — never written directly. */
static inline void rebuild_enc_array(void) {
    aa.enc_count = 0;
    for (int i = 0; i < aa.member_spec_count && aa.enc_count < KI_ENC_MAX; i++) {
        ki_MemberSpec *sp = &aa.member_spec[i];
        int col = sp->color;
        /* MNIST/Fashion (KI_COLORS<=1): the RGB-derived channels do not
         * exist — fold them back to COLOR_MNIST. The grayscale structure
         * channels (edge/bin/lbp/dog/var/dir/range, COLOR_EDGE..RANGE)
         * are KEPT so --channel mnist,lbp works (plan-2026-08-13). */
        if (KI_COLORS <= 1 && col >= 0 && ki_color_is_cifar_rgb(col))
            col = COLOR_MNIST;
        int dup = 0;
        for (int j = 0; j < aa.enc_count && !dup; j++)
            if (aa.enc_array[j].type  == (int8_t)sp->enc_type &&
                aa.enc_array[j].width == (int8_t)sp->enc_width &&
                aa.enc_array[j].color == (int8_t)col) dup = 1;
        if (!dup) {
            aa.enc_array[aa.enc_count].type  = (int8_t)sp->enc_type;
            aa.enc_array[aa.enc_count].width = (int8_t)sp->enc_width;
            aa.enc_array[aa.enc_count].color = (int8_t)col;
            aa.enc_count++;
        }
    }
    /* ── STABLE ENCODING ORDER (bug 2026-08-02) ──────────────
     * enc_array order defines multi_enc_blk_off[] → each member's container
     * offset (slc_off) in the shared input buffer. Without sorting, the order
     * depends on the member-spec ARRIVAL order (--member-file order vs
     * --xform/--encoding product order) — the SAME member spec then points to
     * DIFFERENT containers depending on which other members are loaded, i.e.
     * a member is not autonomous: its score changed 95.77% (.ens from sweep)
     * vs 95.3% (retrain from --member-file) for identical specs.
     * Sort deterministically by (type-name, width, color) so the buffer layout
     * is independent of the member list order. */
    if (aa.enc_count > 1) {
        /* insertion sort (stable, small N) by encoding name then width */
        for (int i = 1; i < aa.enc_count; i++) {
            ki_EncSlot key = aa.enc_array[i];
            int j = i - 1;
            while (j >= 0) {
                const char *a = ki_enc_name_short((int)aa.enc_array[j].type);
                const char *b = ki_enc_name_short((int)key.type);
                int c = strcmp(a, b);
                if (c > 0 || (c == 0 && aa.enc_array[j].width > key.width)) {
                    aa.enc_array[j + 1] = aa.enc_array[j];
                    j--;
                } else break;
            }
            aa.enc_array[j + 1] = key;
        }
    }
    /* Fallback: kein member_spec → ein Default-Encoding */
    if (aa.enc_count == 0 && aa.member_spec_count == 0) {
        int def_enc = aa.debug_binarize ? KI_ENC_LIN7 : KI_ENC_RAW;
        aa.enc_array[0].type  = (int8_t)def_enc;
        aa.enc_array[0].width = (int8_t)KI_ENC_WIDTH_DEFAULT;
        aa.enc_array[0].color = -1;
        aa.enc_count = 1;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * COMPLETION TABLE — for --completion flag (bash auto-completion)
 * ═══════════════════════════════════════════════════════════════════════
 * type: "none", "file", "dir", "num", "float", "token"
 * values: space-separated tokens for "token" type, NULL otherwise.
 * Sentinelle: {NULL, NULL, NULL} */
struct _comp_entry { const char *flag; const char *type; const char *values; };
static const struct _comp_entry _comp_table[] = {
    {"--hiddenN",                       "num",   NULL},
    {"--epochsN",                       "num",   NULL},
    {"--ensembleN",                     "num",   NULL},
    {"--splitVN",                       "token", "1 2 3 4 8 16 32"},
    {"--splitHN",                       "num",   NULL},
    {"--batchN",                        "num",   NULL},
    {"--trainN",                        "num",   NULL},
    {"--evalN",                         "num",   NULL},
    {"--threadN",                       "num",   NULL},
    {"--lr",                            "float", NULL},
    {"--lr-min",                        "float", NULL},
    {"--step-err",                      "token", "cos-time cos-err pow pow-eval const"},
    {"--step-const",                    "num",   NULL},
    {"--step-power",                    "float", NULL},
    {"--gap-k",                         "float", NULL},
    {"--member-threshold",              "num",   NULL},
    {"--err-rollback",                  "none",  NULL},
    {"--maj",                           "token", "1 1r 1p 1rp 3 7"},
    {"--maj1-thresh",                   "num",   NULL},
    {"--maj-step",                      "num",   NULL},
    {"--rows-mode",                     "token", "flat rows"},
    {"--no-precompute",                 "none",  NULL},
    {"--warmup",                        "num",   NULL},
    {"--seed",                          "num",   NULL},
    {"--seed-file",                     "file",  NULL},
    {"--seed-splitmix",                 "none",  NULL},
    {"--seed-member",                   "token", "once const incr"},
    {"--channel",                      "token", "@color"},
    {"--encoding",                      "token", "@enc"},
    {"--encoding-sizeN",                "token", "0 8 12 16 24 32"},
    {"--export-merge-scores",           "dir",   NULL},
    {"--export-scores",                 "file",  NULL},
    {"--export-neurons",                "file",  NULL},
    {"--export",                        "dir",   NULL},
    {"--export-gb",                     "none",  NULL},
    {"--import-gb",                     "none",  NULL},
    {"--member-file",                   "file",  NULL},
    {"--member",                        "token", NULL},
    {"--import",                        "dir",   NULL},
    {"--predictions",                   "file",  NULL},
    {"--no-multi-correct",              "none",  NULL},
    {"--multi-correct",                 "none",  NULL},
    {"--optional",                      "token", "target-norm"},
    {"--target-init",         "token", "count random inverse uniform prior laplace dampen"},
    {"--dry-run",                       "none",  NULL},
    {"--quick",                         "none",  NULL},
    {"--qq",                            "none",  NULL},
    {"--debug",                         "none",  NULL},
    {"--debug-maj",                     "token", "auto container pixel"},
    {"--debug-h0",                      "none",  NULL},
    {"--debug-class-voting",            "none",  NULL},
    {"--debug-class-voting-all",        "none",  NULL},
    {"--debug-confusion-matrix",        "none",  NULL},
    {"--debug-confusion-matrix-all",        "none",  NULL},
    {"--debug-epoch",                       "none",  NULL},
    {"--debug-member",                      "none",  NULL},
    {"--debug-member-stats",                "file",  NULL},
    {"--debug-cache",                       "none",  NULL},
    {"--no-ens-cache",                      "none",  NULL},
    {"--xform-cache-level",                 "token", "1 2"},
    {"--sweep",                             "none",  NULL},
    {"--force",                             "none",  NULL},
    {"--xform",              "token", "@xform"},
    {"--filter",                        "token", NULL},
    {"--shuffle",                       "none",  NULL},
    {"--help",                          "none",  NULL},
    {"--help-channels",                 "none",  NULL},
    {"--help-encoding",                 "none",  NULL},
    {"--help-filter",                   "none",  NULL},
    {"--help-target-init",              "none",  NULL},
    {"--help-xform",                    "none",  NULL},
    {NULL, NULL, NULL}
};

/* ── Dynamic token lists for --completion ─────────────────────────
 * Values are built from the *_table structures (single source of
 * truth) instead of hand-maintained strings that drift out of sync.
 * Markers used in _comp_table values:
 *   "@color" → control tokens + ki_color_table + ki_color_alias_table
 *   "@enc"   → control tokens + ki_enc_table + ki_enc_alias_table
 *   "@xform" → control tokens + ki_xform_table + ki_xform_alias_table
 * Duplicates (case-insensitive, e.g. "y" vs "Y") are dropped. */
static int comp_name_exists(const char *buf, const char *name) {
    size_t nl = strlen(name);
    for (const char *p = buf; *p; ) {
        const char *sp = p;
        while (*sp && *sp != ' ') sp++;
        if ((size_t)(sp - p) == nl && strncasecmp(p, name, nl) == 0) return 1;
        p = (*sp) ? sp + 1 : sp;
    }
    return 0;
}

/* Append name lower-cased (table names like "Y"/"AL" → "y"/"al") so the
 * completion list matches what users type (bash matches case-sensitively). */
static void comp_add_lower(char *buf, size_t *used, size_t sz, const char *name) {
    if (comp_name_exists(buf, name)) return;  /* case-insensitive dedup */
    size_t l = strlen(name);
    if (*used + l + 2 >= sz) return;
    if (*used) buf[(*used)++] = ' ';
    for (size_t i = 0; i < l; i++)
        buf[(*used)++] = (char)tolower((unsigned char)name[i]);
    buf[*used] = '\0';
}

static void comp_tokens_build(const char *marker, char *buf, size_t sz) {
    buf[0] = '\0';
    size_t used = 0;
#define COMP_ADD(_s) comp_add_lower(buf, &used, sz, (_s))
    if (strcmp(marker, "@color") == 0) {
        COMP_ADD("all"); COMP_ADD("packed"); COMP_ADD("full"); COMP_ADD("flat");
        for (size_t _i = 0; _i < sizeof(ki_color_table)/sizeof(ki_color_table[0]); _i++)
            COMP_ADD(ki_color_table[_i].name);
        for (size_t _i = 0; _i < sizeof(ki_color_alias_table)/sizeof(ki_color_alias_table[0]); _i++)
            COMP_ADD(ki_color_alias_table[_i].name);
    } else if (strcmp(marker, "@enc") == 0) {
        COMP_ADD("all"); COMP_ADD("latest");
        for (size_t _i = 0; _i < sizeof(ki_enc_table)/sizeof(ki_enc_table[0]); _i++)
            COMP_ADD(ki_enc_table[_i].name);
        for (size_t _i = 0; _i < sizeof(ki_enc_alias_table)/sizeof(ki_enc_alias_table[0]); _i++)
            COMP_ADD(ki_enc_alias_table[_i].name);
    } else if (strcmp(marker, "@xform") == 0) {
        COMP_ADD("all");
        for (size_t _i = 0; _i < sizeof(ki_xform_table)/sizeof(ki_xform_table[0]); _i++)
            COMP_ADD(ki_xform_table[_i].name);
        for (size_t _i = 0; _i < sizeof(ki_xform_alias_table)/sizeof(ki_xform_alias_table[0]); _i++)
            COMP_ADD(ki_xform_alias_table[_i].name);
    }
#undef COMP_ADD
}

/* ── Completion handler ───────────────────────────────────────────
 * --completion [--<flag>] — the decision which source answers lives
 * HERE, not in the arg parser:
 *   - flags with an "@" marker in _comp_table → token list built
 *     dynamically from the *_table / *_alias_table structures
 *     (--channel, --encoding, --xform; single source of truth)
 *   - all other flags → static values from _comp_table            */
static void ki_completion_print(const char *target) {
    const struct _comp_entry *e;
    for (e = _comp_table; e->flag; e++) {
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

static void ki_completion_dispatch(int argc, char **argv, int *pi) {
    /* argv[*pi] == "--completion"; an optional next arg is the target flag */
    if (*pi + 1 < argc && argv[*pi + 1][0] == '-' && argv[*pi + 1][1] == '-') {
        ki_completion_print(argv[++(*pi)]);
        exit(0);
    }
    /* --completion alone → all valid flag names */
    for (const struct _comp_entry *e = _comp_table; e->flag; e++)
        puts(e->flag);
    exit(0);
}

/* ── Xform list insertion (preserves duplicates) ───────────────── */
static inline void ki_xform_list_add(int xf) {
    if (aa.xform_list_count < KI_XFORM_LIST_MAX)
        aa.xform_list[aa.xform_list_count++] = xf;
}

/* ── Xform bitmask set — SAFE for pipe IDs >= 64 ──────────────────
 * aa.xforms is uint64_t, but pipe IDs run up to KI_XFORM_COUNT +
 * KI_XFORM_PIPE_MAX-1 (41 + 511 = 552) — `1ull << xf` with xf >= 64 is
 * undefined behavior. The bitmask is only used for the SETUP display
 * ("show_xform", "xform_str") and the empty-check; the actual xform
 * selection runs through aa.xform_list[]. So a pipe ID beyond bit 63 is
 * simply NOT set in the bitmask — the list still carries it. The
 * empty-check must then use xform_list_count, not the bitmask.
 * Fix 2026-08-09 (too many pipelines overflow, ki-common.h). */
static inline void ki_xform_bit_set(int xf) {
    if (xf >= 0 && xf < 64) aa.xforms |= (1ull << (unsigned)xf);
}

/* ═══════════════════════════════════════════════════════════════════════
 * XFORM PIPELINE CHAINING — "@" syntax: rot90@avg4 → apply rot90, then avg4
 * ═══════════════════════════════════════════════════════════════════════
 * Pipelines are stored as virtual xform IDs >= KI_XFORM_COUNT.
 * The bitmask aa.xforms uses bits up to KI_XFORM_COUNT + pipe_count - 1.
 * KI_XFORM_BUF_MAX: max slots for xform/pipe data buffers (must cover pipes).
 * DIMENSIONING (2026-08-09): 128 pipes overflowed on real sweep corpora —
 * the run-sweep "@"-cross-product (sweep@X = 16 base xforms × N suffixes)
 * produced 144 distinct pipelines in scores-H196-E10-M1-106-INT32, breaking
 * the merge with "[ERROR] too many pipelines". 512 covers 16 base × 32
 * suffixes; BUF_MAX and LIST_MAX grow in lockstep. */
#define KI_XFORM_PIPE_MAX_STEPS 8
#define KI_XFORM_PIPE_MAX 512
/* X_xform data-buffer slots must cover ALL xforms: table xforms (0..KI_XFORM_COUNT-1)
 * PLUS up to KI_XFORM_PIPE_MAX pipe IDs (KI_XFORM_COUNT .. KI_XFORM_COUNT+511).
 * 64 was too small — member files with many @-pipes (e.g. 129 distinct xforms in
 * a top-50% MNIST list) produced pipe IDs >= 64, making the lazy-load gate
 * `xf < KI_XFORM_BUF_MAX` skip loading while `X_xform[xf]` was read out of bounds
 * → garbage input_buf → SIGSEGV in h0_neuron (bug 2026-08-01). */
#define KI_XFORM_BUF_MAX 1024

static int _xf_pipe_steps[KI_XFORM_PIPE_MAX][KI_XFORM_PIPE_MAX_STEPS];
static int _xf_pipe_nsteps[KI_XFORM_PIPE_MAX];
static int _xf_pipe_count = 0;

/* Find existing pipe by its steps (dedup). Returns virtual xform ID or -1.
 * Serialized with the same critical section as ki_xform_pipe_create so a
 * reader never observes a half-written registry slot (FIX 2026-08-12). */
static inline int ki_xform_pipe_find(const int *steps, int n) {
    int found = -1;
    #pragma omp critical(xform_pipe_registry)
    {
        for (int i = 0; i < _xf_pipe_count; i++) {
            if (_xf_pipe_nsteps[i] != n) continue;
            int match = 1;
            for (int j = 0; j < n; j++)
                if (_xf_pipe_steps[i][j] != steps[j]) { match = 0; break; }
            if (match) { found = KI_XFORM_COUNT + i; break; }
        }
    }
    return found;
}

/* Register a pipeline from an array of xform IDs. Returns virtual xform ID.
 * THREAD-SAFE (FIX 2026-08-12): the PRF trainer exports one .ens per member
 * from parallel threads; ki_xform_parse_or_pipe() → ki_xform_pipe_create()
 * mutates the GLOBAL registry. Without serialization two threads could read
 * the same _xf_pipe_count and hand out the SAME virtual id for two different
 * pipes → wrong internal xform name in the exported .ens (filename ≠ meta).
 * See: bugs/bug-2026-08-12-trainer-pipe-registry-race.md. All trainers build
 * with -fopenmp, so omp critical is always available. */
static inline int ki_xform_pipe_create(const int *steps, int n) {
    int id;
    #pragma omp critical(xform_pipe_registry)
    {
        if (_xf_pipe_count >= KI_XFORM_PIPE_MAX || n < 2 || n > KI_XFORM_PIPE_MAX_STEPS) {
            id = -1;
        } else {
            id = KI_XFORM_COUNT + _xf_pipe_count;
            for (int i = 0; i < n; i++) _xf_pipe_steps[_xf_pipe_count][i] = steps[i];
            _xf_pipe_nsteps[_xf_pipe_count] = n;
            _xf_pipe_count++;
        }
    }
    return id;
}
static inline int  ki_xform_is_pipe(int xf) { return xf >= KI_XFORM_COUNT; }
static inline int  ki_xform_pipe_idx(int xf) { return xf - KI_XFORM_COUNT; }
static inline int  ki_xform_pipe_nsteps(int xf) {
    int idx = xf - KI_XFORM_COUNT;
    return (idx >= 0 && idx < _xf_pipe_count) ? _xf_pipe_nsteps[idx] : 0;
}
static inline const int *ki_xform_pipe_steps(int xf) {
    int idx = xf - KI_XFORM_COUNT;
    return (idx >= 0 && idx < _xf_pipe_count) ? _xf_pipe_steps[idx] : NULL;
}
/* Build display name like "rot90@avg4" into a static buffer.
 * THREAD-SAFE (FIX 2026-08-10): the buffer must be thread-local — the PRF
 * trainer exports .ens files from parallel member threads, and a shared
 * static buffer raced (ens-verify "field 3 mismatch": shuffle@avg4 was
 * overwritten by colswap-3-4@avg4 from another thread). Same pattern as
 * the thread-local t_match buffer in h0_neuron. */
static inline const char *ki_xform_pipe_name(int xf) {
    static __thread char _buf[64];
    int idx = xf - KI_XFORM_COUNT;
    if (idx < 0 || idx >= _xf_pipe_count) return "pipe?";
    int pos = 0;
    for (int i = 0; i < _xf_pipe_nsteps[idx] && pos < 60; i++) {
        if (i > 0) _buf[pos++] = '@';
        const char *n = ki_xform_name(_xf_pipe_steps[idx][i]);
        while (*n && pos < 60) _buf[pos++] = *n++;
    }
    _buf[pos] = '\0';
    return _buf;
}
static inline const char *ki_xform_str(int xf) {
  return ki_xform_is_pipe(xf) ? ki_xform_pipe_name(xf) : ki_xform_name(xf);
}

/* ── Iterative alias expansion (2026-08-12) ─────────────────────────
 * Fully expands an xform alias string into a comma-separated list of
 * PLAIN xform names, resolving nested aliases (e.g. "all" → "all-basic,
 * all-shift,..." → "id,hflip,...,sft-u1,..."). Copies into dst (size).
 * Non-alias input is copied verbatim. Bounded by 5 passes (an alias
 * cycle would loop forever; the table has no cycles). Used by the
 * --xform parser's cross-product so BOTH sides of "a@b" support aliases
 * at any nesting depth. Returns dst. */
static inline char *ki_xform_alias_expand_full(const char *in, char *dst, size_t size) {
    strncpy(dst, in, size - 1);
    dst[size - 1] = '\0';
    for (int _iter = 0; _iter < 5; _iter++) {
        /* Phase 1: full-string alias match */
        const char *_full = ki_xform_alias_expand(dst);
        if (_full) {
            strncpy(dst, _full, size - 1);
            dst[size - 1] = '\0';
            continue;
        }
        /* Phase 2: per-token expansion */
        char _tmp[4096], _new[4096] = "";
        strncpy(_tmp, dst, sizeof(_tmp) - 1);
        _tmp[sizeof(_tmp) - 1] = '\0';
        int _any = 0;
        char *_save2 = NULL;
        for (char *_t = strtok_r(_tmp, ",", &_save2); _t; _t = strtok_r(NULL, ",", &_save2)) {
            while (*_t == ' ' || *_t == '\t') _t++;
            const char *_pe = ki_xform_alias_expand(_t);
            if (_pe) {
                if (_new[0]) strncat(_new, ",", sizeof(_new) - 1);
                strncat(_new, _pe, sizeof(_new) - strlen(_new) - 1);
                _any = 1;
            } else {
                if (_new[0]) strncat(_new, ",", sizeof(_new) - 1);
                strncat(_new, _t, sizeof(_new) - strlen(_new) - 1);
            }
        }
        if (!_any) break;
        strncpy(dst, _new, size - 1);
        dst[size - 1] = '\0';
    }
    return dst;
}

/* ── Parse xform with optional @-pipeline chaining.
 * Returns xform_id (regular or pipe virtual ID), or -1 on error.
 * Does NOT modify aa.xforms or aa.xform_list — caller handles that.
 * INTENTIONAL (2026-08-15): KI_XFORM_ID steps are PRESERVED in pipelines
 * (rot0@id → "rot0@id", id@rot67 → "id@rot67"). The old behaviour skipped
 * them (id@avg4 = avg4), which silently erased the "@id" suffix from
 * member names — that made `--filter regex 'rot[0-9]+@id'` impossible and
 * hid the difference between rotXX (bare) and rotXX@id (explicit). The
 * name must equal what was entered: no representation mangling. */
static inline int ki_xform_parse_or_pipe(const char *xf_str) {
    if (!strchr(xf_str, '@')) {
        return ki_xform_parse(xf_str);
    }
    char _pipe_buf[128];
    strncpy(_pipe_buf, xf_str, sizeof(_pipe_buf) - 1);
    _pipe_buf[sizeof(_pipe_buf) - 1] = '\0';
    int _steps[KI_XFORM_PIPE_MAX_STEPS], _n = 0;
    char *_psave = NULL;
    for (char *_st = strtok_r(_pipe_buf, "@", &_psave);
         _st && _n < KI_XFORM_PIPE_MAX_STEPS;
         _st = strtok_r(NULL, "@", &_psave)) {
        while (*_st == ' ' || *_st == '\t') _st++;
        int _s = ki_xform_parse(_st);
        if (_s < 0) {
            fprintf(stderr, "[ERROR] unknown xform step '%s' in pipeline '%s'\n", _st, xf_str);
            return -1;
        }
        _steps[_n++] = _s;   /* id steps preserved (2026-08-15) */
    }
    if (_n < 1) return KI_XFORM_ID;
    if (_n == 1) return _steps[0];
    int pipe_id = ki_xform_pipe_find(_steps, _n);
    if (pipe_id >= 0) return pipe_id;
    pipe_id = ki_xform_pipe_create(_steps, _n);
    if (pipe_id < 0) {
        fprintf(stderr, "[ERROR] too many pipelines (max %d)\n", KI_XFORM_PIPE_MAX);
        return -1;
    }
    return pipe_id;
}

/* ── Core member spec add (parsed ints) — called by all three paths. */
static inline int ki_member_spec_add_raw(int col, int enc_type, int enc_width,
                                          int xform_id, int hn_idx) {
    int n = aa.member_spec_count;
    if (n >= (int)(sizeof(aa.member_spec) / sizeof(aa.member_spec[0]))) {
        fprintf(stderr, "[ERROR] Too many member specs (max %zu)\n",
                sizeof(aa.member_spec) / sizeof(aa.member_spec[0]));
        return -1;
    }
    aa.member_spec[n].xform_id  = xform_id;
    aa.member_spec[n].color     = col;
    aa.member_spec[n].enc_type  = enc_type;
    aa.member_spec[n].enc_width = enc_width;
    aa.member_spec[n].hn_idx    = hn_idx;
    aa.member_spec_count = n + 1;
    return 0;
}

/* ── Stable xform-major sort of member_spec[].
 * Groups all members of one xform (incl. @-pipes, which share a pipe_id)
 * together so the trainer can hold only ONE input buffer per xform group
 * and free it on group switch (see mlp-bin32-otto-trn-seq.c cur_xf_buf).
 * --member-file order is file order (unsorted); without sorting the
 * CIFAR sweep-style runs OOM (every lazy-loaded xform buffer stays
 * resident forever, 461 members × ~20 xforms × ~2.6 GB > 50 GB).
 * Insertion sort is stable: within one xform group the members are ordered
 * channel-major then encoding-major — sort key (xform_id, color, enc_type,
 * enc_width, hn_idx). Channel-major grouping was added 2026-08-05: the bare
 * --encoding expansion produced a mixed order (R:raw,R:lin,R:exp,G:raw,B:raw,
 * G:lin,B:lin,...) instead of xform → channel → encoding.
 * Called by ALL spec sources: ki_member_parse (--member),
 * ki_member_file_parse (--member-file) and
 * ki_member_spec_generate_from_product (--xform/--encoding product).
 * See: plans/plan-2026-07-31-sweep-xform-memory.md
 *      plans/plan-2026-08-05-xform-cache.md */
static inline int ki_member_spec_less(const ki_MemberSpec *a,
                                      const ki_MemberSpec *b) {
    if (a->xform_id != b->xform_id) return a->xform_id < b->xform_id;
    if (a->color    != b->color)    return a->color    < b->color;
    if (a->enc_type != b->enc_type) return a->enc_type < b->enc_type;
    if (a->enc_width != b->enc_width) return a->enc_width < b->enc_width;
    return a->hn_idx < b->hn_idx;
}
static inline void ki_member_spec_sort_xform(void) {
    int n = aa.member_spec_count;
    if (n < 2) return;
    for (int i = 1; i < n; i++) {
        ki_MemberSpec sp = aa.member_spec[i];
        int j = i - 1;
        while (j >= 0 && !ki_member_spec_less(&aa.member_spec[j], &sp)) {
            aa.member_spec[j + 1] = aa.member_spec[j];
            j--;
        }
        aa.member_spec[j + 1] = sp;
    }
}

/* ── Generate member specs from --xform × --encoding × --channel product.
 * Called AFTER ki_parse_args() when no --member/--member-file was given.
 * Populates aa.member_spec[] from aa.xform_list[], aa.enc_array[], aa.channel. */
static inline int ki_member_spec_generate_from_product(void) {
    if (aa.member_spec_count > 0) return 0;  /* already have explicit specs */
    if (aa.xform_list_count == 0) {
        /* Default: identity only */
        aa.xform_list[0] = KI_XFORM_ID;
        aa.xform_list_count = 1;
    }
    /* Determine active channels/encodings */
    int n_enc = aa.enc_count > 0 ? aa.enc_count : 1;
    if (aa.enc_count == 0) {
        /* No explicit encodings — use channel bitmask */
        n_enc = 0;
        for (int b = 0; b < COLOR_NB; b++)
            if (aa.channel & (1 << b)) n_enc++;
        if (n_enc < 1) n_enc = 1;
    }
    /* Iterate product: xform × encoding × splitHN */
    for (int xf_idx = 0; xf_idx < aa.xform_list_count; xf_idx++) {
        int xf_id = aa.xform_list[xf_idx];
        for (int vi = 0; vi < n_enc; vi++) {
            int col = 0, enc_type = KI_ENC_RAW, enc_width = KI_ENC_WIDTH_DEFAULT;
            if (aa.enc_count > 0 && vi < aa.enc_count) {
                col       = (int)aa.enc_array[vi].color;
                enc_type  = (int)aa.enc_array[vi].type;
                enc_width = (int)aa.enc_array[vi].width;
            } else if (aa.enc_count > 0) {
                col       = (int)aa.enc_array[0].color;
                enc_type  = (int)aa.enc_array[0].type;
                enc_width = (int)aa.enc_array[0].width;
            } else {
                /* Find channel from bitmask */
                int cnt = 0;
                for (int b = 0; b < COLOR_NB; b++) {
                    if (aa.channel & (1 << b)) {
                        if (cnt == vi) { col = b; break; }
                        cnt++;
                    }
                }
                if (cnt < vi) col = 0;
                enc_type  = (int)aa.enc_default_type;
                enc_width = (int)aa.enc_default_width;
            }
            for (int hn = 0; hn < aa.splitHN; hn++) {
                if (ki_member_spec_add_raw(col, enc_type, enc_width, xf_id, hn) < 0)
                    return -1;
            }
        }
    }
    printf("  [PRODUCT] %d members (%d xforms × %d enc × %d HN)\n",
           aa.member_spec_count,
           aa.xform_list_count, n_enc, aa.splitHN);
    ki_member_spec_sort_xform();
    return aa.member_spec_count;
}

/* ── String-based wrapper for --member and --member-file.
 * Parses channel/encoding/xform from strings, hn_idx = 0 (no splitHN). */
static inline int ki_member_spec_add(const char *ch_str,
                                      const char *enc_str,
                                      const char *xf_str) {
    int col = ki_color_parse(ch_str);
    if (col < 0) {
        fprintf(stderr, "[ERROR] --member/--member-file: unknown channel '%s'\n", ch_str);
        return -1;
    }
    int enc_w = 8;
    int enc = ki_enc_parse(enc_str, &enc_w);
    if (enc < 0) {
        fprintf(stderr, "[ERROR] --member/--member-file: unknown encoding '%s'\n", enc_str);
        return -1;
    }
    int xf_id = ki_xform_parse_or_pipe(xf_str);
    if (xf_id < 0) {
        fprintf(stderr, "[ERROR] --member/--member-file: unknown xform '%s'\n", xf_str);
        return -1;
    }
    return ki_member_spec_add_raw(col, enc, enc_w, xf_id, 0);
}

/* ── Parse --member inline: "XF:CHAN:ENC,XF:CHAN:ENC,..."
 * Each comma-separated token defines one explicit member.
 * Calls ki_member_spec_add() for each — same path as --member-file. */
static inline int ki_member_parse(void) {
    if (!aa.member_str[0]) return 0;
    char buf[4096];
    strncpy(buf, aa.member_str, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    int n = 0;
    char *save = NULL;
    for (char *tok = strtok_r(buf, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
        while (*tok == ' ' || *tok == '\t') tok++;
        if (*tok == '\0') continue;
        /* Format: XF:CHAN:ENC — split by ':' (xform first, matches workflow) */
        char *ch_start = strchr(tok, ':');
        if (!ch_start) {
            fprintf(stderr, "[ERROR] --member: expected 'XF:CHAN:ENC', got '%s'\n", tok);
            return -1;
        }
        *ch_start = '\0';
        char *xf_str   = tok;
        char *rest     = ch_start + 1;
        char *enc_str  = strchr(rest, ':');
        if (!enc_str) {
            fprintf(stderr, "[ERROR] --member: expected 'XF:CHAN:ENC', got '%s:%s'\n", xf_str, rest);
            return -1;
        }
        *enc_str = '\0';
        char *ch_str  = rest;
        char *enc_only = enc_str + 1;
        if (ki_member_spec_add(ch_str, enc_only, xf_str) < 0) return -1;
        n++;
    }
    if (n == 0) {
        fprintf(stderr, "[ERROR] --member: no valid members found\n");
        return -1;
    }
    ki_member_spec_sort_xform();  /* xform-major order (memory control, see above) */
    printf("  [MEMBER] %d explicit members\n", n);
    return n;
}

/* ── Parse member file: one member per line.
 * Format:  XF:CHAN:ENC   (same as --member, xform first).
 * Lines starting with # or ; are comments. Empty lines skipped.
 * Delegates to ki_member_spec_add (supports @-pipelines). */
static inline int ki_member_file_parse(void) {
    if (!aa.member_file[0]) return 0;
    FILE *f = fopen(aa.member_file, "r");
    if (!f) { fprintf(stderr, "[ERROR] Cannot open --member-file: %s\n", aa.member_file); return -1; }
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == ';' || *p == '\n' || *p == '\0') continue;
        { size_t l = strlen(p); while (l > 0 && (p[l-1] == '\n' || p[l-1] == '\r')) p[--l] = '\0'; }
        /* Format: XF:CHAN:ENC — two colons */
        char *first = strchr(p, ':');
        char *second = first ? strchr(first + 1, ':') : NULL;
        if (!first || !second) {
            fprintf(stderr, "[WARN] --member-file: expected 'XF:CHAN:ENC', got: %s\n", line);
            continue;
        }
        *first = '\0';   /* p = XF */
        *second = '\0';  /* first+1 = CHAN */
        /* Strip optional W0 marker suffix from encoding (e.g. "exp8  0xABCDEF12") */
        char *_enc_only = second + 1;
        {   char *_sp = strchr(_enc_only, ' ');
            if (_sp) *_sp = '\0';  /* space before W0 marker → truncate */
        }
        if (ki_member_spec_add(first + 1, _enc_only, p) < 0) { fclose(f); return -1; }
    }
    fclose(f);
    int n = aa.member_spec_count;
    if (n == 0) { fprintf(stderr, "[ERROR] --member-file: no valid members found in %s\n", aa.member_file); return -1; }
    ki_member_spec_sort_xform();  /* xform-major order (memory control, see above) */
    printf("  [MEMBER-FILE] %s  (%d members)\n", aa.member_file, n);
    return n;
}

/* ── Adopt ensemble DEFAULTS from a --member-file META header ──────────
 * Called BEFORE ki_parse_args so that values from the file (written by
 * merge-ensemble --member-out) become the DEFAULTS; explicit CLI options
 * still win because they are applied afterwards (feature 2026-08-10).
 * Reads only the `# META: H=.. EP=.. VN=.. HN=.. MAJ=.. MAJ1_THRESH=..`
 * line — the member spec lines are handled by ki_member_file_parse(). */
static inline void ki_member_file_apply_meta(void) {
    if (!aa.member_file[0]) return;
    FILE *f = fopen(aa.member_file, "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "# META:", 7) != 0) continue;
        int _H = 0, _EP = 0, _VN = 0, _HN = 0, _MAJT = -999;
        char _MAJ[8] = "";
        if (sscanf(line, "# META: H=%d EP=%d VN=%d HN=%d MAJ=%7s MAJ1_THRESH=%d",
                   &_H, &_EP, &_VN, &_HN, _MAJ, &_MAJT) < 4) break;
        if (_H > 0) aa.hidden = _H;
        if (_EP > 0) { aa.epochs = _EP; aa.cfg_epochs = _EP; }
        if (_VN > 0) aa.splitVN = _VN;
        if (_HN > 0) aa.splitHN = _HN;
        if (_MAJ[0] && strcmp(_MAJ, "-1") != 0) {
            if (strcmp(_MAJ, "1r") == 0)      aa.maj_mode = KI_MAJ_1R;
            else if (strcmp(_MAJ, "1p") == 0) aa.maj_mode = KI_MAJ_1P;
            else if (strcmp(_MAJ, "1rp") == 0)aa.maj_mode = KI_MAJ_1RP;
            else if (strcmp(_MAJ, "3") == 0)  aa.maj_mode = KI_MAJ_3;
            else if (strcmp(_MAJ, "7") == 0)  aa.maj_mode = KI_MAJ_7;
            else                               aa.maj_mode = KI_MAJ_1;
        }
        if (_MAJT >= -2) aa.maj1_thresh = _MAJT;
        /* TARGET=3,7 (specialist classes, written by merge-ensemble when
         * --target was given). Parsed separately from sscanf so the mask
         * survives even if MAJ/MAJ1_THRESH fields change. Feature
         * 2026-08-12: --debug-class-voting marks the target column(s). */
        char tgt_spec[64] = "";
        char *tg = strstr(line, "TARGET=");
        if (tg) {
            char *sp = strchr(tg, '\n');
            if (sp) *sp = '\0';
            snprintf(tgt_spec, sizeof(tgt_spec), "%.48s", tg + 7);
            aa.member_target_mask = 0;
            for (char *tok = strtok(tgt_spec, ","); tok;
                 tok = strtok(NULL, ",")) {
                int c = atoi(tok);
                if (c >= 0 && c < KI_NCLASSES)
                    aa.member_target_mask |= (1 << c);
            }
        }
        printf("  [MEMBER-FILE] meta defaults: H=%d EP=%d VN=%d HN=%d MAJ=%s MAJ1_THRESH=%d%s%s\n",
               _H, _EP, _VN, _HN, _MAJ, _MAJT,
               aa.member_target_mask ? "  TARGET=" : "",
               aa.member_target_mask ? tgt_spec : "");
        break;
    }
    fclose(f);
}

/* ── Apply xform (or pipe) to a raw image buffer ────────────────────
 * Applies xform_id to n_samples images (in → out, each img_sz bytes).
 * Handles regular xforms and @-chained pipes.
 * For pipes, allocates internal temp buffers (caller provides out only). */
static inline void ki_xform_apply_buf(uint8_t *restrict out,
                                       const uint8_t *restrict in,
                                       int w, int h, int ch,
                                       int n_samples, int xform_id,
                                       int img_sz) {
    if (!ki_xform_is_pipe(xform_id)) {
        for (int s = 0; s < n_samples; s++)
            ki_xform_raw(out + (size_t)s * (size_t)img_sz,
                         in  + (size_t)s * (size_t)img_sz,
                         w, h, ch, xform_id);
    } else {
        int n_steps = ki_xform_pipe_nsteps(xform_id);
        const int *steps = ki_xform_pipe_steps(xform_id);
        size_t total_sz = (size_t)n_samples * (size_t)img_sz;
        uint8_t *t0 = (uint8_t *)malloc(total_sz);
        uint8_t *t1 = (uint8_t *)malloc(total_sz);
        /* Step 0: in → t0 */
        for (int s = 0; s < n_samples; s++)
            ki_xform_raw(t0 + (size_t)s * (size_t)img_sz,
                         in  + (size_t)s * (size_t)img_sz,
                         w, h, ch, steps[0]);
        /* Steps 1..n-2: toggle t0↔t1 */
        int cur = 0;
        for (int i = 1; i < n_steps - 1; i++) {
            uint8_t *src = cur ? t1 : t0;
            uint8_t *dst = cur ? t0 : t1;
            for (int s = 0; s < n_samples; s++)
                ki_xform_raw(dst + (size_t)s * (size_t)img_sz,
                             src + (size_t)s * (size_t)img_sz,
                             w, h, ch, steps[i]);
            cur = !cur;
        }
        /* Step n-1: active → out */
        {   uint8_t *src = cur ? t1 : t0;
            for (int s = 0; s < n_samples; s++)
                ki_xform_raw(out + (size_t)s * (size_t)img_sz,
                             src + (size_t)s * (size_t)img_sz,
                             w, h, ch, steps[n_steps - 1]);
        }
        free(t0); free(t1);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * GB CACHE HASH — key for gb_buf caching (data/gb/)
 * ═══════════════════════════════════════════════════════════════════════
 * All parameters that affect gb_buf (W0, input pipeline, dimensions).
 * W0[0..3] = PRNG proxy (first 4 weights). */
static inline uint32_t gb_cache_hash(const uint32_t *w0_first4,
                                       int total_train, int total_eval,
                                       int H_local, int NC_slice,
                                       size_t n_cont,
                                       const char *color_name,
                                       const char *enc_name, int enc_width,
                                       const char *xf_name) {
    uint32_t h = (uint32_t)KI_DATASET_ID;
    /* w0_first4 may be NULL in KI_BITVOTING mode (no W0) — use a fixed
     * proxy so the gb cache key stays deterministic per mode. */
    if (w0_first4) {
        for (int i = 0; i < 4; i++) h = h * 31 + w0_first4[i];
    } else {
        h = h * 31 + 0x42495654u;   /* "BIVT" — Bit-Voting marker */
    }
    h = h * 31 + (uint32_t)total_train;
    h = h * 31 + (uint32_t)total_eval;
    h = h * 31 + (uint32_t)H_local;
    h = h * 31 + (uint32_t)NC_slice;
    h = h * 31 + (uint32_t)n_cont;
    h = h * 31 + (uint32_t)aa.splitVN;
    h = h * 31 + (uint32_t)aa.splitHN;
    h = h * 31 + (uint32_t)aa.channel;
    h = h * 31 + (uint32_t)aa.enc_size;
    h = h * 31 + (uint32_t)KI_BIT_WIDTH;
    h = h * 31 + (uint32_t)KI_PX;
    h = h * 31 + (uint32_t)KI_COLORS;
    while (color_name && *color_name) h = h * 31 + (uint8_t)*color_name++;
    while (enc_name && *enc_name)   h = h * 31 + (uint8_t)*enc_name++;
    h = h * 31 + (uint32_t)enc_width;
    while (xf_name && *xf_name)     h = h * 31 + (uint8_t)*xf_name++;
    return h;
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
            printf("  --step-err cos-time|cos-err|pow[=NUM]|pow-eval[=NUM]|const[=NUM]                  (default: %s)\n", mode_str());
            printf("                    Step mode: N=err-proportional, auto=compute,\n");
            printf("                    cos-time  : time-based cosine\n");
            printf("                    cos-err   : error cosine\n");
            printf("                    pow       : step_init×(train_err/total)^power\n");
            printf("                    pow-eval  : step_init×(eval_err/total)^power  (self-regularizing)\n");
            printf("                    const     : step_init (const=NUM: fixed step NUM)\n");
            printf("  --step-const N    alias for --step-err const=####                               (default: %d)\n", aa.stepN);
            printf("  --step-power F    alias for --step-err pow=####                                 (default: %.1f, 1.0=linear)\n", (double)aa.step_power);
            printf("  --member-threshold N  Ignore members below N%% training accuracy in ensemble voting  (default: 0=off)\n");
            printf("  --gap-k F         Exp(-K × gap) step damping when overfitting gap widens        (default: %.1f)\n", (double)aa.gap_k);
            printf("                    gap = train_err%% - eval_err%%  |  step *= exp(-K × gap)\n");
            printf("  --err-rollback    Rollback targets when training err increases                  (default: off)\n");
            printf("  --maj 1|1r|1p|1rp|3|7  Majority mode (default: 1):\n");
            printf("       1   = container-level flat (original per-bit)\n");
            printf("       1r  = container-level row-wise (per row, then cross-row)\n");
            printf("       1p  = pixel-genau flat (default for --maj 1)\n");
            printf("       1rp = pixel-genau row-wise (per row pixel, then cross-row)\n");
            printf("       3   = 3-tree Baum (legacy)\n");
            printf("       7   = 7-tree (5/7 threshold)\n");
            printf("  --maj1-thresh N   Exact half threshold for maj=1 (default: %d)\n", aa.maj1_thresh);
            printf("                    -2 = auto per encoding (n*135/256 ≈52.7%% for 8-bit)\n");
            printf("                    -1 = n/2 (standard majority)\n");
            printf("                     0 = every bit passes\n");
            printf("                    >0 = exact count value\n");
            printf("  --maj-step N      Pixel step between majority triples (default: %d, auto when 0)\n", KI_PX_PER_CONT_W);
            printf("                    N=0: auto (KI_PX_PER_CONT). Fast path when N %% KI_PX_PER_CONT == 0\n");
            printf("  --rows-mode flat|rows  Split image into per-row members (default: flat)\n");
            printf("  --no-precompute   Skip h0/gb caches, compute on-the-fly per batch (saves RAM, slower)  (default: off)\n");
            printf("  --warmup N        Linear warmup epochs                                          (default: %d, 0=off)\n", aa.warmup_epochs);
            printf("  ---------------------------------------------------------------------------------------------\n");
            printf("  --seed N          Random seed                                                   (default: %d)\n", aa.seed);
            printf("  --seed-file PATH-TO-RANDOM-FILE                                                 (default: none)\n");
            printf("                    Use true random data from file instead of PRNG\n");
            printf("  --seed-splitmix   Use splitmix64 PRNG                                           (default: on)\n");
            printf("  --seed-member const|incr|once  W0 seeding mode.                                 (default: %s)\n", ensemble_seed_str());
            printf("                    once    : one seed, all sequential.\n");
            printf("                    const   : same W0 per ensemble (tests color diversity).\n");
            printf("                    incr    : unique seed per member.\n");
            printf("  ---------------------------------------------------------------------------------------------\n");
            printf("  --channel [packed|full,][flat,]...                                             (default: %s)\n", color_str());
            printf("                    See --help-channels for details\n");
            printf("  --encoding [all|%s]  (default: latest)\n", ki_enc_names_all());
            printf("                    See --help-encoding for details\n");
            printf("  --encoding-sizeN 0/8/16/24/32     0=RAW passthrough (no encoding), 8..32=thermometer width  (default: %d)\n", KI_BIT_WIDTH);
            printf("  --xform id,hflip,..,sft-d3,sft-r3      Image transform ensemble                 (default: id)\n");
            printf("                    Aliases: all (20 transforms), shift (12 shifts), performance (4×)\n");
            printf("                    See --help-xform for details\n");
            printf("  --export-merge-scores DIR  Save per-member scores to archive files for merge    (default: none)\n");
            printf("  --export-scores FILE  Save per-sample ensemble scores (10×int64+uint8)          (default: none)\n");
            printf("  --export-neurons FILE  Save gb_buf+Target+Offset per member (v3) for Adam..     (default: none)\n");
            printf("  --export-gb           Cache gb_buf to data/gb/ and auto-load on restart          (default: off)\n");
            printf("  --export DIR      Export directory                                              (default: none)\n");
            printf("  --import DIR      Load model for inference                                      (default: none)\n");
            printf("  --predictions FILE                                                              (default: none)\n");
            printf("                    Export per-sample predictions (for vis-errors, eval only)\n");
            printf("  --optional target-norm  Vote normalisierung (equal voting power)                (default: off)\n");
            printf("  --?no-?multi-correct  Only punish argmax, not all over true_k                   (default: multi-correct)\n");
            printf("  --target-init count|random|inverse|uniform|prior|laplace|dampen  Target initialisation              (default: %s)\n", target_init_str());
            printf("                    See --help-target-init for details\n");
            printf("  ---------------------------------------------------------------------------------------------\n");
            printf("  --dry-run         Print architecture and exit                                   (default: off)\n");
            printf("  --quick           5000 train / 2000 eval\n");
            printf("  --qq              5000 train / 2000 eval / 3 epochs\n");
            printf("  --debug           Verbose output                                                (default: off)\n");
            printf("  --debug-maj auto|container|pixel  Force majority path (debug)                   (default: auto)\n");
            printf("  --debug-h0        Per-neuron debug                                              (default: off)\n");
            printf("  --debug-class-voting?-all?  Member × Class accuracy table (end only)            (default: off)\n");
            printf("  --debug-confusion-matrix?-all?  Confusion matrix table (end only)               (default: off)\n");
            printf("  --debug-member    Verbose member-by-member output with channel/encoding/xform     (default: off)\n");
            printf("  --debug-member-stats [FILE]  Per-member quality table (ensemble gain, diversity)     (default: off)\n");
            printf("  --debug-cache     Log EVERY cache create/use (xform cache, cex *.pre, gb cache)   (default: off)\n");
            printf("  --xform-cache-level N  1=transform only, 2=+block cache (CIFAR, ~2 GB slot)      (default: 2)\n");
            printf("  --filter #,#,... or name,name,...  Restrict to specific classes only            (default: none)\n");
            printf("                    See --help-filter for class names\n");
            printf("  --shuffle         Shuffle data before train/eval split                          (default: off)\n");
            exit(1);
        } else if (strcmp(argv[i], "--help-channels") == 0) {
            printf("--channel [packed|full,][flat,]...                                             (default: %s)\n", color_str());
#if KI_COLORS > 1
            printf("  Channel selection (comma-sep).  Encoding via --encoding.\n");
            printf("  packed/full   : 4px/cont or 1px/cont,\n");
            printf("  flat          : all selected blocks in one wide W0,\n");
            printf("  auge          : lum|al=(R+G)/2, rg|am=R-G, by|ap=B-(R+G)/2,\n");
            printf("                :     bl=(R+B)/2,    bm=R-B,    bp=G-(R+B)/2,\n");
            printf("  diff          : rg=R-G, rb=R-B, gb=G-B (color opponent),\n");
            printf("  rgb           : r=R, g=G, b=B,\n");
            printf("  grey          : y|601=ITU-601, yl|709=ITU-709\n");
            printf("  h             : hue (Farbwinkel, atan2-basiert)\n");
            printf("  s             : saturation (Farbsaettigung, max-min)\n");
            printf("  c             : contrast (Sobel-Kanten auf LUM)\n");
            printf("  edge          : edges via Sobel on Y luminance\n");
            printf("  bin           : Otsu-binarized Y luminance (filled black/white regions)\n");
            printf("  lbp           : Local Binary Pattern (8-bit texture descriptor)\n");
            printf("  dog           : Difference of Gaussians (band-pass edges)\n");
            printf("  var           : Local variance (texture roughness)\n");
            printf("  dir           : Gradient direction (8-bin quantized, 0..248)\n");
            printf("  range         : Local range (max-min in 3×3, texture sharpness)\n");
            printf("  lbp-rg        : LBP on RG opponent (chromatic texture)\n");
            printf("  dist          : Center distance (positional encoding, 255=center)\n");
#else
            printf("  Grayscale (KI_COLORS==1, MNIST/Fashion) channels:\n");
            printf("  mnist         : raw grayscale pixels (default, single block)\n");
            printf("  edge          : Sobel edge magnitude on grayscale — button placket\n");
            printf("                  (Shirt), sleeve hems, lapels (Coat) become explicit\n");
            printf("                  edges. The structure layer missing for Shirt/\n");
            printf("                  Pullover/Coat (plan-2026-08-13-fashion-edge-channel.md)\n");
            printf("  bin           : Otsu-binarized grayscale (filled black/white regions)\n");
            printf("  lbp           : Local Binary Pattern (8-bit texture descriptor)\n");
            printf("  dog           : Difference of Gaussians (band-pass edges)\n");
            printf("  var           : Local variance (texture roughness)\n");
            printf("  dir           : Gradient direction (8-bin quantized, 0..248)\n");
            printf("  range         : Local range (max-min in 3×3, texture sharpness)\n");
            printf("  Use comma-sep: --channel edge,mnist  (multiple blocks per member)\n");
            printf("  Combine with --encoding and --xform per block: edge:exp8,mnist:lin8\n");
#endif
            exit(1);   /* INTENTIONAL: non-zero so run-research.sh suppresses logging */
        } else if (strcmp(argv[i], "--help-encoding") == 0) {
            printf("--encoding [all|%s]                              (default: latest)\n", ki_enc_names_all());
            printf("  OR <color>:<enc>[width] per-block: r:exp16,g:lin8,b:sqrt8   \n");
            printf("  Pixel-Encoding pro Farb-Block.\n");
            printf("  Optionaler Width-Suffix: exp16=16-bit, lin32=32-bit\n");
            printf("  8-bit  : 4 px/cont, 8 Stufen (default, exp=0.3)\n");
            printf("  16-bit : 2 px/cont, 16 Stufen (exp=0.5, 2x Breite)\n");
            printf("  32-bit : 1 px/cont, 32 Stufen (exp=0.7, 4x Breite)\n");
            printf("  --encoding-sizeN 0/8/16/24/32     0=RAW passthrough (no encoding), 8..32=thermometer width  (default: %d)\n", KI_BIT_WIDTH);
            printf("    Without explicit width suffix (e.g. \"exp\" instead of \"exp16\"), the\n");
            printf("    global --encoding-sizeN is used. Combine with --splitHN 2 for 2× AND2\n");
            printf("    filter resolution: more bits → finer distribution → more signal.\n");
            printf("    NOTE: KI_BIT_WIDTH in ki-local.h is the master pixel width.\n");
            printf("  lin7   7-level thermometer (old bin),\n");
            printf("  lin8   linear (pv*width/256),\n");
            printf("  down   shadow emphasis,\n");
            printf("  up     highlight emphasis,\n");
            printf("  mid    midtone emphasis,\n");
            printf("  log    logarithmic,\n");
            printf("  exp    exponential,\n");
            printf("  sig    sigmoid.\n");
            printf("  sqrt   square root (soft exp, bright emphasis).\n");
            printf("  cbrt   cube root (even softer, natural image curve).\n");
            printf("  gamma  gamma 0.45 (tunable power-law, complementary).\n");
            printf("  tri    triangle (peaks at midtones, zero at ends).\n");
            printf("  inv-exp inverse exp (dark emphasis, 1−e^(−k·pv)).\n");
            printf("  raw    no encoding (raw 8-bit bytes).\n");
            printf("\n");
            printf("  Dataset group aliases (multi-block, per --encoding):\n");
#if KI_DATASET_ID == 1  /* CIFAR-10 */
            printf("    latest      : 17 members: ey-b,ey-a,ey-h,ey-s-1,ey-s-2\n");
            printf("    latest-2    : 11 members: optimized via sweep (gamma/sqrt/cbrt)\n");
            printf("    performance : 12 members: ey-b-2,ey-a-2,ey-h,ey-s-2 (color-free)\n");
            printf("    ey-a        : b=up,al=down,am=sig,ap=sig          (4 blocks)\n");
            printf("    ey-a-2      : al=down,am=sig,ap=sig               (3 blocks, no color)\n");
            printf("    ey-b        : g=up,bl=down,bm=sig,bp=sig          (4 blocks)\n");
            printf("    ey-b-2      : bl=down,bm=sig,bp=sig               (3 blocks, no color)\n");
            printf("    ey-c        : r=up,cl=down,cm=sig,cp=sig          (4 blocks)\n");
            printf("    ey-c-2      : cl=down,cm=sig,cp=sig               (3 blocks, no color)\n");
            printf("    ey-h        : h=down,c=exp,gb=sig                 (3 blocks)\n");
            printf("    ey-s        : lbp=up,dog=sig,var=exp              (3 spatial/texture)\n");
            printf("    ey-s-1      : lbp=gamma,dog=sig,var=exp           (3 spatial, gamma variant)\n");
            printf("    ey-s-2      : dir=down,range=log,lbp-rg=mid       (3 spatial: dir+range+lbp-rg)\n");
            printf("    top-rgb     : r=down,g=down,b=down                (3 blocks)\n");
#else  /* MNIST, Fashion-MNIST (grayscale) */
            printf("    latest : exp8 (single block)\n");
#endif
            exit(1);   /* INTENTIONAL: non-zero so run-research.sh suppresses logging */
        } else if (strcmp(argv[i], "--help-filter") == 0) {
            printf("--filter #,#,... or name,name,...  Restrict to specific classes only             (default: none)\n");
            printf("                    Examples: --filter 0,1 or --filter name1,name2\n");
            printf("                    Available:");
            for (int _k = 0; _k < KI_NCLASSES; _k++)
                printf(" %s(%d)", ki_class_names[_k], _k);
            printf("\n");
            exit(1);   /* INTENTIONAL: non-zero so run-research.sh suppresses logging */
        } else if (strcmp(argv[i], "--help-target-init") == 0) {
            printf("--target-init count|random|inverse|uniform|prior|laplace|dampen  Target initialisation  (default: %s)\n", target_init_str());
            printf("  count   : count-based (default). Accumulate corrections from counting.\n");
            printf("  random  : uniform random [0, n_k] per neuron×class.\n");
            printf("  inverse : negated count logits. +1 for all classes EXCEPT true class.\n");
            printf("  uniform : all targets = 1 (constant, class prior only).\n");
            printf("  prior   : per-class constant = class_count[k] (no per-neuron variation).\n");
            printf("  laplace : count +1 per entry (additive smoothing, clamped to n_k).\n");
            printf("  dampen  : count >> 1 (shape preserved, peak/valley amplitude halved).\n");
            exit(1);   /* INTENTIONAL: non-zero so run-research.sh suppresses logging */
        } else if (strcmp(argv[i], "--help-xform") == 0) {
            printf("--xform token[,token,...]  Image transform ensemble  (default: id)\n");
            printf("  D4 geometric transforms (8):\n");
            printf("    id       : identity (original image)\n");
            printf("    hflip    : horizontal flip (left-right mirror)\n");
            printf("    vflip    : vertical flip (top-bottom mirror)\n");
            printf("    dflip1   : main diagonal flip (transpose)\n");
            printf("    dflip2   : anti-diagonal flip\n");
            printf("    rot90    : rotate 90° clockwise\n");
            printf("    rot180   : rotate 180° (= hflip+vflip combined)\n");
            printf("    rot270   : rotate 270° clockwise (= rot90⁻¹)\n");
            printf("    rot45    : rotate 45° clockwise (nearest-neighbor, borders filled with 0)\n");
            printf("    spiral   : spiral distortion (bilinear, strongest at center, chromatic per channel)\n");
            printf("    colswap-3-4 : swap col 3+4k ↔ 4+4k (majority triple (0,4,8) → (0,3,7))\n");
            printf("    colswap-2-4 : swap col 2+4k ↔ 4+4k (majority triple (0,4,8) → (0,2,6))\n");
            printf("    colswap-1-4 : swap col 1+4k ↔ 4+4k (majority triple (0,4,8) → (0,1,5))\n");
            printf("  Row-wise running average filters (wrap-right):\n");
            printf("    avg2       : p[i] = (p[i] + p[i+1]) / 2  (2-tap blur, wrap-right)\n");
            printf("    avg3       : p[i] = (p[i] + p[i+1] + p[i+2]) / 3  (3-tap)\n");
            printf("    avg4       : p[i] = (p[i] + p[i+1] + p[i+2] + p[i+3]) / 4  (4-tap)\n");
            printf("  Pipeline chaining via @ (sequential application):\n");
            printf("    X@Y        : apply X first, then Y on the result\n");
            printf("    Example: rot90@avg4  → rotate 90°, then 4-tap blur\n");
            printf("    Example: avg2@avg4   → 2-tap then 4-tap = ~6-tap\n");
             printf("    Example: rot45@avg2@avg4  → rotate 45°, avg2, avg4\n");
             printf("    Note: id steps are preserved (rot0@id stays 'rot0@id',\n");
             printf("          no filtering — the name equals what was entered)\n");
            printf("  Pixel shifts (12) — fill vacated pixels with 0:\n");
            printf("    sft-u1/2/3  : shift up by 1/2/3 px\n");
            printf("    sft-d1/2/3  : shift down by 1/2/3 px\n");
            printf("    sft-l1/2/3  : shift left by 1/2/3 px\n");
            printf("    sft-r1/2/3  : shift right by 1/2/3 px\n");
            printf("  Aliases:\n");
            printf("    all          : all-basic,all-shift,all-shuffle\n");
            printf("    performance  : id,hflip,rot45,rot90,spiral (5×, faster experiments)\n");
            printf("    augmentation : all-basic,all-shift\n");
            printf("    all-basic    : id,hflip,vflip,dflip1,dflip2,rot90,rot180,rot270,rot45,spiral\n");
            printf("    all-shift    : all 12 pixel shifts (= 20 - 8 D4)\n");
            printf("    all-shuffle  : shuffle+shuffle1-shuffle10\n");
            printf("  Multiple transforms create independent members with own W0+Target.\n");
            printf("  Each transform is applied BEFORE channel computation.\n");
            printf("  Example: --xform id,hflip           → 2× member multiplier\n");
            printf("           --xform all                → 30+ member multiplier\n");
            printf("           --xform shift              → 12× member multiplier\n");
            printf("           --xform performance        → 5× member multiplier\n");
            printf("           --xform rot90@avg4         → 1× member (pipeline)\n");
            printf("           --xform id,rot45,rot90     → 3× members (independent W0)\n");
            printf("           --xform avg4,rot45@avg4,rot90@avg4  → 3× members\n");
            exit(1);   /* INTENTIONAL: non-zero so run-research.sh suppresses logging */
        } else if (strcmp(argv[i], "--completion") == 0) {
            ki_completion_dispatch(argc, argv, &i);
        } else if (strcmp(argv[i], "--dry-run") == 0) {
            aa.dry_run = 1;
            /* Execution: 0 epochs (no training). cfg_epochs keeps the user's
             * --epochsN value so the Args:/RESULT display stays truthful
             * (FIX 2026-08-10: dry-run showed Ep=0 despite --epochsN 10). */
            if (aa.cfg_epochs <= 0) aa.cfg_epochs = aa.epochs;
            aa.epochs  = 0;
        } else if (strcmp(argv[i], "--debug") == 0) {
            aa.debug = 1;
        } else if (strcmp(argv[i], "--quick") == 0) {
            aa.trainN = 5000; aa.evalN = 2000;
        } else if (strcmp(argv[i], "--qq") == 0) {
            aa.trainN = 5000; aa.evalN = 2000; aa.epochs = 3; aa.cfg_epochs = 3;
        } else if (strcmp(argv[i], "--hiddenN") == 0 && i + 1 < argc) {
            aa.hidden = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--epochsN") == 0 && i + 1 < argc) {
            aa.epochs = atoi(argv[++i]);
            aa.cfg_epochs = aa.epochs;   /* keep configured value for display */
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
        } else if (strcmp(argv[i], "--eff-lambda") == 0 && i + 1 < argc) {
            aa.eff_lambda = (float)atof(argv[++i]);
            if (aa.eff_lambda < 0.0f) {
                fprintf(stderr, "[WARN] --eff-lambda must be >= 0 — clamping to 0\n");
                aa.eff_lambda = 0.0f;
            }
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
            } else if (strncmp(val, "pow-eval", 8) == 0) {
                aa.step_mode  = STEP_POW_EVAL;
                if (val[8] == '=')               /* --step-err pow-eval=0.5 → setzt step_power */
                    aa.step_power = (float)atof(val + 9);
            } else if (strncmp(val, "pow", 3) == 0) {
                aa.step_mode  = STEP_POW;
                if (val[3] == '=')               /* --step-err pow=0.5 → setzt step_power */
                    aa.step_power = (float)atof(val + 4);
            } else {
                fprintf(stderr, "[ERROR] --step-err: unknown mode '%s'. "
                        "Valid: cos-time, cos-err, pow[=NUM], pow-eval[=NUM], const[=NUM]\n", val);
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
        } else if (strcmp(argv[i], "--member-threshold") == 0 && i + 1 < argc) {
            aa.member_threshold = atoi(argv[++i]);
            if (aa.member_threshold < 0) aa.member_threshold = 0;
        } else if (strcmp(argv[i], "--gap-k") == 0 && i + 1 < argc) {
            aa.gap_k = (float)atof(argv[++i]);
            if (aa.gap_k < 0.0f) aa.gap_k = 0.0f;
        } else if (strcmp(argv[i], "--err-rollback") == 0) {
            aa.err_rollback = 1;
        } else if (strcmp(argv[i], "--maj") == 0 && i + 1 < argc) {
            const char *val = argv[++i];
            if (strcmp(val, "1") == 0) {
                aa.maj_mode = KI_MAJ_1;
            } else if (strcmp(val, "1r") == 0) {
                aa.maj_mode = KI_MAJ_1R;
            } else if (strcmp(val, "1p") == 0) {
                aa.maj_mode = KI_MAJ_1P;
            } else if (strcmp(val, "1rp") == 0) {
                aa.maj_mode = KI_MAJ_1RP;
            } else if (strcmp(val, "3") == 0) {
                aa.maj_mode = KI_MAJ_3;
            } else if (strcmp(val, "7") == 0) {
                aa.maj_mode = KI_MAJ_7;
            } else if (strcmp(val, "0") == 0) {
                aa.maj_mode = KI_MAJ_1;  /* alias for fast testing */
            } else {
                fprintf(stderr, "[ERROR] --maj: expected '1','1r','1p','1rp','3','7', got '%s'\n", val);
                exit(1);
            }
        } else if (strcmp(argv[i], "--maj1-thresh") == 0 && i + 1 < argc) {
            int val = atoi(argv[++i]);
            if (val < -2) { fprintf(stderr, "[ERROR] --maj1-thresh: expected >= -2, -2=auto=encoding, -1=n/2\n"); exit(1); }
            aa.maj1_thresh = val;
        } else if (strcmp(argv[i], "--maj-step") == 0 && i + 1 < argc) {
            int val = atoi(argv[++i]);
            if (val < 0) { fprintf(stderr, "[ERROR] --maj-step: expected non-negative integer\n"); exit(1); }
            aa.maj_step = val;
        } else if (strcmp(argv[i], "--debug-maj") == 0 && i + 1 < argc) {
            const char *val = argv[++i];
            if (strcmp(val, "auto") == 0) aa.debug_maj = 0;
            else if (strcmp(val, "container") == 0) aa.debug_maj = 1;
            else if (strcmp(val, "pixel") == 0) aa.debug_maj = 2;
            else { fprintf(stderr, "[ERROR] --debug-maj: expected 'auto','container','pixel'\n"); exit(1); }
        } else if (strcmp(argv[i], "--rows-mode") == 0 && i + 1 < argc) {
            const char *val = argv[++i];
            if (strcmp(val, "rows") == 0) aa.rows_mode = 1;
            else if (strcmp(val, "flat") == 0) aa.rows_mode = 0;
            else { fprintf(stderr, "[ERROR] --rows-mode: expected 'flat' or 'rows'\n"); exit(1); }
        } else if (strcmp(argv[i], "--no-precompute") == 0) {
            aa.no_precompute = 1;
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
        } else if (strcmp(argv[i], "--export-gb") == 0) {
            aa.export_gb = 1;
        } else if (strcmp(argv[i], "--import-gb") == 0) {
            aa.import_gb = 1;
        } else if (strcmp(argv[i], "--sweep") == 0) {
            aa.sweep = 1;
        } else if (strcmp(argv[i], "--force") == 0) {
            aa.force = 1;
        } else if (strcmp(argv[i], "--member-file") == 0 && i + 1 < argc) {
            strncpy(aa.member_file, argv[++i], sizeof(aa.member_file) - 1);
            aa.member_file[sizeof(aa.member_file) - 1] = '\0';
        } else if (strcmp(argv[i], "--member") == 0 && i + 1 < argc) {
            strncpy(aa.member_str, argv[++i], sizeof(aa.member_str) - 1);
            aa.member_str[sizeof(aa.member_str) - 1] = '\0';
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
        } else if (strcmp(argv[i], "--debug-member") == 0) {
            aa.debug_member = 1;
        } else if (strcmp(argv[i], "--debug-cache") == 0) {
            aa.debug_cache = 1;
        } else if (strcmp(argv[i], "--no-ens-cache") == 0) {
            aa.no_ens_cache = 1;
        } else if (strcmp(argv[i], "--xform-cache-level") == 0 && i + 1 < argc) {
            int lvl = atoi(argv[++i]);
            if (lvl != 1 && lvl != 2) {
                fprintf(stderr, "[ERROR] --xform-cache-level: expected 1 or 2, got %d\n", lvl);
                exit(1);
            }
            aa.xform_cache_level = lvl;
        } else if (strcmp(argv[i], "--debug-member-stats") == 0) {
            aa.debug_member_stats = 1;
            /* Optional: next arg not starting with -- is the filename */
            if (i + 1 < argc && argv[i+1][0] != '-') {
                strncpy(aa.member_scores_path, argv[++i], sizeof(aa.member_scores_path) - 1);
                aa.member_scores_path[sizeof(aa.member_scores_path) - 1] = '\0';
            } else {
                snprintf(aa.member_scores_path, sizeof(aa.member_scores_path), "member-scores.bin");
            }
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
        } else if (strcmp(argv[i], "--xform") == 0 && i + 1 < argc) {
            const char *val = argv[++i];
            aa.xforms = 0;
            aa.xform_list_count = 0;
            char xbuf[128];
            strncpy(xbuf, val, sizeof(xbuf) - 1);
            xbuf[sizeof(xbuf) - 1] = '\0';
            for (char *tok = strtok(xbuf, ","); tok; tok = strtok(NULL, ",")) {
                while (*tok == ' ' || *tok == '\t') tok++;
                /* Try alias expansion first */
                if (ki_xform_alias_expand(tok) != NULL) {
                    char xalias[1024];
                    strncpy(xalias, tok, sizeof(xalias) - 1);
                    xalias[sizeof(xalias) - 1] = '\0';
                    /* Iterative 5-pass expansion: Phase 1 (full) + Phase 2 (per-token) */
                    for (int _iter = 0; _iter < 5; _iter++) {
                        /* Phase 1: full string match */
                        const char *_full = ki_xform_alias_expand(xalias);
                        if (_full) {
                            strncpy(xalias, _full, sizeof(xalias) - 1);
                            xalias[sizeof(xalias) - 1] = '\0';
                            continue;
                        }
                        /* Phase 2: per-token expansion */
                        char _tmp[1024], _new[1024] = "";
                        strncpy(_tmp, xalias, sizeof(_tmp) - 1);
                        _tmp[sizeof(_tmp) - 1] = '\0';
                        int _any = 0;
                        char *_save2 = NULL;
                        for (char *_t = strtok_r(_tmp, ",", &_save2); _t; _t = strtok_r(NULL, ",", &_save2)) {
                            while (*_t == ' ' || *_t == '\t') _t++;
                            const char *_pe = ki_xform_alias_expand(_t);
                            if (_pe) {
                                if (_new[0]) strncat(_new, ",", sizeof(_new) - 1);
                                strncat(_new, _pe, sizeof(_new) - strlen(_new) - 1);
                                _any = 1;
                            } else {
                                if (_new[0]) strncat(_new, ",", sizeof(_new) - 1);
                                strncat(_new, _t, sizeof(_new) - strlen(_new) - 1);
                            }
                        }
                        if (!_any) break;
                        strncpy(xalias, _new, sizeof(xalias) - 1);
                        xalias[sizeof(xalias) - 1] = '\0';
                    }
                    /* Parse final expanded string token by token */
                    char *_save3 = NULL;
                    for (char *_t = strtok_r(xalias, ",", &_save3); _t; _t = strtok_r(NULL, ",", &_save3)) {
                        while (*_t == ' ' || *_t == '\t') _t++;
                        int _x = ki_xform_parse(_t);
                        if (_x >= 0) {
                            ki_xform_bit_set(_x);
                            ki_xform_list_add(_x);
                        }
                    }
                } else if (strchr(tok, '@')) {
                    /* Pipeline: rot90@avg4 → create pipe, store virtual ID.
                     * ALIAS CROSS-PRODUCT (2026-08-12): an alias on EITHER side
                     * of "@" expands to a cross-product of pipelines:
                     *   all-shift@filter-bv → sft-u1@id, sft-u1@spiral, ...,
                     *                          sft-r3@rot90  (12 × 6 = 72 pipes)
                     *   sweep@spiral        → id@spiral, hflip@spiral, ... (16×1)
                     *   id@filter-bv        → id@id, id@spiral, ...        (1×6)
                     * A CHAIN (a@b@c) is supported: the LAST token is the
                     * suffix, everything before it is the prefix — both sides
                     * may be aliases. The cross-product is bounded by
                     * KI_XFORM_PIPE_MAX (512); too many pipelines abort. */
                    const char *_at = strchr(tok, '@');
                    size_t _pre_len = (size_t)(_at - tok);
                    char _pre[256], _suf[256];
                    if (_pre_len >= sizeof(_pre)) _pre_len = sizeof(_pre) - 1;
                    memcpy(_pre, tok, _pre_len); _pre[_pre_len] = '\0';
                    strncpy(_suf, _at + 1, sizeof(_suf) - 1);
                    _suf[sizeof(_suf) - 1] = '\0';
                    /* Suffix may itself be a chain (b@c) — keep it intact for
                     * ki_xform_parse_or_pipe, only expand if it is a plain alias. */
                    const char *_alias_exp = ki_xform_alias_expand(_pre);
                    const char *_suf_exp   = strchr(_suf, '@') ? NULL
                                                               : ki_xform_alias_expand(_suf);
                    if (_alias_exp || _suf_exp) {
                        /* Cross-product: for every expanded prefix token × every
                         * expanded suffix token create one pipeline. Both sides
                         * are fully alias-expanded (nested aliases included).
                         * Suffix tokens are parsed into a fixed array FIRST
                         * (strtok is not reentrant — re-parsing per prefix
                         * token would lose the stream). */
                        char _pre_buf[4096], _suf_buf[4096];
                        ki_xform_alias_expand_full(_alias_exp ? _alias_exp : _pre,
                                                   _pre_buf, sizeof(_pre_buf));
                        ki_xform_alias_expand_full(_suf_exp ? _suf_exp : _suf,
                                                   _suf_buf, sizeof(_suf_buf));
                        char _suf_toks[256][64];
                        int _suf_n = 0;
                        for (char *_st = strtok(_suf_buf, ","); _st && _suf_n < 256;
                             _st = strtok(NULL, ",")) {
                            while (*_st == ' ' || *_st == '\t') _st++;
                            if (!*_st) continue;
                            snprintf(_suf_toks[_suf_n], 64, "%s", _st);
                            _suf_n++;
                        }
                        int _has_pipe = 0;
                        int _prod_count = 0;
                        for (char *_pt = strtok(_pre_buf, ","); _pt;
                             _pt = strtok(NULL, ",")) {
                            while (*_pt == ' ' || *_pt == '\t') _pt++;
                            if (!*_pt) continue;
                            for (int _si = 0; _si < _suf_n; _si++) {
                                char _pipe_token[512];
                                /* %.240s bounds both tokens so the combined
                                 * string always fits _pipe_token (240+1+240). */
                                snprintf(_pipe_token, sizeof(_pipe_token),
                                         "%.240s@%.240s", _pt, _suf_toks[_si]);
                                int _pid = ki_xform_parse_or_pipe(_pipe_token);
                                if (_pid < 0) {
                                    fprintf(stderr, "[ERROR] --xform: bad pipeline '%s' (from alias '%s@%s')\n",
                                            _pipe_token, _pre, _suf);
                                    exit(1);
                                }
                                ki_xform_bit_set(_pid);
                                ki_xform_list_add(_pid);
                                _has_pipe = 1;
                                if (++_prod_count > KI_XFORM_PIPE_MAX) {
                                    fprintf(stderr, "[ERROR] --xform: cross-product '%s@%s' exceeds %d pipelines\n",
                                            _pre, _suf, KI_XFORM_PIPE_MAX);
                                    exit(1);
                                }
                            }
                        }
                        if (!_has_pipe) {
                            ki_xform_bit_set(KI_XFORM_ID);
                            ki_xform_list_add(KI_XFORM_ID);
                        }
                    } else {
                        /* Direct pipeline (no alias expansion) */
                        int _pid = ki_xform_parse_or_pipe(tok);
                        if (_pid < 0) {
                            fprintf(stderr, "[ERROR] --xform: bad pipeline '%s'\n", tok);
                            exit(1);
                        }
                        ki_xform_bit_set(_pid);
                        ki_xform_list_add(_pid);
                    }
                } else {
                    int xf = ki_xform_parse(tok);
                    if (xf >= 0) {
                        ki_xform_bit_set(xf);
                        ki_xform_list_add(xf);
                    } else {
                        fprintf(stderr, "[ERROR] --xform: unknown transform '%s'. "
                                "Valid: all, shift, augmentation, performance, performance-2, id, hflip, vflip, dflip1, dflip2, "
                                "rot90, rot22, rot67, rot45, spiral, colswap-3-4, colswap-2-4, colswap-1-4, "
                                "avg2, avg3, avg4, "
                                "sft-u1/2/3, sft-d1/2/3, sft-l1/2/3, sft-r1/2/3, "
                                "shuffle, shuffle1..10\n", tok);
                        exit(1);
                    }
                }
            }
            /* Empty check via the LIST, not the bitmask: pipe IDs >= 64 are
             * never set in aa.xforms (ki_xform_bit_set guard) — the list is
             * the authoritative source. Fix 2026-08-09. */
            if (aa.xform_list_count == 0) {
                fprintf(stderr, "[ERROR] --xform: at least one transform required\n");
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
        } else if (strcmp(argv[i], "--target-init") == 0 && i + 1 < argc) {
            const char *val = argv[++i];
            if (strcmp(val, "count") == 0) {
                aa.target_init_mode = KI_TARGET_COUNT;
            } else if (strcmp(val, "random") == 0) {
                aa.target_init_mode = KI_TARGET_RANDOM;
            } else if (strcmp(val, "inverse") == 0) {
                aa.target_init_mode = KI_TARGET_INVERSE;
            } else if (strcmp(val, "uniform") == 0) {
                aa.target_init_mode = KI_TARGET_UNIFORM;
            } else if (strcmp(val, "prior") == 0) {
                aa.target_init_mode = KI_TARGET_PRIOR;
            } else if (strcmp(val, "laplace") == 0) {
                aa.target_init_mode = KI_TARGET_LAPLACE;
            } else if (strcmp(val, "dampen") == 0) {
                aa.target_init_mode = KI_TARGET_DAMPEN;
            } else {
                fprintf(stderr, "[ERROR] --target-init: unknown mode '%s'. "
                        "Valid: count, random, inverse, uniform, prior, laplace, dampen\n", val);
                exit(1);
            }
        } else if (strcmp(argv[i], "--channel") == 0 && i + 1 < argc) {
            const char *val = argv[++i];
            /* 5-pass iterative alias expansion (like --xform / --encoding) */
            char buf[4096];
            strncpy(buf, val, sizeof(buf) - 1);
            buf[sizeof(buf) - 1] = '\0';
            for (int _iter = 0; _iter < 5; _iter++) {
                char _tmp[4096], _new[4096] = "";
                strncpy(_tmp, buf, sizeof(_tmp) - 1);
                _tmp[sizeof(_tmp) - 1] = '\0';
                int _any = 0;
                char *_save = NULL;
                for (char *_t = strtok_r(_tmp, ",", &_save); _t; _t = strtok_r(NULL, ",", &_save)) {
                    while (*_t == ' ' || *_t == '\t') _t++;
                    if (*_t == '\0') continue;
                    const char *_exp = ki_color_alias_expand(_t);
                    if (_exp) {
                        if (_new[0]) strncat(_new, ",", sizeof(_new) - strlen(_new) - 1);
                        strncat(_new, _exp, sizeof(_new) - strlen(_new) - 1);
                        _any = 1;
                    } else {
                        if (_new[0]) strncat(_new, ",", sizeof(_new) - strlen(_new) - 1);
                        strncat(_new, _t, sizeof(_new) - strlen(_new) - 1);
                    }
                }
                if (!_any) break;
                strncpy(buf, _new, sizeof(buf) - 1);
                buf[sizeof(buf) - 1] = '\0';
            }
            /* Build bitmask from fully expanded tokens */
            int mask = 0;
            for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
                while (*tok == ' ' || *tok == '\t') tok++;
                for (char *p = tok; *p; p++) *p = (char)tolower((unsigned char)*p);
                if (strcmp(tok, "packed") == 0) { aa.packedB = 1; continue; }
                else if (strcmp(tok, "full") == 0) { aa.packedB = 0; continue; }
                else if (strcmp(tok, "flat") == 0) { aa.debug_flat = 1; continue; }
                if (strcmp(tok, "all") == 0) {
                    for (int b = 0; b <= COLOR_NB; b++) mask |= (1 << b);
                    continue;
                }
                int bit = ki_color_parse(tok);
                if (bit >= 0) { mask |= (1 << bit); continue; }
                fprintf(stderr, "[ERROR] --channel: unknown '%s'. "
                        "Valid: %s\n", tok, ki_color_names_all());
                exit(1);
            }
            if (mask == 0) {
                fprintf(stderr, "[ERROR] --channel: at least one channel required\n");
                exit(1);
            }
            aa.channel = mask;
            aa.channel_explicit = 1;
        } else if (strcmp(argv[i], "--encoding-sizeN") == 0 && i + 1 < argc) {
            int sz = atoi(argv[++i]);
            if (sz != 0 && sz != 8 && sz != 16 && sz != 24 && sz != 32) {
                fprintf(stderr, "[ERROR] --encoding-sizeN: expected 0 (RAW), 8, 16, 24, or 32, got %d\n", sz);
                exit(1);
            }
            aa.enc_size = sz;
        } else if (strcmp(argv[i], "--encoding") == 0 && i + 1 < argc) {
            const char *val = argv[++i];
            /* --encoding impliziert debug_binarize (thermometer mode).
             * Explizites --encoding raw deaktiviert es. */
            int has_raw = 0;
            char buf[4096];
            strncpy(buf, val, sizeof(buf) - 1);
            buf[sizeof(buf) - 1] = '\0';

            /* ── Encoding-Alias-Expansion ─────────────────────────────
             * Dataset-specific aliases defined in ki-local.h
             * via ki_encoding_alias_lookup().  Iterative 5-pass expansion:
             * Phase 1: full-string match, Phase 2: per-token. */
            for (int _iter = 0; _iter < 5; _iter++) {
                /* Phase 1: full string match */
                const char *_full = ki_encoding_alias_lookup(buf);
                if (_full) {
                    strncpy(buf, _full, sizeof(buf) - 1);
                    buf[sizeof(buf) - 1] = '\0';
                    continue;
                }
                /* Phase 2: per-token expand */
                char _tmp[4096], _new[4096] = "";
                strncpy(_tmp, buf, sizeof(_tmp) - 1);
                _tmp[sizeof(_tmp) - 1] = '\0';
                char *_t = strtok(_tmp, ",");
                int _any = 0;
                while (_t) {
                    const char *_val = ki_encoding_alias_lookup(_t);
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

            /* ── Encoding "all": Cartesian product channels × encodings ── */
            if (strcmp(buf, "all") == 0) {
                int _all_width = aa.enc_size > 0 ? aa.enc_size : 8;
                for (int _ch = 1; _ch < COLOR_NB; _ch++) {
                    if (!(aa.channel & (1 << _ch))) continue;
                    for (int _enc = 0; _enc < KI_ENC_COUNT; _enc++) {
                        if (aa.enc_count >= KI_ENC_MAX) break;
                        if (_enc == KI_ENC_LIN7) continue;  /* lin7 = legacy */
                        aa.enc_array[aa.enc_count].type  = (int8_t)_enc;
                        aa.enc_array[aa.enc_count].width = (int8_t)_all_width;
                        aa.enc_array[aa.enc_count].color = (int8_t)_ch;
                        aa.enc_count++;
                    }
                    if (aa.enc_count >= KI_ENC_MAX) break;
                }
                if (aa.enc_count == 0) {
                    aa.enc_array[aa.enc_count].type  = (int8_t)KI_ENC_RAW;
                    aa.enc_array[aa.enc_count].width = (int8_t)_all_width;
                    aa.enc_array[aa.enc_count].color = -1;
                    aa.enc_count++;
                }
                buf[0] = '\0';  /* skip token parsing, fall through to normal post-process */
            }

            for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
                while (*tok == ' ' || *tok == '\t') tok++;
                const char *sep = strchr(tok, ':');
                    int enc;
                    int w  = KI_ENC_WIDTH_DEFAULT;  /* fallback; overridden by post-process if --encoding-sizeN was set */
                    if (sep) {
                        /* Per-block encoding: r:exp16 */
                        char col_buf[32];
                        size_t col_len = (size_t)(sep - tok);
                        if (col_len >= sizeof(col_buf)) col_len = sizeof(col_buf) - 1;
                        memcpy(col_buf, tok, col_len);
                        col_buf[col_len] = '\0';
                        int bit = ki_color_parse(col_buf);
                        if (bit < 0) {
                            fprintf(stderr, "[ERROR] --encoding: unknown color '%s' in '%s'. "
                                    "Valid: %s\n", col_buf, tok, ki_color_names_all());
                            exit(1);
                        }
                        enc = ki_enc_parse(sep + 1, &w);
                        if (enc < 0) {
                        fprintf(stderr, "[ERROR] --encoding: unknown encoding '%s' in '%s'. "
                                "Valid: %s\n", sep + 1, tok, ki_enc_names_all());
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
                     * Each token means ONE entry in enc_array[].
                     * KANAL-KREUZPRODUKT (2026-08-14): when --channel selects
                     * MULTIPLE channels explicitly (e.g. mnist,lbp), one
                     * encoding token expands to one entry PER channel — the
                     * same cross-product --encoding all already does (Z.
                     * 2020-2042). Without this, --channel mnist,lbp with
                     * --encoding sweep silently dropped lbp (only 13 members
                     * instead of 26). Single-channel (default mnist) keeps
                     * the legacy color=-1 entry. */
                    enc = ki_enc_parse(tok, &w);
                    if (enc < 0) {
                        fprintf(stderr, "[ERROR] --encoding: unknown '%s'. "
                                "Valid: %s\n", tok, ki_enc_names_all());
                        exit(1);
                    }
                    int _nch = 0;
                    for (int _ch = 0; _ch < COLOR_NB; _ch++)
                        if (aa.channel & (1 << _ch)) _nch++;
                    if (aa.channel_explicit && _nch > 1) {
                        /* Cross-product: one enc entry per selected channel.
                         * On grayscale, CIFAR-RGB channels in the mask are
                         * skipped (ki_color_is_cifar_rgb); structure channels
                         * (edge/lbp/...) are kept (plan-2026-08-13). */
                        for (int _ch = 0; _ch < COLOR_NB; _ch++) {
                            if (!(aa.channel & (1 << _ch))) continue;
                            if (KI_COLORS <= 1 && ki_color_is_cifar_rgb(_ch)) continue;
                            if (aa.enc_count < KI_ENC_MAX) {
                                aa.enc_array[aa.enc_count].type  = (int8_t)enc;
                                aa.enc_array[aa.enc_count].width = (int8_t)w;
                                aa.enc_array[aa.enc_count].color = (int8_t)_ch;
                                aa.enc_count++;
                            }
                        }
                    } else {
                        if (aa.enc_count < KI_ENC_MAX) {
                            aa.enc_array[aa.enc_count].type  = (int8_t)enc;
                            aa.enc_array[aa.enc_count].width = (int8_t)w;
                            aa.enc_array[aa.enc_count].color = -1;  /* default/all */
                            aa.enc_count++;
                        }
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
            /* Default: resolve "latest" alias with iterative alias expansion
             * (same multi-pass logic as the --encoding parser, lines 609-638).
             * Fallback to lin7/raw if alias is not defined. */
            {
                aa.debug_binarize = 1;  /* thermometer mode */
                char _def_buf[256] = "latest";
                for (int _iter = 0; _iter < 5; _iter++) {
                    /* Phase 1: full-string match */
                    const char *_full = ki_encoding_alias_lookup(_def_buf);
                    if (_full) {
                        strncpy(_def_buf, _full, sizeof(_def_buf) - 1);
                        _def_buf[sizeof(_def_buf) - 1] = '\0';
                        continue;
                    }
                    /* Phase 2: per-token expand */
                char _tmp[4096], _new[4096] = "";
                    strncpy(_tmp, _def_buf, sizeof(_tmp) - 1);
                    _tmp[sizeof(_tmp) - 1] = '\0';
                    char *_t = strtok(_tmp, ",");
                    int _any = 0;
                    while (_t) {
                        while (*_t == ' ' || *_t == '\t') _t++;
                        const char *_val = ki_encoding_alias_lookup(_t);
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
                    if (!_any) break;
                    strncpy(_def_buf, _new, sizeof(_def_buf) - 1);
                    _def_buf[sizeof(_def_buf) - 1] = '\0';
                }
                /* Parse the fully expanded string into enc_array */
                for (char *_tok = strtok(_def_buf, ","); _tok; _tok = strtok(NULL, ",")) {
                    while (*_tok == ' ' || *_tok == '\t') _tok++;
                    const char *_sep = strchr(_tok, ':');
                    int _enc;
                    int _w  = KI_ENC_WIDTH_DEFAULT;
                    if (_sep) {
                        /* per-block: color:enc */
                        char _col_buf[32];
                        size_t _col_len = (size_t)(_sep - _tok);
                        if (_col_len >= sizeof(_col_buf)) _col_len = sizeof(_col_buf) - 1;
                        memcpy(_col_buf, _tok, _col_len); _col_buf[_col_len] = '\0';
                        int _bit = ki_color_parse(_col_buf);
                        if (_bit >= 0) {
                            _enc = ki_enc_parse(_sep + 1, &_w);
                            if (_enc >= 0 && aa.enc_count < KI_ENC_MAX) {
                                aa.enc_array[aa.enc_count].type  = (int8_t)_enc;
                                aa.enc_array[aa.enc_count].width = (int8_t)_w;
                                aa.enc_array[aa.enc_count].color = (int8_t)_bit;
                                aa.enc_count++;
                            }
                        }
                    } else {
                        /* bare token: color=-1, expanded later by first_bare logic */
                        _enc = ki_enc_parse(_tok, &_w);
                        if (_enc >= 0 && aa.enc_count < KI_ENC_MAX) {
                            aa.enc_array[aa.enc_count].type  = (int8_t)_enc;
                            aa.enc_array[aa.enc_count].width = (int8_t)_w;
                            aa.enc_array[aa.enc_count].color = -1;
                            aa.enc_count++;
                        }
                    }
                }
            }
            if (aa.enc_count == 0) {
                /* Fallback: kein Alias definiert → lin7/raw per active channel */
                int def_type = aa.debug_binarize ? KI_ENC_LIN7 : KI_ENC_RAW;
                for (int b = 0; b < COLOR_NB; b++) {
                    if (!(aa.channel & (1 << b))) continue;
                    if (aa.enc_count < KI_ENC_MAX) {
                        aa.enc_array[aa.enc_count].type  = (int8_t)def_type;
                        aa.enc_array[aa.enc_count].width = (int8_t)(aa.enc_size > 0 ? aa.enc_size : KI_ENC_WIDTH_DEFAULT);
                        aa.enc_array[aa.enc_count].color = (int8_t)b;
                        aa.enc_count++;
                    }
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
        /* ── Post-process: override KI_ENC_WIDTH_DEFAULT (8) with --encoding-sizeN ── */
        if (aa.enc_size == 0) {
            /* --encoding-sizeN 0: RAW passthrough, width=8 (no encoding transformation) */
            for (int _i = 0; _i < aa.enc_count; _i++) {
                aa.enc_array[_i].type = KI_ENC_RAW;
                if (aa.enc_array[_i].width < 1)
                    aa.enc_array[_i].width = KI_ENC_WIDTH_DEFAULT;
            }
            /* --encoding-sizeN 0: silent (Structure line shows raw8) */
        } else if (aa.enc_size != KI_ENC_WIDTH_DEFAULT) {
            if (aa.enc_size != KI_BIT_WIDTH) {
                fprintf(stderr, "[WARNING] --encoding-sizeN %d != KI_BIT_WIDTH %d. "
                        "Pixel-majority uses %d bits (compile-time constant).\n",
                        aa.enc_size, KI_BIT_WIDTH, KI_BIT_WIDTH);
            }
            for (int _i = 0; _i < aa.enc_count; _i++) {
                if (aa.enc_array[_i].width == KI_ENC_WIDTH_DEFAULT)
                    aa.enc_array[_i].width = (int8_t)aa.enc_size;
            }
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
 * COUNTER_TYPE — Data format for Target/OFFSET/Correction step
 * ═══════════════════════════════════════════════════════════════════════
 * Default: float (IEEE 754 32-bit, direct, no scaling).
 * Use -DMODE_INT32 for int32_t fixed-point mode.
 */
#ifndef COUNTER_TYPE_IS_FLOAT
# define COUNTER_TYPE_IS_FLOAT 1
#endif
#ifndef COUNTER_TYPE
# define COUNTER_TYPE float
#endif

/* Score accumulator type: double by default — float32 drifts on large
 * ensemble sums (scores reach ~1e9 > 2^24; the merge selection changed
 * 29→44 members with exact accumulation, evidence 2026-08-06).
 * MODE_INT32 sets int64_t. Explicit -DSCORE_TYPE=float reproduces the
 * legacy float32 behavior. */
#ifndef SCORE_TYPE
# define SCORE_TYPE double
#endif

/* ── Real-type labels (shared by trainer AND merge) ───────────────────
 * COUNTER_TYPE / SCORE_TYPE can be overridden per build (-D…); the setup
 * labels must reflect the REAL types, not a hardcoded 2-way assumption
 * (float vs int32). Precision matters: float32 vs double vs int64 give
 * different results (analysis 2026-08-05/06). */
static inline const char *ki_counter_type_str(void) {
    return _Generic((COUNTER_TYPE)0,
        float:   "IEEE 754 float32",
        double:  "IEEE 754 double",
        int32_t: "int32_t",
        int64_t: "int64_t",
        default: "?");
}
static inline const char *ki_score_type_str(void) {
    return _Generic((SCORE_TYPE)0,
        float:   "IEEE 754 float32",
        double:  "IEEE 754 double",
        int64_t: "int64_t",
        default: "?");
}

/* ═══════════════════════════════════════════════════════════════════════
 * OT_PRECISION / OT_F — Scaling factor for fixed-point mode
 * ═══════════════════════════════════════════════════════════════════════
 * int32_t mode (COUNTER_TYPE_IS_FLOAT=0): F = (1<<OT_PRECISION), logit
 *   values scaled by F. ot_precision() applies the fixed-point round
 *   (×F + ±0.5) — needed to store logits in int32.
 * float mode (COUNTER_TYPE_IS_FLOAT=1): F = 1, ot_precision() = IDENTITY.
 *   A float/double stores logits exactly — the old code still applied
 *   ×F + ±0.5 (the "float mode: F=1" comment was a lie; no #ifdef branch
 *   existed). The ±0.5 round is a REAL, non-linear distortion (not a
 *   linear shift) and the ×F is redundant for floats. Lossless since
 *   2026-08-16 (plan-2026-08-16-lossless-float-mode.md). The distinction
 *   is COUNTER_TYPE_IS_FLOAT, NOT MODE_FLT64 — both FLT32 and FLT64 are
 *   lossless (the code already branches on COUNTER_TYPE_IS_FLOAT
 *   everywhere, e.g. merge-ensemble.c fp_scale).
 */
#if COUNTER_TYPE_IS_FLOAT
#  define OT_F (1 << OT_PRECISION)
static inline double ot_precision(double in) { return in * (double)OT_F; }
                        /* Lossless float mode (2026-08-16): scale by F (same
                         * int32 dynamic range) but WITHOUT the ±0.5 rounding.
                         * The F-scale is REQUIRED: the correction branch uses
                         * the SAME gap<OT_F logic as int32 (step×gap/OT_F vs
                         * step), so targets must live on the int32 scale
                         * (±8.7×F) for the comparison to work. The dropped
                         * ±0.5 is the actual precision gain — no rounding
                         * distortion, only the F-scale remains. */
#else
#  define OT_F (1 << OT_PRECISION)  /* always 131072 at OP17 — scaling for gap */
static inline double ot_precision(double in) {
    return in * (double)OT_F + (in >= 0 ? 0.5 : -0.5);
}
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * DEFAULT-NAME-STEM — H512-E10-OT8-M1-105-INT32 (2026-08-18)
 * ═══════════════════════════════════════════════════════════════════════
 * Central derivation of the "stem" from known parameters — identical for
 * trainer (--export-default → export-{STEM}) and MERGE (--member-out-default
 * → member-{STEM}.out). No reference to file/dir names.
 *
 * Schema: H{h}-E{ep}-{OT|BV}{KI_BIT_WIDTH}-M{maj}[-{m1t}]-{INT32|FLT32|FLT64}
 *
 *   - maj1_thresh (m1t) arrives RESOLVED here (explicit value or
 *     ki_default_half() result — resolution is done by the trainer, because only
 *     it includes maj1.h). m1t < 0 ⇒ no suffix (M3 / Bit-Voting).
 *   - M identifier is the majority identifier ("1"/"3"/"-1"), NOT the enum value
 *     (KI_MAJ_1=0, KI_MAJ_3=1). Passed via maj_mode.
 *   - Arch = "OT" (Otto) or "BV" (Bit-Voting), via KI_BITVOTING.
 */
static inline void ki_default_tag(char *buf, size_t bufsz, int H, int ep,
                                  int maj_mode, int m1t) {
    const char *arch =
#ifdef KI_BITVOTING
        "BV";
#else
        "OT";
#endif
    const char *mode =
#ifdef MODE_INT32
        "INT32";
#elif defined(MODE_FLT64)
        "FLT64";
#else
        "FLT32";
#endif
    const char *maj;
    if (maj_mode == KI_MAJ_1) {
        maj = "1";
        if (m1t >= 0)
            snprintf(buf, bufsz, "H%d-E%d-%s%d-M%s-%d-%s", H, ep, arch, KI_BIT_WIDTH, maj, m1t, mode);
        else
            snprintf(buf, bufsz, "H%d-E%d-%s%d-M%s-%s", H, ep, arch, KI_BIT_WIDTH, maj, mode);
    } else if (maj_mode == KI_MAJ_3) {
        maj = "3";
        snprintf(buf, bufsz, "H%d-E%d-%s%d-M%s-%s", H, ep, arch, KI_BIT_WIDTH, maj, mode);
    } else {
        maj = "-1";   /* Bit-Voting: no majority */
        snprintf(buf, bufsz, "H%d-E%d-%s%d-M%s-%s", H, ep, arch, KI_BIT_WIDTH, maj, mode);
    }
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
      case STEP_POW_EVAL:
          snprintf(_mode_buf, sizeof(_mode_buf), "pow-eval(%.3g)", (double)aa.step_power);
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

/* ── Color-String (for --help and SETUP Header) ────────────────── */
/* Returns "R,G,B", "packed:MNIST", "LUM,RG,BY" etc. from the
 * aa.channel mask + aa.packedB flag.
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

/* ── Encoding-String (for --help and TRAINING Header) ──────────── */
/* Returns "R=exp8,G=lin8,B=sig8" etc. — always from enc_array[].
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

/* ── Majority mode string ────────────────────────────────────── */
__attribute__((unused))
static const char *maj_str(void) {
    switch (aa.maj_mode) {
        case KI_MAJ_1:   return "1";
        case KI_MAJ_1R:  return "1r";
        case KI_MAJ_1P:  return "1p";
        case KI_MAJ_1RP: return "1rp";
        case KI_MAJ_3:   return "3";
        case KI_MAJ_7:   return "7";
        default:         return "?";
    }
}

/* ── Target-init string (for --help and SETUP Header) ───────── */
/* Returns "count", "random", or "?" for unknown mode.
 * Uses static buffer, analog to mode_str(). */
__attribute__((unused))
static const char *target_init_str(void) {
    switch (aa.target_init_mode) {
        case KI_TARGET_COUNT:   return "count";
        case KI_TARGET_RANDOM:  return "random";
        case KI_TARGET_INVERSE: return "inverse";
        case KI_TARGET_UNIFORM: return "uniform";
        case KI_TARGET_PRIOR:   return "prior";
        case KI_TARGET_LAPLACE: return "laplace";
        case KI_TARGET_DAMPEN:  return "dampen";
        default:                return "?";
    }
}

/* ── Xform-String (for SETUP Header) ────────────────────────── */
/* Returns "id,hflip" etc. from aa.xforms bitmask.
 * Uses static buffer, analog to color_str(). */
__attribute__((unused))
static const char *xform_str(void) {
    static char _xform_buf[1024];
    int pos = 0;
    for (int i = 0; i < aa.xform_list_count && i < KI_XFORM_LIST_MAX; i++) {
        int x = aa.xform_list[i];
        if (pos > 0) _xform_buf[pos++] = ',';
        const char *n = ki_xform_str(x);
        while (*n && pos < (int)sizeof(_xform_buf) - 2) _xform_buf[pos++] = *n++;
    }
    if (pos == 0) { _xform_buf[pos++] = 'i'; _xform_buf[pos++] = 'd'; }
    _xform_buf[pos] = '\0';
    return _xform_buf;
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
 * (macros defined in mlp-bin32-otto-trn-seq.c)
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
                                   int err, float lr, int members,
                                   int pool_n) {
    float tp  = (train_n > 0) ? (float)train_ok * 100.0f / (float)train_n : 0.0f;
    float ep  = (eval_n  > 0) ? (float)eval_ok  * 100.0f / (float)eval_n  : 0.0f;
    printf("\n============================================================\n");
    printf("REPORT train=%.2f%% (%d) eval=%.2f%% (%d)"
           " err=%d lr=%.4f time=%dms threads=%d",
           tp, train_n, ep, eval_n,
           err, (double)lr, elapsed_ms, threadN);
    if (members > 0) {
        /* eff = eval - lambda*(members-1): eval combined with the member-count
         * complexity penalty (DRAM inference cost scales with member count).
         * λ = aa.eff_lambda (--eff-lambda, default 0.02). */
        float eff = ep - aa.eff_lambda * (float)(members - 1);
        /* members=11/3363: SELECTED members / pool size AFTER filtering
         * (merge-ensemble passes n_blocks; trainers pass 0 = unknown).
         * run-research.sh parses members= into the status log. 2026-08-16. */
        if (pool_n > 0)
            printf(" members=%d/%d eff=%.2f", members, pool_n, (double)eff);
        else
            printf(" members=%d eff=%.2f", members, (double)eff);
    }
    printf("\n");
     printf("============================================================\n");
}

/* ═══════════════════════════════════════════════════════════════════
 * CONFUSION MATRIX — [true][pred] Tabelle (K×K), generisch
 * ═══════════════════════════════════════════════════════════════════
 * Can be used by any trainer.
 * is_final: 1 = Endausgabe, 0 = per-epoch
 * libtprint (2026-08-12): exact column alignment via the vendored table
 * library, same as --debug-class-voting. */
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

    printf("\n  ── Confusion Matrix %s%s%s ─────────────────────────────\n",
           is_final ? "(final" : "(per epoch",
           !is_final ? ", Ep " : "",
           !is_final ? "" : ")");
    if (!is_final) printf("  Ep %d\n", ep + 1);
    else (void)ep;

    TPrint *tp = tprint_create(stdout, TRUE, TRUE, 2, 2);
    tprint_set_double_fmt(tp, "%5.1f%%");
    tprint_column_add(tp, "true \\ pred", TPAlign_center, TPAlign_left);
    for (int ai = 0; ai < n_active; ai++)
        tprint_column_add(tp, ki_class_names[active_cols[ai]],
                          TPAlign_center, TPAlign_right);
    tprint_column_add(tp, "err%", TPAlign_center, TPAlign_right);

    for (int ri = 0; ri < n_active; ri++) {
        int r = active_cols[ri];
        int row_tot = 0, row_err = 0;
        for (int ci = 0; ci < n_active; ci++) {
            int cc = active_cols[ci];
            row_tot += cm[r][cc];
            if (cc != r) row_err += cm[r][cc];
        }
        if (row_tot == 0) continue;  /* only show rows with samples */
        tprint_data_add_str(tp, 0, ki_class_names[r]);
        for (int ci = 0; ci < n_active; ci++) {
            int cc = active_cols[ci];
            double col_pct = (double)cm[r][cc] * 100.0 / (double)row_tot;
            tprint_data_add_double(tp, ci + 1, col_pct);
        }
        double err_pct = (double)row_err * 100.0 / (double)row_tot;
        tprint_data_add_double(tp, n_active + 1, err_pct);
    }

    tprint_print(tp);
    tprint_free(tp);
    printf("\n");
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
