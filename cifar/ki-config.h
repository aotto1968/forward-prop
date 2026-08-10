/*
 * otto-score-ifc/cifar/ki-config.h — CIFAR-10 static configuration
 * =================================================================
 * Always included FIRST (before ki-encoding.h / ki-common.h) so that
 * dataset-specific macros like KI_SWEEP_PERFORMANCE_ENCODING are visible
 * to the shared headers. Symlinked into cifar-1/ki-config.h.
 *
 * Contains ONLY static configuration (macros). Dataset-specific logic
 * (loader, data structs) stays in ki-local.h.
 */
#ifndef KI_CONFIG_H
#define KI_CONFIG_H

/* ═══════════════════════════════════════════════════════════════════════
 * CONSTANTS — CIFAR-10
 * ═══════════════════════════════════════════════════════════════════════ */

#define KI_DATASET_ID             1       /* unique for cache key (overridable) */
#define KI_DATASET_NAME           "CIFAR-10"
/* Prepped-cache write threshold: buffers larger than this are never written
 * to data/prepped/. CIFAR --channel sweep produces ~23 GB buffers (60000 ×
 * 96512 × 4B) — never cache those or the disk fills with ~394 GB for data
 * that is loaded once and freed again. Adjust per dataset as needed.
 * See: plans/plan-2026-07-31-sweep-xform-memory.md */
#define KI_PREPPED_MAX_BYTES      (2ULL * 1024ULL * 1024ULL * 1024ULL)
#ifndef KI_BIT_WIDTH
#define KI_BIT_WIDTH              8       /* bits per pixel (8/16/24/32) */
#endif
#define KI_PX_PER_CONT_W          (32 / KI_BIT_WIDTH)  /* 4 bei 8bit, 2 bei 16bit, 1 bei 32bit */
#define KI_ROWS                   32
#define KI_COLS                   32
#define KI_PX                     3072    /* 32 × 32 × 3 = 3072 pixels per image */
#define KI_NCLASSES               10
#define KI_DEFAULT_LR             0.01f   /* → step = 0.005 × 131072 = 655 */
#define KI_DEFAULT_STEP_POWER     7.0f    /* higher yields smaller trn */
#define KI_DEFAULT_STEP_MODE      STEP_COS_TIME
#define KI_DEFAULT_BATCH_N        128     /* optimum */
#define KI_DEFAULT_ENSEMBLE_SEED  ENS_SEED_ONCE
#define KI_COLORS                 3       /* R, G, B as independent samples, each packed 4px/cont */
#define KI_DEFAULT_COLOR          ((1<<COLOR_R)|(1<<COLOR_G)|(1<<COLOR_B))  /* CIFAR default: raw R+G+B (bits 1,2,3) */
#define KI_NC                     (KI_PX / KI_COLORS / KI_PX_PER_CONT_W)  /* Container per color: 1024/4=256 bei 8bit */
#define KI_NC_TOTAL               (KI_NC * KI_COLORS)
#define KI_PACK                   KI_PX_PER_CONT_W


#ifndef NC
#define NC  KI_NC
#endif

#ifndef OT_PRECISION
#define OT_PRECISION              17
#endif

/* For mlp-flt32-trn-*-adam.c (old trainer) */
#ifndef KI_BITS_PER_CONT
#define KI_BITS_PER_CONT          32
#endif
#ifndef KI_MODEL_DIR
#define KI_MODEL_DIR              "models"
#endif

/* ── Sweep encoding preset (dataset-specific, resolved by
 *    ki_encoding_alias_expand "sweep-performance") ──
 * CIFAR top-6 encodings from the gt 20% analysis (2026-08-01):
 * down8 1391, mid8 1381, cbrt8 1371, lin8 1363, sqrt8 1363, gamma8 1353. */
#define KI_SWEEP_PERFORMANCE_ENCODING "cbrt,down,exp,gamma,inv-exp,lin,log,mid,raw,sig,sqrt,tri,up"
#define KI_SWEEP_PERFORMANCE_XFORM "avg2,avg3,avg4,colswap-1-4,colswap-2-4,colswap-3-4,dflip1,dflip2,hflip,id,rot22,rot45,rot67,rot90,shuffle,spiral,vflip"
#define KI_SWEEP_PERFORMANCE_CHANNEL "AL,AM,AP,B,bin,BL,BM,BP,C,CL,CM,CP,dir,dog,edge,G,GB,H,lbp,lbp-gb,lbp-rb,lbp-rg,R,range,RB,RG,S,var,Y"

#endif /* KI_CONFIG_H */
