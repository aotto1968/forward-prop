/*
 * otto-score-ifc/mnist/ki-config.h — MNIST static configuration
 * =============================================================
 * Always included FIRST (before ki-encoding.h / ki-common.h) so that
 * dataset-specific macros like KI_SWEEP_PERFORMANCE_ENCODING are visible
 * to the shared headers. Symlinked into mnist-1/ki-config.h.
 *
 * Contains ONLY static configuration (macros). Dataset-specific logic
 * (loader, data structs) stays in ki-local.h.
 */
#ifndef KI_CONFIG_H
#define KI_CONFIG_H

/* ═══════════════════════════════════════════════════════════════════════
 * CONSTANTS — MNIST
 * ═══════════════════════════════════════════════════════════════════════ */

#define KI_DATASET_ID           0       /* unique for cache key */
#define KI_DATASET_NAME         "MNIST"
/* Prepped-cache write threshold: buffers larger than this are never written
 * to data/prepped/. MNIST stays small normally (~681 MB), but wide encoding/
 * channel sweeps reach ~18 GB per buffer — never cache those. Adjust per
 * dataset as needed. See: plans/plan-2026-07-31-sweep-xform-memory.md */
#define KI_PREPPED_MAX_BYTES    (2ULL * 1024ULL * 1024ULL * 1024ULL)
#define KI_PX                   784
#ifndef KI_BIT_WIDTH
#define KI_BIT_WIDTH            8      /* bits per pixel (8/16/24/32) */
#endif
#define KI_PX_PER_CONT_W        (32 / KI_BIT_WIDTH)  /* 4 bei 8bit, 2 bei 16bit, 1 bei 32bit */
#define KI_ROWS                 28
#define KI_COLS                 28
#define KI_NCLASSES             10
#define KI_DEFAULT_LR           0.05f   /* → step = 0.05 × 131072 = 6554 */
#define KI_DEFAULT_STEP_POWER   0.1f    /* higher yields smaller trn */
#define KI_DEFAULT_STEP_MODE    STEP_COS_TIME
#define KI_DEFAULT_BATCH_N      64      /* optimum */
#define KI_COLORS               1       /* MNIST is grayscale */
#define KI_DEFAULT_COLOR        (1<<COLOR_MNIST)  /* MNIST: single grayscale block */
#define KI_NC                   (KI_PX / KI_PX_PER_CONT_W)  /* Container per image: 784/4=196 bei 8bit */
#define KI_NC_TOTAL             (KI_NC * KI_COLORS)
#define KI_PACK                 KI_PX_PER_CONT_W


#ifndef NC
#define NC  KI_NC
#endif

#ifndef OT_PRECISION
#define OT_PRECISION 17
#endif

/* ── Sweep encoding preset (dataset-specific, resolved by
 *    ki_encoding_alias_expand "sweep-performance") ──
 * MNIST top-6 encodings from the hierarchy analysis (2026-08-01):
 * log8/cbrt8/exp8/inv-exp8/gamma8/sqrt8 dominate the top-50% archive list. */
#ifndef KI_SWEEP_PERFORMANCE_ENCODING
#define KI_SWEEP_PERFORMANCE_ENCODING "cbrt8,exp8,inv-exp8,gamma8,log8,sqrt8"
#endif

#define KI_SWEEP_PERFORMANCE_XFORM "avg2,avg3,avg4,colswap-1-4,colswap-2-4,colswap-3-4,dflip1,dflip2,hflip,id,rot22,rot45,rot67,rot90,shuffle,spiral,vflip"

#endif /* KI_CONFIG_H */
