/*
 * otto-score-ifc/mlp-bin32-trn-w1-hebbian.c — Majority + Hebbian (DRAM-native)
 * ============================================================================
 *
 * Reference implementation: bitwise Hebbian training for MNIST.
 * Uses XNOR + MAJ3 + popcount — NO floating point, NO matmul.
 *
 * Note: This trainer does NOT converge well (~82% MNIST).
 * It oscillates around 80-82% without improving beyond 82% regardless of epochs.
 * Included as reference for comparison with Otto Score and AdamW baselines.
 *
 * Build:
 *   make mlp-bin32-trn-w1-hebbian-xnor.exe    (XNOR, PACKING=1, NC=196)
 *   make mlp-bin32-trn-w1-hebbian-xor.exe     (XOR,  PACKING=1, NC=196)
 *
 * Original: ki-w1/mlp-bin32-trn-w1-hebbian.c
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#define KI_COMMON_LOAD_INPUT
#include "ki-common.h"
#include "w0_random.h"

/* ═══════════════════════════════════════════════════════════════════════
 * PACKING CONFIG — choose via -DPACKING=N
 * ═══════════════════════════════════════════════════════════════════════
 *   NC = 784 × PACKING × 8 / 32 = 784 × PACKING / 4
 *   PACKING  |  NC  | px/Cont | Encoding
 *   ---------+------+---------+--------------------------------
 *      1     | 196  |   4     | p0|p1<<8|p2<<16|p3<<24
 *      2     | 392  |   2     | p0*0x0101 | (p1*0x0101)<<16
 *      4     | 784  |   1     | p0 * 0x01010101
 *      8     | 1568 |   1/2   | 2× uint32 je p0 * 0x01010101
 */

#if !defined(PACKING)
#  error "PACKING must be 1 (NC=196), 2 (NC=392), 4 (NC=784), or 8 (NC=1568)"
#elif PACKING == 1
#  define NC        196
#elif PACKING == 2
#  define NC        392
#elif PACKING == 4
#  define NC        784
#elif PACKING == 8
#  define NC        1568
#else
#  error "PACKING must be 1, 2, 4, or 8"
#endif

/* ── INVARIANT: NC deckt exakt alle 784 Pixel ab ───────────────────
 *     PACKING=1: 4 px/cont → NC × 4 / 1 = 784
 *     PACKING=2: 2 px/cont → NC × 4 / 2 = 784
 *     PACKING=4: 1 px/cont → NC × 4 / 4 = 784
 *     PACKING=8: ½px/cont  → NC / 2     = 784
 */
_Static_assert(
    PACKING == 8 ? (NC / 2 == 784) : (NC * 4 / PACKING == 784),
    "NC does not cover all 784 pixels (NC*4/PACKING != 784)"
);

/* ── Buffer sizes (safe maximum for all variants) ──────────────── */
#define BUF_MAJ   2048       /* majority_tree internal buffer (>= NC=1568) */
#define H0_BUF    4096       /* predict/MAE h0 buffer (for H up to 4096) */

/* ═══════════════════════════════════════════════════════════════════════
 * H0 MODE — choose via -DH0_XOR (default: XNOR)
 * ═══════════════════════════════════════════════════════════════════════
 *   Define      | match[c]           | Effect
 *   ------------+--------------------+-------------------------------
 *   -DH0_XOR    | in[c] ^ row[c]     | Majority over differences
 *   (default)   | ~(in[c] ^ row[c])  | Majority over agreements
 *
 * Hebbian compensates for the negation → identical accuracy.
 * XOR saves one NOT operation per container.
 */
/* MB function: XOR3 (Parity) or MAJ (Majority) */
#ifdef H0_XOR3
#  define MB_STR   "XOR3"
#  define MB_FUNC  xor3_tree
#else
#  ifdef DIAG_MAJ3
#    define MB_STR   "DIAG"
#  else
#    define MB_STR   "MAJ"
#  endif
#  define MB_FUNC  majority_tree
#endif

#ifdef H0_XOR
#  define H0_MODE_STR  "XOR"
#  define H0_FORMULA   "in ^ W0"
#else
#  define H0_MODE_STR  "XNOR"
#  define H0_FORMULA   "~(in ^ W0)"
#endif

#define BITS      32
#define K         10
#define INPUT_PX  784

static inline int popcnt(uint32_t v) { return __builtin_popcount(v); }

/* ── print uint32_t as 32-bit binary string ──────────────────────── */
static inline void print_binary(uint32_t v) {
    for (int i = 31; i >= 0; i--)
        putchar((v & (1U << (unsigned)i)) ? '1' : '0');
}

/* ── 3er-Majority + Majority-Tree via lib/maj3.h ─────────────────── */
#include "maj3.h"

/* ── 3er-XOR (reines Bit-Gatter: ^) ───────────────────────────────── */
static inline uint32_t xor3(uint32_t a, uint32_t b, uint32_t c) {
    return a ^ b ^ c;
}

/* ── XOR3 tree: n uint32_t → 1 (ONLY ^, no &|~) ─────────────────── */
#ifdef H0_XOR3
static uint32_t xor3_tree(const uint32_t *vals, int n) {
    uint32_t buf[BUF_MAJ];
    if (n > BUF_MAJ) n = BUF_MAJ;
    for (int i = 0; i < n; i++) buf[i] = vals[i];
    while (n > 1) {
        int n_out = 0;
        for (int i = 0; i < n; i += 3) {
            uint32_t a = buf[i];
            if (i + 2 < n)
                buf[n_out++] = xor3(a, buf[i+1], buf[i+2]);
            else if (i + 1 < n)
                buf[n_out++] = a ^ buf[i+1];
            else
                buf[n_out++] = a;
        }
        n = n_out;
    }
    return buf[0];
}
#endif /* H0_XOR3 */

/* ── H0 via Majority (NC×uint32_t → 1) ────────────────────────────── */
static void compute_h0(const uint32_t *in, const uint32_t *W0,
                       int H, uint32_t *out) {
#ifdef DIAG_MAJ3
    /* Diagonal version: MAJ3 over match[(h+c)%H][c] */
    for (int h = 0; h < H; h++) {
        uint32_t match[NC];
        for (int c = 0; c < NC; c++) {
            size_t h2 = (size_t)((h + c) % H);
            uint32_t w = W0[h2 * (size_t)NC + (size_t)c];
#ifdef H0_XOR
            match[c] = in[c] ^ w;
#else
            match[c] = ~(in[c] ^ w);
#endif
        }
        out[h] = MB_FUNC(match, NC);
    }
#else
    /* Horizontale Version: match[h][c] */
    for (int h = 0; h < H; h++) {
        uint32_t match[NC];
        const uint32_t *row = W0 + (size_t)h * NC;
#ifdef H0_XOR
        for (int c = 0; c < NC; c++)
            match[c] = in[c] ^ row[c];
#else
        for (int c = 0; c < NC; c++)
            match[c] = ~(in[c] ^ row[c]);
#endif
        out[h] = MB_FUNC(match, NC);
    }
#endif
}

/* ── XNOR-Score (32-bit) — IMMER XNOR, egal welcher H0-Modus ─────── */
static inline double xnor_score(uint32_t w1, uint32_t h0) {
    return (double)popcnt(~(w1 ^ h0)) * (1.0 / 32.0) - 0.5;
}

/* ── H1 Summe über H Neuronen ─────────────────────────────────────── */
static inline double h1_total(const uint32_t *h0, int H,
                              const uint32_t *w1_row) {
    double sum = 0.0;
    for (int c = 0; c < H; c++)
        sum += xnor_score(w1_row[c], h0[c]);
    return sum;
}

/* ── Prediction ──────────────────────────────────────────────────── */
static inline int predict(const uint32_t *in, const uint32_t *W0,
                          const uint32_t *W1, int H) {
    uint32_t h0[H0_BUF];
    compute_h0(in, W0, H, h0);
    double best = -1e9;
    int best_k = 0;
    for (int k = 0; k < K; k++) {
        double s = h1_total(h0, H, W1 + (size_t)k * (size_t)H);
        if (s > best) { best = s; best_k = k; }
    }
    return best_k;
}

/* ── Weight Export (uint32 binary format) ───────────────────────────
 * Saves best W0 + W1 as binary for inference.
 * Format matches mlp-bin32-ifc.c expectations.
 */
static void export_bin_weights(const uint32_t *W0, const uint32_t *W1,
                                int H_f, const char *dir) {
    char parent[512];
    strncpy(parent, dir, sizeof(parent) - 1);
    parent[sizeof(parent) - 1] = '\0';
    char *slash = strrchr(parent, '/');
    if (slash) {
        *slash = '\0';
        if (parent[0] != '\0') {
            if (mkdir(parent, 0755) != 0 && errno != EEXIST) {
                fprintf(stderr, "[ERROR] Cannot create '%s': %s\n",
                        parent, strerror(errno));
                return;
            }
        }
    }
    if (mkdir(dir, 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "[ERROR] Cannot create '%s': %s\n", dir, strerror(errno));
        return;
    }
    char path[512];
    snprintf(path, sizeof(path), "%s/weights.meta", dir);
    FILE *f = fopen(path, "w");
    if (f) {
        fprintf(f, "2\n%d %d\n%d %d\n", H_f, NC, K, H_f);
        fclose(f);
    }
    snprintf(path, sizeof(path), "%s/W0.bin", dir);
    f = fopen(path, "wb");
    if (f) {
        fwrite(W0, sizeof(uint32_t), (size_t)H_f * (size_t)NC, f);
        fclose(f);
    }
    snprintf(path, sizeof(path), "%s/W1.bin", dir);
    f = fopen(path, "wb");
    if (f) {
        fwrite(W1, sizeof(uint32_t), (size_t)K * (size_t)H_f, f);
        fclose(f);
    }
    printf("══╡ EXPORT ╞═════════════════════════════════════════════════\n");
    printf("  Saved bin32 model to %s/  (W0=%zu KB + W1=%zu KB)\n",
           dir,
           (size_t)H_f * (size_t)NC * sizeof(uint32_t) / 1024,
           (size_t)K * (size_t)H_f * sizeof(uint32_t) / 1024);
    fflush(stdout);
}

/* ── Accuracy ────────────────────────────────────────────────────── */
static inline float accuracy(const uint32_t *X, const uint8_t *Y, int N,
                             const uint32_t *W0, const uint32_t *W1,
                             int H) {
    int ok = 0;
    #pragma omp parallel for reduction(+:ok) schedule(static)
    for (int s = 0; s < N; s++)
        if (predict(X + (size_t)s * NC, W0, W1, H) == (int)Y[s]) ok++;
    return (float)ok * 100.0f / (float)N;
}

/* ═══════════════════════════════════════════════════════════════════
 *  HEBBIAN TRAINING — identical for all variants
 * ═══════════════════════════════════════════════════════════════════ */
static void hebbian_update(uint32_t *W1, const uint32_t *X,
                           const uint8_t *Y, int N,
                           const uint32_t *W0, int H,
                           unsigned int rng_seed,
                           int *flips_out,
                           int hebbian_pct) {
    size_t csize = (size_t)K * (size_t)H * (size_t)BITS;
    int *counter = (int *)calloc(csize, sizeof(int));
    int class_cnt[K] = {0};

    int *idx = (int *)malloc((size_t)N * sizeof(int));
    for (int i = 0; i < N; i++) idx[i] = i;
    srand(rng_seed);
    for (int i = N - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }

    #pragma omp parallel
    {
        int *loc_ctr = (int *)calloc(csize, sizeof(int));
        int loc_cc[K] = {0};
        uint32_t *h0_buf = (uint32_t *)malloc((size_t)H * sizeof(uint32_t));
        int alloc_fail = (!loc_ctr || !h0_buf);

        #pragma omp for schedule(static)
        for (int si = 0; si < N; si++) {
            if (alloc_fail) continue;
            int s = idx[si];
            int lb = (int)Y[s];
            loc_cc[lb]++;

            compute_h0(X + (size_t)s * (size_t)NC, W0, H, h0_buf);

            for (int h = 0; h < H; h++) {
                uint32_t hv = h0_buf[h];
                if (hv == 0) continue;
                size_t base = ((size_t)lb * (size_t)H + (size_t)h) * (size_t)BITS;
                for (int b = 0; b < BITS; b++) {
                    if (hv & (1U << (unsigned int)b))
                        loc_ctr[base + (size_t)b]++;
                }
            }
        }

        #pragma omp critical
        {
            for (size_t i = 0; i < csize; i++)
                counter[i] += loc_ctr[i];
            for (int k = 0; k < K; k++)
                class_cnt[k] += loc_cc[k];
        }
        free(loc_ctr);
        free(h0_buf);
    }

    uint32_t *old_W1 = NULL;
    if (flips_out) {
        old_W1 = (uint32_t *)malloc((size_t)K * (size_t)H * sizeof(uint32_t));
        memcpy(old_W1, W1, (size_t)K * (size_t)H * sizeof(uint32_t));
    }

    for (int k = 0; k < K; k++) {
        int th = (class_cnt[k] * hebbian_pct + 50) / 100;
        if (th < 1) th = 1;
        for (int h = 0; h < H; h++) {
            uint32_t w1v = 0;
            size_t base = ((size_t)k * (size_t)H + (size_t)h) * (size_t)BITS;
            for (int b = 0; b < BITS; b++) {
                if (counter[base + (size_t)b] > th)
                    w1v |= (1U << (unsigned int)b);
            }
            W1[(size_t)k * (size_t)H + (size_t)h] = w1v;
        }
    }

    if (flips_out) {
        int total = 0;
        size_t w1_cnt = (size_t)K * (size_t)H;
        for (size_t i = 0; i < w1_cnt; i++)
            total += popcnt(old_W1[i] ^ W1[i]);
        *flips_out = total;
        free(old_W1);
    }

    free(counter);
    free(idx);
}

/* ── INVARIANT: verify load_input covers all 784 pixels ────────────
 *     pixels_per_container = 4 / PACKING  (for PACKING=1,2,4)
 *     pixels_per_container = 1/2          (for PACKING=8)
 */
static inline void validate_pixel_coverage(void) {
    int covered = (PACKING == 8) ? (NC / 2) : (NC * 4 / PACKING);
    if (covered != INPUT_PX) {
        fprintf(stderr, "[FATAL] load_input: deckt %d/784 Pixel ab (PACKING=%d NC=%d)\n",
                covered, PACKING, NC);
        exit(1);
    }
}

/* ═══════════════════════════════════════════════════════════════════
 *  INPUT LOADING — PACKING-dependent encoding
 * ═══════════════════════════════════════════════════════════════════
 *  Takes raw uint8_t pixels [0,255], no float round-trip.
 *
 *  PACKING=1 (NC=196): 4 px/Cont im Format p0|p1<<8|p2<<16|p3<<24
 *  PACKING=2 (NC=392): 2 px/Cont, 8→16 Expansion via *0x0101
 *  PACKING=4 (NC=784): 1 px/Cont, 8→32 Expansion via *0x01010101
 *  PACKING=8 (NC=1568): 1 px → 2 Cont, je *0x01010101
 */
static uint32_t *load_input(const uint8_t *X_raw, int n_samples) {
    validate_pixel_coverage();
    uint32_t *Xb = ki_xmalloc((size_t)n_samples * (size_t)NC * sizeof(uint32_t));
#if PACKING == 1
    for (int s = 0; s < n_samples; s++) {
        uint32_t *row = Xb + (size_t)s * NC;
        for (int c = 0; c < NC; c++) {
            uint32_t val = 0;
            for (int k = 0; k < 4; k++) {
                size_t p = (size_t)s * (size_t)INPUT_PX + (size_t)c * 4 + (size_t)k;
                val |= ((uint32_t)X_raw[p] & 0xFFU) << (unsigned)(k * 8);
            }
            row[c] = val;
        }
    }
#elif PACKING == 2
    for (int s = 0; s < n_samples; s++) {
        uint32_t *row = Xb + (size_t)s * NC;
        for (int c = 0; c < NC; c++) {
            uint32_t val = 0;
            for (int k = 0; k < 2; k++) {
                size_t p = (size_t)s * (size_t)INPUT_PX + (size_t)c * 2 + (size_t)k;
                uint32_t byte = X_raw[p] & 0xFFU;
                val |= (byte * 0x0101U) << (unsigned)(k * 16);
            }
            row[c] = val;
        }
    }
#elif PACKING == 4
    for (int s = 0; s < n_samples; s++) {
        uint32_t *row = Xb + (size_t)s * NC;
        for (size_t p = 0; p < INPUT_PX; p++) {
            size_t off = (size_t)s * (size_t)INPUT_PX + p;
            row[p] = ((uint32_t)X_raw[off] & 0xFFU) * 0x01010101U;
        }
    }
#elif PACKING == 8
    for (int s = 0; s < n_samples; s++) {
        uint32_t *row = Xb + (size_t)s * NC;
        for (size_t p = 0; p < INPUT_PX; p++) {
            size_t off = (size_t)s * (size_t)INPUT_PX + p;
            uint32_t val = ((uint32_t)X_raw[off] & 0xFFU) * 0x01010101U;
            row[p * 2]     = val;
            row[p * 2 + 1] = val;
        }
    }
#endif
    return Xb;
}

/* ═══════════════════════════════════════════════════════════════════
 *  MAIN
 * ═══════════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[]) {
    ki_Args a = ki_args_defaults();
    ki_parse_args(argc, argv, &a);
    omp_set_num_threads(a.threadN);

    int H = a.hidden, E = a.epochs, Ntrn = a.trainN, Nevl = a.evalN;
    unsigned seed = a.seed;

    if (a.dry_run) {
        ki_SetupInfo setup = {
            .title = "Hebbian-" H0_MODE_STR "-" MB_STR,
            .H = H, .epochs = E, .bits_per_cont = BITS,
            .pixel_bits = 8, .seed = seed,
            .N = Ntrn, .ne = Nevl,
            .n_threads = a.threadN, .sizeof_bn = 4,
            .nc = NC, .C = NC,
            .px = 784,
            .input_bit  = (size_t)NC * BITS,
            .hidden_bit = (size_t)H * BITS,
            .output_bit = (size_t)K * BITS,
            .w0_bit = (size_t)H * (size_t)NC * BITS,
            .w1_bit = (size_t)K * (size_t)H * BITS,
        };
        ki_dry_run_show(&setup);
        printf("  %-18s %-16s  %s\n", "h0", H0_FORMULA, H0_MODE_STR " + " MB_STR " -> 1");
        printf("  %-18s %-16s  %s\n", "w1-score", "~(w1 ^ h0)", "XNOR popcnt (immer)");
        printf("  %-18s %-16s  co-occurrence, threshold=%d%%\n",
               "train", "hebbian", a.hebbian_pct);
        printf("  %-18s PACKING=%d  NC=%d  %s\n", "input", PACKING, NC,
               PACKING == 1 ? "4px packed p0|p1<<8|p2<<16|p3<<24" :
               PACKING == 2 ? "2px 0x0101 expansion" :
               PACKING == 4 ? "1px 0x01010101 byte-repeat" :
               "1px -> 2x 0x01010101");
        return 1;
    }

    ki_MNISTData md;
    if (ki_mnist_read(&md) != 0) return 1;
    if (Ntrn + Nevl > md.num_images) Nevl = md.num_images - Ntrn;
    if (Nevl < 0) Nevl = 0;

    uint32_t *Xtrn = load_input(md.X_raw, Ntrn);
    uint32_t *Xevl = load_input(md.X_raw + (size_t)Ntrn * (size_t)INPUT_PX, Nevl);
    uint8_t *Ytrn = md.y;
    uint8_t *Yevl = md.y + Ntrn;

    if (a.debug_detail) {
        /* Xtrn als Byte-Array — jedes uint32 = 4 uint8_t */
        uint8_t *Xtrn8 = (uint8_t *)Xtrn;
        /* find first non-zero container */
        int first_nz = 0;
        for (int ci = 0; ci < NC; ci++)
            if (Xtrn[ci] != 0) { first_nz = ci; break; }
        printf("  [MNIST] Xtrn (sample 0, ab cont[%d], als bin8):\n", first_nz);
        for (int ci = first_nz; ci < first_nz + 4 && ci < NC; ci++) {
            int b0 = ci * 4;
            printf("    cont[%3d] = %02x %02x %02x %02x  ",
                   ci, Xtrn8[b0+3], Xtrn8[b0+2], Xtrn8[b0+1], Xtrn8[b0+0]);
            print_binary(Xtrn[ci]);
            printf("\n");
        }
        fflush(stdout);
    }

    struct timeval tv_setup;
    gettimeofday(&tv_setup, NULL);

    uint32_t *W0 = ki_xmalloc((size_t)H * (size_t)NC * sizeof(uint32_t));
    uint32_t *W1 = ki_xmalloc((size_t)K * (size_t)H * sizeof(uint32_t));
    w0_srandom(seed);
    for (size_t i = 0; i < (size_t)H * (size_t)NC; i++)
        W0[i] = w0_random();
    memset(W1, 0, (size_t)K * (size_t)H * sizeof(uint32_t));

    ki_SetupInfo setup = {
        .title = "Hebbian-" H0_MODE_STR,
        .H = H, .epochs = E, .bits_per_cont = BITS,
        .pixel_bits = 8, .seed = seed,
        .N = Ntrn, .ne = Nevl,
        .n_threads = a.threadN, .sizeof_bn = 4,
        .nc = NC, .C = NC,
        .px = 784,
        .input_bit  = (size_t)NC * BITS,
        .hidden_bit = (size_t)H * BITS,
        .output_bit = (size_t)K * BITS,
        .w0_bit = (size_t)H * (size_t)NC * BITS,
        .w1_bit = (size_t)K * (size_t)H * BITS,
        .no_close = 1,
    };
    printf("══╡ HEBBIAN-%s-%s  NC=%-4d  H=%-4d  Ep=%-2d ╞══\n",
           H0_MODE_STR, MB_STR, NC, H, E);
    ki_setup_show(&setup);
    printf("══╡ TRAINING ╞══════════════════════════════════════════════════════════\n");

    int evl_n = (Nevl > 2000) ? 2000 : Nevl;
    int mae_n = (Ntrn < 5000) ? Ntrn : 5000;
    int total_bits = K * H * BITS;
    float best_evl = 0.0f;
    uint32_t *W1_best = ki_xmalloc((size_t)K * (size_t)H * sizeof(uint32_t));
    uint32_t *W0_best = ki_xmalloc((size_t)H * (size_t)NC * sizeof(uint32_t));
    memcpy(W1_best, W1, (size_t)K * (size_t)H * sizeof(uint32_t));

    struct timeval tv_trn;
    gettimeofday(&tv_trn, NULL);

    for (int ep = 0; ep < E; ep++) {
        w0_srandom(seed + (unsigned int)ep);
        for (size_t i = 0; i < (size_t)H * (size_t)NC; i++)
            W0[i] = w0_random();

        int hebbian_flips = 0;
        hebbian_update(W1, Xtrn, Ytrn, Ntrn, W0, H,
                       seed + (unsigned int)ep, &hebbian_flips,
                       a.hebbian_pct);

        float cur_trn = accuracy(Xtrn, Ytrn, mae_n, W0, W1, H);
        float cur_evl = accuracy(Xevl, Yevl, evl_n, W0, W1, H);
        if (cur_evl > best_evl) {
            best_evl = cur_evl;
            memcpy(W1_best, W1, (size_t)K * (size_t)H * sizeof(uint32_t));
            memcpy(W0_best, W0, (size_t)H * (size_t)NC * sizeof(uint32_t));
        }

        double sum_ae = 0.0;
        int ae_cnt = 0;
        for (int s = 0; s < mae_n; s++) {
            uint32_t h0[H0_BUF];
            compute_h0(Xtrn + (size_t)s * (size_t)NC, W0, H, h0);
            for (int k0 = 0; k0 < K; k0++) {
                double h1 = h1_total(h0, H, W1 + (size_t)k0 * (size_t)H);
                double t = (k0 == (int)Ytrn[s]) ? 0.5 * (double)H
                                                : -0.5 * (double)H;
                sum_ae += fabs(t - h1);
                ae_cnt++;
            }
        }
        double mae = (ae_cnt > 0) ? sum_ae / (double)ae_cnt : 0.0;

        int flips_pct = (total_bits > 0) ? (hebbian_flips * 100 + total_bits/2) / total_bits : 0;
        unsigned int ep_seed = seed + (unsigned int)ep;
        printf("Ep %3d  seed=%5u  trn=%5.1f%% (%d)  evl=%5.1f%% (%d)  best=%5.1f%%  flips=%5d/%d (%2d%%)  MAE=%.4f  HEB=%d\n",
               ep + 1, ep_seed, cur_trn, mae_n, cur_evl, evl_n, best_evl,
               hebbian_flips, total_bits, flips_pct, mae, a.hebbian_pct);
        fflush(stdout);
    }

    memcpy(W1, W1_best, (size_t)K * (size_t)H * sizeof(uint32_t));
    memcpy(W0, W0_best, (size_t)H * (size_t)NC * sizeof(uint32_t));

    struct timeval tv_end;
    gettimeofday(&tv_end, NULL);
    long trn_ms = (tv_end.tv_sec - tv_trn.tv_sec) * 1000L
                + (tv_end.tv_usec - tv_trn.tv_usec) / 1000L;

    float fin_trn = accuracy(Xtrn, Ytrn, Ntrn, W0, W1, H);
    float fin_evl = accuracy(Xevl, Yevl, Nevl, W0, W1, H);
    int train_ok = (int)(fin_trn * (float)Ntrn / 100.0f);
    int eval_ok  = (int)(fin_evl * (float)Nevl / 100.0f);
    ki_report_show(train_ok, Ntrn, eval_ok, Nevl, trn_ms, a.threadN, 0, 0.0f);

    /* ── Export best weights ─────────────────────────────────── */
    if (a.out[0] != '\0')
        export_bin_weights(W0, W1, H, a.out);
    else {
        char def_dir[256];
        snprintf(def_dir, sizeof(def_dir), KI_MODEL_DIR "/hebbian-h%d", H);
        export_bin_weights(W0, W1, H, def_dir);
    }

    ki_mnist_free(&md);
    free(W0); free(W1); free(W1_best); free(W0_best);
    free(Xtrn); free(Xevl);
    return 0;
}
