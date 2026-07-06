/*
 * otto-score-ifc/mnist/mlp-flt32-adam-trn.c — Float32 AdamW W1-Only (W0 Frozen)
 * =============================================================================
 *
 * Self-contained AdamW trainer with multi-member ensemble.
 * Each enc_array entry becomes a "member" with independent W0+W1.
 * Forward: matmul + LReLU(0.05) | Loss: MSE ±1 (all 10 classes)
 * Optimizer: AdamW per member | Schedule: warmup + cosine decay
 * Inference via --import: loads weights and evaluates.
 * Single source for both MNIST and CIFAR (via ki-local.h dataset aliases).
 */
#include "ki-common.h"
#include "ki-adamw.h"

ki_Args aa = {
    .seed_splitmix = 1,
};
#define N_CLASSES KI_NCLASSES
#define ADAM_WD 1e-4f
#define ADAM_MAX_MEM 128

/* ═══════════════════════════════════════════════════════════════════════
 * FORWARD — single member
 * ═══════════════════════════════════════════════════════════════════════ */
static void forward(const ki_LinearLayer *l0, const ki_LinearLayer *l1,
                     const float *x, float *h0_out, float *output, int batchN) {
    ki_linear_forward(l0, x, h0_out, batchN);
    ki_leaky_relu(h0_out, batchN * l0->out_features);
    ki_linear_forward(l1, h0_out, output, batchN);
}

/* ═══════════════════════════════════════════════════════════════════════
 * ACCURACY — multi-member (sum scores, argmax)
 * ═══════════════════════════════════════════════════════════════════════ */
static float accuracy_multi(const float *X, const uint8_t *Y, int N,
                             const ki_LinearLayer *l0_arr, const ki_LinearLayer *l1_arr,
                             const int *offs, int n_mem, int stride) {
    int ok = 0;
    #pragma omp parallel for reduction(+:ok) schedule(static)
    for (int s = 0; s < N; s++) {
        float total[N_CLASSES];
        for (int k = 0; k < N_CLASSES; k++) total[k] = 0;
        for (int m = 0; m < n_mem; m++) {
            const float *slice = X + (size_t)s * (size_t)stride + offs[m];
            float h0_buf[4096], out[N_CLASSES];
            forward(&l0_arr[m], &l1_arr[m], slice, h0_buf, out, 1);
            for (int k = 0; k < N_CLASSES; k++) total[k] += out[k];
        }
        int pred = 0;
        for (int k = 1; k < N_CLASSES; k++)
            if (total[k] > total[pred]) pred = k;
        if (pred == (int)Y[s]) ok++;
    }
    return 100.0f * (float)ok / (float)N;
}

/* ═══════════════════════════════════════════════════════════════════════
 * EXPORT — per-member weights
 * ═══════════════════════════════════════════════════════════════════════ */
static void export_member(const float *W0, const float *W1,
                           int H, int nc, const char *dir, int mid) {
    char cmd[512], path[1024];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", dir);
    if (system(cmd) != 0) return;
    snprintf(path, sizeof(path), "%s/weights-%d.meta", dir, mid);
    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "%d\n%d %d\n%d %d\n", 2, H, nc, N_CLASSES, H);
    fprintf(f, "lr=%.4f wd=%.6f\n", 0.005, (double)ADAM_WD);
    fclose(f);
    snprintf(path, sizeof(path), "%s/W0-%d.bin", dir, mid);
    f = fopen(path, "wb");
    fwrite(W0, sizeof(float), (size_t)H * (size_t)nc, f);
    fclose(f);
    snprintf(path, sizeof(path), "%s/W1-%d.bin", dir, mid);
    f = fopen(path, "wb");
    fwrite(W1, sizeof(float), (size_t)N_CLASSES * (size_t)H, f);
    fclose(f);
}

/* ═══════════════════════════════════════════════════════════════════════
 * PACKING — MNIST (4px averaged) and CIFAR (channel-aware blocks)
 * ═══════════════════════════════════════════════════════════════════════ */
static float *ki_pack_mnist_float(const uint8_t *X_raw, int n_samples, int nc) {
    size_t total = (size_t)n_samples * (size_t)nc;
    float *out = (float *)malloc(total * sizeof(float));
    if (!out) return NULL;
    for (int s = 0; s < n_samples; s++) {
        float *row = out + (size_t)s * (size_t)nc;
        for (int c = 0; c < nc; c++) {
            int sum = 0;
            for (int k = 0; k < 4; k++)
                sum += (int)X_raw[(size_t)s * KI_PX + (size_t)c * 4 + (size_t)k];
            row[c] = (float)sum / (4.0f * 127.5f) - 1.0f;
        }
    }
    return out;
}

static float *ki_pack_blocks_float(const uint8_t *X_raw, int n_samples,
                                    const int *block_order, int n_blocks,
                                    int packed) {
    if (n_blocks <= 0 || !block_order) return NULL;
    int blk_nc = packed ? KI_NC : (KI_PX / 3);
    int total_nc = n_blocks * blk_nc;
    size_t total = (size_t)n_samples * (size_t)total_nc;
    float *packed_out = (float *)malloc(total * sizeof(float));
    if (!packed_out) return NULL;
    for (int s = 0; s < n_samples; s++) {
        size_t base = (size_t)s * (size_t)KI_PX;
        float *row = packed_out + (size_t)s * (size_t)total_nc;
        uint8_t all_blocks[16][1024];
        for (int p = 0; p < 1024; p++) {
            int r = (int)X_raw[base + (size_t)p];
            int g = (int)X_raw[base + 1024 + (size_t)p];
            int b = (int)X_raw[base + 2048 + (size_t)p];
            uint8_t blk[COLOR_NB];
            ki_blocks_from_rgb(r, g, b, blk);
            for (int bi = 0; bi < n_blocks && bi < 16; bi++)
                if (block_order[bi] >= 0 && block_order[bi] < COLOR_NB)
                    all_blocks[bi][p] = blk[block_order[bi]];
        }
        for (int bi = 0; bi < n_blocks && bi < 16; bi++) {
            float *dst = row + (size_t)bi * (size_t)blk_nc;
            if (packed) {
                for (int c = 0; c < blk_nc; c++) {
                    int sum = 0;
                    for (int k = 0; k < 4; k++)
                        sum += (int)all_blocks[bi][(size_t)c * 4 + (size_t)k];
                    dst[c] = (float)sum / (4.0f * 127.5f) - 1.0f;
                }
            } else {
                for (int c = 0; c < blk_nc; c++)
                    dst[c] = (float)all_blocks[bi][c] / 127.5f - 1.0f;
            }
        }
    }
    return packed_out;
}

/* ═══════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[]) {
    aa.hidden = 64; aa.epochs = 1; aa.batchN = 64; aa.trainN = 50000; aa.evalN = 10000;
    aa.seed = 42; aa.lr = 0.005f; aa.lr_min = 0.1f; aa.threadN = 8; aa.warmup_epochs = 2;
    aa.channel = KI_DEFAULT_COLOR; aa.packedB = 1; aa.ensembleN = 1;
    ki_parse_args(argc, argv);

    /* Sollen Members oder eine Matrix verwendet werden?
     * Default (kein --channels, kein --encoding): single matrix (flat)
     * --channels r,g,b  (ohne flat) → Members pro Block
     * --channels flat              → single matrix
     * --encoding latest            → Members (mixed types) */
    int use_members = 0;
    if (aa.enc_count > 1) {
        /* --encoding mit verschiedenen Typen → Members */
        int8_t first = aa.enc_array[0].type;
        for (int i = 1; i < aa.enc_count; i++)
            if (aa.enc_array[i].type != first) { use_members = 1; break; }
    }
    if (!use_members && aa.channel_explicit && !aa.debug_flat)
        use_members = 1;  /* --channels ohne flat → Members */
    omp_set_num_threads(aa.threadN);

    /* ── Single matrix or multi-member? ────────────────────────── */
    int n_mem = 0;          /* Anzahl Members (für Training) */
    int n_blocks = 0;       /* Anzahl gepackter Blöcke im Buffer */
    int mem_nc[ADAM_MAX_MEM], mem_off[ADAM_MAX_MEM];
    int block_order[ADAM_MAX_MEM];
    int total_stride = 0;
    int blk_nc = aa.packedB ? KI_NC : (KI_PX / 3);

    if (!use_members) {
        /* Single-matrix: alle Blöcke in EINER W0 kombiniert */
        n_mem = 1;
        int blk_idx = 0;
        if (aa.enc_count > 0) {
            for (int i = 0; i < aa.enc_count && i < ADAM_MAX_MEM; i++)
                block_order[blk_idx++] = (int)aa.enc_array[i].color;
        } else {
            int mask = aa.channel;
            if (mask < 0) mask = KI_DEFAULT_COLOR;
            for (int b = 0; b < COLOR_NB && blk_idx < ADAM_MAX_MEM; b++)
                if (mask & (1 << b)) block_order[blk_idx++] = b;
            if (blk_idx == 0) {
                block_order[0] = (KI_COLORS <= 1) ? COLOR_MNIST : COLOR_R;
                if (KI_COLORS > 1) { block_order[1] = COLOR_G; block_order[2] = COLOR_B; blk_idx = 3; }
                else blk_idx = 1;
            }
        }
        n_blocks = blk_idx;
        mem_nc[0] = n_blocks * blk_nc;
        mem_off[0] = 0;
        total_stride = mem_nc[0];
    } else {
        /* Multi-member: eigener (W0,W1) pro Block/Encoding */
        n_mem = aa.enc_count > 0 ? aa.enc_count : 0;
        if (n_mem == 0) {
            int mask = aa.channel;
            if (mask < 0) mask = KI_DEFAULT_COLOR;
            for (int b = 0; b < COLOR_NB && n_mem < ADAM_MAX_MEM; b++)
                if (mask & (1 << b)) {
                    block_order[n_mem] = b;
                    mem_nc[n_mem] = blk_nc;
                    mem_off[n_mem] = total_stride;
                    total_stride += blk_nc;
                    n_mem++;
                }
            if (n_mem == 0) { n_mem = 1; block_order[0] = COLOR_MNIST; }
            n_blocks = n_mem;
        } else {
            for (int i = 0; i < n_mem && i < ADAM_MAX_MEM; i++) {
                int col = (int)aa.enc_array[i].color;
                block_order[i] = col >= 0 ? col : COLOR_MNIST;
                mem_nc[i] = blk_nc;
                mem_off[i] = total_stride;
                total_stride += blk_nc;
            }
            n_blocks = n_mem;
        }
    }
    int nc_per_sample = total_stride;

    /* ── Ensemble expansion: replicate members × ensembleN ───── */
    int base_n_mem = n_mem;
    int total_members = base_n_mem * aa.ensembleN;
    if (total_members > ADAM_MAX_MEM) total_members = ADAM_MAX_MEM;
    for (int e = 1; e < aa.ensembleN; e++) {
        for (int i = 0; i < base_n_mem && e * base_n_mem + i < ADAM_MAX_MEM; i++) {
            int idx = e * base_n_mem + i;
            mem_nc[idx] = mem_nc[i];
            mem_off[idx] = mem_off[i];
            block_order[idx] = block_order[i];
        }
    }
    n_mem = total_members;

    /* ── Load dataset ─────────────────────────────────────────── */
    ki_Dataset data = { .dry_run = aa.dry_run };
    if (ki_dataset_read(&data) != 0) return 1;
    int total_all = data.num_images;
    int total_train = aa.trainN;
    int total_eval  = aa.evalN;
    if (total_train + total_eval > total_all) {
        total_eval = total_all - total_train;
        if (total_eval < 0) { total_train = total_all; total_eval = 0; }
    }
    /* ── Pixel-data-dependent init (skipped for dry-run) ────── */
    uint8_t *y_tr = NULL, *y_te = NULL;
    float *X_all = NULL, *X_tr = NULL, *X_te = NULL;
    if (!aa.dry_run) {
        y_tr = data.y;
        y_te = data.y + total_train;

        /* ── Pack input ────────────────────────────────────────── */
        if (KI_COLORS <= 1) {
            /* MNIST: single block (ignores members/channels) */
            n_mem = 1; mem_nc[0] = KI_NC; mem_off[0] = 0;
            nc_per_sample = KI_NC;
            X_all = ki_pack_mnist_float(data.X_raw, total_train + total_eval, KI_NC);
        } else {
            /* CIFAR: pack blocks in enc_array order (or channel order) */
            X_all = ki_pack_blocks_float(data.X_raw, total_train + total_eval,
                                         block_order, n_blocks, aa.packedB);
            nc_per_sample = total_stride;
        }
        X_tr = X_all;
        X_te = X_all + (size_t)total_train * (size_t)nc_per_sample;
    }
    int H = aa.hidden;

    /* ── IFC MODE: --import → evaluate only ───────────────────── */
    if (aa.importD[0]) {
        printf("\n══╡ INFERENCE ╞══════════════════════════════════════════════════\n");
        /* Load per-member weights */
        int n_loaded = 0;
        ki_LinearLayer l0_arr[ADAM_MAX_MEM], l1_arr[ADAM_MAX_MEM];
        for (int i = 0; i < ADAM_MAX_MEM; i++) {
            char path[1024]; int nlay, hh, ncc, kk, hh2;
            snprintf(path, sizeof(path), "%s/weights-%d.meta", aa.importD, i);
            FILE *f = fopen(path, "r");
            if (!f) break;
            if (fscanf(f, "%d\n%d %d\n%d %d", &nlay, &hh, &ncc, &kk, &hh2) != 5
                || nlay != 2 || kk != N_CLASSES || hh2 != hh) {
                fclose(f); break;
            }
            fclose(f);
            l0_arr[i].in_features = ncc;
            l0_arr[i].out_features = hh;
            l0_arr[i].W = (float *)malloc((size_t)hh * (size_t)ncc * sizeof(float));
            l1_arr[i].in_features = hh;
            l1_arr[i].out_features = N_CLASSES;
            l1_arr[i].W = (float *)malloc((size_t)N_CLASSES * (size_t)hh * sizeof(float));
            snprintf(path, sizeof(path), "%s/W0-%d.bin", aa.importD, i);
            FILE *fw = fopen(path, "rb");
            if (fw) { fread(l0_arr[i].W, sizeof(float), (size_t)hh * (size_t)ncc, fw); fclose(fw); }
            snprintf(path, sizeof(path), "%s/W1-%d.bin", aa.importD, i);
            fw = fopen(path, "rb");
            if (fw) { fread(l1_arr[i].W, sizeof(float), (size_t)N_CLASSES * (size_t)hh, fw); fclose(fw); }
            n_loaded++;
        }
        if (n_loaded == 0) { fprintf(stderr, "[FATAL] No members found\n"); return 1; }
        H = l0_arr[0].out_features;
        struct timeval tv0, tv1; gettimeofday(&tv0, NULL);
        float acc = accuracy_multi(X_te, y_te, total_eval,
                                    l0_arr, l1_arr, mem_off, n_loaded,
                                    nc_per_sample);
        gettimeofday(&tv1, NULL);
        int el = (int)((tv1.tv_sec-tv0.tv_sec)*1000 + (tv1.tv_usec-tv0.tv_usec)/1000);
        printf("\n══╡ RESULT ╞══════════════════════════════════════════════════════\n");
        printf("  Eval:    %.1f%%  (%d samples, %d members)\n", (double)acc, total_eval, n_loaded);
        printf("  Time:    %dms\n", el);
        int evl_ok = (int)(acc * (float)total_eval / 100.0f + 0.5f);
        ki_report_show(0, 0, evl_ok, total_eval, el, aa.threadN, 0, 0.0f);
        for (int i = 0; i < n_loaded; i++) { free(l0_arr[i].W); free(l1_arr[i].W); }
        ki_dataset_free(&data); free(X_all);
        return 0;
    }

    /* ── Architecture display ────────────────────────────────── */
    /* Build channel/block display string */
    char chan_str[128] = "";
    if (n_mem > 1) {
        /* Multi-member: show block names per member */
        int pos = 0;
        for (int i = 0; i < n_mem && i < ADAM_MAX_MEM && pos < 100; i++) {
            if (i > 0) pos += snprintf(chan_str + pos, sizeof(chan_str) - (size_t)pos - 1, ",");
            pos += snprintf(chan_str + pos, sizeof(chan_str) - (size_t)pos - 1, "%s",
                            ki_color_name(block_order[i]));
        }
    } else {
        /* Single matrix: show all block names in the flat input */
        int pos = 0;
        for (int i = 0; i < n_blocks && i < ADAM_MAX_MEM && pos < 100; i++) {
            if (i > 0) pos += snprintf(chan_str + pos, sizeof(chan_str) - (size_t)pos - 1, ",");
            const char *cn = ki_color_name(block_order[i]);
            if (cn) pos += snprintf(chan_str + pos, sizeof(chan_str) - (size_t)pos - 1, "%s", cn);
        }
    }
    printf("══════════════════════════════════════════════════════════════════════\n");
    printf("══╡ ADAM ╞══  Float32  H=%d  Ep=%d  nc=%d  lr=%.4f\n",
           H, aa.epochs, nc_per_sample, (double)aa.lr);
    printf("══╡ SETUP ╞══════════════════════════════════════════════════════════\n");
    printf("  Input:       %d px → %d/%d blocks (%s) × %d = %d total  (packed)\n",
           KI_PX, n_blocks, COLOR_NB, chan_str[0] ? chan_str : "?",
           blk_nc, nc_per_sample);
    printf("  ───────────────────────────────────────────────────────────\n");
    printf("  HIDDEN       %-4d nrn x %2d bit  = %7zu bit  (%5.1f KB)\n",
           H, 32, (size_t)H * 32, (double)((size_t)H * 32) / 8 / 1024);
    size_t w0_total = (size_t)H * (size_t)nc_per_sample;
    size_t w1_total = (size_t)N_CLASSES * (size_t)H;
    printf("  OUTPUT       %-4d nrn x %2d bit  = %7zu bit  (%5.1f KB)\n",
           N_CLASSES, 32, (size_t)N_CLASSES * 32, (double)((size_t)N_CLASSES * 32) / 8 / 1024);
    printf("  ───────────────────────────────────────────────────────────\n");
    printf("  W0 = %4d x %4d x %2d bit  = %9zu bit  (%5.1f KB)  %s, frozen\n",
           H, (int)nc_per_sample, 32, w0_total * 32, (double)(w0_total * 32) / 8 / 1024,
           n_mem > 1 ? "per member" : "");
    printf("  W1 = %4d x %4d x %2d bit  = %9zu bit  (%5.1f KB)  %s, AdamW\n",
           N_CLASSES, H, 32, w1_total * 32, (double)(w1_total * 32) / 8 / 1024,
           n_mem > 1 ? "per member" : "");
    printf("  ───────────────────────────────────────────────────────────\n");
    size_t total_bit = (w0_total + w1_total) * 32;
    printf("  TOTAL                     %9zu bit  (%5.1f KB)",
           total_bit, (double)total_bit / 8 / 1024);
    if (n_mem > 1) printf("  × %d members = %zu KB", n_mem,
           (size_t)n_mem * total_bit / 8 / 1024);
    printf("\n");
    printf("  OMP:         %d threads\n", aa.threadN);
    printf("  Train/Eval:  %d / %d samples  batch=%d  warmup=%d\n",
           total_train, total_eval, aa.batchN, aa.warmup_epochs);
    printf("  Optimizer:   AdamW(lr=%.4f, lr_min=%.2f, wd=%.0e, clip=1.0)\n",
           (double)aa.lr, (double)aa.lr_min, (double)ADAM_WD);
    printf("  Schedule:    warmup + cosine decay  (%s)\n",
           !use_members ? "flat (single W0)" : "members");
    const char *rng_src;
    if (aa.seed_file[0])
        rng_src = "true random file";
    else
        rng_src = "splitmix64";
    printf("  Seed:        %u  %s  seed-member: %s",
           aa.seed, rng_src, ensemble_seed_str());
    if (aa.seed_file[0])
        printf("  from %s", aa.seed_file);
    printf("\n");

    /* ── MEMBER section ──────────────────────────────────────── */
    printf("\n══╡ MEMBER ╞══════════════════════════════════════════════════\n");
    printf("  Grid: EN[%d] × base[%d] = %d members%s\n",
           aa.ensembleN, base_n_mem, n_mem,
           use_members ? "" : " (flat)");
    printf("  Per member: W0[H=%d × I=%d], W1[K=%d × H=%d]\n",
           H, (int)mem_nc[0], N_CLASSES, H);
    /* Build arrays for ki_print_member_structure */
    int c[ADAM_MAX_MEM], t[ADAM_MAX_MEM], w[ADAM_MAX_MEM];
    for (int i = 0; i < n_mem && i < ADAM_MAX_MEM; i++) {
        c[i] = block_order[i % base_n_mem];
        t[i] = -1;
        w[i] = -1;
        if (aa.enc_count > 0 && (i % base_n_mem) < aa.enc_count) {
            t[i] = (int)aa.enc_array[i % base_n_mem].type;
            w[i] = (int)aa.enc_array[i % base_n_mem].width;
        }
    }
    ki_print_member_structure(c, t, w, n_mem, aa.ensembleN);

    /* ── Create per-member W0 + W1 + AdamW ────────────────────── */
    srand((unsigned int)aa.seed);  /* for ki_shuffle_int */
    ki_LinearLayer l0_arr[ADAM_MAX_MEM], l1_arr[ADAM_MAX_MEM];
    ki_AdamWState adamw_arr[ADAM_MAX_MEM];
    float *best_W0[ADAM_MAX_MEM], *best_W1[ADAM_MAX_MEM];
    float best_eval_acc = 0.0f;

    for (int m = 0; m < n_mem && m < ADAM_MAX_MEM; m++) {
        l0_arr[m].in_features = mem_nc[m];
        l0_arr[m].out_features = H;
        l0_arr[m].W = (float *)ki_xcalloc((size_t)H * (size_t)mem_nc[m], sizeof(float));
        l1_arr[m].in_features = H;
        l1_arr[m].out_features = N_CLASSES;
        l1_arr[m].W = (float *)ki_xcalloc((size_t)N_CLASSES * (size_t)H, sizeof(float));
        adamw_arr[m] = ki_adamw_create((size_t)N_CLASSES * (size_t)H, aa.lr, aa.lr_min);
        best_W0[m] = (float *)ki_xmalloc((size_t)H * (size_t)mem_nc[m] * sizeof(float));
        best_W1[m] = (float *)ki_xmalloc((size_t)N_CLASSES * (size_t)H * sizeof(float));
    }

    /* W0/W1 initialisieren (Seed-Modus wie Otto) */
    if (aa.ensemble_seed == ENS_SEED_CONST) {
        w0_srandom((uint64_t)aa.seed);
        ki_init_kaiming_w0_seq(&l0_arr[0]);  /* Member 0 als Vorlage */
        ki_init_kaiming_w0_seq(&l1_arr[0]);
        for (int m = 1; m < n_mem && m < ADAM_MAX_MEM; m++) {
            memcpy(l0_arr[m].W, l0_arr[0].W, (size_t)H * (size_t)mem_nc[0] * sizeof(float));
            memcpy(l1_arr[m].W, l1_arr[0].W, (size_t)N_CLASSES * (size_t)H * sizeof(float));
        }
    } else if (aa.ensemble_seed == ENS_SEED_INCR) {
        for (int m = 0; m < n_mem && m < ADAM_MAX_MEM; m++) {
            ki_init_kaiming_w0(&l0_arr[m], (uint64_t)aa.seed + (uint64_t)m);
            ki_init_kaiming_w0(&l1_arr[m], (uint64_t)(aa.seed + 1000) + (uint64_t)m);
        }
    } else {
        /* once (default): Ein Seed, alle sequential */
        w0_srandom((uint64_t)aa.seed);
        for (int m = 0; m < n_mem && m < ADAM_MAX_MEM; m++) {
            ki_init_kaiming_w0_seq(&l0_arr[m]);
            ki_init_kaiming_w0_seq(&l1_arr[m]);
        }
    }

    int *idx = (int *)ki_xmalloc((size_t)total_train * sizeof(int));

    /* ══════════════════════════════════════════════════════════════
     * TRAINING LOOP
     * ══════════════════════════════════════════════════════════════ */
    printf("\n══╡ TRAINING ╞══  lr=%.4f  lr_min=%.2f  batch=%d  warmup=%d\n",
           (double)aa.lr, (double)aa.lr_min, aa.batchN, aa.warmup_epochs);

    struct timeval tv_start;
    gettimeofday(&tv_start, NULL);

    for (int ep = 0; ep < aa.epochs; ep++) {
        struct timeval tv_ep0;
        gettimeofday(&tv_ep0, NULL);
        float lr_min_val = aa.lr * aa.lr_min;
        float lr_ep = ki_lr_schedule(ep, aa.epochs, aa.warmup_epochs, aa.lr, lr_min_val, 0);
        for (int m = 0; m < n_mem && m < ADAM_MAX_MEM; m++)
            adamw_arr[m].lr = lr_ep;

        for (int i = 0; i < total_train; i++) idx[i] = i;
        ki_shuffle_int(idx, total_train);

        for (int start = 0; start < total_train; start += aa.batchN) {
            int actual = aa.batchN;
            if (start + actual > total_train) actual = total_train - start;

            for (int m = 0; m < n_mem && m < ADAM_MAX_MEM; m++) {
                int off_m = mem_off[m];
                size_t w1_n = (size_t)N_CLASSES * (size_t)H;

                /* Forward (parallel over batch samples) */
                float *h0_buf = (float *)malloc((size_t)actual * (size_t)H * sizeof(float));
                float *out_buf = (float *)malloc((size_t)actual * (size_t)N_CLASSES * sizeof(float));
                #pragma omp parallel for schedule(static)
                for (int bi = 0; bi < actual; bi++) {
                    int sidx = idx[start + bi];
                    const float *x = X_tr + (size_t)sidx * (size_t)nc_per_sample + off_m;
                    forward(&l0_arr[m], &l1_arr[m], x,
                            h0_buf + (size_t)bi * (size_t)H,
                            out_buf + (size_t)bi * (size_t)N_CLASSES, 1);
                }

                /* MSE gradient for W1 */
                float *grad_w1 = (float *)malloc(w1_n * sizeof(float));
                memset(grad_w1, 0, w1_n * sizeof(float));
                for (int k = 0; k < N_CLASSES; k++) {
                    for (int h_id = 0; h_id < H; h_id++) {
                        float acc_g = 0.0f;
                        for (int bi = 0; bi < actual; bi++) {
                            int sidx = idx[start + bi];
                            float tgt = (y_tr[sidx] == k) ? 1.0f : -1.0f;
                            acc_g += (out_buf[bi * N_CLASSES + k] - tgt)
                                   * h0_buf[bi * H + h_id];
                        }
                        grad_w1[k * H + h_id] = acc_g / (float)actual;
                    }
                }
                ki_clip_grad(grad_w1, w1_n);
                ki_adamw_step(&adamw_arr[m], l1_arr[m].W, grad_w1);
                free(h0_buf); free(out_buf); free(grad_w1);
            }
        }

        /* Evaluate */
        float train_acc = accuracy_multi(X_tr, y_tr, total_train,
                                          l0_arr, l1_arr, mem_off, n_mem,
                                          nc_per_sample);
        float eval_acc = (total_eval > 0)
            ? accuracy_multi(X_te, y_te, total_eval,
                             l0_arr, l1_arr, mem_off, n_mem,
                             nc_per_sample)
            : 0.0f;

        if (eval_acc > best_eval_acc) {
            best_eval_acc = eval_acc;
            for (int m = 0; m < n_mem && m < ADAM_MAX_MEM; m++) {
                memcpy(best_W0[m], l0_arr[m].W,
                       (size_t)H * (size_t)mem_nc[m] * sizeof(float));
                memcpy(best_W1[m], l1_arr[m].W,
                       (size_t)N_CLASSES * (size_t)H * sizeof(float));
            }
        }

        struct timeval tv_ep1;
        gettimeofday(&tv_ep1, NULL);
        int ep_ms = (int)((tv_ep1.tv_sec - tv_ep0.tv_sec) * 1000
                        + (tv_ep1.tv_usec - tv_ep0.tv_usec) / 1000);
        printf("  Ep %2d  trn=%.1f%%  evl=%.1f%%  best=%.1f%%  lr=%.6f  time=%dms  mem=%d\n",
               ep + 1, (double)train_acc, (double)eval_acc, (double)best_eval_acc,
               (double)lr_ep, ep_ms, n_mem);
        fflush(stdout);
    }

    struct timeval tv_end;
    gettimeofday(&tv_end, NULL);
    int elapsed_ms = (int)((tv_end.tv_sec - tv_start.tv_sec) * 1000
                         + (tv_end.tv_usec - tv_start.tv_usec) / 1000);

    /* ── EXPORT (per-member, only with --export) ────────────────── */
    if (!aa.dry_run && aa.exportD[0]) {
        for (int m = 0; m < n_mem && m < ADAM_MAX_MEM; m++)
            export_member(best_W0[m], best_W1[m], H, mem_nc[m], aa.exportD, m);
        size_t total_kb = 0;
        for (int m = 0; m < n_mem && m < ADAM_MAX_MEM; m++)
            total_kb += ((size_t)H * (size_t)mem_nc[m] + (size_t)N_CLASSES * (size_t)H) * sizeof(float);
        printf("\n══╡ EXPORT ╞════════════════════════════════════════════════\n");
        printf("  Model:  %s  (%d members, H=%d)\n", aa.exportD, n_mem, H);
        printf("  Total:  %zu KB  (%d × (W0=%zuB + W1=%zuB))\n",
               total_kb / 1024, n_mem,
               (size_t)H * (size_t)mem_nc[0] * sizeof(float) / 1024,
               (size_t)N_CLASSES * (size_t)H * sizeof(float) / 1024);
    }

    /* ── REPORT ──────────────────────────────────────────────── */
    printf("\n══╡ RESULT ╞══════════════════════════════════════════════════════\n");
    printf("  H=%d  mem=%d  ep=%d  trn=%.1f%%  evl=%.1f%%  lr=%.4f  time=%dms\n",
           H, n_mem, aa.epochs, 0.0, (double)best_eval_acc, (double)aa.lr, elapsed_ms);
    int eval_ok = (int)(best_eval_acc * (float)total_eval / 100.0f + 0.5f);
    int train_ok = (int)(best_eval_acc * (float)total_train / 100.0f + 0.5f);
    ki_report_show(train_ok, total_train, eval_ok, total_eval,
                   elapsed_ms, aa.threadN, 0, 0.0f);

    /* ── Confusion matrix (end only) ───────────────────────────── */
    if ((aa.debug_confusion || aa.debug_confusion_all) && !aa.dry_run) {
        uint8_t *pred_tr = (uint8_t *)malloc((size_t)total_train);
        #pragma omp parallel for schedule(static)
        for (int s = 0; s < total_train; s++) {
            float total[N_CLASSES];
            for (int k = 0; k < N_CLASSES; k++) total[k] = 0;
            for (int m = 0; m < n_mem && m < ADAM_MAX_MEM; m++) {
                const float *slice = X_tr + (size_t)s * (size_t)nc_per_sample + mem_off[m];
                float h0_buf[4096], out[N_CLASSES];
                forward(&l0_arr[m], &l1_arr[m], slice, h0_buf, out, 1);
                for (int k = 0; k < N_CLASSES; k++) total[k] += out[k];
            }
            int pred = 0;
            for (int k = 1; k < N_CLASSES; k++)
                if (total[k] > total[pred]) pred = k;
            pred_tr[s] = (uint8_t)pred;
        }
        print_confusion_debug(y_tr, pred_tr, total_train, aa.epochs - 1, !aa.debug_confusion_all);
        free(pred_tr);
    }

    /* ── Cleanup ─────────────────────────────────────────────── */
    for (int m = 0; m < n_mem && m < ADAM_MAX_MEM; m++) {
        ki_adamw_free(&adamw_arr[m]);
        free(l0_arr[m].W); free(l1_arr[m].W);
        free(best_W0[m]); free(best_W1[m]);
    }
    free(idx); free(X_all);
    ki_dataset_free(&data);
    return 0;
}
