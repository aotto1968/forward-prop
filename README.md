# Otto Score — DRAM-Native MLP Classifier

**MNIST: 99.0% in 4s  |  CIFAR-10: 64.9% (maj=1, sizeN=32)  |  Fashion-MNIST: 90.2%** — Zero floating point, zero matmul in inference.
Only `&|~` + int32 + popcount. Also includes float32 AdamW + multi-member Hebbian + **Bit-Voting linear baseline** (proves the 5pp W0 nonlinearity gap).

> 📖 **Full status report:** [`www/papers/status-2026-07.html`](www/papers/status-2026-07.html) — Latest findings: `--export-gb` cache, `--maj 1` default, `avg2/3/4` filters, `@` pipe chaining, 65% ceiling analysis.

---

## 🚀 Quick Start

```bash
# Step 1: Build everything (9 binaries)
make all

# Step 2: Download datasets (MNIST + CIFAR-10 + Fashion-MNIST)
make setup

# Step 3: Train models + run tests (first run ~5min, then cached <1s)
make test
```

**Expected output (cached models):**

```
=== Otto Score MNIST (H=512, 10 ep, latest)       ===      eval=97.0%
=== Float32 AdamW MNIST (H=512, 10 ep)            ===      eval=92.5%
=== Bin32 Hebbian MNIST (H=512, 10 ep, latest)    ===      eval=84.3%
=== Otto Score CIFAR-10 (H=256, 5 ep, latest)     ===      eval=58.0%
=== Float32 AdamW CIFAR-10 (H=256, 5 ep)          ===      eval=44.3%
=== Bin32 Hebbian CIFAR-10 (H=256, 5 ep, latest)  ===      eval=31.0%
=== Otto Score Fashion (H=512, 10 ep, latest)     ===      eval=87.5%
=== Float32 AdamW Fashion (H=512, 10 ep)          ===      eval=82.1%
=== Bin32 Hebbian Fashion (H=512, 10 ep, latest)  ===      eval=71.0%
```

First run trains all 12 models (~5min). Subsequent runs use cached models (<1s total).

**`make models` — Train fresh models (cached):**
```
$ make models
make -s -C mnist model-otto
REPORT train=100.0% (60000) eval=97.1% (10000) ...
make -s -C mnist model-adam
REPORT train=92.5% (60000) eval=92.5% (10000) ...
make -s -C mnist model-hebbian
REPORT train=83.2% (60000) eval=84.3% (10000) ...
make -s -C cifar model-otto
REPORT train=78.5% (50000) eval=58.0% (10000) ...
make -s -C cifar model-adam
REPORT train=44.3% (50000) eval=44.3% (10000) ...
make -s -C cifar model-hebbian
REPORT train=30.6% (50000) eval=31.0% (10000) ...
make -s -C fashion model-otto
REPORT train=99.2% (60000) eval=87.5% (10000) ...
make -s -C fashion model-adam
REPORT train=82.1% (60000) eval=82.1% (10000) ...
make -s -C fashion model-hebbian
REPORT train=71.8% (60000) eval=71.0% (10000) ...
```

---

## 🎯 Test Targets

| Command                     | What it tests                          |
| --------------------------- | -------------------------------------- |
| `make test-mnist`           | All 4 MNIST approaches                 |
| `make test-mnist-otto`      | Otto Score MNIST only                  |
| `make test-mnist-adam`      | Float32 AdamW MNIST only               |
| `make test-mnist-hebbian`   | Bin32 Hebbian MNIST only               |
| `make test-mnist-bitvoting` | Bit-Voting MNIST only                  |
| `make test-cifar`           | All 4 CIFAR-10 approaches              |
| `make test-cifar-otto`      | Otto Score CIFAR-10 only               |
| `make test-cifar-adam`      | Float32 AdamW CIFAR-10 only            |
| `make test-cifar-hebbian`   | Bin32 Hebbian CIFAR-10 only            |
| `make test-cifar-bitvoting` | Bit-Voting CIFAR-10 only               |
| `make test-fashion`         | All 4 Fashion-MNIST approaches         |
| `make test-fashion-otto`    | Otto Score Fashion-MNIST only          |
| `make test-fashion-adam`    | Float32 AdamW Fashion-MNIST only       |
| `make test-fashion-hebbian` | Bin32 Hebbian Fashion-MNIST only       |
| `make test-fashion-bitvoting` | Bit-Voting Fashion-MNIST only        |

First run trains all 12 models. Subsequent runs use cached models (<1s total).

---

## 🔬 Key Findings

### Majority Mode: `maj=1` is Now Default

The old `--maj 3` tree was identified as `--maj 1` with an effective threshold
of ~52.7% (135/256). See [Status Report July 2026](www/papers/status-2026-07.html).

**`--maj 1`** is DRAM-native (pure bit-logic, no tree overhead) and conceptually
cleaner. The threshold is controlled via `--maj-thresh`:
- `-2` (default): auto per encoding (lookup table: 256→135, 512→269, 1024→540)
- `-1`: n/2 (exact 50%)
- `>=0`: exact value

The old maj3 mechanism is considered legacy and is no longer the default.

### `--maj-step N` — Only `N == KI_PX_PER_CONT` Works

The `--maj-step` flag controls the pixel distance between the three members
of each majority triple. **Only the default `--maj-step 4` (= KI_PX_PER_CONT
at 8-bit) produces valid results.** Any other value creates fundamental
problems:

| Step    | Relation to ppc (=4) | Problem                                                                                                                                                                                                                    |
| ------- | -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **4**   | `== ppc`             | **Container-aligned** — each triple draws 1 pixel per container, `uint32_t` bit layout = pixel position 0-3. Fast path `majority_tree3` used. **Default. Works. ✓**                                                        |
| **1-3** | `< ppc`              | Triple (i, i+S, i+2S) fits within 3 containers (12px). Path 1 (container-grouping) works correctly, but **rearranges which bits compose each position** → different match results. W0 encoding expects ppc-aligned layout. |
| **5+**  | `> ppc`              | Triple (i, i+S, i+2S) **spans more than 3 containers**. No container-aligned slot assignment possible → Path 2 (flat packing) used. **~2-3pp accuracy loss** vs step=4.                                                    |

**Root cause:** The `uint32_t` match vector has a fixed bit-layout
(`slot 0 = bits 0-7`, `slot 1 = bits 8-15`, etc.) that corresponds to
pixel position within each container. The W0 random projection and H0
encoding are trained for this layout. Any step other than ppc **shifts
which pixels land in which slot**, destroying the alignment that the
encoding logic depends on.

For step > ppc, the triple even crosses container boundaries — pixel
positions from different containers/slots end up in the same triple,
and the result bit cannot be assigned to any single slot. Flat packing
(assigning results sequentially to the next free slot) is the only
correct approach, but produces a bit layout that the H0 encoding
cannot exploit effectively.

**Implementation details:**
- `--debug-maj container` — forces `majority_tree3` (original, fast)
- `--debug-maj pixel` — forces `majority_tree3_pixel_step` (step-aware)
- `--debug-maj auto` (default) — fast path for step=ppc, pixel_step otherwise

Both paths are validated to produce **identical results** for step=ppc
(train=39.9%, eval=38.0%, err=30038 at H=64 ep=1).

### Row-Wise Running Average Filters (July 2026)

Three new xform types for row-wise moving average (wrap-right):

| Xform | Formula | Effect |
|-------|---------|--------|
| `avg2` | `p[i] = (p[i] + p[i+1]) / 2` | 2-tap blur |
| `avg3` | `p[i] = (p[i] + p[i+1] + p[i+2]) / 3` | 3-tap |
| `avg4` | `p[i] = (p[i] + p[i+1] + p[i+2] + p[i+3]) / 4` | 4-tap, strongest blur |

Usage:
```bash
--xform avg4,rot45@avg4,rot90@avg4
```

**Insight:** Avg filters spread one pixel's information across 4 neighbors.
Combined with the `@` pipe syntax, multiple transforms can be chained,
e.g. `rot45@avg4` = rotate first, then blur.

### Xform Pipeline Chaining `@` (July 2026)

The `@` syntax enables sequential chaining of xforms:

```bash
# One member: rot90, then avg4 on the result
--xform rot90@avg4

# Three members: avg4, rot45+avg4, rot90+avg4
--xform avg4,rot45@avg4,rot90@avg4

# Any number of steps
--xform rot45@avg2@avg4

# id steps are preserved (the name equals what was entered)
--xform id@avg4  # → member name "id@avg4" (no filtering)
```

Each pipeline is **one member** (not multiple). `rot90@avg4 = 1×`, `avg4,rot45@avg4 = 2×`.

### `--encoding-sizeN 0` — RAW Encoding (July 2026)

With `--encoding-sizeN 0`, NO thermometer encoding is applied — the raw 8-bit pixel values
go directly into the container (4 pixels per uint32). This corresponds to `KI_ENC_RAW` with `width=8`.

```bash
# Otto Score with RAW Encoding
./cifar/cifar-mlp-bin32-otto-trn-xnor.exe --encoding-sizeN 0 ...

# BitVoting with RAW Encoding (no W0 bottleneck)
./cifar/cifar-mlp-bin32-otto-trn-bitvoting.exe --encoding-sizeN 0 ...
```

**Result:** Train reaches 99.6% (Otto) resp. 96.8% (BitVoting), but eval generalizes
differently — BitVoting shows **no ceiling** (eval still climbing at 47.4%, 60 Members).

### Otto Score vs Bit-Voting Baseline: +7pp at 3.6× Time

CIFAR-10, H=768, 10 epochs, performance encoding, 12 members:

| Approach                       | Eval      | Time    | Key Difference                |
| ------------------------------ | --------- | ------- | ----------------------------- |
| **Otto Score** (W0 + majority) | **61.3%** | 57s     | Nonlinear feature abstraction |
| Bit-Voting (linear)            | 54.3%     | **16s** | Direct bit voting, no W0      |

Otto's +7pp come from the **dense W0 random projection** (24576 → 768×32 H0 bits)
followed by majority-threshold — the same 24576 input bits distributed overcomplete
and then nonlinearly compressed. The 41s overhead is the one-time W0×I0 computation
(12 members × 768 H × 256 NC × 50K samples). DRAM hardware would execute this at
memory speed with zero software overhead.

### Bit-Masse Equivalence

All three formats (flt32, int32, bin32) share the same information-theoretic
container principle: **every neuron is a 32-bit container**. Float32 and int32
differ only in computation format (multiply-add vs XNOR+popcount), not in
capacity. The Bit-Voting baseline proves that the +5–7pp gap between Otto and
linear voting is due to **nonlinear feature abstraction from W0 + majority**,
not from more parameters or higher precision.

### W0 advantage is dataset-dependent (Fashion-MNIST, 2026-08-16)

On Fashion-MNIST the W0 gap nearly vanishes when both confounders are
controlled (identical member matrix via `--filter-t1` = all-rot@filter-bv,
7800 members; equal ensemble size via `--max 12`):

| Approach | H | Bit-mass | eval (12 members) |
| -------- | - | -------- | ----------------- |
| Bit-Voting (no W0) | 196 | 245 KB | 91.58% |
| Otto Score (W0)    | 196 | 395 KB | **91.93%** (+0.35 pp) |
| Otto Score (W0)    | 198 | 399 KB | 91.79% (+0.21 pp) |

- On CIFAR W0 buys +7pp (dense projection matters for noisy color data);
  on Fashion it buys only +0.35 pp at +61% bit-mass. The projection's value
  scales with input entropy, not with H.
- An apparent H198 > H196 result (92.19% vs 91.93%) was an **ensemble-size
  artifact** (22 vs 12 members): at `--max 12`, H196 beats H198. H-scalability
  is not monotonic around the container count (196).
- Bit-Voting's structural cost is H-fixation (H = container count; scaling
  requires BV16/BV32), while Otto buys free H-scaling with the W0 matrix.

**Precision caveat (2026-08-06):** the container is format-neutral, but the
**score accumulation must be exact (int64/double)**. The float32 accumulator
drifts once the ensemble scores reach ~1e9 (> 2²⁴): the same corpus selected
with float32 vs int64/double gave 29 vs 44 members (66.79% vs 67.72%). The DRAM
chip accumulates popcounts exactly in integer arithmetic — the exact behavior is
chip-faithful, the float32 drift was an emulation artifact. `SCORE_TYPE`
defaults to double; `-DSCORE_TYPE=float` reproduces the legacy behavior. The
`.ens` archive follows the internal format (v12 double / v13 int64) and the
`.meta` records/validates the arithmetic types.

---

```
otto-score-ifc/
├── Makefile              ← orchestrates: delegates to subdirectories
├── mnist/                ← MNIST sources (Otto + Hebbian + Adam, unified)
│   ├── Makefile
│   ├── mlp-bin32-otto-trn.c      ← Otto Score (shared source)
│   ├── mlp-bin32-hebbian-trn.c   ← Hebbian (shared source)
│   ├── mlp-flt32-adam-trn.c      ← Float32 AdamW (unified, --import inference)
│   ├── ki-common.h               ← Shared header (args, parsing, RNG)
│   ├── ki-local.h                ← MNIST dataset config
│   └── merge-ensemble.c          ← Ensemble merge tool (companion to --export-merge-scores)
├── cifar/                ← CIFAR-10 (symlinks to mnist/ + CIFAR ki-local.h)
│   ├── Makefile
│   ├── mlp-bin32-otto-trn.c      → ../mnist/...
│   ├── mlp-bin32-hebbian-trn.c   → ../mnist/...
│   ├── mlp-flt32-adam-trn.c      → ../mnist/...
│   ├── ki-common.h               → ../mnist/ki-common.h
│   └── ki-local.h                ← CIFAR dataset config
├── fashion/              ← Fashion-MNIST (symlinks to mnist/ + Fashion ki-local.h)
│   ├── Makefile
│   ├── mlp-bin32-otto-trn.c      → ../mnist/...
│   ├── mlp-bin32-hebbian-trn.c   → ../mnist/...
│   ├── mlp-flt32-adam-trn.c      → ../mnist/...
│   ├── ki-common.h               → ../mnist/ki-common.h
│   └── ki-local.h                ← Fashion-MNIST dataset config
├── lib/                   ← Shared headers (ki-adamw.h, ki-encoding.h, maj3.h, w0_random.h)
├── models/                ← Cached trained models (-e10 MNIST, -e5 CIFAR)
├── bin/                   ← Helper scripts (run-ensemble.sh, etc.)
├── www/                   ← Publication site (HTML papers, style.css, datasets)
├── fetch_mnist.sh         ← MNIST download (https://forward-prop.nhi1.de/data/mnist/)
├── fetch_cifar10.sh       ← CIFAR-10 download (https://forward-prop.nhi1.de/data/cifar-10-batches-bin/)
├── fetch_fashion.sh       ← Fashion-MNIST download (https://forward-prop.nhi1.de/data/mnist-fashion/)
└── README.md              ← this file
```

All 3 trainers (Otto, Hebbian, Adam) are **unified** across MNIST and CIFAR:
the same `.c` file is compiled with different `ki-local.h` per dataset.
Each trainer doubles as inference engine via `--import`. Zero code drift.

## Build Targets

| Command         | Builds                                                          |
| --------------- | --------------------------------------------------------------- |
| `make` / `all`  | All 9 binaries (Otto + Hebbian + Adam × XNOR/XOR, all datasets) |
| `make otto`     | Otto Score only (mnist/ + cifar/ + fashion/)                    |
| `make hebbian`  | Hebbian only (mnist/ + cifar/ + fashion/)                       |
| `make adam`     | Float32 AdamW only (mnist/ + cifar/ + fashion/)                 |
| `make models`   | Train all 9 models (cached)                                     |
| `make clean`    | Remove executables                                              |
| `make ensemble` | Build merge-ensemble only (mnist/ + cifar/)                     |
| `make bitvote`  | Build Bit-Voting baseline (linear perceptron, no W0)            |

## CLI Flags (unified across all trainers)

| Flag                               | Description                                                                                                                 | Default         |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | --------------- |
| `--hiddenN N`                      | Hidden neurons                                                                                                              | 64              |
| `--epochsN N`                      | Training epochs                                                                                                             | 1               |
| `--splitVN N`                      | Bit-Grouping: 1                                                                                                             | 1,2,3,4,8,16,32 |
| `--encoding TYPE`                  | Input encoding: exp, sig, up8, down8, raw, etc.; `latest` = dataset-specific alias (CIFAR: 11 members, MNIST/Fashion: exp8) | raw8            |
| `--ensembleN N`                    | Independent W0 copies                                                                                                       | 1               |
| `--export DIR`                     | Model export directory                                                                                                      | none            |
| `--import DIR`                     | Load model for inference                                                                                                    | none            |
| `--predictions FILE`               | Export per-sample predictions (for vis-errors tool)                                                                         | none            |
| `--export-merge-scores DIR`        | Save per-member scores to archive files (ensemble)                                                                          | none            |
| `--export-gb`                      | Cache gb_buf to data/gb/ (≥2nd run 35% faster)                                                                              | off             |
| `--debug-cache`                   | Log every cache create/use (xform cache, cex *.pre, gb cache)                                                              | off             |
| `--dry-run`                        | Print architecture and exit (metadata only, instant)                                                                        | off             |
| `--seed N`                         | Random seed                                                                                                                 | 42              |
| `--seed-member MODE`               | Member seed strategy (once, const, incr)                                                                                    | once            |
| `--gap-k F`                        | Exp(-K×gap) step damping when train/eval gap widens                                                                         | 0.0 (off)       |
| `--multi-correct`                  | Punish all wrong classes, not just argmax                                                                                   | off             |
| `--batchN N`                       | Mini-batch size                                                                                                             | 64              |
| `--maj-step N`                     | Pixel step between majority triples (0=auto=KI_PX_PER_CONT)                                                                 | 0               |
| `--debug-maj auto,container,pixel` | Force majority path (debug)                                                                                                 | auto            |
| `--debug-class-voting`             | Per-member per-class accuracy table                                                                                         | off             |
| `--debug-confusion-matrix`         | Confusion matrix table                                                                                                      | off             |
| `--filter CLASSES`                 | Class subset (numeric or name, comma-sep)                                                                                   | none            |
| `--qq`                             | Quick mode: 5000 train / 2000 eval / 3 ep                                                                                   | off             |
| `--threadN N`                      | OpenMP threads                                                                                                              | auto            |

**`--splitVN` — Bit-Grouping (Otto Score only):**
- `--splitVN 1` (default): every bit is its own feature, 50% retention. Optimal for clean data (MNIST).
- `--splitVN 2` (sweet spot): AND2 filter, only `11` counts, **25% retention**. Optimal for noisy data (CIFAR-10).
- `--splitVN 3-32`: strict AND (all bits in the group must be 1). Only useful for very large H.

Retention determines the VN choice:
| VN    | Groups | Retention | Character                            |
| ----- | ------ | --------- | ------------------------------------ |
| 1     | 32     | 50.0%     | Soft — everything counts             |
| **2** | **16** | **25.0%** | **CIFAR champion** — hard AND        |
| 3     | 10     | 12.5%     | Needs 8× more H for same performance |
| 4     | 8      | 6.25%     | Starves at small H                   |
| 8-32  | 4-1    | 0.4%-0%   | Only viable at H>16384+              |

Backward-compat aliases: `--out` = `--export`, `--model` = `--import`.

## Ensemble Workflow — Accumulate Seeds Over Time

Every Otto Score ensemble member is **completely independent** — different W0 (different
random projection), trained from scratch, own target matrix. This means members can be
computed **"on demand"** and accumulated over time. The merge tool then combines any
subset to find the optimal ensemble size.

### Why this works

Each training run uses a different random seed → different W0 projection → **uncorrelated errors**.
The ensemble sum smoothes out individual member mistakes. More seeds = better accuracy,
logarithmically saturating. The merge-ensemble tool shows exactly where
you are on this curve.

### 1. Train with `--export-merge-scores DIR`

Each run creates one `.ens` archive file with all per-member int64 scores:

```bash
# Manual (one seed per call)
./cifar/cifar-mlp-bin32-otto-trn-xnor.exe --hiddenN 512 --epochsN 7 \
  --encoding latest --splitVN 2 --export-merge-scores scores/ --seed 1234

./cifar/cifar-mlp-bin32-otto-trn-xnor.exe --hiddenN 512 --epochsN 7 \
  --encoding latest --splitVN 2 --export-merge-scores scores/ --seed 5678

# Auto-seed: run-ensemble.sh picks a random seed NOT used before
bash bin/run-ensemble.sh ./cifar/cifar-mlp-bin32-otto-trn-xnor.exe \
  --hiddenN 512 --epochsN 7 --encoding latest --splitVN 2

# Repeat N times (each gets its own unique seed)
bash bin/run-ensemble.sh --repeat 20 ./cifar/cifar-mlp-bin32-otto-trn-xnor.exe \
  --hiddenN 512 --epochsN 7 --encoding latest --splitVN 2
```

Each run writes one archive to `scores/`:
`H512_EP7_VN2_HN1_TE40_SD1234_F4_TS1712345678.ens`

### 2. Merge all seeds to an EN curve

```bash
# Build merge-ensemble (public)
make -C cifar ensemble

# Merge all archives in scores/ — shows per-member accuracy gain
./cifar/cifar-merge-ensemble.exe scores/

# Filter: only members from archives with eval > 58%
./cifar/cifar-merge-ensemble.exe scores/ --filter eval gt 58

# Sort by seed (default: ctime = order of creation)
./cifar/cifar-merge-ensemble.exe scores/ --sort seed

# Limit to first 20 members
./cifar/cifar-merge-ensemble.exe scores/ --num 20

# Save curve data (for plotting)
./cifar/cifar-merge-ensemble.exe scores/ --save /tmp/curve.dat
```

Example output (H=512, 20 seeds, total 220 members):
```
== MERGE ENSEMBLE ============================================
  220 score blocks (10000 test samples, 10 classes)
  Config: H=512  EP=7  VN=2  HN=1  TE=40

  EN    acc[%]    correct      gain[%]  member
  ----  -------  -----------  -------  -------------------
  ─── #1 [0.0%] H512_EP7_VN2_HN1_TE40_SD1234_F4_TS....ens ───
  1     43.50     4350/10000   +43.50  G=up8
  2     47.20     4720/10000    +3.70  B=up8
  ...
  ─── #2 [51.2%] H512_EP7_VN2_HN1_TE40_SD5678_F4_TS....ens ───
  12    56.30     5630/10000    +0.60  BL=down8
  ...
  220   61.22     6122/10000    +0.02  GB=sig8
```

The gain column shows the improvement from adding this member. When gain → 0,
the ensemble is saturated at this H — need more neurons or more seeds.

### merge-ensemble options

```
--num N       Only combine first N members (default: all)
--sort MODE   Sort by 'seed' or 'ctime' (default: ctime, = order of creation)
--filter L    Exclude by label substring ("sig8") or eval threshold (eval gt 58.1) -- no quotes needed
--save FILE   Save accuracy curve data to FILE (default: DIR/merge.dat)
-h, --help    Show help
```

### Archive format (.ens)

| Version | Added   | Contents                                                  |
| ------- | ------- | --------------------------------------------------------- |
| v1      | —       | Bare scores + labels                                      |
| v2      | 2026-07 | Per-member encoding metadata (color, enc_type, enc_width) |
| v3      | 2026-07 | Embedded `int64` timestamp in header                      |
| v4      | 2026-07 | Ensemble eval accuracy (`float`) + `_F4` in filename      |

File naming: `H{hidden}_EP{epochs}_VN{splitVN}_HN{splitHN}_TE{te}_SD{seed}_F4_TS{timestamp}.ens`

## Inference via `--import`

Every training binary doubles as inference engine. No separate IFC binaries:

```bash
# MNIST Otto Score (latest = exp8, single member)
./mnist/mnist-mlp-bin32-otto-trn-xnor.exe \
  --import models/mnist-otto-h512-e10 --evalN 10000 --encoding latest

# CIFAR-10 Otto Score (--encoding latest = 11 members)
./cifar/cifar-mlp-bin32-otto-trn-xnor.exe \
  --import models/cifar-otto-h256-e5 --evalN 10000 --encoding latest

# MNIST Hebbian (latest = exp8)
./mnist/mnist-mlp-bin32-hebbian-trn-xnor.exe \
  --import models/mnist-hebbian-h512-e10 --evalN 10000 --encoding latest

# CIFAR Hebbian (multi-member, 11 members via --encoding latest)
./cifar/cifar-mlp-bin32-hebbian-trn-xnor.exe \
  --import models/cifar-hebbian-h256-e5 --evalN 10000 --encoding latest

# AdamW (float32, --import works the same way)
./mnist/mnist-mlp-flt32-adam-trn.exe \
  --import models/mnist-adam-h512-e10 --evalN 10000
./cifar/cifar-mlp-flt32-adam-trn.exe \
  --import models/cifar-adam-h256-e5 --evalN 10000
```

## `--dry-run` — Fast Architecture Preview

Prints the full 5-section layout (SETUP, MEMBER, TRAINING, EXPORT, RESULT)
**without loading pixel data** (metadata only from dataset headers):

```bash
# MNIST — instant
./mnist/mnist-mlp-bin32-otto-trn-xnor.exe --dry-run --hiddenN 512 --epochsN 10

# CIFAR — skips 180MB batch-file reads
./cifar/cifar-mlp-flt32-adam-trn.exe --dry-run --hiddenN 128 --encoding latest
```

## `--export` — Save Trained Models

All trainers export per-member weights via `--export DIR`:

```bash
./mnist/mnist-mlp-bin32-otto-trn-xnor.exe --hiddenN 256 --epochsN 10 \
  --encoding latest --export models/mnist-otto/
# → writes models/mnist-otto/weights-0.meta, W0-0.bin, W1-0.bin, ...
```

Without `--export`, no files are written (training-only mode).

## Error Visualization via `--predictions`

Every trainer can export per-sample predictions. The built-in
**Error Visualizer** generates an `index.html` with all samples sorted
by class — errors are marked in red.

MNIST (grayscale PNG, 28×28):

```bash
# Step 1: generate predictions
./mnist/mnist-mlp-bin32-otto-trn-xnor.exe \
  --import models/mnist-otto-h512-e10 \
  --evalN 10000 --encoding latest \
  --predictions /tmp/mnist-preds.bin

# Step 2: visualize
./mnist/mnist-mlp-otto-vis-errors.exe \
  --predictions /tmp/mnist-preds.bin \
  --export vis/ --max 200
# → vis/index.html (browser-ready)
```

CIFAR-10 (color RGB PNG, 32×32):

```bash
./cifar/cifar-mlp-bin32-otto-trn-xnor.exe \
  --import models/cifar-otto-h256-e5 \
  --evalN 10000 --encoding latest \
  --predictions /tmp/cifar-preds.bin

./cifar/cifar-mlp-otto-vis-errors.exe \
  --predictions /tmp/cifar-preds.bin \
  --export vis-cifar/ --max 200
```

`cifar-mlp-otto-vis-errors.exe` uses the dataset-specific `ki_write_png()`
from `ki-local.h` (color RGB for CIFAR, grayscale for MNIST).

## Results Summary

| Approach                | MNIST     | CIFAR-10  | Fashion-MNIST | Hardware Target  |
| ----------------------- | --------- | --------- | ------------- | ---------------- |
| Otto Score (single)     | **99.0%** | **64.9%** | **90.2%**     | DRAM (bit-logic) |
| Otto Score (ensemble)   | 99.0%     | **64.9%** | 90.2%         | DRAM (bit-logic) |
| Bin32 Hebbian (bitwise) | 84.4%     | 32.4%     | 69.4%         | DRAM (bit-logic) |
| Float32 AdamW (matmul)  | 92.6%     | 44.8%     | 82.1%         | CPU/GPU          |

- **Otto Score**: MAJ3 + iterative Bayesian correction. Pure `&|~` + popcount.
  Better results via `--splitVN 2` (CIFAR) and ensemble (`--ensembleN 7`).
- **Hebbian**: Counter-based co-occurrence learning with multi-encoding members.
  CIFAR: 11 members (`--encoding latest`). MNIST/Fashion: single member (`latest` → exp8).
- **AdamW**: 1-layer float32 MLP (LeakyReLU, AdamW). Unified source with `--import` inference.

### Best Results (Latest)

| Configuration                                                                  | Dataset  | Accuracy   | Time                    |
| ------------------------------------------------------------------------------ | -------- | ---------- | ----------------------- |
| H=128, EN=7, ep=6, `--encoding latest`, evalN=100                              | MNIST    | **99.0%**  | **4s**                  |
| H=1024, ep=10, `--maj 1 --maj-thresh -2`, perf encoding, `--encoding-sizeN 32` | CIFAR-10 | **64.9%**  | 694s                    |
| H=512, ep=10, `--maj 1`, perf encoding, `--encoding-sizeN 16`                  | CIFAR-10 | **64.7%**  | **296s** (sweet spot)   |
| H=1024, EN=7, ep=7, `--splitVN 2`, `--encoding latest`                         | CIFAR-10 | **61.2%**  | 273s (legacy)           |
| H=1024, 132 filtered members, VN=2, `--encoding latest`                        | CIFAR-10 | **61.66%** | merge-ensemble (legacy) |

### Ensemble Workflow (quick overview)

```bash
# Train 20 random seeds (auto seed management)
bash bin/run-ensemble.sh --repeat 20 ./cifar/cifar-mlp-bin32-otto-trn-xnor.exe \
    --hiddenN 512 --epochsN 7 --encoding latest --splitVN 2

# Merge all seeds to EN accuracy curve
./cifar/cifar-merge-ensemble.exe scores/

# Filter by eval threshold
./cifar/cifar-merge-ensemble.exe scores/ --filter eval gt 58
```

### Implementation Details (2026-07)

- **maj=1 Default**: The old `--maj 3` tree was `--maj 1` with effective threshold
  ~52.7% (135/256). `--maj 1` is DRAM-native (bit-logic only) and conceptually cleaner.
  Threshold via `--maj-thresh -2` (auto per encoding), `-1` (n/2), or fixed value.
- **--export-gb**: gb_buf caching to `data/gb/`. Hash key covers dataset, W0[0..3],
  dimensions, input pipeline (xform, channels, encoding). Second run 35% faster.
  `--debug-cache` for cache logs.
- **Lazy Input Loading**: xform input buffers (load_input_cex_cached) are only loaded on
  gb-cache MISS. On cache HIT, no I/O for input buffers.
- **Xform Pipeline Chaining**: `@` syntax for sequential transforms.
  `rot90@avg4` = first rot90, then avg4. Any number of steps.
- **Row-Wise Average Filters**: `avg2`, `avg3`, `avg4` — running average per
  row with wrap-right. Spreads pixel information across neighbors.
- **--encoding-sizeN 0**: RAW encoding (no thermometer). 8-bit pixel values directly
  into containers. 99.6% train on Otto, 96.8% on BitVoting.
- **65% Ceiling Analysis**: The wall comes from the frozen W0, not from training.
  BitVoting (without W0) shows no ceiling — eval scales linearly with member count.
- **VN=2 Sweet Spot**: 25% retention via AND2 filter. Optimal for noisy data (CIFAR).
  For clean data (MNIST), VN=1 is better. VN=3+ (strict AND) only at very large H.
- **gb-cache optimization**: VN group mask computed once from h0 and reused for all
  epochs + evaluation. **−71% training time** (H=1024, EN=7: 934s→273s).
- **h0_eval Cache**: `evaluate_member` uses gb_buf/gb_buf_te for both evaluations
  (train + test). No h0_neuron during training.
- **Flat arrays removed**: `target_ens`/`offset_ens`/`best_ens`/`err_ens` eliminated.
  Each member stores its own targets.
- **Xforms + Encoding help low H, not the ceiling**: Geometric transforms (`--xform`)
  and encoding diversity (`--encoding`) improve accuracy at low hiddenN by providing
  more signal per member — but they do NOT raise the ~65% CIFAR-10 ceiling.
  Example: H=8 reaches 48.8% with 8 transforms vs 47.1% with 4 transforms (+1.7pp),
  but at H=512 both plateau at 64.4% vs 63.7% (+0.7pp). The ceiling is determined
  by H (hidden neurons), not by input diversity. More transforms or encoding members
  simply accelerate the convergence curve at each H — they cannot exceed the
  information capacity of H neurons.

## References

- **Bit-Voting Baseline**: Linear perceptron on bit-level, no W0, no majority.
  Proves the +5–7pp gap comes from nonlinear W0 projection, not information content.
  [`www/papers/bitvoting-baseline.html`](www/papers/bitvoting-baseline.html)
- **Float32 AdamW**: 1-layer MLP with LeakyReLU(0.05), MSE loss, AdamW optimizer.
  MNIST: 92.6% (10 ep). CIFAR: 41.2% (5 ep).
- **Bin32 Hebbian (legacy)**: The old single-member Hebbian (raw pixels, no encoding)
  achieved only 10% on CIFAR (random baseline). The new multi-member version with
  Thermometer encoding (`--encoding latest`, 11 members) reaches 32.4%.
- **VN data dependency**: MNIST clean → VN=1. CIFAR noisy → VN=2. See
  [`plans/plan-2026-07-08-vn3.md`](https://github.com/aotto1968/forward-prop/blob/master/plans/plan-2026-07-08-vn3.md).

## License

Public domain. Research code — use at your own risk.
