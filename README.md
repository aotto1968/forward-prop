# Otto Score — DRAM-Native MLP Classifier

**MNIST: 99.0%  |  CIFAR-10: 61.62%** — Zero floating point, zero matmul in inference.
Only `&|~` + int32 + popcount. Also includes float32 AdamW + multi-member Hebbian baselines.

---

## 🚀 Quick Start

```bash
# Step 1: Build everything (6 binaries)
make all

# Step 2: Download datasets (MNIST + CIFAR-10)
make setup

# Step 3: Train models + run tests (first run ~5min, then cached <1s)
make test
```

**Expected output (cached models):**

```
=== Otto Score MNIST (H=512, 10 ep)       ===      eval=98.8%
=== Float32 AdamW MNIST (H=512, 10 ep)    ===      eval=92.6%
=== Bin32 Hebbian MNIST (H=512, 10 ep)    ===      eval=84.4%
=== Otto Score CIFAR-10 (H=256, 5 ep)     ===      eval=55.0%
=== Float32 AdamW CIFAR-10 (H=256, 5 ep)  ===      eval=41.2%
=== Bin32 Hebbian CIFAR-10 (H=256, 5 ep)  ===      eval=32.4%
```

---

## 🎯 Test Targets

| Command                   | What it tests               |
| ------------------------- | --------------------------- |
| `make test-mnist`         | All 3 MNIST approaches      |
| `make test-mnist-otto`    | Otto Score MNIST only       |
| `make test-mnist-adam`    | Float32 AdamW MNIST only    |
| `make test-mnist-hebbian` | Bin32 Hebbian MNIST only    |
| `make test-cifar`         | All 3 CIFAR-10 approaches   |
| `make test-cifar-otto`    | Otto Score CIFAR-10 only    |
| `make test-cifar-adam`    | Float32 AdamW CIFAR-10 only |
| `make test-cifar-hebbian` | Bin32 Hebbian CIFAR-10 only |

First run trains all 6 models. Subsequent runs use cached models (<1s total).

---

## 📁 Directory Structure

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
│   └── merge-ensemble.c          ← Ensemble merge tool (companion to --save-scores)
├── cifar/                ← CIFAR-10 (symlinks to mnist/ + CIFAR ki-local.h)
│   ├── Makefile
│   ├── mlp-bin32-otto-trn.c      → ../mnist/...
│   ├── mlp-bin32-hebbian-trn.c   → ../mnist/...
│   ├── mlp-flt32-adam-trn.c      → ../mnist/...
│   ├── ki-common.h               → ../mnist/ki-common.h
│   └── ki-local.h                ← CIFAR dataset config
├── lib/                   ← Shared headers (ki-adamw.h, ki-encoding.h, maj3.h, w0_random.h)
├── models/                ← Cached trained models (-e10 MNIST, -e5 CIFAR)
├── bin/                   ← Helper scripts (run-ensemble.sh, etc.)
├── www/                   ← Publication site (HTML papers, style.css, datasets)
├── fetch_mnist.sh         ← MNIST download
├── fetch_cifar10.sh       ← CIFAR-10 download
└── README.md              ← this file
```

All 3 trainers (Otto, Hebbian, Adam) are **unified** across MNIST and CIFAR:
the same `.c` file is compiled with different `ki-local.h` per dataset.
Each trainer doubles as inference engine via `--import`. Zero code drift.

## Build Targets

| Command        | Builds                                            |
| -------------- | ------------------------------------------------- |
| `make` / `all` | All 6 binaries (Otto + Hebbian + Adam × XNOR/XOR) |
| `make otto`    | Otto Score only (mnist/ + cifar/)                 |
| `make hebbian` | Hebbian only (mnist/ + cifar/)                    |
| `make adam`    | Float32 AdamW only (mnist/ + cifar/)              |
| `make models`  | Train all 6 models (cached)                       |
| `make clean`   | Remove executables                                |
| `make ensemble`| Build merge-ensemble only (mnist/ + cifar/)       |

## CLI Flags (unified across all trainers)

| Flag                       | Description                                          | Default         |
| -------------------------- | ---------------------------------------------------- | --------------- |
| `--hiddenN N`              | Hidden neurons                                       | 64              |
| `--epochsN N`              | Training epochs                                      | 1               |
| `--splitVN N`              | Bit-Grouping: 1                                      | 1,2,3,4,8,16,32 |
| `--encoding TYPE`          | Input encoding (exp, sig, up8, down8, raw, etc.)     | raw8            |
| `--ensembleN N`            | Independent W0 copies                                | 1               |
| `--export DIR`             | Model export directory                               | none            |
| `--import DIR`             | Load model for inference                             | none            |
| `--predictions FILE`       | Export per-sample predictions (for vis-errors tool)  | none            |
| `--save-scores DIR`        | Save per-member scores to archive files (ensemble)   | none            |
| `--dry-run`                | Print architecture and exit (metadata only, instant) | off             |
| `--seed N`                 | Random seed                                          | 42              |
| `--seed-member MODE`       | Member seed strategy (once, const, incr)             | once            |
| `--target-err F`           | Target error threshold for early stopping            | 0.0             |
| `--multi-correct`          | Punish all wrong classes, not just argmax            | off             |
| `--batchN N`               | Mini-batch size                                      | 64              |
| `--debug-class-voting`     | Per-member per-class accuracy table                  | off             |
| `--debug-confusion-matrix` | Confusion matrix table                               | off             |
| `--filter CLASSES`         | Class subset (numeric or name, comma-sep)            | none            |
| `--qq`                     | Quick mode: 5000 train / 2000 eval / 3 ep            | off             |
| `--threadN N`              | OpenMP threads                                       | auto            |

**`--splitVN` — Bit-Grouping (nur Otto Score):**
- `--splitVN 1` (default): jedes Bit = eigenes Feature, 50% Retention. Optimal für clean data (MNIST).
- `--splitVN 2` (Süßer Punkt): AND2-Filter, nur `11` zählt, **25% Retention**. Optimal für noisy data (CIFAR-10).
- `--splitVN 3-32`: strict AND (alle Bits der Gruppe müssen 1 sein). Nur für sehr grosse H.

Die Retention (wieviel Signal passiert) bestimmt die VN-Wahl:
| VN    | Gruppen | Retention | Charakter                              |
| ----- | ------- | --------- | -------------------------------------- |
| 1     | 32      | 50.0%     | Soft — alles zählt                     |
| **2** | **16**  | **25.0%** | **Champion CIFAR** — harter AND        |
| 3     | 10      | 12.5%     | Braucht 8× mehr H für gleiche Leistung |
| 4     | 8       | 6.25%     | Verhungert bei kleinen H               |
| 8-32  | 4-1     | 0.4%-0%   | Nur bei H>16384+                       |

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

### 1. Train with `--save-scores DIR`

Each run creates one `.ens` archive file with all per-member int64 scores:

```bash
# Manual (one seed per call)
./cifar/cifar-mlp-bin32-otto-trn-xnor.exe --hiddenN 512 --epochsN 7 \
  --encoding latest --splitVN 2 --save-scores scores/ --seed 1234

./cifar/cifar-mlp-bin32-otto-trn-xnor.exe --hiddenN 512 --epochsN 7 \
  --encoding latest --splitVN 2 --save-scores scores/ --seed 5678

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
# MNIST Otto Score
./mnist/mnist-mlp-bin32-otto-trn-xnor.exe \
  --import models/mnist-otto-h512-e10 --evalN 10000 --encoding exp

# CIFAR-10 Otto Score (--encoding latest = 11 members)
./cifar/cifar-mlp-bin32-otto-trn-xnor.exe \
  --import models/cifar-otto-h256-e5 --evalN 10000 --encoding latest

# MNIST Hebbian
./mnist/mnist-mlp-bin32-hebbian-trn-xnor.exe \
  --import models/mnist-hebbian-h512-e10 --evalN 10000 --encoding exp

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
  --encoding exp --export models/mnist-otto/
# → writes models/mnist-otto/weights-0.meta, W0-0.bin, W1-0.bin, ...
```

Without `--export`, no files are written (training-only mode).

## Error Visualization via `--predictions`

Jeder Trainer kann per-sample predictions exportieren. Der integrierte
**Error Visualizer** erzeugt ein index.html mit allen Samples, sortiert
nach Klasse — Fehler rot markiert.

MNIST (grayscale PNG, 28×28):

```bash
# Schritt 1: predictions erzeugen
./mnist/mnist-mlp-bin32-otto-trn-xnor.exe \
  --import models/mnist-otto-h512-e10 \
  --evalN 10000 --encoding exp \
  --predictions /tmp/mnist-preds.bin

# Schritt 2: visualisieren
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

`cifar-mlp-otto-vis-errors.exe` nutzt dataset-spezifisches `ki_write_png()`
aus `ki-local.h` (color RGB für CIFAR, grayscale für MNIST).

## Results Summary

| Approach                | MNIST     | CIFAR-10   | Hardware Target  |
| ----------------------- | --------- | ---------- | ---------------- |
| Otto Score (single)     | **99.0%** | **61.0%**  | DRAM (bit-logic) |
| Otto Score (ensemble)   | 99.0%     | **61.62%** | DRAM (bit-logic) |
| Bin32 Hebbian (bitwise) | 84.4%     | 32.4%      | DRAM (bit-logic) |
| Float32 AdamW (matmul)  | 92.6%     | 41.2%      | CPU/GPU          |

- **Otto Score**: MAJ3 + iterative Bayesian correction. Pure `&|~` + popcount.
  Bessere Ergebnisse durch `--splitVN 2` (CIFAR) und Ensemble (`--ensembleN 7`).
- **Hebbian**: Counter-based co-occurrence learning with multi-encoding members.
  MNIST: single member (exp8). CIFAR: 11 members (`--encoding latest`).
- **AdamW**: 1-layer float32 MLP (LeakyReLU, AdamW). Unified source with `--import` inference.

### Best Results (Latest)

| Konfiguration                                             | Dataset  | Accuracy   | Zeit                           |
| --------------------------------------------------------- | -------- | ---------- | ------------------------------ |
| H=128, EN=7, ep=8, `--encoding exp8,log8,sig8`, evalN=100 | MNIST    | **99.0%**  | ~30s                           |
| H=1024, EN=7, ep=7, `--splitVN 2`, `--encoding latest`    | CIFAR-10 | **61.2%**  | **273s** (−71% durch gb-cache) |
| H=1024, 17 seeds, VN=2, `--encoding latest` (Ensemble)    | CIFAR-10 | **61.62%** | merge-ensemble                 |
| H=4094, EN=3, ep=7, VN=2, target-err=0.4 (single)         | CIFAR-10 | **61.0%**  | 603s                           |

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

## Key Findings (2026-07)

- **VN=2 Sweet Spot**: 25% Retention durch AND2-Filter. Optimal für noisy data (CIFAR).
  Für clean data (MNIST) ist VN=1 besser. VN=3+ (strict AND) nur bei sehr grossem H.
- **gb-cache Optimierung**: VN-Gruppenmaske wird 1× aus h0 berechnet und für alle
  Epochen + Evaluation wiederverwendet. **−71% Trainingszeit** (H=1024, EN=7: 934s→273s).
- **Ensemble überwindet Ceiling**: Mehrere Seeds + merge-ensemble erreichen 61.62%
  (H=1024, 17 seeds). Single-Run ceiling: 61.0% (H=4094).
- **h0_eval Cache**: `evaluate_member` nutzt gb_buf/gb_buf_te für beide Evaluationen
  (train + test). Kein h0_neuron mehr während des Trainings.
- **Flat arrays entfernt**: `target_ens`/`offset_ens`/`best_ens`/`err_ens` eliminiert.
  Jeder Member speichert seine Targets selbst.

## References

- **Float32 AdamW**: 1-layer MLP with LeakyReLU(0.05), MSE loss, AdamW optimizer.
  MNIST: 92.6% (10 ep). CIFAR: 41.2% (5 ep).
- **Bin32 Hebbian (legacy)**: The old single-member Hebbian (raw pixels, no encoding)
  achieved only 10% on CIFAR (random baseline). The new multi-member version with
  Thermometer encoding (`--encoding latest`, 11 members) reaches 32.4%.
- **VN-Datenabhängigkeit**: MNIST clean → VN=1. CIFAR noisy → VN=2. Siehe
  [`plans/plan-2026-07-08-vn3.md`](https://github.com/aotto1968/forward-prop/blob/master/plans/plan-2026-07-08-vn3.md).

## License

Public domain. Research code — use at your own risk.
