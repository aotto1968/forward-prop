# Otto Score — DRAM-Native MLP Classifier

**MNIST: 97.0%  |  CIFAR-10: 55.0%** — Zero floating point, zero matmul in inference.
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
=== Otto Score MNIST (H=512, 10 ep)       ===      eval=97.0%
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
│   └── ki-local.h                ← MNIST dataset config
├── cifar/                ← CIFAR-10 (symlinks to mnist/ + CIFAR ki-local.h)
│   ├── Makefile
│   ├── mlp-bin32-otto-trn.c      → ../mnist/...
│   ├── mlp-bin32-hebbian-trn.c   → ../mnist/...
│   ├── mlp-flt32-adam-trn.c      → ../mnist/...
│   ├── ki-common.h               → ../mnist/ki-common.h
│   └── ki-local.h                ← CIFAR dataset config
├── lib/                   ← Shared headers (ki-adamw.h, ki-encoding.h, maj3.h, w0_random.h)
├── models/                ← Cached trained models (-e10 MNIST, -e5 CIFAR)
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

## CLI Flags (unified across all trainers)

| Flag                   | Description                                          | Default |
| ---------------------- | ---------------------------------------------------- | ------- |
| `--hiddenN N`          | Hidden neurons                                       | 64      |
| `--epochsN N`          | Training epochs                                      | 1       |
| `--encoding TYPE`      | Input encoding (exp, sig, up8, down8, raw, etc.)     | raw8    |
| `--ensembleN N`        | Independent W0 copies                                | 1       |
| `--export DIR`         | Model export directory                               | none    |
| `--import DIR`         | Load model for inference                             | none    |
| `--predictions FILE`   | Export per-sample predictions (for vis-errors tool)  | none    |
| `--dry-run`            | Print architecture and exit (metadata only, instant) | off     |
| `--seed N`             | Random seed                                          | 42      |
| `--seed-member MODE`   | Member seed strategy (once, const, incr)             | once    |
| `--batchN N`           | Mini-batch size                                      | 64      |
| `--debug-class-voting` | Per-member per-class accuracy table                  | off     |

Backward-compat aliases: `--out` = `--export`, `--model` = `--import`.

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

All approaches use the same dataset split and are comparable at equal H:

| Approach                | MNIST H=512 | CIFAR H=256 | Hardware Target  |
| ----------------------- | ----------- | ----------- | ---------------- |
| Otto Score (bitwise)    | **97.0%**   | **55.0%**   | DRAM (bit-logic) |
| Bin32 Hebbian (bitwise) | **84.4%**   | **32.4%**   | DRAM (bit-logic) |
| Float32 AdamW (matmul)  | 92.6%       | 41.2%       | CPU/GPU          |

- **Otto Score**: MAJ3 + iterative Bayesian correction. Pure `&|~` + popcount.
- **Hebbian**: Counter-based co-occurrence learning with multi-encoding members.
  MNIST: single member (exp8). CIFAR: 11 members (`--encoding latest`).
- **AdamW**: 1-layer float32 MLP (LeakyReLU, AdamW). Unified source with `--import` inference.

## References

- **Float32 AdamW**: 1-layer MLP with LeakyReLU(0.05), MSE loss, AdamW optimizer.
  MNIST: 92.6% (10 ep). CIFAR: 41.2% (5 ep).
- **Bin32 Hebbian (legacy)**: The old single-member Hebbian (raw pixels, no encoding)
  achieved only 10% on CIFAR (random baseline). The new multi-member version with
  Thermometer encoding (`--encoding latest`, 11 members) reaches 32.4%.

## License

Public domain. Research code — use at your own risk.
