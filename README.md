# Otto Score — DRAM-Native MLP Classifier

**MNIST: 97.0%  |  CIFAR-10: 55.0%** — Zero floating point, zero matmul in inference.
Only `&|~` + int32 + popcount. Also includes float32 AdamW + Hebbian baselines.

---

## 🚀 Quick Start

```bash
# Step 1: Build everything
make all

# Step 2: Download datasets (MNIST + CIFAR-10)
make setup

# Step 3: Run tests (trains models on first run, cached thereafter)
make test
```

**Expected output (second run — cached models, <1s total):**

```
=== Otto Score MNIST (H=512, 10 ep)       ===      eval=97.0%
=== Float32 AdamW MNIST (H=512, 10 ep)    ===      eval=92.6%
=== Bin32 Hebbian MNIST (H=512, 10 ep)    ===      eval=82.9%
=== Otto Score CIFAR-10 (H=256, 5 ep)     ===      eval=55.0%
=== Float32 AdamW CIFAR-10 (H=256, 3 ep)  ===      eval=39.4%
=== Bin32 Hebbian CIFAR-10 (H=256, 3 ep)  ===      eval=10.0%
```

---

## 🎯 Test Targets (fast — <1s with cached models)

| Command                   | What it tests                                  |
| ------------------------- | ---------------------------------------------- |
| `make test-mnist`         | All 3 MNIST approaches (Otto + Adam + Hebbian) |
| `make test-mnist-otto`    | Otto Score MNIST only                          |
| `make test-mnist-adam`    | Float32 AdamW MNIST only                       |
| `make test-mnist-hebbian` | Bin32 Hebbian MNIST only                       |
| `make test-cifar`         | All 3 CIFAR-10 approaches                      |
| `make test-cifar-otto`    | Otto Score CIFAR-10 only                       |
| `make test-cifar-adam`    | Float32 AdamW CIFAR-10 only                    |
| `make test-cifar-hebbian` | Bin32 Hebbian CIFAR-10 only                    |

First run trains models (~2-5 min depending on CPU). Subsequent runs use cached models.

---

## 📁 Directory Structure

```
otto-score-ifc/
├── Makefile              ← orchestrates: delegates to subdirectories
├── mnist/                ← MNIST Otto Score (trainer + inference via --model)
│   ├── Makefile
│   ├── mlp-bin32-otto-trn.c    ← Otto Score trainer (shared source)
│   └── ki-common.h             ← symlink to mnist-1/ki-common.h
├── cifar/                ← CIFAR-10 Otto Score (trainer + inference via --model)
│   ├── Makefile
│   ├── mlp-bin32-otto-trn.c    ← same source as mnist/ (CIFAR via ki-local.h)
│   ├── ki-common.h             ← symlink to cifar-1/ki-common.h
│   └── ki-local.h              ← symlink to cifar-1/ki-local.h
├── reference/             ← Reference implementations (AdamW + Hebbian)
│   ├── Makefile
│   ├── cifar-include/          ← CIFAR headers for reference builds
│   ├── mnist-mlp-*.c           ← MNIST AdamW + Hebbian (trainer + inference)
│   └── cifar-mlp-*.c           ← CIFAR AdamW + Hebbian (trainer + inference)
├── lib/                   ← Shared headers (maj3.h, w0_random.h, enc-lut.h)
├── models/                ← Cached trained models
├── fetch_mnist.sh         ← MNIST download script
├── fetch_cifar10.sh       ← CIFAR-10 download script
└── tests/                 ← Test images (single-image classification)
```

## Build Targets

| Command              | Builds                                                     |
| -------------------- | ---------------------------------------------------------- |
| `make` or `make all` | All 14 binaries (MNIST Otto + CIFAR Otto + all references) |
| `make otto`          | Otto Score only (mnist/ + cifar/)                          |
| `make adam`          | Float32 AdamW references (reference/)                      |
| `make hebbian`       | Bin32 Hebbian references (reference/)                      |
| `make clean`         | Removes all executables + cached models                    |

## How Inference Works

Every trainer binary doubles as IFC via `--model`:

```bash
# MNIST: load cached model and evaluate
./mnist/mnist-mlp-bin32-otto-trn-xnor.exe \
  --model models/mnist-otto-h512/model.otto --evalN 10000 --encoding exp

# CIFAR-10: same pattern (--encoding latest for best accuracy)
./cifar/cifar-mlp-bin32-otto-trn-xnor.exe \
  --model models/cifar-otto-h256/model.otto --evalN 10000 --encoding latest

# Reference inference uses separate IFC binaries:
./reference/mnist-mlp-flt32-adam-ifc.exe \
  --model models/mnist-adam-h512 --evalN 10000
```

There are **no separate IFC source files** for Otto Score — the trainer binary IS the inference binary. This guarantees zero code drift between training and evaluation.

## Naming Convention

All binaries follow: `(dataset)-mlp-(bit|flt)(32)-(otto|adam|hebbian)-(trn|ifc)(-xnor|-xor)?.exe`

| Part                | Meaning                             |
| ------------------- | ----------------------------------- |
| `mnist-` / `cifar-` | Dataset prefix                      |
| `mlp`               | Multilayer perceptron               |
| `bin32` / `flt32`   | uint32 containers / float weights   |
| `otto`              | Otto Score (MAJ3 + Bayes)           |
| `adam`              | Float32 AdamW backprop              |
| `hebbian`           | Bitwise Hebbian (no backprop)       |
| `trn`               | Trainer (also IFC via `--model`)    |
| `ifc`               | Inference-only (reference binaries) |
| `-xnor` / `-xor`    | H0 mode                             |

## Dataset Setup

```bash
make setup-mnist    # Download MNIST (60000 train + 10000 test)
make setup-cifar   # Download CIFAR-10 (50000 train + 10000 test)
make setup         # Both
```

Data is stored in `www/data/mnist/` and `www/data/cifar-10/` (project root).

## Architecture Overview

```
  Input pixels (784 / 3072)
        │
        ▼
  Pack → uint32 containers (196 / 768)
        │
        ▼
  MAJ3 (XNOR/XOR) + popcount → 32-bit H0
        │
        ▼
  Bayes log-score → 10 class scores
        │
        ▼
  argmax → predicted class
```

- **W0**: Frozen random projection (never trained)
- **W1**: Trained via iterative correction (Otto Score) or AdamW (float32 ref)
- **No floating point** in Otto Score inference — only `&|~` + popcount + int32 add

## References

- **Float32 AdamW**: 2-layer MLP with LeakyReLU(0.05), MSE loss, AdamW optimizer.
  Achieves 92.6% on MNIST, 39.4% on CIFAR-10 (H=256, 3 epochs).
- **Bin32 Hebbian**: Bitwise co-occurrence learning with sign flips.
  Achieves 82.9% on MNIST, 10.0% on CIFAR-10 (random baseline — does not converge).

## Results Summary

| Approach                | MNIST     | CIFAR-10  | Hardware Target  |
| ----------------------- | --------- | --------- | ---------------- |
| Otto Score (bitwise)    | **97.0%** | **55.0%** | DRAM (bit-logic) |
| Float32 AdamW (matmul)  | 92.6%     | 39.4%     | CPU/GPU          |
| Bin32 Hebbian (bitwise) | 82.9%     | 10.0%     | DRAM (bit-logic) |

## License

Public domain. Research code — use at your own risk.
