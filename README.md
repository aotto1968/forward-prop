# Otto Score — DRAM-Native MLP Classifier

**MNIST: 97.0%  |  CIFAR-10: 55.0%** — Zero floating point, zero matmul in inference.
Only `&|~` + int32 + popcount. Also includes float32 AdamW + multi-member Hebbian baselines.

---

## 🚀 Quick Start

```bash
# Step 1: Build everything (12 binaries)
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

| Command                   | What it tests                                  |
| ------------------------- | ---------------------------------------------- |
| `make test-mnist`         | All 3 MNIST approaches                         |
| `make test-mnist-otto`    | Otto Score MNIST only                          |
| `make test-mnist-adam`    | Float32 AdamW MNIST only                       |
| `make test-mnist-hebbian` | Bin32 Hebbian MNIST only                       |
| `make test-cifar`         | All 3 CIFAR-10 approaches                      |
| `make test-cifar-otto`    | Otto Score CIFAR-10 only                       |
| `make test-cifar-adam`    | Float32 AdamW CIFAR-10 only                    |
| `make test-cifar-hebbian` | Bin32 Hebbian CIFAR-10 only                    |

First run trains all 6 models. Subsequent runs use cached models (<1s total).

---

## 📁 Directory Structure

```
otto-score-ifc/
├── Makefile              ← orchestrates: delegates to subdirectories
├── mnist/                ← MNIST (Otto Score + Hebbian, unified sources)
│   ├── Makefile
│   ├── mlp-bin32-otto-trn.c      ← Otto Score (shared source)
│   ├── mlp-bin32-hebbian-trn.c   ← Hebbian (shared source)
│   ├── ki-common.h               ← Shared header (modern API)
│   └── ki-local.h                ← MNIST dataset config
├── cifar/                ← CIFAR-10 (Otto Score + Hebbian)
│   ├── Makefile
│   ├── mlp-bin32-otto-trn.c      → ../mnist/...
│   ├── mlp-bin32-hebbian-trn.c   → ../mnist/...
│   ├── ki-common.h               → ../mnist/ki-common.h
│   └── ki-local.h                ← CIFAR dataset config
├── reference/             ← AdamW references (MNIST + CIFAR)
│   ├── Makefile
│   ├── cifar-include/            ← CIFAR headers for reference builds
│   └── mnist-mlp-flt32-*.c       ← AdamW (trainer + inference)
├── lib/                   ← Shared headers (maj3.h, w0_random.h, ki-encoding.h)
├── models/                ← Cached trained models  (-e10 MNIST, -e5 CIFAR)
├── fetch_mnist.sh         ← MNIST download
├── fetch_cifar10.sh       ← CIFAR-10 download
└── README.md              ← this file
```

## Build Targets

| Command          | Builds                                                     |
| ---------------- | ---------------------------------------------------------- |
| `make` / `all`   | All 12 binaries (4 MNIST + 4 CIFAR + 4 reference)         |
| `make otto`      | Otto Score only (mnist/ + cifar/)                          |
| `make hebbian`   | Hebbian only (mnist/ + cifar/)                             |
| `make adam`      | Float32 AdamW references (reference/)                      |
| `make models`    | Train all 6 models (cached)                                |
| `make clean`     | Remove executables                                         |
| `make clean-all` | Remove executables + cached models                         |

## Inference via `--model`

Every training binary doubles as inference engine via `--model`:

```bash
# MNIST Otto Score
./mnist/mnist-mlp-bin32-otto-trn-xnor.exe \
  --model models/mnist-otto-h512-e10/model.otto --evalN 10000 --encoding exp

# CIFAR-10 Otto Score (--encoding latest = 11 members)
./cifar/cifar-mlp-bin32-otto-trn-xnor.exe \
  --model models/cifar-otto-h256-e5/model.otto --evalN 10000 --encoding latest

# MNIST Hebbian
./mnist/mnist-mlp-bin32-hebbian-trn-xnor.exe \
  --model models/mnist-hebbian-h512-e10 --evalN 10000 --encoding exp

# CIFAR Hebbian (multi-member, 11 members via --encoding latest)
./cifar/cifar-mlp-bin32-hebbian-trn-xnor.exe \
  --model models/cifar-hebbian-h256-e5 --evalN 10000 --encoding latest

# AdamW references (separate IFC binaries)
./reference/mnist-mlp-flt32-adam-ifc.exe \
  --model models/mnist-adam-h512-e10 --evalN 10000
./reference/cifar-mlp-flt32-adam-ifc.exe \
  --model models/cifar-adam-h256-e5 --evalN 10000
```

There are **no separate IFC source files** for Otto Score or Hebbian — the trainer binary IS the inference binary. Zero code drift between training and evaluation.

## Results Summary

All approaches use the same dataset split and are comparable at equal H:

| Approach                | MNIST H=512  | CIFAR H=256  | Hardware Target  |
| ----------------------- | ------------ | ------------ | ---------------- |
| Otto Score (bitwise)    | **97.0%**    | **55.0%**    | DRAM (bit-logic) |
| Bin32 Hebbian (bitwise) | **84.4%**    | **32.4%**    | DRAM (bit-logic) |
| Float32 AdamW (matmul)  | 92.6%        | 41.2%        | CPU/GPU          |

- **Otto Score**: MAJ3 + iterative Bayesian correction. Pure `&|~` + popcount.
- **Hebbian**: Counter-based co-occurrence learning with multi-encoding members.
  MNIST: single member (exp8). CIFAR: 11 members (`--encoding latest`).
- **AdamW**: 2-layer float32 MLP (LeakyReLU, AdamW). Reference baseline.

## References

- **Float32 AdamW**: 2-layer MLP with LeakyReLU(0.05), MSE loss, AdamW optimizer.
  MNIST: 92.6% (10 ep). CIFAR: 41.2% (5 ep).
- **Bin32 Hebbian (legacy)**: The old single-member Hebbian (raw pixels, no encoding)
  achieved only 10% on CIFAR (random baseline). The new multi-member version with
  Thermometer encoding (`--encoding latest`, 11 members) reaches 32.4%.

## License

Public domain. Research code — use at your own risk.
