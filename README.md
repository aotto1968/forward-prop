# Otto Score Inference — DRAM-Native MNIST Classifier (+ Float32 Reference)

**95.6% MNIST (full 10000-test evaluation). Zero floating point. Zero training. Only `&|~` + int32.**

**Also includes a 2-layer float32 AdamW baseline for comparison: `mlp-flt32-trn-w1-adam` + `mlp-flt32-ifc`.**

**Supports single (v1) and ensemble (v5) models — auto-detected on load.**
**Precision scaling via OT_PRECISION — default F=1024, legacy models at F=131072.**

---

## 🚀 First Run (5 minutes)

This is a **self-contained demo** that classifies handwritten digits using
only bit-logic operations. No GPU, no PyTorch, no training, no floating point.

---

## 🚀 First Run (5 minutes)

```bash
# Step 1: Build the executables
make all

# Step 2: Download MNIST data (70000 handwritten digits)
make setup

# Step 3: Run the test
make test
```

**Expected output:**

```
=== XNOR v1 single (H=512, --evalN 10000) ===
  Eval:    95.4%  (9536/10000)
  Time:    210ms  (21.0 µs/sample)

=== XOR v1 single (H=512, --evalN 10000) ===
  Eval:    95.4%  (9536/10000)
  Time:    210ms  (21.0 µs/sample)

=== XNOR ensemble v6 (H=128x3, --evalN 10000) ===
  Eval:    95.6%  (9555/10000)
  Time:    163ms  (16.3 µs/sample)

=== XOR ensemble v6 (H=128x3, --evalN 10000) ===
  Eval:    95.6%  (9561/10000)
  Time:    157ms  (15.7 µs/sample)

=== Float32 AdamW (H=512, same bit-mass, --evalN 10000) ===
  Eval:    92.6%  (9260/10000)
  Time:    50ms  (5.0 µs/sample)
```

The Otto Score (bitwise) outperforms the AdamW float32 baseline at **equal bit-mass** (H=512) while using **zero multiplication** and **zero floating point** during inference.

The first `make test` run will auto-train the float32 model (~9 seconds). Subsequent runs use the cached model in `models/flt32-w1-h512/`.

That's it. You just classified handwritten digits with 96% accuracy
using only `&`, `|`, `~` — no multiply, no float, no training.

---

## ❓ What just happened?

| You typed…   | What happened                                                |
| ------------ | ------------------------------------------------------------ |
| `make all`   | Compiled the C inference code → two executables              |
| `make setup` | Downloaded 60000 MNIST training + 10000 test digits          |
| `make test`  | Loaded a pre-trained model (v1 single), ran 2000 test digits |

The two executables are:

| File                          | What it does                        |
| ----------------------------- | ----------------------------------- |
| `mlp-otto-score-ifc-xnor.exe` | XNOR mode (default, model included) |
| `mlp-otto-score-ifc-xor.exe`  | XOR mode (model included)           |

Both auto-detect single (v1) and ensemble (v5) model formats.
All values are scaled by `F = (1<<OT_PRECISION)` — default F=1024 for v5,
F=131072 for legacy v1 models. Display divides by the correct F.

**Float32 reference executables (build separately with `make flt32`):**

| File                          | What it does                        |
| ----------------------------- | ----------------------------------- |
| `mlp-flt32-trn-w1-adam.exe`   | 2-layer AdamW trainer (float32)     |
| `mlp-flt32-ifc.exe`           | 2-layer inference (float32)         |

See "Float32 2-Layer Reference" section below.

---

## 📊 What parameters can I change?

```
./mlp-otto-score-ifc-xnor.exe --model models/model-xnor.otto --evalN 10000 --threadN 4
```

| Flag           | What it does                                   | Default  |
| -------------- | ---------------------------------------------- | -------- |
| `--model PATH` | Which `.otto` model file to load (v1 or v5)    | required |
| `--evalN N`    | How many digits to classify                    | 10000    |
| `--image FILE` | Classify a single image (raw 28x28, 784 bytes) | off      |
| `--threadN N`  | CPU threads for parallel processing            | 8        |

**Try these experiments:**

```bash
# Full evaluation on all 10000 test digits
./mlp-otto-score-ifc-xnor.exe --model models/model-xnor.otto --evalN 10000
# Expected: ~86% (1-pass model, not iteratively trained)

# Fewer threads (slower, but works on any machine)
./mlp-otto-score-ifc-xnor.exe --model models/model-xnor.otto --evalN 10000 --threadN 2

# Quick check with only 100 digits
./mlp-otto-score-ifc-xnor.exe --model models/model-xnor.otto --evalN 100
```

## 🖼️ Classify your own handwritten digit

Take a photo, convert it, classify it:

```bash
# Convert to PGM (Portable GrayMap — standard image format)
convert mydigit.jpg -resize 28x28! -negate -depth 8 pgm:- > digit.pgm

# Classify it!
./mlp-otto-score-ifc-xnor.exe --model models/model-xnor.otto --image digit.pgm
```

The classifier accepts **PGM (P5)** images or raw 784-byte files.
PGM is a standard format — every image viewer can open it.

**Expected output:**

```
══╡ SINGLE IMAGE ╞════════════════════════════════════════════════
  File:  digit.pgm
  Pixel mean: 35.1  (0=white, 255=black, 28x28)

  Scores:
    0: -5577.13
    5: -5230.35  ← PREDICTED
    ...

  >>> Predicted digit: 5 <<<
```

### ⚠️ The classifier has no "reject" class

It only knows digits 0-9. Feed it a shoe, a cat, or random noise — it will always predict one of the 10 digits:

```bash
make test-image
```

```
  tests/shoe.pgm (Fashion-MNIST ankle boot — should NOT be a digit)
  >>> Predicted digit: 2 <<<
```

The shoe is classified as "2" because the model has never seen
a shoe in training. There is no "unknown" / "none of the above" class.

## 🤔 What should I expect to see?

Each run prints:

```
Model:   H=512  XNOR          ← model architecture
Eval:    96.3%  (1927/2000)   ← accuracy (correct / total)
Time:    43ms  (21.5 µs/sample)  ← speed
```

The accuracy varies slightly between runs (±0.1pp) due to OpenMP
thread scheduling. This is normal — the model itself is deterministic.

---

## 📁 What's in this directory?

```
├── README.md                        ← this file
├── Makefile                         ← build: make all / make test / make setup
├── fetch_mnist.sh                   ← MNIST download script
├── convert_to_pgm.sh                ← image → MNIST PGM converter
├── convert_to_raw.sh                ← image → raw 784-byte converter
├── mlp-otto-score-ifc.c             ← Otto Score inference (~440 lines)
├── mlp-flt32-trn-w1-adam.c          ← Float32 AdamW trainer (~440 lines)
├── mlp-flt32-ifc.c                  ← Float32 2-layer inference (~300 lines)
├── ki-common.h                      ← Shared infrastructure (MNIST, matmul, AdamW helpers)
├── lib/
│   ├── maj3.h                       ← MAJ3 majority_tree algorithm
│   └── w0_random.h                  ← splitmix64 random number generator
├── models/
│   ├── model-xnor.otto             ← pre-trained XNOR model (v1 single, H=512)
│   ├── model-xor.otto              ← pre-trained XOR model (v1 single, H=512)
│   ├── model-ensemble-xnor.otto    ← ensemble XNOR model (v5, H=128×N=3)
│   └── model-ensemble-xor.otto     ← ensemble XOR model (v5, H=128×N=3)
├── tests/
│   ├── digit5.pgm                  ← MNIST test image (label=5, viewable!)
│   ├── digit9.pgm                  ← MNIST test image (label=9, viewable!)
│   └── shoe.pgm                    ← Fashion-MNIST boot (NOT a digit!)
└── .gitignore
```

---

## 🧠 How does it actually work? (simplified)

```
  Your digit (28×28 pixels)
        │
        ▼
  Pack 4 pixels → 1 uint32  (784 px → 196 numbers)
        │
        ▼
  For each of 512 neurons:
    XNOR with random W0  →  32-bit pattern
        │
        ▼
  Bayes log-score: sum up evidence for each digit class
        │
        ▼
  Pick the class with the highest score
```

**Key insight:** The random projection W0 is **never trained**.
It's initialized once with random numbers and stays frozen.
The model only learns which bit patterns correlate with which digit —
by counting, not by gradient descent.

For ensemble models (v5), all members are evaluated independently and
their scores are summed before argmax:
```
score[k] = Σ_m offset_m[k] + Σ_m Σ_h Σ_b y_m[h][b] × target_m[k][h][b]
```

## 🤝 Ensemble Voting (v5 format)

Training multiple independent W0s and summing their scores (product of experts)
reduces the error rate significantly. The ifc **auto-detects** ensemble models
on load — no extra flags needed.

```bash
# Train an ensemble (from ki-w2 directory)
cd ../ki-w2
./mlp-otto-score-ensemble-xnor.exe --hiddenN 128 --ensembleN 3 --epochsN 20 --out out/otto-h128-e3-xnor

# Evaluate with inference (auto-detects v5 ensemble format)
cd ../otto-score-ifc
./mlp-otto-score-ifc-xnor.exe --model ../ki-w2/out/otto-h128-e3-xnor/model.otto --evalN 10000
# → 95.5%
```

The ensemble is stored as a **single file** containing all members' W0, targets,
and offsets. Inference iterates over all members and sums the scores:

```
score[k] = Σ_m score_m[k]    (member m, class k)
```

| Configuration                   | Eval      | Notes                      |
| ------------------------------- | --------- | -------------------------- |
| H=512, N=1, ep=20 (bundled v1)  | 96.3%     | Legacy single model        |
| H=128, N=3, ep=20 (v5 ensemble) | **95.5%** | 3× smaller, ~3× faster     |
| H=64, N=17, ep=20 (v5 ensemble) | **96.4%** | Same accuracy, 32× less W0 |

---

## 🧪 Float32 2-Layer Reference (Baseline)

This directory also includes a **2-layer float32 AdamW baseline** for comparison.
It uses standard matmul + Leaky ReLU + AdamW — **no bitwise ops, no MAJ3**.

```
Architecture:  W0 (float, random frozen) → LReLU(0.05) → W1 (float, AdamW)
Training:      MSE(±1 targets), AdamW(lr=0.002, wd=1e-4), warmup + cosine decay
Inference:     matmul(W0, x) → LReLU → matmul(W1, h0) → argmax

Typical:       H=512 → 95%+ eval (MNIST, 10 epochs)
```

### Build & Train (reference)

```bash
make flt32
./mlp-flt32-trn-w1-adam.exe --hiddenN 512 --epochsN 10 --out out/flt32-ref
# → Evaluates on 10000 test digits, exports weights to out/flt32-ref/
```

### Run inference on exported model

```bash
# Evaluate MNIST test set
./mlp-flt32-ifc.exe --model out/flt32-ref --evalN 10000

# Classify a single image (PGM or raw 784 bytes)
./mlp-flt32-ifc.exe --model out/flt32-ref --image tests/digit5.pgm
```

### Key differences vs Otto Score

| Aspect            | Otto Score (bitwise)          | Float32 Reference            |
| ----------------- | ----------------------------- | ---------------------------- |
| Forward           | XNOR + MAJ3 + Bayes log-score | matmul + LReLU               |
| Training          | Iterative correction          | AdamW (backprop)             |
| Hardware target   | DRAM (bit-logic)              | CPU/GPU (matmul)             |
| No. of formats    | int32 Target + Offset         | float32 W0 + W1              |
| Best eval (H=512) | 96.3%                         | 95.3%                        |

The Otto Score outperforms the AdamW baseline on MNIST at equal H,
while using **zero floating point** and **zero multiplication** during inference.

---

## 📖 Deeper reading

- **Research results**: [forward-prop.nhi1.de](https://forward-prop.nhi1.de/)
- **The vision**: [forward-prop.nhi1.de/papers/vision.html](https://forward-prop.nhi1.de/papers/vision.html)
- **Full source**: [github.com/aotto1968/forward-prop](https://github.com/aotto1968/forward-prop)

---

## 📝 License

Public domain. This is research code — use at your own risk.
