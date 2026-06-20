# Otto Score Inference — DRAM-Native MNIST Classifier

**95.7% MNIST (full 10000-test evaluation). Zero floating point. Zero training. Only `&|~` + int32.**

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
  Eval:    95.1%  (9512/10000)
  Time:    167ms  (16.7 µs/sample)

=== XOR ensemble v6 (H=128x3, --evalN 10000) ===
  Eval:    94.7%  (9467/10000)
  Time:    153ms  (15.3 µs/sample)
```

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

## 📖 How does it actually work? (simplified)

---

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
├── README.md                  ← this file
├── Makefile                   ← build: make all / make test / make setup
├── fetch_mnist.sh           ← MNIST download script
├── convert_to_pgm.sh        ← image → MNIST PGM converter
├── mlp-otto-score-ifc.c     ← inference source code (~460 lines)
├── ki-common.h                ← MNIST loader (only what's needed)
├── lib/
│   ├── maj3.h                 ← MAJ3 majority_tree algorithm
│   └── w0_random.h            ← splitmix64 random number generator
├── models/
│   ├── model-xnor.otto       ← pre-trained XNOR model (v1 single, H=512)
│   ├── model-xor.otto        ← pre-trained XOR model (v1 single, H=512)
│   ├── model-ensemble-xnor.otto ← ensemble XNOR model (v5, H=128×N=3)
│   └── model-ensemble-xor.otto  ← ensemble XOR model (v5, H=128×N=3)
├── tests/
│   ├── digit5.pgm           ← MNIST test image (label=5, viewable!)
│   ├── digit9.pgm           ← MNIST test image (label=9, viewable!)
│   └── shoe.pgm             ← Fashion-MNIST boot (NOT a digit!)
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

## 📖 Deeper reading

- **Research results**: [forward-prop.nhi1.de](https://forward-prop.nhi1.de/)
- **The vision**: [forward-prop.nhi1.de/papers/vision.html](https://forward-prop.nhi1.de/papers/vision.html)
- **Full source**: [github.com/aotto1968/forward-prop](https://github.com/aotto1968/forward-prop)

---

## 📝 License

Public domain. This is research code — use at your own risk.
