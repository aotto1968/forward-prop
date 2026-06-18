# Otto Score Inference — DRAM-Native MNIST Classifier

**96.4% MNIST. Zero floating point. Zero training. Only `&|~` + int32.**

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
=== XNOR inference with XNOR-trained model ===
  Model:   H=512  XNOR
  Eval:    96.3%  (1927/2000)

=== XOR inference with XOR-trained model ===
  Model:   H=512  XOR
  Eval:    96.7%  (1934/2000)
```

That's it. You just classified handwritten digits with 96% accuracy
using only `&`, `|`, `~` — no multiply, no float, no training.

---

## ❓ What just happened?

| You typed… | What happened |
|:-----------|:--------------|
| `make all` | Compiled the C inference code → two executables |
| `make setup` | Downloaded 60000 MNIST training + 10000 test digits |
| `make test` | Loaded a pre-trained model, ran 2000 test digits, reported accuracy |

The two executables are:

| File | What it does |
|:-----|:-------------|
| `mlp-otto-score-ifc-xnor.exe` | XNOR mode (default, model included) |
| `mlp-otto-score-ifc-xor.exe` | XOR mode (model included) |

Both use the **same math**. XOR saves one NOT gate per bit on hardware.

---

## 📊 What parameters can I change?

```
./mlp-otto-score-ifc-xnor.exe --model models/model-xnor.otto --evalN 10000 --threadN 4
```

| Flag | What it does | Default |
|:-----|:-------------|:--------|
| `--model PATH` | Which `.otto` model file to load | required |
| `--evalN N` | How many digits to classify | 10000 |
| `--threadN N` | CPU threads for parallel processing | 8 |

**Try these experiments:**

```bash
# Full evaluation on all 10000 test digits
./mlp-otto-score-ifc-xnor.exe --model models/model-xnor.otto --evalN 10000
# Expected: 96.3% (9630/10000)

# Fewer threads (slower, but works on any machine)
./mlp-otto-score-ifc-xnor.exe --model models/model-xnor.otto --evalN 10000 --threadN 2

# XOR variant with its matching model
./mlp-otto-score-ifc-xor.exe --model models/model-xor.otto --evalN 10000
# Expected: 96.7% (9670/10000)

# Quick check with only 100 digits
./mlp-otto-score-ifc-xnor.exe --model models/model-xnor.otto --evalN 100
```

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
├── fetch_mnist.sh             ← MNIST download script
├── mlp-otto-score-ifc.c       ← inference source code (only 293 lines!)
├── ki-common.h                ← MNIST loader (only what's needed)
├── lib/
│   ├── maj3.h                 ← MAJ3 majority_tree algorithm
│   └── w0_random.h            ← splitmix64 random number generator
├── models/
│   ├── model-xnor.otto       ← pre-trained XNOR model (512 neurons)
│   └── model-xor.otto        ← pre-trained XOR model (512 neurons)
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

---

## 📖 Deeper reading

- **Research results**: [forward-prop.nhi1.de](https://forward-prop.nhi1.de/)
- **The vision**: [forward-prop.nhi1.de/papers/vision.html](https://forward-prop.nhi1.de/papers/vision.html)
- **Full source**: [github.com/aotto1968/forward-prop](https://github.com/aotto1968/forward-prop)

---

## 📝 License

Public domain. This is research code — use at your own risk.
