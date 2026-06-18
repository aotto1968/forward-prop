# Otto Score Inference — DRAM-Native MNIST Classifier

**Zero floating point. Zero multiply-accumulate. Zero training.**
Only bit-logic (`&`, `|`, `~`) and int32 addition — **96.4% MNIST**.

This directory contains a **self-contained inference demo** for the Otto Score
classifier. Load a pre-trained model and classify MNIST digits in pure `&|~` + int32.

## How It Works

```
Input image (784 px)
       ↓
  Pack into uint32[196] containers  (4 pixels per container)
       ↓
  MAJ3(~(X ⨁ W0[h])) per neuron     (XNOR + majority_tree)
       ↓
  32 × H bits per sample
       ↓
  Bayes log-Score per class:
    score[k] = offset[k] + Σ y × logit[k][h][b]
       ↓
  argmax → predicted digit
```

- **W0**: random frozen projection (never trained, never changed)
- **Target**: pre-computed log-odds `ln((t+1)/(N_k-t+1)) × 100000`
- **offset[k]**: `Σ log(1-P_k) × 100000` — the base cost per class
- **Only operations**: `&`, `|`, `~`, `+` on int32/int64 — **no floats, no multiply**

## Requirements

- **C compiler**: gcc or clang with OpenMP support
- **zlib**: for reading compressed MNIST files (`-lz`)
- **MNIST dataset**: downloaded automatically by the inference (bundled data paths)

Tested on: Linux x86_64, gcc-15, AVX-512. Also works on ARM64 with clang.

## Quick Start

```bash
make all
make setup
make test
make push
```

Expected output:
```
=== XNOR inference with XNOR-trained model ===
  Model:   H=512  XNOR
  Eval:    96.3%

=== XOR inference with XOR-trained model ===
  Model:   H=512  XOR
  Eval:    96.7%
```

## XNOR vs XOR

Both variants are built from the same source using a compile-time switch.
**Each variant requires its matching model.**

| Mode | Build | H0_MATCH | Model file | Accuracy |
|:-----|:------|:---------|:-----------|:---------|
| XNOR | `make xnor` | `~(in ^ W0)` | `models/model-xnor.otto` | **96.3%** |
| XOR  | `make xor`  | `in ^ W0` | `models/model-xor.otto` | **96.7%** |

Both modes achieve identical accuracy at the same H. XOR saves one NOT
operation per bit on DRAM hardware — useful for chip implementations.
To train your own model, use the trainer in `ki-w2/` with the matching
compile flag (`-DH0_XOR` for XOR mode).

## Full MNIST Evaluation

```bash
./mlp-otto-score-ifc-xnor.exe --out models/ --model model-xnor.otto --evalN 10000
./mlp-otto-score-ifc-xor.exe --out models/ --model model-xor.otto --evalN 10000
```

You can also use your own exported model (see the trainer in `ki-w2/`):

```bash
./mlp-otto-score-ifc-xnor.exe --out /path/to/exported/model/
```

## Model Format (`model.otto`)

Single binary file containing everything needed for inference:

```
[Header: 20 bytes]
  magic=0x4F54544F ('OTTO'), version=1, H, NC

[W0: uint32[H × NC]]
  Frozen random projection

[Target: int32[10 × H × 32]]
  Per-class per-bit log-odds

[Offset: int64[10]]
  Per-class base scores
```

| Model | H | Size | Accuracy |
|:------|:-:|:----:|:--------:|
| `models/model-xnor.otto` | 512 | ~1.0 MB | **96.3%** |
| `models/model-xor.otto` | 512 | ~1.0 MB | **96.7%** |

## Training Your Own Model

The trainer lives in `ki-w2/mlp-otto-score.c` (not in this public directory).
To train and export:

```bash
cd ki-w2/
make mlp-otto-score.exe
./mlp-otto-score.exe --hiddenN 2048 --epochsN 20 --out mymodel/
# → mymodel/model.otto — copy to this directory
```

## File Structure

```
├── README.md
├── Makefile                 # Build both XNOR + XOR
├── fetch_mnist.sh           # Download MNIST dataset
├── mlp-otto-score-ifc.c     # Inference source
├── ki-common.h              # MNIST loader + helpers
├── lib/
│   ├── maj3.h               # MAJ3 majority_tree
│   └── w0_random.h          # splitmix64 RNG
├── models/
│   ├── model-xnor.otto      # Pre-trained XNOR (H=512, 20ep, 96.3%)
│   └── model-xor.otto       # Pre-trained XOR (H=512, 20ep, 96.7%)
└── .gitignore
```

## Links

- **Research page**: [forward-prop.nhi1.de](https://forward-prop.nhi1.de/)
- **GitHub**: [github.com/aotto1968/forward-prop](https://github.com/aotto1968/forward-prop)
- **Author**: Andreas Otto — independent research

## License

Public domain. This is research code — use at your own risk.
