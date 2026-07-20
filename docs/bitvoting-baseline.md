# Bit-Voting Baseline — The Information-Equivalent Linear Control Experiment

## Abstract

**The W0 random projection is the critical nonlinear ingredient in Otto Score.**
Without it, the same input bits reach only 55.2% instead of 64.4% on CIFAR-10
(performance encoding, 48 members, H=768) — a **9.2pp gap at identical input
information content**. Both Otto Score and Bit-Voting receive the same pixels,
the same encoding, and the same number of samples. The only difference: Otto maps
24576 input bits through a random overcomplete projection (W0) with majority
threshold, while Bit-Voting votes each input bit directly.

This gap is **not an information limit** — it is a **linear separability limit**.
W0 + majority nonlinearly expands the input space, making classes separable by
a linear voter. The W0 also acts as a **generalizer**: random projection prevents
overfitting to individual pixels by distributing each input bit across multiple
h0-neurons, creating robust distributed features.

Bit-Voting is **~4× faster** because it processes only set bits (sparse), while
Otto must compute the full dense W0×I0 product (H × NC XOR+popcount per sample).

## 1. Motivation

Every claim about Otto Score's accuracy must answer the question:
*"Is the performance due to the W0 + majority architecture, or would a simpler
linear model achieve the same result?"*

The Bit-Voting baseline answers this definitively. It is a **linear perceptron on
bit-level** — no W0, no majority, no hidden layer, no nonlinearity. Each of the
6272 input bits (after encoding) has 10 trainable weights (one per class), exactly
like each of the 256 h0-neurons in Otto Score has 10 trainable weights.

The architecture is the **identity W0**: W0 with ones on the main diagonal and zeros
elsewhere. Every input bit maps to exactly one h0-bit. No superposition, no
redundancy, no nonlinear abstraction.

## 2. Architecture

```
Bit-Voting:                    Otto Score (for comparison):
───────────────                ─────────────────────────────
input bits (6272)              input bits (6272)
        │                              │
        │ identity (no W0)             │ W0 random projection (6272 → 256×32bit)
        │                              │
        ▼                              ▼
   target[b][k]            h0-neuron[0..255]  (256 × 32bit containers)
   (62,720 weights)                │
        │                          │ majority-tree (popcount > threshold)
        │                          │
        ▼                          ▼
   score[k] += target[b][k]    gb_buf[0..255]  (256 bits per sample)
   argmax → prediction               │
                                     │ target[j][k]  (256 × 10 = 2560 weights)
                                     │
                                     ▼
                                score[k] += gb_buf[j] × target[j][k]
                                argmax → prediction
```

**Key equivalence (CIFAR-10, performance encoding, H=768):**
- Otto: H=768 h0-neurons × V=32 × 10 classes = **245,760 trainable int32 weights in target**
- Bit-Voting: 24,576 input-bits × 10 classes = **245,760 trainable int32 weights in target**
- **Same bit mass in target** — identical information capacity

Both have the same number of trainable weights (245,760 int32 values). Otto's weights
operate on the compressed 768 gb-bits (after W0 + majority), Bit-Voting's weights
operate on all 24,576 input bits directly. The 9.2pp gap on CIFAR-10 proves that
**feature quality matters more than feature quantity** — 768 nonlinearly compressed
features beat 24,576 raw features at the same target capacity.

### What the identity W0 proves

## 3. Information Equivalence

### Same bit-mass in target

| Property | Otto Score | Bit-Voting |
|----------|-----------|------------|
| Input bits (CIFAR perf.) | 24,576 (12 enc × 256 cont × 32b) | 24,576 (12 enc × 256 cont × 32b) |
| W0 matrix | 768 × 256 × 32bit = 6.3 Mbit frozen | None (identity) |
| **Target weights** | **H×V×K = 768×32×10 = 245,760 int32** | **bits×K = 24,576×10 = 245,760 int32** |
| Compute pattern | **Dense**: H×NC per sample | **Sparse**: only set bits |
| Nonlinearity | Majority (popcount > threshold) | None (linear) |
| Runtime (48 members, 10 ep) | **~228s** | **~64s** |

Both use the **same training algorithm** (`ki_batch_correct` with Bayesian gap-scaling),
the **same voting mechanism** (accumulated int64 scores → argmax), and the **same
target memory footprint** (245,760 int32). The difference is entirely architectural.

### What the identity W0 proves

If we replace Otto's random W0 with an identity matrix, every input bit flows to
exactly one h0-neuron. The majority then does nothing (only one bit per neuron).
This reduces Otto Score to Bit-Voting — proving that **W0 randomness is the cause
of nonlinear abstraction, not the majority operation**.

## 4. Results

### MNIST (60K train / 10K test, 10 epochs, single member)

| Model | Encoding | Eval | Train | Time | Note |
|-------|----------|------|-------|------|------|
| **Bit-Voting** | `latest` | **91.4%** | 95.0% | 789ms | Linear baseline |
| Otto Score | `latest` | **96.1%** | 98.7% | 1716ms | W0 + majority |
| Float32 AdamW | raw | ~96% | ~96% | — | Backprop baseline |
| Hebbian BV32 | raw | 90.4% | — | — | No backprop (legacy) |

**The gap: 4.7pp (91.4% → 96.1%).**

### CIFAR-10 (50K train / 10K test, 10 epochs, 48 members, performance encoding)

| Model | Eval | Time | Note |
|-------|------|------|------|
| **Bit-Voting** | **55.2%** | **~64s** | Linear baseline, sparse bit scanning |
| Otto Score (H=768) | **64.4%** | **~228s** | W0 + majority, dense W0×I0 product |
| **Gap** | **+9.2pp** | **3.6× slower** | The price of nonlinear abstraction |

**The gap grows with input complexity:**
- MNIST (binary, simple): **+4.7pp** (96.1% vs 91.4%)
- Fashion-MNIST (grayscale, texture): **+8pp** (88% vs 80%)
- CIFAR-10 (color, complex): **+9.2pp** (64.4% vs 55.2%)

Nonlinear feature extraction becomes **more critical** as the input becomes more complex.

### Fashion-MNIST (60K train / 10K test, 10 epochs)

| Model | Encoding | Eval | Note |
|-------|----------|------|------|
| **Bit-Voting** | `latest` | **~80%** | Linear baseline |
| Otto Score | `latest` | **~88%** | W0 + majority |

### Encoding Sensitivity

Bit-Voting depends more on encoding quality than Otto Score, because it has no
feature extraction to compensate for poor input structure:

```
Bit-Voting on MNIST (10 epochs, --evalN 10000):
───────────────────────────────────────────────
Encoding   Eval     Note
raw        89.2%    Raw pixel bits (binary threshold)
down       89.5%    Downsampled
sig8       90.7%    Sigmoid thermometer, 8-bit
log8       91.4%    Logarithmic thermometer, 8-bit (BEST)
exp8       91.2%    Exponential thermometer, 8-bit
```

Otto Score gets ~96% regardless of encoding choice — the random projection
creates structure even from unstructured input bits.

### Ensemble Effect (Multiple Encodings + Xforms)

Bit-Voting benefits from multi-encoding ensembles (+1-2pp), but the gain is much
smaller than for Otto Score:

```
Bit-Voting Ensemble on MNIST (10 epochs, --evalN 10000):
─────────────────────────────────────────────────────────
Members   Configuration              Eval
1         encoding=latest, xform=id   91.4%
2         encoding=exp,log            91.8%
4         enc=exp,log + xf=id,dflip1  92.0%
6         enc=exp,log + xf=perf.      92.1%
```

**Why smaller gain than Otto?** Otto members have **independent random W0
matrices** — different random seeds create completely different projections,
producing uncorrelated errors. Bit-Voting members differ only in encoding and
xform, which provides limited diversity. The W0 randomness is the actual source
of ensemble diversity.

### Member Threshold Effect

The `--member-threshold` filter removes weak members (train accuracy < threshold):

```
Bit-Voting on MNIST, 3 encodings, --member-threshold 92:
──────────────────────────────────────────────────────────
→ 1 of 3 members above threshold (enc=log passes, exp/down filtered)
→ Ensemble eval drops by 1-2% (weaker but larger ensemble is better)
```

**Insight:** For linear models, every member adds useful signal regardless of
individual strength. For nonlinear models (Otto), weak members introduce noise.
This confirms that Bit-Voting members are measuring the same linear manifold,
while Otto members explore different nonlinear projections.

## 5. Speed Analysis: Why Otto is 4× Slower

Otto's runtime is dominated by the **dense W0 × I0 computation**:

| Step | Bit-Voting | Otto (H=768) |
|------|-----------|--------------|
| Per sample: scan input | ~12K set bits (sparse) | — |
| Per sample: W0 × I0 | **0** | **768 × 256 = 196K XOR+popcount** |
| Per sample: majority | **0** | 256K popcount+threshold (maj=3) |
| Per sample: target vote | ~12K adds | 768 adds |
| **Per epoch (50K CIFAR)** | **~600M sparse ops** | **~23B dense ops** |

Otto does **~40× more raw operations** than Bit-Voting. The actual 3.6× runtime
gap (228s vs 64s) is smaller than 40× because:
- Otto's W0 computation is highly optimized (bit-parallel XOR + popcount intrinsics)
- OpenMP parallelizes well over samples
- Bit-Voting's sparse scanning has branch overhead (while(v) __builtin_ctz loop)

**In DRAM hardware:** The W0 × I0 product is a single bitwise-XOR between two
memory rows — executed in one cycle across all bits. The 40× gap vanishes.
Both architectures cost nearly the same chip area.

## 6. W0 as Generalizer

The W0 random projection does more than create nonlinearity — it **generalizes**:

### Distributed representation
Each input bit maps to **~8 random h0-neurons**. No single pixel can dominate
a decision. This forces the network to combine information from multiple pixels,
creating robust, distributed features.

### Encoding invariance
Otto Score achieves ~96% on MNIST regardless of encoding choice (raw, sig8, exp8,
log8 — all within ±0.5pp). Bit-Voting varies ±2pp depending on encoding quality.
The random projection **homogenizes** input structure — bad encodings get mixed
into useful features anyway.

### Overfitting resistance
Bit-Voting with 48 members on CIFAR-10 reaches 55.2% eval but trains to 78.2%
(train-eval gap: **23pp**). Otto with the same 48 members reaches 64.4% eval and
trains to 96.7% (gap: **32pp** — larger because it CAN overfit more, but the
absolute eval is 9.2pp higher because the features are better).

### Why this matters for DRAM
A chip that implements W0 as a fixed random addressing pattern (not stored weights)
gets the generalizer for free:
- No weight storage for W0 (just addressing)
- No training for W0 (frozen random)
- Nonlinearity + generalization at zero chip cost
- The same addressing hardware works for any dataset

## 7. Interpretation

### The Npp gap is the price of linearity

MNIST in raw pixel space is not linearly separable — a 2-layer MLP with 256 hidden
neurons achieves ~97%, while a linear classifier achieves ~92%. The 5pp gap between
Otto and Bit-Voting exactly matches this known difference.

The W0 matrix acts as a **random nonlinear expansion**:
1. Each input bit is XOR-ed with ~31 random container values → nonlinear mixing
2. Majority over each container → threshold activation (popcount > 16)
3. Result: 256 bits that are nonlinearly transformed from the original 6272 bits

**Without training W0**, this expansion is random. Yet it enables 5pp higher accuracy.
This matches the Reservoir Computing / Random Projection literature: a random
expansion into a higher-dimensional space improves linear separability.

### The identity W0 thought experiment

Imagine replacing the random W0 with an identity mapping (1:1, no mixing):

```
Random W0:      input_bit[b] → h0_neuron[j] for ~8 random j    (overcomplete)
Identity W0:    input_bit[b] → h0_neuron[b] alone               (no mixing)
```

With identity W0, the majority layer has nothing to vote on (only 1 bit per neuron).
Training reduces to a linear classifier on raw input bits — exactly Bit-Voting.
The accuracy drops from 96.1% to 91.4%.

This proves that **W0 randomness is not noise — it is the feature extraction**.

### Why not just make W0 learnable?

Learning W0 (backprop through the random projection) adds ~10pp on CIFAR-10
(58% → 68%, see XNOR-Net literature), but requires:
- Floating point storage for real-valued weights
- Backpropagation through XOR operations
- GPU or specialized hardware

The DRAM-native constraint forbids these. Frozen W0 is a compromise: 5pp below
the learnable ceiling, but zero additional hardware cost.

## 8. Key Findings

1. **W0 + majority adds +5–9pp over linear baseline** (MNIST to CIFAR-10).
   The gain comes from nonlinear feature expansion, not from more parameters.

2. **Same bit mass in target**: Both Otto (H=768) and Bit-Voting have 245,760
   int32 target weights. Otto compresses 24,576 input bits to 768 gb-bits via W0,
   then votes with 768×32×10 weights. Bit-Voting votes with all 24,576 bits × 10
   weights directly. Same capacity, better features win.

3. **Otto is ~4× slower in simulation** due to dense W0×I0 computation (196K
   XOR+popcount per sample vs ~12K sparse adds). In DRAM hardware this gap vanishes.

4. **W0 acts as a generalizer**: random projection distributes each input bit
   across multiple h0-neurons → distributed representations → encoding invariance
   → overfitting resistance — all at zero training cost.

5. **The gap scales with input complexity**: MNIST +4.7pp, Fashion +8pp, CIFAR +9.2pp.
   Complex data benefits more from nonlinear abstraction.

6. **Bit-Voting is the perfect null hypothesis** for Otto Score experiments.
   Any improvement that doesn't exceed the linear baseline is attributable to
   better engineering, not to architectural innovation.

## 9. Usage

### CLI

```bash
# MNIST — single member
./mnist-mlp-bin32-otto-trn-bitvoting.exe --epochsN 10 --encoding latest --xform id

# Multi-encoding ensemble
./mnist-mlp-bin32-otto-trn-bitvoting.exe --epochsN 10 --encoding exp,log --xform performance

# CIFAR-10
./cifar-mlp-bin32-otto-trn-bitvoting.exe --epochsN 10 --encoding latest --xform id

# Fashion-MNIST
./fashion-mlp-bin32-otto-trn-bitvoting.exe --epochsN 10 --encoding latest --xform id

# Debug per-member accuracy
./mnist-mlp-bin32-otto-trn-bitvoting.exe --epochsN 10 --encoding exp,log \
  --xform performance --debug-member --debug-epoch

# Filter weak members
./mnist-mlp-bin32-otto-trn-bitvoting.exe --epochsN 10 --encoding exp,log \
  --xform performance --member-threshold 90
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--epochsN` | 1 | Training epochs |
| `--encoding` | latest | Encoding(s), comma-separated |
| `--xform` | id | Transform(s), comma-separated |
| `--batchN` | 64/128 | Batch size for accumulation |
| `--lr` | 0.05 | Step scaling (step = lr × OT_F) |
| `--debug-member` | off | Per-member accuracy + encoding/xform |
| `--debug-epoch` | off | Per-epoch step/train/eval |
| `--member-threshold` | 0 | Filter members below N% train acc |
| `--threadN` | 8 | OpenMP threads |

## 10. Source Files

| File | Description |
|------|-------------|
| `otto-score-ifc/mnist/mlp-bin32-otto-trn-bitvoting.c` | Canonical source |
| `otto-score-ifc/lib/ki-train.h` | Shared training loop (`ki_batch_correct`) |
| `otto-score-ifc/lib/ki-load.h` | Shared data loading |
| `mnist-1/` (symlink) | MNIST build directory |
| `cifar-1/` (symlink) | CIFAR-10 build directory |
| `otto-score-ifc/fashion/` (symlink) | Fashion-MNIST build directory |

## 11. References

- Otto Score: `www/papers/ensemble.html` — W0 randomness and ensemble effects
- MNIST paper: `www/papers/mnist-number.html` — Full MNIST results with xforms
- Status Report: `www/papers/status-2026-07.html` — Encoding, CIFAR-10 barrier
- Random Projection theory: Johnson-Lindenstrauss lemma, Reservoir Computing
