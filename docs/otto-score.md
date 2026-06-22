# Otto Score — DRAM-Native Bayes Classification

**Accuracy:** 95–96% on MNIST (H=512, single pass). Up to 96.4% with ensembles.
**Operations:** Only `& | ~` (bitwise) + int32 addition. **No float, no multiply.**

---

## 1. The Core Idea

The Otto Score treats each neuron as a **32-bit feature detector**.

A random projection W0 maps the input into a 32-bit code (H0). Each of the 32
bit-positions learns a log-odds score for each digit class — how much evidence
a "1" in that position provides for class *k*.

```
Forward:     H0 = MAJ3( ~(input ⊕ W0) )         32-bit code per neuron
             score[k] = offset[k] + Σ Σ active_bit × target[k][h][b]
                                     h   b
             predict = argmax(score)
```

No softmax. No activation function. No matrix multiply.

---

## 2. The Forward Pass (Inference)

### Step 1: Input Packing

784 raw pixels (28×28, uint8) are packed into 196 uint32 containers.
Four consecutive pixels become one uint32:

```
container[c] = p0 | p1<<8 | p2<<16 | p3<<24
```

### Step 2: H0 via MAJ3

For each neuron *h* (H neurons total), compute a 32-bit code:

```
match[c]     = ~(input[c] ⊕ W0[h][c])     (XNOR)
h0_bits[h]   = majority_tree(match, 196)   (MAJ3)
```

**MAJ3 (Majority-of-3):** A binary tree where each node computes `(a & b) | (a & c) | (b & c)`.
Given 196 input bits, MAJ3 returns the majority vote — a single bit.
Applied 32 times → 32-bit code.

### Step 3: Bayes Log-Score

The score for class *k* is a sum of log-odds contributions:

```
score[k] = offset[k] + Σ Σ (h0_bits[h] has bit b) × target[k][h][b]
                       h  b
```

Where:
- `target[k][h][b]` = `round( ln(P(bit=1|class=k) / P(bit=0|class=k)) × F )`
- `offset[k]` = `round( Σ Σ ln(1 - P(bit=1|class=k)) × F )`
- `F = (1<<OT_PRECISION)` — scaling factor (default 1024)

The prediction is simply the class with the highest score: `argmax(score)`.

---

## 3. Training (Iterative Correction)

Training only adjusts `target` and `offset`. W0 stays frozen forever.

### Phase 1: Counting

Count how often each bit-position is active per class:

```
target[k][h][b]++    for every training sample where:
                       class = k AND bit b of h0[h] = 1
```

### Phase 2: Log-Odds Conversion

Convert raw counts to log-odds via Laplace smoothing:

```
p = (count + 1) / (N_k + 2)         (smoothed probability)
target[k][h][b] = round( ln(p/(1-p)) × F )
```

### Phase 3: Iterative Correction (Epochs)

For each misclassified sample, adjust the target to push the correct class
score up and the incorrect prediction down:

```
For each misclassified sample with true class k, predicted p:
  For each active bit b of each neuron h:
    target[k][h][b] += step
    target[p][h][b] -= step
```

The step size follows a cosine decay schedule with linear warmup.

---

## 4. Why It Works on DRAM

Modern processors have fast multiply-add (FMA, SIMD). DRAM does not.
But DRAM rows can compute `& | ~` on all bits in parallel.

**The Otto Score maps directly to DRAM:**
- W0 stored in DRAM rows → `XNOR` with input in one cycle
- MAJ3 tree computed in the row periphery (just AND/OR gates)
- Scores accumulated in int32 registers (no floating point)
- No multiplication, no division (except in training, done offline)

---

## 5. Key Properties

| Property                 | Value                           |
| ------------------------ | ------------------------------- | -------------- |
| Inference operations     | `&                              | ~` + int32 add |
| Training operations      | int32 add/sub (correction)      |
| W0 size (H=512)          | 512 × 196 × 32 bits = 392 KB    |
| W1 size (H=512)          | 10 × 512 × 32 bits × F = 640 KB |
| 1-pass eval (H=512)      | ~86%                            |
| After correction (20 ep) | 96.3%                           |
| Ensemble limit           | ~96.5% (method ceiling)         |

---

## 6. Ensemble Voting

Multiple independent random W0s produce different errors.
Score-summing across members (product of experts) reduces error:

```
total_score[k] = Σ_m score_m[k]
```

At equal bit-mass, 17 small matrices (H=64) outperform one large matrix
(H=1088) — 96.4% vs 95.7%. See `www/papers/random.html` for the full
analysis of this effect.

---

## Source Files

- `mlp-otto-score-ifc.c` — Inference engine (~440 lines)
- `mlp-otto-score-ensemble.c` — Ensemble trainer (~660 lines)
- `ki-otto-common.h` — Shared infrastructure (batch correction, precision)
