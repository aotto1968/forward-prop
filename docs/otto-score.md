# Otto Score — DRAM-Native Bayes Classification

**Accuracy:** 95–96% on MNIST (H=512). Up to 96.4% with ensembles.
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

## 2. Data Structures & Index Layout

```
W0:      uint32[ensemble_n][H][NC]     (frozen, random splitmix64)
Target:  int32[ensemble_n][10][H][32]  (log-odds × F, class × neuron × bit)
Offset:  int64[ensemble_n][10]          (log(1-P) × F, per class)
F = (1 << OT_PRECISION)                (default 10 → F = 1024)
NC = 196 (MNIST, 4 pixels per container)
```

Index macro (identical in all programs):

```c
#define TGT_IDX(k, h, b, H) \
    ((size_t)(k) * (size_t)(H) * 32 + (size_t)(h) * 32 + (size_t)(b))
// Layout:  [10][H][32]  — class outer, neuron middle, bit inner
```

---

## 3. The Forward Pass (Inference)

### Step 1: Input Packing

784 raw pixels (28×28, uint8) are packed into 196 uint32 containers.
Four consecutive pixels become one uint32:

```
container[c] = p0 | p1<<8 | p2<<16 | p3<<24
```

### Step 2: H0 via MAJ3

For each neuron *h*, compute a 32-bit code:

```c
uint32_t h0_neuron(const uint32_t *packed, const uint32_t *W0_row) {
    uint32_t match[NC];
    for (int c = 0; c < NC; c++)
        match[c] = H0_MATCH(packed, W0_row, c);
    return majority_tree(match, NC);
}
// H0_MATCH = ~(in[c] ^ W0_row[c])    XNOR (default)
// H0_MATCH =   in[c] ^ W0_row[c]     XOR  (-DH0_XOR)
```

**MAJ3 (Majority-of-3):** A binary tree where each node computes
`(a & b) | (a & c) | (b & c)`. Applied to 196 containers → 32-bit code.

### Step 3: Bayes Log-Score

**Critical — loop order matters.** The correct implementation:

```c
// ✅ CORRECT: h outer, k middle, b inner
for (int h = 0; h < H; h++) {                       // Σ_h
    uint32_t h0 = h0_neuron(in, W0_row);
    for (int k = 0; k < N_CLASSES; k++) {            // per class
        for (int b = 0; b < 32; b++) {               // Σ_b
            if (h0 & (1U << (unsigned)b)) {          // bit = 1?
                scores[k] += target[TGT_IDX(k, h, b, H)];
                //           ⬆ h = SAME neuron as h0!
            }
        }
    }
}
```

**Why this order?**
- `h` must be outermost because `h0` is computed per neuron
- `target[k][h][b]` uses the **same** `h` — each neuron has its OWN target vector
- `b` must be innermost because we check each bit of `h0`

**❌ WRONG (common bug — all neurons get neuron-0's target):**

```c
for (int b = 0; b < 32; b++) {                      // b outer
    int dot = 0;
    for (int h = 0; h < H; h++)
        dot += (h0[h] >> b) & 1U;                    // Σ_h bits
    scores[k] += dot * target[TGT_IDX(k, 0, b, H)]; // ⚠ h=0 always!
}
```
→ Result: random accuracy (10% for MNIST).

---

## 4. Training (Iterative Correction)

Training only adjusts `target` and `offset`. W0 stays frozen forever.

### Phase 1: ki_build_target — Single Pass Count

Count how often each bit-position is active per class.
Only counts for samples of the correct class:

```c
for (int s = 0; s < N; s++) {
    int k = Y[s];
    const uint32_t *in = X + s * NC;
    for (int h = 0; h < H; h++) {
        uint32_t h0 = h0_neuron(in, W0 + h * NC);
        for (int b = 0; b < 32; b++) {
            if (h0 & (1U << b))
                target[TGT_IDX(k, h, b, H)]++;   // ← k, h, b ALL in index!
        }
    }
}
```

**Essential:** The `h` in `TGT_IDX(k, h, b, H)` must match the neuron's `h`.
Using `TGT_IDX(k, 0, b, H)` (always h=0) would discard all class signal.

### Phase 2: compute_class_offset

Must be called BEFORE logit_convert (needs raw counts):

```c
for (int k = 0; k < 10; k++) {
    int64_t sum = 0;
    int nk = class_counts[k];
    for (int h = 0; h < H; h++)
        for (int b = 0; b < 32; b++) {
            int t = target[TGT_IDX(k, h, b, H)];            // raw count
            double p1 = (double)(nk - t + 1) / (double)(nk + 2);  // Laplace
            sum += (int64_t)ot_precision(log(p1));           // × F
        }
    class_offset[k] = sum;
}
```

### Phase 3: logit_convert

Convert raw counts to log-odds via Laplace smoothing:

```c
for (int k = 0; k < 10; k++) {
    int nk = class_counts[k];
    for (int h = 0; h < H; h++)
        for (int b = 0; b < 32; b++) {
            size_t idx = TGT_IDX(k, h, b, H);
            int t = target[idx];
            double p = (double)(t + 1) / (double)(nk + 2);  // Laplace
            target[idx] = (int32_t)ot_precision(log(p / (1.0 - p)));
        }
}
```

Without this step, targets are raw counts (0..N_k), not log-odds.
The score would compute `raw_count × F` instead of `log_odds × F` — no Bayes signal.

### Phase 4: Iterative Correction (Epochs)

For each epoch:
1. Precompute H0 for ALL training samples (cached)
2. Call `ki_batch_correct()` for each ensemble member:

```c
ki_batch_correct(target_m, H, offset_m, h0_all, Y, trainN,
                 batchN, current_step, tgt_sz_m);
```

For each misclassified sample:
- `target[true_k][h][b] += step` for each active bit
- `target[pred_k][h][b] -= step` for each active bit

Step size follows cosine decay with linear warmup.

---

## 5. Common Implementation Bugs

| Bug | Symptom | Cause |
|-----|---------|-------|
| **Wrong TGT_IDX** | 10% (random) | `TGT_IDX(k, 0, b, H)` instead of `TGT_IDX(k, h, b, H)` — all neurons copy neuron-0's target |
| **Missing Phase 1** | 10% | ki_build_target + logit_convert + class_offset never called → targets = 0 |
| **Wrong W0 seeding** | deterministic, not truly random | `srand()` instead of `w0_srandom()` — splitmix64 has its own seed state |
| **Wrong loop order** | 10% | b outer, h inner → target from h=0 for all neurons |

---

## 6. Why It Works on DRAM

Modern processors have fast multiply-add (FMA, SIMD). DRAM does not.
But DRAM rows can compute `& | ~` on all bits in parallel.

**The Otto Score maps directly to DRAM:**
- W0 stored in DRAM rows → `XNOR` with input in one cycle
- MAJ3 tree computed in the row periphery (just AND/OR gates)
- Scores accumulated in int32 registers (no floating point)
- No multiplication, no division (except in training, done offline)

---

## 7. Key Properties

| Property                 | Value                           |
| ------------------------ | ------------------------------- |
| Inference operations     | `& \| ~` + int32 add            |
| Training operations      | int32 add/sub (correction)      |
| W0 size (H=512)          | 512 × 196 × 32 bits = 392 KB    |
| W1 size (H=512)          | 10 × 512 × 32 bits × F = 640 KB |
| 1-pass eval (H=512)      | ~86%                            |
| After correction (20 ep) | 96.3%                           |
| Ensemble limit           | ~96.5% (method ceiling)         |

---

## 8. Ensemble Voting

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

- `mlp-bin32-otto-ifc.c` — Inference engine (~440 lines)
- `mlp-bin32-otto-trn.c` — Ensemble trainer (~660 lines)
- `ki-otto-common.h` — Shared infrastructure (batch correction, precision)
