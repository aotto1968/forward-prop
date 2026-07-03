# Bin32 Hebbian — Bitwise Co-Occurrence Learning (Negative Reference)

**Accuracy:** ~78–82% on MNIST (H=512, oscillates).
**Operations:** `& | ~` + popcount. **No float, no multiply.**
**Purpose:** Demonstrates that naive co-occurrence counting does *not* converge.

---

## 1. Why This Baseline Exists

The Hebbian trainer represents the most naive form of bitwise learning:
if a bit pattern often co-occurs with a class, set the corresponding weight
bit to 1. This is the simplest possible learning rule — simpler even than
the Otto Score.

It serves as a **negative reference**: pure co-occurrence learning without
Bayes log-odds or iterative correction converges to only ~82% and oscillates
thereafter. This proves that the Otto Score's advantage comes from its
**probabilistic formulation**, not just from being bitwise.

---

## 2. Architecture

```
         W0 (random, fresh each epoch!)    W1 (Hebbian)
Input ──────────────► MAJ3 ──────────────────► popcount ─► Argmax
  196×H              H×10                 10
```

### Forward Pass

The forward pass uses the same MAJ3-based H0 computation as Otto Score,
but scores through **popcount of XNOR** instead of Bayes log-odds:

```
h0[h]      = MAJ3( ~(input ⊕ W0[h]) )                  32-bit code
score[k]   = Σ_h popcount( ~(W1[k][h] ⊕ h0[h]) )       integer score
pred       = argmax(score)
```

Each neuron contributes `popcount(XNOR(W1, h0))` — the number of bit-positions
where W1 and h0 agree. No scaling, no log-odds, no class offset.

---

## 3. Training (Batch Hebbian)

### The Co-Occurrence Rule

For each class *k* and each bit-position *b* of each neuron *h*:

```
counter[k][h][b]++    if input has bit b = 1 AND sample class = k
```

After counting all training samples, apply a threshold:

```
if counter[k][h][b] > threshold:
    W1[k][h][b] = 1
else:
    W1[k][h][b] = 0
```

The threshold is `hebbian_pct` percent of class count (default 50%).

### Why It Fails

This rule has a fundamental flaw: it only learns **positive correlations**
(bit=1 with class=k). It ignores negative evidence (bit=0 with class=k)
and does not model the *absence* of a feature.

Compare with Otto Score, which stores both `P(bit=1|class)` and `P(bit=0|class)`
through the log-odds ratio `ln(p/(1-p))`.

### Per-Epoch W0 Regeneration

Unlike Otto Score (which freezes W0), the Hebbian trainer generates a
**new random W0 every epoch**. This means each epoch starts from scratch
with a fresh random projection. W1 is retrained from zero each epoch.

The best epoch (highest eval accuracy) is saved.

---

## 4. Why Oscillation Occurs

The Hebbian update is discrete and local:

```
if co-occurrence > 50%: set bit to 1
else: set bit to 0
```

This creates a feedback loop:
1. W1 learns some pattern → accuracy improves
2. Next epoch: new W0 → different H0 patterns
3. W1 overwrites with new co-occurrence counts
4. Previous patterns are forgotten

The network oscillates because each epoch destroys the previous epoch's
learning. There is no gradient, no momentum, no persistent memory.

---

## 5. Comparison with Otto Score

| Aspect         | Otto Score                     | Hebbian                   |
| -------------- | ------------------------------ | ------------------------- |
| Score function | Bayes log-odds sum             | popcount(XNOR)            |
| W1 data type   | int32 (log-odds, signed)       | uint32 (binary)           |
| W1 capacity    | 32 bits per weight             | 1 bit per weight          |
| Training       | Iterative correction (add/sub) | Co-occurrence (set/reset) |
| W0             | Frozen (one projection)        | New each epoch            |
| convergence    | ✅ 96%+                        | ❌ ~82%, oscillates       |
| Inference HW   | DRAM (bit-logic)               | DRAM (bit-logic)          |

---

## 6. What We Learn From This

The Hebbian baseline proves that **bitwise operation is not enough**.
The Otto Score's success comes from three key innovations the Hebbian
trainer lacks:

1. **Probabilistic scores** — log-odds capture both positive and negative
   evidence, binary weights capture only positive.

2. **Iterative correction** — instead of counting from scratch each epoch,
   the correction pass makes small adjustments to existing weights.

3. **Frozen W0** — a stable random projection lets W1 accumulate knowledge
   across epochs, instead of chasing a moving target.

---

## Source Files

- `mlp-bin32-w1-hebbian-trn.c` — Hebbian trainer (~570 lines)
- `mlp-bin32-hebbian-ifc.c` — Bin32 inference (~330 lines)
- `ki-common.h` — Shared helpers (MNIST loader, memory)
