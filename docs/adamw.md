# Float32 AdamW Reference — Standard Neural Network Baseline

**Accuracy:** 92.6% on MNIST (H=512, 10 epochs).
**Operations:** Float32 matrix multiply + Leaky ReLU + AdamW optimizer.
**Purpose:** Baseline comparison for Otto Score.

---

## 1. Why This Baseline Exists

The AdamW baseline represents the **standard approach** to MNIST classification:
a two-layer neural network trained with gradient descent. It exists here to
answer the question: *"How well does a simple float32 network perform at the
same model size as the Otto Score?"*

The answer: **Otto Score outperforms it by ~3.7pp** at H=512, using zero
floating-point operations.

---

## 2. Architecture

```
         W0 (float, frozen)         W1 (float, AdamW)
Input ────────────► LReLU ───────────────► Output ─► Argmax
  196×H         H×10                 10
```

### Forward Pass

```
h0     = W0 @ x          (matrix multiply: H × 196 · 196)
h0     = LeakyReLU(h0)   (h0[i] < 0 → h0[i] *= 0.05)
output = W1 @ h0         (matrix multiply: 10 × H · H)
pred   = argmax(output)
```

### Weight Initialization

Both W0 and W1 use Kaiming Uniform initialization:

```
bound = 1 / sqrt(in_features)
W[i]  = uniform_random(-bound, +bound)
```

W0 is frozen after initialization. Only W1 is trained.

---

## 3. Training (AdamW)

### Loss Function

Mean Squared Error with ±1 targets:

```
L = mean( (output[k] - target[k])² )
target[k] = 1.0 if k == true_class else -1.0
```

### Optimizer: AdamW

AdamW is Adam with decoupled weight decay:

```
m_t = β₁ · m_{t-1} + (1 - β₁) · ∇L          (first moment)
v_t = β₂ · v_{t-1} + (1 - β₂) · (∇L)²       (second moment)
m̂_t = m_t / (1 - β₁^t)                      (bias correction)
v̂_t = v_t / (1 - β₂^t)
w   = w - lr · ( m̂_t / (√v̂_t + ε) + λ · w ) (AdamW update)
```

| Hyperparameter | Value        |
| -------------- | ------------ |
| Learning rate  | 0.002        |
| β₁             | 0.9          |
| β₂             | 0.999        |
| ε              | 1e-8         |
| Weight decay   | 1e-4         |
| Batch size     | 64           |
| Warmup epochs  | 2            |
| LR schedule    | Cosine decay |

### Schedule

The learning rate follows cosine decay with linear warmup:

```
lr(epoch) = lr_min + 0.5 × (lr_init - lr_min) × (1 + cos(π × t / T))
```

Where `t = epoch - warmup` and `T = total_epochs - warmup`.

### Gradient Clipping

Gradients are clipped to max norm 1.0 before each AdamW step.

---

## 4. Inference

The trained model is exported as two binary files:

```
weights.meta     — "2\nH 196\n10 H\n"
W0.bin           — float32[H × 196]
W1.bin           — float32[10 × H]
```

The inference engine (`mlp-flt32-adam-ifc.exe`) loads these and runs the
forward pass using the same float32 matmul + LReLU operations.

---

## 5. Input Encoding

Unlike the bitwise methods, the AdamW trainer does not pack pixels into
uint32 containers. Instead it averages groups of 4 raw pixels into one float:

```
packed[c] = avg(p₄c, p₄c₊₁, p₄c₊₂, p₄c₊₃) / 127.5 - 1.0
```

This preserves pixel precision better than bit-packing, at the cost of
requiring floating-point arithmetic.

---

## 6. Comparison with Otto Score

| Aspect       | Otto Score                      | AdamW                 |
| ------------ | ------------------------------- | --------------------- |
| Forward      | XNOR + MAJ3 + int32 add         | matmul + LReLU        |
| Precision    | int32 (scaled log-odds)         | float32 (continuous)  |
| W1 memory    | int32[10×H×32] = 640 KB (H=512) | float32[10×H] = 20 KB |
| Training     | Iterative correction            | Backprop + AdamW      |
| Inference HW | DRAM (bit-logic)                | CPU/GPU (matmul)      |
| Eval (H=512) | **96.3%**                       | 92.6%                 |

The Otto Score uses **32× more W1 memory** but achieves higher accuracy
because the log-odds representation captures per-bit statistics that a
single float weight cannot express.

---

## Source Files

- `mlp-flt32-w1-adam-trn.c` — AdamW trainer (~440 lines)
- `mlp-flt32-adam-ifc.c` — Float32 inference (~300 lines)
- `ki-common.h` — Shared helpers (matmul, AdamW, MNIST loader)
