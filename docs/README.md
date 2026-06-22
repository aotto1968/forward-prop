# Technical Documentation — Otto Score & Reference Baselines

This directory explains the three classification methods implemented in this
repository. Each document walks through the forward pass, training algorithm,
and key design decisions — from theory to C code.

---

## Documents

| Document | Method | Type | Accuracy |
|----------|--------|------|:--------:|
| [otto-score.md](otto-score.md) | **Otto Score** | DRAM-native bit-logic | **96.3%** |
| [adamw.md](adamw.md) | **Float32 AdamW** | CPU/GPU float matmul | 92.6% |
| [hebbian.md](hebbian.md) | **Bin32 Hebbian** | DRAM-native bit-logic | ~82% |

---

## Quick Comparison

```
                Forward Pass              Training Method         Eval
Otto Score     XNOR → MAJ3 → Bayes log   Iterative correction   96.3%
AdamW          matmul → LReLU → matmul   AdamW (backprop)       92.6%
Hebbian        XNOR → MAJ3 → popcount    Batch co-occurrence    ~82%
```

**Key insight:** Otto Score achieves the highest accuracy while using
**zero floating-point** and **zero multiplication** during inference.
It is the only method designed to run natively on DRAM bit-logic hardware.

---

## Architecture Overview

All three methods share the same two-layer structure:

```
         W0 (frozen)          W1 (trained)
Input ───────────────► H0 ───────────────► Score ─► Argmax
        784→196→H              H→10
```

The difference is:
- **How** H0 is computed from the input and W0
- **How** the final score is computed from H0 and W1
- **How** W1 is trained

| Component       | Otto Score       | AdamW            | Hebbian          |
|-----------------|------------------|------------------|------------------|
| W0 data type    | uint32           | float32          | uint32           |
| H0 computation  | MAJ3(XNOR)       | matmul + LReLU   | MAJ3(XNOR)       |
| W1 data type    | int32 (log-odds) | float32          | uint32           |
| Score function  | Bayes log-sum    | matmul           | popcount(XNOR)   |
| Training        | Correction pass  | AdamW            | Co-occurrence    |
| Inference float | **No**           | Yes              | **No**           |
