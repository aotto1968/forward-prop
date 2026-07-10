# KI-DRAM: Boolean-Only Neural Inference in Commodity Memory

**forward-prop Research — July 2026**

---

## Abstract

We present a boolean-only neural network architecture designed for direct execution inside DRAM modules (Processing-in-Memory, PIM). Unlike existing PIM accelerators that embed DSP blocks or FPUs alongside DRAM cells, our approach uses **only bitwise operations (`& | ~`), MAJ3 (3-input majority), and int32 addition** — the minimal logic that can coexist with a DRAM cell without exceeding area or power budgets.

The Otto Score classifier achieves **97% MNIST** and **61.6% CIFAR-10** accuracy using pure `&|~` + int32 at inference time. Training uses float32 SGD as a compromise (gradients need continuous values), but the trained model is exported to pure integer weights for deployment. The entire forward pass is implementable with three logic gate types repeated across the memory array.

---

## 1. Why DRAM-PIM Needs Boolean Logic

### 1.1 The Area Constraint

A DRAM cell occupies approximately 6F² (where F is the minimum feature size). Adding logic to every row must respect this density. A single 32-bit floating-point multiply-accumulate (FMA) unit requires ~10,000 transistors — roughly the area of **1,600 DRAM cells**. Placing one FMA per DRAM row would increase die area by 100×, making the approach economically infeasible.

A **single MAJ3 gate** (3-input majority: `(a & b) | (a & c) | (b & c)`) requires ~12 transistors. A 32-bit majority tree for NC_slice containers requires a few hundred transistors. Total per-neuron logic: **~400 transistors** — comparable to **65 DRAM cells**. At this scale, logic can be integrated directly into the row decoder without measurably increasing die area.

### 1.2 The Power Constraint

DRAM refresh consumes ~10 pJ per bit. A MAJ3 gate switching at DRAM row frequencies consumes <0.1 pJ. An FMA would consume 50–100 pJ — equal to the entire DRAM row power budget. Boolean operations are the only neural network operations that stay within DRAM's thermal envelope.

### 1.3 The Latency Constraint

A DRAM row activation takes ~30 ns. All cells in a row are read simultaneously. The MAJ3 computation can complete within the same cycle because:

- **XNOR**: Compare each stored weight bit against an input bit — 1 gate delay
- **MAJ3**: 3-input majority on (input, weight, ~input⊕weight) — 2 gate delays
- **Majority tree**: log2(NC_slice) levels of MAJ3 gates — ~5 delays for NC=196

The entire H0 forward pass (all H neurons × NC_slice containers) completes in **H row activations** — approximately H × 30 ns. For H=512, this is ~15 µs per inference.

### 1.4 Why Integer Matmul Does NOT Work in DRAM

A 32-bit integer multiply-accumulate requires:
- 32-bit × 32-bit → 64-bit multiplier (~3,000 transistors)
- 64-bit accumulator (~1,500 transistors)
- 32-cycle latency for weight readout

Attempting int32 matmul in DRAM-PIM is fundamentally a DSP-on-DRAM approach — it places a full ALU next to memory, defeating the density advantage of DRAM.

**Only boolean operations (MAJ3 + int32 add) meet the area, power, and latency constraints of DRAM-PIM.**

---

## 2. The Container Model: 32 Boolean Decisions per Neuron

### 2.1 uint32 — A Neuron is NOT a Number

In conventional neural networks, a neuron activation is a real number (float32). In our architecture, a neuron is a **container of 32 independent boolean decisions**:

```c
typedef uint32_t neuron_t;   // 32 bits = 32 yes/no decisions
```

The MAJ3 operation compresses NC_slice containers into 32 bits via bitwise majority voting. Each bit position independently decides whether the majority of containers agree.

### 2.2 Layer 0: Input × W0 → H0

```
For each neuron h (0..H-1):
    For each container c (0..NC_slice-1):
        match[c] = MAJ3( ~(input[c] ⊕ W0[h][c]) )
    h0[h] = majority_tree(match, NC_slice)
```

**Hardware mapping:** NC_slice DRAM rows per neuron. Each cell stores 1 weight bit + XNOR logic. The MAJ3 gate computes per-container agreement. The majority tree combines all containers into one 32-bit result.

**Key property:** The hidden activation `h0[h]` is a 32-bit boolean pattern. Each bit is a yes/no decision — independent, no cross-talk.

### 2.3 Layer 1: H0 × Target → Class Scores

```
For each class k (0..K-1):
    score[k] = offset[k]
    For each neuron h (0..H-1):
        For each active bit b in h0[h]:
            score[k] += target[k][h][b]
    → argmax over classes
```

The target matrix stores **int32 log-odds weights** per (class, neuron, bit-position). This is the only learned parameter. The offset per class centers the score distribution.

**Hardware mapping:** A second DRAM array with H rows × K columns. Each row holds one neuron's target contributions. The score accumulation is a lightweight int32 add — no multiplier needed.

---

## 3. Results

### 3.1 MNIST (Binary Input)

| Configuration                     | Eval      | Time |
| --------------------------------- | --------- | ---- |
| H=128, EN=7, 8 ep, exp8+log8+sig8 | **99.0%** | 40s  |
| H=512, EN=1, 10 ep, encoding=exp  | **97.0%** | 2.6s |
| H=64, EN=5, ep=5                  | **95.3%** | 6.5s |

MNIST works without thermometer encoding — the pixels are already binary (ink/no-ink).

### 3.2 CIFAR-10 (Continuous Color)

| Configuration                              | Eval                   | Time |
| ------------------------------------------ | ---------------------- | ---- |
| H=1024, EN=17, VN=2, 7 ep, encoding=latest | **61.62%**             | 273s |
| H=4094, EN=3, VN=2, target-err=0.4         | **61.0%** (single-run) | —    |
| H=128, EN=17, 8 ep, 10-channel             | **58.7%**              | 357s |

CIFAR-10 requires thermometer encoding — without it, accuracy collapses to ~25% (random).

### 3.3 Key Properties

- **No float at inference** — pure `&|~` + int32
- **No multiplication** — only addition and bitwise logic
- **Training exports to integer** — float32 SGD trains, int32 runs on chip
- **Ensemble members are parallel by design** — each occupies a separate DRAM row
- **Precomputation saves 71% time** — h0/gb computed once, never recomputed

---

## 4. DRAM-PIM Hardware Feasibility

### 4.1 Per-Neuron Logic

```
┌─────────────────────────────────────────────┐
│ DRAM Block (NC_slice containers × 32 bits)  │
│ ┌───┐ ┌───┐         ┌────┐                  │
│ │W₀ │ │W₁ │  . . .  │Wₙ₋₁│ ← 1T+1C          │
│ └─┬─┘ └─┬─┘         └──┬─┘                  │
│   │     │              │                    │
│   ▼     ▼              ▼      MAJ3 gates    │
│  ┌──────┬──────┬──────┬──────┐              │
│  │MAJ3  │MAJ3  │ ...  │MAJ3  │              │
│  └──┬───┴──┬───┴──────┴──┬───┘              │
│     │      │             │                  │
│     └──────┴──────┬──────┘                  │
│                   ▼                         │
│         ┌─────────────────┐                 │
│         │ Majority Tree   │ ← log₂(NC) lvls │
│         └────────┬────────┘                 │
│                  ▼                          │
│       h0[0..31] (32-bit output)             │
└─────────────────────────────────────────────┘
```

### 4.2 Transistor Count

| Component            | Transistors      | Notes               |
| -------------------- | ---------------- | ------------------- |
| DRAM cell (1T1C)     | 1                | Existing            |
| MAJ3 gate (12T)      | 12               | Per container       |
| Majority tree        | ~40 per neuron   | log₂(NC) levels     |
| Score accumulator    | ~100 (int32 add) | Per class           |
| **Total per neuron** | **~400T**        | ~65× DRAM cell area |

At this transistor count, a boolean-KI DRAM chip would be ~5× larger than standard DRAM — but still 20× smaller than a chip with per-row FPUs.

### 4.3 Throughput

- 1 row activation = 30 ns (DDR5 timing)
- All NC_slice containers processed simultaneously per neuron
- H neurons × 30 ns per inference (L0)
- K classes × 30 ns for L1 (score accumulation)
- **Total: (H + K) × 30 ns**

| H    | Latency | Inferences/s/bank |
| ---- | ------- | ----------------- |
| 64   | 2.2 µs  | 450,000           |
| 128  | 4.1 µs  | 240,000           |
| 512  | 15.7 µs | 64,000            |
| 1024 | 31.0 µs | 32,000            |

### 4.4 Ensemble Parallelism

The key architectural insight: ensemble members are **fully independent**. Each occupies separate DRAM rows. A modern DDR5 bank has 65,536 rows — enough for **512 members at H=128** running simultaneously.

```
DRAM Chip:
┌──.─────────────────────────────────────┐
│ Row 0:   W0[0] + Target[0]  (member 0) │  ─→ 32-bit h0 + score
│ Row 1:   W0[1] + Target[1]  (member 0) │
│ ...                                    │
│ Row 127: W0[127] + Target[127]         │
├────────────────────────────────────────┤
│ Row 128: W0[0] + Target[0]  (member 1) │
│ ...                                    │
├────────────────────────────────────────┤
│ Peripheral: Σ scores → argmax          │
└────────────────────────────────────────┘
```

All members compute in parallel — only the score summation (int32 add across 10 classes) is sequential.

### 4.5 Why Backprop Does NOT Work in DRAM

| Operation           | Backprop Training                    | Otto Score Training                      |
| ------------------- | ------------------------------------ | ---------------------------------------- |
| Forward pass        | Float matmul (FMA)                   | MAJ3 + int32 add ← **same as inference** |
| Weight update       | Gradient × LR × momentum (float FMA) | int32 add/sub correction                 |
| Iterations          | Thousands per epoch                  | One correction pass per epoch            |
| Hardware for update | FPU + multiplier                     | int32 adder                              |
| **DRAM-trainable?** | **No** — needs FPU                   | **Yes** — int32 add is DRAM-compatible   |

---

## 5. Comparison: 16-bit vs 32-bit Containers

Earlier experiments used BIN16 (16-bit containers). The switch to **bin32 (uint32)** doubled information capacity per neuron. Key differences:

| Aspect           | BIN16 (legacy)   | Bin32 (current)            |
| ---------------- | ---------------- | -------------------------- |
| Container width  | 16 bits          | 32 bits                    |
| H0 compression   | MAJ3 on 16 bits  | MAJ3 on 32 bits            |
| Training         | Hebbian counting | Iterative Bayes correction |
| MNIST accuracy   | 82%              | 97%                        |
| CIFAR-10         | Not tested       | 61.6%                      |
| Target data type | uint32 (binary)  | int32 (log-odds)           |

The 32-bit container gives **exponentially more information capacity** (2^32 vs 2^16 states per neuron), which directly translates to higher accuracy.

---

## 6. Current Status & Limitations

1. **MNIST: 97-99%** — depends on train/eval ratio
2. **CIFAR-10: 61.6%** — limited by MAJ3 information bottleneck on 256-container input
3. **Ensemble scaling**: H determines convergence speed, not ceiling (all H converge to ~61.5%)
4. **Data-limited**: accuracy scales logarithmically with bit-mass, linearly with training data
5. **No hardware prototype**: estimates based on DRAM timing parameters, not measured silicon
6. **Boolean-only**: suitable for classification, not regression

**The 60% CIFAR barrier is fundamental without dataset-specific prior knowledge.** Every published result above 60% uses data augmentation, transfer learning, or deep architectures — none of which are available in the DRAM bit-logic framework.

---

## 7. Conclusion

Boolean-only neural inference in DRAM-PIM is feasible and achieves **97% MNIST / 61.6% CIFAR-10** with zero floating-point operations at inference time. The key insight: **MAJ3 + int32 add is the only neural operation that fits within DRAM's area, power, and latency constraints.** The Otto Score architecture demonstrates that this minimal instruction set is sufficient for competitive classification accuracy when combined with thermometer encoding and ensemble voting.

**Core innovations:**
1. MAJ3 replaces matmul — 400 transistors per neuron vs 10,000 for FMA
2. Frozen random W0 — never trained, never updated
3. Int32 log-odds target — the only learned parameter
4. Ensemble parallelism — members run simultaneously on separate rows
5. Precomputation — h0 computed once, never recomputed (71% training speedup)

---

## References

- [otto-score.md](otto-score.md) — Full Otto Score architecture
- [ensemble.md](ensemble.md) — Ensemble theory and decoupled workflow
- [encoding.md](encoding.md) — Thermometer encoding for continuous data
- [majority-vote.md](majority-vote.md) — MAJ3 theory, VN and HN specialization
- [human-brain-decoded.md](human-brain-decoded.md) — Extrapolation to biological scale
