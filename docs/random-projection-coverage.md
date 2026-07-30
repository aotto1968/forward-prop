# Random Is Not a Hack — Deterministic Coverage of 8192! Permutations via 67 Million Binary Projections

**forward-prop Research — July 2026**

---

## Abstract

Modern neural architectures constrain the input space through hand-crafted inductive biases: convolutions assume translation invariance, attention assumes pairwise token interactions, and fully connected layers assume dense but trained connectivity. All of these are **subsets of the complete permutation space** — they implicitly bet that the relevant input interactions fall within their chosen subset. We argue that for DRAM-native binary inference, this bet is unnecessary. A **random binary projection ensemble** of width H approximates the complete permutation space `8192!` with `(8192×32) × (H×32)` deterministic connections — for H=256, this is 67,108,864 XOR+popcount operations. As H scales, the coverage converges to the full space almost surely. W0 requires no training, no gradient, and no weight updates; it is a fixed, hardware-addressable index into the permutation space. The only learned component is the Bayesian target matrix W1, which counts class-conditional co-occurrences of the projected bits.

---

## 1. The Problem: 8192! Is Not Learnable

A CIFAR-10 image consists of `32 × 32 × 3 = 3072` pixels, encoded into 256 containers of 32 bits each: **8192 input bits**. In the information-theoretic limit, every input bit should be able to interact with every other input bit. The number of distinct binary relations over the input space is:

| Model                             | Possible Bit Interactions | Coverage                      |
| --------------------------------- | ------------------------- | ----------------------------- |
| Full Permutation Space            | **8192!** ≈ 10^28954      | 100%                          |
| Fully Connected (trained)         | 8192 × H                  | vanishing                     |
| Convolution (3×3)                 | 9 × 32 × channels         | negligible                    |
| Attention (single head)           | 1024 × 1024               | sparse                        |
| **Random Projection (this work)** | **8192 × (H × 32)**       | **deterministic, H-scalable** |

8192! exceeds the number of atoms in the observable universe by a factor of roughly 10^28874. No known or foreseeable algorithm can search, train, or enumerate it.

---

## 2. The Solution: Random Projection as Permutation Coverage

The central insight of this work is: **not knowing which permutation is important implies that all permutations must be equally accessible.** Any deterministic subset of the permutation space (convolution, attention, hand-crafted features) privileges specific interactions and ignores others. Random projection is the only strategy that privileges no subset — it provides **uniform coverage** of the space.

### 2.1 How It Works

Let `x ∈ {0,1}^8192` be the binary input (256 containers × 32 bits). For each neuron `h ∈ [0, H)`:

```
W0_h  ∈ {0,1}^8192      (random, fixed, never trained)
match_h = popcount(x ⊕ W0_h) / 8192
h0_h   = match_h > threshold
```

The projection from `8192` bits to `H` bits is a **random linear map modulo the majority threshold** — a binary Johnson-Lindenstrauss embedding. The key difference from standard JL is that the projection weights are **never optimized**. They are as fixed as a memory address.

### 2.2 Coverage Count

Each neuron connects to all 8192 input bits via W0. The total number of **determined connections** is:

```
8192 input bits × H × 32 (W0 bits per neuron) = 8192 × H × 32
```

For H=256: `8192 × 256 × 32 = 67,108,864 ≈ 6.7 × 10^7`.

This is the number of distinct XOR operations that the inference hardware must support. Each operation is a single bit gate — no multiply, no add, no floating point.

### 2.3 Coverage as H → ∞

The random projection ensemble does not approximate the permutation space — it **is** a deterministic, linearly scaling sample of it. As H increases, the probability that any input bit pair remains unconnected in at least one neuron decreases exponentially:

```
P(unconnected pair) ≤ (1 - 1/8192)^(H × 32) ≈ exp(-H × 32 / 8192)
```

For H=256: `exp(-256 × 32 / 8192) = exp(-1) ≈ 37%` — expected 1 missing interaction per bit pair.
For H=1024: `exp(-4) ≈ 1.8%`.
For H=8192: essentially zero.

### 2.4 Why This Is Not a Hack

- **No free parameters in W0.** The random projection is a fixed computational primitive — like the address decoder in a DRAM chip.
- **No gradient through W0.** Backpropagation never touches W0. It cannot.
- **No assumption about the data.** Unlike convolutions (assume locality), attention (assume pairwise), or MLPs (assume dense+continuous), random projections assume nothing.

The approach is "lazy" in the mathematical sense: we defer all learning to the Bayesian counting layer W1, which simply accumulates class-conditional frequencies of the projected bits.

> "We do not know which interactions matter. Therefore we must make all interactions equally possible. Random projection is the cheapest known primitive that achieves this."

---

## 3. Relation to Known Theory

### 3.1 Johnson-Lindenstrauss Lemma

The JL lemma states that a random linear projection from `ℝ^d → ℝ^k` preserves pairwise distances with high probability when `k = O(log N / ε²)`. Our binary projection is a special case: the dimensionality reduction is from 8192 to H bits, and the distance metric is Hamming distance preserved by XOR+popcount.

### 3.2 Rahimi & Recht (2007) — Random Features

Random Fourier features approximate a shift-invariant kernel by projecting inputs through a random weight matrix. Our W0 is structurally identical, except:

| Aspect         | Random Features   | This Work                |
| -------------- | ----------------- | ------------------------ |
| Projection     | Continuous, float | **Binary, bitwise**      |
| Nonlinearity   | cos / sin         | **popcount + threshold** |
| Output weights | Trained (ridge)   | **Bayesian counting**    |
| Hardware cost  | FMA units         | **XNOR gate + popcount** |

### 3.3 Novelty: The Learning Is in W1, Not W0

Standard random feature methods still train the output layer. In this work, the output layer (W1) is not trained via gradient descent — it is a **class-conditional frequency counter**. Each entry `W1[k][h][v]` records how often virtual neuron `(h, v)` fired for class `k` during training. At inference:

```
score(k) = class_offset[k] + Σ_{h,v active} W1[k][h][v]
```

This is Bayesian inference, not optimization. There is no learning rate, no momentum, no weight decay. The only "learning" is counting.

---

## 4. Implications for DRAM Hardware

### 4.1 W0 Is a Fixed Mask

Since W0 is random and never trained, it can be:

- **Hard-wired** during chip fabrication
- **Initialized once** from a ROM at power-on
- **Shared across all ensemble members** (as done in this work with `--seed-member const`)

There is no need for W0 memory to be writable. This saves significant die area: writable SRAM costs 6 transistors per bit; read-only storage costs 1.

### 4.2 Scaling with H

Coverage scales linearly with H. Since each DRAM row contributes one neuron's majority computation, H is bounded by the number of rows available in the DRAM array:

```
H_max = DRAM_rows / (containers per neuron)
```

For a commodity DDR4 module with 65536 rows and 256 containers per neuron: `H ≈ 256`. Future 3D-stacked memory (HBM) provides orders of magnitude more rows.

### 4.3 The Resulting Chip

| Component      | Type             | Count (H=256)                |
| -------------- | ---------------- | ---------------------------- |
| W0 storage     | ROM / fixed mask | 256 × 8192 = 2,097,152 bits  |
| XNOR gates     | combinatorial    | 8192 per neuron × 256 ≈ 2M   |
| popcount trees | combinatorial    | 256 × 1 (shared per row)     |
| MAJ3 / MAJ1    | threshold gate   | 256                          |
| W1 storage     | DRAM (writable)  | 256 × 10 × 32 = 81,920 int32 |
| Scoring        | int32 adders     | 10 per image                 |

Total transistor count: ~2M for logic — comparable to the row decoders of a single DRAM bank. No FPUs, no multipliers, no gradient logic.

---

## 5. Empirical Validation

The Otto Score classifier (the implementation of this architecture) achieves:

| Dataset       | H   | Epochs | Test Accuracy | Members       |
| ------------- | --- | ------ | ------------- | ------------- |
| MNIST         | 128 | 5      | 97.1%         | 1             |
| Fashion-MNIST | 256 | 5      | 90.2%         | 1             |
| CIFAR-10      | 256 | 10     | 68.19%        | 94 (ensemble) |
| CIFAR-10      | 512 | 10     | 71.4%         | 1             |

These results are **without any training of W0**. The random projection is generated by a splitmix64 PRNG with a fixed seed. The only trained component is the Bayesian target matrix W1.

---

## 6. Conclusion

Random projection in this architecture is not a heuristic, not a shortcut, and not a concession to hardware constraints. It is a **deterministic, scalable strategy for covering the intractable permutation space of binary inputs**. The fixed, untrained W0 ensemble privileges no interaction pattern, reduces the space from `8192!` to `67M × H`, and maps directly onto DRAM-native bitwise logic. The only learning is counting.

> "We do not know which bits matter. Therefore we must give every pair of bits an equal chance to matter. Random projection is the cheapest known way to do this — and for DRAM, it is the only way."
