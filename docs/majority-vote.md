# Majority Vote (MAJ3) — H0 Computation in Bin32

## What It Is

In the forward-prop uint32 (bin32) architecture, the Majority Vote replaces the
standard dot-product forward pass in the first layer (W0 × Input → H0).

Instead of **one** global sum (`popcount(XNOR(w, x))` → scalar), forward-prop
runs **32 independent** per-bit majority votes organized as a **3-input majority
function (MAJ3)**:

```
For each container position c ∈ [0..NC_slice-1]:
    match[c] = MAJ3( ~(input[c] ⊕ W0[neuron][c]) )
    # MAJ3(a,b,c) = (a & b) | (a & c) | (b & c)
    # → 1 if at least 2 of 3 bits are 1

H0[neuron] = majority_tree(match, NC_slice)
# → 1 bit per position if majority of containers agree
```

Each **32-bit uint32 container neuron** makes 32 independent yes/no decisions.
Unlike standard BNNs that sum all bits into one scalar, forward-prop keeps each
bit-position decision intact — a neuron is a 32-bit vector, not a number.

### How MAJ3 Works

MAJ3 takes three bits (a, b, c) and returns 1 if at least two of them are 1.
This maps directly to 3-input AND-OR logic:

```
MAJ3(a, b, c) = (a & b) | (a & c) | (b & c)
```

On a DRAM row, this can be evaluated in a single cycle using three bitwise
operations — no addition, no carry propagation.

The full H0 computation for one neuron is:

```
match[c] = MAJ3( ~(input[c] ⊕ W0[c]) )       // per container
H0       = majority_tree(match, NC_slice)      // per neuron
```

## Historical Lineage

| Concept                                 | Origin                                                                                           | Year      |
| --------------------------------------- | ------------------------------------------------------------------------------------------------ | --------- |
| Majority Function (Boolesche Logik)     | McCulloch & Pitts (threshold neuron), von Neumann (reliable computation)                         | 1943–1956 |
| MAJ3 (3-input majority)                 | Standard cell library primitive in ASIC design                                                   | 1970s+    |
| XNOR + Popcount for BNNs                | Rastegari et al., "XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks" | 2016      |
| BNN Inference (BinaryConnect)           | Courbariaux et al., "BinaryConnect" / "Binarized Neural Networks"                                | 2015–2016 |
| FINN Framework (XNOR-Popcount Pipeline) | Umuroglu et al., "FINN: A Framework for Fast, Scalable Binarized Neural Network Inference"       | 2017      |

## What Makes the Forward-Prop Version Novel

None of the prior BNN papers (XNOR-Net, FINN, BinaryConnect) use a per-bit
majority vote. They all compute a **single global popcount** per output neuron:

```
# Standard BNN inference (XNOR-Net, FINN):
output[neuron] = sign(popcount(XNOR(w, x)) - threshold)
# → one scalar decision, 32 bits collapse into one number
```

```
# Forward-Prop per-bit majority vote (bin32 via MAJ3):
for c in 0..NC_slice-1:
    match[c] = MAJ3( ~(input[c] ⊕ W0[c]) )
H0 = majority_tree(match, NC_slice)
# → 32 independent decisions, each bit preserved individually
```

### Key Differences

1. **Container Architecture**: Each neuron is a **uint32 container** (32 bits),
   not a scalar. Weight matrices are `H × NC` containers of `uint32`.
2. **Bit Independence**: All 32 bits of a neuron are decided independently.
   Information flows through bit positions without cross-talk.
3. **No Summation Across Bits**: Unlike standard BNNs that sum all 32 XNOR bits
   into one popcount, forward-prop keeps bits separate.
4. **MAJ3 over XNOR+popcount**: DRAM rows can evaluate MAJ3 directly using
   bitwise AND/OR — no addition or popcount needed for H0.

## VN and HN — Specializing and Generalizing MAJ3

The MAJ3 computation can be specialized or generalized via two orthogonal
parameters: **HN** (Horizontal Split) and **VN** (Vertical Neurons / splitVN).

```
                    Specialized ← MAJ3 → Generalized
                          ↑                   ↑
                        HN>1                VN>1
                (narrower input)       (broader bit-group)
```

### HN > 1 — Specialization (narrower input)

When `--splitHN` is greater than 1, the input containers are divided into
**HN independent slices**. Each slice is processed by a separate member with
its own W0, seeing only `NC / HN` containers of the input.

```
HN=1:  neuron sees all 196 containers → broad but shallow
HN=2:  neuron sees 98 containers      → narrower, more specialized
HN=4:  neuron sees 49 containers      → highly specialized
```

HN specialization is useful when the input has **many redundant channels**
(e.g., RGB color channels for CIFAR-10 with opponent encodings). Each HN
member learns a different aspect of the input — one may focus on luminance,
another on color opposition.

### VN > 1 — Generalization (broader bit-groups)

When `--splitVN` is greater than 1, the 32 bits of each neuron are **grouped
into VN virtual neurons**, each consisting of `32 / VN` bits. The MAJ3 output
for a group is set only if **all bits in that group agree** (strict AND).

```
VN=1:  32 independent bits  → 32 virtual neurons  (most specific)
VN=2:  16 groups × 2 bits   → 16 virtual neurons  (AND2: both bits must fire)
VN=3:  10 groups × 3 bits   → 10 virtual neurons  (AND3: all 3 must fire)
VN=4:   8 groups × 4 bits   →  8 virtual neurons  (AND4)
```

Each virtual neuron requires **broader agreement** across bit-positions,
forcing the neuron to detect more robust patterns. This prevents overfitting
to narrow pixel correlations.

#### VN=2 and CIFAR-10 — Forcing Generalization

The fixed input vector for CIFAR-10 is only **256 containers** (4 RGB blocks
× 64 containers each). With H=128 and VN=1, the network has 128 × 32 = 4096
virtual neurons — but too many degrees of freedom relative to the input width.

**VN=2** reduces this to 128 × 16 = 2048 virtual neurons and forces each
virtual neuron to be **twice as selective** (both bits in the group must
agree). This consistently improves CIFAR accuracy by **3–6 percentage points**
compared to VN=1 at the same H.

```
CIFAR-10, H=128, 7 epochs:
  VN=1:  51.7%  (4096 virtual neurons, too many degrees of freedom)
  VN=2:  55.5%  (2048 virtual neurons, forced generalization)  ← champion
  VN=3:  35.5%  (1280 virtual neurons, too strict for H=128)
```

For MNIST (clean binary data, wider input), VN=1 is optimal — no
generalization pressure needed.

### VN + HN Combined

VN and HN compose orthogonally: each HN slice gets its own MAJ3 evaluation,
and within each MAJ3 result, VN grouping applies.

```
Total virtual neurons = HN × VN × H
```

Example: `H=128, HN=2, VN=2` → 128 × 2 × 16 = 4096 virtual neurons,
each specialized by HN (half the input) and generalized by VN (AND2).

## Development Timeline

| Date       | Milestone                                                                  |
| ---------- | -------------------------------------------------------------------------- |
| 2026-05-22 | BV32 relaunch: first XNOR+popcount experiments with 8-bit containers       |
| 2026-05-23 | Per-bit majority vote formulated as core L0 forward pass                   |
| 2026-06-01 | BIN16 Container: 16-bit majority vote, Hebbian K-iteration                 |
| 2026-06-19 | **Bin32 switch**: uint32 containers replace bin16 — more bits → more information capacity |
| 2026-06-22 | **MAJ3 replaces per-bit popcount**: 3-input majority on (W, X, ~X⊕W) eliminates popcount from H0 layer |
| 2026-07    | **int32 Container MLP**: Otto Score uses int32 targets + offsets for classification |

## Encoding

Input pixels (8-bit) are encoded into 32-bit containers using configurable
encoding schemes (`--encoding`): raw, exp, sig, lin, etc. Each encoding maps
the pixel value to a 32-bit pattern that preserves information density.

The MAJ3 operation works on the encoded input containers and the frozen random
W0 containers — no per-pixel encoding overhead after the initial `load_input`.

## Related Concepts (Not Used Here)

- **Majority Logic Decoding** (error-correcting codes): Meggitt 1963 — majority
  vote over received bits to correct errors. Related but different domain.
- **Hopfield Networks** (1982): Majority update rule for associative memory.
  Used majority to settle into attractor states.
- **Hyperdimensional Computing** (Kanerva 2009): Majority vote for bundling
  hypervectors. Operates on entire vectors, not per-bit over inputs.
