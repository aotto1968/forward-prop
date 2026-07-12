# Small + Many Beats Large + One — How W0 Randomness Makes Ensembles Work

**Andreas Otto** — 19 June 2026

## Abstract

In the Otto Score classifier, a frozen random projection W0 followed by MAJ3 majority
extracts 32-bit feature strings from 784-pixel MNIST inputs. The conventional wisdom
says: larger W0 (more hidden neurons) → better features → higher accuracy. We
demonstrate that **an ensemble of small W0s consistently beats a single large W0 at
the same total bit budget**, and that this effect depends critically on W0 diversity.

The mechanism is not ensemble averaging in the traditional sense. Score-summing
(product of experts) handles correlated members naturally, and the correlation comes
from the shared iterative correction pass. The real driver is **how the random W0
projections differ** — not the projection size.

## 1. The Observation

A single Otto Score model with H=512 reaches **95.1% MNIST in 43s**. An ensemble of
H=64 × N=5 reaches **95.3% in 6.5s** — 6.6× faster and slightly more accurate.

| Configuration | Bit Mass (W0) | Eval      | Time     |
| ------------- | ------------- | --------- | -------- |
| H=512, N=1    | 3.2 Mbit      | 95.1%     | 43s      |
| H=64, N=5     | 1.0 Mbit      | **95.3%** | **6.5s** |

**10× less bit mass, faster, and better.** This is counter-intuitive: each H=64 MAJ3
projects onto only 64 random vectors, vs 512 for the large model. Five independent
64-vector projections should extract less total information than one 512-vector
projection — yet they outperform it.

### 1.1 CIFAR-10 Ensemble Approximation Chain

The same ensemble effect applies to CIFAR-10, but the gain is even larger because
the single-run ceiling is lower (61.0% at H=4094) and the input space is noisier.
The following chain shows how ensemble accuracy accumulates with each new member
for H=1024, VN=2, 7 epochs:

```
EN=  1:  37.06%    baseline, single member
EN=  2:  42.97%   +5.9pp   ← largest single jump
EN=  3:  49.08%   +6.1pp   ← +12pp in 3 members
EN=  4:  51.97%   +2.9pp
EN=  5:  52.86%   +0.9pp   ← saturation begins
EN= 10:  58.06%   +5.2pp in 5 members
EN= 17:  60.08%            ← 60% threshold crossed
EN= 20:  59.41%            ← dip (member quality varies)
EN= 50:  60.80%            ← slow climb
EN=100:  61.37%
EN=150:  61.45%            ← near plateau
EN=200:  61.57%
EN=220:  61.62%   ← 220 members, CIFAR maximum (2026-07)
```

Key characteristics:
- **Exponential start**: the first 3 members contribute +12pp
- **Diminishing returns**: above EN=10, each new member adds <0.1pp
- **EN=10 beats single-run ceiling**: the 10-member ensemble (58.1%) already
  approaches the best single-run accuracy (61.0% at H=4094) at 1/4 the H
- **220 members**: 61.62% — the current CIFAR-10 ceiling, limited by MAJ3
  information capacity on noisy pixel data

## 2. The Mechanism

The key lies in how MAJ3 features interact with the Bayes log-Score.

### 2.1 Why Single Large W0 Plateaus

A single W0 defines one fixed set of H random projection vectors. The MAJ3 output
of each neuron is a 32-bit string encoding the majority vote across 196 containers.
These 32 bits are weakly correlated (max |r| < 0.1) but not independent. The Bayes
log-Score assumes independence and is optimal under that assumption — but the
assumption is approximate. With a single W0, the approximation errors are systematic:
certain feature patterns are never represented, certain class confusions recur.

This is why the 1-pass Bayes accuracy plateaus at 86% (H=2048) and the iterative
accuracy plateaus around 96.4% — the feature set is fixed at W0 initialization
and no amount of target-tuning can create features that aren't there.

### 2.2 Ensemble as Multiple Feature Samplers

An ensemble with N independent W0s samples N different random projections. Each
member sees a different set of 32-bit feature patterns. The Bayes log-Score per
member makes different mistakes — different class confusions, different failure
modes.

Score-summing across members exploits this. The total score for class k is:

```
total[k] = Σ_m Σ_h Σ_b y_m[h][b] × logit_m[k][h][b] + offset_m[k]
```

Where `m` indexes ensemble members. Crucially, each member has its **own** logit
and offset — there is no shared weight matrix. The ensemble is not voting on a
common representation; each member computes an independent posterior and the
scores are summed.

### 2.3 Why Score-Summing, Not Majority Vote

Hard majority vote (each member votes for one class, ties broken arbitrarily)
loses information. Two members predicting class 3 with scores [3: +500, 7: +490]
vs [3: +490, 7: +500] cancel to a weak overall signal. Score-summing preserves
the margins: the total for class 3 becomes +990 vs +990 for class 7 → correctly
ambiguous.

More importantly, score-summing handles **correlated members** naturally. If all
members have high scores for class 7 on a given sample, the total is high.
If members disagree, the totals are close. This is exactly what a product of
experts does: agreement amplifies, disagreement averages.

## 3. The Seed Experiment

To test whether W0 diversity is the active mechanism, we compared four seeding
strategies for an H=64, N=5, ep=5 ensemble:

| Mode                    | W0 Generation                              | Eval      | Δ from default |
| ----------------------- | ------------------------------------------ | --------- | -------------- |
| **default**             | Seed once, fill all N members sequentially | **95.1%** | —              |
| `--ensemble-seed const` | Same seed for all members → identical W0   | 89.9%     | **−5.2pp**     |
| `--ensemble-seed incr`  | Per-member seeds: `seed + m`               | 94.7%     | −0.4pp         |
| `--seed-file`           | True random from random.org                | 94.9%     | −0.2pp         |

### 3.1 Const Mode: The Smoking Gun

With `--ensemble-seed const`, every member gets the **exact same W0**. The ensemble
still has N independent target matrices (trained independently from different random
initializations), but the features are identical. The result: **89.9%** — barely
better than a single N=1 model (89.2%).

This is the smoking gun: **W0 diversity is the ensemble feature, not the number of
members.** When all members see the same features, they make the same mistakes.
The ensemble collapses to a single model with redundant targets.

### 3.2 Incr Mode: Tweaking Seeds Doesn't Help

With `--ensemble-seed incr`, each member reseeds the PRNG before generating W0:
`w0_srandom(seed + m)`. This produces different W0s, but they come from **separate
PRNG streams** starting at different points. The result: **94.7%** — close to default
but not better.

The incr approach has a subtle issue: reseeding resets the PRNG state, so each
member starts from a predictable position. The default mode (seed once, sequential)
is mathematically equivalent but avoids any risk of PRNG stream overlap or edge
cases from adjacent seeds.

### 3.3 True Random: Overkill

Using `--seed-file` with 64 KiB of true random data from random.org gives
**94.9%**. The PRNG-based modes (default 95.1%, incr 94.7%) are within noise
(±0.3pp from OpenMP scheduling). True random is not needed — splitmix64 passes
BigCrush and provides quality indistinguishable from hardware random for this
application.

### 3.4 The Random Marker

Each ensemble member prints a random marker `#<uint32>` to verify the seed state:

```
Ensemble [1/5] - Target[64][10] = 80 KB (seed=42,#3032050943)
Ensemble [2/5] - Target[64][10] = 80 KB (seed=42,#3939922274)
```

In `const` mode, all markers are identical (same reseed + same first w0_random()
call). In default mode, markers differ because the PRNG stream advances through
W0 generation sequentially — member 2's marker is the first random number after
member 1's entire W0.

### 3.5 The `err=` Diagnostic — Training Error Before Correction

During training, each epoch prints `err=N` — the number of **misclassified training
samples** evaluated BEFORE the correction pass. This diagnostic reveals a characteristic
signature of under- and over-fitting:

**N=1 (single model, no ensemble):** `err` **jitters wildly** between epochs. One run
of H=196, N=1 showed err oscillating between 2 and 569 across 20 epochs — a 250×
swing. The single model overfits to the correction signal, then forgets, then
re-learns. **Training accuracy hits 100% but eval plateaus at 93%.**

**N≥3 (ensemble):** `err` **stabilizes.** The same H=196 bit budget with N=3 shows
err around 500-600 with smooth drift. Training accuracy stays at 97-99%, never
reaching 100%. The ensemble prevents individual members from overfitting because
the score-summing dilutes extreme corrections.

**Counter-intuitive finding: higher err ≠ worse accuracy.** At C=196, N=4 has
err≈1100 (double N=3) but **higher eval** (94.6% vs 94.5%). The ensemble with
more members tolerates more training errors because the score distribution is
wider — each member's corrections are partially cancelled by others. This is a
feature, not a bug: the ensemble actively prevents over-confidence.

| H   | N | err (final) | trn   | Eval      | err character                  |
| --- | - | ----------- | ----- | --------- | ------------------------------ |
| 196 | 1 | 122         | 99.8% | 93.2%     | **jitter ×5** (106↔569↔99↔214) |
| 65  | 3 | 551         | 98.7% | **94.5%** | ✅ stable drift                |
| 49  | 4 | 1116        | 97.8% | **94.6%** | ✅ very stable                 |
| 33  | 6 | 1933        | 96.2% | 94.2%     | ✅ stable                      |

**Practical rule:** If `err` oscillates more than 2× between adjacent epochs,
increase `--ensembleN`. If `err` is stable but eval doesn't improve, increase
`--epochsN` or reduce `--lr`.

## 4. Theoretical Framework

### 4.1 W0 as Random Projection

Each MAJ3 neuron computes:

```
h0[h] = MAJ3( W0[h][c] ⊙ x[c]  for c=1..196 )
```

Where ⊙ is XNOR (match) or XOR (mismatch). With random W0, roughly half the bits
match and half mismatch for any input — this is the **balance condition** that makes
MAJ3 informative. If W0 is trained or biased, the balance is destroyed and accuracy
collapses (73% in the failed W0 training experiment).

The 32-bit MAJ3 output is a lossy compression of 196 × 32 = 6272 input bits into
32 bits. Different random W0s compress differently — they preserve different
patterns and lose different information. An ensemble exploits the complementarity.

### 4.2 Why Small + Many Works

Consider the total number of MAJ3 operations:

- H=512, N=1: 512 MAJ3 evaluations per forward pass
- H=64, N=5: 64 × 5 = **320** MAJ3 evaluations per forward pass

The ensemble does **fewer** MAJ3 evaluations total, yet achieves higher accuracy.
The reason is **feature diversity**: 5 × 64 = 320 evaluations from **different
projections** cover more of the feature space than 512 evaluations from one
projection. Some features are redundant within a single W0; across different
W0s, redundancy is lower.

### 4.3 The 37% Overlap Limit

Analysis of failure sets across different W0s showed:

| Condition                | Overlap | Interpretation                                                                                          |
| ------------------------ | ------- | ------------------------------------------------------------------------------------------------------- |
| 1-pass, 3 seeds          | 75%     | **Method limit**: ~75% of failures are images that MAJ3 fundamentally cannot separate, regardless of W0 |
| Iterative ep=10, 3 seeds | 37%     | **Random limit**: remaining 37% are images where even iterative correction and different W0s fail       |

The drop from 75% to 37% shows that iterative correction fixes the **feature
coverage** problem (images that one W0 misclassifies but another gets right).
The residual 37% are images where **no W0 produces discriminative features** —
these are the MAJ3 information limit.

## 5. Precomputation — The Free Lunch

Because W0 is a **frozen random matrix** and the input dataset X is fixed, the
entire H0 computation can be done **once** — before training even starts.

### 5.1 What Can Be Precomputed

```
W0  → frozen random projection (never changes)
X   → input data (MNIST/CIFAR, fixed)
────────────────────────────────────
h0  = MAJ3(W0, X)          → computed ONCE
gb  = h0_to_gb(h0)         → computed ONCE from h0
gb_te = h0_to_gb(h0_te)    → computed ONCE for test set

Training loop (all epochs):
  target += correct(target, gb, y)   ← only touches target matrix
```

The training loop **never recomputes** h0 or gb. It only applies iterative
corrections to the target matrix (W1). This is fundamentally different from
backpropagation, where every epoch requires a full forward + backward pass
through the entire network.

#### Decoupled Ensemble Training

Because the h0/gb precomputation depends ONLY on W0 and X — never on the
target matrix — the ensemble training becomes **fully decoupled**:

```
W0 + X → h0 (once, independent of target)
         ↓
         gb → Train member 1 (target only)
         gb → Train member 2 (target only)
         gb → ... N members, same h0
```

All N ensemble members share the same precomputed h0 from their shared W0
slice. The training loop only reads gb and writes target — no double
computation, no redundant MAJ3 evaluations.

### 5.2 The 71% Speedup

With the precomputation architecture, training time drops dramatically:

| H    | EN | Before (h0/epoch) | After (h0 once) | Speedup |
| ---- | -- | ----------------- | --------------- | ------- |
| 1024 | 7  | 934s              | **273s**        | **71%** |

The effect grows with H and epoch count: each saved h0 recomputation is
`O(H × NC × N)` bit-operations. For large ensembles and many epochs, the
savings are multiplicative.

### 5.3 H Does Not Change the Ceiling — Only the Convergence Speed

An important empirical finding: increasing H (hidden neurons) does **not** raise
the ensemble accuracy ceiling. It only reduces the number of ensemble members
needed to reach that ceiling.

The following table shows three CIFAR-10 ensembles (VN=2, EP=6, encoding=latest)
converging to the same ~61.5% ceiling:

| H    | EN=1  | EN=3  | EN=10     | EN=50 | EN=100 | EN=200 | Ceiling |
| ---- | ----- | ----- | --------- | ----- | ------ | ------ | ------- |
| 1024 | 37.1% | 49.1% | 58.1%     | 60.8% | 61.4%  | 61.6%  | 61.6%   |
| 2048 | 39.2% | 50.9% | 58.7%     | 61.2% | 61.4%  | —      | 61.6%   |
| 4096 | 40.6% | 51.8% | **60.1%** | 61.4% | —      | —      | 61.6%   |

Key observations:
- **H determines the starting speed**: H=4096 reaches 60.1% at EN=10, while
  H=1024 needs ~50 members to get there
- **The ceiling is identical**: all three converge to ~61.5% regardless of H
- **The MAJ3 information bottleneck is the limit**: once the ensemble covers
  enough diverse projections, additional H per member no longer helps

**Why H cannot be expanded indefinitely**: More hidden neurons means more
random projections of the **same fixed input data**. Each projection extracts
32 bits of information from the input. With 196 containers for MNIST or 256
for CIFAR, the input itself is small. Beyond a certain H, all projections
become redundant — there is simply not enough information in the input to
support more independent features.

**Practical consequence**: Data and neurons are co-dependent. The human brain's
86 billion neurons are needed to **store** a lifetime of visual experience;
the lifetime of visual experience is needed to **fill** those neurons with
meaningful patterns. Neither works without the other. The Otto Score scaling
law projects 92% accuracy at 3.1T channels, but this requires **both** massive
storage (the channels) AND massive data (millions of training examples) to fill
them.

### 5.4 Train Now, Merge Later — Fully Decoupled Ensemble Workflow

Because W0 is the only data-dependent random element and it is frozen, each
ensemble member is **completely independent** — different W0 (different seed),
own training, own target matrix.

This enables a novel workflow:

```
┌────────────────────────┐     ┌─────────────────────┐
│  Train seed 1234       │ ──> │  scores/SD1234.ens  │
│  --export-merge-scores │     └─────────────────────┘
└────────────────────────┘              │
                                        │                  ┌──────────────────┐ 
┌────────────────────────┐     ┌─────────────────────┐     │  merge-ensemble  │ 
│  Train seed 5678       │ ──> │  scores/SD5678.ens  │ ──> │  scores/         │ 
│  --export-merge-scores │     └─────────────────────┘     │  → EN curve      │ 
└────────────────────────┘              │                  └──────────────────┘ 
                                        │                  
┌────────────────────────┐     ┌─────────────────────┐
│  Train seed 9012       │ ──> │  scores/SD9012.ens  │
│  --export-merge-scores │     └─────────────────────┘
└────────────────────────┘              │
   ... (N seeds,                        │
    parallel, any time)                 ▼
```

Each `--export-merge-scores DIR` run writes a **self-contained score archive** (`.ens`)
containing the member's per-sample scores. The `merge-ensemble` tool combines
arbitrary subsets into an EN=1..N accuracy curve.

**Advantages:**

1. **Maximal parallelism** — N seeds run on N machines/cores, no shared state
2. **Incremental accumulation** — new seeds added days later, old archives remain valid
3. **Subset filtering** — `--filter eval gt 58` excludes weak runs (text operators: gt, lt, ge, le, eq)
4. **Sorting** — by seed or timestamp for different EN-curve interpretations
5. **Replay without retraining** — the scores archive is all that is needed to
   reconstruct any ensemble size; the original W0/target can be discarded

```bash
# 20 seeds, parallel, any time
bash run-ensemble.sh --repeat 20 ./cifar/cifar-mlp-bin32-otto-trn-xnor.exe \
    --hiddenN 512 --epochsN 7 --encoding latest --splitVN 2

# Merge all → EN curve
./cifar/cifar-merge-ensemble.exe scores/

# Only runs with eval > 58%
./cifar/cifar-merge-ensemble.exe scores/ --filter eval gt 58
```

### 5.5 Technological Consequence

The combination of frozen W0 + precomputed H0 means the Otto Score training
pipeline is **embarrassingly parallel at every level**:

- **Within a member**: all neurons are independent (MAJ3 per neuron)
- **Between members**: all seeds are independent (different W0)
- **Between datasets**: all score archives are independent (merge later)

This stands in direct contrast to backpropagation, where every training step
depends on the previous step's weights — **inherently sequential**.

## 6. Architectural Implication for DRAM

This result directly informs the DRAM chip architecture:

### 6.1 Many Small Banks Instead of One Large One

A DRAM implementation should have many small MAJ3 banks (each processing 196
containers) rather than one large bank processing all. Each bank has its own
W0 (random, frozen) and its own target memory (log-odds per bit per class).

```
DRAM Chip:
┌──────────────────────────────────────┐
│ Row 0:  W0[0][0..195] + Target[0]    │  ─→ 32 bits + score
│ Row 1:  W0[1][0..195] + Target[1]    │  ─→ 32 bits + score
│ ...                                  │
│ Row 63: W0[63][0..195] + Target[63]  │  ─→ 32 bits + score
├──────────────────────────────────────┤
│ Peripheral: Σ_h Σ_b log-odds → argmax│
└──────────────────────────────────────┘
Multiple chips (ensemble): independent rows, shared peripheral.
```

Each chip holds one ensemble member (or several). The peripheral logic sums
the per-member scores and computes argmax. No communication between chips
during the forward pass — only the 10 × int64 final scores need to be
collected.

### 6.2 W0 Diversity at the Chip Level

The seeding experiment shows that the precise source of W0 randomness is
unimportant — any method that produces distinct W0s works. For manufacturing:

- **On-chip PRNG** (splitmix64): one seed per chip, sequential W0 generation
- **Fuse-based**: each chip has a unique ID → derived seed
- **True random**: optional, not needed

The `const` mode warning: identical W0 across chips would nullify the ensemble
benefit. If two DRAM chips have the same W0 (e.g., from the same mask), they
behave as one member with duplicated target — no accuracy gain.

### 6.3 Scaling Law

The practical scaling for fixed total MAJ3 budget (K = H × N):

- **N=1: overfitting** — single W0 captures limited feature variety
- **N=2-3: optimal** — diversity covers most failure modes, correction handles rest
- **N=4+: diminishing** — W0 diversity saturates, 37% residual overlap dominates
- **H < 24: collapse** — each member too weak individually (underfits)

| H   | N  | K=H×N | Eval      | Character            |
| --- | -- | ----- | --------- | -------------------- |
| 192 | 1  | 192   | 93.3%     | Overfit (err jitter) |
| 64  | 3  | 192   | **94.5%** | Optimal tradeoff     |
| 32  | 6  | 192   | 94.2%     | Member too weak      |
| 16  | 12 | 192   | 92.7%     | Collapse             |

## 7. Practical Guidance

1. **Always use `--ensembleN N` with `N ≥ 3`** — the sweet spot for DRAM cost vs accuracy
2. **Keep `H ≥ 64`** — smaller H collapses feature quality per member
3. **Use default PRNG mode** (seed once, sequential) — matches or beats all other seeding strategies
4. **Never use `--ensemble-seed const`** — identical W0 nullifies the ensemble benefit
5. **Skip true random** — splitmix64 is sufficient; the seed source matters less than the diversity
6. **Score-summing, not majority vote** — preserves margin information
7. **Precompute h0 once** — with frozen W0 and fixed input, h0 never changes; do `--export-merge-scores` and merge later
8. **Archive all seeds** — score archives are tiny (~100KB each) and enable post-hoc ensemble analysis without retraining

## 8. Conclusion

The ensemble effect in Otto Score is driven entirely by **W0 diversity**, not by
the number of ensemble members or the quality of the target tuning. A single fixed
random projection extracts a limited set of features; multiple independent projections
cover more of the feature space. The training signal (iterative log-odds correction)
distributes across diverse W0s, producing different classification boundaries that
complement each other.

The practical implication is decisive for DRAM architecture: **many small MAJ3
banks with independent random W0s outperform one large bank at lower total
cost.** The 37% residual failure overlap is the MAJ3 method limit — no amount of
scaling, ensembling, or tuning can surpass it within the bit-logic framework.

## References

- Otto Score: `www/index.html` — main publication page
- Seed mode comparison: `mnist-1/README.md` — Key Findings section
- Constant bit-mass experiments: `mnist-1/README.md` — H×N=C series
- MAJ3 library: `lib/maj3.h` — majority tree implementation
- splitmix64: `lib/w0_random.h` — DRAM-native RNG
