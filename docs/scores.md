# Scores — Exact Definition and Computation

**Scope:** What a *score* is, how it is computed, and how it is stored in the
`.ens` archive format. This is the reference for the `--export-merge-scores`
pipeline (trainer → `.ens` files → `merge-ensemble`).

**TL;DR:** A score `sc[s][k]` is a **Bayes log-likelihood value** (a logit)
of test sample `s` for class `k`. It is **not** a weight, **not** a
probability, and **not** a model. The `.ens` file stores only the score
table (plus labels + header) — the W1 weights are consumed during
computation and discarded.

---

## 1. Data Flow Overview

```
Input (784 px)
   │  ki-load.h: encode + pack into 32-bit containers
   ▼
Container in[h]  (uint32, 4 px @ 8-bit / 2 px @ 16-bit)
   │  gb = ki_gb_for_neuron(in, W0, h, ...)
   ▼
gb[s][h]  (32-bit activity mask)
   │  training: ki_build_target_from_gb → raw hit counters
   ▼
target[k][h][v]  (raw counts over 60000 train samples)
   │  logit_convert: logit = log(p/(1-p)), p = (t+1)/(nk+2)
   ▼
target[k][h][v]  (logits × F, int32/int64)
   │  + compute_class_offset → offset[k]
   ▼
score[s][k] = offset[k] + Σ_{active (h,v)} target[k][h][v]
   │  export_one_member_ens → ens_write
   ▼
.ens file: header + scores[10000][10] + labels[10000]
   │  merge-ensemble: sum_scores[i] += sc[i]
   ▼
pred = argmax_k sum_scores[s][k]
```

---

## 2. Step by Step — the Exact Computation

### Step 1: Input → Container

The raw grayscale image (0-255 per pixel) is encoded and packed into 32-bit
containers (`lib/ki-load.h`):

```c
int pack = 32 / w, shift = w;          /* w = encoding width (8/16/32) */
uint32_t ev = enc_lut_get(typ, w, pv); /* thermometer: pv → bitmask   */
val |= ev << (k * shift);              /* pack pixels into container  */
```

- 8-bit: 4 px/container → 196 containers (MNIST, `H=196`)
- 16-bit: 2 px/container → 392 containers (`H=392`)
- The container content is the **activity bitmask** of the thermometer levels.

### Step 2: gb — the Activity Mask

`mlp-bin32-otto-trn-seq-prof.c:1938`:

```c
static inline uint32_t ki_gb_for_neuron(const uint32_t *in, const uint32_t *W0_row,
                                        int h, int w0_step, int NC_slice, int half) {
#ifdef KI_BITVOTING
    (void)W0_row; (void)w0_step; (void)NC_slice; (void)half;
    return in[h];                          /* identity: gb = input     */
#else
    return h0_to_gb(h0_neuron(in, W0_row + h * w0_step, NC_slice, half));
#endif
}
```

- **Bit-Voting:** `gb[s][h] = in[h]` — no W0, no majority, direct pixel bits.
- **Otto:** `h0 = majority(W0 × in)` (XNOR + majority tree), then
  `h0_to_gb` reduces it to the bit-mask used below.

### Step 2b: What `(h, v)` Means — One Single Bit in the H Vector

**`(h, v)` selects exactly ONE bit of the H0 output vector.** The H0 vector
is a flat bit-vector of `H × 32` bits:

```
H0 vector:  [Container 0 (bits 0-31)] [Container 1 (bits 0-31)] ... [Container H-1 (bits 0-31)]
              └── v = 0..31 ──┘
                          (h, v) = bit v of container h  →  a single bit
```

- `h` = container index (neuron of H0): `0..195` at 8-bit (`H=196`),
  `0..391` at 16-bit (`H=392`).
- `v` = bit position inside the 32-bit container (`0..31`), i.e. the bit
  position, NOT the bit value. With `splitVN=1` (`V = 32/1 = 32`) every bit
  is its own "virtual neuron", so `v` is the bit position directly. With
  `splitVN>1`, `G = splitVN` bits are grouped into one virtual neuron and
  `v` counts groups (`0..V-1`, `V = 32/G`) instead of single bits.
- `target[k][h][v]` is the hit counter / logit of **exactly that one bit**
  for class `k` — each bit is an independent per-bit classifier ("mini
  witness") that learned a log-odds contribution for every class.
- Index addressing (`TGT_IDX`, `seq-prof.c:53`):
  `idx = h·V·K + v·K + k` — the bit's flat address in the H0 vector × class.

Example at 8-bit `H=196`: the H0 vector has `196 × 32 = 6272` bits.
`(h=5, v=3)` = the 4th bit of the 6th container = one specific bit among
the 6272.

### Step 3: Target Counting

`ki_build_target_from_gb` (`seq-prof.c:647`) counts, over all **training**
samples, how often each single bit `(h, v)` of the H0 vector is active for
each class `k`:

```c
for (int s = 0; s < N; s++) {
    int k = Y[s];                       /* ground-truth label of sample s */
    for (int h = 0; h < H_local; h++) {
        uint32_t gbits = gb_row[h];     /* activity mask */
        while (gbits) {
            int v = __builtin_ctz(gbits);
            lt[TGT_IDX(k, h, v, H_local, V)]++;   /* hit counter */
            gbits &= gbits - 1;
        }
    }
}
```

Layout: `target[k][h][v]`, `V = 32 / splitVN` (with `splitVN=1` → `V=32`).

### Step 4: Logit Transformation

`logit_convert` (`seq-prof.c:802`) converts raw counts to log-odds:

```c
double p = (double)(t + 1) / (double)(nk + 2);          /* Laplace smoothing */
target[idx] = (COUNTER_TYPE)ot_precision(log(p / (1.0 - p)));
```

- **The log is the REAL `log()`** (natural logarithm, double precision from
  math.h) — **not** an approximation. No LUT, no series expansion.
- `nk` = number of training samples of class `k`.

#### Why `log(p / (1.0 - p))`? — the logit (log-odds)

`log(p / (1-p))` is the **logit** — the log of the *odds*
`p/(1-p) = P(hit)/P(no hit)`. It is the standard way to turn a Bernoulli
probability into an **additive score**. The reason is the score formula in
the setup header (`seq-prof.c:1153`):

```
Score:  Σ_h Σ_b [ y·log(P_k) + (1-y)·log(1-P_k) ]
```

This is the **Bernoulli log-likelihood**: each bit `(h,v)` is a Bernoulli
event (fires: yes/no) with class-conditional probability `P_k`, and `y` is
1 if the bit fired, 0 otherwise. Log-likelihoods are *additive* — the total
evidence for class `k` is the sum over all bits of their individual
`log(P)` (fired) or `log(1-P)` (not fired) contributions.

The logit is exactly the *difference* of those two terms:

```
logit = log(p/(1-p)) = log(p) − log(1-p)
```

So the score decomposes cleanly (Step 6 + Step 7):

| bit state | contribution to `score[k]`                               |
| --------- | -------------------------------------------------------- |
| fired     | `log(1-p)` (offset) + `log(p/(1-p))` (target) = `log(p)` |
| not fired | `log(1-p)` (offset only)                                 |

Summed over all bits this is precisely `Σ [ y·log(P) + (1-y)·log(1-P) ]` —
the Bernoulli log-likelihood. **Prediction stays integer-only:** `argmax`
over log-likelihoods equals `argmax` over probabilities (log is monotonic),
so no softmax/exp/division is ever needed at inference.

**Why not store `p` directly?** Because evidence must be *summed* across
hundreds of bits and (in the merge) across members. Probabilities would
have to be *multiplied* (product of experts) — logs turn multiplication
into addition, which is what `sum_scores[i] += sc[i]` does. The logit
(`log(p/(1-p))`) is the natural additive unit: it is symmetric around 0
(50/50 odds = 0), positive for "fires more often in class k", negative for
"fires less often".

#### Why `p = (t+1)/(nk+2)`? — Laplace smoothing and what it fixes

This is **Laplace smoothing** (add-one smoothing), the standard Bayesian
estimate of a Bernoulli probability. Two problems force it:

**Problem 1 — infinite logits.** Without smoothing the hit probability
would be `p = t/nk`, and `logit = log(p/(1-p))` would be:

| Case                                             | p   | logit           |
| ------------------------------------------------ | --- | --------------- |
| `t = 0` (bit `(h,v)` never active for class `k`) | `0` | `log(0/1) = -∞` |
| `t = nk` (bit always active for class `k`)       | `1` | `log(1/0) = +∞` |

Both happen **regularly**: with 6272 bits × 10 classes there are many
`(h,v)` slots that never (or always) fire for a given class. A `±∞` logit
would overflow the int32/int64 quantization (`ot_precision`, ×F) and corrupt
the whole target matrix.

**Problem 2 — the offset needs consistency.** The class offset
(`compute_class_offset`, Step 6) uses the complement `P(no hit)`:

```c
double p1 = (double)(nk - t + 1) / (double)(nk + 2);   /* P(no hit) */
```

The `+2` in the denominator is not arbitrary — it keeps the two estimates
complementary:

```
p + p1 = (t+1)/(nk+2) + (nk-t+1)/(nk+2) = (nk+2)/(nk+2) = 1   ✓
```

A naive `(t+1)/(nk+1)` would break this sum; clamping to `[ε, 1-ε]` would
be an arbitrary constant instead of a principled Bayesian estimate.

**The mathematics:** `p = (t+1)/(nk+2)` is the **MAP estimate of a
Bernoulli probability with a uniform Beta(1,1) prior** — the Bayesian
"most honest" choice. It pretends one pseudo-success and one
pseudo-failure exist in addition to the `nk` real observations. This keeps
`0 < p < 1` for all `t ∈ [0, nk]`, so `log(p/(1-p))` is always finite.

**In one sentence:** the smoothing (a) prevents `±∞` logits at `t=0` or
`t=nk` (numerical stability), (b) is the principled Bayesian estimate
(Beta(1,1) prior) rather than an arbitrary clamp, and (c) preserves the
`p + P(no hit) = 1` consistency that the score offset relies on. It is the
same idea as add-one smoothing in Naive Bayes text classification: never
assign a probability of exactly 0 or 1, always keep a small reserve of
"surprise".

### Step 5: OT_PRECISION Scaling (fixed-point quantization)

`ki-common.h:2233`:

```c
#define OT_F (1 << OT_PRECISION)            /* F = 2^17 = 131072 (default) */
static inline double ot_precision(double in) {
    return in * (double)OT_F + (in >= 0 ? 0.5 : -0.5);   /* ×F + rounding */
}
```

This is **not** a log approximation — it quantizes the (exact) log into an
integer so the target fits in `int32_t`/`int64_t`. The precision loss is the
fixed-point resolution `1/F ≈ 7.6e-6`, far below measurement noise.

### Step 6: Class Offset

`compute_class_offset` (`seq-prof.c:880`) is the "all neurons inactive"
baseline per class:

```c
double p1 = (double)(nk - t + 1) / (double)(nk + 2);   /* P(no hit) */
sum += (SCORE_TYPE)ot_precision(log(p1));               /* ×F, negative */
```

`offset[k] = Σ_{h,v} F · log(p1)` — the score contribution a class gets when
**no** bit `(h,v)` of the H0 vector is active.

**Note:** `p1 = (nk - t + 1)/(nk + 2)` is the Laplace-smoothed complement of
the hit probability — it uses the same `+2` denominator as Step 4 so that
`p + p1 = 1` exactly (see Step 4, "Why `p = (t+1)/(nk+2)`?").

### Step 7: The Score Itself

`scores_otto_from_gb` (`seq-prof.c:935`):

```c
for (int k = 0; k < KI_NCLASSES; k++)
    scores[k] = (SCORE_TYPE)class_offset[k];        /* baseline          */
for (int h = 0; h < H_local; h++)
    VN_SCORE_FROM_GB(gb_row[h], h, H_local, NG, target, scores);
```

with `VN_SCORE_FROM_GB` (`seq-prof.c:821`):

```c
uint32_t _b = (gb);
while (_b) { int _v = __builtin_ctz(_b);
    for (int _k = 0; _k < KI_NCLASSES; _k++)
        (SC)[_k] += (SCORE_TYPE)(TGT)[TGT_IDX(_k, h, _v, H, NG)];
    _b &= _b - 1; }
```

In words — every ACTIVE bit of the H0 vector adds its learned logit:

```
score[s][k] = offset[k]  +  Σ_{bits (h,v) active in gb[s][h]}  target[k][h][v]
```

Each bit `(h,v)` is an independent per-bit classifier: the score is the sum
of the log-odds contributions of all bits that fired for sample `s`.

Prediction: `pred = argmax_k score[s][k]`.

---

## 3. The .ens File Format

Written by `ens_write` (`lib/ki-ens.h`), one file per member.

### 3.1 Layout (v13 = int64)

```
magic(4) ver(4) n_test(4) n_classes(4) n_members(4)
hidden(4) epochs(4) split_vn(4) split_hn(4) target_err(4) seed(4)
v>=3:  timestamp i64(8)
v>=4:  ensemble_eval f32(4)
v>=10: w0_marker u32(4)
v>=11: maj_token char[8] + maj1_thresh i32(4)
v>=7:  per member: 4 length-prefixed strings (color, enc, enc-width, xform)
scores: n_members × (n_test × n_classes) elements
labels: n_test bytes
```

| Version | SCORE_TYPE | bytes/score | `ens_score_bytes` | Notes |
| ------- | ---------- | ----------- | ----------------- | ----- |
| v1-7    | int64      | 8           | 8                 | legacy |
| v8      | int32      | 4           | 4                 | legacy |
| v9-11   | float32    | 4           | 4                 | **drift** — int32 sum loses precision >1e9 |
| v12     | double     | 8           | 8                 | exact accumulation |
| v13     | int64      | 8           | 8                 | exact (MODE_INT32 default) |
| v14     | int64      | 8           | 8                 | **+ precision block** (ot_precision, bit_width, counter_type) |
| v15     | double     | 8           | 8                 | **+ precision block** |

The writer version follows the internal `SCORE_TYPE`
(`ens_version_for_score_type`: float→11, double→12, int64→13) — the export
preserves the computation precision (decision 2026-08-06).

**v14/v15 Precision Block (2026-08-12):**
Headers v14+ add a precision block: `ot_precision(4)`, `bit_width(4)`, `counter_type[24]`. This records how the stored logits were computed — the merge `--check` validates archives against these AND against the `.meta`. An archive must not mix float32/double/int64-built scores (float32 drifts on large sums → selection 29→44 members, 2026-08-06).

### 3.2 Size Computation (Fashion-MNIST, v13)

```
scores: 10000 × 10 × 8 B  = 800,000 B
labels: 10000 × 1 B       =  10,000 B
header:                    =      97 B   (72 B fields + 25 B member strings)
────────────────────────────────────────
total:                     = 810,097 B   ≈ 791 KB
```

Verified against `scores-H196-E10-BV8-INT32/vflip@spiral:mnist:raw8.ens`
(810,097 B) and the BV16 twin (810,098 B).

> **IMPORTANT:** The file contains **only the score table**, not the W1
> weights. The (double) bit mass of the model lives in RAM during training;
> the archive stores the *result* of the computation. Therefore a 16-bit
> corpus (2× W1 bit mass) produces **identically sized** files — same
> 10000×10 score table.

---

## 4. The Merge

`merge-ensemble.c:1446`:

```c
SCORE_TYPE *sum_scores = calloc(score_sz, sizeof(SCORE_TYPE));
for (int en = 1; en <= n_blocks; en++) {
    const SCORE_TYPE *sc = blocks[m].scores;
    for (size_t i = 0; i < score_sz; i++)
        sum_scores[i] += sc[i];              /* additive ensemble vote */

    #pragma omp parallel for reduction(+:correct)
    for (int s = 0; s < g_n_test; s++) {
        const SCORE_TYPE *row = sum_scores + s * row_sz;
        int pred = 0;
        for (int k = 1; k < g_n_classes; k++)
            if (row[k] > row[pred]) pred = k;
        if (pred == g_labels[s]) correct++;
    }
    float acc = 100.0f * correct / g_n_test;
    /* track best_acc / best_en, print EN curve */
}
```

- Members are added one at a time (greedy/beam order, see `--expansion-sort`,
  `--beam`, `--max`, `--min-gain`).
- The accuracy is measured against the **ground-truth labels** stored in the
  `.ens` files (`has_labels`). Without labels no selection is possible.
- `best_en`/`best_acc` = optimal subset size / accuracy.

---

## 5. Bit-Voting vs Otto — Differences in the Chain

| Step          | Bit-Voting                 | Otto                           |
| ------------- | -------------------------- | ------------------------------ |
| W0 projection | none (identity)            | `h0_neuron` + `h0_to_gb`       |
| Majority      | none ("direct bit vote")   | MAJ1/MAJ3 (+ maj1_thresh)      |
| gb            | `gb = in[h]`               | `gb = h0_to_gb(majority(...))` |
| Target init   | count → logit (TRN recipe) | same                           |
| Score         | same `scores_otto_from_gb` | same                           |

Everything downstream of `gb` (target counting, logit, offset, score, .ens)
is shared code.

---

## 6. FAQ / Common Pitfalls

**Q: Why are BV8 and BV16 .ens files the same size?**
A: The file stores the score table (10000×10), not the weights. The 2× W1
bit mass of BV16 exists only during training and is discarded after the
score computation.

**Q: Is the log approximated?**
A: No. `log()` is the real natural logarithm in double precision. Only the
*result* is quantized by `ot_precision` (×F + rounding) for int storage.

**Q: What is `s` in `sc[s][k]`?**
A: The test-sample index (0..9999). Row `s` = test image `s`, column `k` =
class 0..9. `sc[s][k]` = logit of image `s` for class `k`.

**Q: What is `(h, v)`?**
A: One single bit of the H0 vector — `h` = container index, `v` = bit
position inside the 32-bit container. See Step 2b.

**Q: Are the stored values probabilities?**
A: No — they are log-odds (logits). The prediction is `argmax_k`, not a
thresholded probability.

**Q: What does the merge optimize?**
A: Ensemble accuracy: it selects the member subset whose summed scores
maximize `argmax`-vs-truth correctness on the test set.

---

## 7. References

- Trainer: `otto-score-ifc/mnist/mlp-bin32-otto-trn-seq-prof.c`
  (shared: `seq.c`)
- Archive format: `otto-score-ifc/lib/ki-ens.h`
- Encoding/packing: `otto-score-ifc/lib/ki-load.h`
- Merge: `otto-score-ifc/mnist/merge-ensemble.c`
- Scaling: `otto-score-ifc/mnist/ki-common.h` (`OT_F`, `ot_precision`)
- Ensemble workflow: [ensemble.md](ensemble.md)
- Bit-voting baseline: [bitvoting-baseline.md](bitvoting-baseline.md)
