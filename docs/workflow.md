# Otto Score Workflow — Result Generation & Validation

This document describes the complete workflow for generating and validating Otto Score results, using the **sweep-fashion** target (Fashion-MNIST with `--encoding`/`--channel`/`--xform` sweeps) as the canonical example.

---

## Overview

The workflow consists of **4 phases** that transform raw hyperparameter sweeps into validated, deployable models:

```
┌─────────────┐     ┌───────────────────┐     ┌─────────────┐     ┌─────────────┐
│  1. SWEEP   │────▶│ 2. MERGE-ENSEMBLE │────▶│ 3. EXPORT   │────▶│ 4. IMPORT   │
│  (--sweep)  │     │  (merge-ensemble) │     │  (--export) │     │  (--import) │
└─────────────┘     └───────────────────┘     └─────────────┘     └─────────────┘
  Generate .ens     Select best members      Create model.otto      Validate on
  archives          per setup                from member file       new data
```

---

## Phase 1: SWEEP — `--sweep` (Generate .ens Archives)

### Purpose
Generate a **corpus of score archives** (`.ens` files) by training all combinations of `--encoding`, `--channel`, `--xform` (and optionally `--ensembleN`, `--splitVN`, `--splitHN`).

### Command (sweep-fashion)
```bash
# From mnist-fashion/
./fashion-mlp-bin32-otto-trn-xnor-8-int32.exe \
  --hiddenN 256 --epochsN 10 \
  --encoding sweep --channel sweep --xform sweep \
  --sweep --export-merge-scores scores-H256-E10-M1-105-INT32
```

### What Happens
1. **Cartesian product** of all active `--encoding`, `--channel`, `--xform` values (defined in `ki-local.h` + CLI)
2. **Per-member training**: Each combination trains independently with its own W0, target, offset
3. **Per-member .ens export**: After training, each member's scores are written to a single `.ens` archive in the output directory
4. **Directory structure**:
   ```
   scores-H256-E10-M1-105-INT32/
   ├── .meta                    # Config identity (H, EP, VN, HN, MAJ, MAJ1_THRESH, BITS, OT_PRECISION, EXE)
   ├── .index                   # Binary cache of all .ens metadata (fast reload)
   ├── rot0@id:mnist:exp8.ens   # One .ens per member (XF:CHAN:ENC naming)
   ├── rot90@avg4:mnist:log8.ens
   └── ...
   ```

### Parallelism (`--threadN`)
- **PRF mode** (default in `mlp-bin32-otto-trn-seq-prof.c`): **N members in parallel, each with 1 thread**
- `--threadN N` = parallel member width (num threads for OUTER member loop)
- Default: all HW threads (`omp_get_max_threads()`)
- Inner training parallelism capped at 8 threads (batch=64 scaling limit)
- **Speed**: ~221 members (17 xforms × 13 encodings) in ~1 minute on 16 threads

### Output
- **`.ens` archives**: Per-member scores + labels + metadata (v7+ format: channel/enc/width/xform strings)
- **`.meta`**: Config identity for validation on reload
- **`.index`**: Binary cache of all .ens metadata (name, ctime, size, per-member eval) — enables incremental rebuild
- **`.storebackup_dont_backup`**: Marker to exclude from backups (regenerable)

---

## Phase 2: MERGE-ENSEMBLE — Select Best Members

### Purpose
From a corpus of `.ens` archives, **select the optimal subset of members** that maximizes ensemble accuracy for the given setup.

### Tool
`otto-score-ifc/mnist/merge-ensemble.c` → builds `fashion-merge-ensemble-int32.exe` (or `mnist-merge-ensemble.exe`, `cifar-merge-ensemble.exe`)

### Core Algorithm
1. **Load directory**: Read all `.ens` files (or cached `.index`)
2. **Validate config**: Check `.meta` identity (H, EP, VN, HN, MAJ, MAJ1_THRESH, BITS, OT_PRECISION, COUNTER_TYPE, SCORE_TYPE, EXE)
3. **Filter**: Apply `--filter` (eval threshold, regex, class subset)
3. **Search**: Beam search / Greedy / Tries to find optimal member subset
4. **Output**: `--member-out` or `--member-out-default` file with selected member specs

### Key Search Modes

| Mode                | Flag                        | Description                                                                                            |
| ------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------ |
| **Beam**            | `--beam N`                  | Keep top N candidates per level (add-only attractor)                                                   |
| **Beam + Max**      | `--beam N --max`            | Cumulative acceptance: buffer sub-threshold members, commit block when accumulated gain ≥ `--min-gain` |
| **Greedy**          | `--greedy`                  | Single-path optimal subset (O(N²))                                                                     |
| **Tries**           | `--tries N`                 | Shuffle restarts, lock found members (`pool_exclude`)                                                  |
| **Tries + No Lock** | `--tries N --tries-no-lock` | Each try gets fresh pool → escapes greedy attractors                                                   |
| **BestN**           | `--beam-bestN K`            | Start selection at K-th best candidate per level (breaks 1st-best fixation)                            |
| **Optimal**         | `--optimal`                 | 2-opt exchange (removal allowed) on beam result                                                        |

### Filtering Options

| Filter         | Syntax                      | Effect                                  |
| -------------- | --------------------------- | --------------------------------------- | ------------------------------------------ |
| Eval threshold | `--filter eval gt 58.1`     | Keep members with eval > 58.1%          |
| Percentile     | `--filter eval gt 50%`      | Keep top 50% of members by eval         |
| Regex INCLUDE  | `--filter regex ':(cbrt8\   | exp8)$'`                                | Keep members matching regex on XF:CHAN:ENC |
| Regex EXCLUDE  | `--filter regex '!colswap'` | Drop members matching regex             |
| Class target   | `--target 3,7`              | Optimize for recall on classes 3,7 only |
| Sample filter  | `--filter-sample 3,7`       | Only evaluate samples of classes 3,7    |

### Output: `--member-out` / `--member-out-default`

**`--member-out member.out`** — Explicit path
**`--member-out-default`** — Auto-derive `member-{DIR}.out` from corpus dir name

**Format** (one member per line):
```
avg4@spiral:mnist:log8
rot90@avg4:mnist:exp8
colswap-1-4@avg4:mnist:gamma8
...
```
Each line = `XF:CHAN:ENC` spec (matches trainer's member spec format)

### Verification: `--check`
```bash
./fashion-merge-ensemble-int32.exe scores-H256-E10-M1-105-INT32 --check
```
- Validates all `.ens` headers against `.meta`
- Recomputes true member eval (scores vs labels)
- Detects broken header evals (|header-computed| > 1.0%)
- Writes corrected specs to `--member-out` for retraining

---

## Phase 3: EXPORT — Create Deployable Model (`model.otto`)

### Purpose
Take the selected member list (`--member-file`) and **export a self-contained `model.otto`** for inference.

### Command
```bash
# From mnist-fashion/
./fashion-mlp-bin32-otto-trn-xnor-8-int32.exe \
  --member-file member-H256-E10-OT8-M1-105-INT32.out \
  --export export-H256-E10-OT8-M1-105-INT32
```

### What Happens (in `mlp-bin32-otto-trn-seq-prof.c`)

1. **Parse `--member-file`**: Load member specs (XF:CHAN:ENC per line)
2. **Reconstruct members**: For each spec, rebuild:
   - W0 slice (from saved W0 marker or regenerated from seed)
   - Target + Offset (from .ens archives)
   - Input buffer (CEX: load_input_cex_cached for each member's XF:CHAN:ENC)
3. **Verify eval parity**: Run evaluation on test set → **must match merge-ensemble eval%**
   - If mismatch: **abort** (member file / corpus mismatch)
4. **Export `model.otto`** (v8 format):
   - Header: magic, version=8, H0_MODE, H, NC, OT_PRECISION, MAJ_MODE, MAJ1_THRESH, SPLIT_VN, SPLIT_HN
   - Per-member: W0[H_local × NC_slice], Target[H_local × K × V], Offset[K]
   - Per-member metadata: channel, encoding, width, xform strings (v7+)

### Export Verification (Critical)
```bash
# The trainer re-evaluates each member and the ensemble during export
# Output shows:
[  1/9] ens=88.34%  mem=82.1%  avg4@spiral:mnist:log8
[  2/9] ens=90.12%  mem=79.8%  rot90@avg4:mnist:exp8
...
[  9/9] ens=91.78%  mem=81.4%  colswap-1-4@avg4:mnist:gamma8

# Final ensemble eval MUST match merge-ensemble result
REPORT train=99.43% eval=91.78% ...
```

**If eval differs**: Member file references wrong corpus, or archives corrupted → fix before deploying.

---

## Phase 4: IMPORT — Inference on New Data

### Purpose
Load `model.otto` and run **pure inference** (no training) on test/eval data.

### Command
```bash
./fashion-mlp-bin32-otto-trn-xnor-8-int32.exe \
  --import export-H256-E10-OT8-M1-105-INT32 \
  --evalN 10000 --encoding latest
```

### What Happens (Same Binary, Different Mode)

1. **Load `model.otto`** (v8 header + per-member metadata)
   - Overrides `aa.*` with model's MAJ_MODE, MAJ1_THRESH, SPLIT_VN, SPLIT_HN
   - Reads per-member W0, Target, Offset
   - Reads per-member channel/encoding/width/xform strings
2. **Rebuild input pipeline**: For each member, init encoding LUT + load CEX input buffer
   - `load_input_cex_cached()` per member's XF:CHAN:ENC
3. **Run inference**: `ki_evaluate_member()` with `use_gb=0` (raw pixel path)
   - No training loop, no target building, no correction
   - Scores summed across members → argmax → accuracy
4. **Output**: `REPORT` line with eval%, optional `--predictions` file

### Key Differences from Training

| Aspect          | Training                                 | Import (Inference)                           |
| --------------- | ---------------------------------------- | -------------------------------------------- |
| Mode            | `--sweep` / `--member-file` + `--export` | `--import DIR`                               |
| Target building | Yes (count VN firings)                   | No (loaded from model)                       |
| Correction loop | Yes (iterative)                          | No                                           |
| Epochs          | `--epochsN N`                            | N/A (single pass)                            |
| W0              | Generated (random/seeded)                | Loaded from model                            |
| Target/Offset   | Built + trained                          | Loaded from model                            |
| Speed           | Minutes                                  | **Milliseconds** (IFC: 382ms for 55 members) |

### Dry-Run (Architecture Preview)
```bash
./fashion-mlp-bin32-otto-trn-xnor-8-int32.exe --import export-... --dry-run
```
Shows: model version, H, NC, ensemble size, maj config, per-member metadata — **no pixel data loaded**.

---

## Complete Example: sweep-fashion → Validated Model

```bash
# 1. SWEEP: Generate corpus (221 members: 17 xforms × 13 encodings)
cd mnist-fashion
./fashion-mlp-bin32-otto-trn-xnor-8-int32.exe \
  --hiddenN 256 --epochsN 10 \
  --encoding sweep --channel sweep --xform sweep \
  --sweep --export-merge-scores scores-H256-E10-M1-105-INT32

# 2. MERGE: Find best 12 members (beam 10, cumulative acceptance)
./fashion-merge-ensemble-int32.exe scores-H256-E10-M1-105-INT32 \
  --beam 10 --max --min-gain 0.1 \
  --member-out-default

# → Creates member-H256-E10-OT8-M1-105-INT32.out (12 lines)

# 3. EXPORT: Create model.otto + verify eval parity
./fashion-mlp-bin32-otto-trn-xnor-8-int32.exe \
  --member-file member-H256-E10-OT8-M1-105-INT32.out \
  --export export-H256-E10-OT8-M1-105-INT32

# 4. IMPORT: Validate on test set (no training)
./fashion-mlp-bin32-otto-trn-xnor-8-int32.exe \
  --import export-H256-E10-OT8-M1-105-INT32 \
  --evalN 10000 --encoding latest

# Optional: Dry-run for metadata only
./fashion-mlp-bin32-otto-trn-xnor-8-int32.exe \
  --import export-H256-E10-OT8-M1-105-INT32 --dry-run
```

---

## Validation Checklist

| Step       | Check               | Pass Criteria                                                                   |
| ---------- | ------------------- | ------------------------------------------------------------------------------- |
| **Sweep**  | All .ens created    | `.meta` + `.index` + N × `.ens` (N = xforms × channels × encodings × ensembleN) |
| **Merge**  | Member file created | `--member-out` has K lines, each `XF:CHAN:ENC`                                  |
| **Merge**  | Config valid        | `.meta` matches CLI (H, EP, VN, HN, MAJ, MAJ1_THRESH, BITS, OT_PRECISION)       |
| **Export** | Eval parity         | Export-time ensemble eval == Merge-ensemble eval (±0.05pp)                      |
| **Export** | Model version       | `model.otto` v8 (self-describing: maj_mode, maj1_thresh, splitVN, splitHN)      |
| **Import** | Eval match          | Import eval == Export eval (±0.05pp)                                            |
| **Import** | Speed               | IFC < 1s for 10K samples (Otto: ~382ms, BV: ~26ms)                              |

---

## Key Files Reference

| File                                                 | Role                                                                 |
| ---------------------------------------------------- | -------------------------------------------------------------------- |
| `otto-score-ifc/mnist/mlp-bin32-otto-trn-seq-prof.c` | PRF trainer (sweep, export, import, --member-file)                   |
| `otto-score-ifc/mnist/merge-ensemble.c`              | Corpus merge + member selection                                      |
| `otto-score-ifc/lib/ki-ens.h`                        | .ens format (v1-15): read/write/verify                               |
| `otto-score-ifc/mnist/ki-common.h`                   | CLI parsing (`ki_parse_args`), `--member-file` logic                 |
| `otto-score-ifc/mnist/ki-config.h`                   | Config constants (KI_NC, KI_NCLASSES, defaults)                      |
| `mnist-fashion/ki-local.h`                           | Fashion dataset config (class names, default encoding/xform/channel) |

---

## Common Pitfalls

| Issue                        | Cause                                        | Fix                                                                          |
| ---------------------------- | -------------------------------------------- | ---------------------------------------------------------------------------- |
| `config mismatch (H=64/784)` | `.meta` H differs from CLI                   | Ensure `--hiddenN` matches sweep; Bit-Voting repurposes H as container count |
| `maj mismatch (maj=1/3)`     | Trainer `--maj` differs from corpus          | Use same `--maj` / `--maj1-thresh` for sweep and merge                       |
| `SCORE_TYPE mismatch`        | float32 vs int64/double accumulation         | Use same `SCORE_TYPE` (default double) for sweep and merge                   |
| `EXE mismatch`               | Different trainer binary re-populated scores | Rebuild same binary, or use `--force` (with caution)                         |
| Export eval ≠ Merge eval     | Member file references wrong corpus          | Verify `--member-file` points to correct `scores-*` dir                      |
| Import 36.6% (random)        | Missing `--maj1-thresh` on import            | Model v8 carries maj config; v5-v7 need CLI override                         |

---

## Version Compatibility

| .ens Version | Score Type         | Read Support | Write Support   |
| ------------ | ------------------ | ------------ | --------------- |
| v1-v7        | int64              | ✅           | ❌ (legacy)     |
| v8           | int32              | ✅           | ❌ (legacy)     |
| v9-v11       | float32            | ✅           | ❌ (drift)      |
| **v12**      | **double**         | ✅           | ✅ (default)    |
| **v13**      | **int64**          | ✅           | ✅ (MODE_INT32) |
| v14          | int64 + precision  | ✅           | ✅              |
| v15          | double + precision | ✅           | ✅              |

**Default write**: v12 (double) for float builds, v13 (int64) for MODE_INT32.  
**Import validates**: `.meta` `COUNTER_TYPE`/`SCORE_TYPE` + `EXE` + `BITS` + `OT_PRECISION`.

---

## Performance Notes

- **Sweep parallelism**: `--threadN` controls member-level parallelism (outer loop)
- **Merge speed**: `.index` cache avoids re-reading .ens files (stat-based incremental rebuild)
- **Export verification**: Re-evaluates each member — adds ~30s for 50 members (once per export)
- **IFC inference**: ~19× faster for Bit-Voting (no W0 projection) vs Otto Score
- **gb-cache**: `--export-gb` / `--import-gb` caches H0→VN mask to `data/gb/` (35% faster on 2nd run)

---

*Generated from code analysis of `mlp-bin32-otto-trn-seq-prof.c`, `merge-ensemble.c`, `ki-ens.h`, `ki-common.h` (2026-08-19).*
