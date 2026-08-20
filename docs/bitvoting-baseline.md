# Bit-Voting Baseline — Technical Reference

> **Key statements, results and analysis:** see the public publication
> **https://forward-prop.nhi1.de/papers/bitvoting-baseline.html**
> (local source: `www/papers/bitvoting-baseline.html`)
>
> This file maintains **only the technical details** not belonging on the website:
> CLI usage, source files and references. The content sections (Motivation,
> Architecture, Results, Speed Analysis, Generalizer, Interpretation, Key Findings)
> live **exclusively** in the HTML — no duplication, no drift.

## Summary of Findings (see publication)

| Finding                        | Detail                                                                                                                   |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| W0 = free width, not nonlinearity | At **equal H** W0 ≈ BV (gap ~0.5pp); the earlier 4–9pp gap came from **unequal width** (unequal-width, outdated — CIFAR BV superseded) |
| **Equal H → on par**           | At **equal H** (H196) BV ≈ W0: MNIST 98.17 vs 98.75 (+0.58pp), Fashion 91.94 vs 92.47 (+0.53pp) → W0 strength is **not** a per-member effect |
| **W0 strength = free width**   | W0 projects to **arbitrary H** and scales (Fashion 92.47 → 92.51 → 92.80 → 92.96); BV with identity W0 is stuck at **H = I = 196** (no H knob) |
| Bit-density trade-off (TRN)    | Otto compresses gb to ~2.76 bits, BV keeps ~14.38 → BV only ~1.12× faster serial, ~equal parallel                        |
| **IFC inference**              | `maj1(W0×I)` is dominant and needed **1× per member** → **no cache** → BV **14.7× faster** (26ms vs 382ms) |

## Usage

> **Implementation note (2026-08-02):** Bit-Voting is no longer a separate
> tool. It is a compile-time variant of the shared Otto trainer
> (`mlp-bin32-otto-trn-seq-prof.c`) built with `-DKI_BITVOTING`: identity W0
> (gb = input), `splitVN=1` forced, `H = I` width. The binary names below
> are unchanged — the Makefile `bitvote` targets now build the seq-prof.c
> trainer with the flag. Target init is the standard TRN recipe (count →
> logit_convert). Results in section 4 of the HTML paper are historical
> (old tool, custom likelihood-ratio init) and were re-verified under TRN init.
> See: `plans/plan-2026-08-02-ki-bitvoting-flag.md`.
> Bit-Voting `.ens` archives carry `w0_marker = 0` (no W0 exists);
> merge-ensemble refuses to mix them with Otto archives.

### CLI

```bash
# MNIST — single member
./mnist-mlp-bin32-otto-trn-bitvoting.exe --epochsN 10 --encoding latest --xform id

# Multi-encoding ensemble
./mnist-mlp-bin32-otto-trn-bitvoting.exe --epochsN 10 --encoding exp,log --xform performance

# CIFAR-10
./cifar-mlp-bin32-otto-trn-bitvoting.exe --epochsN 10 --encoding latest --xform id

# Fashion-MNIST
./fashion-mlp-bin32-otto-trn-bitvoting.exe --epochsN 10 --encoding latest --xform id

# Debug per-member accuracy
./mnist-mlp-bin32-otto-trn-bitvoting.exe --epochsN 10 --encoding exp,log \
  --xform performance --debug-member --debug-epoch

# Filter weak members
./mnist-mlp-bin32-otto-trn-bitvoting.exe --epochsN 10 --encoding exp,log \
  --xform performance --member-threshold 90
```

### Options

| Flag                 | Default | Description                          |
| -------------------- | ------- | ------------------------------------ |
| `--epochsN`          | 1       | Training epochs                      |
| `--encoding`         | latest  | Encoding(s), comma-separated         |
| `--xform`            | id      | Transform(s), comma-separated        |
| `--batchN`           | 64/128  | Batch size for accumulation          |
| `--lr`               | 0.05    | Step scaling (step = lr × OT_F)      |
| `--debug-member`     | off     | Per-member accuracy + encoding/xform |
| `--debug-epoch`      | off     | Per-epoch step/train/eval            |
| `--member-threshold` | 0       | Filter members below N% train acc    |
| `--threadN`          | 8       | OpenMP threads                       |

## Source Files

| File                                              | Description                                                  |
| ------------------------------------------------- | ------------------------------------------------------------ |
| `otto-score-ifc/mnist/mlp-bin32-otto-trn-seq-prof.c` | Shared PRF trainer — Bit-Voting via `-DKI_BITVOTING` (canonical) |
| `otto-score-ifc/lib/ki-train.h`                   | Shared training loop (`ki_batch_correct`)                    |
| `otto-score-ifc/lib/ki-load.h`                    | Shared data loading                                          |
| ~~`out/archived-trn/mlp-bin32-otto-trn-bitvoting.c`~~ | Original standalone tool (archived 2026-08-02, **removed** — see note below) |
| `mnist-1/` (symlink)                              | MNIST build directory                                        |
| `cifar-1/` (symlink)                              | CIFAR-10 build directory                                     |
| `otto-score-ifc/fashion/` (symlink)               | Fashion-MNIST build directory                                |

## References

- Otto Score: `www/papers/otto-score.html` — Otto Score architecture paper
- Bit-Voting Baseline: `www/papers/bitvoting-baseline.html` — the publication
- MNIST paper: `www/papers/mnist-number.html` — Full MNIST results with xforms
- Status Report: `www/papers/status-2026-07.html` — Encoding, CIFAR-10 barrier
- Random Projection theory: Johnson-Lindenstrauss lemma, Reservoir Computing
