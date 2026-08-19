# Bit-Voting Baseline — Technical Reference

> **Kernaussagen, Ergebnisse und Analyse:** siehe die öffentliche Publikation
> **https://forward-prop.nhi1.de/papers/bitvoting-baseline.html**
> (lokale Quelle: `www/papers/bitvoting-baseline.html`)
>
> Diese Datei pflegt **nur die technischen Details**, die nicht auf die Website
> gehören: CLI-Nutzung, Source-Dateien und Referenzen. Die inhaltlichen
> Sektionen (Motivation, Architektur, Results, Speed-Analyse, Generalizer,
> Interpretation, Key Findings) leben **ausschließlich** im HTML — keine
> Doppelpflege, keine Drift.

## Kurzfassung der Erkenntnisse (Verweis auf die Publikation)

| Erkenntnis                     | Detail                                                                                                                   |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| W0 = freie Breite, nicht Nichtlinearität | Bei **gleichem H** ist W0 ≈ BV (Gap ~0.5pp); der frühere 4–9pp-Gap kam von **ungleicher Breite** (unequal-width, veraltet — CIFAR BV superseded) |
| **Gleiches H → ebenbürtig**    | Bei **gleichem H** (H196) ist BV ≈ W0: MNIST 98.17 vs 98.75 (+0.58pp), Fashion 91.94 vs 92.47 (+0.53pp) → W0-Stärke ist **kein** Per-Member-Effekt |
| **W0-Stärke = freie Breite**   | W0 projiziert auf **beliebiges H** und skaliert damit (Fashion 92.47 → 92.51 → 92.80 → 92.96); BV ist mit Identity-W0 an **H = I = 196 gefangen** (kein H-Knopf) |
| Bit-Dichte-Trade-off (TRN)     | Otto komprimiert gb auf ~2.76 Bits, BV behält ~14.38 → BV nur ~1.12× schneller serial, ~gleich parallel                  |
| **IFC-Inference**              | `maj1(W0×I)` ist dominant und wird **1× pro Member** gebraucht → **kein Cache** → BV **14.7× schneller** (26ms vs 382ms) |

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
- Bit-Voting Baseline: `www/papers/bitvoting-baseline.html` — die Publikation
- MNIST paper: `www/papers/mnist-number.html` — Full MNIST results with xforms
- Status Report: `www/papers/status-2026-07.html` — Encoding, CIFAR-10 barrier
- Random Projection theory: Johnson-Lindenstrauss lemma, Reservoir Computing
