# Otto Score — DRAM-Native MLP Classifier

**MNIST: 98.4% in 14s | CIFAR-10: 69.99% (ensemble beam-search) | Fashion-MNIST: 91.58%** — Zero floating point, zero matmul in inference.
Only `&|~` + int32 + popcount. Also includes float32 AdamW + multi-member Hebbian + **Bit-Voting linear baseline** (proves the W0 nonlinearity gap is dataset-dependent).

> 📖 **Full status report:** [`www/papers/status-2026-07.html`](www/papers/status-2026-07.html) — Latest findings: `--export-gb` cache, `--maj 1` default, `avg2/3/4` filters, `@` pipe chaining, ensemble beam-search breaks 65% ceiling.

---

## 🚀 Quick Start

```bash
# Step 1: Build everything (9 binaries)
make all

# Step 2: Download datasets (MNIST + CIFAR-10 + Fashion-MNIST)
make setup

# Step 3: Train models + run tests (first run ~5min, then cached <1s)
make test
```

**Expected output (cached models):**

```
=== Otto Score MNIST (H=512, 10 ep, latest)       ===      eval=97.0%
=== Float32 AdamW MNIST (H=512, 10 ep)            ===      eval=92.5%
=== Bin32 Hebbian MNIST (H=512, 10 ep, latest)    ===      eval=84.3%
=== Otto Score CIFAR-10 (H=256, 5 ep, latest)     ===      eval=58.0%
=== Float32 AdamW CIFAR-10 (H=256, 5 ep)          ===      eval=44.3%
=== Bin32 Hebbian CIFAR-10 (H=256, 5 ep, latest)  ===      eval=31.0%
=== Otto Score Fashion (H=512, 10 ep, latest)     ===      eval=87.5%
=== Float32 AdamW Fashion (H=512, 10 ep)          ===      eval=82.1%
=== Bin32 Hebbian Fashion (H=512, 10 ep, latest)  ===      eval=71.0%
```

First run trains all 12 models (~5min). Subsequent runs use cached models (<1s total).

**`make models` — Train fresh models (cached):**

```
$ make models
make -s -C mnist model-otto
REPORT train=100.0% (60000) eval=97.1% (10000) ...
make -s -C mnist model-adam
REPORT train=92.5% (60000) eval=92.5% (10000) ...
make -s -C mnist model-hebbian
REPORT train=83.2% (60000) eval=84.3% (10000) ...
make -s -C cifar model-otto
REPORT train=78.5% (50000) eval=58.0% (10000) ...
make -s -C cifar model-adam
REPORT train=44.3% (50000) eval=44.3% (10000) ...
make -s -C cifar model-hebbian
REPORT train=30.6% (50000) eval=31.0% (10000) ...
make -s -C fashion model-otto
REPORT train=99.2% (60000) eval=87.5% (10000) ...
make -s -C fashion model-adam
REPORT train=82.1% (60000) eval=82.1% (10000) ...
make -s -C fashion model-hebbian
REPORT train=71.8% (60000) eval=71.0% (10000) ...
```

---

## 🎯 Test Targets

| Command                     | What it tests                          |
| --------------------------- | -------------------------------------- |
| `make test-mnist`           | All 4 MNIST approaches                 |
| `make test-mnist-otto`      | Otto Score MNIST only                  |
| `make test-mnist-adam`      | Float32 AdamW MNIST only               |
| `make test-mnist-hebbian`   | Bin32 Hebbian MNIST only               |
| `make test-mnist-bitvoting` | Bit-Voting MNIST only                  |
| `make test-cifar`           | All 4 CIFAR-10 approaches              |
| `make test-cifar-otto`      | Otto Score CIFAR-10 only               |
| `make test-cifar-adam`      | Float32 AdamW CIFAR-10 only            |
| `make test-cifar-hebbian`   | Bin32 Hebbian CIFAR-10 only            |
| `make test-cifar-bitvoting` | Bit-Voting CIFAR-10 only               |
| `make test-fashion`         | All 4 Fashion-MNIST approaches         |
| `make test-fashion-otto`    | Otto Score Fashion-MNIST only          |
| `make test-fashion-adam`    | Float32 AdamW Fashion-MNIST only       |
| `make test-fashion-hebbian` | Bin32 Hebbian Fashion-MNIST only       |
| `make test-fashion-bitvoting` | Bit-Voting Fashion-MNIST only        |

First run trains all 12 models. Subsequent runs use cached models (<1s total).

---

## 🔬 Key Findings (August 2026)

### Ensemble Beam-Search Breaks CIFAR-10 65% Ceiling → **69.99%**
**New best: 69.99%** — beam-search over **19,707 stored score archives** (`scores-H256-E10-M3`, maj=3, H=256, EP=10). The old ceiling was a *training* limit; the stored scores contain far more, reachable via smarter ensemble selection.

| Corpus | Archives | eval | Members |
| ------ | -------- | ---- | ------- |
| M1 (maj1, INT32) | 10,588 | 68.62% | 36 |
| M3 (maj3, FLT32) | 10,561 | 69.36% | 109 |
| **M3 + M3.add** | **19,707** | **69.99%** | **116** |

### Float32 Drift Fixed — Exact Accumulation (int64/double)
The float32 score accumulator drifts once ensemble scores reach ~1e9 (> 2²⁴). Resolution: **DRAM chip accumulates popcounts exactly in integer arithmetic** — the exact behavior is chip-faithful.

- `SCORE_TYPE` defaults to **double** (`-DSCORE_TYPE=float` reproduces legacy)
- `.ens` archive follows internal format: **v12 (double) / v13 (int64)**
- `.meta` records/validates `COUNTER_TYPE=` / `SCORE_TYPE=`
- Same corpus: float32 → 29 members (66.79%) vs int64/double → 44 members (67.72%)

### Majority Mode: `maj=1` is Now Default
The old `--maj 3` tree was identified as `--maj 1` with effective threshold ~52.7% (135/256). `--maj 1` is DRAM-native (pure bit-logic, no tree overhead). Threshold via `--maj1-thresh`:
- `-2` (default): auto per encoding (lookup: 256→135, 512→269, 1024→540)
- `-1`: n/2 (exact 50%)
- `>=0`: exact value

### `--max` Cumulative Acceptance (New)
`--max` alone activates **cumulative acceptance** — block commit when accumulated gain crosses `--min-gain`. Accepts sub-threshold members, buffers them, commits when sum of gains crosses min-gain. Window stop counts every step since last commit.

### `--sample-index` Parquet Export (Optional)
`--sample-index FILE` writes member↔sample recognition pairs as 3 Parquet files (samples/members/pairs). Optional build flag: `make USE_CARQUET=1`. Queries via DuckDB `read_parquet()`.

### IFC v7 — Self-Describing Model Format
`model.otto` v7 adds per-member metadata (channel, encoding, width, xform strings). Import initializes enc-LUT from model metadata. **Merge = Retrain = IFC** (exact roundtrip).

### Bit-Voting Gap is Dataset-Dependent
On CIFAR W0 buys +7pp; on Fashion it buys only +0.35pp at +61% bit-mass. The projection's value scales with input entropy, not with H.

### MNIST New Record: **98.4%** (Pipe Xforms)
**Best MNIST: 98.4%** — H=196, 9 members, 10 epochs, 14.2s. Key: xform pipe chains (`@` syntax): `avg4@spiral` = spiral then blur. `avg4@avg4` (double running average) produces heavily smoothed image where MAJ3 generalizes better from overall digit shape.

### Fashion-MNIST: 91.58% Ceiling (Architectural Trap)
Every search strategy lands in 88.7–91.6% basin (~0.1pp spread = noise). The 91% trap is architectural (frozen random W0 + maj1 voting, no convolution). **Speed argument**: PRF trainer does 221-member sweep in ~1 minute — orders of magnitude faster than CNN training.

---

## 🎯 Best Results (Latest)

| Configuration | Dataset | Accuracy | Time |
| --- | --- | --- | --- |
| H=196, EN=1, ep=10, pipe xforms | **MNIST** | **98.4%** | **14s** |
| H=256, EP=10, maj=3, beam-search 19.7k archives | **CIFAR-10** | **69.99%** | merge-ensemble |
| H=512, ep=10, `--maj 1`, perf encoding, `--encoding-sizeN 16` | CIFAR-10 | 64.7% | 296s (sweet spot single-config) |
| H=512, ep=10, m1t=105 | **Fashion-MNIST** | **91.58%** | ~1 min sweep |

---

## 🧪 Test Targets

| Command | What it tests |
| --- | --- |
| `make test-mnist` | All 4 MNIST approaches |
| `make test-cifar` | All 4 CIFAR-10 approaches |
| `make test-fashion` | All 4 Fashion-MNIST approaches |
| `make test` | All 12 models (cached <1s) |

---

## 🔧 Build Targets

| Command | Builds |
| --- | --- |
| `make` / `all` | All 9 binaries (Otto + Hebbian + Adam × XNOR/XOR, all datasets) |
| `make otto` | Otto Score only (mnist/ + cifar/ + fashion/) |
| `make hebbian` | Hebbian only |
| `make adam` | Float32 AdamW only |
| `make models` | Train all 9 models (cached) |
| `make clean` | Remove executables |
| `make ensemble` | Build merge-ensemble only |
| `make bitvote` | Build Bit-Voting baseline (linear, no W0) |

---

## 📋 CLI Flags (Unified Across All Trainers)

| Flag | Description | Default |
| --- | --- | --- |
| `--hiddenN N` | Hidden neurons | 64 |
| `--epochsN N` | Training epochs | 1 |
| `--splitVN N` | Bit-grouping: 1=50%, **2=25% (CIFAR sweet spot)**, 3=12.5%, ... | 1 |
| `--encoding TYPE` | Input encoding: exp, sig, up8, down8, raw, `latest` = dataset alias | raw8 |
| `--ensembleN N` | Independent W0 copies | 1 |
| `--export DIR` | Model export directory | none |
| `--import DIR` | Load model for inference | none |
| `--export-merge-scores DIR` | Save per-member scores to archive files | none |
| `--export-gb` | Cache gb_buf to data/gb/ (≥2nd run 35% faster) | off |
| `--sample-index FILE` | Write member↔sample pairs as Parquet (opt-in `USE_CARQUET=1`) | off |
| `--predictions FILE` | Export per-sample predictions (for vis-errors) | none |
| `--dry-run` | Print architecture and exit (metadata only) | off |
| `--seed N` | Random seed | 42 |
| `--seed-member MODE` | Member seed strategy (once, const, incr) | once |
| `--maj1-thresh N` | Majority threshold: `-2`=auto, `-1`=n/2, `>=0`=exact | -2 |
| `--max N` | Cumulative acceptance (block commit when gain sum crosses min-gain) | off |
| `--min-gain F` | Minimum cumulative gain for block commit | 0.01 |
| `--tries N` | Shuffle restarts (escapes greedy attractors) | 1 |
| `--tries-no-lock` | Each try gets fresh pool (full-pool search) | off |
| `--beam-bestN N` | Start beam selection at N-th best candidate per level | 1 |
| `--filter eval gt N%` | Percentile filter (top N% of member evals) | off |
| `--filter regex 'PATTERN'` | POSIX regex INCLUDE on full XF:CHAN:ENC spec | off |
| `--threadN N` | OpenMP threads | auto |

---

## 🏗️ Architecture

```
otto-score-ifc/
├── Makefile              ← orchestrates: delegates to subdirectories
├── mnist/                ← MNIST sources (Otto + Hebbian + Adam, unified)
│   ├── Makefile
│   ├── mlp-bin32-otto-trn-seq-prof.c  ← PRF trainer (FINAL: N members × 1 thread)
│   ├── mlp-bin32-otto-trn-seq.c       ← Archived sequential (A/B testing)
│   ├── mlp-bin32-hebbian-trn.c        ← Hebbian (shared source)
│   ├── mlp-flt32-adam-trn.c           ← Float32 AdamW (unified, --import inference)
│   ├── ki-common.h                    ← Shared header (args, parsing, RNG)
│   ├── ki-local.h                     ← MNIST dataset config
│   └── merge-ensemble.c               ← Ensemble merge tool
├── cifar/                ← CIFAR-10 (symlinks to mnist/ + CIFAR ki-local.h)
│   ├── Makefile
│   ├── mlp-bin32-otto-trn.c      → ../mnist/...
│   ├── ki-local.h                ← CIFAR dataset config
├── fashion/              ← Fashion-MNIST (symlinks to mnist/ + Fashion ki-local.h)
│   ├── Makefile
│   ├── ki-local.h                ← Fashion-MNIST dataset config
├── lib/                   ← Shared headers (ki-adamw.h, ki-encoding.h, maj1.h, w0_random.h)
├── models/                ← Cached trained models
├── bin/                   ← Helper scripts (run-ensemble.sh, run-research.sh)
├── www/                   ← Publication site (HTML papers, datasets)
├── fetch_mnist.sh         ← MNIST download
├── fetch_cifar10.sh       ← CIFAR-10 download
├── fetch_fashion.sh       ← Fashion-MNIST download
└── README.md              ← this file
```

All 3 trainers (Otto, Hebbian, Adam) are **unified** across MNIST and CIFAR:
the same `.c` file is compiled with different `ki-local.h` per dataset.
Each trainer doubles as inference engine via `--import`. Zero code drift.

---

## 🔄 Ensemble Workflow — Accumulate Seeds Over Time

Every Otto Score ensemble member is **completely independent** — different W0, trained from scratch, own target matrix. Members computed "on demand" and accumulated over time. The merge tool combines any subset to find optimal ensemble size.

### 1. Train with `--export-merge-scores DIR`
```bash
# Manual (one seed per call)
./cifar/cifar-mlp-bin32-otto-trn-xnor.exe --hiddenN 512 --epochsN 7 \
  --encoding latest --splitVN 2 --export-merge-scores scores/ --seed 1234

# Auto-seed: run-ensemble.sh picks unused random seed
bash bin/run-ensemble.sh ./cifar/cifar-mlp-bin32-otto-trn-xnor.exe \
  --hiddenN 512 --epochsN 7 --encoding latest --splitVN 2

# Repeat N times (each gets unique seed)
bash bin/run-ensemble.sh --repeat 20 ./cifar/cifar-mlp-bin32-otto-trn-xnor.exe \
  --hiddenN 512 --epochsN 7 --encoding latest --splitVN 2
```
Each run writes one `.ens` archive: `H512_EP7_VN2_HN1_TE40_SD1234_F4_TS1712345678.ens`

### 2. Merge all seeds to EN curve
```bash
# Build merge-ensemble (public)
make -C cifar ensemble

# Merge all archives in scores/ — shows per-member accuracy gain
./cifar/cifar-merge-ensemble.exe scores/

# Filter: only members from archives with eval > 58%
./cifar/cifar-merge-ensemble.exe scores/ --filter eval gt 58

# Limit to first 20 members
./cifar/cifar-merge-ensemble.exe scores/ --num 20

# Save curve data (for plotting)
./cifar/cifar-merge-ensemble.exe scores/ --save /tmp/curve.dat
```

---

## 📦 Inference via `--import`

Every training binary doubles as inference engine. No separate IFC binaries:

```bash
# MNIST Otto Score (latest = exp8, single member)
./mnist/mnist-mlp-bin32-otto-trn-xnor.exe \
  --import models/mnist-otto-h512-e10 --evalN 10000 --encoding latest

# CIFAR-10 Otto Score (--encoding latest = 11 members)
./cifar/cifar-mlp-bin32-otto-trn-xnor.exe \
  --import models/cifar-otto-h256-e5 --evalN 10000 --encoding latest

# Fashion-MNIST Otto Score
./fashion/fashion-mlp-bin32-otto-trn-xnor-8-int32.exe \
  --import models/fashion-otto-h512-e10 --evalN 10000 --encoding latest
```

---

## 🧰 Advanced Features

### `--export-gb` — gb_buf Cache
Caches H0→VN group mask to `data/gb/`. Hash key covers dataset, W0[0..3], dimensions, input pipeline. Second run 35% faster. `--debug-cache` for cache logs.

### Xform Pipeline Chaining `@`
```bash
# One member: rot90, then avg4 on result
--xform rot90@avg4

# Three members: avg4, rot45+avg4, rot90+avg4
--xform avg4,rot45@avg4,rot90@avg4

# Any number of steps
--xform rot45@avg2@avg4
```

### Row-Wise Average Filters (`avg2`, `avg3`, `avg4`)
Running average per row with wrap-right. Spreads pixel information across neighbors. Combined with `@` chaining: `rot45@avg4` = rotate then blur.

### `--encoding-sizeN 0` — RAW Encoding
No thermometer encoding — raw 8-bit pixel values directly into containers (4 px/uint32). Train reaches 99.6% (Otto), 96.8% (BitVoting). BitVoting shows **no ceiling** (eval still climbing at 47.4%, 60 members).

### VN=2 Sweet Spot (CIFAR)
`--splitVN 2` = AND2 filter, 25% retention. Optimal for noisy data (CIFAR). For clean data (MNIST), VN=1 is better. VN=3+ (strict AND) only at very large H.

---

## 📊 Container Principle — Same Bit-Mass, Same Outcome

Every neuron is a container of 32 yes/no decisions (bits). Float32 and int32 are just different container formats — the information capacity is identical.

| Format | Container | Bits/Neuron | Forward | Training |
| --- | --- | --- | --- | --- |
| flt32 | 1 float32 | 32 | `W @ x` (matmul) | SGD / AdamW |
| int32 | 1 int32 | 32 | `W · x` (int matmul) | SGD (float) |

**Key insight:** The DRAM chip accumulates popcounts exactly in integer arithmetic. Float32 drift was an emulation artifact. The `.ens` archive follows the internal format (v12 double / v13 int64).

---

## 📚 Documentation

| Document | Purpose |
| --- | --- |
| **[Workflow](docs/workflow.md)** | Complete 4-phase result generation & validation (sweep → merge → export → import) |
| **[Architecture](docs/architecture.md)** | Internal pipeline, caches, parallelism model, .ens contract |
| **[Scores](docs/scores.md)** | Score math: target counting, logit conversion, class offset, voting |
| **[Ensemble](docs/ensemble.md)** | Merge-ensemble search algorithms (beam, greedy, tries, optimal) |
| **[Encoding](docs/encoding.md)** | Input encodings (thermometer, raw, sizeN) |
| **[Bit-Mass](docs/bitmass.md)** | Container principle: every neuron = 32-bit container |

---

---

## License

Public domain. Research code — use at your own risk.