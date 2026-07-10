# Technical Documentation — Otto Score & Reference Baselines

This directory contains all end-user-facing documentation for the
**forward-prop** / **otto-score-ifc** project.

---

## Core Methods

| Document | Method | Type | Accuracy |
|----------|--------|------|:--------:|
| [otto-score.md](otto-score.md) | **Otto Score** | DRAM-native bit-logic | **96–97%** |
| [adamw.md](adamw.md) | **Float32 AdamW** | CPU/GPU float matmul | 92.6% |
| [hebbian.md](hebbian.md) | **Bin32 Hebbian** | DRAM-native bit-logic | ~82% |

## Theory & Concepts

| Document | Topic |
|----------|-------|
| [encoding.md](encoding.md) | **Input encoding** — required for continuous data (CIFAR) |
| [majority-vote.md](majority-vote.md) | Majority vote — how MAJ3 replaces matmul |
| [popcount.md](popcount.md) | Popcount off-table — the proof |
| [bitmass.md](bitmass.md) | Bit-Mass theory — the container principle |
| [DRAM.md](DRAM.md) | KI-DRAM: Boolean-only inference in commodity memory |
| [color-vision-opponent-channels.md](color-vision-opponent-channels.md) | Opponent channel theory for CIFAR-10 |

## Research Results

| Document | Topic |
|----------|-------|
| [scaling-law-otto-score-vs-human-brain.md](scaling-law-otto-score-vs-human-brain.md) | Scaling law from 4 neurons to human-level intelligence |
| [status-2026-07-04.md](status-2026-07-04.md) | Project status report — July 2026 |
| [paper-2026-06-29-otto-score-summary.md](paper-2026-06-29-otto-score-summary.md) | Otto Score research summary |

## Workflow & Tools

| Document | Topic |
|----------|-------|
| [ensemble.md](ensemble.md) | Ensemble workflow — train now, merge later |
| [vision_en.md](vision_en.md) | The evolution of life and perfect randomness |

---

## Quick Comparison

```
                Forward Pass              Training Method         Eval
Otto Score     XNOR → MAJ3 → Bayes log   Iterative correction   96–97%
AdamW          matmul → LReLU → matmul   AdamW (backprop)       92.6%
Hebbian        XNOR → MAJ3 → popcount    Batch co-occurrence    ~82%
```

**Key insight:** Otto Score achieves the highest accuracy while using
**zero floating-point** and **zero multiplication** during inference.
It is the only method designed to run natively on DRAM bit-logic hardware.

---

## Mirror

All documents in this directory are symlinked from the project root `docs/`
directory for convenient access. Edits should be made here, in
`otto-score-ifc/docs/`.
