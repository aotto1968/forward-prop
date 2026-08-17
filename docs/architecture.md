# Architecture — Internal Pipeline & Concurrency Model

**Scope:** How the trainer (`mlp-bin32-otto-trn-seq-prof.c`) turns raw pixels
into trained members, how the caches work, and how parallel members are safe
despite shared global state. This is the "how it really works inside"
reference — the score math itself lives in [scores.md](scores.md).

---

## 1. The Big Picture

```
www/data/mnist/*.idx.gz
   │  ki-local.h: decompress + load raw grayscale (uint8, 0-255)
   ▼
X_raw  (n_samples × 784 bytes)
   │  per member (parallel, see §4):
   │  1. xform transform     → transformed grayscale (cached, §3)
   │  2. channel/encoding    → per-pixel thermometer bits
   │  3. container packing   → uint32 containers (member's input_buf)
   │  4. gb computation      → h0/gb (the activity mask)
   │  5. target counting     → logits per (h,v) per class
   │  6. score computation   → .ens archive (scores + labels)
   ▼
scores-*/<member>.ens   (the score archive, see scores.md)
   │  merge-ensemble (--beam/--greedy/--target)
   ▼
optimal member subset (--member-out)  →  best accuracy
```

Two independent "worlds" meet here:

- **Training world** (this file): pixels → per-member `.ens` archives.
  Parallel, cached, one member per file.
- **Merge world** ([scores.md](scores.md), [ensemble.md](ensemble.md)):
  `.ens` files → ensemble accuracy. Serial reading, additive voting.

The two worlds touch ONLY through the `.ens` file format — that is the
contract. Everything the trainer computes about a member is either in the
`.ens` header (config + precision) or in the score table.

---

## 2. The Member Pipeline (per member)

Each member is one `(xform, channel, encoding, seed)` combination. The
trainer's member loop (`seq-prof.c:3668`) processes members in PARALLEL
(see §4). Per member, in order:

### 2.1 Xform transform (grayscale)
`ki_xform_apply_buf()` (`ki-common.h`) transforms the raw grayscale buffer:
hflip, rot90, colswap, avg2/3/4, `@`-pipes (chained). Operates on
`uint8_t` grayscale (0-255) — **bit-width independent** (verified 2026-08-11,
see the xform note in `ki-encoding.h`). Identity members skip this entirely.

### 2.2 Channel + encoding (per pixel)
`enc_lut_get(enc, width, pv)` thermometers the grayscale value into a
bitmask (8/16/32 levels). The encoding LUT is initialized once per
(encoding, width) pair before the member loop.

### 2.3 Container packing
`pack = 32 / width` pixels per uint32 container (`ki-load.h`). 8-bit → 4
px/cont (196 containers), 16-bit → 2 px/cont (392), 32-bit → 1 px/cont
(784). The result is the member's `input_buf`.

### 2.4 gb computation (activity mask)
`ki_gb_for_neuron()`: Otto → `h0_to_gb(h0_neuron(W0 × in))`; Bit-Voting →
`in[h]` (identity, no W0). The gb is the 32-bit activity mask that the
score math consumes.

### 2.5 Target counting → logits
`ki_build_target_from_gb()` counts hits per `(k, h, v)` over the training
set, then `logit_convert()` turns counts into scaled log-odds (`F =
1<<OT_PRECISION`). This is the "learning" — see [scores.md](scores.md)
steps 3-5 for the exact math.

### 2.6 Export
`export_one_member_ens()` computes the eval scores over the test set and
writes ONE `.ens` file (header + scores + labels, v14/v15 with the
precision block). Roundtrip-verified at write time (`ens_verify`).

---

## 3. The Caches (and why they are safe)

### 3.1 The single-slot XFORM cache (in-memory)

`ki-load.h` keeps ONE global buffer for the transformed grayscale:

```c
static uint8_t *_xf_cache_buf = NULL;   /* owned by the cache */
static int      _xf_cache_id  = -1;     /* xform_id of the cached buffer */
static int      _xf_cache_n   = 0;      /* n_samples it was built for  */
```

`ki_xform_get(X_raw, n, xform_id)`:
- identity → returns `X_raw` directly (no slot)
- same `(xform_id, n)` as cached → returns the slot (HIT)
- anything else → frees the slot, recomputes, returns the NEW slot (MISS)

**Why ONE slot is enough:** members are sorted xform-major, so all members
of one xform group are processed consecutively. The slot is overwritten
exactly once per group switch. Old groups are never re-read (in `--sweep`
each member is trained exactly once).

### 3.2 The DISK cache (CEX / .pre files)

`load_input_cex_cached()` (`ki-load.h`) keys a file per
`(channel, encoding, width, xform)`:
`data/prepped/cex_<chan>_<enc><w>_b<width>_<xform>_v<impl>_<N>x<slice>.pre`

- HIT: loads the packed container buffer from disk, NO xform transform.
- MISS: runs the transform once, packs, saves (unless `--sweep`).

The key includes `KI_BIT_WIDTH` and the xform implementation version — a
changed bit width or xform logic produces a NEW file name, so stale caches
are never served (bug 2026-08-04 colswap cache).

### 3.3 Why parallel members are safe (the concurrency contract)

The member loop is parallel (`#pragma omp parallel for schedule(dynamic,1)`,
`seq-prof.c:3664`). The caches are GLOBAL single slots. The rule that keeps
this correct:

> **The cache is only touched INSIDE `#pragma omp critical(cex)`.**
> `load_input_cex_cached()` is the ONLY caller of `ki_xform_get()`, and
> the trainer wraps that call in `critical(cex)` (`seq-prof.c:3790`).

The critical section serializes the CACHE ACCESS (0.9% of runtime), not
the training. After the call, the member owns its own `input_buf` (a fresh
`ki_xmalloc`) — **it never holds a pointer into the cache**. So:

```
Thread A (xform A):  critical(cex) → cache=A → X_A (own alloc) → exit lock
Thread B (xform B):  critical(cex) [waits] → cache=B → X_B (own alloc) → exit lock
```

- The cache slot is a TEMPORARY springboard (grayscale → packed
  containers), not a long-lived buffer.
- Once a member's `X` is built, the cache can be overwritten by any other
  thread without corrupting that member — it never reads the cache again.
- The danger the lock prevents: Thread A reading `src` (cache pointer)
  while Thread B `free()`s it (use-after-free). The lock covers the whole
  read+copy window.
- `schedule(dynamic,1)`: a member of the NEXT xform group can start (and
  even finish) before its group's predecessors — fully safe, no barriers.

### 3.4 The gb cache (optional, disk)

`--export-gb` / `--import-gb` write/read `data/gb/<hash>.gb` keyed by
`(W0[0..3], dims, channel, encoding, width, xform)`. Same single-writer
pattern: the gb computation is per-member and does not share buffers
across threads (the member owns its `gb_buf`).

---

## 4. Parallelism Model (exclusive)

```
OUTER (member loop):  omp parallel for ... if(g_parallel_members)
    └─ INNER (gb/samples): omp parallel for ... if(!g_parallel_members)
```

- **Many members** (sweep): the OUTER loop parallelizes; the INNER loops
  stay single (no nested OpenMP → no deadlock).
- **Few members** (H=512, 1 member): the OUTER stays single; the INNER gb
  loop parallelizes over samples with all cores.

The switch is `g_parallel_members` (`active_members >= 3`):
- `>= 3` members → outer parallel (width = `--threadN` or auto)
- `< 3` members → inner parallel (all cores on the gb loop)

This is the "exclusive parallelism" fix (bug 2026-08-09): never both at
once, so no nested OpenMP, no oversubscription.

---

## 5. The .ens Contract (trainer ↔ merge)

The `.ens` file is the ONLY interface between the two worlds. Everything a
consumer needs is in the header:

| Field                                          | Meaning                                 | Checked by merge?     |
| ---------------------------------------------- | --------------------------------------- | --------------------- |
| `version` (11/12/13/14/15)                     | SCORE_TYPE width + precision block      | ✅                    |
| `hidden` (H)                                   | container count (indirect KI_BIT_WIDTH) | ✅                    |
| `epochs, split_vn, split_hn, seed`             | training config                         | ✅                    |
| `maj_token, maj1_thresh`                       | majority mode                           | ✅ (skip for BV "-1") |
| `w0_marker`                                    | Otto vs Bit-Voting identity             | ✅                    |
| `ot_precision, bit_width, counter_type` (v14+) | how logits were computed                | ✅ (--check)          |
| member strings (color/enc/width/xform)         | member identity                         | filters use it        |

The `.meta` file (per directory) records the SAME config plus `EXE=` (the
trainer that produced the archive). `--check` validates every `.ens`
against `.meta`: mismatch → ERROR (precision), old v<14 → OK + [WARN:]
text (archive stays valid, retrain to get v14). See
[scores.md](scores.md) §3 for the exact byte layout.

### 5b. Merge CLI — the search controls (2026-08-12)

`merge-ensemble DIR [options]` combines `.ens` members into an ensemble.
The important knobs:

| Option | Purpose |
|--------|---------|
| `--beam N` / `--greedy` | search algorithm (beam width N, default; greedy single-path) |
| `--max [--min-gain F]` | cumulative acceptance — buffer sub-threshold steps, commit as ONE block |
| `--tries N [--tries-no-lock] [--tries-random-seed]` | multi-start search, keep the global best |
| `--optimal [best\|first]` | 2-opt exchange on the beam result (removal allowed). `best` = global-best swap (deterministic); `first` = first improving swap (different local optimum, faster). `--optimal-passes N` caps the search |
| `--target N[,M,...]` | optimize for specific class(es): correct counts ONLY when label ∈ targets AND pred == label; denominator = target-class samples (recall 0-100%) |
| `--filter-sample N[,M,...]` | SAMPLE-level filter: only test samples whose label is in the list enter the eval (correct + denominator) |
| `--filter ...` | MEMBER-level filter (eval threshold, label include/exclude, regex) |
| `--member-start FILE` | start the beam from a saved attractor (floor) — the set is shown as EN=0, only ADDITION is allowed (monotone) |
| `--member-out FILE` | export the optimal subset as XF:CHAN:ENC specs (feeds TRN `--member-file`) |
| `--check` | validate all `.ens` vs `.meta` (incl. v14+ precision), write repair specs to `--member-out` |

The beam is add-only and path-dependent: the result is a local optimum
(verified 2-opt-stable with `--optimal`), NOT a proven global maximum —
more/fewer members land in the same accuracy band (the "89% wall" for
Bit-Voting, the "91% wall" for XNOR — architectural, not search-limited).

---

## 6. Reading Order (for newcomers)

1. [scores.md](scores.md) — what a score IS and how it is computed (the math).
2. This file — how the trainer produces the archives (the pipeline).
3. [ensemble.md](ensemble.md) — how the merge finds the best subset.
4. [encoding.md](encoding.md) — input encoding for continuous data.
5. [bitmass.md](bitmass.md) — the container principle (bit mass = capacity).
