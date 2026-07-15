# Input Encoding — The Number World vs Binary World

**Every number from the number world must be converted into an amplitude in the
binary world before MAJ3 can process it.** This is a fundamental requirement:
MAJ3 operates on bits (0/1), not on scalar values. Floating-point addition,
multiplication, and division — the native operations of the number world — do
not exist in a DRAM bit-logic chip.

What the ML community calls **quantization** is exactly this: a deliberate,
lossy generalization of continuous data into discrete levels. In the Otto Score
framework, encoding IS quantization — but specialized for bit-logic hardware.

The challenge is: the encoding must preserve the **information structure** of
the original data. Different data domains need different encodings. Pixel
intensities (0–255) need a different mapping than hue angles (cyclic 0–255)
or edge magnitudes (power-law distributed). The encoding schemes `log`, `exp`,
`sig`, `lin`, `up`, `down`, `mid` each match a specific statistical distribution.

**Both Otto Score and Hebbian classifiers require input encoding for continuous
data (CIFAR).** Currently, we apply encoding to **color values** (the pixel
intensity coming from the number world). Raw pixel values (8-bit, 0–255) cannot
be fed directly into MAJ3 — a random projection W0 sees bit patterns, not
numeric magnitudes. Encoding transforms each pixel into a 32-bit container
whose bit pattern preserves the information MAJ3 needs.

---

## 1. The Core Problem

A DRAM chip can perform bitwise operations on entire rows in parallel — but it
cannot efficiently multiply or add floating-point numbers. Every input pixel must
be represented as a **bit pattern** that the chip can compare via XNOR + MAJ3.

This creates a fundamental tension: **integer/floating-point values are continuous,
but binary patterns are discrete.**

### 1.1 MNIST: When Raw Input Works

MNIST handwritten digits are **binary by nature**: each pixel is either "ink"
(>128, treat as 1) or "paper" (<128, treat as 0). When packed into uint32
containers (4 pixels × 8 bits each), the resulting 32-bit pattern directly
represents the digit's shape. No encoding transformation is needed.

```
Raw MNIST pixel (8-bit):   0x00 0xFF 0xFF 0x00  → 0x00FF00FF
                             (paper, ink, ink, paper)
```

This is why standard Otto Score reaches **86% in 1 pass** and **97%+ with iteration**
on MNIST without any special encoding — the signal is already binary.

### 1.2 CIFAR: The Continuity Requirement

CIFAR-10 color images have **grayscale gradients** (smooth transitions between
shades). A pixel value of 127 and 128 differ by only 1 in the number world, but
in raw 8-bit binary they differ by up to 8 bits (0x7F → 0x80 = bit pattern flip
from 01111111 to 10000000).

This **discontinuity** breaks the XNOR+popcount similarity measure: two nearly-
identical pixels can have very different popcounts. Without encoding, CIFAR
accuracy collapses to ~25% (random) regardless of H or ensemble size.

### 1.3 The Solution: Thermometer Encoding

**Thermometer encoding** restores continuity: each value `v` (0–255) is mapped
to a pattern where the first `v>>s` bits are 1, the rest 0:

```
raw 8-bit:  127→0x7F (popcount=7)    128→0x80 (popcount=1)    ← DISCONTINUOUS
thermometer:127→0x1F (popcount=5)     128→0x20 (popcount=6)    ← CONTINUOUS
```

Two adjacent values now have **similar popcount** — the bit-level similarity
reflects the numerical similarity. This is the key insight that makes Otto Score
work on photographic images.

### 1.4 Rule of Thumb

| Dataset                | Encoding required? | Default encoding |
| ---------------------- | ------------------ | ---------------- |
| MNIST                  | No (raw works)     | `latest` → `exp` |
| Fashion-MNIST          | No (raw works)     | `latest` → `exp` |
| CIFAR-10               | **Yes**            | `latest` → 17 members (ey-b+ey-a+ey-h+ey-s-1+ey-s-2) |
| CIFAR-100              | **Yes**            | `latest` → 17 members |
| Custom continuous data | **Yes**            | `exp` or `sig`   |

---

## 2. How Encoding Works

Each encoding first computes a **level** (0 to `max_lev`, where `max_lev = width`
for most encodings, except `lin7` where `max_lev = 7`). The level is then
converted to a **thermometer bitmask**:

```
output = ((uint32_t)1 << level) - 1    // level lower bits set to 1
// Examples: level=0 → 0x00000000, level=3 → 0x00000007 (popcount=3),
//           level=8 → 0x000000FF (popcount=8), level=32 → 0xFFFFFFFF
```

**Term definitions:**
- **`pv`** = pixel value (0–255) — the raw 8-bit input from the image after channel transformation.
- **`width`** = encoding output width in bits (8, 16, or 32). Determines the number of thermometer levels: `max_lev = width` (except `lin7`).
- **`max_lev`** = maximum thermometer level = number of bits in the output pattern. Default: `width` (so width=8 → `max_lev=8` → output ranges from 0x00 to 0xFF).
- **`low_lev`** = segment boundary level used by multi-segment encodings (`down`, `up`, `mid`) to allocate thermometer resolution across bright/dark/mid ranges.
- **`mid_lev`** = `max_lev − low_lev` — the remaining levels allocated to the opposite segment.

After computing `level`, the final output is always: `((uint32_t)1 << level) - 1`
(thermometer mask: exactly `level` lower bits set to 1).

---

## 3. Encoding Schemes

| Encoding       | level formula                                                                                                                                                                 | max_lev | Best For                            |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ----------------------------------- |
| `raw`          | `return pv` (identity, no thermometer)                                                                                                                                        | —       | MNIST, binary signals               |
| `lin7`         | `pv >> 5`                                                                                                                                                                     | **7**   | Legacy 7-level linear               |
| `lin` / `lin8` | `(pv × max_lev) >> 8`                                                                                                                                                         | width   | Luminance — uniform slope           |
| `down`         | `pv < 64` → `(pv × low_lev) / 64`, else `low_lev + ((pv-64) × (max_lev-low_lev)) / 192`; `low_lev = max_lev × 3/7`                                                            | width   | Shadows — more levels in dark       |
| `up`           | `pv > 192` → `low_lev + ((pv-192) × (max_lev-low_lev)) / 63`, else `(pv × low_lev) / 192`; `low_lev = max_lev × 4/7`                                                          | width   | Highlights — more levels in bright  |
| `mid`          | 3-segment: `pv<80` → `(pv×low_lev)/80`, `pv>175` → `low_lev + ((pv-175)×mid_lev)/80`, else `low_lev + ((pv-80)×mid_lev)/95`; `low_lev=max_lev×3/8`, `mid_lev=max_lev-low_lev` | width   | Mid-tones — band-pass               |
| `log`          | `log₂(pv+1) × max_lev / log₂(256)`                                                                                                                                            | width   | Power-law — high res for dark       |
| `exp`          | `(pv/255)^e × max_lev`, with `e=0.30` (width 8/32) or `e=0.35` (width 16)                                                                                                     | width   | Bright-sparse — high res for bright |
| `sig`          | `σ(12 × (pv/255 − 0.5)) × max_lev`, `σ(x)=1/(1+e⁻ˣ)`                                                                                                                          | width   | Opponent channels — S-shaped        |
| `sqrt`         | `√(pv/255) × max_lev`                                                                                                                                                         | width   | Softer bright emphasis (e=0.5)      |
| `cbrt`         | `∛(pv/255) × max_lev`                                                                                                                                                         | width   | Natural image curve (e=0.33)        |
| `gamma`        | `(pv/255)^0.45 × max_lev`                                                                                                                                                     | width   | Between exp(0.30) and sqrt(0.50)    |
| `tri`          | `min(pv, 255-pv) × 2/255 × max_lev`                                                                                                                                           | width   | Mid-tone emphasis (triangle peak)   |
| `inv-exp`      | `(1−e^(−4·pv/255)) × max_lev`                                                                                                                                                 | width   | Dark emphasis                        |

### Examples (width=8, all produce thermometer values 0..255)

| pv  | raw      | lin      | exp      | log      | sig      | up       | down     | mid      |
| --- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| 0   | 0x00 (0) | 0x00 (0) | 0x00 (0) | 0x00 (0) | 0x00 (0) | 0x00 (0) | 0x00 (0) | 0x00 (0) |
| 32  | 0x20 (1) | 0x01 (1) | 0x07 (3) | 0x3F (6) | 0x00 (0) | 0x00 (0) | 0x03 (2) | 0x07 (3) |
| 64  | 0x40 (1) | 0x03 (2) | 0x1F (5) | 0x7F (7) | 0x00 (0) | 0x01 (1) | 0x0F (4) | 0x0F (4) |
| 128 | 0x80 (1) | 0x0F (4) | 0xFF (8) | 0xFF (8) | 0x0F (4) | 0x07 (3) | 0x7F (7) | 0xFF (8) |
| 192 | 0xC0 (2) | 0x3F (6) | 0xFF (8) | 0xFF (8) | 0xFF (8) | 0x1F (5) | 0xFF (8) | 0x7F (7) |
| 255 | 0xFF (8) | 0xFF (8) | 0xFF (8) | 0xFF (8) | 0xFF (8) | 0xFF (8) | 0xFF (8) | 0xFF (8) |

Note: `raw` encoding's popcount is unrelated to brightness (`0x80` = 128 has
popcount 1, same as `0x01` = 1) — hence thermometer is essential for
photographic images.

**Width suffix:** `exp8` = width 8 bits, `exp16` = width 16 bits.

---

## 4. Channels + Encodings Define a Member

Every ensemble member is defined by the pair (channel, encoding). Multiple
members can share the same channel with different encodings, or the same
encoding on different channels. Each combination creates an independent binary
view of the input.

### 4.1 Channels (CIFAR-10)

Built-in channels derived from CIFAR-10's raw R, G, B values. Many are inspired
by **human opponent-color vision** (see [color-vision-opponent-channels.md](color-vision-opponent-channels.md)).

| Token        | Name                 | Formula                  | Description                      |
| ------------ | -------------------- | ------------------------ | -------------------------------- |
| `r`          | Red                  | raw R                    | Raw red pixel                    |
| `g`          | Green                | raw G                    | Raw green pixel                  |
| `b`          | Blue                 | raw B                    | Raw blue pixel                   |
| `y`          | Luminance (BT.601)   | `(77R+150G+29B)>>8`      | ITU-R BT.601 broadcast luminance |
| `lum` / `al` | Luminance (R+G)      | `(R+G)>>1`               | Simple luminance sum             |
| `rg` / `am`  | Red-Green opponent   | `(R-G+255)>>1`           | Red vs Green (L-M)               |
| `by` / `ap`  | Blue-Yellow opponent | `(2B-R-G+510)>>2`        | Blue vs Yellow (S-(L+M))         |
| `bl`         | Luminance (R+B)      | `(R+B)>>1`               | Alternative luminance            |
| `bm`         | R-B opponent         | `(R-B+255)>>1`           | Red-Blue difference              |
| `bp`         | G-(R+B)/2 opponent   | `(2G-R-B+510)>>2`        | Green vs Red+Blue                |
| `h`          | Hue                  | `atan2(2R-G-B, G-B)`     | Color angle (HSV)                |
| `c`          | Contrast             | Sobel 3×3 edge magnitude | Spatial edges on LUM             |
| `edge`       | Sobel Edge           | Gradient magnitude on Y  | Edge strength (post-process)     |
| `bin`        | Otsu Binary          | Otsu threshold on Y      | Filled black/white regions       |
| `lbp`        | LBP (texture)        | 8-bit Local Binary Pattern on Y | Luminance texture descriptor |
| `dog`        | DoG (band-pass)      | Box 3×3 − Box 5×5 + 128 | Band-pass edge filter            |
| `var`        | Variance (roughness) | Mean absolute deviation 3×3 | Local texture roughness      |
| `dir`        | Gradient direction   | 8-bin quantized from Sobel gx/gy | Edge orientation (0..248)  |
| `range`      | Local range          | max − min in 3×3         | Texture sharpness (no division)  |
| `lbp-rg`     | LBP on RG opponent   | 8-bit LBP on R-G color   | Chromatic texture                 |

**How to combine:** `--channels h,c,lum,by,rg` selects Hue + Contrast +
Luminance + Blue-Yellow + Red-Green. Each channel becomes a separate block
of containers.

### 4.2 Member Definition Examples

```
# One member: channel=mnist, encoding=exp8
--channels mnist --encoding exp8

# Two members: same channel, different encodings
--channels mnist --encoding exp8,log8

# Three members: different channels with per-channel encoding
--channels g,bl,bm --encoding g=up8,bl=down8,bm=sig

# 10 members: full CIFAR best config
--encoding g=up8,bl=down8,bm=sig,bp=sig,b=up8,al=down8,am=sig,ap=sig,h=up8,c=log8
```

Each comma-separated entry in `--encoding` creates one virtual block = one
member. When `--channels` has fewer entries than `--encoding`, the last channel
is reused.

---

## 5. Container Width: 8/16/32 Does Not Help

The natural question: does widening the thermometer (8 → 16 → 32 bits per pixel)
improve accuracy? Empirically: **no** — and the reason is fundamental.

Each thermometer value is passed through **MAJ3**, which compresses NC_slice
containers into exactly 32 bits via bitwise majority voting. After MAJ3, every
bit-width produces exactly 32 bits. A wider container (16 or 32 bits per pixel)
does not increase the information reaching the target — it only spreads the same
32-bit budget across fewer containers.

```
Width 8:   4 pixels × 8 bit  = 1 uint32 container  → MAJ3 → 32 bits
Width 16:  2 pixels × 16 bit = 1 uint32 container  → MAJ3 → 32 bits
Width 32:  1 pixel  × 32 bit = 1 uint32 container  → MAJ3 → 32 bits
```

**All widths give the same popcount range for MAJ3 voting.** The popcount (after
XNOR between W0 and containers) collapses the width back to a single integer —
wider containers do not provide more gradient resolution.

**The correct lever is more members, not wider containers.** Duplication (channel
+ encoding variants) creates independent MAJ3 projections with different random
W0, providing genuinely new information. Width variation just redistributes the
same information.

---

## 6. Encoding Orthogonality

Encodings should be **orthogonal**: two encodings on the same pixel should
produce bit patterns that are as decorrelated as possible. If they produce
similar patterns, the ensemble gains nothing — the second encoding is redundant.

The best MNIST config (`exp8,log8,sig8`) exploits this:

| Encoding | Maps pixel value v=128 to | Popcount | Character                   |
| -------- | ------------------------- | -------- | --------------------------- |
| `exp8`   | `0x01`                    | 1        | Compressed dark range       |
| `log8`   | `0xFE`                    | 7        | Compressed bright range     |
| `sig8`   | `0x60`                    | 2        | S-curve, mid-range emphasis |

Same pixel, three completely different bit patterns. The MAJ3 on each encoding
produces different error patterns → the ensemble averages them out.

For CIFAR, orthogonality is achieved by pairing channels with encodings matched
to their statistical distribution:

| Channel         | Distribution         | Best Encoding | Why                                       |
| --------------- | -------------------- | ------------- | ----------------------------------------- |
| G (raw green)   | Uniform 0–255        | `up8`         | Linear ascending highlights bright greens |
| BL (R+B lum)    | Uniform 0–255        | `down8`       | Linear descending highlights dark areas   |
| BM (R-B opp.)   | Peaked at 128        | `sig8`        | Sigmoid saturates opponent extremes       |
| H (hue angle)   | Cyclic 0–255         | `up8`         | Ascending preserves angle order           |
| C (Sobel edges) | Power-law (80% < 30) | `log8`        | Logarithmic gives dark edges more levels  |

---

## 7. Encoding and Container Layout

The output of encoding is a flat array of uint32 containers. The number of
containers per sample depends on the encoding width:

```
Containers per sample = (pixels × encoding_width_bytes) / 4
```

For MNIST (784 pixels, `--encoding exp`):
`784 × 2 / 4 = 392` uint32 containers → fixed at 196 after packing.

For CIFAR-10 (3072 bytes, `--encoding exp`):
`3072 × 2 / 4 = 1536` uint32 containers → distributed across active channels.

The frozen W0 matrix is sized to match the total container count. The encoding
choice therefore affects the memory footprint of the entire model.

---

## 8. Encoding Is Part of the Trained System

Encoding is **not a preprocessing convenience** — it is a structural part of
the classifier. Changing the encoding between training and inference produces
incorrect results because W0 sees different bit patterns.

The encoding configuration is therefore recorded in:

- The score archive (`.ens`) header for ensemble merging
- The `--export` model file for inference
- The input cache hash (`load_input_cached`) for deterministic replay

The `--encoding latest` shorthand selects the best-known configuration for the
current dataset (expanded at runtime into the full encoding table). For CIFAR-10,
`latest` now maps to 17 members: `ey-b,ey-a,ey-h,ey-s-1,ey-s-2` (reaching 63.7%).

---

## 9. Encoding Group Aliases (CIFAR-10)

The CIFAR-10 dataset defines multi-block encoding aliases that combine
channel selection with per-channel encoding. The core pattern `up+down+sig+sig`
is applied to different raw color channels:

| Alias   | Channels                              | Pattern               |
| ------- | ------------------------------------- | --------------------- |
| `ey-b`  | `g=up, bl=down, bm=sig, bp=sig`      | G + opponent pairs    |
| `ey-a`  | `b=up, al=down, am=sig, ap=sig`      | B + opponent pairs    |
| `ey-c`  | `r=up, cl=down, cm=sig, cp=sig`      | R + opponent pairs    |
| `ey-h`  | `h=down, c=exp, gb=sig`              | Hue + Contrast + Diff |
| `top-rgb` | `r=down, g=down, b=down`          | Raw RGB with down     |
| `latest`   | `ey-b,ey-a,ey-h`                          | 11 members (optimal)     |
| `latest-2` | `g=down,bl=gamma,bm=sig,bp=sig,b=sqrt,al=down,am=sig,ap=sig,h=lin,c=cbrt,gb=sig` | 11 members (2026-07-14 sweep) |
| `ey-s`     | `lbp=up,dog=sig,var=exp`                          | 3 spatial/texture members   |
| `ey-s-1`   | `lbp=gamma,dog=sig,var=exp`                       | 3 spatial (gamma variant)   |
| `ey-s-2`   | `dir=down,range=log,lbp-rg=mid`                   | 3 spatial: dir+range+lbp-rg |
| `ey-c`     | `r=up,cl=down,cm=sig,cp=sig`                      | R + opponent pairs (same pattern as ey-a/ey-b) |
| `top-rgb`  | `r=down,g=down,b=down`                            | Raw RGB with down           |
| `latest`   | `ey-b,ey-a,ey-h,ey-s-1,ey-s-2`                   | **17 members** (best, 63.7%) |
| `latest-2` | `g=down,bl=gamma,bm=sig,bp=sig,b=sqrt,al=down,am=sig,ap=sig,h=lin,c=cbrt,gb=sig` | 11 members (2026-07-14 sweep) |

### 9.1 Key Findings (2026-07-15)

**Spatial channels break the 62% ceiling.** Adding `ey-s-1` (lbp+dog+var with gamma)
and `ey-s-2` (dir+range+lbp-rg) to the color channels reaches **63.7%** at H=1024,
EN=3, ep=16. Best config: `--encoding ey-b,ey-a,ey-h,ey-s-1,ey-s-2`.

**ey-a, ey-b, ey-c are the same pattern permuted** over R/G/B. After two in the
system, the third adds nothing. `ey-c` is retained for compatibility but not used
in `latest`.

**`latest` now has 17 members** (4+4+3+3+3 = ey-b + ey-a + ey-h + ey-s-1 + ey-s-2).
The previous combination of color-only channels plateaued at 61.4%. Spatial channels
(dir/range/lbp-rg) provide orthogonal information that pushes past the ceiling.

### 9.2 Practical Guidance

- **Best config:** `--encoding ey-b,ey-a,ey-h,ey-s-1,ey-s-2` (17 members, 63.7%).
- **splitVN=2 hurts spatial channels** — they need full bit resolution. Use no splitVN.
- **Sweet spot:** H=512, EN=3, ep=16 → 63.5% at 251s (vs 63.7% at 538s for H=1024).
- **`ey-c` is redundant** with ey-a and ey-b (same pattern, different channel).

## 10. New Encodings Implemented (2026-07-14)

Five new thermometer-compatible encodings were added to `ki-encoding.h`,
extending the set from 9 to 14 total. All are orthogonal to the existing set:

| Encoding | Level formula | Orthogonal to | Status |
|----------|--------------|---------------|--------|
| `sqrt` | `√(pv/255) × max_lev` | exp, up | ✅ implemented |
| `cbrt` | `∛(pv/255) × max_lev` | exp, sqrt | ✅ implemented |
| `gamma` | `(pv/255)^0.45 × max_lev` | exp, up, down | ✅ implemented |
| `tri` | `min(pv, 255-pv) × 2 / 255 × max_lev` | sig, mid | ✅ implemented |
| `inv-exp` | `(1−e^(−4·pv/255)) × max_lev` | exp, down | ✅ implemented |

All five use the same thermometer output as the existing encodings:
`((uint32_t)1 << level) - 1`. Only the `level` computation differs.

---

## References

- [color-vision-opponent-channels.md](color-vision-opponent-channels.md) — Opponent channel theory for CIFAR-10
- [otto-score.md](otto-score.md) — Otto Score classifier overview
- [hebbian.md](hebbian.md) — Hebbian baseline (also requires encoding)
- `src/encoding.c` — Encoding LUT generation
- `src/load_input.c` — Input encoding + container packing
