# Color Vision — Opponent Channel Theory for CIFAR-10

## Background (from ophthalmology/AI research)

Human color vision uses **3 cone types** (L=red, M=green, S=blue) that are immediately
transformed into **opponent channels** in the retina:

| Channel     | Formula   | Meaning        |
| ----------- | --------- | -------------- |
| Red–Green   | L − M     | Red vs Green   |
| Blue–Yellow | S − (L+M) | Blue vs Yellow |
| Luminance   | L+M       | Brightness     |

This means: the brain does NOT process R, G, B independently. It compares them.
A pure R channel alone carries very limited information — almost no luminance,
no color contrast.

## Implication for Otto Score on CIFAR-10

CIFAR-10 images provide 3072 raw bytes (R,G,B interleaved). Our color-split
approach packs each color independently (KI_PACK=4, KI_NC=256 per color) and
treats R, G, B as 3 independent input vectors.

**Experiment (2026-06-24):** Training only the R channel (`--channels r`)
with H=512, ens=1, 1 epoch gives ~30.3% eval. Adding more H or more ensemble
members does NOT improve — the signal in a single color channel is intrinsically
limited (no opponent contrast, no luminance sum).

Expected result with all 3 colors combined: significantly higher, because
the opponent relationships (R−G, B−(R+G)) can be learned across members.

## `--channels` Bitmask

The `--channels` flag accepts a bitfield string to simulate color blindness
or isolate channels:

| Value         | Active colors  | Simulates                      |
| ------------- | -------------- | ------------------------------ |
| `r`           | Red only       | Missing M+S cones              |
| `g`           | Green only     | Missing L+S cones              |
| `b`           | Blue only      | Missing L+M cones              |
| `rg`          | Red + Green    | Missing S cones (Tritanopia)   |
| `rb`          | Red + Blue     | Missing M cones (Deuteranopia) |
| `gb`          | Green + Blue   | Missing L cones (Protanopia)   |
| `rgb`         | All 3          | Normal vision                  |
| `y`           | Luminance only | Achromatopsia (complete)       |
| `rgy`         | R+G+Y          | —                              |
| `rb`          | R+B (legacy)   | —                              |
| `lum,rg,by,y` | All 4 opponent | Full opponent set              |

Internally stored as bitmask (legacy): `R=1, G=2, B=4, Y=8`.

## Opponent Channels (current, 2026-06-24)

Replaces raw R,G,B with biologically-inspired opponent channels:

| Channel | Formula      | Mapping to 0..255    | Meaning               |
| ------- | ------------ | -------------------- | --------------------- |
| LUM     | R+G          | (R+G)>>1             | Luminance (L+M)       |
| RG      | R-G          | (R-G+255)>>1         | Red–Green (L-M)       |
| BY      | B-(R+G)/2    | (2B-R-G+510)>>2      | Blue–Yellow (S-(L+M)) |
| Y       | ITU-R BT.601 | (R*77+G*150+B*29)>>8 | ITU-Y (legacy)        |

### Bitmask (new)

| Bit | Token   | Default | Channel |
| --- | ------- | ------- | ------- |
| 0   | lum / l | ✓       | LUM     |
| 1   | rg      | ✓       | RG      |
| 2   | by      | ✓       | BY      |
| 3   | y       | —       | Y       |

Default (no `--channels`): LUM+RG+BY (bits 0+1+2).  
Comma-separated: `--channels lum,by` (LUM+BY only).

### Data layout

Default (no Y): `[LUM(0..255), RG(256..511), BY(512..767)]` stride=768.  
With Y (e.g. `--channels lum,rg,by,y`): `[LUM, RG, BY, Y(768..1023)]` stride=1024.

### Benchmark (2026-06-24, H=64, ens=1, 1 epoch, splitHN=2, 5000/2000)

| Channels  | eval  | Description         |
| --------- | ----- | ------------------- |
| LUM+RG+BY | 28.6% | Default opponent    |
| LUM only  | 27.1% | Luminance dominates |
| RG only   | 17.0% | Weak alone          |
| BY only   | 18.1% | Weak alone          |
| (old) RGB | 27.8% | Old approach        |

Opponent channels (28.6%) slightly outperform raw RGB (27.8%) at same bit-mass.
The opponent difference (RG, BY) carries little signal alone but adds ~1.5% to LUM.
