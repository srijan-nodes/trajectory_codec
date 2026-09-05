# Ablation Analysis: TrajCDDec V5 Mode Decomposition

This document provides deep analytical interpretation of the exhaustive $2^{11}$ ablation study and the controlled Leave-One-Out (LOO) benchmark.

---

## 1. The Pareto Frontier

### 1.1 What the Top 20 Share
Every configuration in the Top 20 has these modes **ON**: `m1_spatial`, `m7_solid`, `m11_cached`, `m13_faded`. Every configuration has `m6_bg` **OFF**. This identifies 5 structural invariants:

| Mode | State in All Top 20 | Role |
|------|---------------------|------|
| `m1_spatial` | **Always ON** | Spatial residual fallback — essential for handling blocks that no predictive mode can capture |
| `m7_solid` | **Always ON** | Solid-fill fast path — dominant on static background blocks |
| `m11_cached` | **Always ON** | Cached motion vector reuse — zero-byte payload for repeated motion |
| `m13_faded` | **Always ON** | Luma shift for brightness transitions — 1-byte payload |
| `m6_bg` | **Always OFF** | Background model — apparently adds overhead without compression benefit |

### 1.2 The Compression Plateau
The Top 20 span an extremely narrow range: **55.32% – 55.38%** unweighted mean reduction. This 0.06% spread across 20 configurations means the compression performance is dominated by the 5 core modes above. The remaining 6 modes (`m4_skip`, `m4_search`, `m9_dct`, `m10_dct`, `m12_palette`, `m15_dither`) are **interchangeable** — toggling them produces negligible compression impact.

### 1.3 The Irreducible Core
The algorithmic backbone of TrajCDDec V5 is a **5-mode quintet**:

```
m1_spatial + m7_solid + m11_cached + m13_faded + m4_search
```

This quintet appears in 16 of the Top 20 configurations. When `m4_search` is removed (Ranks 5–8), compression drops by only 0.02%, but MSE drops from 7.33 to 5.17 — a significant quality improvement. This suggests `m4_search` trades quality for marginal compression.

---

## 2. Mode-by-Mode Analysis

### m1_spatial — Spatial Residual Quantization
| Metric | Value |
|--------|-------|
| **Compression impact** | Essential — present in all Top 20 |
| **Enc FPS delta** | -2.30 (costs 2.30 FPS to enable) |
| **Dec FPS delta** | -9.99 |
| **Verdict** | 🔴 Essential for compression, significant FPS cost |

The most computationally expensive mode in the cascade. Computes `np.round((curr - prev) / QP)` for every block, identifies non-zero residuals, and serializes them. Despite costing 2.30 FPS encode and 9.99 FPS decode, it is **irreplaceable** — without it, the codec cannot handle blocks with novel content that doesn't match any predictive mode.

### m4_skip — Fast Skip Detection
| Metric | Value |
|--------|-------|
| **Compression impact** | Negligible — varies across Top 20 |
| **Enc FPS delta** | +1.42 overhead when enabled |
| **Dec FPS delta** | +16.38 overhead when enabled |
| **Verdict** | ⚪ Neutral — skip detection is already handled by the base cascade |

The skip detection fast path is redundant with the zero-motion check in the cascade's first gate (`base_sad / 256 < 1.0`). Enabling `m4_skip` adds a separate check that rarely triggers on content where the base check hasn't already caught it.

### m4_search — Local Bounding Box Motion Search
| Metric | Value |
|--------|-------|
| **Compression impact** | Marginal (+0.02% when enabled) |
| **Enc FPS delta** | +0.56 (slight contribution) |
| **Dec FPS delta** | +0.58 (negligible) |
| **Verdict** | 🟡 Marginal — small compression gain, small FPS cost |

Uses `cv2.matchTemplate` to search a 32×32 window for motion vectors. Contributes a barely-measurable +0.02% compression improvement but increases MSE from 5.17 to 7.33 in the Top 20 table. This mode finds more motion matches but at the cost of accepting higher-error matches. A candidate for removal if quality is prioritized over compression.

### m6_bg — Background Model
| Metric | Value |
|--------|-------|
| **Compression impact** | Harmful — excluded from all Top 20 |
| **Enc FPS delta** | -0.19 (negligible cost) |
| **Dec FPS delta** | +19.17 overhead when enabled |
| **Verdict** | 🔴 Harmful — excluded from optimal set |

The background model maintains a running average of static pixels. On synthetic video, this is redundant — the codec's skip and cache modes already handle static regions perfectly. The background model adds 19 FPS of decode overhead (the decoder must update `bg_model` every frame via exponential moving average) with no compression benefit.

### m7_solid — Solid Fill Detection
| Metric | Value |
|--------|-------|
| **Compression impact** | Essential — present in all Top 20 |
| **Enc FPS delta** | **+2.61** (contributes 2.61 FPS!) |
| **Dec FPS delta** | **+52.33** (contributes 52.33 FPS!) |
| **Verdict** | 🟢 Essential — massive speed AND compression benefit |

**The single most impactful mode.** Solid fill detection checks if a block has near-uniform intensity (`max - min < threshold`). If so, it writes a single byte (the median value) instead of 256 bytes of raw data. This is both:
- **Faster to encode:** One comparison + one byte write vs 256 bytes of residual computation
- **Faster to decode:** `np.full((16, 16, 1), val)` vs any other reconstruction

Disabling `m7_solid` drops encode FPS from 16.18 to 13.56 and decode FPS from 168.92 to 116.59. This is the rare **win-win mode** — it improves both speed and compression simultaneously.

### m9_dct — Low-Frequency DCT (4×4)
| Metric | Value |
|--------|-------|
| **Compression impact** | Zero — 0% hit rate on ball.mp4 |
| **Enc FPS delta** | +0.71 overhead when enabled |
| **Dec FPS delta** | +18.52 overhead when enabled |
| **Verdict** | 🔴 Vestigial — pure overhead on this test set |

DCT modes are architecturally present but never win the cost function competition on the test videos. Enabling `m9_dct` adds `cv2.dct()` and `cv2.idct()` calls that compute transform coefficients, evaluate them against the cost threshold, and always reject them in favor of cheaper modes (Raw, Dither, or Spatial).

### m10_dct — Mid-Frequency DCT (8×8)
| Metric | Value |
|--------|-------|
| **Compression impact** | Zero — 0% hit rate on ball.mp4 |
| **Enc FPS delta** | +1.07 overhead when enabled |
| **Dec FPS delta** | +13.33 overhead when enabled |
| **Verdict** | 🔴 Vestigial — same as m9_dct |

Identical analysis to `m9_dct`. The 8×8 DCT computes 64 coefficients (vs 16 for 4×4) but still loses to cheaper modes on cost. The combined overhead of both DCT modes is ~1.78 FPS encode and ~31.85 FPS decode.

### m11_cached — Cached Motion Vector Reuse
| Metric | Value |
|--------|-------|
| **Compression impact** | Essential — present in all Top 20 |
| **Enc FPS delta** | **+1.17** (contributes 1.17 FPS) |
| **Dec FPS delta** | -3.18 (slight decode overhead) |
| **Verdict** | 🟢 Essential — zero-byte payload mode |

Checks if the current block matches the previous frame at the last-known motion vector offset. If it does, the encoder writes **zero payload bytes** — just a mode flag. This is one of the most bandwidth-efficient modes in the codec. The 1.17 FPS encode contribution comes from early termination: when the cached vector matches, the encoder skips all subsequent mode evaluations.

### m12_palette — 4-Color Palette Quantization
| Metric | Value |
|--------|-------|
| **Compression impact** | Varies — present in Ranks 1–8, absent in Ranks 9–12 |
| **Enc FPS delta** | -2.38 (costs 2.38 FPS) |
| **Dec FPS delta** | -23.00 (costs 23.00 FPS) |
| **Verdict** | 🟡 Beneficial for compression, expensive for speed |

K-means clusters the block into ≤4 colors and writes a 69-byte palette+index payload. This is more expensive than Dither (Mode 15) because it includes a `num_colors` byte and runs the full K-means algorithm. The 23 FPS decode cost comes from the palette lookup and index unpacking operations.

### m13_faded — Luma Shift Motion
| Metric | Value |
|--------|-------|
| **Compression impact** | Essential — present in all Top 20 |
| **Enc FPS delta** | -1.37 (costs 1.37 FPS) |
| **Dec FPS delta** | -18.31 (costs 18.31 FPS) |
| **Verdict** | 🟢 Essential — handles fade transitions with 5-byte payload |

Detects blocks where the content is the same as the motion-predicted block but with a global brightness shift. Writes only 5 bytes (dy, dx, shift_value). Essential for fade-to-black and lighting transitions. The FPS cost comes from computing `np.mean()` on both blocks and evaluating the shifted prediction.

### m15_dither — 4-Level Dithered Quantization
| Metric | Value |
|--------|-------|
| **Compression impact** | Varies — present in Ranks 1–12, absent in Ranks 13–20 |
| **Enc FPS delta** | -0.97 (costs 0.97 FPS) |
| **Dec FPS delta** | -17.91 (costs 17.91 FPS) |
| **Verdict** | 🟡 Beneficial for compression, moderate speed cost |

Similar to `m12_palette` but without the `num_colors` byte (always 4 colors). Writes a 68-byte payload. Serves as the final fallback before Raw mode (256 bytes). The compression benefit comes from reducing 256 bytes to 68 bytes for blocks with limited color diversity.

---

## 3. The Negative Byte-Weighted Aggregate Explained

### The Math
For the Rank 1 configuration across all 11 videos:
- **Total original bytes:** ~8.7 MB (dominated by `05_break_entropy.mp4` at 7.6 MB)
- **Total compressed bytes:** ~21.5 MB (dominated by `05_break_entropy.mp4` expanding to 25.8 MB)
- **Byte-Weighted Aggregate:** $100 - (21.5 / 8.7 \times 100) = -147\%$

### Why This Is Expected
`05_break_entropy.mp4` contains 100% random noise. The Shannon entropy is maximal. No prediction-based codec can compress it — the codec's overhead (mode flags, frame headers, motion vector metadata) **adds** bytes on top of the raw data. This is a fundamental information-theoretic limit, not a codec bug.

### What Happens Without the Adversarial Video
Excluding `05_break_entropy.mp4`, the remaining 10 videos achieve positive compression across all Top 20 configurations. The Unweighted Mean Reduction (55.38%) is the fairer metric because it treats each video equally, preventing a single adversarial file from dominating the ranking.

---

## 4. The Dynamic Boxing Effect

Dynamic boxing merges adjacent blocks that share the same mode into larger rectangular operations. This reduces the number of decoder iterations but was disabled during the ablation study.

| Setting | Baseline Enc FPS | Baseline Dec FPS |
|---------|-----------------|-----------------|
| `dynamic_boxing: True` | ~16.74 | ~249.93 |
| `dynamic_boxing: False` | ~16.18 | ~168.92 |

The 80 FPS decode improvement from dynamic boxing comes from:
1. Fewer mode bytes to parse (merged blocks share one mode flag)
2. Fewer numpy slice operations (one large assignment vs N small ones)
3. Better cache locality (sequential memory access patterns)

**Recommendation:** Production deployments should use `dynamic_boxing: True` for maximum decode speed. The ablation study used `False` to isolate the per-mode effects without the confounding variable of block merging.

---

## 5. Cross-Mode Interactions

### The m1_spatial Paradox
`m1_spatial` is **essential for compression** (present in all Top 20) but **costs 2.30 FPS encode**. This is because:
- Spatial residual computation (`np.round((curr - prev) / QP)`) runs on every block
- Non-zero index extraction and serialization is expensive
- Without it, blocks with novel content fall through to Raw mode (256 bytes) — destroying compression

The trade-off is clear: accept 2.30 FPS slower encoding to avoid a catastrophic compression regression.

### The m7_solid Win-Win
`m7_solid` is the only mode that improves **both** speed and compression. This is because:
- The solid-fill check (`max - min < threshold`) is cheaper than any other mode evaluation
- A single-byte payload (vs 256 bytes raw) saves bandwidth
- The decoder reconstruction (`np.full()`) is cheaper than any other mode's reconstruction

### DCT Vestigiality
Both DCT modes (`m9_dct`, `m10_dct`) are never selected by the cost function on the test videos. This means:
- The `cv2.dct()` and `cv2.idct()` calls execute but their results are always rejected
- Combined overhead: ~1.78 FPS encode, ~31.85 FPS decode
- **Recommendation:** Disable both DCT modes in production unless the content contains smooth continuous gradients that span more than 4 intensity levels

### The Palette/Dither/Faded Trio
`m12_palette`, `m13_faded`, and `m15_dither` all cost similar decode FPS (-17 to -23 FPS each) but are essential for compression on blocks with limited color diversity or brightness transitions. They form a complementary trio:
- `m13_faded`: Handles brightness-shifted motion (5 bytes)
- `m15_dither`: Handles 4-level quantized blocks (68 bytes)
- `m12_palette`: Handles arbitrary ≤4-color blocks (69 bytes)

---

## 6. Recommendations

### Maximum Compression Configuration
```python
config = {
    'm1_spatial': True,    # Essential
    'm4_skip': False,      # Redundant
    'm4_search': True,     # +0.02% compression
    'm6_bg': False,        # Harmful
    'm7_solid': True,      # Essential
    'm9_dct': False,       # Vestigial
    'm10_dct': False,      # Vestigial
    'm11_cached': True,    # Essential
    'm12_palette': True,   # Beneficial
    'm13_faded': True,     # Essential
    'm15_dither': True,    # Beneficial
    'dynamic_boxing': True # For decode speed
}
```
This is the Pareto Rank 1 configuration with `dynamic_boxing: True` for production.

### Maximum Speed Configuration
```python
config = {
    'm1_spatial': False,   # -2.30 FPS cost
    'm4_skip': False,      # Redundant
    'm4_search': False,    # Skip matchTemplate
    'm6_bg': False,        # Harmful
    'm7_solid': True,      # +2.61 FPS AND compression
    'm9_dct': False,       # -0.71 FPS overhead
    'm10_dct': False,      # -1.07 FPS overhead
    'm11_cached': True,    # +1.17 FPS contribution
    'm12_palette': False,  # -2.38 FPS cost
    'm13_faded': False,    # -1.37 FPS cost
    'm15_dither': False,   # -0.97 FPS cost
    'dynamic_boxing': True
}
```
This maximizes encode FPS at the expense of compression ratio. Only `m7_solid` and `m11_cached` are retained as they are both speed-positive and compression-positive.

### The Pareto Trade-Off
There is no single "best" configuration — the optimal choice depends on whether you prioritize compression or speed. The Top 20 Pareto frontier shows that compression plateaus at ~55.38% regardless of which optional modes are enabled, suggesting that the **5-mode core** captures nearly all achievable compression and the remaining modes are fine-tuning.
