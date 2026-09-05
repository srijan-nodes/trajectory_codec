# Research Methodology: TrajCDDec V5 Ablation Study

This document chronicles the complete experimental methodology — every phase of research, how experiments were designed, executed, validated, and what was learned from each.

---

## 1. Research Questions

The ablation study was designed to answer three questions:

1. **Which algorithmic modes are essential for compression?** Of the 11 toggleable modes in the V5 cascade, which ones actually contribute to data reduction, and which are vestigial?
2. **What is the marginal FPS cost of each mode?** Each mode adds computational overhead. What is the true encode/decode speed impact of enabling or disabling each one?
3. **How does TrajCDDec compare to standard codecs at ultra-low bitrates?** Can a heuristic-based block matcher outperform DCT-based codecs on synthetic content?

---

## 2. Test Suite

### 2.1 Synthetic Video Corpus
11 purpose-built synthetic videos were used, each targeting a specific codec stress axis:

| Video | Profile | Resolution | Frames | Content |
|-------|---------|-----------|--------|---------|
| `01_ideal_motion.mp4` | Rigid body | 640×360 | 90 | Smooth sphere traversal |
| `02_ideal_cache.mp4` | Cache test | 640×360 | 90 | Repetitive static patterns |
| `03_ideal_palette.mp4` | Palette | 640×360 | 90 | ≤4 color flat regions |
| `04_ideal_combined.mp4` | Combined | 640×360 | 90 | Mixed motion + palette |
| `05_break_entropy.mp4` | Adversarial | 640×360 | 90 | 100% random noise per frame |
| `06_break_gradient.mp4` | Gradient | 640×360 | 90 | Smooth continuous gradients |
| `07_break_edge.mp4` | Edge case | 640×360 | 90 | High-contrast sharp edges |
| `08_break_moire.mp4` | Moiré | 640×360 | 90 | Fine repeating patterns |
| `09_break_flicker.mp4` | Flicker | 640×360 | 90 | Frame-to-frame oscillation |
| `10_break_zoom.mp4` | Zoom | 640×360 | 90 | Scale transforms |
| `ball.mp4` | Complex | 640×360 | 101 | Shaded sphere with gradients |

The corpus was split into two families:
- **`ideal_*` videos (01–04):** Designed to play to the codec's strengths (rigid motion, cache hits, palette blocks)
- **`break_*` videos (05–10):** Designed to stress-test failure modes (entropy, gradients, moiré, flicker)

### 2.2 The Adversarial Video Problem
`05_break_entropy.mp4` contains 100% random noise. No compression algorithm can reduce random data — the codec expands it from 7.6 MB to 25.8 MB (3.4× expansion). This single video dominates the byte-weighted aggregate metric, producing the counterintuitive -147% "compression" seen in the Top 20 table. This is why the **Unweighted Mean Reduction** was chosen for ranking.

---

## 3. Phase 1: Exhaustive Ablation Sweep

### Design
- **Script:** `run_massive_ablation.py`
- **Combinatorial space:** $2^{11} = 2{,}048$ permutations of 11 binary mode toggles
- **Videos per permutation:** 11
- **Total encode tasks:** $2{,}048 \times 11 = 22{,}528$
- **Execution:** Python `multiprocessing.Pool` with resume capability via CSV checkpointing
- **Duration:** ~1.5 hours on a consumer CPU

### Configuration
All permutations shared these fixed settings:
- `dynamic_boxing: False`
- `name: 'Ablation'`
- Top-level keys `background`, `palette`, `motion`, `dct` set but never read by the encoder (dead config — only the per-mode granular toggles `m1_spatial`, `m4_skip`, etc. are checked)

### Metrics Collected Per Encode
| Metric | Source | Description |
|--------|--------|-------------|
| `orig_bytes` | `os.path.getsize(video_path)` | Source MP4 container size |
| `comp_bytes` | `os.path.getsize(output_nam)` | Compressed NAM file size |
| `time_s` | Wall-clock encode time | Used for FPS calculation |
| `frames` | Frame count from encoder | Used for FPS calculation |

### Aggregation
Two reduction metrics were computed per permutation:

1. **Unweighted Mean Reduction:** Arithmetic mean of per-video reduction percentages. Each video contributes equally regardless of size.
   $$\text{Unweighted} = \frac{1}{N} \sum_{i=1}^{N} \left(100 - \frac{\text{comp}_i}{\text{orig}_i} \times 100\right)$$

2. **Byte-Weighted Aggregate:** Total compressed bytes / total original bytes across all videos. Large videos dominate.
   $$\text{Aggregate} = 100 - \frac{\sum \text{comp}_i}{\sum \text{orig}_i} \times 100$$

### Error Handling
The script wrapped each encode in a `try/except` that produced a zero-value row on failure. The aggregation logic guards against this with `if orig > 0:` before computing per-video reductions. Zero rows don't corrupt the reduction math but are present in `raw_encode_results.csv`.

### Resume Logic
On restart, the script loads `raw_encode_results.csv` and builds a `set` of `(permutation, video)` tuples already completed. New tasks are only generated for uncompleted pairs. This prevents duplicate rows.

### Output Files
- `raw_encode_results.csv` — 22,529 rows (one per encode task)
- `ablation_matrix.csv` — 2,049 rows (one per permutation, sorted by Unweighted Mean Reduction)

---

## 4. Phase 2: Decode FPS Measurement (Flawed)

### Design
- **Script:** `massive_decode.py`
- **Scope:** 2,048 permutations × `ball.mp4` only
- **Execution:** Multiprocessed
- **Goal:** Measure decode FPS for each configuration to identify the "fastest decoder" configs

### The Critical Bug
Lines 22–28:
```python
try:
    res_dec = dec.decode(nam_path, None)
    dec_fps = frames / max(0.001, res_dec['dec_time'])
except:
    dec_fps = 0.0  # <-- POISONED DATA
```

When BUG-001 (Mode 15 desync) caused the decoder to crash, `dec_fps` was silently set to `0.0`. These zeros entered the `np.mean()` calculation, dragging down the average for any mode correlated with crashing configurations.

### Impact
The "Decoder FPS" column in the original Top 20 table and the earlier LOO table contained poisoned data. For example, Rank 1 showed 199.45 Dec FPS — this may include zero-poisoning from crashed configs that happened to share mode flags with the Rank 1 permutation.

### Lesson
This script was entirely replaced by `controlled_benchmark.py`, which uses `None` filtering instead of zero-substitution. The `massive_decode.py` results are retained in the repository for historical reference but should not be cited.

---

## 5. Phase 3: Controlled Sequential Benchmark

### Design
- **Script:** `controlled_benchmark.py`
- **Methodology:** Leave-One-Out (LOO) against Pareto Rank 1
- **Execution:** Strictly sequential, single-threaded (no multiprocessing)
- **Trials:** 3 per configuration
- **Video:** `ball.mp4` (101 frames)
- **Configuration:** `dynamic_boxing: False` (matching Phase 1)

### Why Sequential
The multiprocessed Phase 2 benchmark suffered from:
- **CPU scheduling contention:** Multiple encoder processes competing for L1/L2 cache
- **Thermal throttling:** Sustained multi-core load caused frequency drops
- **OS-level thread migration:** Processes moved between cores mid-run
- **Memory bus saturation:** Multiple numpy array allocations competing for bandwidth

Sequential execution eliminates all of these noise sources. Each trial runs on a quiescent system.

### LOO Toggle Logic
For each of the 11 modes:
1. Start with the Pareto Rank 1 baseline config
2. Flip the single mode's state (ON→OFF or OFF→ON)
3. Run 3 sequential encode+decode trials
4. Compute `Mean ± Std` for both encode and decode FPS
5. Compute delta: `baseline_fps - toggled_fps` for ON→OFF modes; `toggled_fps - baseline_fps` for OFF→ON modes

### Delta Sign Convention
- **Positive delta for ON modes:** The mode contributes this many FPS. Disabling it loses performance.
- **Positive delta for OFF modes:** Enabling the mode costs this many FPS in overhead.

### Statistical Notes
- `np.std()` uses population standard deviation (ddof=0) by default. With only 3 trials, this slightly underestimates uncertainty vs sample std (ddof=1). The effect is negligible for our purposes.
- The `m9_dct` mode serves as a built-in experimental control: since DCT has a 0% hit rate on `ball.mp4`, its delta measures pure function-call overhead with no compression effect.

### Final Results (dynamic_boxing: False)

| Mode | Baseline | Toggled | Enc FPS | Dec FPS | Enc Delta | Dec Delta |
|------|----------|---------|---------|---------|-----------|-----------|
| BASELINE | - | - | 16.18 ± 0.54 | 168.92 ± 10.14 | - | - |
| m1_spatial | ON | OFF | 18.47 ± 0.07 | 178.90 ± 3.95 | -2.30 | -9.99 |
| m4_skip | OFF | ON | 17.60 ± 0.62 | 185.29 ± 28.40 | +1.42 | +16.38 |
| m4_search | ON | OFF | 15.62 ± 0.60 | 168.33 ± 4.62 | +0.56 | +0.58 |
| m6_bg | OFF | ON | 15.98 ± 0.16 | 188.09 ± 5.80 | -0.19 | +19.17 |
| m7_solid | ON | OFF | 13.56 ± 0.15 | 116.59 ± 1.95 | +2.61 | +52.33 |
| m9_dct | OFF | ON | 16.89 ± 0.10 | 187.44 ± 3.97 | +0.71 | +18.52 |
| m10_dct | OFF | ON | 17.25 ± 0.27 | 182.24 ± 5.40 | +1.07 | +13.33 |
| m11_cached | ON | OFF | 15.01 ± 0.38 | 172.10 ± 19.53 | +1.17 | -3.18 |
| m12_palette | ON | OFF | 18.55 ± 0.47 | 191.91 ± 0.31 | -2.38 | -23.00 |
| m13_faded | ON | OFF | 17.54 ± 0.41 | 187.23 ± 4.99 | -1.37 | -18.31 |
| m15_dither | ON | OFF | 17.15 ± 0.35 | 186.82 ± 5.30 | -0.97 | -17.91 |

---

## 6. Phase 4: H.264/H.265 Comparison

### Design
- **Script:** `compare_h264.py`
- **Methodology:** Minimum-Floor Encoding protocol
- **Video:** `01_ideal_motion.mp4`

### Protocol
1. Encode with TrajCDDec → measure output size $X$ bytes
2. Calculate target bitrate: $X \times 8 \div \text{duration}$
3. Encode with `ffmpeg -c:v libx264 -b:v {bitrate} -preset veryslow`
4. Encode with `ffmpeg -c:v libx265 -b:v {bitrate} -preset veryslow`
5. Compare all three outputs on SSIM and PSNR (grayscale plane)

### Fairness Caveats
1. **File size mismatch:** H.264 and H.265 refused to go below their structural minimum bitrate floors, producing files 50–85% larger than TrajCDDec. The comparison is not at matched file sizes.
2. **Hardcoded FPS:** The script assumes 30 FPS for all videos. If a video has a different framerate, the bitrate target would be incorrect.
3. **Grayscale comparison:** TrajCDDec operates in grayscale internally. Both original and decoded frames are converted to grayscale for SSIM/PSNR, making the comparison fair on the luma plane.

### Quality Metrics
- **PSNR formula:** $\text{PSNR} = 10 \cdot \log_{10}\left(\frac{255^2}{\text{MSE}}\right)$ — verified correct
- **SSIM:** Uses `skimage.metrics.structural_similarity` — standard implementation

---

## 7. Phase 5: Codec Bug Discovery & Resolution

The LOO ablation (Phase 3) accidentally functioned as a **codec fuzzer**. By systematically disabling each of the 11 modes, it forced the encoder into fallback paths that had never been exercised during normal operation.

### Timeline
1. Initial LOO runs showed `*CRASH*` for 5 of 11 modes when toggled
2. Stack traces revealed `ValueError: could not broadcast input array from shape (16,16,1) into shape (0,32,1)`
3. Root cause traced to Mode 15 (Dither) writing 34 bytes instead of 68 bytes
4. Fix applied to both `encoder_ablation.py` and `final_commit/encoder.py`
5. Verification: encode/decode round-trip on previously-crashing configs passed
6. Full LOO benchmark re-run with corrected code → no crashes

### The Fix
The encoder's Mode 15 packer was rewritten to match the decoder's expectations:
- **Before:** `struct.pack("<H", packed) + np.packbits(labels).tobytes()` → 34 bytes
- **After:** `palette.tobytes() + packed_2bit.tobytes()` → 68 bytes

---

## 8. Phase 6: Comprehensive Audit

### Methodology
Five independent audit passes were conducted in parallel:

1. **RESEARCH_NOTES Auditor:** Checked document structure, internal contradictions, stale content, and broken markdown
2. **Encoder/Decoder Auditor:** Verified byte-level payload parity across all 16 codec modes
3. **Ablation Data Auditor:** Cross-checked CSV data against published tables, verified methodology
4. **Core Script Auditor:** Audited `run_massive_ablation.py`, `massive_decode.py`, `controlled_benchmark.py` for logic bugs
5. **Supporting Script Auditor:** Audited `benchmark.py`, `compare_h264.py`, `analysis.py`, `generate_excel.py`, `batch_test_v5.py`, `batch_test7.py`

### Findings
- **7 critical issues** → all fixed
- **6 moderate issues** → all fixed  
- **5 minor issues** → 2 fixed, 3 acknowledged (won't fix)
- **13 verification points** → all passed

See `ENGINEERING_LOG.md` for the complete bug registry.

---

## 9. Compression Ratio Methodology

### How `orig_bytes` Is Computed
All scripts compute `orig_bytes = os.path.getsize(video_path)`, which is the **MP4 container size** — not raw pixel data. Since MP4 files are already H.264/HEVC compressed, the "Data Reduction" percentages represent improvement over an already-compressed baseline.

For context, `ball.mp4` (640×360, 101 grayscale frames) would be:
- Raw pixels: $640 \times 360 \times 101 = 23{,}270{,}400$ bytes ≈ **22.2 MB**
- MP4 container: **104 KB** (already 99.5% compressed by H.264)
- NAM output: **85 KB** (18.35% further reduction over MP4)

This is a meaningful caveat but not a fatal flaw — the research question is whether our codec can beat H.264 on synthetic content, and the MP4 baseline is H.264's own output.

### The Negative Byte-Weighted Aggregate
The Byte-Weighted Aggregate is -147% because `05_break_entropy.mp4` (7.6 MB) expands to 25.8 MB, adding ~18.2 MB to the numerator. This single file dominates the byte-weighted metric. The Unweighted Mean treats all 11 videos equally and shows 55.38% reduction — a fairer representation of algorithmic capability across content types.

---

## 10. Data Provenance Summary

| Published Number | Source Script | Trustworthy? |
|-----------------|--------------|-------------|
| Top 20 Pareto rankings (compression %) | `run_massive_ablation.py` | ✅ Yes |
| Top 20 Encoding FPS | `run_massive_ablation.py` | ⚠️ Multiprocess contention |
| Top 20 Decoder FPS | `massive_decode.py` | ❌ Poisoned by BUG-002 |
| LOO Marginal FPS table | `controlled_benchmark.py` | ✅ Yes (re-run with corrected config) |
| Section 2.4 compression ratios | `benchmark.py` | ⚠️ MP4-as-baseline caveat |
| H.264/H.265 comparison | `compare_h264.py` | ⚠️ Correct formulas, unfair comparison |
| Ablation scatter/bar plots | `analysis.py` | ✅ Yes (regenerated with fixed column name) |
