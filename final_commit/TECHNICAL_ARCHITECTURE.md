# Technical Architecture: TrajCDDec V5 Codec

This document is a complete technical reference for the TrajCDDec V5 video codec. It covers the bitstream format, all 16 encoding modes, the encoder/decoder pipelines, and the configuration system. A developer could reimplement the codec from this document alone.

---

## 1. Bitstream Format (NAM_V5)

### 1.1 File Header (21 bytes, uncompressed)

| Offset | Size | Format | Field | Description |
|--------|------|--------|-------|-------------|
| 0 | 6 | ASCII | Magic | `b'NAM_V5'` — format identifier |
| 6 | 4 | `<I` | Height | Frame height in pixels |
| 10 | 4 | `<I` | Width | Frame width in pixels |
| 14 | 4 | `<I` | Total Frames | Number of frames in the video |
| 18 | 2 | `<H` | QP | Quantization parameter (used by Mode 1) |
| 20 | 1 | `<B` | has_bg | Background model flag (0 or 1) |

After the 21-byte header, the remainder of the file is **Zstandard compressed**. The decoder wraps the file handle in `zstd.ZstdDecompressor().stream_reader(f)` and reads from the decompressed stream.

### 1.2 Frame Structure

Each frame begins with a 1-byte type indicator:

```
<B> frame_type:
  0 = I-frame (keyframe)
  1 = P-frame (predicted)
```

#### I-frame (type 0)
| Size | Format | Field |
|------|--------|-------|
| 4 | `<I` | Payload size (= H × W bytes) |
| H×W | raw | Grayscale pixel data (uint8) |

#### P-frame (type 1)
| Size | Format | Field |
|------|--------|-------|
| 4 | `<I` | Length of mode_data array |
| 4 | `<I` | Length of payload_data array |
| var | raw | mode_data — one byte per block (mode flags) |
| var | raw | payload_data — concatenated payloads for all blocks |

The mode_data and payload_data are separated to allow the decoder to index into modes independently from payloads. The decoder iterates over the 16×16 block grid in raster order (top-to-bottom, left-to-right), reading one mode byte per block and consuming the corresponding payload bytes.

---

## 2. Complete Mode Reference

### Mode 0xFF — Skip Run
| Property | Value |
|----------|-------|
| **Trigger** | `base_sad / 256 < 1.0` (block is identical to previous frame) |
| **Payload** | `<H` skip_count (2 bytes) |
| **Decoder** | Block remains unchanged from `prev` frame. Counter decrements for subsequent blocks. |
| **Complexity** | $O(1)$ per skipped block |

The encoder accumulates consecutive skip blocks into a run-length count. The decoder reads the count and skips that many blocks without reading any further payload.

---

### Mode 0 — I-frame (Keyframe)
| Property | Value |
|----------|-------|
| **Trigger** | First frame, or global scene change (`MAE > 25.0`) |
| **Payload** | `<I` size + raw grayscale pixels (H×W bytes) |
| **Decoder** | Direct copy to frame buffer |
| **Complexity** | $O(H \times W)$ |

---

### Mode 1 — Spatial Residual Quantization
| Property | Value |
|----------|-------|
| **Trigger** | All predictive modes fail; `nnz < 128` |
| **Payload (sparse)** | `<B` nnz + nnz index bytes + nnz value bytes = `1 + 2×nnz` bytes |
| **Payload (dense)** | Never emitted (encoder gates `nnz < 128`) |
| **Decoder** | `prev_block + (quantized * QP)` clamped to [0, 255] |
| **Complexity** | $O(N^2)$ where N=16 |

The encoder computes `delta = round((curr - prev) / QP)`, finds non-zero positions, and serializes indices and values (biased by +128 for uint8 storage). The decoder reverses: `values = uint8_values - 128`, then `reconstructed = prev + delta * QP`.

**Note:** The decoder has a dead-code branch for `nnz >= 128` (dense mode) that reads 256 bytes. The encoder never emits this because it falls through to Raw mode (Mode 2) for dense blocks.

---

### Mode 2 — Raw Block
| Property | Value |
|----------|-------|
| **Trigger** | All other modes fail or exceed cost threshold |
| **Payload** | 256 bytes (16×16 grayscale pixels, uint8) |
| **Decoder** | Direct copy to frame buffer |
| **Complexity** | $O(N^2)$ |

The ultimate fallback — writes the block's raw pixel values with no compression.

---

### Mode 3 — Dynamic Box Wrapper
| Property | Value |
|----------|-------|
| **Trigger** | Adjacent blocks share the same optimal mode |
| **Payload** | `<BBB` w_mult, h_mult, inner_mode (3 bytes) + inner mode's payload |
| **Decoder** | Reads multipliers, then dispatches to the inner mode's handler with enlarged block dimensions |
| **Complexity** | Varies by inner mode |

Dynamic boxing merges a rectangle of `w_mult × h_mult` adjacent 16×16 blocks into a single larger block. The inner mode (4, 6, 7, or 11) operates on the merged region. This reduces mode flag overhead and improves decode speed.

**Encoder constraint:** Only modes 4, 6, 7, and 11 are wrapped in dynamic boxes. Modes with variable or complex payloads (1, 2, 9, 10, 12, 13, 15) are not boxed.

---

### Mode 4 — Local Motion Search
| Property | Value |
|----------|-------|
| **Trigger** | `cv2.matchTemplate` finds a match with `MSE < 150.0` in a ±16px window |
| **Payload** | `<hh` dy, dx (4 bytes, signed int16) |
| **Decoder** | `prev[y-dy : y-dy+16, x-dx : x-dx+16]` copied to current block |
| **Complexity** | $O(K \log K)$ via FFT-accelerated template matching |

Updates `last_dy, last_dx` for Mode 11 (cached motion).

---

### Mode 6 — Background Model Copy
| Property | Value |
|----------|-------|
| **Trigger** | Block matches the running background model |
| **Payload** | 0 bytes (empty) |
| **Decoder** | `bg_model[y:y+16, x:x+16]` copied to current block |
| **Complexity** | $O(N^2)$ |

---

### Mode 7 — Solid Fill
| Property | Value |
|----------|-------|
| **Trigger** | Block intensity range `max - min < threshold` |
| **Payload** | `<B` val (1 byte — the median intensity value) |
| **Decoder** | `np.full((16, 16, 1), val)` |
| **Complexity** | $O(1)$ |

The most efficient mode — 1 byte represents 256 pixels.

---

### Mode 9 — Low-Frequency DCT (4×4)
| Property | Value |
|----------|-------|
| **Trigger** | Cost function selects DCT with 4×4 coefficient mask |
| **Payload** | 16 float16 coefficients = 32 bytes |
| **Decoder** | Zero-pad 4×4 coefficients into 16×16 matrix, `cv2.idct()` |
| **Complexity** | $O(N^2 \log N)$ |

---

### Mode 10 — Mid-Frequency DCT (8×8)
| Property | Value |
|----------|-------|
| **Trigger** | Cost function selects DCT with 8×8 coefficient mask |
| **Payload** | 64 float16 coefficients = 128 bytes |
| **Decoder** | Zero-pad 8×8 coefficients into 16×16 matrix, `cv2.idct()` |
| **Complexity** | $O(N^2 \log N)$ |

---

### Mode 11 — Cached Motion Vector
| Property | Value |
|----------|-------|
| **Trigger** | Block matches `prev[y-last_dy, x-last_dx]` with `SAD/256 < 4.0` and `MSE < 60.0` |
| **Payload** | 0 bytes (empty) |
| **Decoder** | Reuses `last_dy, last_dx` from the most recent Mode 4 or Mode 13 |
| **Complexity** | $O(N^2)$ |

---

### Mode 12 — Palette Quantization (≤4 colors)
| Property | Value |
|----------|-------|
| **Trigger** | K-means produces ≤4 clusters with acceptable SAD |
| **Payload** | `<B` num_colors (1) + 4-byte palette (uint8) + 64-byte packed indices = **69 bytes** |
| **Decoder** | Unpack 2-bit indices: `(byte >> 6) & 0x03`, etc. Map through palette. |
| **Complexity** | $O(N^2)$ |

---

### Mode 13 — Faded Motion (Luma Shift)
| Property | Value |
|----------|-------|
| **Trigger** | Motion-predicted block + global brightness shift matches with `MSE < 250.0` |
| **Payload** | `<hhb` dy, dx, luma_shift = **5 bytes** |
| **Decoder** | `np.clip(prev[y-dy:, x-dx:] + shift, 0, 255)` |
| **Complexity** | $O(N^2)$ |

Updates `last_dy, last_dx` for Mode 11.

---

### Mode 15 — Fast Dithered Palette (4 levels)
| Property | Value |
|----------|-------|
| **Trigger** | K-means with 4 centers produces acceptable dither quality |
| **Payload** | 4-byte palette (uint8) + 64-byte packed 2-bit indices = **68 bytes** |
| **Decoder** | Same unpacking as Mode 12 but without the `num_colors` prefix byte |
| **Complexity** | $O(N^2)$ |

Note: Mode 15 differs from Mode 12 only in the absence of the 1-byte `num_colors` header (always 4 colors assumed).

---

## 3. Encoder Pipeline

### 3.1 Pre-Processing
1. Read video frame via `cv2.VideoCapture`
2. Convert to grayscale: `cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)`
3. Reshape to `(H, W, 1)` as `int16` (allows negative residuals)

### 3.2 Scene Change Detection
If `np.mean(np.abs(curr_gray - prev_gray)) > 25.0`, emit an I-frame (Mode 0) and reset the background model.

### 3.3 Block Grid Iteration
For each 16×16 block in raster order (top-left to bottom-right):

```
1. Skip Check:     SAD < 1.0/pixel  → Mode 0xFF (Skip)
2. Cached Motion:  SAD < 4.0 and MSE < 60.0 at (last_dy, last_dx) → Mode 11
3. Solid Fill:     max-min < threshold → Mode 7
4. Motion Search:  matchTemplate MSE < 150.0 → Mode 4
5. Luma Shift:     shifted_MSE < 250.0 → Mode 13
6. DCT Low:        if m9_dct enabled → Mode 9
7. DCT Mid:        if m10_dct enabled → Mode 10
8. Palette:        if m12_palette enabled, ≤4 colors → Mode 12
9. Dither:         if m15_dither enabled → Mode 15
10. Spatial:       if m1_spatial enabled, nnz < 128 → Mode 1
11. Raw Fallback:  → Mode 2
```

### 3.4 Cost Function
All modes compete via a unified cost function:
$$\text{Cost} = \text{PayloadBytes} + \lambda \times \text{SAD}$$

Where $\lambda$ is a rate-distortion trade-off parameter. The mode with the lowest cost wins. Modes are evaluated lazily — the cascade terminates early if a zero-cost mode (Skip, Cached Motion) matches.

### 3.5 Dynamic Boxing
After the per-block mode decisions, the encoder attempts to merge adjacent blocks that share the same mode. It grows rectangles by expanding `w_mult` and `h_mult` (up to 8×8 blocks) as long as the merged region's mode produces acceptable quality. Merged blocks are emitted as Mode 3 wrappers.

---

## 4. Decoder Pipeline

### 4.1 Initialization
1. Read 21-byte header
2. Initialize Zstandard decompression stream
3. Set `prev = None`, `bg_model = None`

### 4.2 Frame Loop
For each frame:
1. Read 1-byte frame type
2. **I-frame:** Read raw pixels, initialize `prev` and `bg_model`
3. **P-frame:** Copy `prev` → `next_frame`, then:
   - Read `len_modes` and `len_payloads` (8 bytes)
   - Read the complete `mode_data` and `payload_data` arrays
   - Iterate block grid in raster order:
     - Check `processed[y, x]` — skip if already handled (by dynamic boxing)
     - Check `skip_remaining` — decrement and skip if in a skip run
     - Read mode byte from `mode_data`
     - If Mode 3: read `w_mult, h_mult, inner_mode` from payload
     - Dispatch to mode handler, consuming payload bytes
     - Mark `processed[y:y+h_mult, x:x+w_mult] = True`

### 4.3 Background Model Update
After each P-frame, if `has_bg`:
```python
diff_mask = |next_frame - prev| < 5
bg_model[diff_mask] = 0.95 * bg_model[diff_mask] + 0.05 * next_frame[diff_mask]
```

---

## 5. Configuration System

### 5.1 The 11 Ablatable Toggles

| Key | Default | Controls |
|-----|---------|----------|
| `m1_spatial` | True | Mode 1 (Spatial Residual) evaluation |
| `m4_skip` | False | Explicit skip detection (redundant with base check) |
| `m4_search` | True | Mode 4 (Motion Search via matchTemplate) |
| `m6_bg` | False | Mode 6 (Background Model Copy) |
| `m7_solid` | True | Mode 7 (Solid Fill) evaluation |
| `m9_dct` | False | Mode 9 (4×4 Low-Freq DCT) evaluation |
| `m10_dct` | False | Mode 10 (8×8 Mid-Freq DCT) evaluation |
| `m11_cached` | True | Mode 11 (Cached Motion Vector) evaluation |
| `m12_palette` | True | Mode 12 (Palette Quantization) evaluation |
| `m13_faded` | True | Mode 13 (Faded Motion / Luma Shift) evaluation |
| `m15_dither` | True | Mode 15 (Fast Dither) evaluation |

### 5.2 Fixed Settings

| Key | Value | Description |
|-----|-------|-------------|
| `dynamic_boxing` | True (production) / False (ablation) | Block merging optimization |
| `name` | String | Display name for tqdm progress bar |

### 5.3 Coarse vs Granular Config Keys
The `final_commit/encoder.py` uses coarse top-level keys (`background`, `palette`, `motion`, `dct`) that map to groups of modes. The `encoder_ablation.py` uses the granular per-mode toggles listed above. Only `encoder_ablation.py` supports the full $2^{11}$ ablation space.

---

## 6. Payload Size Summary

| Mode | Name | Payload Size | Notes |
|------|------|-------------|-------|
| 0xFF | Skip | 2 bytes | Run-length count |
| 0 | I-frame | 4 + H×W bytes | Header + raw pixels |
| 1 | Spatial (sparse) | 1 + 2×nnz bytes | Variable length |
| 2 | Raw | 256 bytes | Fixed 16×16 block |
| 3 | Dynamic Box | 3 + inner bytes | Wrapper |
| 4 | Motion | 4 bytes | (dy, dx) signed int16 |
| 6 | Background | 0 bytes | Zero payload |
| 7 | Solid | 1 byte | Single intensity value |
| 9 | DCT Low | 32 bytes | 16 float16 coefficients |
| 10 | DCT Mid | 128 bytes | 64 float16 coefficients |
| 11 | Cached Motion | 0 bytes | Zero payload |
| 12 | Palette | 69 bytes | 1B count + 4B palette + 64B indices |
| 13 | Faded | 5 bytes | (dy, dx, shift) |
| 15 | Dither | 68 bytes | 4B palette + 64B indices |
