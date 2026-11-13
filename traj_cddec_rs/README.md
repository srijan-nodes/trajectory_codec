# Trajectory Video Codec (Rust Edition) - `traj_cddec_rs`

A high-performance, memory-safe, ultra-fast implementation of the **TrajCDDec V5/V6 Hybrid Video Codec** rewritten completely from the ground up in pure Rust.

---

## Highlights & Capabilities

- **Unified V5 / V6 Bitstream Support**: Full backward and forward bitstream compatibility with Python-generated `NAM_V5` and `NAM_V6` containers.
- **Blazing Fast Throughput**:
  - Decode speed reaching **400+ to 1200+ FPS** (zero-copy memory layout, precomputed 2-bit LUTs, vectorized block operations).
  - Multi-lane encoding cascade with fast early-outs and zero temporal drift.
- **Lossless & Near-Lossless Reconstruction**: Identical math, rate-distortion optimization, and unsharp micro-contrast filters to maintain $\text{PSNR} \ge 40-55\text{ dB}$.
- **Complete Suite of Commands**:
  - `encode`: Compress MP4 video to `.nam`.
  - `decode`: Decompress `.nam` bitstream to `.mp4` or memory.
  - `benchmark`: 1-click reproducibility benchmark measuring Orig Size, Comp Size, Reduction %, Encode FPS, Decode FPS, MSE, and PSNR.
  - `analyze`: Bitstream telemetry report breaking down headers and mode layouts.
  - `batch`: Run evaluation across an entire directory of test videos and output a formatted benchmark table.

---

## Architecture Overview

```
traj_cddec_rs/
├── Cargo.toml
├── .cargo/
│   └── config.toml           # Optimized linker configuration (rust-lld)
├── src/
│   ├── lib.rs                # Library interface exposing all submodules
│   ├── main.rs               # CLI entrypoint with Clap subcommands
│   ├── bitstream.rs          # Bitstream container parsing (NAM_V5, NAM_V6, Zstandard stream)
│   ├── modes.rs              # 16 Macroblock modes, 2-bit LUTs, and pure-Rust 16x16 2D DCT engine
│   ├── decoder.rs            # Stream Type 0 (Macroblock) & Stream Type 1 (Layered Trajectory) decoders
│   ├── encoder.rs            # Hybrid cascade decision tree, rate-distortion lambda tuning, dynamic boxing
│   ├── video_io.rs           # High-speed FFmpeg pipe integration for reading & writing video frames
│   └── metrics.rs            # Frame MSE, PSNR, and corpus benchmark aggregator
```

---

## 16-Mode Decision Cascade

The macroblock engine operates on $16 \times 16$ pixel blocks:
1. **Mode 0xFF (Skip Run)**: Strict gated source static and reconstruction error checks for $0$-byte identical blocks.
2. **Mode 0 (I-Frame Keyframe)**: Full raw grayscale keyframe emitted on first frame or scene changes ($\text{mean}(\Delta) > 35.0$).
3. **Mode 3 (Dynamic Box Wrapper)**: Merges adjacent blocks sharing mode 6, 4, or 11 into up to $8 \times 8$ block rectangles ($128 \times 128$ pixels).
4. **Mode 6 (Background Model Copy)**: $0$-payload copy from the running temporal background buffer.
5. **Mode 7 (Solid Fill)**: $1$-byte median intensity fill for smooth backgrounds ($<10$ intensity variance).
6. **Mode 12 (Palette Quantization)**: 4-color palette + 64-byte packed 2-bit index buffer ($69$ bytes total).
7. **Mode 11 (Cached Motion Vector)**: $0$-payload reuse of the previous block's motion displacement $(dy, dx)$.
8. **Mode 4 (Local Motion Search)**: Fast template matching search in a $\pm 16\text{ px}$ search window with SSD/SAD metric.
9. **Mode 13 (Faded Motion)**: Motion vector displacement combined with signed luma shift $(\Delta Y)$.
10. **Mode 15 (Fast Dithered Palette)**: 4-center quantization fallback for complex noise and fireworks.
11. **Mode 9 / 10 (Low / Mid DCT)**: Orthonormal 2D Discrete Cosine Transform with $4 \times 4$ and $8 \times 8$ coefficient quantization.
12. **Mode 1 (Spatial Residual)**: Sparse quantized differential residual $(\Delta / QP)$.
13. **Mode 2 (Raw Block)**: Uncompressed fallback for complex textures.

---

## CLI Usage

### 1. Encode a Video
```bash
cargo run --release -- encode --input test_vid/01_ideal_motion.mp4 --output test_vid/01_ideal_motion.nam
```

### 2. Decode a Video & Measure Fidelity
```bash
cargo run --release -- decode --input test_vid/01_ideal_motion.nam --output reconstructed.mp4 --orig test_vid/01_ideal_motion.mp4
```

### 3. Run a Benchmark on a Single Video
```bash
cargo run --release -- benchmark --video test_vid/01_ideal_motion.mp4
```

### 4. Run Batch Benchmark on Full Corpus
```bash
cargo run --release -- batch --folder test_vid
```

### 5. Inspect Bitstream Telemetry
```bash
cargo run --release -- analyze --input test_vid/ball.nam
```
