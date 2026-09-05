# Encoder Technical Reference

The `encoder.py` module is the heavy lifter of the V5 Hybrid Codec. Its primary job is to take raw `.mp4` video frames (in RGB), compute the differences between sequential frames, and pack that delta into a custom binary stream format (`.nam`).

## The `encode()` Pipeline

1. **Initialization**: We establish a `cv2.VideoCapture` object and extract video metadata (FPS, Width, Height, Frame Count). We then write a custom 16-byte header to the output `.nam` file.
2. **Color Space**: The codec operates natively in YUV space for luma calculations, but uses RGB for block matching in complex modes.
3. **Block Iteration**: Every frame is divided into a grid of `16x16` pixel blocks. The encoder evaluates each block sequentially.
4. **Fast Lane Filtering**: Before running expensive computations, we do a basic Sum of Absolute Differences (SAD) check against the `previous_frame`. If the SAD is extremely low (the block hasn't changed), we immediately encode it as Mode 4 (Perfect Cache).

## The Cascade Logic

If a block is not static, it enters the cascade:
- **Mode 11 (Cached Motion)**: Tests if the block shifted exactly by the same motion vector (`last_dy`, `last_dx`) as the previous block. Highly efficient for continuous camera pans.
- **Mode 13 (Luma Shift)**: Assesses if the block matches the cached position but is universally brighter or darker. Great for fade-to-black or flash frames.
- **Mode 6 / Mode 7 (DCT)**: Converts the `16x16` block into frequency data using the Discrete Cosine Transform. 
  - Mode 6 encodes a highly-compressed 8x8 sub-grid (low frequency).
  - Mode 7 encodes a more detailed 16x16 grid (high frequency).
  - We strictly enforce constraints on DCT to ensure it is only used on smooth gradients (like rendering shadows or spherical objects), preventing pixelation.
- **Mode 9 / Mode 10 (Global Search)**: Expands the motion vector search to the entire frame (expensive but highly accurate).
- **Mode 15 (Forced Dither)**: If all other modes fail to accurately or cheaply represent the block, we crush the 16x16 block down to 4 distinct luma levels. This acts as a safety net against entropy spikes.
- **Mode 1 (Raw I-Frame)**: A brutal, uncompressed safety fallback if literally nothing else is suitable.

## Dynamic Boxing and Artifact Prevention

The encoder features a sophisticated `dynamic_boxing` system. If a block significantly differs from its prediction (MSE > 60), we draw a "box" around the area and force adjacent blocks into high-fidelity modes. This specifically mitigates "wild tearing" or "stray line" artifacts that can occur at the edges of rapidly moving objects.

## Output Format

The bitstream is packed using Python's `struct` library. Each block is preceded by a single byte indicating its Mode, followed by a variable-length payload (from 0 bytes for a Perfect Cache, up to 768 bytes for Raw Mode).
