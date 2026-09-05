# Trajectory Video Codec (V5 Hybrid)

Welcome to the Trajectory Video Codec project, a custom, high-performance hybrid video compression algorithm built from the ground up in Python. This codec is specifically designed to handle a variety of video inputs ranging from static text presentation and high-speed motion to complex, chaotic confetti patterns.

## Overview

Traditional video codecs (like H.264 or HEVC) are highly complex and optimized for general-purpose hardware. This project is an experimental custom codec implementation that utilizes a heavily cascaded, multi-mode block-matching algorithm (V5 Hybrid) to achieve excellent visual fidelity while aggressively compressing the payload.

The codec relies on a `16x16` block-based architecture, analyzing every block of every frame and routing it through a cascading decision tree to find the most optimal encoding mode (e.g., perfect cache, motion estimation, DCT compression, dither fallbacks).

## Key Components

The system is split into three main modules:
- **`main.py`**: The GUI and orchestrator. It provides an intuitive Tkinter interface with two tabs: one for manual single-file Encode/Decode, and one for running batch processing over an entire directory of test videos (which outputs a color-coded `.xlsx` diagnostic report).
- **`encoder.py`**: The brain of the compression algorithm. Contains the `HybridV5` mode cascade, block selection logic, Discrete Cosine Transform (DCT) computations, and bitstream packing. 
- **`decoder.py`**: The reconstructor. Reads the binary `.nam` bitstream, unpacks the payloads, and perfectly rebuilds the frame state using the inverse operations of the encoder.

## The Algorithm: Hybrid V5 Cascade

The V5 Encoder uses a strict cascading logic to determine the cheapest way to store a `16x16` block:
1. **Mode 4 (Perfect Cache)**: If the block hasn't changed from the previous frame, store it with just 2 bytes.
2. **Mode 11 (Cached Motion)**: If the block moved predictably based on the last known motion vector, store it cheaply.
3. **Mode 13 (Luma Shift)**: If the block is undergoing a fade (brightness change), apply a flat luma offset to the cached block.
4. **Mode 6/7 (Low/High Frequency DCT)**: If the block is new but smooth or easily compressible, convert it to the frequency domain using DCT and quantize it.
5. **Mode 15 (Forced Dither)**: If the block is highly complex (e.g., confetti) and all other modes fail, fall back to a highly compressed 4-level posterized representation to save bandwidth.

## Testing and Results

We extensively tested this codec on a comprehensive suite of edge-case videos:
- `01_ideal_motion.mp4` (Smooth moving objects)
- `02_ideal_cache.mp4` (Static backgrounds)
- `03_ideal_palette.mp4` (Rich colors)
- `04_ideal_baseline.mp4` (Standard video)
- `05_break_entropy.mp4` (Pure noise/confetti)
- `07_break_fade.mp4` (Gradual brightness fades)

### Favorable Results
Our custom V5 algorithm achieved fantastic results across our testing suite:
- **Peak Encode FPS**: Achieved blazing fast encode speeds peaking at **90+ FPS** on predictable sequences (like `ideal_motion`), successfully skipping unnecessary calculations.
- **Peak Decode FPS**: The decoder is extremely lightweight, regularly exceeding **140+ FPS** during playback reconstruction.
- **Visual Fidelity**: By dynamically tuning the `dynamic_boxing` constraints, we successfully eliminated tearing artifacts ("wild extra lines") and achieved visually lossless performance on static elements and structured motion.
- **Robustness**: The encoder gracefully handles chaotic inputs (like `break_entropy.mp4`) without crashing, heavily utilizing the Mode 15 Dither fallback to maintain a playable framerate.

## Repository Structure

```
final_commit/
│
├── main.py             # GUI Application & Batch Orchestrator
├── encoder.py          # Core encoding logic & bitstream packing
├── decoder.py          # Core decoding logic & frame reconstruction
│
├── README.md           # Project overview (You are here)
├── encoder_readme.md   # Deep-dive into the encoding pipeline
└── decoder_readme.md   # Deep-dive into the decoding architecture
```

For more in-depth technical details on the specific bit-packing and mode structures, please see `encoder_readme.md` and `decoder_readme.md`.
