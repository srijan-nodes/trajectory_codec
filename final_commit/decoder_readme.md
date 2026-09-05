# Decoder Technical Reference

The `decoder.py` module is responsible for reading the custom `.nam` format and translating it back into a sequence of RGB video frames. The decoder operates essentially as the inverse of the encoder, but is significantly faster because it performs zero algorithmic decision-making.

## The `decode()` Pipeline

1. **Header Parsing**: The decoder reads the first 16 bytes of the file, unpacking the standard `H D M P` magic signature, followed by the video dimensions (Width, Height), framerate (FPS), and total Frame Count.
2. **State Management**: The decoder initializes a blank canvas (`prev`) that will hold the reconstructed state of the previous frame.
3. **Block Reconstruction**: For every block in the grid, the decoder reads exactly 1 byte to determine the Mode. Based on that mode, it reads a strict, predetermined number of bytes (the payload) and updates the corresponding 16x16 pixel area on the canvas.

## Supported Modes & Payloads

- **Mode 1 (Raw I-Frame)**: Reads exactly 768 bytes (16x16x3). The payload is directly applied as raw RGB pixels.
- **Mode 4 (Perfect Cache)**: Reads 0 additional bytes. The decoder simply copies the 16x16 block from the exact same position in the `prev` frame canvas.
- **Mode 6 (Low Freq DCT)**: Reads 128 bytes (8x8 half-precision floats). Applies a 2D Inverse Discrete Cosine Transform (IDCT) and scales the result back up to 16x16 using cubic interpolation.
- **Mode 7 (High Freq DCT)**: Reads 512 bytes (16x16 half-precision floats). Applies a 16x16 Inverse DCT.
- **Mode 11 (Cached Motion)**: Reads 0 additional bytes. It relies on the *globally persistent* `cached_dy` and `cached_dx` variables that were set by the last successful motion block (like Mode 9 or 10), and shifts the block accordingly.
- **Mode 13 / 14 (Luma Shifts)**: Reads a 1-byte integer offset. It takes the cached or motion-predicted block and simply adds/subtracts the offset to all channels to perfectly simulate lighting changes.
- **Mode 15 (Forced Dither)**: Reads 64 bytes (the 2-bit quantization indices) plus a 1-byte baseline and 1-byte span limit. It rapidly scales the values to 4 static luma bins, achieving high-speed posterization.

## Performance

The decoding process is incredibly fast (averaging 140+ FPS) because all expensive search functions (like `cv2.matchTemplate`) are entirely handled during encoding. The decoder only performs direct memory copies and simple arithmetic shifts.
