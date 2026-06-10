import cv2
import numpy as np
import struct
import zstandard as zstd
from tqdm import tqdm

def encode_video(video_path, output_path, iframe_interval=300, QP=8, BIAS_BLOB_THRESHOLD=12.0):
    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read video")

    h, w, _ = frame.shape
    # I420 format requires dimensions to be perfectly divisible by 2
    h = h - (h % 2)
    w = w - (w % 2)
    
    cctx = zstd.ZstdCompressor(level=3)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with open(output_path, "wb") as f:
        # HEADER (Upgraded to NAM1 for YUV 4:2:0)
        f.write(b'NAM1')
        f.write(struct.pack("III", h, w, 1)) # Channels is now effectively 1
        f.write(struct.pack("H", QP))

        # Convert first frame to I420 Flat Array
        curr_i420 = cv2.cvtColor(frame[:h, :w], cv2.COLOR_BGR2YUV_I420)
        prev = curr_i420.reshape((int(h * 1.5), w, 1)).astype(np.int16)
        
        frame_idx = 0
        pbar = tqdm(total=total_frames, desc="Encoding NAM1 (I420)", unit="frame")

        while True:
            if frame_idx > 0:
                ret, frame = cap.read()
                if not ret:
                    break
                    
            # Convert incoming frame to flat I420
            curr_i420 = cv2.cvtColor(frame[:h, :w], cv2.COLOR_BGR2YUV_I420)
            curr_i = curr_i420.reshape((int(h * 1.5), w, 1)).astype(np.int16)

            # ---------- I-FRAME ----------
            if frame_idx == 0 or frame_idx % iframe_interval == 0:
                raw = curr_i420.tobytes()
                compressed = cctx.compress(raw)

                f.write(struct.pack("B", 0))
                f.write(struct.pack("HHHH", 0, 0, 0, 0))
                f.write(struct.pack("I", len(compressed)))
                f.write(compressed)

                prev = curr_i

            else:
                delta = curr_i - prev

                # ---------- QUANTIZATION ----------
                quantized = np.round(delta / QP).astype(np.int16)

                # ---------- MASK + DILATION ----------
                mask = np.any(quantized != 0, axis=2).astype(np.uint8)
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)

                # ---> THE C++ SPEED HACK <---
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

                components = []
                next_prev = prev.copy()  # Prevents Encoder-Decoder Drift

                for label in range(1, num_labels):
                    x_min = stats[label, cv2.CC_STAT_LEFT]
                    y_min = stats[label, cv2.CC_STAT_TOP]
                    w_box = stats[label, cv2.CC_STAT_WIDTH]
                    h_box = stats[label, cv2.CC_STAT_HEIGHT]
                    area  = stats[label, cv2.CC_STAT_AREA]

                    # Instant noise filter using pre-calculated area
                    if area < 25:
                        continue

                    y_max = y_min + h_box - 1
                    x_max = x_min + w_box - 1

                    cropped_curr = curr_i[y_min:y_max+1, x_min:x_max+1]
                    cropped_quant = quantized[y_min:y_max+1, x_min:x_max+1]

                    # ---------- DECISION ENGINE ----------
                    variance = np.var(cropped_curr, axis=(0, 1))
                    max_variance = np.max(variance)

                    if max_variance < BIAS_BLOB_THRESHOLD:
                        # Path A: Solid Color Blob (Now 1 byte)
                        flag = 3
                        mean_val = int(np.mean(cropped_curr))
                        compressed = struct.pack("B", mean_val) 
                        next_prev[y_min:y_max+1, x_min:x_max+1] = mean_val
                    else:
                        # Path B: Complex Texture
                        if cropped_quant.min() >= -128 and cropped_quant.max() <= 127:
                            packed = (cropped_quant + 128).astype(np.uint8)
                            flag = 1
                            payload = packed.tobytes()
                        else:
                            flag = 2
                            payload = cropped_quant.astype(np.int16).tobytes()

                        compressed = cctx.compress(payload)
                        recon_delta = cropped_quant * QP
                        next_prev[y_min:y_max+1, x_min:x_max+1] += recon_delta

                    components.append((flag, y_min, y_max, x_min, x_max, compressed))

                # ---------- BINARY STREAM WRITE ----------
                if len(components) == 0:
                    f.write(struct.pack("B", 2))  # empty frame
                else:
                    f.write(struct.pack("B", 1))  # delta frame
                    f.write(struct.pack("H", len(components)))

                    for flag, y_min, y_max, x_min, x_max, compressed in components:
                        f.write(struct.pack("B", flag))
                        f.write(struct.pack("HHHH", y_min, y_max, x_min, x_max))
                        f.write(struct.pack("I", len(compressed)))
                        f.write(compressed)

                prev = next_prev

            frame_idx += 1
            pbar.update(1)
            
        pbar.close()
    cap.release()