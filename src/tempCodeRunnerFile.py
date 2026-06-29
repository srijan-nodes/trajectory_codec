import cv2
import numpy as np
import struct
import zstandard as zstd


def encode_video(video_path, output_path, iframe_interval=60, QP=2):
    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read video")

    h, w, c = frame.shape
    cctx = zstd.ZstdCompressor(level=3)

    with open(output_path, "wb") as f:
        # HEADER
        f.write(b'NAM0')
        f.write(struct.pack("III", h, w, c))
        f.write(struct.pack("H", QP))

        prev = frame.astype(np.int16)
        frame_idx = 0

        while True:
            if frame_idx == 0:
                curr = frame
            else:
                ret, curr = cap.read()
                if not ret:
                    break

            curr_i = curr.astype(np.int16)

            # ---------- I-FRAME ----------
            if frame_idx == 0 or frame_idx % iframe_interval == 0:
                raw = curr.tobytes()
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

                # ---------- CONNECTED COMPONENTS ----------
                num_labels, labels = cv2.connectedComponents(mask)

                components = []

                for label in range(1, num_labels):
                    ys, xs = np.where(labels == label)
                    if len(ys) == 0:
                        continue

                    y_min, y_max = ys.min(), ys.max()
                    x_min, x_max = xs.min(), xs.max()

                    # noise filter
                    if (y_max - y_min) * (x_max - x_min) < 25:
                        continue

                    cropped = quantized[y_min:y_max+1, x_min:x_max+1]

                    # ---------- UINT8 PACK ----------
                    if cropped.min() >= -128 and cropped.max() <= 127:
                        packed = (cropped + 128).astype(np.uint8)
                        flag = 1
                        payload = packed.tobytes()
                    else:
                        flag = 2
                        payload = cropped.astype(np.int16).tobytes()

                    compressed = cctx.compress(payload)

                    components.append((flag, y_min, y_max, x_min, x_max, compressed))

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

                prev = curr_i

            frame_idx += 1

    cap.release()
    print(f"Encoding complete ({frame_idx} frames)")