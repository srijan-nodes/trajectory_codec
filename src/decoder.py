import numpy as np
import struct
import zstandard as zstd
import cv2

def decode_video_stream(encoded_path):
    dctx = zstd.ZstdDecompressor()

    with open(encoded_path, "rb") as f:
        magic = f.read(4)
        if magic != b'NAM1':
            raise ValueError("Invalid file. Expected NAM1 (I420 YUV Format).")

        h, w, _ = struct.unpack("III", f.read(12))
        i420_h = int(h * 1.5)
        shape = (i420_h, w, 1)

        QP = struct.unpack("H", f.read(2))[0]
        prev = None
        frame_idx = 0

        while True:
            type_byte = f.read(1)
            if not type_byte:
                break # Clean EOF

            frame_type = struct.unpack("B", type_byte)[0]

            if frame_type == 0:
                f.read(8)
                size = struct.unpack("I", f.read(4))[0]
                data = dctx.decompress(f.read(size))
                frame = np.frombuffer(data, dtype=np.uint8).reshape(shape)
                prev = frame.astype(np.int16)

            elif frame_type == 1:
                num_components = struct.unpack("H", f.read(2))[0]
                next_frame = prev.copy()

                for _ in range(num_components):
                    # --- RESILIENCE LAYER 1: Catch Stream Misalignment ---
                    try:
                        flag = struct.unpack("B", f.read(1))[0]
                        y_min, y_max, x_min, x_max = struct.unpack("HHHH", f.read(8))
                        size = struct.unpack("I", f.read(4))[0]
                        raw_payload = f.read(size)
                    except struct.error:
                        # If the stream is broken or we hit EOF mid-component, stop processing this frame
                        break

                    h_box = y_max - y_min + 1
                    w_box = x_max - x_min + 1

                    if flag == 3:
                        if len(raw_payload) == 1:
                            val = struct.unpack("B", raw_payload)[0]
                            next_frame[y_min:y_max+1, x_min:x_max+1] = val
                    else:
                        # --- RESILIENCE LAYER 2: Catch Zstd Corruption ---
                        try:
                            data = dctx.decompress(raw_payload)
                        except zstd.ZstdError:
                            # Drop the corrupted block and continue processing the rest of the frame
                            continue 

                        if flag == 1:
                            packed = np.frombuffer(data, dtype=np.uint8).reshape((h_box, w_box, 1))
                            quantized = packed.astype(np.int16) - 128
                        else:
                            quantized = np.frombuffer(data, dtype=np.int16).reshape((h_box, w_box, 1))

                        delta = quantized * QP
                        next_frame[y_min:y_max+1, x_min:x_max+1] += delta

                prev = next_frame

            elif frame_type == 2:
                pass

            # --- YIELD BGR FRAME ---
            yield_frame = np.clip(prev, 0, 255).astype(np.uint8).reshape((i420_h, w))
            bgr_frame = cv2.cvtColor(yield_frame, cv2.COLOR_YUV2BGR_I420)
            yield bgr_frame
            frame_idx += 1