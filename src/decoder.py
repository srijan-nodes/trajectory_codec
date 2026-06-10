import numpy as np
import struct
import zstandard as zstd

def decode_video_stream(encoded_path):
    dctx = zstd.ZstdDecompressor()

    with open(encoded_path, "rb") as f:
        magic = f.read(4)
        if magic != b'NAM0':
            raise ValueError("Invalid file")

        h, w, c = struct.unpack("III", f.read(12))
        shape = (h, w, c)

        QP = struct.unpack("H", f.read(2))[0]

        prev = None

        while True:
            type_byte = f.read(1)
            if not type_byte:
                break

            frame_type = struct.unpack("B", type_byte)[0]

            # ---------- I-FRAME ----------
            if frame_type == 0:
                f.read(8)
                size = struct.unpack("I", f.read(4))[0]
                data = dctx.decompress(f.read(size))

                frame = np.frombuffer(data, dtype=np.uint8).reshape(shape)
                prev = frame.astype(np.int16)

                yield frame

            # ---------- DELTA / HYBRID ----------
            elif frame_type == 1:
                num_components = struct.unpack("H", f.read(2))[0]

                next_frame = prev.copy()

                for _ in range(num_components):
                    flag = struct.unpack("B", f.read(1))[0]
                    y_min, y_max, x_min, x_max = struct.unpack("HHHH", f.read(8))
                    size = struct.unpack("I", f.read(4))[0]
                    
                    raw_payload = f.read(size)

                    h_box = y_max - y_min + 1
                    w_box = x_max - x_min + 1

                    if flag == 3:
                        # Solid Vector Blob (Decision Engine)
                        b, g, r = struct.unpack("BBB", raw_payload)
                        next_frame[y_min:y_max+1, x_min:x_max+1] = [b, g, r]
                    else:
                        # Standard Zstd Delta
                        data = dctx.decompress(raw_payload)
                        
                        if flag == 1:
                            packed = np.frombuffer(data, dtype=np.uint8).reshape((h_box, w_box, 3))
                            quantized = packed.astype(np.int16) - 128
                        else:
                            quantized = np.frombuffer(data, dtype=np.int16).reshape((h_box, w_box, 3))

                        delta = quantized * QP
                        next_frame[y_min:y_max+1, x_min:x_max+1] += delta

                frame = np.clip(next_frame, 0, 255).astype(np.uint8)
                prev = next_frame

                yield frame

            # ---------- EMPTY ----------
            elif frame_type == 2:
                yield np.clip(prev, 0, 255).astype(np.uint8)