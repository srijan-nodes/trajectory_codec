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

        while True:
            type_byte = f.read(1)
            if not type_byte:
                break 

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
                    try:
                        flag = struct.unpack("B", f.read(1))[0]
                        y_min, y_max, x_min, x_max = struct.unpack("HHHH", f.read(8))
                        size = struct.unpack("I", f.read(4))[0]
                        raw_payload = f.read(size)
                    except struct.error:
                        break

                    h_box = y_max - y_min + 1
                    w_box = x_max - x_min + 1

                    if flag == 3:
                        if len(raw_payload) == 1:
                            val = struct.unpack("B", raw_payload)[0]
                            next_frame[y_min:y_max+1, x_min:x_max+1] = val
                            
                    # ---------- DECODE TYPE 4: TEMPORAL MOTION ----------
                    elif flag == 4:
                        if len(raw_payload) == 4:
                            dy, dx = struct.unpack("hh", raw_payload)
                            src_y = y_min - dy
                            src_x = x_min - dx
                            
                            # Safety clamp
                            src_y = max(0, min(i420_h - h_box, src_y))
                            src_x = max(0, min(w - w_box, src_x))
                            
                            # Cut and paste the entity from the previous frame!
                            next_frame[y_min:y_max+1, x_min:x_max+1] = prev[src_y:src_y+h_box, src_x:src_x+w_box]
                            
                    else:
                        try:
                            data = dctx.decompress(raw_payload)
                        except zstd.ZstdError:
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

            yield_frame = np.clip(prev, 0, 255).astype(np.uint8).reshape((i420_h, w))
            bgr_frame = cv2.cvtColor(yield_frame, cv2.COLOR_YUV2BGR_I420)
            yield bgr_frame