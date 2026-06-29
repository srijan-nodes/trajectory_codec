import numpy as np
import struct
import zstandard as zstd
import cv2

def decode_video_stream(encoded_path):
    dctx = zstd.ZstdDecompressor()

    with open(encoded_path, "rb") as f:
        magic = f.read(4)
        if magic != b'NAM1':
            raise ValueError("Invalid file. Expected NAM1 format.")

        h, w, _ = struct.unpack("III", f.read(12))
        QP = struct.unpack("H", f.read(2))[0]
        
        # Dimensions
        y_shape = (h, w)
        uv_shape = (h//2, w//2)

        prev_y = np.zeros(y_shape, dtype=np.int16)
        prev_u = np.zeros(uv_shape, dtype=np.int16)
        prev_v = np.zeros(uv_shape, dtype=np.int16)
        
        entity_cache = []

        while True:
            type_byte = f.read(1)
            if not type_byte:
                break 

            frame_type = struct.unpack("B", type_byte)[0]

            if frame_type == 0:
                size = struct.unpack("I", f.read(4))[0]
                data = dctx.decompress(f.read(size))
                
                y_size = h * w
                uv_size = (h//2) * (w//2)
                
                prev_y = np.frombuffer(data[:y_size], dtype=np.uint8).reshape(y_shape).astype(np.int16)
                prev_u = np.frombuffer(data[y_size:y_size+uv_size], dtype=np.uint8).reshape(uv_shape).astype(np.int16)
                prev_v = np.frombuffer(data[y_size+uv_size:], dtype=np.uint8).reshape(uv_shape).astype(np.int16)

            elif frame_type == 1:
                next_y = prev_y.copy()
                next_u = prev_u.copy()
                next_v = prev_v.copy()

                for y_mb in range(0, h, 16):
                    for x_mb in range(0, w, 16):
                        y_c, x_c = y_mb // 2, x_mb // 2

                        flag = struct.unpack("B", f.read(1))[0]

                        if flag == 0:
                            continue
                        
                        elif flag == 5:
                            size = struct.unpack("I", f.read(4))[0]
                            idx = struct.unpack("B", f.read(size))[0]
                            cy, cu, cv = entity_cache[idx]
                            
                            next_y[y_mb:y_mb+16, x_mb:x_mb+16] = cy
                            next_u[y_c:y_c+8, x_c:x_c+8] = cu
                            next_v[y_c:y_c+8, x_c:x_c+8] = cv
                            
                        elif flag == 4:
                            size = struct.unpack("I", f.read(4))[0]
                            dy, dx = struct.unpack("hh", f.read(size))
                            
                            src_y = y_mb - dy
                            src_x = x_mb - dx
                            cy = prev_y[src_y:src_y+16, src_x:src_x+16].copy()
                            
                            src_yc, src_xc = src_y // 2, src_x // 2
                            cu = prev_u[src_yc:src_yc+8, src_xc:src_xc+8].copy()
                            cv = prev_v[src_yc:src_yc+8, src_xc:src_xc+8].copy()
                            
                            next_y[y_mb:y_mb+16, x_mb:x_mb+16] = cy
                            next_u[y_c:y_c+8, x_c:x_c+8] = cu
                            next_v[y_c:y_c+8, x_c:x_c+8] = cv

                            entity_cache.append((cy, cu, cv))
                            if len(entity_cache) > 255:
                                entity_cache.pop(0)

                        elif flag == 2:
                            size = struct.unpack("I", f.read(4))[0]
                            data = dctx.decompress(f.read(size))
                            
                            qy = np.frombuffer(data[:512], dtype=np.int16).reshape((16, 16))
                            qu = np.frombuffer(data[512:640], dtype=np.int16).reshape((8, 8))
                            qv = np.frombuffer(data[640:768], dtype=np.int16).reshape((8, 8))
                            
                            cy = prev_y[y_mb:y_mb+16, x_mb:x_mb+16] + qy * QP
                            cu = prev_u[y_c:y_c+8, x_c:x_c+8] + qu * QP
                            cv = prev_v[y_c:y_c+8, x_c:x_c+8] + qv * QP
                            
                            next_y[y_mb:y_mb+16, x_mb:x_mb+16] = cy
                            next_u[y_c:y_c+8, x_c:x_c+8] = cu
                            next_v[y_c:y_c+8, x_c:x_c+8] = cv

                            entity_cache.append((cy.copy(), cu.copy(), cv.copy()))
                            if len(entity_cache) > 255:
                                entity_cache.pop(0)

                prev_y, prev_u, prev_v = next_y, next_u, next_v

            # Merge Y, U, V back into YUV I420 format and yield BGR frame
            yuv = np.zeros((int(h * 1.5), w), dtype=np.uint8)
            yuv[:h, :] = np.clip(prev_y, 0, 255).astype(np.uint8)
            
            u_flat = np.clip(prev_u, 0, 255).astype(np.uint8).flatten()
            v_flat = np.clip(prev_v, 0, 255).astype(np.uint8).flatten()
            
            u_rows = h // 4
            yuv[h:h+u_rows, :] = u_flat.reshape((u_rows, w))
            yuv[h+u_rows:, :] = v_flat.reshape((u_rows, w))

            bgr_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            yield bgr_frame