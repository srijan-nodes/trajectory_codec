import cv2
import numpy as np
import struct
import zstandard as zstd
from tqdm import tqdm

def encode_video(video_path, output_path, iframe_interval=300, QP=15, BIAS_BLOB_THRESHOLD=12.0, MOTION_SEARCH_RANGE=15, MOTION_MSE_THRESHOLD=50.0, MIN_AREA=150, CACHE_MSE_THRESHOLD=5.0):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read video")

    # Crop to multiples of 16 for Macroblock processing
    h, w, _ = frame.shape
    h = h - (h % 16)
    w = w - (w % 16)
    
    cctx = zstd.ZstdCompressor(level=3)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    stats = {"I-Frames": 0, "Type 0 (Skip)": 0, "Type 1/2 (Zstd Delta)": 0, "Type 3 (Vector Blob)": 0, "Type 4 (Motion)": 0, "Type 5 (Entity Cache)": 0}

    with open(output_path, "wb") as f:
        f.write(b'NAM1')
        f.write(struct.pack("III", h, w, 1)) 
        f.write(struct.pack("H", QP))

        # We will keep Y, U, V separated instead of flattened
        def get_yuv(bgr_frame):
            yuv = cv2.cvtColor(bgr_frame[:h, :w], cv2.COLOR_BGR2YUV_I420)
            y = yuv[:h, :].astype(np.int16)
            u = yuv[h:h+(h//4), :].reshape((h//2, w//2)).astype(np.int16)
            v = yuv[h+(h//4):, :].reshape((h//2, w//2)).astype(np.int16)
            return y, u, v

        prev_y, prev_u, prev_v = get_yuv(frame)
        
        # Long term memory cache for 16x16 macroblocks
        entity_cache = []
        
        frame_idx = 0
        pbar = tqdm(total=total_frames, desc="Encoding NAM1 (MB)", unit="frame")

        while True:
            if frame_idx > 0:
                ret, frame = cap.read()
                if not ret:
                    break
                    
            curr_y, curr_u, curr_v = get_yuv(frame)

            if frame_idx == 0 or frame_idx % iframe_interval == 0:
                raw_y = curr_y.astype(np.uint8).tobytes()
                raw_u = curr_u.astype(np.uint8).tobytes()
                raw_v = curr_v.astype(np.uint8).tobytes()
                compressed = cctx.compress(raw_y + raw_u + raw_v)

                f.write(struct.pack("B", 0))
                f.write(struct.pack("I", len(compressed)))
                f.write(compressed)

                prev_y, prev_u, prev_v = curr_y.copy(), curr_u.copy(), curr_v.copy()
                stats["I-Frames"] += 1

            else:
                f.write(struct.pack("B", 1)) # P-Frame
                next_prev_y = prev_y.copy()
                next_prev_u = prev_u.copy()
                next_prev_v = prev_v.copy()

                components = []

                # Iterate over Macroblocks
                for y_mb in range(0, h, 16):
                    for x_mb in range(0, w, 16):
                        y_c, x_c = y_mb // 2, x_mb // 2

                        curr_y_mb = curr_y[y_mb:y_mb+16, x_mb:x_mb+16]
                        curr_u_mb = curr_u[y_c:y_c+8, x_c:x_c+8]
                        curr_v_mb = curr_v[y_c:y_c+8, x_c:x_c+8]

                        prev_y_mb = prev_y[y_mb:y_mb+16, x_mb:x_mb+16]
                        prev_u_mb = prev_u[y_c:y_c+8, x_c:x_c+8]
                        prev_v_mb = prev_v[y_c:y_c+8, x_c:x_c+8]

                        # 1. Check for Skip
                        mse_y = np.mean((curr_y_mb - prev_y_mb)**2)
                        if mse_y < 2.0: # Skip threshold
                            components.append((0, b""))
                            stats["Type 0 (Skip)"] += 1
                            continue

                        # 2. Check Entity Cache
                        matched_idx = -1
                        for idx, (cy, cu, cv) in enumerate(entity_cache):
                            if np.mean((cy - curr_y_mb)**2) < CACHE_MSE_THRESHOLD:
                                matched_idx = idx
                                break

                        if matched_idx != -1:
                            components.append((5, struct.pack("B", matched_idx)))
                            cy, cu, cv = entity_cache[matched_idx]
                            next_prev_y[y_mb:y_mb+16, x_mb:x_mb+16] = cy
                            next_prev_u[y_c:y_c+8, x_c:x_c+8] = cu
                            next_prev_v[y_c:y_c+8, x_c:x_c+8] = cv
                            stats["Type 5 (Entity Cache)"] += 1
                            continue

                        # 3. Motion Search
                        search_y_min = max(0, y_mb - MOTION_SEARCH_RANGE)
                        search_y_max = min(h, y_mb + 16 + MOTION_SEARCH_RANGE)
                        search_x_min = max(0, x_mb - MOTION_SEARCH_RANGE)
                        search_x_max = min(w, x_mb + 16 + MOTION_SEARCH_RANGE)

                        search_window = prev_y[search_y_min:search_y_max, search_x_min:search_x_max]
                        if search_window.shape[0] >= 16 and search_window.shape[1] >= 16:
                            res = cv2.matchTemplate(search_window.astype(np.float32), curr_y_mb.astype(np.float32), cv2.TM_SQDIFF)
                            min_val, _, min_loc, _ = cv2.minMaxLoc(res)
                            mse = min_val / 256.0

                            if mse < MOTION_MSE_THRESHOLD:
                                best_prev_x = search_x_min + min_loc[0]
                                best_prev_y = search_y_min + min_loc[1]
                                
                                dy = y_mb - best_prev_y
                                dx = x_mb - best_prev_x
                                
                                # Apply to Y
                                recon_y = prev_y[best_prev_y:best_prev_y+16, best_prev_x:best_prev_x+16].copy()
                                next_prev_y[y_mb:y_mb+16, x_mb:x_mb+16] = recon_y

                                # Apply to U, V
                                dy_c, dx_c = dy // 2, dx // 2
                                best_prev_yc = max(0, min(h//2 - 8, y_c - dy_c))
                                best_prev_xc = max(0, min(w//2 - 8, x_c - dx_c))

                                recon_u = prev_u[best_prev_yc:best_prev_yc+8, best_prev_xc:best_prev_xc+8].copy()
                                recon_v = prev_v[best_prev_yc:best_prev_yc+8, best_prev_xc:best_prev_xc+8].copy()
                                
                                next_prev_u[y_c:y_c+8, x_c:x_c+8] = recon_u
                                next_prev_v[y_c:y_c+8, x_c:x_c+8] = recon_v

                                entity_cache.append((recon_y, recon_u, recon_v))
                                if len(entity_cache) > 255:
                                    entity_cache.pop(0)

                                components.append((4, struct.pack("hh", dy, dx)))
                                stats["Type 4 (Motion)"] += 1
                                continue

                        # 4. Residual (Zstd)
                        delta_y = curr_y_mb - prev_y_mb
                        delta_u = curr_u_mb - prev_u_mb
                        delta_v = curr_v_mb - prev_v_mb
                        
                        qy = np.round(delta_y / QP).astype(np.int16)
                        qu = np.round(delta_u / QP).astype(np.int16)
                        qv = np.round(delta_v / QP).astype(np.int16)

                        next_prev_y[y_mb:y_mb+16, x_mb:x_mb+16] = prev_y_mb + qy * QP
                        next_prev_u[y_c:y_c+8, x_c:x_c+8] = prev_u_mb + qu * QP
                        next_prev_v[y_c:y_c+8, x_c:x_c+8] = prev_v_mb + qv * QP

                        recon_y = next_prev_y[y_mb:y_mb+16, x_mb:x_mb+16].copy()
                        recon_u = next_prev_u[y_c:y_c+8, x_c:x_c+8].copy()
                        recon_v = next_prev_v[y_c:y_c+8, x_c:x_c+8].copy()

                        entity_cache.append((recon_y, recon_u, recon_v))
                        if len(entity_cache) > 255:
                            entity_cache.pop(0)

                        payload = qy.tobytes() + qu.tobytes() + qv.tobytes()
                        compressed = cctx.compress(payload)
                        components.append((2, compressed))
                        stats["Type 1/2 (Zstd Delta)"] += 1

                for flag, payload in components:
                    f.write(struct.pack("B", flag))
                    if flag != 0:
                        f.write(struct.pack("I", len(payload)))
                        f.write(payload)

                prev_y, prev_u, prev_v = next_prev_y, next_prev_u, next_prev_v

            frame_idx += 1
            pbar.update(1)
            
        pbar.close()
        
    cap.release()
    
    print("\n" + "="*30)
    print("ENCODER TELEMETRY")
    print("="*30)
    for key, val in stats.items():
        print(f"- {key}: {val} packets")
    print("="*30 + "\n")