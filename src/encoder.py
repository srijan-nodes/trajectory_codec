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

    h, w, _ = frame.shape
    h = h - (h % 2)
    w = w - (w % 2)
    
    cctx = zstd.ZstdCompressor(level=3)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    stats = {"I-Frames": 0, "Type 1/2 (Zstd Delta)": 0, "Type 3 (Vector Blob)": 0, "Type 4 (Motion)": 0, "Type 5 (Entity Cache)": 0}

    with open(output_path, "wb") as f:
        f.write(b'NAM1')
        f.write(struct.pack("III", h, w, 1)) 
        f.write(struct.pack("H", QP))

        curr_i420 = cv2.cvtColor(frame[:h, :w], cv2.COLOR_BGR2YUV_I420)
        prev = curr_i420.reshape((int(h * 1.5), w, 1)).astype(np.int16)
        
        # --- INITIALIZE LONG-TERM MEMORY ---
        entity_cache = []
        
        frame_idx = 0
        pbar = tqdm(total=total_frames, desc="Encoding NAM1 (Memory Cache)", unit="frame")

        while True:
            if frame_idx > 0:
                ret, frame = cap.read()
                if not ret:
                    break
                    
            curr_i420 = cv2.cvtColor(frame[:h, :w], cv2.COLOR_BGR2YUV_I420)
            curr_i = curr_i420.reshape((int(h * 1.5), w, 1)).astype(np.int16)

            if frame_idx == 0 or frame_idx % iframe_interval == 0:
                raw = curr_i420.tobytes()
                compressed = cctx.compress(raw)

                f.write(struct.pack("B", 0))
                f.write(struct.pack("HHHH", 0, 0, 0, 0))
                f.write(struct.pack("I", len(compressed)))
                f.write(compressed)

                prev = curr_i
                stats["I-Frames"] += 1

            else:
                delta = curr_i - prev
                quantized = np.round(delta / QP).astype(np.int16)

                mask = np.any(quantized != 0, axis=2).astype(np.uint8)
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)

                num_labels, labels, component_stats, centroids = cv2.connectedComponentsWithStats(mask)

                components = []
                next_prev = prev.copy()

                for label in range(1, num_labels):
                    x_min = component_stats[label, cv2.CC_STAT_LEFT]
                    y_min = component_stats[label, cv2.CC_STAT_TOP]
                    w_box = component_stats[label, cv2.CC_STAT_WIDTH]
                    h_box = component_stats[label, cv2.CC_STAT_HEIGHT]
                    area  = component_stats[label, cv2.CC_STAT_AREA]

                    if area < MIN_AREA:
                        continue

                    y_max = y_min + h_box - 1
                    x_max = x_min + w_box - 1

                    cropped_curr = curr_i[y_min:y_max+1, x_min:x_max+1]
                    
                    # ---------- PATH 0: THE ENTITY CACHE (LONG-TERM MEMORY) ----------
                    matched_cache_idx = -1
                    for idx, cached_block in enumerate(entity_cache):
                        # Strict shape check prevents expensive math on mismatched blocks
                        if cached_block.shape == cropped_curr.shape:
                            mse = np.mean((cached_block.astype(np.float32) - cropped_curr.astype(np.float32))**2)
                            if mse < CACHE_MSE_THRESHOLD:
                                matched_cache_idx = idx
                                break

                    if matched_cache_idx != -1:
                        flag = 5
                        # Just pack the array index (0-254) as a single byte!
                        compressed = struct.pack("B", matched_cache_idx)
                        next_prev[y_min:y_max+1, x_min:x_max+1] = entity_cache[matched_cache_idx]
                        recon_block = entity_cache[matched_cache_idx].copy()
                        stats["Type 5 (Entity Cache)"] += 1
                    else:
                        # ---------- PATH 1: TEMPORAL MOTION SEARCH ----------
                        search_y_min = max(0, y_min - MOTION_SEARCH_RANGE)
                        search_y_max = min(int(h * 1.5), y_max + 1 + MOTION_SEARCH_RANGE)
                        search_x_min = max(0, x_min - MOTION_SEARCH_RANGE)
                        search_x_max = min(w, x_max + 1 + MOTION_SEARCH_RANGE)

                        search_window = prev[search_y_min:search_y_max, search_x_min:search_x_max]
                        
                        res = cv2.matchTemplate(search_window.astype(np.float32), cropped_curr.astype(np.float32), cv2.TM_SQDIFF)
                        min_val, _, min_loc, _ = cv2.minMaxLoc(res)
                        
                        mse = min_val / (h_box * w_box)

                        if mse < MOTION_MSE_THRESHOLD:
                            flag = 4
                            best_prev_x = search_x_min + min_loc[0]
                            best_prev_y = search_y_min + min_loc[1]
                            
                            dy = y_min - best_prev_y
                            dx = x_min - best_prev_x
                            compressed = struct.pack("hh", dy, dx)
                            
                            recon_block = prev[best_prev_y:best_prev_y+h_box, best_prev_x:best_prev_x+w_box].copy()
                            next_prev[y_min:y_max+1, x_min:x_max+1] = recon_block
                            stats["Type 4 (Motion)"] += 1

                        else:
                            # ---------- PATH 2 & 3: SPATIAL DEVIATION ----------
                            cropped_quant = quantized[y_min:y_max+1, x_min:x_max+1]
                            variance = np.var(cropped_curr, axis=(0, 1))
                            max_variance = np.max(variance)

                            if max_variance < BIAS_BLOB_THRESHOLD:
                                flag = 3
                                mean_val = int(np.mean(cropped_curr))
                                compressed = struct.pack("B", mean_val) 
                                
                                recon_block = np.full((h_box, w_box, 1), mean_val, dtype=np.int16)
                                next_prev[y_min:y_max+1, x_min:x_max+1] = recon_block
                                stats["Type 3 (Vector Blob)"] += 1
                            else:
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
                                
                                recon_block = next_prev[y_min:y_max+1, x_min:x_max+1].copy()
                                stats["Type 1/2 (Zstd Delta)"] += 1

                    # --- SYNCHRONIZE MEMORY BANK ---
                    # Add every processed block to the cache so the decoder can reference it later
                    entity_cache.append(recon_block)
                    if len(entity_cache) > 255:
                        entity_cache.pop(0)

                    components.append((flag, y_min, y_max, x_min, x_max, compressed))

                if len(components) == 0:
                    f.write(struct.pack("B", 2))
                else:
                    f.write(struct.pack("B", 1))
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
    
    print("\n" + "="*30)
    print("📡 ENCODER TELEMETRY")
    print("="*30)
    for key, val in stats.items():
        print(f"- {key}: {val} packets")
    print("="*30 + "\n")