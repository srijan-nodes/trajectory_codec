import cv2
import numpy as np
import struct
import zstandard as zstd
import os
from tqdm import tqdm

def run_encoder(video_path, output_path, config, max_frames=150, QP=15):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret: return {}
    h, w, _ = frame.shape
    h, w = h - (h % 2), w - (w % 2)
    cctx = zstd.ZstdCompressor(level=3)

    curr_i420 = cv2.cvtColor(frame[:h, :w], cv2.COLOR_BGR2YUV_I420)
    
    palette = np.zeros(16, dtype=np.uint8)
    if config.get('palette'):
        y_sample = curr_i420[:h, :w].flatten()[::4].astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(y_sample, 16, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
        palette = np.sort(centers.flatten()).astype(np.uint8)

    with open(output_path, "wb") as f:
        f.write(b'NAM_LAB')
        f.write(struct.pack("III", h, w, 1)) 
        f.write(struct.pack("H", QP))
        f.write(palette.tobytes()) 

        prev = curr_i420.reshape((int(h * 1.5), w, 1)).astype(np.int16)
        entity_cache = []
        frame_idx = 0
        pbar = tqdm(total=max_frames, desc=f"Encoding: {config['name']}", unit="f", leave=False)

        telemetry = {
            "scene_cuts": 0,
            
            "motion_attempts": 0,
            "motion_accepts": 0,
            "motion_rejected": 0,
            "total_motion_mse": 0.0,
            "total_rejected_mse": 0.0,
            
            "motion_bytes_saved": 0,
            
            "scale_attempts": 0,
            "scale_accepts": 0,
            
            # COMPONENT FRAGMENTATION TRACKERS
            "total_components": 0,
            "total_component_area": 0,
            "max_components_frame": 0,
            "frames_with_components": 0,
            
            # ENGINE MODE COUNTS
            "spatial_count": 0,
            "motion_count": 0,
            "cache_count": 0,
            "scale_count": 0,
            "palette_count": 0,
        }

        while frame_idx < max_frames:
            if frame_idx > 0:
                ret, frame = cap.read()
                if not ret: break
            curr_i = cv2.cvtColor(frame[:h, :w], cv2.COLOR_BGR2YUV_I420).reshape((int(h * 1.5), w, 1)).astype(np.int16)

            is_scene_change = False
            if frame_idx > 0:
                hist_curr = cv2.calcHist([curr_i[:h].astype(np.uint8)], [0], None, [256], [0, 256])
                hist_prev = cv2.calcHist([prev[:h].astype(np.uint8)], [0], None, [256], [0, 256])
                cv2.normalize(hist_curr, hist_curr, 0, 1, cv2.NORM_MINMAX)
                cv2.normalize(hist_prev, hist_prev, 0, 1, cv2.NORM_MINMAX)
                
                corr = cv2.compareHist(hist_curr, hist_prev, cv2.HISTCMP_CORREL)
                mean_diff = np.mean(np.abs(curr_i.astype(np.float32) - prev.astype(np.float32)))
                
                if corr < 0.40 or mean_diff > 20.0:
                    is_scene_change = True
                    telemetry["scene_cuts"] += 1

            if frame_idx == 0 or is_scene_change:
                compressed = cctx.compress(curr_i.astype(np.uint8).tobytes())
                f.write(struct.pack("B", 0))
                f.write(struct.pack("I", len(compressed)))
                f.write(compressed)
                prev = curr_i
                entity_cache.clear() 
            else:
                quantized = np.round((curr_i - prev) / QP).astype(np.int16)
                
                # --- MORPHOLOGICAL SEGMENTATION UPGRADE ---
                base_mask = np.any(np.abs(quantized) > 0, axis=2).astype(np.uint8)
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.dilate(mask, kernel, iterations=1)
                # ------------------------------------------

                num_labels, _, component_stats, _ = cv2.connectedComponentsWithStats(mask)
                components = []
                next_prev = prev.copy()
                valid_components_this_frame = 0

                for label in range(1, num_labels):
                    x_min, y_min, w_box, h_box, area = component_stats[label][:5]
                    if area < 100: continue
                    
                    # TELEMETRY: Component Tracking
                    telemetry["total_components"] += 1
                    telemetry["total_component_area"] += area
                    valid_components_this_frame += 1
                    
                    y_max, x_max = y_min + h_box - 1, x_min + w_box - 1
                    cropped_curr = curr_i[y_min:y_max+1, x_min:x_max+1]
                    flag = -1

                    # 1. FUZZY CACHE
                    if config.get('cache') and flag == -1:
                        for idx, cached_block in enumerate(entity_cache):
                            ch, cw, _ = cached_block.shape
                            if abs(ch - h_box) <= 4 and abs(cw - w_box) <= 4:
                                resized_cache = cv2.resize(cached_block, (w_box, h_box), interpolation=cv2.INTER_NEAREST).reshape(h_box, w_box, 1)
                                if np.mean((resized_cache.astype(np.float32) - cropped_curr.astype(np.float32))**2) < 10.0:
                                    flag = 5
                                    payload = struct.pack("B", idx)
                                    next_prev[y_min:y_max+1, x_min:x_max+1] = resized_cache
                                    recon_block = resized_cache.copy()
                                    telemetry["cache_count"] += 1
                                    break

                    # 2. DEEP MOTION SEARCH
                    if config.get('motion') and flag == -1:
                        telemetry["motion_attempts"] += 1
                        s_y_min, s_y_max = max(0, y_min - 40), min(int(h * 1.5), y_max + 41)
                        s_x_min, s_x_max = max(0, x_min - 40), min(w, x_max + 41)
                        
                        search_h, search_w = s_y_max - s_y_min, s_x_max - s_x_min
                        
                        if search_h >= h_box and search_w >= w_box:
                            res = cv2.matchTemplate(prev[s_y_min:s_y_max, s_x_min:s_x_max].astype(np.float32), cropped_curr.astype(np.float32), cv2.TM_CCOEFF_NORMED)
                            _, max_val, _, max_loc = cv2.minMaxLoc(res)
                            
                            if max_val > 0.98: 
                                best_y, best_x = s_y_min + max_loc[1], s_x_min + max_loc[0]
                                candidate = prev[best_y:best_y+h_box, best_x:best_x+w_box].copy()
                                motion_mse = np.mean((candidate.astype(np.float32) - cropped_curr.astype(np.float32))**2)
                                
                                if motion_mse < 50.0:
                                    flag = 4
                                    payload = struct.pack("hh", y_min - best_y, x_min - best_x)
                                    next_prev[y_min:y_max+1, x_min:x_max+1] = candidate
                                    recon_block = candidate.copy()
                                    telemetry["motion_count"] += 1
                                    telemetry["motion_accepts"] += 1
                                    telemetry["total_motion_mse"] += motion_mse
                                    
                                    spatial_estimate = len(cctx.compress((cropped_curr - prev[y_min:y_max+1, x_min:x_max+1]).astype(np.int16).tobytes()))
                                    telemetry["motion_bytes_saved"] += (spatial_estimate - len(payload))
                                else:
                                    telemetry["motion_rejected"] += 1
                                    telemetry["total_rejected_mse"] += motion_mse

                    # 3. AFFINE SCALE VECTOR
                    if config.get('scale') and flag == -1:
                        telemetry["scale_attempts"] += 1
                        if w_box > 10 and h_box > 10:
                            src_w, src_h = max(10, int(w_box * 0.8)), max(10, int(h_box * 0.8))
                            if y_min + src_h < int(h * 1.5) and x_min + src_w < w:
                                anchor_block = prev[y_min:y_min+src_h, x_min:x_min+src_w]
                                sc_recon = cv2.resize(anchor_block, (w_box, h_box), interpolation=cv2.INTER_NEAREST).reshape(h_box, w_box, 1)
                                if np.mean((sc_recon.astype(np.float32) - cropped_curr.astype(np.float32))**2) < 50.0:
                                    flag = 6
                                    payload = struct.pack("HH", src_w, src_h)
                                    next_prev[y_min:y_max+1, x_min:x_max+1] = sc_recon
                                    recon_block = sc_recon.copy()
                                    telemetry["scale_count"] += 1
                                    telemetry["scale_accepts"] += 1

                    # 4. 4-BIT ALIAS PALETTE
                    if config.get('palette') and flag == -1:
                        if np.max(np.var(cropped_curr, axis=(0, 1))) > 12.0:
                            diffs = np.abs(cropped_curr - palette.reshape((1, 1, 16)))
                            indices = np.argmin(diffs, axis=-1).astype(np.uint8).flatten()
                            
                            pal_recon = palette[indices].reshape((h_box, w_box, 1)).astype(np.int16)
                            if np.mean((pal_recon.astype(np.float32) - cropped_curr.astype(np.float32))**2) < 15.0:
                                flag = 8
                                if len(indices) % 2 != 0: indices = np.append(indices, 0)
                                packed_bytes = (indices[0::2] << 4) | indices[1::2]
                                payload = cctx.compress(packed_bytes.tobytes())
                                recon_block = pal_recon
                                next_prev[y_min:y_max+1, x_min:x_max+1] = recon_block
                                telemetry["palette_count"] += 1

                    # 5. SPATIAL FALLBACK
                    if flag == -1:
                        telemetry["spatial_count"] += 1
                        cropped_quant = quantized[y_min:y_max+1, x_min:x_max+1]
                        if cropped_quant.min() >= -128 and cropped_quant.max() <= 127:
                            flag = 1
                            payload = cctx.compress((cropped_quant + 128).astype(np.uint8).tobytes())
                        else:
                            flag = 2
                            payload = cctx.compress(cropped_quant.astype(np.int16).tobytes())
                        next_prev[y_min:y_max+1, x_min:x_max+1] += (cropped_quant * QP)
                        recon_block = next_prev[y_min:y_max+1, x_min:x_max+1].copy()

                    entity_cache.append(recon_block)
                    if len(entity_cache) > 255: entity_cache.pop(0)
                    components.append((flag, y_min, y_max, x_min, x_max, payload))

                # Update per-frame component stats
                if valid_components_this_frame > 0:
                    telemetry["frames_with_components"] += 1
                telemetry["max_components_frame"] = max(telemetry["max_components_frame"], valid_components_this_frame)

                if len(components) == 0 and not is_scene_change:
                    f.write(struct.pack("B", 2))
                elif not is_scene_change:
                    f.write(struct.pack("B", 1))
                    f.write(struct.pack("H", len(components)))
                    for flag, y_min, y_max, x_min, x_max, payload in components:
                        f.write(struct.pack("B", flag))
                        f.write(struct.pack("HHHH", y_min, y_max, x_min, x_max))
                        f.write(struct.pack("I", len(payload)))
                        f.write(payload)
                prev = next_prev
            frame_idx += 1
            pbar.update(1)
        pbar.close()
        
    cap.release()
    return telemetry

def calculate_mse(original_video, encoded_path, max_frames=150):
    dctx = zstd.ZstdDecompressor()
    cap = cv2.VideoCapture(original_video)
    mse_total = 0
    
    with open(encoded_path, "rb") as f:
        magic = f.read(7)
        h, w, _ = struct.unpack("III", f.read(12))
        QP = struct.unpack("H", f.read(2))[0]
        palette = np.frombuffer(f.read(16), dtype=np.uint8)
        i420_h = int(h * 1.5)
        
        prev = None
        entity_cache = []
        frame_idx = 0

        while frame_idx < max_frames:
            ret, orig_frame = cap.read()
            if not ret: break
            orig_yuv = cv2.cvtColor(orig_frame[:h, :w], cv2.COLOR_BGR2YUV_I420)
            
            type_byte = f.read(1)
            if not type_byte: break 
            frame_type = struct.unpack("B", type_byte)[0]

            if frame_type == 0:
                size = struct.unpack("I", f.read(4))[0]
                prev = np.frombuffer(dctx.decompress(f.read(size)), dtype=np.uint8).reshape((i420_h, w, 1)).astype(np.int16)
            elif frame_type == 1:
                num_components = struct.unpack("H", f.read(2))[0]
                next_frame = prev.copy()
                for _ in range(num_components):
                    flag = struct.unpack("B", f.read(1))[0]
                    y_min, y_max, x_min, x_max = struct.unpack("HHHH", f.read(8))
                    size = struct.unpack("I", f.read(4))[0]
                    raw_payload = f.read(size)
                    h_box, w_box = y_max - y_min + 1, x_max - x_min + 1

                    if flag == 5:
                        cached_block = entity_cache[struct.unpack("B", raw_payload)[0]]
                        recon_block = cv2.resize(cached_block, (w_box, h_box), interpolation=cv2.INTER_NEAREST).reshape((h_box, w_box, 1))
                        next_frame[y_min:y_max+1, x_min:x_max+1] = recon_block
                    elif flag == 4:
                        dy, dx = struct.unpack("hh", raw_payload)
                        src_y, src_x = max(0, min(i420_h - h_box, y_min - dy)), max(0, min(w - w_box, x_min - dx))
                        recon_block = prev[src_y:src_y+h_box, src_x:src_x+w_box].copy()
                        next_frame[y_min:y_max+1, x_min:x_max+1] = recon_block
                    elif flag == 6: 
                        src_w, src_h = struct.unpack("HH", raw_payload)
                        anchor_block = prev[y_min:y_min+src_h, x_min:x_min+src_w]
                        recon_block = cv2.resize(anchor_block, (w_box, h_box), interpolation=cv2.INTER_NEAREST).reshape((h_box, w_box, 1))
                        next_frame[y_min:y_max+1, x_min:x_max+1] = recon_block
                    elif flag == 8: 
                        packed_bytes = np.frombuffer(dctx.decompress(raw_payload), dtype=np.uint8)
                        flat_idx = np.empty(packed_bytes.size * 2, dtype=np.uint8)
                        flat_idx[0::2] = packed_bytes >> 4
                        flat_idx[1::2] = packed_bytes & 0x0F
                        recon_block = palette[flat_idx[:h_box * w_box]].reshape((h_box, w_box, 1)).astype(np.int16)
                        next_frame[y_min:y_max+1, x_min:x_max+1] = recon_block
                    else:
                        data = dctx.decompress(raw_payload)
                        quantized = (np.frombuffer(data, dtype=np.uint8).astype(np.int16) - 128) if flag == 1 else np.frombuffer(data, dtype=np.int16)
                        next_frame[y_min:y_max+1, x_min:x_max+1] += (quantized.reshape((h_box, w_box, 1)) * QP)
                        recon_block = next_frame[y_min:y_max+1, x_min:x_max+1].copy()

                    entity_cache.append(recon_block)
                    if len(entity_cache) > 255: entity_cache.pop(0)
                prev = next_frame

            recon_yuv = np.clip(prev, 0, 255).astype(np.uint8).reshape((i420_h, w))
            mse_total += np.mean((orig_yuv.astype(np.float32) - recon_yuv.astype(np.float32)) ** 2)
            frame_idx += 1
            
    cap.release()
    return mse_total / frame_idx