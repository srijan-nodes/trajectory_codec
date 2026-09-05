"""
encoder.py — Hybrid Video Codec Encoder
=========================================
"""
import cv2
import numpy as np
import struct
import zlib
import zstandard as zstd
import time
from tqdm import tqdm
import os

def encode(video_path: str, output_path: str, config=None) -> dict:
    start_time = time.time()
    if config is None:
        config = {
            "name": "HybridV5",
            "motion": True,
            "dct": True,
            "faded_motion": True,
            "dynamic_boxing": True
        }
    LAMBDA = 0.012
    QP = 27
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret: return {}
    
    h, w, _ = frame.shape
    h = h - (h % 16) if h % 16 != 0 else h
    w = w - (w % 16) if w % 16 != 0 else w
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    cap = cv2.VideoCapture(video_path)
    
    cctx = zstd.ZstdCompressor(level=9)
    BLOCK_SIZE = 16
    grid_h, grid_w = h // BLOCK_SIZE, w // BLOCK_SIZE

    telemetry = {
        "blocks_total": 0, "blocks_skip": 0, "blocks_spatial": 0, 
        "blocks_raw": 0, "blocks_motion": 0, "blocks_background": 0, 
        "blocks_solid": 0, "blocks_palette": 0, 
        "blocks_faded_mot": 0, "blocks_dct": 0, "blocks_dithered": 0,
        "dynamic_boxes": 0
    }
    
    with open(output_path, "wb") as f:
        f.write(b'NAM_V5')
        f.write(struct.pack("<III", int(h), int(w), int(total_frames))) 
        f.write(struct.pack("<H", int(QP)))
        f.write(struct.pack("<B", 1 if config.get('background') else 0))

        prev = None
        bg_model = None
        
        mode_buffer, payload_buffer = bytearray(), bytearray()
        frame_idx = 0
        pbar = tqdm(total=total_frames, desc=f"Enc V5: {config['name']}", unit="f", leave=False)
        stream = cctx.stream_writer(f)

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            curr_y = cv2.cvtColor(frame[:h, :w], cv2.COLOR_BGR2GRAY).reshape((h, w, 1)).astype(np.int16)
            
            is_scene_change = False
            if prev is not None and np.mean(np.abs(curr_y.astype(np.float32) - prev.astype(np.float32))) > 25.0:
                is_scene_change = True

            if prev is None or is_scene_change:
                raw_bytes = curr_y.astype(np.uint8).tobytes()
                stream.write(struct.pack("<B", 0))
                stream.write(struct.pack("<I", len(raw_bytes)))
                stream.write(raw_bytes)
                prev = curr_y.copy()
                bg_model = curr_y.copy().astype(np.float32)
                bg_model_i32 = bg_model.astype(np.int32)
                frame_idx += 1
                pbar.update(1)
                continue

            stream.write(struct.pack("<B", 1))
            next_prev = prev.copy()
            
            curr_y_i32 = curr_y.astype(np.int32)
            prev_i32 = prev.astype(np.int32)
            
            diff = np.abs(curr_y_i32 - prev_i32)
            sad_grid = diff.squeeze().reshape(grid_h, BLOCK_SIZE, grid_w, BLOCK_SIZE).sum(axis=(1, 3))
            max_grid = diff.squeeze().reshape(grid_h, BLOCK_SIZE, grid_w, BLOCK_SIZE).max(axis=(1, 3))
            
            skip_run = 0
            processed = np.zeros((grid_h, grid_w), dtype=bool)
            last_dy, last_dx = 0, 0

            for y_idx in range(grid_h):
                for x_idx in range(grid_w):
                    if processed[y_idx, x_idx]: continue

                    y_min, x_min = y_idx * BLOCK_SIZE, x_idx * BLOCK_SIZE
                    y_max, x_max = y_min + BLOCK_SIZE, x_min + BLOCK_SIZE
                    
                    curr_block = curr_y[y_min:y_max, x_min:x_max]
                    prev_block = prev[y_min:y_max, x_min:x_max]
                    base_sad = sad_grid[y_idx, x_idx]
                    
                    # 1. FAST LANE: SKIP (V5 + HYBRID)

                    # [HYBRID] MAX_GRID SKIP
                    if max_grid[y_idx, x_idx] < 4:
                        skip_run += 1
                        telemetry["blocks_skip"] += 1
                        telemetry["blocks_total"] += 1
                        processed[y_idx, x_idx] = True
                        continue

                    if max_grid[y_idx, x_idx] < 4 and config.get('m4_skip', True):
                        skip_run += 1
                        telemetry["blocks_skip"] += 1
                        telemetry["blocks_total"] += 1
                        processed[y_idx, x_idx] = True
                        continue 
                    
                    if skip_run > 0:
                        mode_buffer.append(0xFF)
                        payload_buffer.extend(struct.pack("<H", int(skip_run)))
                        skip_run = 0

                    best_mode = 2
                    best_payload = curr_block.astype(np.uint8).tobytes()
                    best_recon = curr_block.copy()
                    best_cost = 257 + (LAMBDA * base_sad) 
                    log_dy, log_dx = 0, 0
                    
                    curr_block_i32 = curr_y_i32[y_min:y_max, x_min:x_max]

                    # 2. FAST LANE: BACKGROUND (WITH HARD MSE GATE)
                    if config.get('m6_bg', True):
                        bg_cand = bg_model[y_min:y_max, x_min:x_max].astype(np.int32)
                        bg_sad = np.sum(np.abs(curr_block_i32 - bg_cand))
                        bg_mse = np.mean((curr_block_i32 - bg_cand) ** 2)
                        
                        if bg_sad / 256 < 6.0 and bg_mse < 60.0:
                            cost = 1 + (LAMBDA * bg_sad)
                            if cost < best_cost:
                                best_mode, best_cost, best_payload = 6, cost, b''
                                best_recon = bg_model[y_min:y_max, x_min:x_max].astype(np.int16)

                    # 3. FAST LANE: SOLID
                    if config.get('m7_solid', True):
                        b_min, b_max = np.min(curr_block), np.max(curr_block)
                        if (b_max - b_min) <= 2:
                            solid_val = int(np.mean(curr_block))
                            solid_recon = np.full((BLOCK_SIZE, BLOCK_SIZE, 1), solid_val, dtype=np.int16)
                            solid_sad = np.sum(np.abs(curr_block_i32 - solid_recon.astype(np.int32)))
                            cost = 2 + (LAMBDA * solid_sad)
                            if cost < best_cost:
                                best_mode, best_cost = 7, cost
                                best_payload = struct.pack("<B", np.clip(solid_val, 0, 255))
                                best_recon = solid_recon

                    # 4. FAST LANE: PRECISE PALETTE
                    if config.get('m12_palette', True) and best_cost > 10:
                        unique_vals = np.unique(curr_block)
                        if len(unique_vals) <= 4:
                            palette = np.zeros(4, dtype=np.uint8)
                            palette[:len(unique_vals)] = unique_vals
                            indices = np.zeros_like(curr_block, dtype=np.uint8)
                            for i, v in enumerate(unique_vals): indices[curr_block == v] = i
                            flat = indices.flatten()
                            shift = np.array([6, 4, 2, 0], dtype=np.uint8)
                            packed = np.bitwise_or.reduce(flat.reshape(-1, 4) << shift, axis=1).astype(np.uint8)
                            cost = 1 + len(unique_vals) + 64 + (LAMBDA * 0)
                            if cost < best_cost:
                                best_mode, best_cost = 12, cost
                                best_payload = struct.pack("<B", len(unique_vals)) + palette.tobytes() + packed.tobytes()
                                best_recon = curr_block.copy()

                    # 5. FAST LANE: PREDICTED & FAST MOTION (WITH HARD MSE GATES)
                    if base_sad > 256 and best_cost > 15:
                        pred_y = np.clip(y_min - last_dy, 0, h - BLOCK_SIZE)
                        pred_x = np.clip(x_min - last_dx, 0, w - BLOCK_SIZE)
                        cand_pred = prev_i32[pred_y:pred_y+BLOCK_SIZE, pred_x:pred_x+BLOCK_SIZE]
                        
                        pred_sad = np.sum(np.abs(curr_block_i32 - cand_pred))
                        pred_mse = np.mean((curr_block_i32 - cand_pred) ** 2)
                        
                        if pred_sad / 256 < 4.0 and pred_mse < 60.0 and config.get('m11_cached', True):
                            cost = 1 + (LAMBDA * pred_sad)
                            if cost < best_cost:
                                best_mode, best_cost, best_payload = 11, cost, b''
                                best_recon = prev[pred_y:pred_y+BLOCK_SIZE, pred_x:pred_x+BLOCK_SIZE].copy()
                                log_dy, log_dx = last_dy, last_dx
                        else:
                            s_y_min, s_y_max = max(0, pred_y - 16), min(h, pred_y + BLOCK_SIZE + 16)
                            s_x_min, s_x_max = max(0, pred_x - 16), min(w, pred_x + BLOCK_SIZE + 16)
                            if (s_y_max - s_y_min >= BLOCK_SIZE) and (s_x_max - s_x_min >= BLOCK_SIZE):
                                res = cv2.matchTemplate(prev[s_y_min:s_y_max, s_x_min:s_x_max].astype(np.uint8), curr_block.astype(np.uint8), cv2.TM_SQDIFF)
                                min_val, _, min_loc, _ = cv2.minMaxLoc(res)
                                best_match_y = s_y_min + min_loc[1]
                                best_match_x = s_x_min + min_loc[0]
                                candidate = prev_i32[best_match_y:best_match_y+BLOCK_SIZE, best_match_x:best_match_x+BLOCK_SIZE]
                                
                                cost = 3 + (LAMBDA * np.sum(np.abs(curr_block_i32 - candidate)))
                                cand_mse = np.mean((curr_block_i32 - candidate) ** 2)
                                
                                log_dy = int(y_min - best_match_y)
                                log_dx = int(x_min - best_match_x)
                                
                                if cost < best_cost and cand_mse < 150.0 and config.get('m4_search', True):
                                    best_mode, best_cost = 4, cost
                                    best_payload = struct.pack("<hh", log_dy, log_dx)
                                    best_recon = prev[best_match_y:best_match_y+BLOCK_SIZE, best_match_x:best_match_x+BLOCK_SIZE].copy()

                    # ------------------------------------------------------------------
                    # 🚨 THE DESPERATION ARENA 🚨
                    # ------------------------------------------------------------------
                    if best_cost > 30 and config.get('m13_faded', True): 
                        
                        # A. Faded Motion (With Hard Gate) - Mode 13
                        ref_y = np.clip(y_min - log_dy, 0, h - BLOCK_SIZE)
                        ref_x = np.clip(x_min - log_dx, 0, w - BLOCK_SIZE)
                        fade_cand = prev_i32[ref_y:ref_y+BLOCK_SIZE, ref_x:ref_x+BLOCK_SIZE]
                        
                        luma_shift = int(np.mean(curr_block_i32) - np.mean(fade_cand))
                        if -128 <= luma_shift <= 127:
                            shifted_cand = np.clip(fade_cand + luma_shift, 0, 255)
                            fade_sad = np.sum(np.abs(curr_block_i32 - shifted_cand))
                            cost = 4 + (LAMBDA * fade_sad)
                            
                            if cost < best_cost:
                                best_mode, best_cost = 13, cost
                                best_payload = struct.pack("<hhb", log_dy, log_dx, luma_shift)
                                best_recon = shifted_cand.astype(np.int16).reshape((16, 16, 1))

                        # C. Fast Forced Dither (Uniform Binning) - Mode 15
                        dither_sad = 0
                        if (best_cost > 75 or (np.max(curr_block) - np.min(curr_block) > 40)) and config.get('m15_dither', True):
                            b_min, b_max = np.min(curr_block), np.max(curr_block)
                            centers = np.linspace(b_min, b_max, 4).astype(np.uint8)
                            labels = np.abs(curr_block_i32.reshape(-1, 1) - centers).argmin(axis=1).astype(np.uint8)
                            dither_recon = centers[labels].reshape((16, 16))
                            dither_sad = np.sum(np.abs(curr_block_i32 - dither_recon))
                            
                            cost_15 = 69 + (LAMBDA * dither_sad)
                            if cost_15 < best_cost:
                                best_mode, best_cost = 15, cost_15
                                palette = np.zeros(4, dtype=np.uint8)
                                palette[:] = centers
                                packed = np.zeros(64, dtype=np.uint8)
                                for i in range(256):
                                    byte_idx = i // 4
                                    bit_shift = 6 - (2 * (i % 4))
                                    packed[byte_idx] |= (labels[i] << bit_shift)
                                best_payload = palette.tobytes() + packed.tobytes()
                                best_recon = dither_recon.astype(np.int16).reshape((16, 16, 1))
                                
                        # B. DCT COMPRESSION (Mode 9 & 10)
                        if config.get('m9_dct', True) and best_cost > 45:
                            if np.max(curr_block) - np.min(curr_block) <= 40 or dither_sad > 500:
                                curr_f32 = curr_block.astype(np.float32)
                                dct = cv2.dct(curr_f32)
                                
                                dct_low = np.zeros_like(dct)
                                dct_low[:4, :4] = dct[:4, :4]
                                recon_low = cv2.idct(dct_low).astype(np.int32)
                                
                                cost_9 = 33 + (LAMBDA * 0.6 * np.sum(np.abs(curr_block_i32 - recon_low)))
                                if cost_9 < best_cost:
                                    best_mode, best_cost = 9, cost_9
                                    best_payload = dct_low[:4, :4].flatten().astype(np.float16).tobytes()
                                    best_recon = recon_low.astype(np.int16).reshape((16, 16, 1))

                                if best_cost > 100 and config.get('m10_dct', True):
                                    dct_mid = np.zeros_like(dct)
                                    dct_mid[:8, :8] = dct[:8, :8]
                                    recon_mid = cv2.idct(dct_mid).astype(np.int32)
                                    
                                    cost_10 = 129 + (LAMBDA * 0.8 * np.sum(np.abs(curr_block_i32 - recon_mid)))
                                    if cost_10 < best_cost:
                                        best_mode, best_cost = 10, cost_10
                                        best_payload = dct_mid[:8, :8].flatten().astype(np.float16).tobytes()
                                        best_recon = recon_mid.astype(np.int16).reshape((16, 16, 1))

                    # ------------------------------------------------------------------
                    # 🧱 THE ABSOLUTE FALLBACK
                    # ------------------------------------------------------------------
                    if best_cost > 10 and config.get('m1_spatial', True):
                        delta = np.round((curr_block.astype(np.float32) - prev_block.astype(np.float32)) / QP).astype(np.int16)
                        if delta.min() >= -128 and delta.max() <= 127:
                            nnz = np.count_nonzero(delta)
                            if nnz < 128:
                                indices = np.nonzero(delta.flatten())[0].astype(np.uint8)
                                values = (delta.flatten()[indices] + 128).astype(np.uint8)
                                payload = struct.pack("<B", nnz) + indices.tobytes() + values.tobytes()
                                recon = np.clip(prev_block + (delta * QP), 0, 255).astype(np.int16)
                                cost = (1 + nnz*2) + (LAMBDA * np.sum(np.abs(curr_block_i32 - recon.astype(np.int32))))
                                if cost < best_cost:
                                    best_mode, best_cost, best_payload, best_recon = 1, cost, payload, recon

                    # Vector Logging
                    if best_mode in [4, 11, 13]:
                        last_dy, last_dx = int(log_dy), int(log_dx)

                    w_mult, h_mult = 1, 1
                    if config.get('dynamic_boxing') and best_mode in [4, 6, 7, 11]:
                        max_w = min(8, grid_w - x_idx)
                        max_h = min(8, grid_h - y_idx)
                        for w_test in range(2, max_w + 1):
                            if np.any(processed[y_idx, x_idx : x_idx + w_test]): break
                            test_curr = curr_y_i32[y_min : y_min + BLOCK_SIZE, x_min : x_min + w_test*BLOCK_SIZE]
                            if best_mode == 7:
                                solid_val = struct.unpack("<B", best_payload)[0]
                                if np.max(np.abs(test_curr - solid_val)) < 12: w_mult = w_test
                                else: break
                            elif best_mode == 6:
                                bg_cand = bg_model_i32[y_min : y_min + BLOCK_SIZE, x_min : x_min + w_test*BLOCK_SIZE]
                                if np.max(np.abs(test_curr - bg_cand)) < 15: w_mult = w_test
                                else: break
                            elif best_mode in [4, 11]:
                                ref_y = np.clip(y_min - log_dy, 0, h - BLOCK_SIZE)
                                ref_x = np.clip(x_min - log_dx, 0, w - w_test*BLOCK_SIZE)
                                cand_pred = prev_i32[ref_y : ref_y + BLOCK_SIZE, ref_x : ref_x + w_test*BLOCK_SIZE]
                                grow_gate = 60.0 if best_mode == 11 else 150.0
                                if np.mean((test_curr - cand_pred) ** 2) < grow_gate and np.max(np.abs(test_curr - cand_pred)) < 15: w_mult = w_test
                                else: break
                        for h_test in range(2, max_h + 1):
                            if np.any(processed[y_idx : y_idx + h_test, x_idx : x_idx + w_mult]): break
                            test_curr = curr_y_i32[y_min : y_min + h_test*BLOCK_SIZE, x_min : x_min + w_mult*BLOCK_SIZE]
                            if best_mode == 7:
                                solid_val = struct.unpack("<B", best_payload)[0]
                                if np.max(np.abs(test_curr - solid_val)) < 12: h_mult = h_test
                                else: break
                            elif best_mode == 6:
                                bg_cand = bg_model_i32[y_min : y_min + h_test*BLOCK_SIZE, x_min : x_min + w_mult*BLOCK_SIZE]
                                if np.max(np.abs(test_curr - bg_cand)) < 15: h_mult = h_test
                                else: break
                            elif best_mode in [4, 11]:
                                ref_y = np.clip(y_min - log_dy, 0, h - h_test*BLOCK_SIZE)
                                ref_x = np.clip(x_min - log_dx, 0, w - w_mult*BLOCK_SIZE)
                                cand_pred = prev_i32[ref_y : ref_y + h_test*BLOCK_SIZE, ref_x : ref_x + w_mult*BLOCK_SIZE]
                                grow_gate = 60.0 if best_mode == 11 else 150.0
                                if np.mean((test_curr - cand_pred) ** 2) < grow_gate and np.max(np.abs(test_curr - cand_pred)) < 15: h_mult = h_test
                                else: break

                    blocks_in_box = w_mult * h_mult
                    if blocks_in_box > 1:
                        telemetry["dynamic_boxes"] += 1
                        mode_buffer.append(3)
                        payload_buffer.extend(struct.pack("<BBB", w_mult, h_mult, best_mode))
                        payload_buffer.extend(best_payload)
                        if best_mode == 7:
                            solid_val = struct.unpack("<B", best_payload)[0]
                            best_recon = np.full((h_mult*BLOCK_SIZE, w_mult*BLOCK_SIZE, 1), solid_val, dtype=np.int16)
                        elif best_mode == 6:
                            best_recon = bg_model[y_min:y_min+h_mult*BLOCK_SIZE, x_min:x_min+w_mult*BLOCK_SIZE].astype(np.int16).copy()
                        elif best_mode in [4, 11]:
                            ref_y = np.clip(y_min - log_dy, 0, h - h_mult*BLOCK_SIZE)
                            ref_x = np.clip(x_min - log_dx, 0, w - w_mult*BLOCK_SIZE)
                            best_recon = prev[ref_y : ref_y + h_mult*BLOCK_SIZE, ref_x : ref_x + w_mult*BLOCK_SIZE].copy()
                    else:
                        mode_buffer.append(best_mode)
                        payload_buffer.extend(best_payload)
                        
                    next_prev[y_min:y_min+(h_mult*BLOCK_SIZE), x_min:x_min+(w_mult*BLOCK_SIZE)] = best_recon
                    processed[y_idx:y_idx+h_mult, x_idx:x_idx+w_mult] = True

                    telemetry["blocks_total"] += blocks_in_box
                    if best_mode == 1: telemetry["blocks_spatial"] += blocks_in_box
                    elif best_mode == 2: telemetry["blocks_raw"] += blocks_in_box
                    elif best_mode in [4, 11]: telemetry["blocks_motion"] += blocks_in_box
                    elif best_mode == 6: telemetry["blocks_background"] += blocks_in_box
                    elif best_mode == 7: telemetry["blocks_solid"] += blocks_in_box
                    elif best_mode == 9 or best_mode == 10: telemetry["blocks_dct"] += blocks_in_box
                    elif best_mode == 12: telemetry["blocks_palette"] += blocks_in_box
                    elif best_mode == 13: telemetry["blocks_faded_mot"] += blocks_in_box
                    elif best_mode == 15: telemetry["blocks_dithered"] += blocks_in_box

            if skip_run > 0:
                mode_buffer.append(0xFF)
                payload_buffer.extend(struct.pack("<H", int(skip_run)))
                
            if config.get('background'):
                # FIX: Force encoder to update BG model exactly like the decoder
                diff_mask = np.abs(next_prev.astype(np.int32) - prev.astype(np.int32)) < 5
                bg_model[diff_mask] = (0.95 * bg_model[diff_mask] + 0.05 * next_prev[diff_mask]).astype(np.float32)

            prev = next_prev.copy()
            frame_idx += 1
            pbar.update(1)

            stream.write(struct.pack("<II", len(mode_buffer), len(payload_buffer)))
            stream.write(mode_buffer)
            stream.write(payload_buffer)
            mode_buffer.clear()
            payload_buffer.clear()
        pbar.close()
        stream.close()
        cap.release()
        
    enc_time = time.time() - start_time
    file_size = os.path.getsize(output_path)
    orig_bytes = os.path.getsize(video_path) if os.path.exists(video_path) else 1
    comp_bytes = file_size
    return {
        "encode_time_s": enc_time,
        "frames": frame_idx,
        "total_frames": frame_idx,
        "blocks": telemetry,
        "orig_bytes": orig_bytes,
        "compressed_bytes": comp_bytes,
        "ratio": comp_bytes / orig_bytes,
        "v5_frames": frame_idx,
        "v6_frames": 0,
        "dynamic_boxes": telemetry["dynamic_boxes"]
    }
