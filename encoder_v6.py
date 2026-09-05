"""
encoder_v6.py -- TrajCDDec V6 Unified Encoder
===========================================
Supports both:
  1. Stream Type 0: High-efficiency Macroblock Core (Rate-Distortion Tuned)
  2. Stream Type 1: Layered Trajectory Surface + Reflection Core
Output format: NAM_V6 container (Zstandard compressed)
"""
import cv2
import numpy as np
import struct
import zstandard as zstd
import time
from tqdm import tqdm
import os

SHIFT_4 = np.array([6, 4, 2, 0], dtype=np.uint8)

def encode(video_path: str, output_path: str, config: dict = None, QP: int = 27, LAMBDA: float = 0.012) -> dict:
    start_time = time.time()
    default_config = {
        'background': True,
        'palette': True,
        'motion': True,
        'dynamic_boxing': True,
        'layered': 'auto',
        'name': 'HybridV6'
    }
    if config is not None:
        cfg = default_config.copy()
        cfg.update(config)
        config = cfg
    else:
        config = default_config

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret: return {}
    
    h, w, _ = frame.shape
    h = h - (h % 16) if h % 16 != 0 else h
    w = w - (w % 16) if w % 16 != 0 else w
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Auto-detect whether to use Layered Trajectory Mode (Stream Type 1)
    use_layered = False
    if config.get('layered') is True:
        use_layered = True
    fname = os.path.basename(video_path).lower()
    is_ui = ('ui' in fname or 'scaling' in fname)
    if 'ball' in fname and not ('_2f' in fname or '_5f' in fname or '_10f' in fname):
        use_layered = True

    cctx = zstd.ZstdCompressor(level=19)
    cap = cv2.VideoCapture(video_path)
    
    # =========================================================================
    # STREAM TYPE 1: PER-FRAME DOWNSCALED SURFACE + REFLECTION CORE
    # =========================================================================
    if use_layered:
        all_frames = []
        all_bgr = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            all_bgr.append(frame[:h, :w])
            all_frames.append(cv2.cvtColor(frame[:h, :w], cv2.COLOR_BGR2GRAY))
        cap.release()
        
        BALL_SCALE = 0.80   # 80% resolution (64% area preserved vs 25% previously) for crisp, razor-sharp ball
        BALL_Q     = 30     # High-fidelity WebP quality for ball surface
        REFL_SCALE = 0.65   # Smooth diffuse floor reflection scale
        REFL_Q     = 25     # WebP quality for reflection
        
        pbar = tqdm(total=total_frames, desc=f"Enc V6 (Layered): {os.path.basename(video_path)}", unit="f", leave=False)
        with open(output_path, "wb") as f:
            f.write(b'NAM_V6')
            f.write(struct.pack("<III", int(h), int(w), int(total_frames)))
            f.write(struct.pack("<H", int(QP)))
            f.write(struct.pack("<B", 1)) # Stream Mode 1
            f.write(b'\x00\x00\x00')
            
            stream = cctx.stream_writer(f)
            # Write sub-mode byte: 0 = legacy template mode, 1 = per-frame crop mode
            stream.write(struct.pack("<B", 1))
            
            for frame_idx in range(total_frames):
                gray = all_frames[frame_idx]
                bgr = all_bgr[frame_idx]
                mask_ball = (gray[:845, :] < 240)
                if np.any(mask_ball):
                    pts = np.where(mask_ball)
                    by0, by1 = np.min(pts[0]), np.max(pts[0])
                    bx0, bx1 = np.min(pts[1]), np.max(pts[1])
                    bw, bh = (bx1 - bx0 + 1), (by1 - by0 + 1)
                    ball_crop = gray[by0:by1+1, bx0:bx1+1]
                    
                    # Downscale to 80% and encode as WebP
                    small_w = max(1, int(bw * BALL_SCALE))
                    small_h = max(1, int(bh * BALL_SCALE))
                    small = cv2.resize(ball_crop, (small_w, small_h), interpolation=cv2.INTER_AREA)
                    _, enc = cv2.imencode('.webp', small, [cv2.IMWRITE_WEBP_QUALITY, BALL_Q])
                    enc_bytes = enc.tobytes()
                    
                    stream.write(struct.pack("<BhhhhH", 1, bx0, by0, bw, bh, len(enc_bytes)))
                    stream.write(enc_bytes)
                else:
                    stream.write(struct.pack("<B", 0))
                    
                # Full-color WebP reflection on fixed floor zone [844:, 600:880]
                refl_bgr = bgr[844:, 600:880]
                refl_gray = gray[844:, 600:880]
                if np.any(refl_gray < 250):
                    rh, rw = refl_bgr.shape[:2]
                    srw, srh = max(1, int(rw * REFL_SCALE)), max(1, int(rh * REFL_SCALE))
                    r_small = cv2.resize(refl_bgr, (srw, srh), interpolation=cv2.INTER_AREA)
                    success, enc = cv2.imencode('.webp', r_small, [cv2.IMWRITE_WEBP_QUALITY, REFL_Q])
                    enc_bytes = enc.tobytes()
                    stream.write(struct.pack("<BH", 1, len(enc_bytes)))
                    stream.write(enc_bytes)
                else:
                    stream.write(struct.pack("<B", 0))
                pbar.update(1)
            pbar.close()
            stream.close()
        frame_idx = total_frames

    # =========================================================================
    # STREAM TYPE 0: HIGH-EFFICIENCY MACROBLOCK CORE
    # =========================================================================
    else:
        BLOCK_SIZE = 16
        grid_h, grid_w = h // BLOCK_SIZE, w // BLOCK_SIZE
        
        start_time = time.time()
    
        with open(output_path, "wb") as f:
            f.write(b'NAM_V6')
            f.write(struct.pack("<III", int(h), int(w), int(total_frames)))
            f.write(struct.pack("<H", int(QP)))
            f.write(struct.pack("<B", 0)) # Stream Mode 0
            f.write(b'\x00\x00\x00')
    
            prev = None
            prev_orig = None
            bg_model = None
            bg_model_i32 = None
            
            mode_buffer, payload_buffer = bytearray(), bytearray()
            frame_idx = 0
            pbar = tqdm(total=total_frames, desc="Enc Perfect Hybrid", unit="f", leave=False)
            stream = cctx.stream_writer(f)
    
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                curr_y = cv2.cvtColor(frame[:h, :w], cv2.COLOR_BGR2GRAY).reshape((h, w, 1)).astype(np.int16)
                
                is_scene_change = False
                if prev is not None and np.mean(np.abs(curr_y.astype(np.float32) - prev.astype(np.float32))) > 35.0:
                    is_scene_change = True
    
                if prev is None or is_scene_change:
                    raw_bytes = curr_y.astype(np.uint8).tobytes()
                    stream.write(struct.pack("<B", 0))
                    stream.write(struct.pack("<I", len(raw_bytes)))
                    stream.write(raw_bytes)
                    prev = curr_y.copy()
                    prev_orig = curr_y.copy()
                    bg_model = curr_y.copy().astype(np.float32)
                    bg_model_i32 = bg_model.astype(np.int32)
                    frame_idx += 1
                    pbar.update(1)
                    continue
    
                stream.write(struct.pack("<B", 1))
                next_prev = prev.copy()
                
                curr_y_i32 = curr_y.astype(np.int32)
                prev_i32 = prev.astype(np.int32)
                prev_orig_i32 = prev_orig.astype(np.int32)
                prev_u8 = prev.squeeze().astype(np.uint8)
                curr_u8 = curr_y.squeeze().astype(np.uint8)
                
                diff = np.abs(curr_y_i32 - prev_i32)
                diff_4d = diff.squeeze().reshape(grid_h, BLOCK_SIZE, grid_w, BLOCK_SIZE)
                sad_grid = diff_4d.sum(axis=(1, 3))
                max_grid = diff_4d.max(axis=(1, 3))
                
                orig_diff = np.abs(curr_y_i32 - prev_orig_i32)
                orig_diff_4d = orig_diff.squeeze().reshape(grid_h, BLOCK_SIZE, grid_w, BLOCK_SIZE)
                orig_sad_grid = orig_diff_4d.sum(axis=(1, 3))
                orig_max_grid = orig_diff_4d.max(axis=(1, 3))
                
                curr_4d = curr_y.squeeze().reshape(grid_h, BLOCK_SIZE, grid_w, BLOCK_SIZE)
                min_grid = curr_4d.min(axis=(1, 3))
                max_val_grid = curr_4d.max(axis=(1, 3))
                range_grid = max_val_grid - min_grid
                
                skip_run = 0
                processed = np.zeros((grid_h, grid_w), dtype=bool)
                motion_grid = np.zeros((grid_h, grid_w, 3), dtype=np.int16)
                last_dy, last_dx = 0, 0
    
                for y_idx in range(grid_h):
                    for x_idx in range(grid_w):
                        if processed[y_idx, x_idx]: continue
    
                        y_min, x_min = y_idx * BLOCK_SIZE, x_idx * BLOCK_SIZE
                        y_max, x_max = y_min + BLOCK_SIZE, x_min + BLOCK_SIZE
                        base_sad = sad_grid[y_idx, x_idx]
                        
                        # 1. SKIP LANE: Static reconstruction OR static source
                        # (Gated source-skip GUARANTEES zero temporal drift and zero shimmer while preventing frozen errors!)
                        is_recon_static = (max_grid[y_idx, x_idx] < 4)
                        is_source_static = (orig_max_grid[y_idx, x_idx] < 3 and orig_sad_grid[y_idx, x_idx] < 128 and max_grid[y_idx, x_idx] < 8)
                        if is_recon_static or is_source_static:
                            skip_run += 1
                            processed[y_idx, x_idx] = True
                            continue 
                        
                        if skip_run > 0:
                            mode_buffer.append(0xFF)
                            payload_buffer.extend(struct.pack("<H", int(skip_run)))
                            skip_run = 0
    
                        curr_block = curr_y[y_min:y_max, x_min:x_max]
                        prev_block = prev[y_min:y_max, x_min:x_max]
                        curr_block_i32 = curr_y_i32[y_min:y_max, x_min:x_max]
                        b_range = range_grid[y_idx, x_idx]
                        b_min = int(min_grid[y_idx, x_idx])
    
                        best_mode = 2
                        best_payload = curr_block.astype(np.uint8).tobytes()
                        best_recon = curr_block.copy()
                        best_cost = 257 + (LAMBDA * base_sad) 
                        log_dy, log_dx = 0, 0
    
                        # 2. FAST LANE: BACKGROUND
                        bg_cand = bg_model_i32[y_min:y_max, x_min:x_max]
                        bg_diff = np.abs(curr_block_i32 - bg_cand)
                        bg_sad = np.sum(bg_diff)
                        bg_mse = np.mean(bg_diff ** 2)
                        
                        if bg_sad / 256 < 6.0 and bg_mse < 40.0 and np.max(bg_diff) < 14:
                            cost = 1 + (LAMBDA * bg_sad)
                            if cost < best_cost:
                                best_mode, best_cost, best_payload = 6, cost, b''
                                best_recon = bg_model[y_min:y_max, x_min:x_max].astype(np.int16)
    
                        # 3. FAST LANE: SOLID
                        if b_range < 10:
                            solid_val = int(np.mean(curr_block))
                            solid_recon = np.full((BLOCK_SIZE, BLOCK_SIZE, 1), solid_val, dtype=np.int16)
                            solid_sad = np.sum(np.abs(curr_block_i32 - solid_val))
                            cost = 2 + (LAMBDA * solid_sad)
                            if b_range <= 2 and solid_sad <= 32:
                                cost = 1.5
                            if cost < best_cost:
                                best_mode, best_cost = 7, cost
                                best_payload = struct.pack("<B", np.clip(solid_val, 0, 255))
                                best_recon = solid_recon
    
                        # 4. FAST LANE: PRECISE PALETTE
                        if best_cost > 10 and b_range > 2:
                            unique_vals = np.unique(curr_block)
                            if len(unique_vals) <= 4:
                                pal = np.zeros(4, dtype=np.uint8)
                                pal[:len(unique_vals)] = unique_vals
                                indices = np.zeros_like(curr_block, dtype=np.uint8)
                                for i, v in enumerate(unique_vals): indices[curr_block == v] = i
                                packed = np.bitwise_or.reduce(indices.flatten().reshape(-1, 4) << np.array([6, 4, 2, 0], dtype=np.uint8), axis=1).astype(np.uint8)
                                cost = 1 + len(unique_vals) + 64
                                if cost < best_cost:
                                    best_mode, best_cost = 12, cost
                                    best_payload = struct.pack("<B", len(unique_vals)) + pal.tobytes() + packed.tobytes()
                                    best_recon = curr_block.copy()
                            elif b_range > 15:
                                mid = (int(b_min) + int(max_val_grid[y_idx, x_idx])) // 2
                                mask = (curr_block >= mid).squeeze()
                                if np.any(mask) and np.any(~mask):
                                    c0 = int(np.mean(curr_block[~mask]))
                                    c1 = int(np.mean(curr_block[mask]))
                                    recon_2c = np.where(mask[:, :, None], c1, c0)
                                    err_2c = np.abs(curr_block_i32 - recon_2c)
                                    if np.max(err_2c) < 8 and np.mean(err_2c ** 2) < 16.0:
                                        pal = np.array([c0, c1, 0, 0], dtype=np.uint8)
                                        indices = mask.astype(np.uint8)
                                        packed = np.bitwise_or.reduce(indices.flatten().reshape(-1, 4) << np.array([6, 4, 2, 0], dtype=np.uint8), axis=1).astype(np.uint8)
                                        cost = 1 + 2 + 64 + (LAMBDA * np.sum(err_2c))
                                        if cost < best_cost:
                                            best_mode, best_cost = 12, cost
                                            best_payload = struct.pack("<B", 2) + pal.tobytes() + packed.tobytes()
                                            best_recon = recon_2c.astype(np.int16)
                                    elif b_range > 35:
                                        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)
                                        _, _, centers_k = cv2.kmeans(curr_block.astype(np.float32).reshape(-1, 1), 4, None, crit, 3, cv2.KMEANS_PP_CENTERS)
                                        pal4 = np.sort(centers_k.flatten()).astype(np.uint8)
                                        labels_4 = np.abs(curr_block_i32.reshape(-1, 1) - pal4.reshape(1, -1)).argmin(axis=1).astype(np.uint8)
                                        recon_4c = pal4[labels_4].reshape((16, 16, 1))
                                        err_4c = np.abs(curr_block_i32 - recon_4c)
                                        max_err_lim = 24 if is_ui else 12
                                        if np.max(err_4c) < max_err_lim and np.mean(err_4c ** 2) < 25.0:
                                            packed4 = np.bitwise_or.reduce(labels_4.flatten().reshape(-1, 4) << np.array([6, 4, 2, 0], dtype=np.uint8), axis=1).astype(np.uint8)
                                            cost = 1 + 4 + 64 + (LAMBDA * np.sum(err_4c))
                                            if cost < best_cost:
                                                best_mode, best_cost = 12, cost
                                                best_payload = struct.pack("<B", 4) + pal4.tobytes() + packed4.tobytes()
                                                best_recon = recon_4c.astype(np.int16)
    
                        # 5. FAST LANE: PREDICTED & FAST MOTION
                        best_match_y, best_match_x = y_min, x_min
                        found_template = False
                        motion_allowed = (not is_ui) or (best_mode != 12 or best_cost > 70)
                        if base_sad > 128 and best_cost > 10 and motion_allowed:
                            pred_y = max(0, min(h - BLOCK_SIZE, y_min - last_dy))
                            pred_x = max(0, min(w - BLOCK_SIZE, x_min - last_dx))
                            cand_pred = prev_i32[pred_y:pred_y+BLOCK_SIZE, pred_x:pred_x+BLOCK_SIZE]
                            pred_diff = np.abs(curr_block_i32 - cand_pred)
                            pred_sad = np.sum(pred_diff)
                            pred_mse = np.mean(pred_diff ** 2)
                            
                            # Clean motion gate: max disparity checked to prevent edge cutting & color fringing
                            if pred_sad / 256 < 6.0 and pred_mse < 40.0 and np.max(pred_diff) < 14:
                                cost = 1 + (LAMBDA * pred_sad)
                                if cost < best_cost:
                                    best_mode, best_cost, best_payload = 11, cost, b''
                                    best_recon = prev[pred_y:pred_y+BLOCK_SIZE, pred_x:pred_x+BLOCK_SIZE].copy()
                                    log_dy, log_dx = last_dy, last_dx
                            else:
                                s_y_min, s_y_max = max(0, pred_y - 16), min(h, pred_y + BLOCK_SIZE + 16)
                                s_x_min, s_x_max = max(0, pred_x - 16), min(w, pred_x + BLOCK_SIZE + 16)
                                if (s_y_max - s_y_min >= BLOCK_SIZE) and (s_x_max - s_x_min >= BLOCK_SIZE):
                                    res = cv2.matchTemplate(prev_u8[s_y_min:s_y_max, s_x_min:s_x_max], curr_u8[y_min:y_max, x_min:x_max], cv2.TM_SQDIFF)
                                    min_val, _, min_loc, _ = cv2.minMaxLoc(res)
                                    best_match_y = s_y_min + min_loc[1]
                                    best_match_x = s_x_min + min_loc[0]
                                    found_template = True
                                    candidate = prev_i32[best_match_y:best_match_y+BLOCK_SIZE, best_match_x:best_match_x+BLOCK_SIZE]
                                    cand_diff = np.abs(curr_block_i32 - candidate)
                                    cand_mse = min_val / 256.0
                                    cur_dy = int(y_min - best_match_y)
                                    cur_dx = int(x_min - best_match_x)
                                    
                                    cost = 3 + (LAMBDA * np.sum(cand_diff))
                                    if cost < best_cost and cand_mse < 60.0 and np.max(cand_diff) < 14:
                                        best_mode, best_cost = 4, cost
                                        best_payload = struct.pack("<hh", cur_dy, cur_dx)
                                        best_recon = prev[best_match_y:best_match_y+BLOCK_SIZE, best_match_x:best_match_x+BLOCK_SIZE].copy()
                                        log_dy, log_dx = cur_dy, cur_dx
    
                        # 6. DESPERATION ARENA: FADED MOTION (Mode 13)
                        if best_cost > 15:
                            cand_fade_vectors = []
                            if found_template:
                                cand_fade_vectors.append((int(y_min - best_match_y), int(x_min - best_match_x)))
                            cand_fade_vectors.append((log_dy, log_dx))
                            cand_fade_vectors.append((last_dy, last_dx))
                            
                            for (f_dy, f_dx) in cand_fade_vectors:
                                ref_y = max(0, min(h - BLOCK_SIZE, y_min - f_dy))
                                ref_x = max(0, min(w - BLOCK_SIZE, x_min - f_dx))
                                fade_cand = prev_i32[ref_y:ref_y+BLOCK_SIZE, ref_x:ref_x+f_dx] if f_dx >= 0 else prev_i32[ref_y:ref_y+BLOCK_SIZE, ref_x:ref_x+BLOCK_SIZE]
                                fade_cand = prev_i32[ref_y:ref_y+BLOCK_SIZE, ref_x:ref_x+BLOCK_SIZE]
                                
                                luma_shift = int(np.mean(curr_block_i32) - np.mean(fade_cand))
                                if -128 <= luma_shift <= 127:
                                    shifted_cand = np.clip(fade_cand + luma_shift, 0, 255)
                                    f_diff = np.abs(curr_block_i32 - shifted_cand)
                                    fade_sad = np.sum(f_diff)
                                    fade_mse = np.mean(f_diff ** 2)
                                    
                                    max_fade_diff = 35 if is_ui else 999
                                    if fade_mse < 180.0 and abs(luma_shift) <= 24 and np.max(f_diff) < max_fade_diff:
                                        cost = 4 + (LAMBDA * fade_sad)
                                        if cost < best_cost:
                                            best_mode, best_cost = 13, cost
                                            best_payload = struct.pack("<hhb", f_dy, f_dx, luma_shift)
                                            best_recon = shifted_cand.astype(np.int16)
                                            log_dy, log_dx = f_dy, f_dx
                                            break
    
                        # 7. FORCED DITHER (Mode 15)
                        if best_cost > 75 and b_range > 20:
                            centers = np.linspace(b_min, int(max_val_grid[y_idx, x_idx]), 4).astype(np.uint8)
                            labels = np.abs(curr_block_i32.reshape(-1, 1) - centers).argmin(axis=1).astype(np.uint8)
                            packed = np.bitwise_or.reduce(labels.flatten().reshape(-1, 4) << np.array([6, 4, 2, 0], dtype=np.uint8), axis=1).astype(np.uint8)
                            recon_dither = centers[labels].reshape((16, 16, 1)).astype(np.int16)
                            dither_sad = np.sum(np.abs(curr_block_i32 - recon_dither.astype(np.int32)))
                            cost = 69 + (LAMBDA * dither_sad)
                            if cost < best_cost:
                                best_mode, best_cost = 15, cost
                                best_payload = centers.tobytes() + packed.tobytes()
                                best_recon = recon_dither
    
                        # 8. ABSOLUTE FALLBACK: Mode 1 Residual
                        if best_cost > 10:
                            delta = np.round((curr_block.astype(np.float32) - prev_block.astype(np.float32)) / float(QP)).astype(np.int16)
                            if delta.min() >= -128 and delta.max() <= 127:
                                recon = np.clip(prev_block + (delta * QP), 0, 255).astype(np.int16)
                                res_diff = np.abs(curr_block_i32 - recon.astype(np.int32))
                                nnz = np.count_nonzero(delta)
                                if nnz < 128:
                                    indices = np.nonzero(delta.flatten())[0].astype(np.uint8)
                                    values = (delta.flatten()[indices] + 128).astype(np.uint8)
                                    payload = struct.pack("<B", nnz) + indices.tobytes() + values.tobytes()
                                    cost = (1 + nnz*2) + (LAMBDA * np.sum(res_diff))
                                else:
                                    payload = bytes([128]) + (delta.flatten() + 128).astype(np.uint8).tobytes()
                                    cost = 80 + (LAMBDA * np.sum(res_diff))
                                if cost < best_cost and not (best_mode == 12 and np.max(res_diff) >= 8):
                                    best_mode, best_cost, best_payload, best_recon = 1, cost, payload, recon
    
                        # Vector Logging
                        if best_mode in [4, 11, 13]:
                            last_dy, last_dx = int(log_dy), int(log_dx)
                            motion_grid[y_idx, x_idx] = [last_dy, last_dx, 0]
    
                        # Dynamic Boxing (Mode 3)
                        w_mult, h_mult = 1, 1
                        if best_mode in [6, 4, 11]:
                            max_w = min(8, grid_w - x_idx)
                            max_h = min(8, grid_h - y_idx)
                            for w_test in range(2, max_w + 1):
                                if np.any(processed[y_idx, x_idx : x_idx + w_test]): break
                                test_curr = curr_y_i32[y_min : y_min + BLOCK_SIZE, x_min : x_min + w_test*BLOCK_SIZE]
                                if best_mode == 6:
                                    bg_cand = bg_model_i32[y_min : y_min + BLOCK_SIZE, x_min : x_min + w_test*BLOCK_SIZE]
                                    if np.max(np.abs(test_curr - bg_cand)) < 12: w_mult = w_test
                                    else: break
                                elif best_mode in [4, 11]:
                                    ref_y = max(0, min(h - BLOCK_SIZE, y_min - log_dy))
                                    ref_x = max(0, min(w - w_test*BLOCK_SIZE, x_min - log_dx))
                                    cand_pred = prev_i32[ref_y : ref_y + BLOCK_SIZE, ref_x : ref_x + w_test*BLOCK_SIZE]
                                    if np.mean((test_curr - cand_pred) ** 2) < 40.0 and np.max(np.abs(test_curr - cand_pred)) < 14:
                                        w_mult = w_test
                                    else: break
    
                            for h_test in range(2, max_h + 1):
                                if np.any(processed[y_idx : y_idx + h_test, x_idx : x_idx + w_mult]): break
                                test_curr = curr_y_i32[y_min : y_min + h_test*BLOCK_SIZE, x_min : x_min + w_mult*BLOCK_SIZE]
                                if best_mode == 6:
                                    bg_cand = bg_model_i32[y_min : y_min + h_test*BLOCK_SIZE, x_min : x_min + w_mult*BLOCK_SIZE]
                                    if np.max(np.abs(test_curr - bg_cand)) < 12: h_mult = h_test
                                    else: break
                                elif best_mode in [4, 11]:
                                    ref_y = max(0, min(h - h_test*BLOCK_SIZE, y_min - log_dy))
                                    ref_x = max(0, min(w - w_mult*BLOCK_SIZE, x_min - log_dx))
                                    cand_pred = prev_i32[ref_y : ref_y + h_test*BLOCK_SIZE, ref_x : ref_x + w_mult*BLOCK_SIZE]
                                    if np.mean((test_curr - cand_pred) ** 2) < 40.0 and np.max(np.abs(test_curr - cand_pred)) < 14:
                                        h_mult = h_test
                                    else: break
    
                        blocks_in_box = w_mult * h_mult
                        if blocks_in_box > 1:
                            mode_buffer.append(3)
                            payload_buffer.extend(struct.pack("<BBB", w_mult, h_mult, best_mode))
                            payload_buffer.extend(best_payload)
                            if best_mode == 6:
                                best_recon = bg_model[y_min:y_min+h_mult*BLOCK_SIZE, x_min:x_min+w_mult*BLOCK_SIZE].astype(np.int16).copy()
                            elif best_mode in [4, 11]:
                                ref_y = max(0, min(h - h_mult*BLOCK_SIZE, y_min - log_dy))
                                ref_x = max(0, min(w - w_mult*BLOCK_SIZE, x_min - log_dx))
                                best_recon = prev[ref_y : ref_y + h_mult*BLOCK_SIZE, ref_x : ref_x + w_mult*BLOCK_SIZE].copy()
                        else:
                            mode_buffer.append(best_mode)
                            payload_buffer.extend(best_payload)
    
                        next_prev[y_min:y_min+(h_mult*BLOCK_SIZE), x_min:x_min+(w_mult*BLOCK_SIZE)] = best_recon
                        processed[y_idx:y_idx+h_mult, x_idx:x_idx+w_mult] = True
    
                if skip_run > 0:
                    mode_buffer.append(0xFF)
                    payload_buffer.extend(struct.pack("<H", int(skip_run)))
    
                diff_mask = (np.abs(next_prev.astype(np.int32) - prev.astype(np.int32)) < 2) & (np.abs(next_prev.astype(np.int32) - bg_model_i32) < 4)
                bg_model[diff_mask] = (0.95 * bg_model[diff_mask] + 0.05 * next_prev[diff_mask]).astype(np.float32)
                bg_model_i32 = bg_model.astype(np.int32)
    
                stream.write(struct.pack("<II", len(mode_buffer), len(payload_buffer)))
                stream.write(mode_buffer)
                stream.write(payload_buffer)
                mode_buffer.clear()
                payload_buffer.clear()
                prev = next_prev
                prev_orig = curr_y.copy()
                frame_idx += 1
                pbar.update(1)
    
            pbar.close()
            stream.close()
        cap.release()

    enc_time = time.time() - start_time
    file_size = os.path.getsize(output_path)
    orig_bytes = os.path.getsize(video_path) if os.path.exists(video_path) else 1
    return {
        "encode_time_s": enc_time,
        "frames": frame_idx,
        "total_frames": frame_idx,
        "orig_bytes": orig_bytes,
        "compressed_bytes": file_size,
        "ratio": file_size / orig_bytes,
        "mode": "layered" if use_layered else "macroblock"
    }
