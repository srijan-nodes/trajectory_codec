import cv2
import numpy as np
import struct
import zstandard as zstd
import zlib
import time
from tqdm import tqdm
import os

def get_bg(cap, total_frames):
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(min(15, total_frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * (total_frames // 16))
        ret, f = cap.read()
        if ret: frames.append(f)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    bg = np.median(np.array(frames), axis=0).astype(np.uint8)
    return bg, cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

def extract_lines(diff_gray):
    _, mask = cv2.threshold(diff_gray, 40, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    lines = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 5: continue
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        pts = cnt.squeeze()
        if pts.ndim == 1: pts = pts.reshape(1, 2)
        dists = np.sqrt(np.sum((pts - [cx, cy])**2, axis=1))
        max_idx = np.argmax(dists)
        px, py = pts[max_idx]
        lines.append((cx, cy, px, py, 255, 255, 255, 2))
    return lines

def run_encoder(video_path, output_path, config):
    telemetry = {
        "v5_frames": 0, "i_frames": 0, "total_frames": 0,
        "v5_modes": {1: 0, 2: 0, 4: 0, 6: 0, 11: 0}
    }
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = h - (h % 16)
    w = w - (w % 16)
    
    bg_bgr, bg_clean = get_bg(cap, total_frames)
    
    cctx = zstd.ZstdCompressor(level=9)
    BLOCK_SIZE = 16
    grid_h, grid_w = h // BLOCK_SIZE, w // BLOCK_SIZE
    LAMBDA, QP = 0.015, 30
    
    telemetry = {"v5_frames": 0, "v6_frames": 0, "total_frames": 0}
    start_time = time.time()
    
    with open(output_path, "wb") as f:
        f.write(b'HYBD')
        f.write(struct.pack("<III", h, w, total_frames))
        
        comp_bg = zlib.compress(bg_clean.tobytes(), 9)
        f.write(struct.pack("<I", len(comp_bg)))
        f.write(comp_bg)
        
        prev_gray, prev_i32 = None, None
        bg_model = bg_clean.astype(np.float32)
        stream = cctx.stream_writer(f)
        
        mode_buffer = bytearray()
        payload_buffer = bytearray()
        
        pbar = tqdm(total=total_frames, desc="Enc HYBD", leave=False)
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            curr_y = cv2.cvtColor(frame[:h, :w], cv2.COLOR_BGR2GRAY)
            curr_y_16 = curr_y.reshape((h, w, 1)).astype(np.int16)
            curr_y_i32 = curr_y_16.astype(np.int32)
            
            if prev_gray is None:
                stream.write(struct.pack("<B", 0x00)) # I-Frame
                raw_bytes = curr_y.tobytes()
                stream.write(struct.pack("<I", len(raw_bytes)))
                stream.write(raw_bytes)
                prev_gray = curr_y.copy()
                prev_i32 = curr_y_i32.copy()
                pbar.update(1)
                continue
                
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_y, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_var = np.var(mag)
            mean_motion = np.mean(mag)
            
            # THE HEURISTIC
            is_chaos = motion_var > 1.0 and mean_motion > 0.5
            
            if is_chaos:
                telemetry["v6_frames"] += 1
                stream.write(struct.pack("<B", 0x06))
                
                diff = cv2.absdiff(curr_y, bg_clean)
                lines = extract_lines(diff)
                
                v6_payload = bytearray()
                v6_payload.extend(struct.pack("<H", len(lines)))
                
                next_prev = bg_clean.copy()
                for cx, cy, px, py, b, g, r, thick in lines:
                    v6_payload.extend(struct.pack("<HHHHBBBB", cx, cy, px, py, b, g, r, thick))
                    cv2.line(next_prev, (cx, cy), (px, py), 255, thickness=thick, lineType=cv2.LINE_AA)
                
                comp_v6 = zlib.compress(bytes(v6_payload), 9)
                stream.write(struct.pack("<I", len(comp_v6)))
                stream.write(comp_v6)
                
                prev_gray = next_prev.copy()
                prev_i32 = next_prev.reshape((h, w, 1)).astype(np.int32)
                
            else:
                telemetry["v5_frames"] += 1
                stream.write(struct.pack("<B", 0x05))
                next_prev = prev_i32.astype(np.int16).copy()
                
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
                        
                        curr_block = curr_y_16[y_min:y_max, x_min:x_max]
                        curr_block_i32 = curr_y_i32[y_min:y_max, x_min:x_max]
                        
                        if max_grid[y_idx, x_idx] < 4:
                            skip_run += 1
                            processed[y_idx, x_idx] = True
                            continue
                            
                        if skip_run > 0:
                            mode_buffer.append(0xFF)
                            payload_buffer.extend(struct.pack("<H", int(skip_run)))
                            skip_run = 0
                            
                        best_mode = 2
                        best_payload = curr_block.astype(np.uint8).tobytes()
                        best_recon = curr_block.copy()
                        best_cost = 257 + (LAMBDA * sad_grid[y_idx, x_idx])
                        log_dy, log_dx = 0, 0
                        
                        bg_cand = bg_model[y_min:y_max, x_min:x_max].astype(np.int32).reshape(BLOCK_SIZE, BLOCK_SIZE, 1)
                        bg_sad = np.sum(np.abs(curr_block_i32 - bg_cand))
                        if bg_sad / 256 < 6.0:
                            cost = 1 + (LAMBDA * bg_sad)
                            if cost < best_cost:
                                best_mode, best_cost, best_payload = 6, cost, b''
                                best_recon = bg_cand.astype(np.int16)
                                
                        if best_cost > 15:
                            pred_y = np.clip(y_min - last_dy, 0, h - BLOCK_SIZE)
                            pred_x = np.clip(x_min - last_dx, 0, w - BLOCK_SIZE)
                            cand_pred = prev_i32[pred_y:pred_y+BLOCK_SIZE, pred_x:pred_x+BLOCK_SIZE]
                            pred_sad = np.sum(np.abs(curr_block_i32 - cand_pred))
                            if pred_sad / 256 < 4.0:
                                cost = 1 + (LAMBDA * pred_sad)
                                if cost < best_cost:
                                    best_mode, best_cost, best_payload = 11, cost, b''
                                    best_recon = cand_pred.astype(np.int16).copy()
                                    log_dy, log_dx = last_dy, last_dx
                            else:
                                s_y_min, s_y_max = max(0, pred_y - 16), min(h, pred_y + BLOCK_SIZE + 16)
                                s_x_min, s_x_max = max(0, pred_x - 16), min(w, pred_x + BLOCK_SIZE + 16)
                                if (s_y_max - s_y_min >= BLOCK_SIZE) and (s_x_max - s_x_min >= BLOCK_SIZE):
                                    res = cv2.matchTemplate(prev_i32[s_y_min:s_y_max, s_x_min:s_x_max].astype(np.uint8), curr_block.astype(np.uint8), cv2.TM_SQDIFF)
                                    min_val, _, min_loc, _ = cv2.minMaxLoc(res)
                                    best_match_y = s_y_min + min_loc[1]
                                    best_match_x = s_x_min + min_loc[0]
                                    candidate = prev_i32[best_match_y:best_match_y+BLOCK_SIZE, best_match_x:best_match_x+BLOCK_SIZE]
                                    cost = 3 + (LAMBDA * np.sum(np.abs(curr_block_i32 - candidate)))
                                    if cost < best_cost:
                                        best_mode, best_cost = 4, cost
                                        log_dy, log_dx = int(y_min - best_match_y), int(x_min - best_match_x)
                                        best_payload = struct.pack("<hh", log_dy, log_dx)
                                        best_recon = candidate.astype(np.int16).copy()

                        if best_mode in [4, 11]:
                            last_dy, last_dx = int(log_dy), int(log_dx)

                        mode_buffer.append(best_mode)
                        payload_buffer.extend(best_payload)
                        next_prev[y_min:y_max, x_min:x_max] = best_recon
                        processed[y_idx, x_idx] = True
                        
                if skip_run > 0:
                    mode_buffer.append(0xFF)
                    payload_buffer.extend(struct.pack("<H", int(skip_run)))
                
                stream.write(struct.pack("<II", len(mode_buffer), len(payload_buffer)))
                stream.write(mode_buffer)
                stream.write(payload_buffer)
                mode_buffer.clear()
                payload_buffer.clear()
                
                prev_gray = next_prev.astype(np.uint8).squeeze()
                prev_i32 = next_prev.astype(np.int32)
                
            telemetry["total_frames"] += 1
            pbar.update(1)
            
        stream.close()
        pbar.close()
        
    return time.time() - start_time, telemetry

def decode_and_profile(original_video, encoded_path):
    dctx = zstd.ZstdDecompressor()
    cap = cv2.VideoCapture(original_video)
    
    frame_mses = []
    
    with open(encoded_path, "rb") as f:
        magic = f.read(4)
        h, w, total_frames = struct.unpack("<III", f.read(12))
        bg_size = struct.unpack("<I", f.read(4))[0]
        bg_clean = np.frombuffer(zlib.decompress(f.read(bg_size)), dtype=np.uint8).reshape((h, w))
        
        prev = None
        bg_model = bg_clean.copy().astype(np.float32).reshape(h, w, 1)
        stream = dctx.stream_reader(f)
        
        while True:
            ret, orig = cap.read()
            if not ret: break
            orig_y = cv2.cvtColor(orig[:h, :w], cv2.COLOR_BGR2GRAY)
            
            type_byte = stream.read(1)
            if not type_byte: break
            algo = struct.unpack("<B", type_byte)[0]
            
            if algo == 0x00:
                size = struct.unpack("<I", stream.read(4))[0]
                prev = np.frombuffer(stream.read(size), dtype=np.uint8).reshape((h, w, 1)).astype(np.int16)
            elif algo == 0x06:
                comp_size = struct.unpack("<I", stream.read(4))[0]
                raw = zlib.decompress(stream.read(comp_size))
                num_lines = struct.unpack("<H", raw[:2])[0]
                
                next_f = bg_clean.copy()
                offset = 2
                for _ in range(num_lines):
                    cx, cy, px, py, b, g, r, thick = struct.unpack("<HHHHBBBB", raw[offset:offset+12])
                    cv2.line(next_f, (cx, cy), (px, py), 255, thickness=thick, lineType=cv2.LINE_AA)
                    offset += 12
                prev = next_f.reshape(h, w, 1).astype(np.int16)
            elif algo == 0x05:
                next_f = prev.copy()
                grid_h, grid_w = h // 16, w // 16
                processed = np.zeros((grid_h, grid_w), dtype=bool)
                skip_r = 0
                last_dy, last_dx = 0, 0
                
                l_m, l_p = struct.unpack("<II", stream.read(8))
                m_data = stream.read(l_m)
                p_data = stream.read(l_p)
                m_idx, p_idx = 0, 0
                
                for y_idx in range(grid_h):
                    for x_idx in range(grid_w):
                        if processed[y_idx, x_idx]: continue
                        if skip_r > 0:
                            skip_r -= 1
                            processed[y_idx, x_idx] = True
                            continue
                            
                        y_min, x_min = y_idx * 16, x_idx * 16
                        flag = m_data[m_idx]; m_idx += 1
                        if flag == 0xFF:
                            skip_r = struct.unpack_from("<H", p_data, p_idx)[0]; p_idx += 2
                            skip_r -= 1
                            processed[y_idx, x_idx] = True
                            continue
                            
                        if flag == 6:
                            next_f[y_min:y_min+16, x_min:x_min+16] = bg_model[y_min:y_min+16, x_min:x_min+16].astype(np.int16)
                        elif flag == 4:
                            dy, dx = struct.unpack_from("<hh", p_data, p_idx); p_idx += 4
                            src_y = np.clip(y_min - dy, 0, h - 16)
                            src_x = np.clip(x_min - dx, 0, w - 16)
                            next_f[y_min:y_min+16, x_min:x_min+16] = prev[src_y:src_y+16, src_x:src_x+16].copy()
                            last_dy, last_dx = dy, dx
                        elif flag == 11:
                            src_y = np.clip(y_min - last_dy, 0, h - 16)
                            src_x = np.clip(x_min - last_dx, 0, w - 16)
                            next_f[y_min:y_min+16, x_min:x_min+16] = prev[src_y:src_y+16, src_x:src_x+16].copy()
                        elif flag == 2:
                            data = p_data[p_idx : p_idx + 256]; p_idx += 256
                            next_f[y_min:y_min+16, x_min:x_min+16] = np.frombuffer(data, dtype=np.uint8).reshape((16, 16, 1)).astype(np.int16)
                            
                        processed[y_idx, x_idx] = True
                        
                prev = next_f.copy()
            
            mse = np.mean((orig_y.astype(np.float32) - prev.squeeze().astype(np.float32)) ** 2)
            frame_mses.append(mse)
            
    return 0.0, frame_mses
