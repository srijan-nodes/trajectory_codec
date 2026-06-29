import cv2
import numpy as np
import struct
import zstandard as zstd
import os
from tqdm import tqdm

def dhash(block):
    small = cv2.resize(np.clip(block, 0, 255).astype(np.uint8), (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    return int(sum([1 << i for i, x in enumerate(diff.flatten()) if x]))

def generate_clean_plate(video_path, max_frames, h, w):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    step = max(1, max_frames // 45) 
    while count < max_frames:
        ret, frame = cap.read()
        if not ret: break
        if count % step == 0:
            y = cv2.cvtColor(frame[:h, :w], cv2.COLOR_BGR2GRAY)
            frames.append(y)
        count += 1
    cap.release()
    if not frames: return np.zeros((h, w, 1), dtype=np.uint8)
    return np.median(frames, axis=0).astype(np.uint8).reshape((h, w, 1))

# --- V4 DISCOVERY ENGINE ---
class TemporalTensor:
    def __init__(self, grid_h, grid_w, history_len=10):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.history_len = history_len
        # Stores [Mode, dy, dx, isActive]
        self.buffer = np.zeros((grid_h, grid_w, history_len, 4), dtype=np.int16)
        self.head = 0
        self.frame_count = 0

    def log_block(self, y_idx, x_idx, mode, dy=0, dx=0, is_active=1):
        self.buffer[y_idx, x_idx, self.head] = [mode, dy, dx, is_active]

    def advance_frame(self):
        self.head = (self.head + 1) % self.history_len
        self.frame_count += 1

    def cluster_regions(self):
        """Runs a relaxed Union-Find to discover cohesive behavioral regions."""
        if self.frame_count < self.history_len: return 0, 0
        
        parent = {}
        size = {}

        def find(i):
            if parent[i] == i: return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                if size[root_i] < size[root_j]: root_i, root_j = root_j, root_i
                parent[root_j] = root_i
                size[root_i] += size[root_j]

        # Initialize disjoint set for active blocks
        active_blocks = []
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                # If block was active (not skip) in at least 3 of last 10 frames
                if np.sum(self.buffer[y, x, :, 3]) > 3:
                    idx = y * self.grid_w + x
                    parent[idx] = idx
                    size[idx] = 1
                    active_blocks.append((y, x, idx))

        # Horizontal & Vertical Adjacency Checks
        for y, x, idx in active_blocks:
            neighbors = []
            if x + 1 < self.grid_w and (y * self.grid_w + x + 1) in parent: neighbors.append((y, x+1, y * self.grid_w + x + 1))
            if y + 1 < self.grid_h and ((y+1) * self.grid_w + x) in parent: neighbors.append((y+1, x, (y+1) * self.grid_w + x))

            for ny, nx, nidx in neighbors:
                # Cohesion Logic: Relaxed Equality over history
                hist_a = self.buffer[y, x]
                hist_b = self.buffer[ny, nx]
                
                # Check Mode Similarity (Must match exactly)
                if not np.array_equal(hist_a[:, 0], hist_b[:, 0]): continue
                
                # Check Motion Similarity (Relaxed L1 Norm <= 1)
                motion_diff = np.abs(hist_a[:, 1:3] - hist_b[:, 1:3])
                if np.max(np.sum(motion_diff, axis=1)) > 1: continue
                
                union(idx, nidx)

        # Analyze clusters
        cluster_sizes = {}
        for idx in parent.keys():
            root = find(idx)
            cluster_sizes[root] = cluster_sizes.get(root, 0) + 1

        valid_regions = [s for s in cluster_sizes.values() if s >= 4] # Min 4 blocks to be a "Region"
        if not valid_regions: return 0, 0
        
        return len(valid_regions), max(valid_regions)


def run_encoder(video_path, output_path, config, max_frames=150, QP=15, LAMBDA=0.5):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret: return {}
    
    h, w, _ = frame.shape
    h = h - (h % 16) if h % 16 != 0 else h
    w = w - (w % 16) if w % 16 != 0 else w
    
    cctx = zstd.ZstdCompressor(level=3)
    
    if config.get('background'):
        clean_plate = generate_clean_plate(video_path, max_frames, h, w).astype(np.int16)
    else:
        clean_plate = None

    with open(output_path, "wb") as f:
        f.write(b'NAM_V3')
        f.write(struct.pack("<III", int(h), int(w), 1)) 
        f.write(struct.pack("<H", int(QP)))
        
        f.write(struct.pack("<B", 1 if config.get('background') else 0))
        if config.get('background'):
            compressed_bg = cctx.compress(clean_plate.astype(np.uint8).tobytes())
            f.write(struct.pack("<I", len(compressed_bg)))
            f.write(compressed_bg)

        prev = cv2.cvtColor(frame[:h, :w], cv2.COLOR_BGR2GRAY).reshape((h, w, 1)).astype(np.int16)
        frame_idx = 0
        pbar = tqdm(total=max_frames, desc=f"Encoding: {config['name']}", unit="f", leave=False)

        telemetry = {
            "scene_cuts": 0, "blocks_total": 0, "blocks_skip": 0, 
            "blocks_intra": 0, "blocks_motion": 0, "blocks_solid": 0, 
            "blocks_spatial": 0, "blocks_background": 0,
            "motion_searches_performed": 0, "dynamic_boxes": 0,
            # V4 Telemetry
            "v4_regions_discovered": 0, "v4_max_region_size": 0
        }
        BLOCK_SIZE = 16
        
        # Initialize V4 Tensor
        grid_h = h // BLOCK_SIZE
        grid_w = w // BLOCK_SIZE
        temporal_tensor = TemporalTensor(grid_h, grid_w)

        while frame_idx < max_frames:
            if frame_idx > 0:
                ret, frame = cap.read()
                if not ret: break
            
            curr_y = cv2.cvtColor(frame[:h, :w], cv2.COLOR_BGR2GRAY).reshape((h, w, 1)).astype(np.int16)
            next_prev = prev.copy()
            
            intra_hash_table = {}
            mode_buffer, payload_buffer = bytearray(), bytearray()
            skip_run = 0

            processed = np.zeros((grid_h, grid_w), dtype=bool)

            is_scene_change = False
            if frame_idx > 0 and np.mean(np.abs(curr_y.astype(np.float32) - prev.astype(np.float32))) > 25.0:
                is_scene_change = True
                telemetry["scene_cuts"] += 1

            if frame_idx == 0 or is_scene_change:
                compressed = cctx.compress(curr_y.astype(np.uint8).tobytes())
                f.write(struct.pack("<B", 0))
                f.write(struct.pack("<I", len(compressed)))
                f.write(compressed)
                prev = curr_y.copy()
            else:
                f.write(struct.pack("<B", 1))
                for y_idx in range(grid_h):
                    for x_idx in range(grid_w):
                        if processed[y_idx, x_idx]: 
                            continue

                        y_min, x_min = y_idx * BLOCK_SIZE, x_idx * BLOCK_SIZE
                        y_max, x_max = y_min + BLOCK_SIZE, x_min + BLOCK_SIZE
                        
                        curr_block = curr_y[y_min:y_max, x_min:x_max]
                        prev_block = prev[y_min:y_max, x_min:x_max]
                        
                        base_sad = np.sum(np.abs(curr_block.astype(np.int32) - prev_block.astype(np.int32)))
                        
                        if base_sad / 256 < 2.0:
                            skip_run += 1
                            telemetry["blocks_skip"] += 1
                            telemetry["blocks_total"] += 1
                            processed[y_idx, x_idx] = True
                            temporal_tensor.log_block(y_idx, x_idx, mode=0xFF, is_active=0)
                            continue 
                        
                        if skip_run > 0:
                            mode_buffer.append(0xFF)
                            payload_buffer.extend(struct.pack("<H", int(skip_run)))
                            skip_run = 0

                        # SPATIAL DEFAULT
                        delta = np.round((curr_block - prev_block).astype(np.float32) / QP).astype(np.int16)
                        delta[np.abs(delta) <= 1] = 0 
                        
                        if delta.min() >= -128 and delta.max() <= 127:
                            best_mode, best_payload = 1, (delta + 128).astype(np.uint8).tobytes()
                        else:
                            best_mode, best_payload = 2, delta.astype(np.int16).tobytes()
                            
                        best_recon = prev_block + (delta * QP)
                        best_cost = len(best_payload) + (LAMBDA * np.sum(np.abs(delta)))
                        log_dy, log_dx = 0, 0

                        # MODE 6: GLOBAL BACKGROUND REVEAL
                        if config.get('background'):
                            bg_block = clean_plate[y_min:y_max, x_min:x_max]
                            bg_sad = np.sum(np.abs(curr_block.astype(np.int32) - bg_block.astype(np.int32)))
                            if bg_sad / 256 < 2.0:
                                cost = 1 + (LAMBDA * bg_sad)
                                if cost < best_cost:
                                    best_mode, best_cost = 6, cost
                                    best_payload = b'' 
                                    best_recon = bg_block.copy()
                            
                        # SOLID 
                        if config.get('palette'):
                            if (int(np.max(curr_block)) - int(np.min(curr_block))) < 10:
                                solid_val = np.mean(curr_block)
                                solid_recon = np.full((BLOCK_SIZE, BLOCK_SIZE, 1), solid_val, dtype=np.int16)
                                solid_sad = np.sum(np.abs(curr_block.astype(np.int32) - solid_recon.astype(np.int32)))
                                cost = 2 + (LAMBDA * solid_sad)
                                if cost < best_cost:
                                    best_mode, best_cost = 7, cost
                                    best_payload = struct.pack("<B", int(np.clip(solid_val, 0, 255)))
                                    best_recon = solid_recon

                        # INTRA-COPY
                        if config.get('cache'):
                            b_hash = dhash(curr_block)
                            if b_hash in intra_hash_table:
                                for src_y, src_x in intra_hash_table[b_hash][-8:]:
                                    candidate = next_prev[src_y:src_y+BLOCK_SIZE, src_x:src_x+BLOCK_SIZE]
                                    intra_sad = np.sum(np.abs(curr_block.astype(np.int32) - candidate.astype(np.int32)))
                                    cost = 4 + (LAMBDA * intra_sad)
                                    if cost < best_cost:
                                        best_mode, best_cost = 5, cost
                                        best_payload = struct.pack("<HH", int(src_y), int(src_x))
                                        best_recon = candidate.copy()

                        # MOTION
                        if config.get('motion') and base_sad > 256:
                            telemetry["motion_searches_performed"] += 1
                            s_y_min, s_y_max = max(0, y_min - 32), min(h, y_max + 32)
                            s_x_min, s_x_max = max(0, x_min - 32), min(w, x_max + 32)
                            res = cv2.matchTemplate(prev[s_y_min:s_y_max, s_x_min:s_x_max].astype(np.float32), curr_block.astype(np.float32), cv2.TM_SQDIFF_NORMED)
                            min_val, _, min_loc, _ = cv2.minMaxLoc(res)
                            if min_val < 0.1:
                                best_y, best_x = s_y_min + min_loc[1], s_x_min + min_loc[0]
                                candidate = prev[best_y:best_y+BLOCK_SIZE, best_x:best_x+BLOCK_SIZE].copy()
                                cost = 4 + (LAMBDA * np.sum(np.abs(curr_block.astype(np.int32) - candidate.astype(np.int32))))
                                if cost < best_cost:
                                    best_mode, best_cost = 4, cost
                                    best_payload = struct.pack("<hh", int(y_min - best_y), int(x_min - best_x))
                                    best_recon = candidate
                                    log_dy, log_dx = int(y_min - best_y), int(x_min - best_x)

                        # --- DYNAMIC BOXING ---
                        w_mult, h_mult = 1, 1
                        if config.get('dynamic_boxing') and best_mode in [4, 6, 7]:
                            max_w = grid_w - x_idx
                            max_h = grid_h - y_idx
                            
                            for w_test in range(2, max_w + 1):
                                conflict = False
                                for yt in range(h_mult):
                                    if processed[y_idx + yt, x_idx + w_test - 1]:
                                        conflict = True; break
                                if conflict: break
                                
                                test_curr = curr_y[y_min : y_min + h_mult*BLOCK_SIZE, x_min : x_min + w_test*BLOCK_SIZE]
                                if best_mode == 7:
                                    solid_recon_test = np.full((h_mult*BLOCK_SIZE, w_test*BLOCK_SIZE, 1), struct.unpack("<B", best_payload)[0], dtype=np.int16)
                                    if np.max(np.abs(test_curr.astype(np.int32) - solid_recon_test.astype(np.int32))) < 12: 
                                        w_mult = w_test
                                    else: break
                                elif best_mode == 6:
                                    bg_cand = clean_plate[y_min : y_min + h_mult*BLOCK_SIZE, x_min : x_min + w_test*BLOCK_SIZE]
                                    if np.max(np.abs(test_curr.astype(np.int32) - bg_cand.astype(np.int32))) < 15:
                                        w_mult = w_test
                                    else: break
                                elif best_mode == 4:
                                    dy, dx = struct.unpack("<hh", best_payload)
                                    sy1, sy2 = y_min - dy, y_min + h_mult*BLOCK_SIZE - dy
                                    sx1, sx2 = x_min - dx, x_min + w_test*BLOCK_SIZE - dx
                                    if sy1 >= 0 and sy2 <= h and sx1 >= 0 and sx2 <= w:
                                        cand = prev[sy1:sy2, sx1:sx2]
                                        if np.max(np.abs(test_curr.astype(np.int32) - cand.astype(np.int32))) < 15: w_mult = w_test
                                        else: break
                                    else: break

                            for h_test in range(2, max_h + 1):
                                conflict = False
                                for xt in range(w_mult):
                                    if processed[y_idx + h_test - 1, x_idx + xt]:
                                        conflict = True; break
                                if conflict: break
                                
                                test_curr = curr_y[y_min : y_min + h_test*BLOCK_SIZE, x_min : x_min + w_mult*BLOCK_SIZE]
                                if best_mode == 7:
                                    solid_recon_test = np.full((h_test*BLOCK_SIZE, w_mult*BLOCK_SIZE, 1), struct.unpack("<B", best_payload)[0], dtype=np.int16)
                                    if np.max(np.abs(test_curr.astype(np.int32) - solid_recon_test.astype(np.int32))) < 12: 
                                        h_mult = h_test
                                    else: break
                                elif best_mode == 6:
                                    bg_cand = clean_plate[y_min : y_min + h_test*BLOCK_SIZE, x_min : x_min + w_mult*BLOCK_SIZE]
                                    if np.max(np.abs(test_curr.astype(np.int32) - bg_cand.astype(np.int32))) < 15:
                                        h_mult = h_test
                                    else: break
                                elif best_mode == 4:
                                    dy, dx = struct.unpack("<hh", best_payload)
                                    sy1, sy2 = y_min - dy, y_min + h_test*BLOCK_SIZE - dy
                                    sx1, sx2 = x_min - dx, x_min + w_mult*BLOCK_SIZE - dx
                                    if sy1 >= 0 and sy2 <= h and sx1 >= 0 and sx2 <= w:
                                        cand = prev[sy1:sy2, sx1:sx2]
                                        if np.max(np.abs(test_curr.astype(np.int32) - cand.astype(np.int32))) < 15: h_mult = h_test
                                        else: break
                                    else: break

                        blocks_count = w_mult * h_mult
                        bh = dhash(best_recon[:16, :16])
                        intra_hash_table.setdefault(bh, []).append((y_min, x_min))

                        if blocks_count > 1:
                            telemetry["dynamic_boxes"] += 1
                            mode_buffer.append(3) 
                            
                            if best_mode == 7:
                                final_block = curr_y[y_min : y_min + h_mult*16, x_min : x_min + w_mult*16]
                                best_payload = struct.pack("<B", int(np.clip(np.mean(final_block), 0, 255)))
                                best_recon = np.full((h_mult*16, w_mult*16, 1), struct.unpack("<B", best_payload)[0], dtype=np.int16)
                            elif best_mode == 6:
                                best_recon = clean_plate[y_min : y_min + h_mult*16, x_min : x_min + w_mult*16].copy()
                            elif best_mode == 4:
                                dy, dx = struct.unpack("<hh", best_payload)
                                best_recon = prev[y_min-dy : y_min+(h_mult*16)-dy, x_min-dx : x_min+(w_mult*16)-dx].copy()
                                
                            payload_buffer.extend(struct.pack("<BBB", w_mult, h_mult, best_mode) + best_payload)
                            processed[y_idx : y_idx + h_mult, x_idx : x_idx + w_mult] = True
                            next_prev[y_min : y_min + h_mult*16, x_min : x_min + w_mult*16] = best_recon
                        else:
                            mode_buffer.append(best_mode)
                            payload_buffer.extend(best_payload)
                            processed[y_idx, x_idx] = True
                            next_prev[y_min:y_min+16, x_min:x_min+16] = best_recon

                        # Log to V4 Tensor
                        for yt in range(h_mult):
                            for xt in range(w_mult):
                                temporal_tensor.log_block(y_idx + yt, x_idx + xt, best_mode, log_dy, log_dx, is_active=1)

                        telemetry["blocks_total"] += blocks_count
                        if best_mode == 1 or best_mode == 2: telemetry["blocks_spatial"] += blocks_count
                        elif best_mode == 4: telemetry["blocks_motion"] += blocks_count
                        elif best_mode == 5: telemetry["blocks_intra"] += blocks_count
                        elif best_mode == 6: telemetry["blocks_background"] += blocks_count
                        elif best_mode == 7: telemetry["blocks_solid"] += blocks_count
                
                if skip_run > 0:
                    mode_buffer.append(0xFF)
                    payload_buffer.extend(struct.pack("<H", int(skip_run)))
                    
                comp_modes = cctx.compress(bytes(mode_buffer))
                comp_payloads = cctx.compress(bytes(payload_buffer))
                f.write(struct.pack("<II", len(comp_modes), len(comp_payloads)))
                f.write(comp_modes + comp_payloads)
                prev = next_prev
            
            # V4 Discovery Cycle
            if frame_idx % 5 == 0 and config.get('background'): # Run on the heaviest config
                regions, max_size = temporal_tensor.cluster_regions()
                if regions > telemetry["v4_regions_discovered"]:
                    telemetry["v4_regions_discovered"] = regions
                if max_size > telemetry["v4_max_region_size"]:
                    telemetry["v4_max_region_size"] = max_size

            temporal_tensor.advance_frame()
            frame_idx += 1
            pbar.update(1)
        pbar.close()
    return telemetry

# ... [calculate_mse function remains exactly the same] ...
def calculate_mse(original_video, encoded_path, max_frames=150):
    dctx = zstd.ZstdDecompressor()
    cap = cv2.VideoCapture(original_video)
    mse_total = 0
    BLOCK_SIZE = 16
    
    with open(encoded_path, "rb") as f:
        magic = f.read(6)
        h, w, _ = struct.unpack("<III", f.read(12))
        QP = struct.unpack("<H", f.read(2))[0]
        
        has_bg = struct.unpack("<B", f.read(1))[0]
        if has_bg:
            bg_size = struct.unpack("<I", f.read(4))[0]
            clean_plate = np.frombuffer(dctx.decompress(f.read(bg_size)), dtype=np.uint8).reshape((h, w, 1)).astype(np.int16)
        else:
            clean_plate = None

        prev = None
        frame_idx = 0

        while frame_idx < max_frames:
            ret, orig_frame = cap.read()
            if not ret: break
            orig_y = cv2.cvtColor(orig_frame[:h, :w], cv2.COLOR_BGR2GRAY)
            
            type_byte = f.read(1)
            if not type_byte: break 
            frame_type = struct.unpack("<B", type_byte)[0]

            if frame_type == 0:
                size = struct.unpack("<I", f.read(4))[0]
                prev = np.frombuffer(dctx.decompress(f.read(size)), dtype=np.uint8).reshape((h, w, 1)).astype(np.int16)
            elif frame_type == 1:
                len_modes, len_payloads = struct.unpack("<II", f.read(8))
                mode_data = dctx.decompress(f.read(len_modes))
                payload_data = dctx.decompress(f.read(len_payloads))
                
                mode_idx, pay_idx = 0, 0
                next_frame = prev.copy()
                
                grid_h, grid_w = h // BLOCK_SIZE, w // BLOCK_SIZE
                processed = np.zeros((grid_h, grid_w), dtype=bool)
                skip_remaining = 0
                
                for y_idx in range(grid_h):
                    for x_idx in range(grid_w):
                        if processed[y_idx, x_idx]: continue
                        
                        if skip_remaining > 0:
                            skip_remaining -= 1
                            processed[y_idx, x_idx] = True
                            continue
                        
                        y_min, x_min = y_idx * BLOCK_SIZE, x_idx * BLOCK_SIZE
                        flag = mode_data[mode_idx]; mode_idx += 1
                        
                        if flag == 0xFF:
                            skip_remaining = struct.unpack_from("<H", payload_data, pay_idx)[0]
                            pay_idx += 2
                            skip_remaining -= 1 
                            processed[y_idx, x_idx] = True
                            continue 
                        
                        w_mult, h_mult, current_mode = 1, 1, flag
                        if flag == 3: 
                            w_mult, h_mult, current_mode = struct.unpack_from("<BBB", payload_data, pay_idx)
                            pay_idx += 3
                            
                        y_max, x_max = y_min + (h_mult * BLOCK_SIZE), x_min + (w_mult * BLOCK_SIZE)
                        
                        if current_mode == 7:
                            val = payload_data[pay_idx]; pay_idx += 1
                            next_frame[y_min:y_max, x_min:x_max] = val
                        elif current_mode == 6: # BACKGROUND MODE
                            next_frame[y_min:y_max, x_min:x_max] = clean_plate[y_min:y_max, x_min:x_max].copy()
                        elif current_mode == 5:
                            src_y, src_x = struct.unpack_from("<HH", payload_data, pay_idx); pay_idx += 4
                            next_frame[y_min:y_max, x_min:x_max] = next_frame[src_y:src_y+(h_mult*16), src_x:src_x+(w_mult*16)].copy()
                        elif current_mode == 4:
                            dy, dx = struct.unpack_from("<hh", payload_data, pay_idx); pay_idx += 4
                            src_y, src_x = y_min - dy, x_min - dx
                            next_frame[y_min:y_max, x_min:x_max] = prev[src_y:src_y+(h_mult*16), src_x:src_x+(w_mult*16)].copy()
                        elif current_mode == 1:
                            data = payload_data[pay_idx : pay_idx + 256]; pay_idx += 256
                            quantized = (np.frombuffer(data, dtype=np.uint8).astype(np.int16) - 128)
                            next_frame[y_min:y_max, x_min:x_max] += (quantized.reshape((BLOCK_SIZE, BLOCK_SIZE, 1)) * QP)
                        elif current_mode == 2:
                            data = payload_data[pay_idx : pay_idx + 512]; pay_idx += 512
                            quantized = np.frombuffer(data, dtype=np.int16)
                            next_frame[y_min:y_max, x_min:x_max] += (quantized.reshape((BLOCK_SIZE, BLOCK_SIZE, 1)) * QP)
                            
                        processed[y_idx : y_idx + h_mult, x_idx : x_idx + w_mult] = True
                            
                prev = next_frame

            recon_y = np.clip(prev, 0, 255).astype(np.uint8).reshape((h, w))
            mse_total += np.mean((orig_y.astype(np.float32) - recon_y.astype(np.float32)) ** 2)
            frame_idx += 1
            
    cap.release()
    return mse_total / frame_idx