import cv2
import numpy as np
import struct
import zstandard as zstd
import os
from tqdm import tqdm

def run_encoder(video_path, output_path, config, max_frames=150, QP=15):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret: return
    h, w, _ = frame.shape
    h, w = h - (h % 2), w - (w % 2)
    cctx = zstd.ZstdCompressor(level=3)

    curr_i420 = cv2.cvtColor(frame[:h, :w], cv2.COLOR_BGR2YUV_I420)
    
    palette = np.zeros(16, dtype=np.uint8)
    if config['palette']:
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
        pbar = tqdm(total=max_frames, desc=f"Encoding: {config['name']}", unit="f")

        while frame_idx < max_frames:
            if frame_idx > 0:
                ret, frame = cap.read()
                if not ret: break
            curr_i = cv2.cvtColor(frame[:h, :w], cv2.COLOR_BGR2YUV_I420).reshape((int(h * 1.5), w, 1)).astype(np.int16)

            if frame_idx == 0:
                compressed = cctx.compress(curr_i.astype(np.uint8).tobytes())
                f.write(struct.pack("B", 0))
                f.write(struct.pack("I", len(compressed)))
                f.write(compressed)
                prev = curr_i
            else:
                quantized = np.round((curr_i - prev) / QP).astype(np.int16)
                mask = cv2.dilate(np.any(quantized != 0, axis=2).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
                num_labels, _, component_stats, _ = cv2.connectedComponentsWithStats(mask)
                components = []
                next_prev = prev.copy()

                for label in range(1, num_labels):
                    x_min, y_min, w_box, h_box, area = component_stats[label][:5]
                    if area < 100: continue
                    y_max, x_max = y_min + h_box - 1, x_min + w_box - 1
                    cropped_curr = curr_i[y_min:y_max+1, x_min:x_max+1]
                    flag = -1

                    # UPGRADE 1: FUZZY CACHE (Tolerance + Snap-to-Fit)
                    if config['cache'] and flag == -1:
                        for idx, cached_block in enumerate(entity_cache):
                            ch, cw, _ = cached_block.shape
                            if abs(ch - h_box) <= 4 and abs(cw - w_box) <= 4:
                                resized_cache = cv2.resize(cached_block, (w_box, h_box), interpolation=cv2.INTER_NEAREST).reshape(h_box, w_box, 1)
                                if np.mean((resized_cache.astype(np.float32) - cropped_curr.astype(np.float32))**2) < 10.0:
                                    flag = 5
                                    payload = struct.pack("B", idx)
                                    next_prev[y_min:y_max+1, x_min:x_max+1] = resized_cache
                                    recon_block = resized_cache.copy()
                                    break

                    # UPGRADE 2: DEEP MOTION (Expanded Radius + Cross-Correlation)
                    if config['motion'] and flag == -1:
                        s_y_min, s_y_max = max(0, y_min - 40), min(int(h * 1.5), y_max + 41)
                        s_x_min, s_x_max = max(0, x_min - 40), min(w, x_max + 41)
                        res = cv2.matchTemplate(prev[s_y_min:s_y_max, s_x_min:s_x_max].astype(np.float32), cropped_curr.astype(np.float32), cv2.TM_CCORR_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(res)
                        if max_val > 0.94: # 94% Structural Correlation
                            flag = 4
                            best_y, best_x = s_y_min + max_loc[1], s_x_min + max_loc[0]
                            payload = struct.pack("hh", y_min - best_y, x_min - best_x)
                            recon_block = prev[best_y:best_y+h_box, best_x:best_x+w_box].copy()
                            next_prev[y_min:y_max+1, x_min:x_max+1] = recon_block

                    # UPGRADE 3: RLE PALETTE (Run-Length Encoding Entropy Crunch)
                    if config['palette'] and flag == -1:
                        if np.max(np.var(cropped_curr, axis=(0, 1))) > 12.0:
                            flag = 8
                            diffs = np.abs(cropped_curr - palette.reshape((1, 1, 16)))
                            indices = np.argmin(diffs, axis=-1).astype(np.uint8).flatten()
                            
                            rle_data = bytearray()
                            if len(indices) > 0:
                                current_idx = indices[0]
                                count = 1
                                for idx_val in indices[1:]:
                                    if idx_val == current_idx and count < 255:
                                        count += 1
                                    else:
                                        rle_data.extend([current_idx, count])
                                        current_idx = idx_val
                                        count = 1
                                rle_data.extend([current_idx, count])
                                
                            payload = cctx.compress(bytes(rle_data))
                            recon_block = palette[indices[:h_box * w_box]].reshape((h_box, w_box, 1)).astype(np.int16)
                            next_prev[y_min:y_max+1, x_min:x_max+1] = recon_block

                    if flag == -1:
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

                if len(components) == 0:
                    f.write(struct.pack("B", 2))
                else:
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
                    elif flag == 8:
                        rle_bytes = dctx.decompress(raw_payload)
                        flat_idx = np.empty(h_box * w_box, dtype=np.uint8)
                        pos = 0
                        for i in range(0, len(rle_bytes), 2):
                            idx_val, count = rle_bytes[i], rle_bytes[i+1]
                            flat_idx[pos:pos+count] = idx_val
                            pos += count
                        recon_block = palette[flat_idx].reshape((h_box, w_box, 1)).astype(np.int16)
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

if __name__ == "__main__":
    video_file = "ball.mp4" 
    
    test_configs = [
        {"name": "V2 Baseline (Spatial)", "motion": False, "cache": False, "palette": False},
        {"name": "V2 Deep Motion (CCORR)", "motion": True, "cache": False, "palette": False},
        {"name": "V2 Fuzzy Cache (Snap)", "motion": False, "cache": True, "palette": False},
        {"name": "V2 RLE Palette", "motion": False, "cache": False, "palette": True}
    ]

    results = []
    print("\n🔬 STARTING V2 ABLATION STUDY...\n")
    
    for i, config in enumerate(test_configs):
        out_file = f"test_v2_{i}.nam_lab"
        run_encoder(video_file, out_file, config, max_frames=150)
        mse = calculate_mse(video_file, out_file, max_frames=150)
        size_kb = os.path.getsize(out_file) / 1024
        results.append((config['name'], size_kb, mse))
        os.remove(out_file)

    print("\n" + "="*50)
    print(f"{'FEATURE PROFILE':<25} | {'SIZE (150 frames)':<15} | {'QUALITY (MSE)'}")
    print("-" * 50)
    for name, size, mse in results:
        print(f"{name:<25} | {size:>10.2f} KB   | {mse:>10.2f}")
    print("="*50 + "\n")