import cv2
import numpy as np
import struct
import zstandard as zstd
import os
import time

# Import the V5 encoder from your existing file
from lab_v5 import run_encoder

MODE_NAMES = {
    0: "Skip (Static/BG)",
    1: "Spatial Delta",
    2: "Raw Bytes (Fallback)",
    3: "Dynamic Box (Merge)",
    4: "Fast Motion",
    6: "Adaptive Background",
    7: "Solid Color",
    9: "Low-Freq DCT (4x4)",
    10: "Mid-Freq DCT (8x8)",
    11: "Predicted Motion",
    12: "Precise Palette (4-Color)",
    13: "Faded Motion (Luma Shift)",
    14: "Subsampled (8x8 Chunk)",
    15: "Forced Dither (K-Means)",
    255: "I-Frame (Raw Keyframe)"
}

def decode_and_investigate(original_video, encoded_path):
    dctx = zstd.ZstdDecompressor()
    cap = cv2.VideoCapture(original_video)
    BLOCK_SIZE = 16
    
    # Tracking dictionary
    mode_stats = {k: {"count": 0, "sum_mse": 0.0, "max_mse": 0.0} for k in range(256)}
    
    with open(encoded_path, "rb") as f:
        magic = f.read(6)
        h, w, total_frames = struct.unpack("<III", f.read(12))
        QP = struct.unpack("<H", f.read(2))[0]
        has_bg = struct.unpack("<B", f.read(1))[0]
        
        grid_h, grid_w = h // BLOCK_SIZE, w // BLOCK_SIZE
        prev = None
        bg_model = None

        while True:
            ret, orig_frame = cap.read()
            if not ret: break
            orig_y = cv2.cvtColor(orig_frame[:h, :w], cv2.COLOR_BGR2GRAY).astype(np.float32)
            
            type_byte = f.read(1)
            if not type_byte: break 
            frame_type = struct.unpack("<B", type_byte)[0]

            frame_modes = np.zeros((grid_h, grid_w), dtype=np.uint8) 

            if frame_type == 0:
                size = struct.unpack("<I", f.read(4))[0]
                prev = np.frombuffer(dctx.decompress(f.read(size)), dtype=np.uint8).reshape((h, w, 1)).astype(np.int16)
                bg_model = prev.copy().astype(np.float32)
                frame_modes.fill(255) 
                
            elif frame_type == 1:
                next_frame = prev.copy()
                processed = np.zeros((grid_h, grid_w), dtype=bool)
                last_dy, last_dx = 0, 0
                skip_remaining = 0
                
                len_modes, len_payloads = struct.unpack("<II", f.read(8))
                mode_data = dctx.decompress(f.read(len_modes))
                payload_data = dctx.decompress(f.read(len_payloads))
                mode_idx, pay_idx = 0, 0
                
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
                        
                        frame_modes[y_idx : y_idx + h_mult, x_idx : x_idx + w_mult] = current_mode
                        
                        if current_mode == 7:
                            val = payload_data[pay_idx]; pay_idx += 1
                            next_frame[y_min:y_max, x_min:x_max] = val
                        elif current_mode == 6:
                            next_frame[y_min:y_max, x_min:x_max] = bg_model[y_min:y_max, x_min:x_max].astype(np.int16)
                        elif current_mode == 4:
                            dy, dx = struct.unpack_from("<hh", payload_data, pay_idx); pay_idx += 4
                            src_y = np.clip(y_min - dy, 0, h - (h_mult * BLOCK_SIZE))
                            src_x = np.clip(x_min - dx, 0, w - (w_mult * BLOCK_SIZE))
                            next_frame[y_min:y_max, x_min:x_max] = prev[src_y:src_y+(h_mult*16), src_x:src_x+(w_mult*16)].copy()
                            last_dy, last_dx = int(dy), int(dx)
                        elif current_mode == 11:
                            src_y = np.clip(y_min - last_dy, 0, h - (h_mult * BLOCK_SIZE))
                            src_x = np.clip(x_min - last_dx, 0, w - (w_mult * BLOCK_SIZE))
                            next_frame[y_min:y_max, x_min:x_max] = prev[src_y:src_y+(h_mult*16), src_x:src_x+(w_mult*16)].copy()
                        elif current_mode == 13: 
                            dy, dx, shift = struct.unpack_from("<hhb", payload_data, pay_idx); pay_idx += 5
                            src_y = np.clip(y_min - dy, 0, h - (h_mult * BLOCK_SIZE))
                            src_x = np.clip(x_min - dx, 0, w - (w_mult * BLOCK_SIZE))
                            cand = prev[src_y:src_y+(h_mult*16), src_x:src_x+(w_mult*16)]
                            next_frame[y_min:y_max, x_min:x_max] = np.clip(cand.astype(np.int32) + shift, 0, 255).astype(np.int16)
                            last_dy, last_dx = int(dy), int(dx)
                        elif current_mode == 9:
                            flat_coeffs = np.frombuffer(payload_data[pay_idx : pay_idx + 32], dtype=np.float16)
                            pay_idx += 32
                            dct_low = np.zeros((16, 16), dtype=np.float32)
                            dct_low[:4, :4] = flat_coeffs.reshape((4, 4)).astype(np.float32)
                            recon_low = cv2.idct(dct_low)
                            next_frame[y_min:y_max, x_min:x_max] = recon_low.astype(np.int16).reshape((16, 16, 1))
                        elif current_mode == 10:
                            flat_coeffs = np.frombuffer(payload_data[pay_idx : pay_idx + 128], dtype=np.float16)
                            pay_idx += 128
                            dct_mid = np.zeros((16, 16), dtype=np.float32)
                            dct_mid[:8, :8] = flat_coeffs.reshape((8, 8)).astype(np.float32)
                            recon_mid = cv2.idct(dct_mid)
                            next_frame[y_min:y_max, x_min:x_max] = recon_mid.astype(np.int16).reshape((16, 16, 1))
                        elif current_mode == 15: 
                            palette = np.frombuffer(payload_data[pay_idx : pay_idx + 4], dtype=np.uint8); pay_idx += 4
                            packed = np.frombuffer(payload_data[pay_idx : pay_idx + 64], dtype=np.uint8); pay_idx += 64
                            unpacked = np.zeros(256, dtype=np.uint8)
                            for i in range(64):
                                byte = packed[i]
                                unpacked[i*4]   = (byte >> 6) & 0x03
                                unpacked[i*4+1] = (byte >> 4) & 0x03
                                unpacked[i*4+2] = (byte >> 2) & 0x03
                                unpacked[i*4+3] = byte & 0x03
                            mapped = palette[unpacked].reshape((16, 16, 1))
                            next_frame[y_min:y_max, x_min:x_max] = mapped.astype(np.int16)
                        elif current_mode == 12:
                            num_colors = payload_data[pay_idx]; pay_idx += 1
                            palette = np.frombuffer(payload_data[pay_idx : pay_idx + 4], dtype=np.uint8); pay_idx += 4
                            packed = np.frombuffer(payload_data[pay_idx : pay_idx + 64], dtype=np.uint8); pay_idx += 64
                            unpacked = np.zeros(256, dtype=np.uint8)
                            for i in range(64):
                                byte = packed[i]
                                unpacked[i*4]   = (byte >> 6) & 0x03
                                unpacked[i*4+1] = (byte >> 4) & 0x03
                                unpacked[i*4+2] = (byte >> 2) & 0x03
                                unpacked[i*4+3] = byte & 0x03
                            mapped = palette[unpacked].reshape((16, 16, 1))
                            next_frame[y_min:y_max, x_min:x_max] = mapped.astype(np.int16)
                        elif current_mode == 1:
                            nnz = payload_data[pay_idx]; pay_idx += 1
                            quantized = np.zeros(256, dtype=np.int16)
                            if nnz < 128:
                                indices = np.frombuffer(payload_data[pay_idx : pay_idx + nnz], dtype=np.uint8); pay_idx += nnz
                                values = np.frombuffer(payload_data[pay_idx : pay_idx + nnz], dtype=np.uint8).astype(np.int16) - 128; pay_idx += nnz
                                quantized[indices] = values
                            else:
                                data = payload_data[pay_idx : pay_idx + 256]; pay_idx += 256
                                quantized = np.frombuffer(data, dtype=np.uint8).astype(np.int16) - 128
                            next_frame[y_min:y_max, x_min:x_max] = np.clip(next_frame[y_min:y_max, x_min:x_max] + (quantized.reshape((BLOCK_SIZE, BLOCK_SIZE, 1)) * QP), 0, 255)
                        elif current_mode == 2:
                            data = payload_data[pay_idx : pay_idx + (256 * w_mult * h_mult)]; pay_idx += (256 * w_mult * h_mult)
                            next_frame[y_min:y_max, x_min:x_max] = np.frombuffer(data, dtype=np.uint8).reshape((h_mult*16, w_mult*16, 1)).astype(np.int16)
                            
                        processed[y_idx : y_idx + h_mult, x_idx : x_idx + w_mult] = True

                if has_bg:
                    diff_mask = np.abs(next_frame.astype(np.int32) - prev.astype(np.int32)) < 5
                    bg_model[diff_mask] = (0.95 * bg_model[diff_mask] + 0.05 * next_frame[diff_mask]).astype(np.float32)

                prev = next_frame.copy()

            # --- FORENSIC MSE CALCULATOR ---
            diff_sq = (orig_y - prev.squeeze().astype(np.float32)) ** 2
            block_mse_grid = diff_sq.reshape(grid_h, BLOCK_SIZE, grid_w, BLOCK_SIZE).mean(axis=(1, 3))
            
            for m in np.unique(frame_modes):
                mask = (frame_modes == m)
                count = np.sum(mask)
                if count > 0:
                    mse_values = block_mse_grid[mask]
                    mode_stats[m]["count"] += count
                    mode_stats[m]["sum_mse"] += np.sum(mse_values)
                    mode_stats[m]["max_mse"] = max(mode_stats[m]["max_mse"], np.max(mse_values))
                
    cap.release()
    return mode_stats

def batch_investigate(media_folder="test_vid"):
    videos = sorted([f for f in os.listdir(media_folder) if f.endswith(('.mp4', '.y4m'))])
    if not videos:
        print(f"No videos found in {media_folder}/")
        return

    config = {
        "name": "V5 Investigator", 
        "motion": True, "palette": True, 
        "dynamic_boxing": True, "background": True, "dct": True
    }
    temp_nam = "temp_batch_investigator.nam"

    # Colors
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'

    print("🚀 STARTING FORENSIC BATCH INVESTIGATION...")

    for video in videos:
        video_path = os.path.join(media_folder, video)
        print(f"\n" + "="*85)
        print(f"📊 FORENSIC ALGORITHM REPORT: {video}")
        print("="*85)

        try:
            print(f"Encoding & Analyzing...")
            enc_time, _ = run_encoder(video_path, temp_nam, config)
            stats = decode_and_investigate(video_path, temp_nam)
        except Exception as e:
            print(f"Crashed on {video}: {e}")
            continue
        finally:
            if os.path.exists(temp_nam): os.remove(temp_nam)

        print(f"\n{'Algorithm Mode':<30} | {'Blocks Used':>12} | {'Avg Block MSE':>15} | {'Worst Block MSE':>18}")
        print("-" * 85)

        valid_modes = [m for m in range(256) if stats[m]["count"] > 0]
        valid_modes.sort(key=lambda m: stats[m]["sum_mse"] / stats[m]["count"])

        total_blocks = sum(stats[m]["count"] for m in valid_modes)
        overall_mse_sum = sum(stats[m]["sum_mse"] for m in valid_modes)

        for m in valid_modes:
            count = stats[m]["count"]
            avg_mse = stats[m]["sum_mse"] / count
            max_mse = stats[m]["max_mse"]
            
            mode_name = MODE_NAMES.get(m, f"Unknown Mode {m}")
            
            avg_color = GREEN if avg_mse < 5.0 else (YELLOW if avg_mse < 15.0 else RED)
            
            avg_str = f"{avg_mse:.2f}"
            max_str = f"{max_mse:.2f}"
            
            avg_fmt = f"{' ' * (13 - len(avg_str))}{avg_color}{avg_str} {RESET}"
            max_fmt = f"{' ' * (16 - len(max_str))}{RED if max_mse > 50 else YELLOW if max_mse > 20 else GREEN}{max_str} {RESET}"

            print(f"{mode_name:<30} | {count:>12,d} | {avg_fmt} | {max_fmt}")

        print("-" * 85)
        overall_avg = overall_mse_sum / total_blocks if total_blocks > 0 else 0
        print(f"{'OVERALL VIDEO AVERAGE':<30} | {total_blocks:>12,d} | {overall_avg:>13.2f}   |")
        print("="*85)

if __name__ == "__main__":
    batch_investigate()