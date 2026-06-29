import cv2
import numpy as np
import struct
import zstandard as zstd
import tkinter as tk
from tkinter import filedialog
import os

# Import the perfected V5 encoder
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

# X-Ray Color Map (OpenCV uses BGR format)
COLOR_MAP = {
    0: (0, 0, 0),         # Skip: Black
    1: (255, 255, 255),   # Spatial Delta: Pure White
    2: (128, 128, 128),   # Raw Bytes: Gray
    4: (0, 165, 255),     # Fast Motion: Orange
    6: (50, 50, 50),      # Adaptive BG: Dark Gray
    7: (255, 0, 0),       # Solid Color: Blue
    9: (255, 255, 0),     # DCT Low: Cyan
    10: (255, 255, 0),    # DCT Mid: Cyan
    11: (0, 255, 255),    # Predicted Motion: Yellow
    12: (0, 255, 0),      # Precise Palette: Green
    13: (0, 0, 255),      # Faded Motion: Red
    15: (255, 0, 255),    # Forced Dither: Magenta
    255: (0, 0, 0)        # I-Frame: Black
}

def create_legend_sidebar(sidebar_w, sidebar_h):
    """Generates the static color key panel for the UI."""
    sidebar = np.zeros((sidebar_h, sidebar_w, 3), dtype=np.uint8)
    
    y_offset = 40
    # Added cv2.LINE_AA for crisp, anti-aliased title
    cv2.putText(sidebar, "V5 CODEC X-RAY", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(sidebar, (20, y_offset + 10), (sidebar_w - 20, y_offset + 10), (100, 100, 100), 1, cv2.LINE_AA)
    
    y_offset += 40
    
    # Sort modes to group them logically
    display_order = [255, 0, 6, 7, 12, 11, 4, 13, 9, 10, 15, 1, 2]
    
    for mode in display_order:
        if mode not in COLOR_MAP: continue
        color = COLOR_MAP[mode]
        name = MODE_NAMES.get(mode, "Unknown")
        
        # Draw the color box
        cv2.rectangle(sidebar, (20, y_offset - 15), (40, y_offset + 5), color, -1)
        if mode in [0, 255]:
            cv2.rectangle(sidebar, (20, y_offset - 15), (40, y_offset + 5), (100, 100, 100), 1)
            
        # Added cv2.LINE_AA for crisp labels
        cv2.putText(sidebar, f"{name}", (55, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        y_offset += 30
        
    # Added cv2.LINE_AA for crisp control instructions
    cv2.putText(sidebar, "CONTROLS:", (20, sidebar_h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(sidebar, "[SPACE] Pause/Play", (20, sidebar_h - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(sidebar, "[N] Next Frame (While Paused)", (20, sidebar_h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(sidebar, "[Q] Quit & Dump Report", (20, sidebar_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
    
    return sidebar

def run_xray_decoder(original_video, encoded_path):
    print(f"\n🔍 Booting Codec X-Ray for {os.path.basename(original_video)}...")
    dctx = zstd.ZstdDecompressor()
    cap = cv2.VideoCapture(original_video)
    BLOCK_SIZE = 16
    
    mode_stats = {k: {"count": 0, "sum_mse": 0.0, "max_mse": 0.0} for k in range(256)}
    
    with open(encoded_path, "rb") as f:
        # Read the 21-byte Header
        magic = f.read(6)
        h, w, total_frames = struct.unpack("<III", f.read(12))
        QP = struct.unpack("<H", f.read(2))[0]
        has_bg = struct.unpack("<B", f.read(1))[0]
        
        grid_h, grid_w = h // BLOCK_SIZE, w // BLOCK_SIZE
        prev = None
        bg_model = None

        # Build UI layout variables
        sidebar_w = 320
        ui_height = max(h, 540) # Ensure window is tall enough to fit the legend
        legend_sidebar = create_legend_sidebar(sidebar_w, ui_height)

        cv2.namedWindow("Codec X-Ray", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Codec X-Ray", w + sidebar_w, ui_height)

        is_paused = False
        frame_counter = 0

        while True:
            if not is_paused:
                ret, orig_frame = cap.read()
                type_byte = f.read(1)
                
                # --- INFINITE LOOP LOGIC ---
                if not ret or not type_byte:
                    print("🔄 Reached end of video. Looping...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Rewind video
                    f.seek(21) # Rewind .nam file exactly past the 21-byte header
                    prev = None
                    bg_model = None
                    continue 
                
                orig_y = cv2.cvtColor(orig_frame[:h, :w], cv2.COLOR_BGR2GRAY).astype(np.float32)
                frame_type = struct.unpack("<B", type_byte)[0]
                frame_modes = np.zeros((grid_h, grid_w), dtype=np.uint8) 

                # --- DECODING ---
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
                            elif current_mode == 9: pay_idx += 32
                            elif current_mode == 10: pay_idx += 128
                            elif current_mode == 15: pay_idx += 68
                            elif current_mode == 12: pay_idx += 69
                            elif current_mode == 1:
                                nnz = payload_data[pay_idx]; pay_idx += 1
                                if nnz < 128: pay_idx += (nnz * 2)
                                else: pay_idx += 256
                            elif current_mode == 2:
                                pay_idx += (256 * w_mult * h_mult)
                                
                            processed[y_idx : y_idx + h_mult, x_idx : x_idx + w_mult] = True

                    if has_bg:
                        diff_mask = np.abs(next_frame.astype(np.int32) - prev.astype(np.int32)) < 5
                        bg_model[diff_mask] = (0.95 * bg_model[diff_mask] + 0.05 * next_frame[diff_mask]).astype(np.float32)

                    prev = next_frame.copy()

                # --- FORENSIC STATS (Only calculate on the 1st loop to preserve report purity) ---
                if frame_counter < total_frames:
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
                    frame_counter += 1

                # --- X-RAY RENDERER ---
                color_grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
                for mode, color in COLOR_MAP.items():
                    color_grid[frame_modes == mode] = color
                    
                heatmap_full = cv2.resize(color_grid, (w, h), interpolation=cv2.INTER_NEAREST)
                display_frame = orig_frame[:h, :w].copy()
                
                # Blend (Black Skip blocks will nicely dim the background, making colors pop)
                blended = cv2.addWeighted(display_frame, 0.45, heatmap_full, 0.55, 0)
                
                # Draw subtle grid lines
                for x in range(0, w, 16): cv2.line(blended, (x, 0), (x, h), (40, 40, 40), 1)
                for y in range(0, h, 16): cv2.line(blended, (0, y), (w, y), (40, 40, 40), 1)

                # Center the video vertically if it's shorter than the UI
                if h < ui_height:
                    pad_top = (ui_height - h) // 2
                    pad_bottom = ui_height - h - pad_top
                    video_panel = cv2.copyMakeBorder(blended, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[15, 15, 15])
                else:
                    video_panel = blended

                # Stitch the Video and the Sidebar together
                final_ui = np.hstack((video_panel, legend_sidebar))

            # Display UI
            cv2.imshow("Codec X-Ray", final_ui)
            
            # Control Logic
            delay = 0 if is_paused else 30
            key = cv2.waitKey(delay) & 0xFF
            
            if key == ord('q'): 
                break
            elif key == ord(' '): 
                is_paused = not is_paused
            elif key == ord('n') and is_paused: 
                
                continue

    cap.release()
    cv2.destroyAllWindows()
    return mode_stats

def main():
    root = tk.Tk()
    root.withdraw() 
    
    print("Waiting for file selection...")
    video_path = filedialog.askopenfilename(
        title="Select Video to X-Ray (.mp4)", 
        filetypes=[("Video Files", "*.mp4 *.y4m *.avi")]
    )
    if not video_path:
        return

    root.update()

    config = {
        "name": "V5 X-Ray", 
        "motion": True, "palette": True, 
        "dynamic_boxing": True, "background": True, "dct": True
    }
    temp_nam = "temp_xray.nam"
    
    print(f"\nEncoding {os.path.basename(video_path)}...")
    try:
        run_encoder(video_path, temp_nam, config)
    except Exception as e:
        print(f"Encoder crashed: {e}")
        return

    try:
        stats = run_xray_decoder(video_path, temp_nam)
    except Exception as e:
        print(f"Decoder crashed: {e}")
        return
    finally:
        if os.path.exists(temp_nam): os.remove(temp_nam)

    # Print the Final Report
    print("\n" + "="*85)
    print(f"📊 FORENSIC ALGORITHM REPORT: {os.path.basename(video_path)}")
    print("="*85)
    print(f"{'Algorithm Mode':<30} | {'Blocks Used':>12} | {'Avg Block MSE':>15} | {'Worst Block MSE':>18}")
    print("-" * 85)

    GREEN, YELLOW, RED, RESET = '\033[92m', '\033[93m', '\033[91m', '\033[0m'
    valid_modes = [m for m in range(256) if stats[m]["count"] > 0]
    valid_modes.sort(key=lambda m: stats[m]["sum_mse"] / stats[m]["count"])
    total_blocks = sum(stats[m]["count"] for m in valid_modes)
    overall_mse_sum = sum(stats[m]["sum_mse"] for m in valid_modes)

    for m in valid_modes:
        count = stats[m]["count"]
        avg_mse = stats[m]["sum_mse"] / count
        max_mse = stats[m]["max_mse"]
        mode_name = MODE_NAMES.get(m, f"Unknown Mode {m}")
        avg_col = GREEN if avg_mse < 5.0 else (YELLOW if avg_mse < 15.0 else RED)
        max_col = GREEN if max_mse < 20.0 else (YELLOW if max_mse < 50.0 else RED)
        
        print(f"{mode_name:<30} | {count:>12,d} | {avg_col}{avg_mse:>15.2f}{RESET} | {max_col}{max_mse:>18.2f}{RESET}")

if __name__ == "__main__":
    main()