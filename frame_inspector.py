import cv2
import numpy as np
import struct
import zstandard as zstd
import tkinter as tk
from tkinter import filedialog
import os

def decode_video_to_memory(original_video, encoded_path):
    print(f"Loading {os.path.basename(original_video)} and decoding {os.path.basename(encoded_path)}...")
    dctx = zstd.ZstdDecompressor()
    cap = cv2.VideoCapture(original_video)
    BLOCK_SIZE = 16
    
    orig_frames = []
    decoded_frames = []
    mses = []
    all_frame_modes = [] # V5 Enhancement: Store the chosen mode for every block
    
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
            orig_y = cv2.cvtColor(orig_frame[:h, :w], cv2.COLOR_BGR2GRAY)
            orig_frames.append(orig_y)
            
            type_byte = f.read(1)
            if not type_byte: break 
            frame_type = struct.unpack("<B", type_byte)[0]

            if frame_type == 0:
                size = struct.unpack("<I", f.read(4))[0]
                prev = np.frombuffer(dctx.decompress(f.read(size)), dtype=np.uint8).reshape((h, w, 1)).astype(np.int16)
                bg_model = prev.copy().astype(np.float32)
                decoded_frames.append(prev.copy().squeeze().astype(np.uint8))
                mses.append(np.mean((orig_y.astype(np.float32) - prev.squeeze().astype(np.float32)) ** 2))
                # I-Frames are fully raw, log as mode 255
                all_frame_modes.append(np.full((grid_h, grid_w), 255, dtype=np.uint8))
                
            elif frame_type == 1:
                next_frame = prev.copy()
                processed = np.zeros((grid_h, grid_w), dtype=bool)
                frame_modes = np.zeros((grid_h, grid_w), dtype=np.uint8) # Default 0 = Skip
                
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
                            frame_modes[y_idx, x_idx] = 0 # Skip Mode
                            continue
                            
                        y_min, x_min = y_idx * BLOCK_SIZE, x_idx * BLOCK_SIZE
                        flag = mode_data[mode_idx]; mode_idx += 1
                        
                        if flag == 0xFF:
                            skip_remaining = struct.unpack_from("<H", payload_data, pay_idx)[0]
                            pay_idx += 2
                            skip_remaining -= 1 
                            processed[y_idx, x_idx] = True
                            frame_modes[y_idx, x_idx] = 0 # Skip Mode
                            continue 
                            
                        w_mult, h_mult, current_mode = 1, 1, flag
                        if flag == 3: 
                            w_mult, h_mult, current_mode = struct.unpack_from("<BBB", payload_data, pay_idx)
                            pay_idx += 3
                            
                        y_max, x_max = y_min + (h_mult * BLOCK_SIZE), x_min + (w_mult * BLOCK_SIZE)
                        
                        # Log the chosen mode to the heatmap tensor
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
                        elif current_mode == 12:
                            num_colors = payload_data[pay_idx]; pay_idx += 1
                            palette = np.frombuffer(payload_data[pay_idx : pay_idx + 4], dtype=np.uint8)
                            pay_idx += 4
                            packed = np.frombuffer(payload_data[pay_idx : pay_idx + 64], dtype=np.uint8)
                            pay_idx += 64
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
                decoded_frames.append(prev.copy().squeeze().astype(np.uint8))
                mses.append(np.mean((orig_y.astype(np.float32) - prev.squeeze().astype(np.float32)) ** 2))
                all_frame_modes.append(frame_modes)
                
    cap.release()
    print("Decoding complete. Launching Inspector...")
    return orig_frames, decoded_frames, mses, all_frame_modes, grid_h, grid_w, h

def render_mode_overlay(d_bgr, modes, grid_h, grid_w, img_h):
    overlay = d_bgr.copy()
    
    # Mode Color Mapping (BGR Format)
    COLORS = {
        1: (0, 0, 255),       # Spatial (Red)
        2: (0, 0, 150),       # Raw Bytes (Dark Red)
        4: (255, 0, 0),       # Motion (Blue)
        6: (255, 0, 255),     # Background (Magenta)
        7: (0, 255, 255),     # Solid Color (Yellow)
        11: (255, 255, 0),    # Pred Motion (Cyan)
        12: (0, 165, 255),    # Palette (Orange)
        255: (255, 255, 255)  # I-Frame (White flash)
    }
    
    for y in range(grid_h):
        for x in range(grid_w):
            m = modes[y, x]
            if m in COLORS:
                y1, x1 = y * 16, x * 16
                cv2.rectangle(overlay, (x1, y1), (x1+16, y1+16), COLORS[m], -1)
                
    # Blend the colors at 40% opacity
    blended = cv2.addWeighted(overlay, 0.4, d_bgr, 0.6, 0)
    
    # Draw Legend at bottom
    legend = [
        ("Skip", (0,0,0)), ("Spat", (0, 0, 255)), ("Mot", (255, 0, 0)), 
        ("Pred Mot", (255, 255, 0)), ("BG", (255, 0, 255)), 
        ("Solid", (0, 255, 255)), ("Pal", (0, 165, 255))
    ]
    
    cv2.rectangle(blended, (0, img_h - 30), (blended.shape[1], img_h), (0,0,0), -1)
    x_off = 10
    for name, color in legend:
        if name == "Skip":
            cv2.rectangle(blended, (x_off, img_h-22), (x_off+12, img_h-10), (255,255,255), 1)
        else:
            cv2.rectangle(blended, (x_off, img_h-22), (x_off+12, img_h-10), color, -1)
        cv2.putText(blended, name, (x_off+16, img_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        x_off += 85

    return blended

def main():
    # 1. Spawn Native File Picker GUI
    root = tk.Tk()
    root.withdraw() # Hide the main window
    
    print("Waiting for file selection...")
    orig_path = filedialog.askopenfilename(title="Select Original Video (.mp4)", filetypes=[("Video Files", "*.mp4 *.y4m *.avi")])
    if not orig_path:
        print("No original video selected. Exiting.")
        return
        
    nam_path = filedialog.askopenfilename(title="Select Encoded Video (.nam)", filetypes=[("NAM Files", "*.nam")])
    if not nam_path:
        print("No encoded video selected. Exiting.")
        return
        
    root.update() # Ensure dialog fully closes

    # 2. Decode completely into memory
    orig_frames, decoded_frames, mses, modes, grid_h, grid_w, img_h = decode_video_to_memory(orig_path, nam_path)
    total_frames = len(orig_frames)
    
    # 3. Viewer State
    cv2.namedWindow("V5 Inspector", cv2.WINDOW_NORMAL)
    state = {"idx": 0, "playing": False, "overlay": False}
    
    def on_trackbar(val):
        state["idx"] = val
        
    cv2.createTrackbar("Frame", "V5 Inspector", 0, total_frames - 1, on_trackbar)
    
    print("\n" + "="*30)
    print("🎮 V5 INSPECTOR CONTROLS")
    print("="*30)
    print(" [Space] : Play / Pause")
    print(" [Right] : Next Frame")
    print(" [Left]  : Prev Frame")
    print(" [ M ]   : Toggle Macroblock Overlay")
    print(" [Esc]/q : Quit")
    
    while True:
        idx = state["idx"]
        o_frame, d_frame = orig_frames[idx], decoded_frames[idx]
        mse, frame_mode = mses[idx], modes[idx]
        
        o_bgr = cv2.cvtColor(o_frame, cv2.COLOR_GRAY2BGR)
        d_bgr = cv2.cvtColor(d_frame, cv2.COLOR_GRAY2BGR)
        
        if state["overlay"]:
            d_bgr = render_mode_overlay(d_bgr, frame_mode, grid_h, grid_w, img_h)
            
        cv2.putText(o_bgr, f"ORIGINAL (Frame {idx})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        color_mse = (0, 0, 255) if mse > 5.0 else (0, 255, 0)
        cv2.putText(d_bgr, f"V5 DECODED (MSE: {mse:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_mse, 2)
        
        # Overlay Status Text
        if state["overlay"]:
            cv2.putText(d_bgr, "OVERLAY: ON (Press 'M')", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(d_bgr, "OVERLAY: OFF (Press 'M')", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        canvas = np.hstack((o_bgr, d_bgr))
        cv2.imshow("V5 Inspector", canvas)
        
        key = cv2.waitKey(30 if state["playing"] else 0) & 0xFF
        
        if key == 27 or key == ord('q'): # Esc or q
            break
        elif key == 32: # Space
            state["playing"] = not state["playing"]
        elif key == 83 or key == ord('d'): # Right Arrow
            state["playing"] = False
            state["idx"] = min(total_frames - 1, state["idx"] + 1)
            cv2.setTrackbarPos("Frame", "V5 Inspector", state["idx"])
        elif key == 81 or key == ord('a'): # Left Arrow
            state["playing"] = False
            state["idx"] = max(0, state["idx"] - 1)
            cv2.setTrackbarPos("Frame", "V5 Inspector", state["idx"])
        elif key == ord('m'): # M key
            state["overlay"] = not state["overlay"]
            
        if state["playing"]:
            state["idx"] = min(total_frames - 1, state["idx"] + 1)
            cv2.setTrackbarPos("Frame", "V5 Inspector", state["idx"])
            if state["idx"] == total_frames - 1:
                state["playing"] = False 

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()