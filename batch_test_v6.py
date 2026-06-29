import os
import cv2
import time
import numpy as np
import builtins

from encoder_v6 import encode_v6
from decoder_v6 import decode_v6

# Monkey-patch print to keep the batch layout pristine and suppress encoder logs
original_print = builtins.print
def quiet_print(*args, **kwargs):
    text = str(args[0]) if args else ""
    if text.startswith("▶") or text.startswith("=") or text.startswith("-") or text.startswith("V6") or text.startswith("🚀") or text.startswith("Algorithm"):
        original_print(*args, **kwargs)
builtins.print = quiet_print

def calculate_mse(orig_frames, dec_frames):
    min_len = min(len(orig_frames), len(dec_frames))
    if min_len == 0: return 0.0
    
    mse_sum = sum(np.mean((orig_frames[i].astype(np.float32) - dec_frames[i].astype(np.float32)) ** 2) 
                  for i in range(min_len))
    return mse_sum / min_len

def run_v6_batch(media_folder="test_vid"):
    videos = sorted([f for f in os.listdir(media_folder) if f.endswith(('.mp4', '.y4m'))])
    if not videos:
        original_print(f"No videos found in {media_folder}/")
        return

    original_print("\n🚀 STARTING V6 SOBVC ABLATION ON SYNTHETIC VIDEO CORPUS...\n")

    for video in videos:
        video_path = os.path.join(media_folder, video)
        encoded_path = f"temp_{video}.nam6"
        
        # --- Load Original ---
        cap = cv2.VideoCapture(video_path)
        orig_frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            orig_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        cap.release()
        total_frames = len(orig_frames)
        orig_size = os.path.getsize(video_path) / 1024
        
        original_print(f"▶ Testing: {video} (Orig: {orig_size:.2f} KB | Frames: {total_frames}/{total_frames})")
        original_print("=" * 115)
        original_print(f"{'Algorithm':<22} | {'Size (KB)':>9} | {' Ratio':>6} | {' MSE':>5} | {'Enc FPS':>7} | {'Dec FPS':>7} | {'Notes'}")
        original_print("-" * 115)
        
        # --- Encode ---
        t0 = time.time()
        encode_v6(video_path, encoded_path)
        enc_time = time.time() - t0
        enc_fps = total_frames / max(enc_time, 0.001)
        new_size = os.path.getsize(encoded_path) / 1024
        ratio = new_size / orig_size
        
        # --- Decode ---
        t0 = time.time()
        dec_frames = decode_v6(encoded_path)
        dec_time = time.time() - t0
        dec_fps = total_frames / max(dec_time, 0.001)
        
        # --- MSE Check ---
        mse = calculate_mse(orig_frames, dec_frames)
        
        original_print(f"{'V6 Semantic SOBVC':<22} | {new_size:>9.2f} | {ratio:>5.2f}x | {mse:>5.2f} | {enc_fps:>7.1f} | {dec_fps:>7.1f} | Quadtree + Bloom")
        original_print("=" * 115 + "\n")
        
        if os.path.exists(encoded_path):
            os.remove(encoded_path)

if __name__ == "__main__":
    try:
        run_v6_batch()
    finally:
        builtins.print = original_print