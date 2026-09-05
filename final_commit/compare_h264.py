import os
import sys
import time
import imageio
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import warnings
warnings.filterwarnings("ignore") # ignore divide by zero for inf PSNR

from encoder import encode
from decoder import decode

def print_header(title):
    print(f"\n{'='*60}\n{title.center(60)}\n{'='*60}")

def load_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    return frames

def calculate_metrics(orig_frames, decoded_frames):
    total_ssim = 0
    total_psnr = 0
    
    count = min(len(orig_frames), len(decoded_frames))
    for i in range(count):
        orig = orig_frames[i]
        dec = decoded_frames[i]
        
        # Ensure 2D shapes
        if len(orig.shape) == 3 and orig.shape[2] == 3: orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        if len(dec.shape) == 3 and dec.shape[2] == 3: dec = cv2.cvtColor(dec, cv2.COLOR_BGR2GRAY)
        orig = np.squeeze(orig)
        dec = np.squeeze(dec)
        
        # Calculate SSIM
        s = ssim(orig, dec, data_range=255)
        total_ssim += s
        
        # Accumulate MSE for global PSNR
        err = np.mean((orig.astype(np.float64) - dec.astype(np.float64)) ** 2)
        total_psnr += err # reusing variable for mse
        
    avg_mse = total_psnr / count
    if avg_mse == 0:
        final_psnr = float('inf')
    else:
        final_psnr = 10 * np.log10((255**2) / avg_mse)
        
    return total_ssim / count, final_psnr

def main():
    videos_to_test = ['../test_vid/01_ideal_motion.mp4', '../test_vid/countdown.mp4']
    
    for vid in videos_to_test:
        if not os.path.exists(vid):
            print(f"Skipping {vid}, not found.")
            continue
            
        print_header(f"Evaluating: {os.path.basename(vid)}")
        orig_frames = load_frames(vid)
        fps = 30 # standard synthetic
        duration = len(orig_frames) / fps
        
        # 1. Encode with TrajCDDec
        print("1. Encoding with TrajCDDec...")
        nam_out = vid.replace('.mp4', '_traj.nam')
        enc_stats = encode(vid, nam_out)
        traj_size = os.path.getsize(nam_out)
        
        target_bitrate = int((traj_size * 8) / duration)
        print(f"   TrajCDDec Size: {traj_size} bytes ({target_bitrate/1000:.2f} kbps target)")
        
        # 2. Encode with H.264 matching bitrate
        print(f"2. Encoding with H.264 (Targeting {target_bitrate/1000:.2f} kbps)...")
        h264_out = vid.replace('.mp4', '_h264.mp4')
        writer = imageio.get_writer(
            h264_out, 
            fps=fps, 
            macro_block_size=None,
            ffmpeg_params=[
                '-vcodec', 'libx264',
                '-b:v', str(target_bitrate),
                '-maxrate', str(target_bitrate),
                '-bufsize', str(target_bitrate * 2),
                '-preset', 'veryslow' # Give H.264 the best possible compression efficiency
            ]
        )
        for f in orig_frames:
            writer.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        writer.close()
        h264_size = os.path.getsize(h264_out)
        print(f"   H.264 Size:     {h264_size} bytes")
        
        # 3. Encode with H.265 matching bitrate
        print(f"3. Encoding with H.265 (Targeting {target_bitrate/1000:.2f} kbps)...")
        h265_out = vid.replace('.mp4', '_h265.mp4')
        writer = imageio.get_writer(
            h265_out, 
            fps=fps, 
            macro_block_size=None,
            ffmpeg_params=[
                '-vcodec', 'libx265',
                '-b:v', str(target_bitrate),
                '-maxrate', str(target_bitrate),
                '-bufsize', str(target_bitrate * 2),
                '-preset', 'veryslow' # Give H.265 the best possible compression efficiency
            ]
        )
        for f in orig_frames:
            writer.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        writer.close()
        h265_size = os.path.getsize(h265_out)
        print(f"   H.265 Size:     {h265_size} bytes")
        
        # 4. Decode TrajCDDec
        print("4. Decoding & Computing Metrics...")
        dec_stats = decode(nam_out)
        traj_frames = dec_stats['frames']
        traj_ssim, traj_psnr = calculate_metrics(orig_frames, traj_frames)
        
        # 5. Decode H.264
        h264_frames = []
        reader = imageio.get_reader(h264_out)
        for f in reader:
            h264_frames.append(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        reader.close()
        h264_ssim, h264_psnr = calculate_metrics(orig_frames, h264_frames)
        
        # 6. Decode H.265
        h265_frames = []
        reader = imageio.get_reader(h265_out)
        for f in reader:
            h265_frames.append(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        reader.close()
        h265_ssim, h265_psnr = calculate_metrics(orig_frames, h265_frames)
        
        print("\n--- RESULTS ---")
        print(f"TrajCDDec : SSIM = {traj_ssim:.4f} | PSNR = {traj_psnr:.2f} dB | Size = {traj_size} bytes")
        print(f"H.264     : SSIM = {h264_ssim:.4f} | PSNR = {h264_psnr:.2f} dB | Size = {h264_size} bytes")
        print(f"H.265     : SSIM = {h265_ssim:.4f} | PSNR = {h265_psnr:.2f} dB | Size = {h265_size} bytes")
        
        best_ssim = max(traj_ssim, h264_ssim, h265_ssim)
        if traj_ssim == best_ssim:
            print("[WINNER] TrajCDDec WINS on Visual Quality (SSIM)!")
        elif h265_ssim == best_ssim:
            print("[WINNER] H.265 WINS on Visual Quality (SSIM)!")
        else:
            print("[WINNER] H.264 WINS on Visual Quality (SSIM)!")
            
        # Clean up
        if os.path.exists(nam_out): os.remove(nam_out)
        if os.path.exists(h264_out): os.remove(h264_out)
        if os.path.exists(h265_out): os.remove(h265_out)

if __name__ == "__main__":
    main()
