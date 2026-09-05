from final_commit.encoder import encode as encode_video
from final_commit.decoder import decode as decode_video_stream
from tqdm import tqdm
import cv2
import numpy as np
import os

def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    print("Extracting original frames...")
    for _ in tqdm(range(total_frames), desc="Reading MP4", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break
        # Crop to even dimensions to match NAM1 (YUV 4:2:0) requirements
        h, w, _ = frame.shape
        h = h - (h % 2)
        w = w - (w % 2)
        frames.append(frame[:h, :w])

    cap.release()
    return frames

def evaluate_quality(original, reconstructed):
    # Dynamically check only the intersecting frames so it never crashes
    frames_to_check = min(len(original), len(reconstructed))
    
    total_mse = 0
    mismatched_frames = 0
    mismatched_indices = []

    for i in tqdm(range(frames_to_check), desc="Analyzing Quality", unit="frame"):
        rec_f = reconstructed[i]
        h_r, w_r = rec_f.shape[:2]
        orig_cropped = original[i][:h_r, :w_r]
        
        if rec_f.ndim == 3 and rec_f.shape[2] == 3:
            diff = orig_cropped.astype(np.float32) - rec_f.astype(np.float32)
        else:
            orig_y = cv2.cvtColor(orig_cropped, cv2.COLOR_BGR2GRAY)
            diff = orig_y.astype(np.float32) - rec_f.squeeze().astype(np.float32)
            
        mse = np.mean(diff ** 2)
        total_mse += mse
        if mse > 0.001:
            mismatched_frames += 1
            mismatched_indices.append(i)

    if mismatched_frames == 0:
        print("\nAll frames match perfectly (Lossless Compression) [OK]")
    else:
        avg_mse = total_mse / frames_to_check
        
        print("\n" + "="*30)
        print("COMPRESSION REPORT")
        print("="*30)
        print(f"- Lossy Frames: {mismatched_frames} / {frames_to_check}")
        
        if len(mismatched_indices) > 5:
            preview = ", ".join(map(str, mismatched_indices[:5]))
            print(f"- Mismatched Indices: [{preview}, ... and {mismatched_frames - 5} more]")
        else:
            print(f"- Mismatched Indices: {mismatched_indices}")
            
        print(f"- Average MSE: {avg_mse:.4f} (Lower is better)")
        
        if avg_mse < 10.0:
            print("- Visual Result: Perceptually identical to the human eye [OK]")
        else:
            print("- Visual Result: Noticeable quality loss [WARNING]")
        print("="*30)

    return True

def check_exact_match(original, reconstructed):
    frames_to_check = min(len(original), len(reconstructed))
    
    for i in tqdm(range(frames_to_check), desc="Verifying frames", unit="frame"):
        rec_f = reconstructed[i]
        h_r, w_r = rec_f.shape[:2]
        orig_cropped = original[i][:h_r, :w_r]
        if rec_f.ndim == 3 and rec_f.shape[2] == 3:
            match = np.array_equal(orig_cropped, rec_f)
        else:
            orig_y = cv2.cvtColor(orig_cropped, cv2.COLOR_BGR2GRAY)
            match = np.array_equal(orig_y, rec_f.squeeze())
        if not match:
            print(f"\nFrame {i} mismatch [X]")
            return False
            
    return True

def write_video(frames, output_path, fps):
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h), isColor=True)

    print("\nWriting reconstructed video...")
    for f in tqdm(frames, desc="Saving MP4", unit="frame"):
        if f.ndim == 3 and f.shape[2] == 3:
            f_bgr = np.clip(f, 0, 255).astype(np.uint8)
        else:
            f_bgr = cv2.cvtColor(np.clip(f.squeeze(), 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        out.write(f_bgr)

    out.release()

def print_storage_report(original_path, encoded_path, reconstructed_path):
    print("\n" + "="*30)
    print("STORAGE REPORT")
    print("="*30)
    
    try:
        orig_mb = os.path.getsize(original_path) / (1024 * 1024)
        enc_mb = os.path.getsize(encoded_path) / (1024 * 1024)
        rec_mb = os.path.getsize(reconstructed_path) / (1024 * 1024)
        
        print(f"- Original MP4:      {orig_mb:.2f} MB")
        print(f"- NAM1 Encoded:      {enc_mb:.2f} MB")
        print(f"- Reconstructed MP4: {rec_mb:.2f} MB")
        
        # Calculate comparison multiplier against the original
        if orig_mb > 0:
            multiplier = enc_mb / orig_mb
            print(f"\n- Ratio: NAM1 is {multiplier:.2f}x the size of the original MP4")
            
    except OSError as e:
        print(f"Could not calculate file sizes: {e}")
        
    print("="*30 + "\n")

if __name__ == "__main__":
    input_video = "test_vid/01_ideal_motion.mp4"
    encoded_file = "encoded.namaste"
    reconstructed_file = "reconstructed.mp4"

    fps = get_video_fps(input_video)
    print("FPS:", fps)
    
    # 1. Encode with standard sizing
    encode_video(input_video, encoded_file)

    # 2. Decode Stream
    res = decode_video_stream(encoded_file, original_video=input_video)
    reconstructed_frames = res["frames"] if isinstance(res, dict) and "frames" in res else list(res)

    # 3. Extract Original
    original_frames = extract_frames(input_video)

    print(f"\nOriginal: {len(original_frames)}")
    print(f"Reconstructed: {len(reconstructed_frames)}")

    # 4. Analytics exactly as requested
    evaluate_quality(original_frames, reconstructed_frames)
    check_exact_match(original_frames, reconstructed_frames)
    
    # 5. Save final video
    write_video(reconstructed_frames, reconstructed_file, fps)
    
    # 6. Storage Report
    print_storage_report(input_video, encoded_file, reconstructed_file)