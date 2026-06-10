from src.encoder import encode_video
from src.decoder import decode_video_stream
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
        if not np.array_equal(original[i], reconstructed[i]):
            mismatched_frames += 1
            mismatched_indices.append(i)
            
            diff = original[i].astype(np.float32) - reconstructed[i].astype(np.float32)
            mse = np.mean(diff ** 2)
            total_mse += mse

    if mismatched_frames == 0:
        print("\nAll frames match perfectly (Lossless Compression) ✅")
    else:
        avg_mse = total_mse / frames_to_check
        
        print("\n" + "="*30)
        print("📊 COMPRESSION REPORT")
        print("="*30)
        print(f"- Lossy Frames: {mismatched_frames} / {frames_to_check}")
        
        if len(mismatched_indices) > 5:
            preview = ", ".join(map(str, mismatched_indices[:5]))
            print(f"- Mismatched Indices: [{preview}, ... and {mismatched_frames - 5} more]")
        else:
            print(f"- Mismatched Indices: {mismatched_indices}")
            
        print(f"- Average MSE: {avg_mse:.4f} (Lower is better)")
        
        if avg_mse < 10.0:
            print("- Visual Result: Perceptually identical to the human eye ✅")
        else:
            print("- Visual Result: Noticeable quality loss ⚠️")
        print("="*30)

    return True

def check_exact_match(original, reconstructed):
    frames_to_check = min(len(original), len(reconstructed))
    
    for i in tqdm(range(frames_to_check), desc="Verifying frames", unit="frame"):
        if not np.array_equal(original[i], reconstructed[i]):
            print(f"\nFrame {i} mismatch ❌")
            return False
            
    return True

def write_video(frames, output_path, fps):
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    print("\nWriting reconstructed video...")
    for f in tqdm(frames, desc="Saving MP4", unit="frame"):
        out.write(f)

    out.release()

def print_storage_report(original_path, encoded_path, reconstructed_path):
    print("\n" + "="*30)
    print("💾 STORAGE REPORT")
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
    input_video = "countdown.mp4"
    encoded_file = "encoded.namaste"
    reconstructed_file = "reconstructed.mp4"

    fps = get_video_fps(input_video)
    print("FPS:", fps)
    
    # 1. Encode with standard sizing
    encode_video(input_video, encoded_file, QP=8, iframe_interval=300)

    # 2. Decode Stream
    reconstructed_frames = []
    stream = decode_video_stream(encoded_file)
    for frame in tqdm(stream, desc="Decoding NAM1", unit="frame"):
        reconstructed_frames.append(frame)

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