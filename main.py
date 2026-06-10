from src.encoder import encode_video
from src.decoder import decode_video_stream
from tqdm import tqdm
import cv2
import numpy as np

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
        frames.append(frame)

    cap.release()
    return frames

def check_exact_match(original, reconstructed):
    if len(original) != len(reconstructed):
        print(f"Frame count mismatch ❌ ({len(original)} vs {len(reconstructed)})")
        return False

    # Check match with progress bar
    for i in tqdm(range(len(original)), desc="Verifying frames", unit="frame"):
        if not np.array_equal(original[i], reconstructed[i]):
            print(f"\nFrame {i} mismatch ❌")
            return False

    print("\nAll frames match perfectly ✅")
    return True

def evaluate_quality(original, reconstructed):
    if len(original) != len(reconstructed):
        print(f"Frame count mismatch ❌ ({len(original)} vs {len(reconstructed)})")
        return False

    total_mse = 0
    mismatched_frames = 0
    mismatched_indices = []

    for i in tqdm(range(len(original)), desc="Analyzing Quality", unit="frame"):
        if not np.array_equal(original[i], reconstructed[i]):
            mismatched_frames += 1
            mismatched_indices.append(i)
            
            # Calculate Mean Squared Error (how far off the pixels are)
            diff = original[i].astype(np.float32) - reconstructed[i].astype(np.float32)
            mse = np.mean(diff ** 2)
            total_mse += mse

    if mismatched_frames == 0:
        print("\nAll frames match perfectly (Lossless Compression) ✅")
    else:
        avg_mse = total_mse / len(original)
        
        print("\n" + "="*30)
        print("📊 COMPRESSION REPORT")
        print("="*30)
        print(f"- Lossy Frames: {mismatched_frames} / {len(original)}")
        
        # Show a preview of which frames were lossy (so it doesn't spam the console)
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
        print("="*30 + "\n")

    return True

def write_video(frames, output_path, fps):
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    print("Writing reconstructed video...")
    for f in tqdm(frames, desc="Saving MP4", unit="frame"):
        out.write(f)

    out.release()

if __name__ == "__main__":
    input_video = "countdown.mp4"
    encoded_file = "encoded.namaste"

    fps = get_video_fps(input_video)
    print("FPS:", fps)
    
    # 1. Encode
    encode_video(input_video, encoded_file, QP=8)

    # 2. Decode (Streamed into a list, wrapped with tqdm)
    reconstructed_frames = []
    stream = decode_video_stream(encoded_file)
    for frame in tqdm(stream, desc="Decoding NAM0", unit="frame"):
        reconstructed_frames.append(frame)

    # 3. Extract Original
    original_frames = extract_frames(input_video)

    print(f"Original: {len(original_frames)}")
    print(f"Reconstructed: {len(reconstructed_frames)}")
    # check_exact_match(original_frames, reconstructed_frames)  <-- Delete or comment this out
    evaluate_quality(original_frames, reconstructed_frames)

    # 4. Check & Write
    check_exact_match(original_frames, reconstructed_frames)
    write_video(reconstructed_frames, "reconstructed.mp4", fps)