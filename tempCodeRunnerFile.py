import os
import csv
import argparse
import cv2
from lab_v2 import run_encoder, calculate_mse

def run_batch_ablation(media_folder="test_vid", output_csv="rdo_report.csv", full_test=False):
    videos = sorted([f for f in os.listdir(media_folder) if f.endswith(('.mp4', '.y4m'))])
    
    if not videos:
        print(f"No videos found in {media_folder}/")
        return

    mode_string = "FULL TEST (All Frames)" if full_test else "QUICK TEST (45 Frames)"

    test_configs = [
        {"name": "V2 Baseline (Spatial)", "motion": False, "cache": False, "palette": False, "scale": False},
        {"name": "V2 Deep Motion (CCORR)", "motion": True, "cache": False, "palette": False, "scale": False},
        {"name": "V2 Fuzzy Cache (Snap)", "motion": False, "cache": True, "palette": False, "scale": False},
        {"name": "V2 RLE Palette", "motion": False, "cache": False, "palette": True, "scale": False},
        {"name": "V2 Affine Scale", "motion": False, "cache": False, "palette": False, "scale": True}
    ]

    print(f"🔬 STARTING BATCH ABLATION STUDY ({mode_string}) ON {len(videos)} VIDEOS...\n")

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Video", "Algorithm", "Orig Size (KB)", "Encoded (KB)", "Ratio", "MSE"])

        for video in videos:
            video_path = os.path.join(media_folder, video)
            orig_size_kb = os.path.getsize(video_path) / 1024
            
            # --- THE FIX: Get the exact frame count dynamically ---
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            current_max_frames = total_frames if full_test else min(45, total_frames)
            # --------------------------------------------------------

            print(f"\n▶ Testing: {video} (Original Size: {orig_size_kb:.2f} KB | Frames: {current_max_frames})")
            print("-" * 75)
            
            for i, config in enumerate(test_configs):
                out_file = f"temp_{i}.nam_lab"
                try:
                    run_encoder(video_path, out_file, config, max_frames=current_max_frames)
                    mse = calculate_mse(video_path, out_file, max_frames=current_max_frames)
                    
                    size_kb = os.path.getsize(out_file) / 1024
                    ratio = size_kb / orig_size_kb
                    
                    writer.writerow([video, config['name'], f"{orig_size_kb:.2f}", f"{size_kb:.2f}", f"{ratio:.3f}x", f"{mse:.2f}"])
                    print(f"{config['name']:<25} | {size_kb:>8.2f} KB ({ratio:>5.2f}x) | MSE: {mse:>6.2f}")
                except Exception as e:
                    print(f"{config['name']:<25} | FAILED: {e}")
                finally:
                    if os.path.exists(out_file):
                        os.remove(out_file)

    print(f"\n✅ Batch complete! Data saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NAM Codec Batch Testing Laboratory")
    parser.add_argument(
        "--full", 
        action="store_true", 
        help="Run encoding matrix on all frames instead of a quick 45-frame sample"
    )
    args = parser.parse_args()

    run_batch_ablation(media_folder="test_vid", output_csv="rdo_report.csv", full_test=args.full)