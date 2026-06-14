import os
import csv
import argparse
import cv2
from lab_v3 import run_encoder, calculate_mse

def run_batch_ablation(media_folder="test_vid", output_csv="rdo_report_v3.csv", full_test=False):
    videos = sorted([f for f in os.listdir(media_folder) if f.endswith(('.mp4', '.y4m'))])
    
    if not videos:
        print(f"No videos found in {media_folder}/")
        return

    mode_string = "FULL TEST (All Frames)" if full_test else "QUICK TEST (45 Frames)"

    test_configs = [
        {"name": "V3 Baseline (Spatial)", "motion": False, "cache": False, "palette": False, "dynamic_boxing": False},
        {"name": "V3 Deep Motion",        "motion": True,  "cache": False, "palette": False, "dynamic_boxing": False},
        {"name": "V3 Solid Color",        "motion": False, "cache": False, "palette": True,  "dynamic_boxing": False},
        {"name": "V3 DYNAMIC BOX (Merge)", "motion": True,  "cache": True,  "palette": True,  "dynamic_boxing": True},
        {"name": "V3 BG Model (Sprite)",    "motion": True,  "cache": True,  "palette": True,  "dynamic_boxing": True,  "background": True},
    ]
    
    print(f"🔬 STARTING V3 MACROBLOCK ABLATION ({mode_string}) ON {len(videos)} VIDEOS...\n")

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Video", "Algorithm", "Orig Size", "Encoded", "Ratio", "MSE", "Skip%", "Intra%", "Motion%", "Solid%", "Spatial%"])

        results = []
        for video in videos:
            video_path = os.path.join(media_folder, video)
            orig_size_kb = os.path.getsize(video_path) / 1024
            
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            current_max_frames = total_frames if full_test else min(45, total_frames)

            print(f"\n▶ Testing: {video} (Orig: {orig_size_kb:.2f} KB | Frames: {current_max_frames})")
            print("=" * 125)
            print(f"{'Algorithm':<26} | {'Size (KB)':>10} | {'Ratio':>7} | {'MSE':>7} | {'Skip%':>7} | {'Intra%':>7} | {'Motion%':>8} | {'Solid%':>7} | {'Spatial%':>9}")
            print("-" * 125)
            
            for i, config in enumerate(test_configs):
                out_file = f"temp_{i}.nam_lab"
                try:
                    telemetry = run_encoder(video_path, out_file, config, max_frames=current_max_frames)
                    mse = calculate_mse(video_path, out_file, max_frames=current_max_frames)
                    
                    size_kb = os.path.getsize(out_file) / 1024
                    ratio = size_kb / orig_size_kb
                    tel = telemetry if telemetry else {}
                    
                    total = max(1, tel.get("blocks_total", 1))
                    skip_p = (tel.get("blocks_skip", 0) / total) * 100
                    intra_p = (tel.get("blocks_intra", 0) / total) * 100
                    mot_p = (tel.get("blocks_motion", 0) / total) * 100
                    sol_p = (tel.get("blocks_solid", 0) / total) * 100
                    spa_p = (tel.get("blocks_spatial", 0) / total) * 100
                    
                    writer.writerow([
                        video, config['name'], f"{orig_size_kb:.2f}", f"{size_kb:.2f}", f"{ratio:.3f}x", 
                        f"{mse:.2f}", f"{skip_p:.1f}", f"{intra_p:.1f}", f"{mot_p:.1f}", f"{sol_p:.1f}", f"{spa_p:.1f}"
                    ])
                    
                    print(
                        f"{config['name']:<26} | "
                        f"{size_kb:>10.2f} | "
                        f"{ratio:>6.2f}x | "
                        f"{mse:>7.2f} | "
                        f"{skip_p:>7.1f} | "
                        f"{intra_p:>7.1f} | "
                        f"{mot_p:>8.1f} | "
                        f"{sol_p:>7.1f} | "
                        f"{spa_p:>9.1f}"
                    )
                except Exception as e:
                    print(f"{config['name']:<26} | FAILED: {str(e)}")
                finally:
                    if os.path.exists(out_file):
                        os.remove(out_file)
            print("=" * 125)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    run_batch_ablation(media_folder="test_vid", output_csv="rdo_report_v3.csv", full_test=args.full)