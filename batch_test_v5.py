import os
import argparse
import csv
import cv2
import numpy as np
from lab_v5 import run_encoder, decode_and_profile

def run_batch_ablation(media_folder="test_vid", output_csv="rdo_report_v5.csv", mse_csv="frame_mse_report.csv"):
    videos = sorted([f for f in os.listdir(media_folder) if f.endswith(('.mp4', '.y4m'))])
    if not videos:
        print(f"No videos found in {media_folder}/")
        return

    test_configs = [
        {"name": "V5 Hybrid Arena", "motion": True, "palette": True, "dynamic_boxing": True, "background": True, "dct": True}
    ]

    with open(mse_csv, mode='w', newline='') as mse_file:
        mse_writer = csv.writer(mse_file)
        mse_writer.writerow(["Video", "Config", "Frame_Index", "MSE"])

        print("🚀 STARTING V5 HYBRID ABLATION ON SYNTHETIC VIDEO CORPUS...\n")

        for video in videos:
            video_path = os.path.join(media_folder, video)
            orig_size_kb = os.path.getsize(video_path) / 1024
            
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            print(f"\n▶ Testing: {video} (Orig: {orig_size_kb:.2f} KB | Frames: {total_frames}/{total_frames})")
            print("=" * 145)
            print(f"{'Algorithm':<22} | {'Size (KB)':>9} | {'Ratio':>6} | {'MSE':>5} | {'Enc FPS':>7} | {'Dec FPS':>7} | Modes Used (%)")
            print("-" * 145)
            
            for config in test_configs:
                out_file = f"temp_v5.nam"
                try:
                    enc_time, tel = run_encoder(video_path, out_file, config)
                    dec_time, frame_mses = decode_and_profile(video_path, out_file)
                    
                    for f_idx, f_mse in enumerate(frame_mses):
                        mse_writer.writerow([video, config['name'], f_idx, f"{f_mse:.4f}"])
                    
                    size_kb = os.path.getsize(out_file) / 1024
                    ratio = size_kb / orig_size_kb
                    mean_mse = np.mean(frame_mses)
                    
                    enc_fps = total_frames / max(enc_time, 0.001)
                    dec_fps = total_frames / max(dec_time, 0.001)
                    
                    std_mse = np.std(frame_mses)
                    spikes = [i for i, mse in enumerate(frame_mses) if mse > (mean_mse + 2 * std_mse) and mse > 3.0]
                    
                    total = max(1, tel.get("blocks_total", 1))
                    
                    # Fast Lane
                    sk = (tel.get("blocks_skip", 0) / total) * 100
                    mo = (tel.get("blocks_motion", 0) / total) * 100
                    bg = (tel.get("blocks_background", 0) / total) * 100
                    so = (tel.get("blocks_solid", 0) / total) * 100
                    pa = (tel.get("blocks_palette", 0) / total) * 100
                    sp = (tel.get("blocks_spatial", 0) / total) * 100
                    rw = (tel.get("blocks_raw", 0) / total) * 100
                    
                    # Desperation Arena
                    fm = (tel.get("blocks_faded_mot", 0) / total) * 100
                    dc = (tel.get("blocks_dct", 0) / total) * 100
                    di = (tel.get("blocks_dithered", 0) / total) * 100
                    
                    mode_str = (f"Fast: Sk:{sk:.0f} Mo:{mo:.0f} BG:{bg:.0f} So:{so:.0f} Pa:{pa:.0f} Sp:{sp:.0f} || "
                                f"Arena: Fd:{fm:.1f} DCT:{dc:.1f} Di:{di:.1f}")
                    
                    print(
                        f"{config['name']:<22} | "
                        f"{size_kb:>9.2f} | "
                        f"{ratio:>5.2f}x | "
                        f"{mean_mse:>5.2f} | "
                        f"{enc_fps:>7.1f} | "
                        f"{dec_fps:>7.1f} | "
                        f"{mode_str}"
                    )
                    
                    if spikes:
                        spike_str = ", ".join([str(f) for f in spikes[:12]])
                        if len(spikes) > 12: spike_str += f" (+{len(spikes)-12} more)"
                        print(f"   ⚠️ MSE Spikes detected at frames: {spike_str}")
                    
                except Exception as e:
                    print(f"{config['name']:<22} | FAILED: {str(e)}")
                    import traceback
                    traceback.print_exc()
                finally:
                    if os.path.exists(out_file): os.remove(out_file)
            print("=" * 145)
            
        print(f"\n✅ Full frame-by-frame MSE report saved to: {mse_csv}")

if __name__ == "__main__":
    run_batch_ablation()