import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import os
import argparse
import csv
import cv2
import numpy as np
import subprocess
from encoder import run_encoder
from decoder import decode_and_profile

class GitBenchmarkTracker:
    @staticmethod
    def get_reports_head_csv(filepath):
        """Reads the version of the CSV specifically from the 'reports' branch."""
        try:
            result = subprocess.run(['git', 'show', f'reports:{filepath}'], capture_output=True, text=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError:
            return None 

    @staticmethod
    def parse_csv_to_dict(csv_content):
        if not csv_content: return {}
        reader = csv.DictReader(io.StringIO(csv_content))
        return {(row['Video'], row['Config']): row for row in reader if 'Video' in row}

    @staticmethod
    def generate_report_and_commit(summary_csv, mse_csv):
        print("\n" + "="*90)
        print("📊 GIT BENCHMARK DELTA REPORT")
        print("="*90)

        try:
            subprocess.run(['git', 'status'], capture_output=True, check=True)
        except subprocess.CalledProcessError:
            print("⚠️ Git repository not found. Skipping auto-commit.")
            return

        old_csv_data = GitBenchmarkTracker.get_reports_head_csv(summary_csv)
        old_data = GitBenchmarkTracker.parse_csv_to_dict(old_csv_data)
        
        with open(summary_csv, 'r', encoding='utf-8') as f:
            new_csv_content = f.read()
            new_data = GitBenchmarkTracker.parse_csv_to_dict(new_csv_content)
            
        with open(mse_csv, 'r', encoding='utf-8') as f:
            new_mse_content = f.read()

        if not old_data:
            print("No previous benchmark found in 'reports' branch. This will be the baseline commit.")
        else:
            print(f"{'Video':<20} | {'Size (KB)':>12} | {'MSE':>10} | {'Enc FPS':>12}")
            print("-" * 65)
            
            for key, new_row in new_data.items():
                vid_name = key[0][:18] 
                old_row = old_data.get(key)
                
                if not old_row:
                    print(f"{vid_name:<20} | [NEW VIDEO IN CORPUS]")
                    continue
                
                size_delta = float(new_row['Size_KB']) - float(old_row['Size_KB'])
                mse_delta = float(new_row['MSE']) - float(old_row['MSE'])
                fps_delta = float(new_row['Enc_FPS']) - float(old_row['Enc_FPS'])
                
                size_str = f"{size_delta:+.2f} KB"
                mse_str = f"{mse_delta:+.2f}"
                fps_str = f"{fps_delta:+.1f}"
                
                size_sym = "🟢" if size_delta < -0.5 else ("🔴" if size_delta > 0.5 else "⚪")
                mse_sym = "🟢" if mse_delta < -0.1 else ("🔴" if mse_delta > 0.1 else "⚪")
                fps_sym = "🟢" if fps_delta > 1.0 else ("🔴" if fps_delta < -1.0 else "⚪")
                
                print(f"{vid_name:<20} | {size_str:>8} {size_sym} | {mse_str:>6} {mse_sym} | {fps_str:>8} {fps_sym}")

        print("-" * 90)
        print("Executing Auto-Commit to 'reports' branch...")
        
        try:
            curr_branch = subprocess.run(['git', 'branch', '--show-current'], capture_output=True, text=True, check=True).stdout.strip()
            
            if curr_branch != "reports":
                # Stash active work to ensure clean branch switch
                stash_out = subprocess.run(['git', 'stash', 'push', '--include-untracked', '-m', 'auto-stash-benchmarks'], capture_output=True, text=True)
                did_stash = "No local changes to save" not in stash_out.stdout

                # Jump to reports branch
                try:
                    subprocess.run(['git', 'checkout', 'reports'], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, check=True)
                except subprocess.CalledProcessError:
                    subprocess.run(['git', 'checkout', '-b', 'reports'], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, check=True)
                
                # Write CSVs natively into the reports branch
                with open(summary_csv, 'w', encoding='utf-8') as f: f.write(new_csv_content)
                with open(mse_csv, 'w', encoding='utf-8') as f: f.write(new_mse_content)
                
                # Commit
                subprocess.run(['git', 'add', summary_csv, mse_csv], check=True)
                subprocess.run(['git', 'commit', '-m', 'Auto-commit: Benchmark results updated'], capture_output=True, check=True)
                
                # Jump back to Dev branch
                subprocess.run(['git', 'checkout', curr_branch], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, check=True)
                
                # Pop stash and restore CSVs to the local IDE view
                if did_stash:
                    subprocess.run(['git', 'stash', 'pop'], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, check=True)
                
                with open(summary_csv, 'w', encoding='utf-8') as f: f.write(new_csv_content)
                with open(mse_csv, 'w', encoding='utf-8') as f: f.write(new_mse_content)
                
                print(f"✅ Successfully committed to 'reports' branch and returned safely to '{curr_branch}'.")
            else:
                subprocess.run(['git', 'add', summary_csv, mse_csv], check=True)
                subprocess.run(['git', 'commit', '-m', 'Auto-commit: Benchmark results updated'], capture_output=True, check=True)
                print("✅ Successfully committed to current 'reports' branch.")

        except subprocess.CalledProcessError as e:
            print("⚠️ Auto-commit to branch failed:", e)
            
def run_batch_ablation(media_folder=".", summary_csv="sobvc_report_v1.csv", mse_csv="frame_mse_report.csv"):
    videos = sorted([f for f in os.listdir(media_folder) if f.endswith('.mp4')])
    if not videos:
        print(f"No videos found in {media_folder}/")
        return

    test_configs = [
        {"name": "SOBVC Default", "motion": True, "palette": True, "dynamic_boxing": True, "background": True, "dct": True}
    ]

    # Open both CSVs for writing
    with open(summary_csv, mode='w', newline='', encoding='utf-8') as sum_file, \
         open(mse_csv, mode='w', newline='', encoding='utf-8') as mse_file:
        
        sum_writer = csv.writer(sum_file)
        mse_writer = csv.writer(mse_file)
        
        # Write Headers
        sum_writer.writerow(["Video", "Config", "Size_KB", "Ratio", "MSE", "Enc_FPS", "Dec_FPS"])
        mse_writer.writerow(["Video", "Config", "Frame_Index", "MSE"])

        print("🚀 STARTING SOBVC BATCH ABLATION ON VIDEO CORPUS...\n")

        for video in videos:
            video_path = os.path.join(media_folder, video)
            orig_size_kb = os.path.getsize(video_path) / 1024
            
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            print(f"\n▶ Testing: {video} (Orig: {orig_size_kb:.2f} KB | Frames: {total_frames}/{total_frames})")
            print("=" * 145)
            print(f"{'Algorithm':<22} | {'Size (KB)':>9} | {'Ratio':>6} | {'MSE':>5} | {'Enc FPS':>7} | {'Dec FPS':>7} | SOBVC Telemetry")
            print("-" * 145)
            
            for config in test_configs:
                out_file = f"temp_sobvc.nam"
                try:
                    enc_time, tel = run_encoder(video_path, out_file, config)
                    dec_time, frame_mses = decode_and_profile(video_path, out_file)
                    
                    # Write frame granular MSE
                    for f_idx, f_mse in enumerate(frame_mses):
                        mse_writer.writerow([video, config['name'], f_idx, f"{f_mse:.4f}"])
                    
                    size_kb = os.path.getsize(out_file) / 1024
                    ratio = size_kb / orig_size_kb
                    mean_mse = np.mean(frame_mses) if frame_mses else 0.0
                    
                    enc_fps = total_frames / max(enc_time, 0.001)
                    dec_fps = total_frames / max(dec_time, 0.001)
                    
                    # Write to Summary CSV
                    sum_writer.writerow([video, config['name'], f"{size_kb:.2f}", f"{ratio:.3f}", f"{mean_mse:.2f}", f"{enc_fps:.1f}", f"{dec_fps:.1f}"])
                    
                    std_mse = np.std(frame_mses) if frame_mses else 0.0
                    spikes = [i for i, mse in enumerate(frame_mses) if mse > (mean_mse + 2 * std_mse) and mse > 3.0]
                    
                    # Update Telemetry String for SOBVC
                    spawns = tel.get('spawns', 0)
                    moves = tel.get('moves', 0)
                    transitions = tel.get('transitions', 0)
                    despawns = tel.get('despawns', 0)
                    locks = tel.get('locks', 0)
                    
                    mode_str = f"Spw:{spawns} Mov:{moves} Trns:{transitions} Lk:{locks} Dspw:{despawns}"
                    
                    # --- Color Logic ---
                    GREEN = '\033[92m'
                    YELLOW = '\033[93m'
                    RED = '\033[91m'
                    RESET = '\033[0m'

                    if ratio < 1.0 and mean_mse < 5.0:
                        ratio_color, mse_color = GREEN, GREEN
                    elif ratio > 1.0 and mean_mse > 10.0:
                        ratio_color, mse_color = RED, RED
                    else:
                        ratio_color = GREEN if ratio < 1.0 else (RED if ratio > 1.0 else YELLOW)
                        mse_color = GREEN if mean_mse < 5.0 else (RED if mean_mse > 10.0 else YELLOW)

                    r_txt = f"{ratio:.2f}x"
                    m_txt = f"{mean_mse:.2f}"
                    
                    # Pad strings manually before applying invisible ANSI codes
                    ratio_fmt = f"{' ' * (6 - len(r_txt))}{ratio_color}{r_txt}{RESET}"
                    mse_fmt = f"{' ' * (5 - len(m_txt))}{mse_color}{m_txt}{RESET}"
                    
                    print(
                        f"{config['name']:<22} | "
                        f"{size_kb:>9.2f} | "
                        f"{ratio_fmt} | "
                        f"{mse_fmt} | "
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

    # Trigger the Git tracker after all tests finish
    GitBenchmarkTracker.generate_report_and_commit(summary_csv, mse_csv)

if __name__ == "__main__":
    run_batch_ablation()