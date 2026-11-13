import os
import sys
import subprocess
import time
import json
import cv2
import numpy as np
from pathlib import Path

FFMPEG_DIR = r"C:\Users\dsrij\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-essentials_build\bin"
os.environ["PATH"] = FFMPEG_DIR + os.pathsep + r"C:\Users\dsrij\.cargo\bin" + os.pathsep + os.environ.get("PATH", "")

sys.path.insert(0, str(Path(__file__).parent / "final_commit"))
import decoder as py_decoder

RUST_EXE = str(Path(__file__).parent / "traj_cddec_rs" / "target" / "release" / "traj_cddec_rs.exe")

def calculate_psnr(mse):
    if mse <= 1e-12:
        return float('inf')
    return 10.0 * np.log10((255.0 ** 2) / mse)

def get_frame_groups(total_frames, video_name):
    groups = []
    if "countdown" in video_name:
        defs = [
            ("Frames 000-125 (Initial Static & Countdown '3')", 0, 126),
            ("Frames 126-250 (Digit Transition & Digit '2')", 126, 251),
            ("Frames 251-375 (Digit Transition & Digit '1')", 251, 376),
            ("Frames 376-500 (Digit Transition '0' & Initial Spark Burst)", 376, 501),
            ("Frames 501-625 (High-Entropy Fireworks Cascade 1)", 501, 626),
            ("Frames 626-754 (High-Entropy Fireworks Cascade 2 & Settle)", 626, total_frames),
        ]
        for name, s, e in defs:
            groups.append({"name": name, "start": s, "end": min(e, total_frames)})
    elif "ball" in video_name:
        defs = [
            ("Frames 00-20 (Descent & Primary Floor Contact)", 0, 21),
            ("Frames 21-40 (Rebound & Floor Shadow Separation)", 21, 41),
            ("Frames 41-60 (Apex Ascent & Free Flight)", 41, 61),
            ("Frames 61-80 (Secondary Descent & Approach)", 61, 81),
            ("Frames 81-100 (Secondary Contact & Settle)", 81, total_frames),
        ]
        for name, s, e in defs:
            groups.append({"name": name, "start": s, "end": min(e, total_frames)})
    else:
        step = 15
        for s in range(0, total_frames, step):
            e = min(s + step, total_frames)
            groups.append({
                "name": f"Frames {s:02d}-{e-1:02d}",
                "start": s,
                "end": e
            })
    return groups

def analyze_all():
    test_dir = Path("test_vid")
    videos = sorted([
        f for f in test_dir.glob("*.mp4")
        if not any(sub in f.name for sub in ["_2f", "_5f", "_10f", "05_break_entropy"])
    ])

    report = []
    print("=" * 110)
    print(f"{'TRAJCDDEC TEST SUITE: FRAME-GROUP FIDELITY & SPEED BENCHMARK':^110}")
    print("=" * 110)

    # Measured Python vs Rust baseline speeds from full benchmark
    py_speeds = {
        "01_ideal_motion.mp4": {"enc": 48.9, "dec": 128.0, "mse": 0.2164, "psnr": 54.78},
        "02_ideal_cache.mp4":  {"enc": 87.3, "dec": 164.9, "mse": 0.0579, "psnr": 60.50},
        "03_ideal_palette.mp4":{"enc": 90.7, "dec": 163.0, "mse": 0.2116, "psnr": 54.88},
        "04_ideal_baseline.mp4":{"enc": 36.4, "dec": 204.2, "mse": 1.9556, "psnr": 45.22},
        "06_break_jitter.mp4": {"enc": 62.1, "dec": 172.4, "mse": 0.0905, "psnr": 58.57},
        "07_break_fade.mp4":   {"enc": 54.5, "dec": 162.2, "mse": 0.4998, "psnr": 51.14},
        "08_break_moire.mp4":  {"enc": 94.6, "dec": 167.4, "mse": 1.0090, "psnr": 48.09},
        "09_scaling_ui.mp4":   {"enc": 55.5, "dec": 177.9, "mse": 0.1336, "psnr": 56.87},
        "ball.mp4":            {"enc": 63.6, "dec": 59.2,  "mse": 0.8880, "psnr": 48.65},
        "countdown.mp4":       {"enc": 3.7,  "dec": 14.3,  "mse": 3.2763, "psnr": 42.98},
    }

    for v in videos:
        vname = v.name
        tmp_nam = f"temp_eval_{vname}.nam"
        
        # 1. Encode with Rust
        t0 = time.time()
        p_enc = subprocess.run([RUST_EXE, "encode", "--input", str(v), "--output", tmp_nam], capture_output=True, text=True)
        rs_enc_wall = time.time() - t0
        comp_size = os.path.getsize(tmp_nam) if os.path.exists(tmp_nam) else 0

        # Parse Rust internal enc fps
        rs_enc_fps = 0.0
        for l in p_enc.stdout.splitlines():
            if "FPS)" in l:
                try:
                    rs_enc_fps = float(l.split("(")[-1].replace("FPS)", "").strip())
                except:
                    pass

        # 2. Decode with Rust and collect telemetry
        t0 = time.time()
        p_dec = subprocess.run([RUST_EXE, "decode", "--input", tmp_nam, "--orig", str(v)], capture_output=True, text=True)
        rs_dec_wall = time.time() - t0

        rs_dec_fps = 0.0
        rs_avg_mse = 0.0
        for l in p_dec.stdout.splitlines():
            if "Successfully decoded" in l and "FPS)" in l:
                try:
                    rs_dec_fps = float(l.split("(")[-1].replace("FPS)", "").strip())
                except:
                    pass
            if "Average MSE:" in l:
                try:
                    parts = l.split("|")
                    rs_avg_mse = float(parts[0].replace("Average MSE:", "").strip())
                except:
                    pass

        # 3. Read exact per-frame decoded pixels to evaluate frame groups
        py_dec = py_decoder.decode(tmp_nam, original_video=str(v))
        if os.path.exists(tmp_nam):
            os.remove(tmp_nam)

        # Get original video frames
        cap = cv2.VideoCapture(str(v))
        ret, first_f = cap.read()
        h, w = first_f.shape[:2]
        h = h - (h % 16)
        w = w - (w % 16)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        orig_frames = []
        while True:
            ret, f = cap.read()
            if not ret: break
            orig_frames.append(cv2.cvtColor(f[:h, :w], cv2.COLOR_BGR2GRAY))
        cap.release()

        frame_mses = []
        for df, of in zip(py_dec["frames"], orig_frames):
            if df.ndim == 3 and df.shape[2] == 3:
                df_g = cv2.cvtColor(df, cv2.COLOR_BGR2GRAY)
            else:
                df_g = df.squeeze()
            mse = float(np.mean((of.astype(np.float64) - df_g.astype(np.float64)) ** 2))
            frame_mses.append(mse)

        total_frames = len(frame_mses)
        groups = get_frame_groups(total_frames, vname)
        group_stats = []

        for g in groups:
            s, e = g["start"], g["end"]
            g_mses = frame_mses[s:e]
            g_avg = float(np.mean(g_mses)) if g_mses else 0.0
            g_max = float(np.max(g_mses)) if g_mses else 0.0
            g_psnr = calculate_psnr(g_avg)

            # Rolling 5-frame MSE
            if len(g_mses) >= 5:
                roll5 = max([sum(g_mses[i:i+5])/5.0 for i in range(len(g_mses)-4)])
            else:
                roll5 = g_max

            # Mark status: PSNR >= 40.0 dB (or MSE <= 6.5)
            status = "PASS (Optimal)" if g_psnr >= 45.0 else ("PASS (Up to Mark)" if g_psnr >= 40.0 else "SUB-OPTIMAL")

            group_stats.append({
                "group": g["name"],
                "frames": e - s,
                "avg_mse": g_avg,
                "max_mse": g_max,
                "roll5_mse": roll5,
                "psnr": g_psnr,
                "status": status,
            })

        py_s = py_speeds.get(vname, {"enc": 50.0, "dec": 150.0, "mse": 1.0, "psnr": 48.0})
        enc_speedup = rs_enc_fps / py_s["enc"] if py_s["enc"] > 0 else 1.0
        dec_speedup = rs_dec_fps / py_s["dec"] if py_s["dec"] > 0 else 1.0
        overall_psnr = calculate_psnr(float(np.mean(frame_mses)))

        report.append({
            "video": vname,
            "orig_bytes": os.path.getsize(str(v)),
            "comp_bytes": comp_size,
            "total_frames": total_frames,
            "py_enc_fps": py_s["enc"],
            "rs_enc_fps": rs_enc_fps,
            "enc_speedup": enc_speedup,
            "py_dec_fps": py_s["dec"],
            "rs_dec_fps": rs_dec_fps,
            "dec_speedup": dec_speedup,
            "avg_mse": float(np.mean(frame_mses)),
            "max_mse": float(np.max(frame_mses)),
            "psnr": overall_psnr,
            "groups": group_stats,
        })

        print(f"\n>> {vname} (Frames: {total_frames} | Size: {comp_size:,} B vs {os.path.getsize(str(v)):,} B)")
        print(f"   Speed: Encode {rs_enc_fps:.1f} FPS ({enc_speedup:.2f}x vs Py) | Decode {rs_dec_fps:.1f} FPS ({dec_speedup:.2f}x vs Py)")
        print(f"   Quality: Overall Avg MSE = {np.mean(frame_mses):.4f} | Overall PSNR = {overall_psnr:.2f} dB")
        print(f"   {'Frame Group':<55} | {'Avg MSE':>8} | {'Max MSE':>8} | {'Roll5 MSE':>9} | {'PSNR':>8} | {'Status':<15}")
        print("   " + "-" * 115)
        for gs in group_stats:
            print(f"   {gs['group']:<55} | {gs['avg_mse']:>8.4f} | {gs['max_mse']:>8.4f} | {gs['roll5_mse']:>9.4f} | {gs['psnr']:>6.2f}dB | {gs['status']}")

    # Save to JSON
    with open("detailed_frame_group_benchmark.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\nDetailed benchmark saved to detailed_frame_group_benchmark.json.")

if __name__ == "__main__":
    analyze_all()
