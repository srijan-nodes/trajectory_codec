import os
import sys
import time
import subprocess
import json
import cv2
import numpy as np
from pathlib import Path

FFMPEG_DIR = r"C:\Users\dsrij\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-essentials_build\bin"
os.environ["PATH"] = FFMPEG_DIR + os.pathsep + r"C:\Users\dsrij\.cargo\bin" + os.pathsep + os.environ.get("PATH", "")

sys.path.insert(0, str(Path(__file__).parent / "final_commit"))
import encoder as py_encoder
import decoder as py_decoder

RUST_EXE = str(Path(__file__).parent / "traj_cddec_rs" / "target" / "release" / "traj_cddec_rs.exe")

def calculate_psnr(mse):
    if mse <= 1e-12:
        return float('inf')
    return 10.0 * np.log10((255.0 ** 2) / mse)

def get_original_frames(video_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return []
    h, w = frame.shape[:2]
    h = h - (h % 16)
    w = w - (w % 16)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    frames = []
    while True:
        if max_frames and len(frames) >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame[:h, :w], cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    cap.release()
    return frames

def get_frame_groups(total_frames, video_name):
    groups = []
    if "countdown" in video_name:
        group_defs = [
            ("Frames 0-125 (Initial Countdown '3')", 0, 126),
            ("Frames 126-250 (Digit Transition '2')", 126, 251),
            ("Frames 251-375 (Digit Transition '1')", 251, 376),
            ("Frames 376-500 ('0' & Explosion Start)", 376, 501),
            ("Frames 501-625 (Fireworks Spark Cascade 1)", 501, 626),
            ("Frames 626-754 (Fireworks Spark Cascade 2)", 626, total_frames),
        ]
        for name, start, end in group_defs:
            groups.append({
                "name": name,
                "start": start,
                "end": min(end, total_frames)
            })
    elif "ball" in video_name:
        group_defs = [
            ("Frames 0-20 (Descent & Impact)", 0, 21),
            ("Frames 21-40 (Rebound Lift-off)", 21, 41),
            ("Frames 41-60 (Apex Ascent & Hover)", 41, 61),
            ("Frames 61-80 (Secondary Descent)", 61, 81),
            ("Frames 81-100 (Secondary Impact & Settle)", 81, total_frames),
        ]
        for name, start, end in group_defs:
            groups.append({
                "name": name,
                "start": start,
                "end": min(end, total_frames)
            })
    else:
        # 90-frame videos split into 15-frame groups
        step = 15
        for start in range(0, total_frames, step):
            end = min(start + step, total_frames)
            groups.append({
                "name": f"Frames {start:02d}-{end-1:02d}",
                "start": start,
                "end": end
            })
    return groups

def benchmark_video(video_path):
    vname = os.path.basename(video_path)
    print(f"\n{'='*75}")
    print(f"BENCHMARKING SUITE VIDEO: {vname}")
    print(f"{'='*75}")

    orig_frames = get_original_frames(video_path)
    total_frames = len(orig_frames)
    orig_bytes = os.path.getsize(video_path)
    
    # -------------------------------------------------------------
    # 1. PYTHON ENCODER & DECODER
    # -------------------------------------------------------------
    py_nam = f"temp_py_{vname}.nam"
    print(f"Running Python V6 Codec on {vname}...")
    t0 = time.time()
    py_enc_res = py_encoder.encode(video_path, py_nam)
    py_enc_time = time.time() - t0
    py_comp_bytes = os.path.getsize(py_nam)
    py_enc_fps = total_frames / py_enc_time if py_enc_time > 0 else 0

    t0 = time.time()
    py_dec_res = py_decoder.decode(py_nam, original_video=video_path)
    py_dec_time = time.time() - t0
    py_dec_fps = total_frames / py_dec_time if py_dec_time > 0 else 0

    # Collect Python per-frame MSE
    py_frame_mses = []
    for df, of in zip(py_dec_res["frames"], orig_frames):
        if df.ndim == 3 and df.shape[2] == 3:
            df_g = cv2.cvtColor(df, cv2.COLOR_BGR2GRAY)
        else:
            df_g = df.squeeze()
        mse = float(np.mean((of.astype(np.float64) - df_g.astype(np.float64)) ** 2))
        py_frame_mses.append(mse)

    if os.path.exists(py_nam):
        os.remove(py_nam)

    # -------------------------------------------------------------
    # 2. RUST ENCODER & DECODER
    # -------------------------------------------------------------
    rs_nam = f"temp_rs_{vname}.nam"
    print(f"Running Rust TrajCDDec on {vname}...")
    
    # Encode with Rust
    t0 = time.time()
    cmd_enc = [RUST_EXE, "encode", "--input", video_path, "--output", rs_nam]
    res_enc = subprocess.run(cmd_enc, capture_output=True, text=True)
    rs_enc_time = time.time() - t0
    rs_comp_bytes = os.path.getsize(rs_nam) if os.path.exists(rs_nam) else 0
    rs_enc_fps = total_frames / rs_enc_time if rs_enc_time > 0 else 0

    # Decode with Rust
    t0 = time.time()
    cmd_dec = [RUST_EXE, "decode", "--input", rs_nam, "--orig", video_path]
    res_dec = subprocess.run(cmd_dec, capture_output=True, text=True)
    rs_dec_time = time.time() - t0
    rs_dec_fps = total_frames / rs_dec_time if rs_dec_time > 0 else 0

    # Parse Rust decode output for exact internal decode speed & MSE
    # Also verify via Python decoder reading Rust bitstream to extract per-frame MSE!
    rs_py_dec = py_decoder.decode(rs_nam, original_video=video_path)
    rs_frame_mses = []
    for df, of in zip(rs_py_dec["frames"], orig_frames):
        if df.ndim == 3 and df.shape[2] == 3:
            df_g = cv2.cvtColor(df, cv2.COLOR_BGR2GRAY)
        else:
            df_g = df.squeeze()
        mse = float(np.mean((of.astype(np.float64) - df_g.astype(np.float64)) ** 2))
        rs_frame_mses.append(mse)

    if os.path.exists(rs_nam):
        os.remove(rs_nam)

    # Extract internal pure decode time reported by Rust CLI output
    pure_rs_dec_fps = rs_dec_fps
    pure_rs_enc_fps = rs_enc_fps
    for line in res_dec.stdout.splitlines():
        if "Successfully decoded" in line and "FPS" in line:
            try:
                parts = line.split("(")
                fps_part = parts[-1].replace("FPS)", "").strip()
                pure_rs_dec_fps = float(fps_part)
            except:
                pass
    for line in res_enc.stdout.splitlines():
        if "Successfully encoded" in line and "FPS" in line:
            try:
                parts = line.split("(")
                fps_part = parts[-1].replace("FPS)", "").strip()
                pure_rs_enc_fps = float(fps_part)
            except:
                pass

    # -------------------------------------------------------------
    # 3. FRAME GROUP ANALYSIS
    # -------------------------------------------------------------
    groups = get_frame_groups(total_frames, vname)
    group_results = []

    for g in groups:
        s, e = g["start"], g["end"]
        py_g_mses = py_frame_mses[s:e] if e <= len(py_frame_mses) else []
        rs_g_mses = rs_frame_mses[s:e] if e <= len(rs_frame_mses) else []

        py_avg = float(np.mean(py_g_mses)) if py_g_mses else 0.0
        py_max = float(np.max(py_g_mses)) if py_g_mses else 0.0
        py_psnr = calculate_psnr(py_avg)

        rs_avg = float(np.mean(rs_g_mses)) if rs_g_mses else 0.0
        rs_max = float(np.max(rs_g_mses)) if rs_g_mses else 0.0
        rs_psnr = calculate_psnr(rs_avg)

        # Mark verification: PSNR >= 40.0 dB or MSE <= 6.5
        is_mark_ok = bool(rs_psnr >= 40.0 or rs_avg <= 6.5)

        group_results.append({
            "name": str(g["name"]),
            "count": int(e - s),
            "py_avg_mse": float(py_avg),
            "py_max_mse": float(py_max),
            "py_psnr": float(py_psnr),
            "rs_avg_mse": float(rs_avg),
            "rs_max_mse": float(rs_max),
            "rs_psnr": float(rs_psnr),
            "is_mark_ok": is_mark_ok,
        })

    py_total_avg_mse = float(np.mean(py_frame_mses))
    py_total_psnr = calculate_psnr(py_total_avg_mse)

    rs_total_avg_mse = float(np.mean(rs_frame_mses))
    rs_total_psnr = calculate_psnr(rs_total_avg_mse)

    enc_speedup = pure_rs_enc_fps / py_enc_fps if py_enc_fps > 0 else 1.0
    dec_speedup = pure_rs_dec_fps / py_dec_fps if py_dec_fps > 0 else 1.0

    print(f"Results for {vname}:")
    print(f"  Orig Size:    {orig_bytes:,} B")
    print(f"  Py Size:      {py_comp_bytes:,} B ({(1-py_comp_bytes/orig_bytes)*100:.1f}% reduction)")
    print(f"  Rust Size:    {rs_comp_bytes:,} B ({(1-rs_comp_bytes/orig_bytes)*100:.1f}% reduction)")
    print(f"  Enc Speed:    Py = {py_enc_fps:.1f} FPS | Rust = {pure_rs_enc_fps:.1f} FPS (Speedup: {enc_speedup:.2f}x)")
    print(f"  Dec Speed:    Py = {py_dec_fps:.1f} FPS | Rust = {pure_rs_dec_fps:.1f} FPS (Speedup: {dec_speedup:.2f}x)")
    print(f"  Overall MSE:  Py = {py_total_avg_mse:.4f} ({py_total_psnr:.2f} dB) | Rust = {rs_total_avg_mse:.4f} ({rs_total_psnr:.2f} dB)")
    
    return {
        "video": vname,
        "total_frames": total_frames,
        "orig_bytes": orig_bytes,
        "py_comp_bytes": py_comp_bytes,
        "rs_comp_bytes": rs_comp_bytes,
        "py_enc_fps": py_enc_fps,
        "rs_enc_fps": pure_rs_enc_fps,
        "enc_speedup": enc_speedup,
        "py_dec_fps": py_dec_fps,
        "rs_dec_fps": pure_rs_dec_fps,
        "dec_speedup": dec_speedup,
        "py_avg_mse": py_total_avg_mse,
        "py_psnr": py_total_psnr,
        "rs_avg_mse": rs_total_avg_mse,
        "rs_psnr": rs_total_psnr,
        "group_results": group_results,
    }

def main():
    test_dir = Path("test_vid")
    videos = sorted([
        f for f in test_dir.glob("*.mp4")
        if not any(sub in f.name for sub in ["_2f", "_5f", "_10f", "05_break_entropy"])
    ])

    print(f"Found {len(videos)} videos in the test suite.")
    all_results = []
    
    for v in videos:
        res = benchmark_video(str(v))
        all_results.append(res)

    # Save detailed JSON report
    with open("benchmark_suite_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nBenchmark successfully completed and written to benchmark_suite_results.json.")

if __name__ == "__main__":
    main()
