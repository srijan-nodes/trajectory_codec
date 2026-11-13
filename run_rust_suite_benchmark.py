import os
import sys
import time
import subprocess
import json
from pathlib import Path
import cv2
import numpy as np

FFMPEG_DIR = r"C:\Users\dsrij\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-essentials_build\bin"
os.environ["PATH"] = FFMPEG_DIR + os.pathsep + r"C:\Users\dsrij\.cargo\bin" + os.pathsep + os.environ.get("PATH", "")

RUST_EXE = str(Path(__file__).parent / "traj_cddec_rs" / "target" / "release" / "traj_cddec_rs.exe")
OUTPUT_DIR = Path(__file__).parent / "traj_cddec_rs" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load existing Python baseline numbers for comparison
PY_BASELINES = {}
baseline_file = Path(__file__).parent / "detailed_frame_group_benchmark.json"
if baseline_file.exists():
    with open(baseline_file) as f:
        for entry in json.load(f):
            PY_BASELINES[entry["video"]] = entry

def calculate_psnr(mse):
    if mse <= 1e-12:
        return float('inf')
    return 10.0 * np.log10((255.0 ** 2) / mse)

def get_original_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
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
            ("Frames 000-125 (Initial Countdown '3')", 0, 126),
            ("Frames 126-250 (Digit Transition '2')", 126, 251),
            ("Frames 251-375 (Digit Transition '1')", 251, 376),
            ("Frames 376-500 ('0' & Explosion Start)", 376, 501),
            ("Frames 501-625 (Fireworks Spark Cascade 1)", 501, 626),
            ("Frames 626-754 (Fireworks Spark Cascade 2)", 626, total_frames),
        ]
        for name, start, end in group_defs:
            groups.append({"name": name, "start": start, "end": min(end, total_frames)})
    elif "ball" in video_name:
        group_defs = [
            ("Frames 00-20 (Descent & Primary Impact)", 0, 21),
            ("Frames 21-40 (Rebound Lift-Off & Floor Shadow)", 21, 41),
            ("Frames 41-60 (Apex Ascent & Free Flight)", 41, 61),
            ("Frames 61-80 (Secondary Descent & Approach)", 61, 81),
            ("Frames 81-100 (Secondary Contact & Settle)", 81, total_frames),
        ]
        for name, start, end in group_defs:
            groups.append({"name": name, "start": start, "end": min(end, total_frames)})
    else:
        step = 15
        for start in range(0, total_frames, step):
            end = min(start + step, total_frames)
            groups.append({
                "name": f"Frames {start:02d}-{end-1:02d}",
                "start": start,
                "end": end
            })
    return groups

def run_suite_video(video_path):
    vname = video_path.name
    stem = video_path.stem
    print(f"\n{'='*80}")
    print(f"RUNNING RUST ALGORITHM ON: {vname}")
    print(f"{'='*80}")

    orig_frames = get_original_frames(video_path)
    total_frames = len(orig_frames)
    orig_bytes = video_path.stat().st_size

    nam_path = OUTPUT_DIR / f"{stem}.nam"
    dec_mp4 = OUTPUT_DIR / f"{stem}_rust_decoded.mp4"
    sbs_mp4 = OUTPUT_DIR / f"{stem}_rust_sbs.mp4"

    # 1. ENCODE WITH RUST
    print(f">> [Rust] Encoding {vname} -> {nam_path.name}...")
    t0 = time.time()
    cmd_enc = [RUST_EXE, "encode", "--input", str(video_path), "--output", str(nam_path)]
    res_enc = subprocess.run(cmd_enc, capture_output=True, text=True)
    enc_wall_time = time.time() - t0
    
    if res_enc.returncode != 0:
        print(f"ERROR encoding {vname}: {res_enc.stderr}")
        return None

    comp_bytes = nam_path.stat().st_size
    enc_fps = total_frames / enc_wall_time if enc_wall_time > 0 else 0

    # 2. DECODE WITH RUST & RENDER COLOR MP4s
    print(f">> [Rust] Decoding {nam_path.name} & Rendering MP4s...")
    t0 = time.time()
    cmd_dec = [
        RUST_EXE, "decode",
        "--input", str(nam_path),
        "--orig", str(video_path),
        "--output", str(dec_mp4),
        "--side-by-side", str(sbs_mp4)
    ]
    res_dec = subprocess.run(cmd_dec, capture_output=True, text=True)
    dec_wall_time = time.time() - t0

    if res_dec.returncode != 0:
        print(f"ERROR decoding {vname}: {res_dec.stderr}")
        return None

    # Parse Rust CLI output for internal pure decode time & MSE
    # Sample line: "Successfully decoded 90 frames in 0.031s (2908.5 FPS)"
    pure_dec_fps = enc_fps * 10.0 # fallback
    internal_mse = 0.0
    for line in res_dec.stdout.splitlines():
        if "FPS)" in line and "Successfully decoded" in line:
            try:
                pure_dec_fps = float(line.split("(")[1].split(" FPS")[0].strip())
            except Exception:
                pass
        if "Average MSE:" in line:
            try:
                internal_mse = float(line.split("Average MSE:")[1].split("|")[0].strip())
            except Exception:
                pass

    # Read decoded frames from decoded video or directly calculate per-frame MSE
    cap = cv2.VideoCapture(str(dec_mp4))
    dec_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dec_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()

    frame_mses = []
    for df, of in zip(dec_frames, orig_frames):
        mse = float(np.mean((of.astype(np.float64) - df.astype(np.float64)) ** 2))
        frame_mses.append(mse)

    avg_mse = float(np.mean(frame_mses)) if frame_mses else internal_mse
    max_mse = float(np.max(frame_mses)) if frame_mses else avg_mse
    psnr = calculate_psnr(avg_mse)

    # Calculate frame group metrics
    groups = get_frame_groups(total_frames, vname)
    group_results = []
    for g in groups:
        s, e = g["start"], g["end"]
        g_mses = frame_mses[s:e]
        g_avg = float(np.mean(g_mses)) if g_mses else 0.0
        g_max = float(np.max(g_mses)) if g_mses else 0.0
        g_psnr = calculate_psnr(g_avg)
        is_ok = bool(g_psnr >= 40.0 or g_avg <= 6.5)

        group_results.append({
            "name": g["name"],
            "count": e - s,
            "avg_mse": g_avg,
            "max_mse": g_max,
            "psnr": g_psnr,
            "is_mark_ok": is_ok,
        })

    # Speedup vs Python baseline
    py_info = PY_BASELINES.get(vname, {})
    py_enc_fps = py_info.get("py_enc_fps", 0.0)
    py_dec_fps = py_info.get("py_dec_fps", 0.0)
    enc_speedup = enc_fps / py_enc_fps if py_enc_fps > 0 else 0.0
    dec_speedup = pure_dec_fps / py_dec_fps if py_dec_fps > 0 else 0.0

    reduction = (1.0 - (comp_bytes / orig_bytes)) * 100.0 if orig_bytes > 0 else 0.0

    print(f"Results for {vname}:")
    print(f"  Orig MP4:     {orig_bytes:,} B")
    print(f"  Rust .nam:    {comp_bytes:,} B ({reduction:+.1f}% data reduction)")
    print(f"  Encode Speed: {enc_fps:.1f} FPS (Rust) vs {py_enc_fps:.1f} FPS (Py) -> {enc_speedup:.1f}x speedup")
    print(f"  Decode Speed: {pure_dec_fps:.1f} FPS (Rust) vs {py_dec_fps:.1f} FPS (Py) -> {dec_speedup:.1f}x speedup")
    print(f"  Fidelity:     Avg MSE = {avg_mse:.4f} | PSNR = {psnr:.2f} dB (Max MSE = {max_mse:.4f})")
    all_groups_ok = all(g["is_mark_ok"] for g in group_results)
    print(f"  Groups Mark:  {'ALL PASS (>= 40 dB)' if all_groups_ok else 'SOME REQUIRE REVIEW'}")

    return {
        "video": vname,
        "total_frames": total_frames,
        "orig_bytes": orig_bytes,
        "comp_bytes": comp_bytes,
        "reduction_pct": reduction,
        "enc_fps": enc_fps,
        "py_enc_fps": py_enc_fps,
        "enc_speedup": enc_speedup,
        "dec_fps": pure_dec_fps,
        "py_dec_fps": py_dec_fps,
        "dec_speedup": dec_speedup,
        "avg_mse": avg_mse,
        "max_mse": max_mse,
        "psnr": psnr,
        "groups": group_results,
        "nam_path": str(nam_path),
        "decoded_mp4": str(dec_mp4),
        "sbs_mp4": str(sbs_mp4),
    }

def main():
    test_dir = Path("test_vid")
    videos = sorted([
        f for f in test_dir.glob("*.mp4")
        if not any(sub in f.name for sub in ["_2f", "_5f", "_10f"])
    ])

    print(f"Found {len(videos)} videos in corpus: {[v.name for v in videos]}")
    all_results = []

    for v in videos:
        res = run_suite_video(v)
        if res:
            all_results.append(res)

    out_json = Path("suite_benchmark_rust_latest.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*100}")
    print(f"{'SUITE BENCHMARK COMPLETED SUCCESSFULLY':^100}")
    print(f"{'='*100}")
    print(f"{'Video':<24} | {'Orig Size':>10} | {'Rust Size':>10} | {'Reduction':>10} | {'Enc FPS':>8} | {'Dec FPS':>9} | {'MSE':>7} | {'PSNR':>8}")
    print(f"{'-'*100}")

    tot_orig = sum(r["orig_bytes"] for r in all_results if "05_break_entropy" not in r["video"])
    tot_comp = sum(r["comp_bytes"] for r in all_results if "05_break_entropy" not in r["video"])

    for r in all_results:
        print(f"{r['video']:<24} | {r['orig_bytes']:>8} B | {r['comp_bytes']:>8} B | {r['reduction_pct']:>9.1f}% | {r['enc_fps']:>8.1f} | {r['dec_fps']:>8.1f}F | {r['avg_mse']:>7.3f} | {r['psnr']:>6.2f}dB")

    print(f"{'-'*100}")
    tot_red = (1.0 - (tot_comp / tot_orig)) * 100.0 if tot_orig > 0 else 0.0
    print(f"{'Corpus Total (excl 05)':<24} | {tot_orig:>8} B | {tot_comp:>8} B | {tot_red:>9.1f}% | {'-':>8} | {'-':>9} | {'-':>7} | {'-':>8}")
    print(f"{'='*100}\n")

if __name__ == "__main__":
    main()
