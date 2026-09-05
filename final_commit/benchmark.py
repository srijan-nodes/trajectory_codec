"""
benchmark.py — 1-Click Reproducibility Benchmark (Optimized)
============================================================
Benchmarks encoding speed, compression ratio, decoding speed,
and reconstruction fidelity (MSE / PSNR) for TrajCDDec V5.
"""
import argparse
import os
import time
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from encoder_v6 import encode
from decoder_v6 import decode

def print_header(title):
    print(f"\n{'='*50}\n{title.center(50)}\n{'='*50}")

def benchmark_video(video_path: str):
    if not os.path.exists(video_path):
        print(f"Error: Could not find video file at '{video_path}'")
        return None

    output_nam = video_path.replace('.mp4', '_bench.nam')
    
    # 1. ENCODING
    print_header(f"BENCHMARK: {os.path.basename(video_path)}")
    print("Starting Encoder (V5 Hybrid)...")
    enc_stats = encode(video_path, output_nam)
    
    orig_mb = enc_stats['orig_bytes'] / (1024 * 1024)
    comp_mb = enc_stats['compressed_bytes'] / (1024 * 1024)
    ratio = (enc_stats['compressed_bytes'] / enc_stats['orig_bytes']) * 100
    enc_fps = enc_stats['frames'] / enc_stats['encode_time_s']
    
    print(f"\n[ENCODER RESULTS]")
    print(f"Original Size:   {orig_mb:.2f} MB ({enc_stats['orig_bytes']:,} bytes)")
    print(f"Compressed Size: {comp_mb:.2f} MB ({enc_stats['compressed_bytes']:,} bytes)")
    print(f"Data Reduction:  {100 - ratio:.2f}%")
    print(f"Encoding Speed:  {enc_fps:.2f} FPS")
    
    # 2. DECODING
    print_header("DECODING")
    print(f"Verifying bytestream integrity & fidelity for '{os.path.basename(output_nam)}'...")
    
    try:
        dec_stats = decode(output_nam, original_video=video_path)
        frames_decoded = dec_stats['frame_count']
        dec_time = dec_stats['dec_time']
        avg_mse = dec_stats['avg_mse']
    except Exception as e:
        print(f"\n[DECODER ERROR] Failed to decode bytestream: {str(e)}")
        if os.path.exists(output_nam):
            os.remove(output_nam)
        return None
        
    dec_fps = frames_decoded / dec_time
    psnr = 10 * np.log10((255.0 ** 2) / avg_mse) if avg_mse > 0 else float('inf')
    
    print(f"\n[DECODER RESULTS]")
    print(f"Frames Decoded:  {frames_decoded}")
    print(f"Decoding Speed:  {dec_fps:.2f} FPS")
    print(f"Average MSE:     {avg_mse:.4f}")
    print(f"Recon PSNR:      {psnr:.2f} dB")
    
    # Clean up
    if os.path.exists(output_nam):
        os.remove(output_nam)
        
    print_header("BENCHMARK COMPLETE")
    return {
        "video": os.path.basename(video_path),
        "orig_bytes": enc_stats['orig_bytes'],
        "comp_bytes": enc_stats['compressed_bytes'],
        "ratio_pct": ratio,
        "enc_fps": enc_fps,
        "dec_fps": dec_fps,
        "mse": avg_mse,
        "psnr": psnr
    }

def main():
    parser = argparse.ArgumentParser(description="TrajCDDec: 1-Click Reproducibility Benchmark")
    parser.add_argument("--video", type=str, default=None, help="Path to a single .mp4 video file")
    parser.add_argument("--folder", type=str, default=None, help="Path to folder of .mp4 videos to benchmark")
    args = parser.parse_args()

    if args.video:
        benchmark_video(args.video)
    elif args.folder:
        folder = Path(args.folder)
        videos = sorted([v for v in folder.glob("*.mp4") if not any(sub in v.name for sub in ['_2f', '_5f', '_10f', '05_break_entropy'])])
        if not videos:
            print(f"No .mp4 files found in {args.folder}")
            sys.exit(1)
        
        results = []
        for v in videos:
            res = benchmark_video(str(v))
            if res: results.append(res)
            
        print("\n" + "="*105)
        print(f"{'CORPUS BENCHMARK SUMMARY':^105}")
        print("="*105)
        print(f"{'Video':<24} | {'Orig Size':<10} | {'Comp Size':<10} | {'Reduction':<10} | {'Enc FPS':<8} | {'Dec FPS':<8} | {'MSE':<8} | {'PSNR':<8}")
        print("-" * 105)
        for r in results:
            red_pct = (1.0 - (r['comp_bytes'] / r['orig_bytes'])) * 100.0
            print(f"{r['video']:<24} | {r['orig_bytes']:>8,} B | {r['comp_bytes']:>8,} B | {red_pct:>8.1f}% | {r['enc_fps']:>7.1f}F | {r['dec_fps']:>7.1f}F | {r['mse']:>8.4f} | {r['psnr']:>6.2f}dB")
        print("="*105)
    else:
        # Default to test_vid if available
        default_dir = Path(__file__).parent.parent / "test_vid"
        if default_dir.exists():
            print(f"No video specified. Benchmarking test corpus at {default_dir}...")
            parser.print_help()
        else:
            parser.print_help()

if __name__ == "__main__":
    main()
