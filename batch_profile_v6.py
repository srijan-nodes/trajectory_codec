import cv2
import numpy as np
import os
import time
from tqdm import tqdm

# Import the vision engine from your V6 file
from lab_v6 import build_quadtree

def traverse_and_count(node, stats):
    """Recursively calculates the pixel area assigned to each semantic tag."""
    if node.is_leaf:
        area = node.size * node.size
        if node.tag:
            stats[node.tag] += area
        stats['total_area'] += area
        stats['leaf_count'] += 1
    else:
        for child in node.children:
            traverse_and_count(child, stats)

def run_v6_batch_profiler(media_folder="test_vid"):
    videos = sorted([f for f in os.listdir(media_folder) if f.endswith(('.mp4', '.y4m'))])
    if not videos:
        print(f"No videos found in {media_folder}/")
        return

    print("🚀 STARTING V6 SEMANTIC BATCH PROFILER...")
    print("=" * 105)
    print(f"{'Video':<20} | {'Frames':>6} | {'FPS':>6} | {'Nodes/Fr':>8} | {'FLAT %':>8} | {'EDGE %':>8} | {'MOTION %':>8} | {'CHAOS %':>8}")
    print("-" * 105)

    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'

    overall_stats = {'FLAT': 0, 'EDGE': 0, 'MOTION': 0, 'CHAOS': 0, 'total_area': 0}

    for video in videos:
        video_path = os.path.join(media_folder, video)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        ret, prev_frame = cap.read()
        if not ret: continue
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        h, w = prev_gray.shape
        
        vid_stats = {'FLAT': 0, 'EDGE': 0, 'MOTION': 0, 'CHAOS': 0, 'total_area': 0, 'leaf_count': 0}
        
        start_time = time.time()
        
        pbar = tqdm(total=total_frames, desc=f"Profiling {video[:15]}", unit="f", leave=False)
        frame_idx = 1
        
        while True:
            ret, curr_frame = cap.read()
            if not ret: break
            
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            # Optical Flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Quadtree Build
            MAX_BLOCK = 64
            for y in range(0, h, MAX_BLOCK):
                for x in range(0, w, MAX_BLOCK):
                    root_node = build_quadtree(curr_gray, flow, x, y, MAX_BLOCK, min_size=8)
                    traverse_and_count(root_node, vid_stats)
            
            prev_gray = curr_gray
            frame_idx += 1
            pbar.update(1)
            
        pbar.close()
        cap.release()
        
        time_taken = time.time() - start_time
        fps = frame_idx / max(time_taken, 0.001)
        
        # Calculate percentages based on pixel area, not node count
        area = vid_stats['total_area']
        if area == 0: area = 1
        
        p_flat = (vid_stats['FLAT'] / area) * 100
        p_edge = (vid_stats['EDGE'] / area) * 100
        p_motion = (vid_stats['MOTION'] / area) * 100
        p_chaos = (vid_stats['CHAOS'] / area) * 100
        avg_nodes = vid_stats['leaf_count'] // max(frame_idx, 1)

        # Accumulate for overall
        for k in ['FLAT', 'EDGE', 'MOTION', 'CHAOS', 'total_area']:
            overall_stats[k] += vid_stats[k]

        print(f"{video[:20]:<20} | {frame_idx:>6} | {fps:>6.1f} | {avg_nodes:>8,d} | "
              f"{GREEN}{p_flat:>7.1f}%{RESET} | {CYAN}{p_edge:>7.1f}%{RESET} | "
              f"{YELLOW}{p_motion:>7.1f}%{RESET} | {MAGENTA}{p_chaos:>7.1f}%{RESET}")

    print("=" * 105)
    area = overall_stats['total_area']
    if area > 0:
        p_flat = (overall_stats['FLAT'] / area) * 100
        p_edge = (overall_stats['EDGE'] / area) * 100
        p_motion = (overall_stats['MOTION'] / area) * 100
        p_chaos = (overall_stats['CHAOS'] / area) * 100
        print(f"{'CORPUS AVERAGE':<20} | {'-':>6} | {'-':>6} | {'-':>8} | "
              f"{GREEN}{p_flat:>7.1f}%{RESET} | {CYAN}{p_edge:>7.1f}%{RESET} | "
              f"{YELLOW}{p_motion:>7.1f}%{RESET} | {MAGENTA}{p_chaos:>7.1f}%{RESET}")
    print("=" * 105)

if __name__ == "__main__":
    run_v6_batch_profiler()