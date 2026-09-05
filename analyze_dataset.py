import pandas as pd
import numpy as np

def main():
    print("=" * 60)
    print(" TRAJ_CDDEC ABLATION DATASET ANALYSIS ")
    print("=" * 60)
    
    # 1. Load Data
    print("Loading datasets...")
    try:
        matrix_df = pd.read_csv('ablation_matrix.csv')
        raw_df = pd.read_csv('raw_encode_results.csv')
        print(f"Loaded {len(matrix_df)} permutations and {len(raw_df)} raw encodes.")
    except Exception as e:
        print(f"Error loading CSVs: {e}")
        return

    print("\n--- PER-VIDEO DCT IMPACT ANALYSIS ---")
    
    # Calculate reduction for each row
    raw_df['reduction'] = np.where(raw_df['orig_bytes'] > 0, 100 - (raw_df['comp_bytes'] / raw_df['orig_bytes'] * 100), 0)
    
    # Parse permutation strings to boolean columns for m9 and m10
    raw_df['m9_dct'] = raw_df['perm'].str.contains(r'\+m9_dct')
    raw_df['m10_dct'] = raw_df['perm'].str.contains(r'\+m10_dct')
    
    videos = raw_df['video'].unique()
    
    print("\nm9_dct Impact by Video:")
    for vid in videos:
        vid_df = raw_df[raw_df['video'] == vid]
        on_mean = vid_df[vid_df['m9_dct'] == True]['reduction'].mean()
        off_mean = vid_df[vid_df['m9_dct'] == False]['reduction'].mean()
        diff = on_mean - off_mean
        print(f"{vid.ljust(30)} : Impact -> {diff:>+7.2f}% (ON: {on_mean:>7.2f}%, OFF: {off_mean:>7.2f}%)")
        
    print("\n------------------------------------------------------------\n")
    print("m10_dct Impact by Video:")
    for vid in videos:
        vid_df = raw_df[raw_df['video'] == vid]
        on_mean = vid_df[vid_df['m10_dct'] == True]['reduction'].mean()
        off_mean = vid_df[vid_df['m10_dct'] == False]['reduction'].mean()
        diff = on_mean - off_mean
        print(f"{vid.ljust(30)} : Impact -> {diff:>+7.2f}% (ON: {on_mean:>7.2f}%, OFF: {off_mean:>7.2f}%)")
        
    print("\n--- ISOLATED DCT IMPACT (CASCADE BYPASS PROOF) ---")
    print("Checking impact of m9_dct on ball.mp4 when all major heuristics (m1, m4, m7, m11, m13) are OFF.")
    
    # Isolate permutations where m1, m4_skip, m4_search, m7, m11, m13 are OFF
    isolated_df = raw_df[
        (raw_df['perm'].str.contains(r'-m1_spatial')) &
        (raw_df['perm'].str.contains(r'-m4_skip')) &
        (raw_df['perm'].str.contains(r'-m4_search')) &
        (raw_df['perm'].str.contains(r'-m7_solid')) &
        (raw_df['perm'].str.contains(r'-m11_cached')) &
        (raw_df['perm'].str.contains(r'-m13_faded')) &
        (raw_df['video'] == 'test_vid/ball.mp4')
    ]
    
    iso_on = isolated_df[isolated_df['m9_dct'] == True]['reduction'].mean()
    iso_off = isolated_df[isolated_df['m9_dct'] == False]['reduction'].mean()
    iso_diff = iso_on - iso_off
    print(f"\nball.mp4 Isolated m9_dct Impact: {iso_diff:>+7.2f}% (ON: {iso_on:>7.2f}%, OFF: {iso_off:>7.2f}%)")
    
    print("\n" + "=" * 60)
    print(" ANALYSIS COMPLETE ")
    print("=" * 60)

if __name__ == "__main__":
    main()
