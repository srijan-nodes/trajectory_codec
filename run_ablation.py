import os
import glob
import itertools
from encoder_ablation import encode

test_vids = glob.glob('test_vid/*.mp4')
print(f'Found {len(test_vids)} videos for ablation testing (Max 40 frames each)')

toggles = ['motion', 'dct', 'faded_motion', 'dynamic_boxing']
permutations = list(itertools.product([False, True], repeat=4))

results = []

for perm in permutations:
    config = {
        'name': 'Ablation',
        'motion': perm[0],
        'dct': perm[1],
        'faded_motion': perm[2],
        'dynamic_boxing': perm[3]
    }
    
    perm_name = ""
    for idx, t in enumerate(toggles):
        if perm[idx]: perm_name += f"+{t} "
        else: perm_name += f"-{t} "
    
    print(f'\nRunning Permutation: {perm_name}')
    
    total_orig = 0
    total_comp = 0
    total_time = 0
    
    for vid in test_vids:
        # Hide standard output if it gets noisy, but tqdm might still print
        # encode_ablation already has tqdm disabled or we can just let it run
        try:
            enc = encode(vid, 'temp_ablation.nam', config=config)
            total_orig += enc['orig_bytes']
            total_comp += enc['compressed_bytes']
            total_time += enc['encode_time_s']
        except Exception as e:
            print(f'Error on {vid}: {e}')
            
    avg_ratio = (total_comp / total_orig) * 100
    reduction = 100 - avg_ratio
    fps = (len(test_vids) * 40) / total_time if total_time > 0 else 0
    
    results.append({
        'name': perm_name,
        'reduction': reduction,
        'fps': fps
    })
    
    print(f'-> Reduction: {reduction:.2f}% | FPS: {fps:.2f}')

print('\n\n--- FINAL ABLATION RANKING (By Compression Reduction) ---')
results.sort(key=lambda x: x['reduction'], reverse=True)

for i, r in enumerate(results):
    print(f"{i+1}. {r['name'].strip()}")
    print(f"   Compression: {r['reduction']:.2f}% | Speed: {r['fps']:.2f} FPS")

