import itertools
import multiprocessing as mp
import time
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath('final_commit'))
import encoder_ablation as enc
import decoder as dec

def worker(config_dict):
    # Unique temp files per process to avoid race conditions
    pid = os.getpid()
    out_nam = f"temp_{pid}.nam"
    
    start_enc = time.time()
    res_enc = enc.encode('test_vid/ball.mp4', out_nam, config_dict)
    enc_time = time.time() - start_enc
    
    start_dec = time.time()
    try:
        res_dec = dec.decode(out_nam, None)
        dec_time = time.time() - start_dec
        frames = res_dec.get('frame_count', 1)
        dec_fps = frames / max(0.001, res_dec['dec_time'])
    except Exception as e:
        dec_fps = 0.0
        frames = 1
    
    if os.path.exists(out_nam):
        os.remove(out_nam)
        
    return {
        'config': config_dict,
        'enc_fps': frames / max(0.001, enc_time),
        'dec_fps': dec_fps
    }

if __name__ == '__main__':
    modes = ['m1_spatial', 'm4_skip', 'm4_search', 'm6_bg', 'm7_solid', 'm9_dct', 'm10_dct', 'm11_cached', 'm12_palette', 'm13_faded', 'm15_dither']
    
    all_configs = []
    for perm in itertools.product([True, False], repeat=len(modes)):
        c = {"name": "Test", "dynamic_boxing": True}
        for i, m in enumerate(modes):
            c[m] = perm[i]
        all_configs.append(c)
        
    print(f"Running {len(all_configs)} permutations on ball.mp4...")
    start_time = time.time()
    
    results = []
    with mp.Pool(mp.cpu_count()) as pool:
        for i, res in enumerate(pool.imap_unordered(worker, all_configs)):
            results.append(res)
            if (i+1) % 100 == 0:
                print(f"Done {i+1}/2048 in {time.time()-start_time:.1f}s")
                
    # Now analyze the marginal effects
    print("\n| Mode | Avg Enc FPS (Enabled) | Avg Enc FPS (Disabled) | Avg Dec FPS (Enabled) | Avg Dec FPS (Disabled) |")
    print("|---|---|---|---|---|")
    
    for m in modes:
        enc_en = np.mean([r['enc_fps'] for r in results if r['config'][m] == True])
        enc_dis = np.mean([r['enc_fps'] for r in results if r['config'][m] == False])
        dec_en = np.mean([r['dec_fps'] for r in results if r['config'][m] == True])
        dec_dis = np.mean([r['dec_fps'] for r in results if r['config'][m] == False])
        
        print(f"| `{m}` | {enc_en:.2f} | {enc_dis:.2f} | {dec_en:.2f} | {dec_dis:.2f} |")
