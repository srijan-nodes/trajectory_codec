import time
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath('final_commit'))
import encoder_ablation as enc
import decoder as dec

# We will use the Rank 1 configuration as our controlled baseline
baseline_config = {
    'name': 'Baseline',
    'dynamic_boxing': False,
    'm1_spatial': True,
    'm4_skip': False,
    'm4_search': True,
    'm6_bg': False,
    'm7_solid': True,
    'm9_dct': False,
    'm10_dct': False,
    'm11_cached': True,
    'm12_palette': True,
    'm13_faded': True,
    'm15_dither': True
}

modes = ['m1_spatial', 'm4_skip', 'm4_search', 'm6_bg', 'm7_solid', 'm9_dct', 'm10_dct', 'm11_cached', 'm12_palette', 'm13_faded', 'm15_dither']
trials = 3

def run_trial(config_dict, out_nam):
    start_enc = time.time()
    enc.encode('test_vid/ball.mp4', out_nam, config_dict)
    enc_time = time.time() - start_enc
    
    start_dec = time.time()
    try:
        res_dec = dec.decode(out_nam, None)
        dec_time = res_dec['dec_time']
        frames = res_dec.get('frame_count', 101)
        dec_fps = frames / max(0.001, dec_time)
    except Exception as e:
        print(f"Decode failed: {e}")
        dec_fps = None
        frames = 101
        
    if os.path.exists(out_nam):
        os.remove(out_nam)
        
    enc_fps = frames / max(0.001, enc_time)
    return enc_fps, dec_fps

if __name__ == '__main__':
    print("Running controlled sequential benchmarks (no thread contention noise)...")
    results = {}
    
    out_nam = "temp_benchmark.nam"
    
    # 1. Benchmark Baseline
    print(f"Benchmarking Baseline...")
    b_enc_fps, b_dec_fps = [], []
    for _ in range(trials):
        e, d = run_trial(baseline_config, out_nam)
        b_enc_fps.append(e)
        b_dec_fps.append(d)
        
    results['Baseline'] = {
        'enc': b_enc_fps,
        'dec': b_dec_fps
    }
    
    # 2. Benchmark Marginal Toggles
    for m in modes:
        print(f"Benchmarking toggle for {m}...")
        test_config = baseline_config.copy()
        # Toggle the mode
        test_config[m] = not test_config[m]
        
        t_enc_fps, t_dec_fps = [], []
        for _ in range(trials):
            e, d = run_trial(test_config, out_nam)
            if e is not None: t_enc_fps.append(e)
            if d is not None: t_dec_fps.append(d)
            
        results[m] = {
            'enc': t_enc_fps,
            'dec': t_dec_fps,
            'toggled_to': test_config[m]
        }
        
    print("\n| Mode | Baseline State | Toggled State | Enc FPS (Mean ± Std) | Dec FPS (Mean ± Std) | Enc Delta | Dec Delta |")
    print("|---|---|---|---|---|---|---|")
    
    b_enc_mean, b_enc_std = np.mean(results['Baseline']['enc']), np.std(results['Baseline']['enc'])
    b_dec_mean, b_dec_std = np.mean(results['Baseline']['dec']), np.std(results['Baseline']['dec'])
    print(f"| `BASELINE` | - | - | {b_enc_mean:.2f} ± {b_enc_std:.2f} | {b_dec_mean:.2f} ± {b_dec_std:.2f} | - | - |")
    
    for m in modes:
        res = results[m]
        e_mean = np.mean(res['enc']) if res['enc'] else 0.0
        e_std = np.std(res['enc']) if res['enc'] else 0.0
        d_mean = np.mean(res['dec']) if res['dec'] else 0.0
        d_std = np.std(res['dec']) if res['dec'] else 0.0
        
        enc_delta = e_mean - b_enc_mean
        dec_delta = d_mean - b_dec_mean
        
        if baseline_config[m]:
            true_enc_delta = b_enc_mean - e_mean
            true_dec_delta = b_dec_mean - d_mean
        else:
            true_enc_delta = e_mean - b_enc_mean
            true_dec_delta = d_mean - b_dec_mean
            
        d_mean_str = f"{d_mean:.2f} ± {d_std:.2f}" if res['dec'] else "FAIL"
        d_delta_str = f"{true_dec_delta:+.2f}" if res['dec'] else "FAIL"
        
        print(f"| `{m}` | {'ON' if baseline_config[m] else 'OFF'} | {'OFF' if baseline_config[m] else 'ON'} | {e_mean:.2f} ± {e_std:.2f} | {d_mean_str} | {true_enc_delta:+.2f} | {d_delta_str} |")
