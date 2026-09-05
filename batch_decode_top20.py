import time
import cv2
import sys
import os
import zstandard as zstd
import struct
import numpy as np
sys.path.insert(0, os.path.abspath('final_commit'))
import encoder_ablation as enc
import decoder as dec

configs = [
    '+m1_spatial -m4_skip +m4_search -m6_bg +m7_solid -m9_dct -m10_dct +m11_cached +m12_palette +m13_faded +m15_dither',
    '+m1_spatial -m4_skip +m4_search -m6_bg +m7_solid -m9_dct +m10_dct +m11_cached +m12_palette +m13_faded +m15_dither',
    '+m1_spatial +m4_skip +m4_search -m6_bg +m7_solid -m9_dct -m10_dct +m11_cached +m12_palette +m13_faded +m15_dither',
    '+m1_spatial +m4_skip +m4_search -m6_bg +m7_solid -m9_dct +m10_dct +m11_cached +m12_palette +m13_faded +m15_dither',
    '+m1_spatial -m4_skip -m4_search -m6_bg +m7_solid -m9_dct -m10_dct +m11_cached +m12_palette +m13_faded +m15_dither',
    '+m1_spatial -m4_skip -m4_search -m6_bg +m7_solid -m9_dct +m10_dct +m11_cached +m12_palette +m13_faded +m15_dither',
    '+m1_spatial +m4_skip -m4_search -m6_bg +m7_solid -m9_dct -m10_dct +m11_cached +m12_palette +m13_faded +m15_dither',
    '+m1_spatial +m4_skip -m4_search -m6_bg +m7_solid -m9_dct +m10_dct +m11_cached +m12_palette +m13_faded +m15_dither',
    '+m1_spatial -m4_skip +m4_search -m6_bg +m7_solid -m9_dct -m10_dct +m11_cached -m12_palette +m13_faded +m15_dither',
    '+m1_spatial -m4_skip +m4_search -m6_bg +m7_solid -m9_dct +m10_dct +m11_cached -m12_palette +m13_faded +m15_dither',
    '+m1_spatial +m4_skip +m4_search -m6_bg +m7_solid -m9_dct -m10_dct +m11_cached -m12_palette +m13_faded +m15_dither',
    '+m1_spatial +m4_skip +m4_search -m6_bg +m7_solid -m9_dct +m10_dct +m11_cached -m12_palette +m13_faded +m15_dither',
    '+m1_spatial -m4_skip +m4_search -m6_bg +m7_solid -m9_dct -m10_dct +m11_cached -m12_palette +m13_faded -m15_dither',
    '+m1_spatial -m4_skip +m4_search -m6_bg +m7_solid -m9_dct +m10_dct +m11_cached -m12_palette +m13_faded -m15_dither',
    '+m1_spatial +m4_skip +m4_search -m6_bg +m7_solid -m9_dct -m10_dct +m11_cached -m12_palette +m13_faded -m15_dither',
    '+m1_spatial +m4_skip +m4_search -m6_bg +m7_solid -m9_dct +m10_dct +m11_cached -m12_palette +m13_faded -m15_dither',
    '+m1_spatial -m4_skip +m4_search -m6_bg +m7_solid -m9_dct -m10_dct +m11_cached +m12_palette +m13_faded -m15_dither',
    '+m1_spatial -m4_skip +m4_search -m6_bg +m7_solid -m9_dct +m10_dct +m11_cached +m12_palette +m13_faded -m15_dither',
    '+m1_spatial +m4_skip +m4_search -m6_bg +m7_solid -m9_dct -m10_dct +m11_cached +m12_palette +m13_faded -m15_dither',
    '+m1_spatial +m4_skip +m4_search -m6_bg +m7_solid -m9_dct +m10_dct +m11_cached +m12_palette +m13_faded -m15_dither'
]

vids = ["test_vid/01_ideal_motion.mp4", "test_vid/ball.mp4"]

print("| Rank | Permutation (Active Modes) | Avg Grayscale MSE | Decoder FPS |")
print("| :--- | :--- | :--- | :--- |")

for i, conf_str in enumerate(configs):
    conf_dict = {"name": "Test"}
    for token in conf_str.split():
        sign, name = token[0], token[1:]
        conf_dict[name] = (sign == '+')
    
    total_mse = 0.0
    total_dec_fps = 0.0
    
    for vid in vids:
        out_nam = vid.replace('.mp4', '_tmp.nam')
        out_avi = vid.replace('.mp4', '_tmp.avi')
        
        # encode
        enc.encode(vid, out_nam, conf_dict)
        
        # decode
        res = dec.decode(out_nam, vid)
        total_mse += res['avg_mse']
        total_dec_fps += (res['frame_count'] / res['dec_time'])
        
    avg_mse = total_mse / len(vids)
    avg_fps = total_dec_fps / len(vids)
    print(f"| {i+1} | `{conf_str}` | {avg_mse:.2f} | {avg_fps:.2f} |")
