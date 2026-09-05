import sys
import cv2
import numpy as np
import os
sys.path.insert(0, os.path.abspath('final_commit'))
from decoder import decode

def generate_overlay(video_path, encoded_path, out_path):
    dec = decode(encoded_path, original_video=video_path, return_mode_map=True)
    frames = dec["frames"]
    mode_maps = dec.get("mode_maps")
    if not mode_maps:
        print("No mode maps returned")
        return
        
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), isColor=True)
    
    BLOCK_SIZE = 16
    
    # BGR Mode color mapping
    colors = {
        255: (0, 255, 0),      # Green (Skip)
        7:   (0, 0, 255),      # Red (Solid/Flat)
        12:  (0, 255, 255),    # Yellow (Color Palette Mode 12)
        15:  (0, 128, 255),    # Orange (Fast Forced Dither)
        9:   (255, 0, 255),    # Magenta (DCT Low)
        10:  (255, 0, 128),    # Deep Pink (DCT Mid)
        4:   (255, 255, 0),    # Cyan (Motion)
        11:  (255, 128, 0),    # Light Blue (Cached Motion)
        13:  (128, 255, 0),    # Lime (Fade)
        1:   (128, 128, 128),  # Gray (Raw delta)
        2:   (255, 255, 255),  # White (Raw uncompressed)
    }
    
    for i, dec_f in enumerate(frames):
        ret, orig_f = cap.read()
        if not ret: break
        
        orig_bgr = orig_f[:h, :w].copy()
        
        orig_ycrcb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2YCrCb)
        dec_ycrcb = orig_ycrcb.copy()
        dec_ycrcb[:,:,0] = np.clip(dec_f.squeeze(), 0, 255).astype(np.uint8)
        dec_bgr = cv2.cvtColor(dec_ycrcb, cv2.COLOR_YCrCb2BGR)
        
        overlay = dec_bgr.copy()
        m_map = mode_maps[i]
        
        for y in range(m_map.shape[0]):
            for x in range(m_map.shape[1]):
                m = m_map[y, x]
                c = colors.get(m, (50, 50, 50))
                overlay[y*BLOCK_SIZE:(y+1)*BLOCK_SIZE, x*BLOCK_SIZE:(x+1)*BLOCK_SIZE] = c
                
        # Blend overlay (70% original, 30% color map)
        blended = cv2.addWeighted(dec_bgr, 0.7, overlay, 0.3, 0)
        vw.write(blended)
        
    vw.release()
    cap.release()
    print(f"Overlay video saved to {out_path}")

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python diagnostic_map.py <orig_vid> <encoded.nam> <out.mp4>")
        sys.exit(1)
    generate_overlay(sys.argv[1], sys.argv[2], sys.argv[3])
