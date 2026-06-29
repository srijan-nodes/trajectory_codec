import cv2
import numpy as np
import os

def create_video(filename, frame_generator, frames=90, size=(640, 480)):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 30.0, size)
    for i in range(frames):
        out.write(frame_generator(i, size[1], size[0]))
    out.release()
    print(f"Generated {filename}")

# --- THE IDEALS ---
def ideal_motion(f, h, w):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(img, (50 + f * 5, h // 2), 40, (0, 255, 0), -1)
    return img

def ideal_cache(f, h, w):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    if (f // 15) % 2 == 0:
        cv2.rectangle(img, (200, 200), (440, 280), (0, 0, 255), -1)
        cv2.putText(img, "WARNING", (240, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return img

def ideal_palette(f, h, w):
    img = np.full((h, w, 3), 50, dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (255, 0, 0), -1)
    cv2.rectangle(img, (250, 200), (450, 400), (0, 255, 255), -1)
    return img

def ideal_baseline(f, h, w):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    shift = int(127 * (1 + np.sin(f / 10.0)))
    img[:, :] = (shift, shift // 2, 100)
    return img

# --- THE UNIDEALS (BREAKER TESTS) ---
def break_entropy(f, h, w):
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

def break_jitter(f, h, w):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    jitter_x, jitter_y = np.random.randint(-3, 4, 2)
    cv2.rectangle(img, (200 + jitter_x, 200 + jitter_y), (400 - jitter_x, 400 - jitter_y), (255, 100, 100), -1)
    return img

def break_fade(f, h, w):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    intensity = max(10, 255 - (f * 2))
    cv2.circle(img, (50 + f * 4, h // 3), 50, (intensity, intensity, intensity), -1)
    return img

def break_moire(f, h, w):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(0, w, max(1, 4 - (f // 30))): # Lines get tighter over time
        cv2.line(img, (i, 0), (i, h), (255, 255, 255), 1)
    return img
# Add this above your __main__ block
def scaling_ui(f, h, w):
    img = np.full((h, w, 3), 30, dtype=np.uint8)
    
    # 1. An expanding window (scales in 2 directions)
    box_w = 100 + (f * 4)
    box_h = 50 + (f * 2)
    cv2.rectangle(img, (100, 100), (min(w-50, 100+box_w), min(h-50, 100+box_h)), (200, 100, 50), -1)
    
    # 2. A loading bar (scales in 1 direction)
    bar_w = min(400, f * 10)
    cv2.rectangle(img, (100, 300), (100 + bar_w, 320), (0, 255, 100), -1)
    return img

# Add this line inside your __main__ block:
# create_video("test_vid/07_scaling_ui.mp4", scaling_ui)
if __name__ == "__main__":
    print("Synthesizing Video Test Suite...")
    create_video("01_ideal_motion.mp4", ideal_motion)
    create_video("02_ideal_cache.mp4", ideal_cache)
    create_video("03_ideal_palette.mp4", ideal_palette)
    create_video("04_ideal_baseline.mp4", ideal_baseline)
    create_video("05_break_entropy.mp4", break_entropy)
    create_video("06_break_jitter.mp4", break_jitter)
    create_video("07_break_fade.mp4", break_fade)
    create_video("08_break_moire.mp4", break_moire)
    create_video("09_scaling_ui.mp4", scaling_ui)
    print("Synthetic Suite Complete.")