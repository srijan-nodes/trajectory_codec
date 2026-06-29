import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

# --- SEMANTIC TAGS ---
TAG_COLORS = {
    'FLAT': (50, 50, 50),       # Dark Gray (Adaptive BG candidate)
    'EDGE': (0, 255, 0),        # Green (Text/UI candidate)
    'MOTION': (0, 165, 255),    # Orange (Translational Vector candidate)
    'CHAOS': (255, 0, 255)      # Magenta (Radial Bloom/Dither candidate)
}

class QuadNode:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size
        self.is_leaf = True
        self.tag = None
        self.children = []

def classify_block(block, flow_block):
    """Phase 3: Semantic Routing Classifier"""
    # 1. Edge Energy (Sobel)
    sobelx = cv2.Sobel(block, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(block, cv2.CV_32F, 0, 1, ksize=3)
    edge_energy = np.mean(np.abs(sobelx) + np.abs(sobely))

    # 2. Motion Energy (Farneback)
    mag, _ = cv2.cartToPolar(flow_block[..., 0], flow_block[..., 1])
    motion_var = np.var(mag)
    mean_motion = np.mean(mag)

    if edge_energy > 40.0: return 'EDGE'
    if motion_var > 1.5: return 'CHAOS'
    if mean_motion > 0.5: return 'MOTION'
    return 'FLAT'

def build_quadtree(frame, flow, x, y, size, min_size=8):
    """Phase 2: Recursive Quadtree Decomposition"""
    node = QuadNode(x, y, size)
    
    # Boundary safety
    h, w = frame.shape
    if y >= h or x >= w: return node
    
    block_h = min(size, h - y)
    block_w = min(size, w - x)
    
    flow_block = flow[y:y+block_h, x:x+block_w]
    mag, _ = cv2.cartToPolar(flow_block[..., 0], flow_block[..., 1])
    motion_var = np.var(mag)
    
    # Split Criteria: High motion variance requires finer detail
    if motion_var > 0.5 and size > min_size:
        node.is_leaf = False
        half = size // 2
        node.children.append(build_quadtree(frame, flow, x, y, half, min_size))
        node.children.append(build_quadtree(frame, flow, x + half, y, half, min_size))
        node.children.append(build_quadtree(frame, flow, x, y + half, half, min_size))
        node.children.append(build_quadtree(frame, flow, x + half, y + half, half, min_size))
    else:
        block = frame[y:y+block_h, x:x+block_w]
        node.tag = classify_block(block, flow_block)
        
    return node

def draw_quadtree(vis_frame, node):
    """Recursively draws the bounding boxes and semantic tags"""
    if node.is_leaf:
        if node.tag:
            color = TAG_COLORS[node.tag]
            # Draw semi-transparent fill for non-flat blocks
            if node.tag != 'FLAT':
                overlay = vis_frame.copy()
                cv2.rectangle(overlay, (node.x, node.y), (node.x + node.size, node.y + node.size), color, -1)
                cv2.addWeighted(overlay, 0.3, vis_frame, 0.7, 0, vis_frame)
            
            # Draw borders
            cv2.rectangle(vis_frame, (node.x, node.y), (node.x + node.size, node.y + node.size), color, 1, cv2.LINE_AA)
    else:
        for child in node.children:
            draw_quadtree(vis_frame, child)

def run_v6_vision(video_path):
    print(f"\n🚀 Booting V6 Semantic Engine for {os.path.basename(video_path)}")
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret: return
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    cv2.namedWindow("V6 Quadtree Vision", cv2.WINDOW_NORMAL)
    is_paused = False

    while True:
        if not is_paused:
            ret, curr_frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            # Phase 1: Dense Optical Flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            vis_frame = curr_frame.copy()
            h, w = curr_gray.shape
            
            # Phase 2: Build the tree from massive 64x64 chunks
            MAX_BLOCK = 64
            for y in range(0, h, MAX_BLOCK):
                for x in range(0, w, MAX_BLOCK):
                    root_node = build_quadtree(curr_gray, flow, x, y, MAX_BLOCK, min_size=8)
                    draw_quadtree(vis_frame, root_node)
            
            prev_gray = curr_gray

        cv2.imshow("V6 Quadtree Vision", vis_frame)
        
        key = cv2.waitKey(0 if is_paused else 30) & 0xFF
        if key == ord('q'): break
        elif key == ord(' '): is_paused = not is_paused
        elif key == ord('n') and is_paused: is_paused = False

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    video = filedialog.askopenfilename(title="Select Video", filetypes=[("Video", "*.mp4 *.y4m")])
    if video:
        run_v6_vision(video)