import cv2
import numpy as np
import struct
import zstandard as zstd
import os
from tqdm import tqdm

# --- INLINED BITSTREAM PACKER ---
MODE_TO_BITS = {
    6: 0b00,   # V5 Mode 6: Skip / Background
    12: 0b01,  # V5 Mode 12: Precise Palette
    1: 0b10,   # V5 Mode 1: Spatial Delta
    15: 0b11   # V5 Mode 15: Chaos / Dither
}

class BitWriter:
    def __init__(self):
        self.data = bytearray()
        self.accumulator = 0
        self.bit_count = 0

    def write(self, value, num_bits):
        for i in range(num_bits - 1, -1, -1):
            bit = (value >> i) & 1
            self.accumulator = (self.accumulator << 1) | bit
            self.bit_count += 1
            if self.bit_count == 8:
                self.data.append(self.accumulator)
                self.accumulator = 0
                self.bit_count = 0

    def flush(self):
        if self.bit_count > 0:
            self.accumulator <<= (8 - self.bit_count)
            self.data.append(self.accumulator)
            self.accumulator = 0
            self.bit_count = 0
        return bytes(self.data)

def serialize_quadtree(node, writer):
    if not node.is_leaf:
        writer.write(1, 1)  # SPLIT
        for child in node.children:
            serialize_quadtree(child, writer)
    else:
        writer.write(0, 1)  # LEAF
        tag = node.tag if node.tag is not None else 6  # Defensive fallback
        writer.write(MODE_TO_BITS[tag], 2)  # V5 MODE ID

# --- V6 SEMANTIC ENGINE ---
class QuadNode:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size
        self.is_leaf = True
        self.tag = None
        self.children = []

def classify_block(block, flow_block):
    if block is None or block.size == 0: return 6  # Mode 6: Skip
    
    sobelx = cv2.Sobel(block, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(block, cv2.CV_32F, 0, 1, ksize=3)
    edge_energy = np.mean(np.abs(sobelx) + np.abs(sobely))
    hist_var = np.var(block)

    mag, _ = cv2.cartToPolar(flow_block[..., 0], flow_block[..., 1])
    motion_var = np.var(mag)
    mean_motion = np.mean(mag)

    if motion_var > 1.2: return 15  # CHAOS
    if mean_motion > 0.4: return 1  # MOTION
    if edge_energy > 35.0 and hist_var > 250.0: return 12  # EDGE (Threshold adjusted to 250 for dark UI)
    
    return 6  # FLAT

def build_quadtree(frame, flow, x, y, size, min_size=8):
    node = QuadNode(x, y, size)
    h, w = frame.shape
    
    # Boundary guard: Force FLAT if out of bounds
    if y >= h or x >= w: 
        node.tag = 6
        return node
    
    block_h = min(size, h - y)
    block_w = min(size, w - x)
    if block_h <= 0 or block_w <= 0:
        node.tag = 6
        return node

    if size <= min_size:
        node.tag = classify_block(frame[y:y+block_h, x:x+block_w], flow[y:y+block_h, x:x+block_w])
        if node.tag is None: node.tag = 6
        return node

    block = frame[y:y+block_h, x:x+block_w]
    flow_block = flow[y:y+block_h, x:x+block_w]
    
    mag, _ = cv2.cartToPolar(flow_block[..., 0], flow_block[..., 1])
    motion_var = np.var(mag)
    
    sobelx = cv2.Sobel(block, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(block, cv2.CV_32F, 0, 1, ksize=3)
    edge_energy = np.mean(np.abs(sobelx) + np.abs(sobely))
    
    if (motion_var > 0.4 or edge_energy > 35.0):
        node.is_leaf = False
        half = size // 2
        node.children.append(build_quadtree(frame, flow, x, y, half, min_size))
        node.children.append(build_quadtree(frame, flow, x + half, y, half, min_size))
        node.children.append(build_quadtree(frame, flow, x, y + half, half, min_size))
        node.children.append(build_quadtree(frame, flow, x + half, y + half, half, min_size))
    else:
        node.tag = classify_block(block, flow_block)
        
    if node.is_leaf and node.tag is None:
        node.tag = 6
        
    return node

def apply_semantic_rdo(node, curr_gray, prev_gray, flow):
    if not node.is_leaf:
        for child in node.children: 
            apply_semantic_rdo(child, curr_gray, prev_gray, flow)
        return
        
    y, x, s = node.y, node.x, node.size
    h, w = curr_gray.shape
    block_h = min(s, h - y)
    block_w = min(s, w - x)
    if block_h <= 0 or block_w <= 0: return
    
    block = curr_gray[y:y+block_h, x:x+block_w]
    recon_flat = prev_gray[y:y+block_h, x:x+block_w]
    
    mse_flat = np.mean((block.astype(np.float32) - recon_flat.astype(np.float32))**2)
    if node.tag == 6 and mse_flat > 64.0:
        node.tag = 1

def encode_v6(video_path, output_path, gop_size=60):
    print(f"\n🚀 STARTING HARDENED V6 ENCODE: {os.path.basename(video_path)}")
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret: return
    
    h, w, _ = prev_frame.shape
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cctx = zstd.ZstdCompressor(level=3)
    
    with open(output_path, "wb") as f:
        f.write(b'V6SOBVC')
        f.write(struct.pack("<III", h, w, total_frames))
        
        compressed_kf = cctx.compress(prev_gray.tobytes())
        f.write(struct.pack("<I", len(compressed_kf)))
        f.write(compressed_kf)
        
        pbar = tqdm(total=total_frames - 1, desc="Encoding", unit="f")
        frame_idx = 1
        
        while True:
            ret, curr_frame = cap.read()
            if not ret: break
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            if frame_idx % gop_size == 0:
                comp_kf = cctx.compress(curr_gray.tobytes())
                f.write(struct.pack("<BII", 0, len(comp_kf), 0))
                f.write(comp_kf)
                prev_gray = curr_gray
                frame_idx += 1
                pbar.update(1)
                continue

            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            MAX_BLOCK = 64
            bit_writer = BitWriter()
            payloads = bytearray()
            roots = []
            
            for y in range(0, h, MAX_BLOCK):
                for x in range(0, w, MAX_BLOCK):
                    root_node = build_quadtree(curr_gray, flow, x, y, MAX_BLOCK, min_size=8)
                    apply_semantic_rdo(root_node, curr_gray, prev_gray, flow)
                    serialize_quadtree(root_node, bit_writer)
                    roots.append(root_node)
            
            def process_payloads(node):
                if not node.is_leaf:
                    for child in node.children: process_payloads(child)
                    return
                
                ny, nx, ns = node.y, node.x, node.size
                nb_h = min(ns, h - ny)
                nb_w = min(ns, w - nx)
                if nb_h <= 0 or nb_w <= 0: return

                n_block = curr_gray[ny:ny+nb_h, nx:nx+nb_w]
                
                if node.tag == 6:  # Skip
                    pass
                    
                elif node.tag == 1:  # Spatial Delta
                    flow_block = flow[ny:ny+nb_h, nx:nx+nb_w]
                    dy = int(np.clip(np.median(flow_block[..., 1]), -128, 127))
                    dx = int(np.clip(np.median(flow_block[..., 0]), -128, 127))
                    
                    src_y = np.clip(ny - dy, 0, h - nb_h)
                    src_x = np.clip(nx - dx, 0, w - nb_w)
                    pred = prev_gray[src_y:src_y+nb_h, src_x:src_x+nb_w]
                    
                    residual = n_block.astype(np.int16) - pred.astype(np.int16)
                    residual[np.abs(residual) < 6] = 0
                    quant_res = np.clip(residual // 2, -128, 127).astype(np.int8)
                    
                    payloads.extend(struct.pack("<bb", dy, dx))
                    payloads.extend(quant_res.tobytes())

                elif node.tag == 12:  # Precise Palette
                    min_v = int(np.min(n_block))
                    max_v = int(np.max(n_block))
                    payloads.extend(struct.pack("<BB", min_v, max_v))
                    
                    if min_v != max_v:
                        idx = np.round((n_block.astype(np.float32) - min_v) / (max_v - min_v) * 3).astype(np.uint8).flatten()
                        
                        pad_len = (4 - (len(idx) % 4)) % 4
                        if pad_len > 0:
                            idx = np.append(idx, np.zeros(pad_len, dtype=np.uint8))
                            
                        packed = (idx[0::4] << 6) | (idx[1::4] << 4) | (idx[2::4] << 2) | idx[3::4]
                        payloads.extend(packed.tobytes())

                elif node.tag == 15:  # Chaos / Bloom
                    flow_block = flow[ny:ny+nb_h, nx:nx+nb_w]
                    dy = int(np.clip(np.median(flow_block[..., 1]), -128, 127))
                    dx = int(np.clip(np.median(flow_block[..., 0]), -128, 127))
                    payloads.extend(struct.pack("<bb", dy, dx))
            
            for root_node in roots:
                process_payloads(root_node)
                
            tree_bits = bit_writer.flush()
            comp_payloads = cctx.compress(payloads)
            
            f.write(struct.pack("<BII", 1, len(tree_bits), len(comp_payloads)))
            f.write(tree_bits)
            f.write(comp_payloads)
            
            prev_gray = curr_gray
            frame_idx += 1
            pbar.update(1)
            
        pbar.close()
        cap.release()

if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    video = filedialog.askopenfilename(title="Select Video", filetypes=[("Video", "*.mp4 *.y4m")])
    if video:
        encode_v6(video, "test_v6.nam6")