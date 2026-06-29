import cv2
import numpy as np
import struct
import zstandard as zstd

# Bitstream mapped directly to V5 Modes
BITS_TO_MODE = {
    0b00: 6,   # FLAT / Adaptive Background
    0b01: 12,  # EDGE / Precise Palette
    0b10: 1,   # MOTION / Spatial Delta
    0b11: 15   # CHAOS / Forced Dither Bloom
}

class QuadNode:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size
        self.is_leaf = True
        self.tag = None
        self.children = []

class BitReader:
    def __init__(self, data_bytes):
        self.data = data_bytes
        self.byte_pos = 0
        self.bit_pos = 7

    def read(self, num_bits):
        value = 0
        for _ in range(num_bits):
            if self.byte_pos >= len(self.data):
                raise EOFError("BitReader out of bounds")
            bit = (self.data[self.byte_pos] >> self.bit_pos) & 1
            value = (value << 1) | bit
            self.bit_pos -= 1
            if self.bit_pos < 0:
                self.bit_pos = 7
                self.byte_pos += 1
        return value
def deserialize_quadtree(reader, x, y, size, min_size=8):

    node = QuadNode(x, y, size)

    try:
        is_split = reader.read(1)
    except EOFError:
        node.tag = 6
        return node

    if is_split == 1 and size > min_size:

        node.is_leaf = False

        half = size // 2

        node.children = [

            deserialize_quadtree(
                reader,
                x,
                y,
                half,
                min_size
            ),

            deserialize_quadtree(
                reader,
                x + half,
                y,
                half,
                min_size
            ),

            deserialize_quadtree(
                reader,
                x,
                y + half,
                half,
                min_size
            ),

            deserialize_quadtree(
                reader,
                x + half,
                y + half,
                half,
                min_size
            )
        ]

    else:

        try:
            tag_val = reader.read(2)
        except EOFError:
            tag_val = 0

        node.tag = BITS_TO_MODE.get(tag_val, 6)

    return node
def decode_v6(encoded_path):
    with open(encoded_path, "rb") as f:
        header = f.read(7)
        if header != b'V6SOBVC':
            raise ValueError(f"Invalid file header: {header}")
            
        h, w, total_frames = struct.unpack("<III", f.read(12))
        
        kf_len = struct.unpack("<I", f.read(4))[0]
        dctx = zstd.ZstdDecompressor()
        prev_gray = np.frombuffer(dctx.decompress(f.read(kf_len)), dtype=np.uint8).reshape((h, w))
        
        frames = [prev_gray.copy()]
        
        for _ in range(total_frames - 1):
            type_byte = f.read(1)
            if not type_byte: break
            
            # GOP Reset: Properly intercepts the Type 0 I-Frame
            type_val = type_byte[0]
            if type_val == 0:
                comp_kf_len = struct.unpack("<I", f.read(4))[0]
                _ = f.read(4) # Consume the 4 dummy alignment bytes packed by the encoder
                curr_gray = np.frombuffer(dctx.decompress(f.read(comp_kf_len)), dtype=np.uint8).reshape((h, w))
                frames.append(curr_gray)
                prev_gray = curr_gray
                continue
            
            len_tree, len_payloads = struct.unpack("<II", f.read(8))
            tree_bits = f.read(len_tree)
            comp_payloads = f.read(len_payloads)
            
            payloads = dctx.decompress(comp_payloads)
            reader = BitReader(tree_bits)
            
            curr_gray = np.zeros_like(prev_gray)
            pay_idx = 0
            
            def process_node(node):
                nonlocal pay_idx
                if not node.is_leaf:
                    for child in node.children: process_node(child)
                    return
                    
                y, x, s = node.y, node.x, node.size
                block_h = min(s, h - y)
                block_w = min(s, w - x)
                if block_h <= 0 or block_w <= 0: return
                
                # V5 Mode 6: Skip / Adaptive Background
                if node.tag == 6:
                    curr_gray[y:y+block_h, x:x+block_w] = prev_gray[y:y+block_h, x:x+block_w]
                    
                # V5 Mode 1: Spatial Delta
                elif node.tag == 1:
                    if pay_idx + 2 > len(payloads): raise ValueError("MOTION Vector overrun")
                    dy, dx = struct.unpack_from("<bb", payloads, pay_idx)
                    pay_idx += 2
                    src_y = np.clip(y - dy, 0, h - block_h)
                    src_x = np.clip(x - dx, 0, w - block_w)
                    pred = prev_gray[src_y:src_y+block_h, src_x:src_x+block_w]
                    
                    res_size = block_h * block_w
                    if pay_idx + res_size > len(payloads): raise ValueError("MOTION Residual overrun")
                    quant_res = np.frombuffer(payloads[pay_idx:pay_idx+res_size], dtype=np.int8).reshape((block_h, block_w))
                    pay_idx += res_size
                    
                    curr_gray[y:y+block_h, x:x+block_w] = np.clip(pred.astype(np.int16) + (quant_res.astype(np.int16) * 2), 0, 255).astype(np.uint8)
                    
                # V5 Mode 12: Precise Palette
                elif node.tag == 12:
                    if pay_idx + 2 > len(payloads): raise ValueError("EDGE Header overrun")
                    min_v, max_v = struct.unpack_from("<BB", payloads, pay_idx)
                    pay_idx += 2
                    
                    if min_v == max_v:
                        curr_gray[y:y+block_h, x:x+block_w] = min_v
                    else:
                        N = block_w * block_h
                        packed_len = (N + 3) // 4
                        if pay_idx + packed_len > len(payloads): raise ValueError("EDGE Palette overrun")
                        
                        packed = np.frombuffer(payloads[pay_idx:pay_idx+packed_len], dtype=np.uint8)
                        pay_idx += packed_len
                        
                        idx = np.empty(packed_len * 4, dtype=np.uint8)
                        idx[0::4] = (packed >> 6) & 3
                        idx[1::4] = (packed >> 4) & 3
                        idx[2::4] = (packed >> 2) & 3
                        idx[3::4] = packed & 3
                        
                        idx = idx[:N] # Safely strip encoder padding
                        curr_gray[y:y+block_h, x:x+block_w] = (min_v + idx * ((max_v - min_v) // 3)).reshape(block_h, block_w)
                
                # V5 Mode 15: Chaos / Bloom        
                elif node.tag == 15:
                    if pay_idx + 2 > len(payloads): raise ValueError("CHAOS Vector overrun")
                    dy, dx = struct.unpack_from("<bb", payloads, pay_idx)
                    pay_idx += 2
                    src_y = np.clip(y - dy, 0, h - block_h)
                    src_x = np.clip(x - dx, 0, w - block_w)
                    
                    q_h, q_w = block_h // 4, block_w // 4
                    center_patch = prev_gray[src_y+q_h : src_y+block_h-q_h, src_x+q_w : src_x+block_w-q_w]
                    
                    if center_patch.size > 0 and block_h >= 4 and block_w >= 4:
                        curr_gray[y:y+block_h, x:x+block_w] = cv2.resize(center_patch, (block_w, block_h), interpolation=cv2.INTER_LINEAR)
                    else:
                        curr_gray[y:y+block_h, x:x+block_w] = prev_gray[src_y:src_y+block_h, src_x:src_x+block_w]

            MAX_BLOCK = 64
            for y in range(0, h, MAX_BLOCK):
                for x in range(0, w, MAX_BLOCK):
                    root_node = deserialize_quadtree(reader, x, y, MAX_BLOCK, min_size=8)
                    process_node(root_node)
                    
            frames.append(curr_gray)
            prev_gray = curr_gray
            
        return frames