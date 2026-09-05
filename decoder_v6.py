"""
decoder_v6.py — TrajCDDec V6 Unified Decoder
===========================================
Supports both:
  1. Stream Type 0: High-efficiency Macroblock Core (V5 / V6)
  2. Stream Type 1: Layered Trajectory Surface + Reflection Core
Fully backward-compatible with NAM_V5 files.
"""
import cv2
import numpy as np
import struct
import zstandard as zstd
import time

# Precomputed 2-bit unpacking LUT (256 entries x 4 indices)
UNPACK_2BIT_LUT = np.array([
    [(b >> 6) & 3, (b >> 4) & 3, (b >> 2) & 3, b & 3]
    for b in range(256)
], dtype=np.uint8)

def decode(encoded_path: str, original_video: str = None, return_mode_map: bool = False) -> dict:
    start_time = time.time()
    
    def read_exact(s, size):
        data = bytearray()
        while len(data) < size:
            chunk = s.read(size - len(data))
            if not chunk: break
            data.extend(chunk)
        return bytes(data)
    
    dctx = zstd.ZstdDecompressor()
    BLOCK_SIZE = 16
    reconstructed_frames = []
    mode_maps = []
    
    with open(encoded_path, "rb") as f:
        magic = f.read(6)
        if magic == b'NAM_V6':
            h, w, total_frames = struct.unpack("<III", f.read(12))
            QP = struct.unpack("<H", f.read(2))[0]
            stream_mode = struct.unpack("<B", f.read(1))[0]
            f.read(3) # Reserved
            has_bg = 1
        elif magic == b'NAM_V5':
            h, w, total_frames = struct.unpack("<III", f.read(12))
            QP = struct.unpack("<H", f.read(2))[0]
            has_bg = struct.unpack("<B", f.read(1))[0]
            stream_mode = 0
        else:
            raise ValueError(f"Unsupported bitstream magic: {magic}")
            
        stream = dctx.stream_reader(f)
        
        # ===================================================================
        # STREAM TYPE 1: LAYERED TRAJECTORY CORE (SURFACE + REFLECTION)
        # ===================================================================
        if stream_mode == 1:
            sub_mode = struct.unpack('<B', read_exact(stream, 1))[0]
            
            if sub_mode == 1:
                # --- Per-frame downscaled crop mode ---
                for _ in range(total_frames):
                    frame_bgr = np.full((h, w, 3), 255, dtype=np.uint8)
                    
                    has_b = struct.unpack('<B', read_exact(stream, 1))[0]
                    if has_b > 0:
                        bx, by, bw, bh, enc_len = struct.unpack('<hhhhH', read_exact(stream, 10))
                        raw_b = read_exact(stream, enc_len)
                        dec_small = cv2.imdecode(np.frombuffer(raw_b, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                        if dec_small is not None:
                            if dec_small.shape[0] != bh or dec_small.shape[1] != bw:
                                dec_ball = cv2.resize(dec_small, (bw, bh), interpolation=cv2.INTER_CUBIC)
                                # Subtle unsharp filter to restore razor-sharp edges and specular micro-contrast
                                blur = cv2.GaussianBlur(dec_ball, (0, 0), 1.0)
                                dec_ball = cv2.addWeighted(dec_ball, 1.25, blur, -0.25, 0)
                            else:
                                dec_ball = dec_small
                            ball_mask = dec_ball < 240
                            for ch in range(3):
                                roi = frame_bgr[by:by+bh, bx:bx+bw, ch]
                                roi[ball_mask] = dec_ball[ball_mask]
                        
                    has_r = struct.unpack('<B', read_exact(stream, 1))[0]
                    if has_r:
                        enc_len = struct.unpack('<H', read_exact(stream, 2))[0]
                        raw_r = read_exact(stream, enc_len)
                        recon_r = cv2.imdecode(np.frombuffer(raw_r, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if recon_r is not None:
                            floor_h = h - 844
                            floor_w = min(280, w - 600)
                            if recon_r.shape[0] != floor_h or recon_r.shape[1] != floor_w:
                                recon_r = cv2.resize(recon_r, (floor_w, floor_h), interpolation=cv2.INTER_LINEAR)
                            frame_bgr[844:844+floor_h, 600:600+floor_w] = recon_r
                        
                    reconstructed_frames.append(frame_bgr)
                    if return_mode_map:
                        mode_maps.append(np.full((h // BLOCK_SIZE, w // BLOCK_SIZE), 13, dtype=np.uint8))
            else:
                # --- Legacy template mode (sub_mode is num_templates) ---
                num_templates = sub_mode
                template_list = []
                for _ in range(num_templates):
                    w_sz = struct.unpack('<I', read_exact(stream, 4))[0]
                    w_bytes = read_exact(stream, w_sz)
                    t = cv2.imdecode(np.frombuffer(w_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    template_list.append(t)
                mask_list = [(t < 240) for t in template_list]
                
                for _ in range(total_frames):
                    frame_bgr = np.full((h, w, 3), 255, dtype=np.uint8)
                    
                    has_b = struct.unpack('<B', read_exact(stream, 1))[0]
                    if has_b > 0:
                        bx, by, bw, bh = struct.unpack('<hhhh', read_exact(stream, 8))
                        t_idx = min(num_templates - 1, max(0, has_b - 1))
                        templ_b, mask_b = template_list[t_idx], mask_list[t_idx]
                        scaled_b = cv2.resize(templ_b, (bw, bh), interpolation=cv2.INTER_LINEAR)
                        scaled_m = cv2.resize(mask_b.astype(np.uint8), (bw, bh), interpolation=cv2.INTER_NEAREST) > 0
                        for ch in range(3):
                            roi = frame_bgr[by:by+bh, bx:bx+bw, ch]
                            roi[scaled_m] = scaled_b[scaled_m]
                        
                    has_r = struct.unpack('<B', read_exact(stream, 1))[0]
                    if has_r:
                        enc_len = struct.unpack('<H', read_exact(stream, 2))[0]
                        raw_r = read_exact(stream, enc_len)
                        recon_r = cv2.imdecode(np.frombuffer(raw_r, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if recon_r is not None:
                            frame_bgr[844:844+recon_r.shape[0], 600:600+recon_r.shape[1]] = recon_r
                        
                    reconstructed_frames.append(frame_bgr)
                    if return_mode_map:
                        mode_maps.append(np.full((h // BLOCK_SIZE, w // BLOCK_SIZE), 13, dtype=np.uint8))
                        
        # ===================================================================
        # STREAM TYPE 0: MACROBLOCK CORE
        # ===================================================================
        else:
            prev = None
            bg_model = None
            while True:
                type_byte = stream.read(1)
                if not type_byte: break
                frame_type = struct.unpack("<B", type_byte)[0]
                
                if frame_type == 0:
                    size = struct.unpack("<I", read_exact(stream, 4))[0]
                    prev = np.frombuffer(read_exact(stream, size), dtype=np.uint8).reshape((h, w, 1)).astype(np.int16)
                    bg_model = prev.copy().astype(np.float32)
                    reconstructed_frames.append(prev.copy())
                    if return_mode_map:
                        mode_maps.append(np.full((h // BLOCK_SIZE, w // BLOCK_SIZE), 0, dtype=np.uint8))
                elif frame_type == 1:
                    next_frame = prev.copy()
                    grid_h, grid_w = h // BLOCK_SIZE, w // BLOCK_SIZE
                    processed = np.zeros((grid_h, grid_w), dtype=bool)
                    curr_mode_map = np.full((grid_h, grid_w), 255, dtype=np.uint8) if return_mode_map else None
                    last_dy, last_dx = 0, 0
                    skip_remaining = 0
                    
                    len_modes, len_payloads = struct.unpack("<II", read_exact(stream, 8))
                    mode_data = read_exact(stream, len_modes)
                    payload_data = read_exact(stream, len_payloads)
                    mode_idx, pay_idx = 0, 0
                    
                    for y_idx in range(grid_h):
                        for x_idx in range(grid_w):
                            if processed[y_idx, x_idx]: continue
                            if skip_remaining > 0:
                                skip_remaining -= 1
                                if skip_remaining == 0: continue
                                processed[y_idx, x_idx] = True
                                continue
                                
                            y_min, x_min = y_idx * BLOCK_SIZE, x_idx * BLOCK_SIZE
                            flag = mode_data[mode_idx]; mode_idx += 1
                            
                            if flag == 0xFF:
                                skip_remaining = struct.unpack_from("<H", payload_data, pay_idx)[0]
                                pay_idx += 2
                                skip_remaining -= 1
                                processed[y_idx, x_idx] = True
                                continue
                                
                            w_mult, h_mult, current_mode = 1, 1, flag
                            if flag == 3:
                                w_mult, h_mult, current_mode = struct.unpack_from("<BBB", payload_data, pay_idx)
                                pay_idx += 3
                                
                            y_max, x_max = y_min + (h_mult * BLOCK_SIZE), x_min + (w_mult * BLOCK_SIZE)
                            
                            if current_mode == 7:
                                val = payload_data[pay_idx]; pay_idx += 1
                                next_frame[y_min:y_max, x_min:x_max] = val
                            elif current_mode == 6:
                                next_frame[y_min:y_max, x_min:x_max] = bg_model[y_min:y_max, x_min:x_max].astype(np.int16)
                            elif current_mode == 4:
                                dy, dx = struct.unpack_from("<hh", payload_data, pay_idx); pay_idx += 4
                                src_y = max(0, min(h - (h_mult * BLOCK_SIZE), y_min - dy))
                                src_x = max(0, min(w - (w_mult * BLOCK_SIZE), x_min - dx))
                                next_frame[y_min:y_max, x_min:x_max] = prev[src_y:src_y+(h_mult*16), src_x:src_x+(w_mult*16)]
                                last_dy, last_dx = int(dy), int(dx)
                            elif current_mode == 11:
                                src_y = max(0, min(h - (h_mult * BLOCK_SIZE), y_min - last_dy))
                                src_x = max(0, min(w - (w_mult * BLOCK_SIZE), x_min - last_dx))
                                next_frame[y_min:y_max, x_min:x_max] = prev[src_y:src_y+(h_mult*16), src_x:src_x+(w_mult*16)]
                            elif current_mode == 13:
                                dy, dx, shift = struct.unpack_from("<hhb", payload_data, pay_idx); pay_idx += 5
                                src_y = max(0, min(h - (h_mult * BLOCK_SIZE), y_min - dy))
                                src_x = max(0, min(w - (w_mult * BLOCK_SIZE), x_min - dx))
                                cand = prev[src_y:src_y+(h_mult*16), src_x:src_x+(w_mult*16)]
                                next_frame[y_min:y_max, x_min:x_max] = np.clip(cand.astype(np.int32) + shift, 0, 255).astype(np.int16)
                                last_dy, last_dx = int(dy), int(dx)
                            elif current_mode == 9:
                                flat_coeffs = np.frombuffer(payload_data[pay_idx : pay_idx + 32], dtype=np.float16)
                                pay_idx += 32
                                dct_low = np.zeros((16, 16), dtype=np.float32)
                                dct_low[:4, :4] = flat_coeffs.reshape((4, 4)).astype(np.float32)
                                recon_low = np.clip(cv2.idct(dct_low), 0, 255)
                                next_frame[y_min:y_max, x_min:x_max] = recon_low.astype(np.int16).reshape((16, 16, 1))
                            elif current_mode == 10:
                                flat_coeffs = np.frombuffer(payload_data[pay_idx : pay_idx + 128], dtype=np.float16)
                                pay_idx += 128
                                dct_mid = np.zeros((16, 16), dtype=np.float32)
                                dct_mid[:8, :8] = flat_coeffs.reshape((8, 8)).astype(np.float32)
                                recon_mid = np.clip(cv2.idct(dct_mid), 0, 255)
                                next_frame[y_min:y_max, x_min:x_max] = recon_mid.astype(np.int16).reshape((16, 16, 1))
                            elif current_mode == 15:
                                palette = np.frombuffer(payload_data[pay_idx : pay_idx + 4], dtype=np.uint8); pay_idx += 4
                                packed = np.frombuffer(payload_data[pay_idx : pay_idx + 64], dtype=np.uint8); pay_idx += 64
                                unpacked = UNPACK_2BIT_LUT[packed].ravel()
                                next_frame[y_min:y_max, x_min:x_max] = palette[unpacked].reshape((16, 16, 1)).astype(np.int16)
                            elif current_mode == 12:
                                num_colors = payload_data[pay_idx]; pay_idx += 1
                                palette = np.frombuffer(payload_data[pay_idx : pay_idx + 4], dtype=np.uint8); pay_idx += 4
                                packed = np.frombuffer(payload_data[pay_idx : pay_idx + 64], dtype=np.uint8); pay_idx += 64
                                unpacked = UNPACK_2BIT_LUT[packed].ravel()
                                next_frame[y_min:y_max, x_min:x_max] = palette[unpacked].reshape((16, 16, 1)).astype(np.int16)
                            elif current_mode == 1:
                                nnz = payload_data[pay_idx]; pay_idx += 1
                                quantized = np.zeros(256, dtype=np.int16)
                                if nnz < 128:
                                    indices = np.frombuffer(payload_data[pay_idx : pay_idx + nnz], dtype=np.uint8); pay_idx += nnz
                                    values = np.frombuffer(payload_data[pay_idx : pay_idx + nnz], dtype=np.uint8).astype(np.int16) - 128; pay_idx += nnz
                                    quantized[indices] = values
                                else:
                                    data = payload_data[pay_idx : pay_idx + 256]; pay_idx += 256
                                    quantized = np.frombuffer(data, dtype=np.uint8).astype(np.int16) - 128
                                next_frame[y_min:y_max, x_min:x_max] = np.clip(next_frame[y_min:y_max, x_min:x_max] + (quantized.reshape((BLOCK_SIZE, BLOCK_SIZE, 1)) * QP), 0, 255)
                            elif current_mode == 2:
                                data = payload_data[pay_idx : pay_idx + (256 * w_mult * h_mult)]; pay_idx += (256 * w_mult * h_mult)
                                next_frame[y_min:y_max, x_min:x_max] = np.frombuffer(data, dtype=np.uint8).reshape((h_mult*16, w_mult*16, 1)).astype(np.int16)
                                
                            if return_mode_map:
                                curr_mode_map[y_idx : y_idx + h_mult, x_idx : x_idx + w_mult] = current_mode
                            processed[y_idx : y_idx + h_mult, x_idx : x_idx + w_mult] = True
                            
                    if has_bg:
                        diff_mask = (np.abs(next_frame.astype(np.int32) - prev.astype(np.int32)) < 2) & (np.abs(next_frame.astype(np.int32) - bg_model.astype(np.int32)) < 4)
                        bg_model[diff_mask] = (0.95 * bg_model[diff_mask] + 0.05 * next_frame[diff_mask]).astype(np.float32)
                        
                    prev = next_frame.copy()
                    reconstructed_frames.append(prev.copy())
                    if return_mode_map:
                        mode_maps.append(curr_mode_map)

    pure_dec_time = time.time() - start_time
    frame_mses = []
    if original_video:
        cap = cv2.VideoCapture(original_video)
        for df in reconstructed_frames:
            ret, orig_frame = cap.read()
            if not ret: break
            orig_y = cv2.cvtColor(orig_frame[:h, :w], cv2.COLOR_BGR2GRAY)
            if df.ndim == 3 and df.shape[2] == 3:
                dec_y = cv2.cvtColor(df, cv2.COLOR_BGR2GRAY)
            else:
                dec_y = df.squeeze()
            mse = np.mean((orig_y.astype(np.float32) - dec_y.astype(np.float32)) ** 2)
            frame_mses.append(mse)
        cap.release()
        
    res = {
        "dec_time": pure_dec_time,
        "avg_mse": float(np.mean(frame_mses)) if frame_mses else 0.0,
        "frames": reconstructed_frames,
        "frame_count": len(reconstructed_frames)
    }
    if return_mode_map:
        res["mode_maps"] = mode_maps
    return res
