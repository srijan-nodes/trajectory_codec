"""
decoder.py — V5 Full Engine Decoder
=========================================
"""
import cv2
import numpy as np
import struct
import zlib
import zstandard as zstd
import time

def decode(encoded_path: str, original_video: str = None, return_mode_map: bool = False) -> dict:
    global telemetry_bytes
    telemetry_bytes = {i: 0 for i in range(16)}; telemetry_bytes[0xFF] = 0
    start_time = time.time()
    
    def read_exact(s, size):
        data = bytearray()
        while len(data) < size:
            chunk = s.read(size - len(data))
            if not chunk:
                break
            data.extend(chunk)
        return bytes(data)
    
    dctx = zstd.ZstdDecompressor()
    cap = cv2.VideoCapture(original_video) if original_video else None
    BLOCK_SIZE = 16
    frame_mses = []
    reconstructed_frames = []
    mode_maps = []
    
    with open(encoded_path, "rb") as f:
        magic = f.read(6)
        h, w, total_frames = struct.unpack("<III", f.read(12))
        QP = struct.unpack("<H", f.read(2))[0]
        has_bg = struct.unpack("<B", f.read(1))[0]
        
        prev = None
        bg_model = None
        stream = dctx.stream_reader(f)

        while True:
            if cap:
                ret, orig_frame = cap.read()
                if not ret: break
                orig_y = cv2.cvtColor(orig_frame[:h, :w], cv2.COLOR_BGR2GRAY)
            else:
                orig_y = None
            
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
                if orig_y is not None:
                    mse = np.mean((orig_y.astype(np.float32) - prev.squeeze().astype(np.float32)) ** 2)
                    frame_mses.append(mse)
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
                            if skip_remaining == 0:
                                continue
                            processed[y_idx, x_idx] = True
                            continue
                            
                        y_min, x_min = y_idx * BLOCK_SIZE, x_idx * BLOCK_SIZE
                        flag = mode_data[mode_idx]; mode_idx += 1
                        start_pay = pay_idx
                        telemetry_bytes[flag] += 1
                        
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
                            src_y = np.clip(y_min - dy, 0, h - (h_mult * BLOCK_SIZE))
                            src_x = np.clip(x_min - dx, 0, w - (w_mult * BLOCK_SIZE))
                            next_frame[y_min:y_max, x_min:x_max] = prev[src_y:src_y+(h_mult*16), src_x:src_x+(w_mult*16)].copy()
                            last_dy, last_dx = int(dy), int(dx)
                        elif current_mode == 11:
                            src_y = np.clip(y_min - last_dy, 0, h - (h_mult * BLOCK_SIZE))
                            src_x = np.clip(x_min - last_dx, 0, w - (w_mult * BLOCK_SIZE))
                            next_frame[y_min:y_max, x_min:x_max] = prev[src_y:src_y+(h_mult*16), src_x:src_x+(w_mult*16)].copy()
                        elif current_mode == 13: 
                            dy, dx, shift = struct.unpack_from("<hhb", payload_data, pay_idx); pay_idx += 5
                            src_y = np.clip(y_min - dy, 0, h - (h_mult * BLOCK_SIZE))
                            src_x = np.clip(x_min - dx, 0, w - (w_mult * BLOCK_SIZE))
                            cand = prev[src_y:src_y+(h_mult*16), src_x:src_x+(w_mult*16)]
                            next_frame[y_min:y_max, x_min:x_max] = np.clip(cand.astype(np.int32) + shift, 0, 255).astype(np.int16)
                            last_dy, last_dx = int(dy), int(dx)
                        elif current_mode == 9:
                            flat_coeffs = np.frombuffer(payload_data[pay_idx : pay_idx + 32], dtype=np.float16)
                            pay_idx += 32
                            dct_low = np.zeros((16, 16), dtype=np.float32)
                            dct_low[:4, :4] = flat_coeffs.reshape((4, 4)).astype(np.float32)
                            recon_low = cv2.idct(dct_low)
                            next_frame[y_min:y_max, x_min:x_max] = recon_low.astype(np.int16).reshape((16, 16, 1))
                        elif current_mode == 10:
                            flat_coeffs = np.frombuffer(payload_data[pay_idx : pay_idx + 128], dtype=np.float16)
                            pay_idx += 128
                            dct_mid = np.zeros((16, 16), dtype=np.float32)
                            dct_mid[:8, :8] = flat_coeffs.reshape((8, 8)).astype(np.float32)
                            recon_mid = cv2.idct(dct_mid)
                            next_frame[y_min:y_max, x_min:x_max] = recon_mid.astype(np.int16).reshape((16, 16, 1))
                        elif current_mode == 15: 
                            palette = np.frombuffer(payload_data[pay_idx : pay_idx + 4], dtype=np.uint8); pay_idx += 4
                            packed = np.frombuffer(payload_data[pay_idx : pay_idx + 64], dtype=np.uint8); pay_idx += 64
                            unpacked = np.zeros(256, dtype=np.uint8)
                            for i in range(64):
                                byte = packed[i]
                                unpacked[i*4]   = (byte >> 6) & 0x03
                                unpacked[i*4+1] = (byte >> 4) & 0x03
                                unpacked[i*4+2] = (byte >> 2) & 0x03
                                unpacked[i*4+3] = byte & 0x03
                            mapped = palette[unpacked].reshape((16, 16, 1))
                            next_frame[y_min:y_max, x_min:x_max] = mapped.astype(np.int16)
                        elif current_mode == 12:
                            num_colors = payload_data[pay_idx]; pay_idx += 1
                            palette = np.frombuffer(payload_data[pay_idx : pay_idx + 4], dtype=np.uint8); pay_idx += 4
                            packed = np.frombuffer(payload_data[pay_idx : pay_idx + 64], dtype=np.uint8); pay_idx += 64
                            unpacked = np.zeros(256, dtype=np.uint8)
                            for i in range(64):
                                byte = packed[i]
                                unpacked[i*4]   = (byte >> 6) & 0x03
                                unpacked[i*4+1] = (byte >> 4) & 0x03
                                unpacked[i*4+2] = (byte >> 2) & 0x03
                                unpacked[i*4+3] = byte & 0x03
                            mapped = palette[unpacked].reshape((16, 16, 1))
                            next_frame[y_min:y_max, x_min:x_max] = mapped.astype(np.int16)
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
                            
                        telemetry_bytes[flag] += (pay_idx - start_pay)
                        if return_mode_map:
                            curr_mode_map[y_idx : y_idx + h_mult, x_idx : x_idx + w_mult] = current_mode
                        processed[y_idx : y_idx + h_mult, x_idx : x_idx + w_mult] = True

                if has_bg:
                    diff_mask = np.abs(next_frame.astype(np.int32) - prev.astype(np.int32)) < 5
                    bg_model[diff_mask] = (0.95 * bg_model[diff_mask] + 0.05 * next_frame[diff_mask]).astype(np.float32)

                prev = next_frame.copy()
                reconstructed_frames.append(prev.copy())
                if return_mode_map:
                    mode_maps.append(curr_mode_map)
                if orig_y is not None:
                    mse = np.mean((orig_y.astype(np.float32) - prev.squeeze().astype(np.float32)) ** 2)
                    frame_mses.append(mse)

    if cap:
        cap.release()
    dec_time = time.time() - start_time
    res = {
        "dec_time": dec_time,
        "avg_mse": np.mean(frame_mses) if frame_mses else 0.0,
        "frames": reconstructed_frames,
        "frame_count": len(reconstructed_frames)
    }
    if return_mode_map:
        res["mode_maps"] = mode_maps
    print("\n--- TELEMETRY ---")
    total = sum(telemetry_bytes.values())
    if total > 0:
        print(f"Total Bytes: {total}")
        for k, v in telemetry_bytes.items():
            if v > 0:
                print(f"Mode {k if k != 0xFF else 'Skip'}: {v} bytes ({v/total*100:.2f}%)")
    return res
