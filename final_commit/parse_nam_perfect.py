import struct
import numpy as np

def analyze_nam_bytes(encoded_path):
    with open(encoded_path, "rb") as stream:
        header = stream.read(20)
        if len(header) < 20 or header[:6] != b"NAM_V5":
            return
        w, h, fps = struct.unpack("<IIf", header[6:18])
        
        mode_bytes = {0xFF: 0} # 0xFF is skip run
        for i in range(16): mode_bytes[i] = 0
        
        frame_idx = 0
        while True:
            ftype_data = stream.read(1)
            if not ftype_data: break
            frame_type = ftype_data[0]
            
            if frame_type == 0:
                l_data = stream.read(4)
                if not l_data: break
                l = struct.unpack("<I", l_data)[0]
                stream.read(l) # skip keyframe payload
            elif frame_type == 1:
                l_data = stream.read(8)
                if not l_data: break
                len_modes, len_payloads = struct.unpack("<II", l_data)
                
                mode_data = stream.read(len_modes)
                payload_data = stream.read(len_payloads)
                
                # Now we know mode_data and payload_data.
                # Let's count them!
                mode_idx, pay_idx = 0, 0
                grid_h, grid_w = h // 16, w // 16
                processed = np.zeros((grid_h, grid_w), dtype=bool)
                skip_remaining = 0
                
                for y_idx in range(grid_h):
                    for x_idx in range(grid_w):
                        if processed[y_idx, x_idx]: continue
                        if skip_remaining > 0:
                            skip_remaining -= 1
                            if skip_remaining == 0: continue
                            processed[y_idx, x_idx] = True
                            continue
                            
                        if mode_idx >= len(mode_data): break
                        flag = mode_data[mode_idx]
                        mode_idx += 1
                        
                        start_pay = pay_idx
                        if flag == 0xFF:
                            skip_remaining = struct.unpack_from("<H", payload_data, pay_idx)[0]
                            pay_idx += 2
                        elif flag == 0: pay_idx += 256
                        elif flag == 1: pay_idx += 2
                        elif flag == 2: pay_idx += 256
                        elif flag == 3: pay_idx += 2
                        elif flag == 4: pay_idx += 2
                        elif flag == 5: pay_idx += 256 # Added mode 5 (solid color with error) just in case
                        elif flag == 6: pay_idx += 0
                        elif flag == 7: pay_idx += 1
                        elif flag == 8 or flag == 9 or flag == 10:
                            num_coeffs = struct.unpack_from("<H", payload_data, pay_idx)[0]
                            pay_idx += 2 + (num_coeffs * 3)
                        elif flag == 11: pay_idx += 0
                        elif flag == 12: pay_idx += 2
                        elif flag == 13: pay_idx += 1
                        elif flag == 14:
                            bh, bw = struct.unpack_from("<BB", payload_data, pay_idx)
                            pay_idx += 2 + (bh*bw)
                        elif flag == 15: pay_idx += 32
                            
                        mode_bytes[flag] += (pay_idx - start_pay)
                        mode_bytes[flag] += 1 # mode byte itself
                        
        print(f"--- Telemetry for {encoded_path} ---")
        total_p_bytes = sum(mode_bytes.values())
        print(f"Total P-Frame Payload/Mode Bytes: {total_p_bytes}")
        for m, b in mode_bytes.items():
            if b > 0:
                print(f"Mode {m if m != 0xFF else 'Skip'}: {b} bytes ({b/total_p_bytes*100:.2f}%)")

if __name__ == "__main__":
    analyze_nam_bytes("../test_vid/ball.nam")
