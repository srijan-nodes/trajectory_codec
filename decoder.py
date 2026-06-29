import sys
import struct
import cv2
import zstandard as zstd
import numpy as np
import time

class DecodedObject:
    def __init__(self, id, x, y, w, h):
        self.id = id
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.locked = False
        self.c_class = 0
        self.states = {} # state_id -> binary mask
        self.current_state = None

class BinaryDecoder:
    def __init__(self, filepath):
        self.f = open(filepath, 'rb')
        header = self.f.read(4)
        if header != b'NAM\x01':
            raise ValueError("Invalid .nam file header")
        
        self.width, self.height = struct.unpack('<HH', self.f.read(4))
        self.dctx = zstd.ZstdDecompressor()
        
        self.objects = {}

    def decode_frame(self):
        # Reads events until 0xFF (FRAME_END) or EOF
        while True:
            op_byte = self.f.read(1)
            if not op_byte:
                return False # EOF
            
            op = op_byte[0]
            if op == 0xFF:
                return True # Frame complete
                
            elif op == 0x01: # SPAWN
                data = self.f.read(10)
                obj_id, x, y, w, h = struct.unpack('<HHHHH', data)
                self.objects[obj_id] = DecodedObject(obj_id, x, y, w, h)
                
            elif op == 0x02: # LOCK
                data = self.f.read(3)
                obj_id, c_class = struct.unpack('<HB', data)
                if obj_id in self.objects:
                    self.objects[obj_id].locked = True
                    self.objects[obj_id].c_class = c_class
                    
            elif op == 0x03: # INIT_STATE
                data = self.f.read(12)
                obj_id, state_id, mask_w, mask_h = struct.unpack('<HHII', data)
                payload_size = struct.unpack('<I', self.f.read(4))[0]
                compressed = self.f.read(payload_size)
                
                packed = self.dctx.decompress(compressed)
                mask_bool = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))
                mask_bool = mask_bool[:mask_h * mask_w].reshape((mask_h, mask_w))
                mask = mask_bool * 255
                
                if obj_id in self.objects:
                    self.objects[obj_id].states[state_id] = mask.astype(np.uint8)
                    self.objects[obj_id].current_state = state_id
                    self.objects[obj_id].w = mask_w
                    self.objects[obj_id].h = mask_h
                    
            elif op == 0x04: # STATE_CHANGE
                data = self.f.read(14)
                obj_id, old_state, new_state, new_w, new_h = struct.unpack('<HHHII', data)
                payload_size = struct.unpack('<I', self.f.read(4))[0]
                
                if obj_id in self.objects:
                    obj = self.objects[obj_id]
                    if payload_size > 0:
                        compressed = self.f.read(payload_size)
                        packed = self.dctx.decompress(compressed)
                        xor_bool = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))
                        
                        mh = max(obj.h, new_h)
                        mw = max(obj.w, new_w)
                        xor_bool = xor_bool[:mh * mw].reshape((mh, mw))
                        xor_mask = xor_bool * 255
                        
                        old_mask = obj.states[old_state]
                        padded_old = np.zeros((mh, mw), dtype=np.uint8)
                        padded_old[:old_mask.shape[0], :old_mask.shape[1]] = old_mask
                        
                        new_mask = cv2.bitwise_xor(padded_old, xor_mask.astype(np.uint8))
                        # Crop to new size
                        new_mask = new_mask[:new_h, :new_w]
                        
                        obj.states[new_state] = new_mask
                        
                    obj.current_state = new_state
                    obj.w = new_w
                    obj.h = new_h
                    
            elif op == 0x05: # MOVE
                data = self.f.read(4)
                obj_id, dx, dy = struct.unpack('<Hbb', data)
                if obj_id in self.objects:
                    self.objects[obj_id].x += dx
                    self.objects[obj_id].y += dy
                    
            elif op == 0x06: # DESPAWN
                data = self.f.read(2)
                obj_id = struct.unpack('<H', data)[0]
                if obj_id in self.objects:
                    del self.objects[obj_id]
                    
    def close(self):
        self.f.close()


def play_decoded_video(nam_path):
    print(f"Decoding {nam_path}...")
    decoder = BinaryDecoder(nam_path)
    
    cv2.namedWindow("SOBVC Player", cv2.WINDOW_NORMAL)
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        has_frame = decoder.decode_frame()
        if not has_frame:
            break
            
        frame_count += 1
        canvas = np.zeros((decoder.height, decoder.width, 3), dtype=np.uint8)
        
        for obj_id, obj in decoder.objects.items():
            color = (0, 255, 0) if obj.locked else (0, 0, 255)
            
            # Draw bounding box
            cv2.rectangle(canvas, (int(obj.x), int(obj.y)), (int(obj.x + obj.w), int(obj.y + obj.h)), color, 1)
            
            # Draw mask if it has one
            if obj.current_state is not None and obj.current_state in obj.states:
                mask = obj.states[obj.current_state]
                y1 = int(obj.y)
                y2 = y1 + mask.shape[0]
                x1 = int(obj.x)
                x2 = x1 + mask.shape[1]
                
                # Clip to screen
                canvas_h, canvas_w = canvas.shape[:2]
                if x1 < canvas_w and y1 < canvas_h and x2 > 0 and y2 > 0:
                    mx1 = max(0, -x1)
                    my1 = max(0, -y1)
                    mx2 = mask.shape[1] - max(0, x2 - canvas_w)
                    my2 = mask.shape[0] - max(0, y2 - canvas_h)
                    
                    cx1 = max(0, x1)
                    cy1 = max(0, y1)
                    cx2 = min(canvas_w, x2)
                    cy2 = min(canvas_h, y2)
                    
                    roi = canvas[cy1:cy2, cx1:cx2]
                    m_roi = mask[my1:my2, mx1:mx2]
                    
                    # Add mask to blue channel for visual
                    roi[m_roi > 0] = (255, 200, 0) # cyan color for mask
                    
        cv2.imshow("SOBVC Player", canvas)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"Decoded {frame_count} frames at {fps:.1f} FPS")
    decoder.close()
    cv2.destroyAllWindows()

def decode_and_profile(video_path, nam_path):
    start_time = time.time()
    decoder = BinaryDecoder(nam_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video")
        return 0, []

    frame_mses = []
    
    while True:
        has_frame = decoder.decode_frame()
        if not has_frame:
            break
            
        ret, orig_frame = cap.read()
        if not ret: break
            
        canvas = np.zeros((decoder.height, decoder.width, 3), dtype=np.uint8)
        
        for obj_id, obj in decoder.objects.items():
            if obj.current_state is not None and obj.current_state in obj.states:
                mask = obj.states[obj.current_state]
                y1 = int(obj.y)
                y2 = y1 + mask.shape[0]
                x1 = int(obj.x)
                x2 = x1 + mask.shape[1]
                
                canvas_h, canvas_w = canvas.shape[:2]
                if x1 < canvas_w and y1 < canvas_h and x2 > 0 and y2 > 0:
                    mx1 = max(0, -x1)
                    my1 = max(0, -y1)
                    mx2 = mask.shape[1] - max(0, x2 - canvas_w)
                    my2 = mask.shape[0] - max(0, y2 - canvas_h)
                    
                    cx1 = max(0, x1)
                    cy1 = max(0, y1)
                    cx2 = min(canvas_w, x2)
                    cy2 = min(canvas_h, y2)
                    
                    roi = canvas[cy1:cy2, cx1:cx2]
                    m_roi = mask[my1:my2, mx1:mx2]
                    
                    # Instead of cyan color, copy the original video's patch where mask is 1
                    orig_patch = orig_frame[cy1:cy2, cx1:cx2]
                    # We just use the raw orig_patch where m_roi > 0.
                    # Wait, SOBVC is supposed to reconstruct video. If we cheat and copy from orig_patch,
                    # the encoder didn't actually transmit the texture! 
                    # The user's architecture only sends binary masks for TEXT, meaning it's just green boxes and cyan text visually.
                    # If I calculate MSE of cyan boxes against the original video, the MSE will be huge!
                    # What if I just return dummy MSE for now? Or I can compare just the structure?
                    # The user provided a template for a full video codec. Our SOBVC is a semantic object codec. 
                    roi[m_roi > 0] = (255, 200, 0)
        
        # Calculate MSE
        err = np.sum((canvas.astype("float") - orig_frame.astype("float")) ** 2)
        err /= float(canvas.shape[0] * canvas.shape[1] * canvas.shape[2])
        frame_mses.append(err)

    dec_time = time.time() - start_time
    decoder.close()
    cap.release()
    return dec_time, frame_mses

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python decoder.py <input.nam>")
        sys.exit(1)
    play_decoded_video(sys.argv[1])
