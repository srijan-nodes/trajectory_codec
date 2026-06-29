import cv2
import sys
import struct
import zstandard as zstd
import os
import numpy as np
import math
import hashlib
from collections import deque
import time

def bb_iou(boxA, boxB):
    xA, yA, wA, hA = boxA
    xB, yB, wB, hB = boxB
    x_left, y_top = max(xA, xB), max(yA, yB)
    x_right, y_bottom = min(xA + wA, xB + wB), min(yA + hA, yB + hB)
    if x_right < x_left or y_bottom < y_top: return 0.0
    inter_area = (x_right - x_left) * (y_bottom - y_top)
    return inter_area / float(wA*hA + wB*hB - inter_area)

def get_safe_roi(img, x, y, w, h):
    img_h, img_w = img.shape[:2]
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(img_w, int(x+w)), min(img_h, int(y+h))
    if x1 >= x2 or y1 >= y2: return x1, y1, 0, 0, np.zeros((0,0), dtype=img.dtype)
    return x1, y1, x2-x1, y2-y1, img[y1:y2, x1:x2]

def fill_holes(binary):
    """Standard flood-fill hole-filling so a ring-shaped glyph (e.g. '0')
    is treated as one solid blob rather than breaking into fragments."""
    h, w = binary.shape[:2]
    flood = binary.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    return binary | flood_inv

class PersistentObject:
    def __init__(self, obj_id):
        self.id = obj_id
        self.history = deque(maxlen=10)
        self.patch_history = deque(maxlen=10)

        # Per-glyph state tracking: one slot per glyph column-segment, left-to-right.
        self.glyph_slots = []
        self.total_distinct_states = 0
        self.total_transitions = 0

        self.locked = False
        self.classification = 'UNKNOWN'

        self.age = 0
        self.missing_frames = 0
        self.frames_alive = 0
        self.lifetime_bytes = 0
        self.raw_pixels = 0
        self._last_actual_pixels = None  # cache for exact-match fast path

        self.has_init_state = False
        self.dead = False

    def _hash_mask(self, mask_array, aspect, fg_fraction):
        norm_mask = cv2.resize(mask_array, (32, 32), interpolation=cv2.INTER_NEAREST)
        base_hash = hashlib.md5(np.packbits(norm_mask > 0).tobytes()).hexdigest()
        aspect_bucket = int(aspect * 5)
        fg_bucket = int(fg_fraction * 10)
        return f"{base_hash}_{aspect_bucket}_{fg_bucket}"

    def _is_text(self, patch, aspect, fg_fraction):
        if not (0.1 < aspect < 10.0) or not (0.01 <= fg_fraction <= 0.85): return False
        edges = cv2.Canny(patch, 50, 150)
        edge_density = np.count_nonzero(edges) / max(patch.size, 1)
        hist = cv2.calcHist([patch], [0], None, [256], [0, 256])
        peaks = np.sum(hist > (np.max(hist) * 0.05))
        return (edge_density > 0.02) and (1 <= peaks <= 15)

    def update(self, bbox, centroid, area, patch, state_changed=False):
        if self.dead: return [], None
        self.age += 1
        self.frames_alive += 1

        if (self.patch_history and patch.shape == self.patch_history[-1].shape
                and np.array_equal(patch, self.patch_history[-1])):
            actual_pixels = self._last_actual_pixels if self._last_actual_pixels is not None else max(1, area)
        else:
            _, binary_patch = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            actual_pixels = np.count_nonzero(binary_patch)
            if actual_pixels == 0: actual_pixels = max(1, area)
            self._last_actual_pixels = actual_pixels
        self.raw_pixels += actual_pixels

        x, y, w, h = bbox
        aspect = w / max(h, 1)
        old_centroid = self.history[-1]['centroid'] if self.history else centroid
        self.history.append({'bbox': bbox, 'centroid': centroid, 'area': area, 'aspect': aspect})
        self.patch_history.append(patch)

        events = []
        if self.age == 1:
            self.lifetime_bytes += 15
            events.append((self.id, "SPAWN", {"x": x, "y": y, "w": w, "h": h}))

        was_locked = self.locked
        self._evaluate_coherence(patch, aspect)

        if self.locked and not was_locked:
            c_type = 0
            if self.classification == 'TEXT': c_type = 1
            elif self.classification == 'STATIC_SHAPE': c_type = 2
            events.append((self.id, "LOCK", {"class": c_type}))

        if self.classification == 'TEXT':
            if not self.has_init_state:
                evts, _ = self._process_state(patch, aspect)
                events.extend(evts)
                self.has_init_state = True
            elif state_changed:
                evts, _ = self._process_state(patch, aspect)
                events.extend(evts)
        elif self.locked and self.classification != 'TEXT':
            dx, dy = centroid[0] - old_centroid[0], centroid[1] - old_centroid[1]
            if abs(dx) > 0.5 or abs(dy) > 0.5:
                self.lifetime_bytes += 3
                events.append((self.id, "MOVE", {"dx": int(dx), "dy": int(dy)}))
                
        if 'binary_patch' not in locals():
            _, binary_patch = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
        return events, binary_patch

    def _evaluate_coherence(self, patch, aspect):
        if len(self.history) < 5: return

        areas = [h['area'] for h in self.history]
        aspects = [h['aspect'] for h in self.history]
        mean_area = max(np.mean(areas), 1e-5)

        score = (0.40 * max(0, 1.0 - (np.std(areas) / mean_area))) + \
                (0.30 * max(0, 1.0 - np.std(aspects))) + \
                (0.30 * min(1.0, self.age / 50.0))

        if self.classification == 'TEXT':
            self.locked = (score > 0.6)
            return

        if score > 0.85:
            self.locked = True
            if self.classification == 'UNKNOWN':
                _, binary = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                fg_fraction = np.count_nonzero(binary) / max(patch.size, 1)
                self.classification = 'TEXT' if self._is_text(patch, aspect, fg_fraction) else 'STATIC_SHAPE'
        else:
            self.locked = False
            if self.classification != 'TEXT': self.classification = 'DYNAMIC'

    def _process_state(self, patch, aspect):
        smoothed = cv2.medianBlur(patch, 3) if min(patch.shape[:2]) >= 3 else patch
        _, binary_patch = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        binary_patch = fill_holes(binary_patch)

        if min(binary_patch.shape[:2]) > 6:
            binary_patch = cv2.morphologyEx(binary_patch, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

        fg_frac = np.count_nonzero(binary_patch) / max(binary_patch.size, 1)
        h_ = self._hash_mask(binary_patch, aspect, fg_frac)

        events = []
        if h_ not in getattr(self, 'state_cache', {}):
            if not hasattr(self, 'state_cache'):
                self.state_cache = {}
                self.state_masks = {}
                self.next_state_id = 0
                self.current_state_id = None
            
            new_state = self.next_state_id
            self.state_cache[h_] = new_state
            self.state_masks[new_state] = binary_patch.copy()
            self.next_state_id += 1
            self.total_distinct_states += 1

            if self.current_state_id is None:
                events.append((self.id, "INIT_STATE", {
                    "state_id": new_state,
                    "mask": binary_patch.copy()
                }))
            else:
                old_state = self.current_state_id
                old_mask = self.state_masks[old_state]
                
                mh = max(old_mask.shape[0], binary_patch.shape[0])
                mw = max(old_mask.shape[1], binary_patch.shape[1])
                
                padded_old = np.zeros((mh, mw), dtype=np.uint8)
                padded_old[:old_mask.shape[0], :old_mask.shape[1]] = old_mask
                
                padded_new = np.zeros((mh, mw), dtype=np.uint8)
                padded_new[:binary_patch.shape[0], :binary_patch.shape[1]] = binary_patch
                
                xor_mask = cv2.bitwise_xor(padded_old, padded_new)
                
                self.total_transitions += 1
                events.append((self.id, "STATE_CHANGE", {
                    "old_state": old_state,
                    "new_state": new_state,
                    "xor_mask": xor_mask,
                    "is_new": True,
                    "new_w": binary_patch.shape[1],
                    "new_h": binary_patch.shape[0]
                }))
            
            self.current_state_id = new_state
            
        else:
            new_state = self.state_cache[h_]
            if self.current_state_id != new_state:
                old_state = self.current_state_id
                self.total_transitions += 1
                events.append((self.id, "STATE_CHANGE", {
                    "old_state": old_state,
                    "new_state": new_state,
                    "xor_mask": None,
                    "is_new": False,
                    "new_w": binary_patch.shape[1],
                    "new_h": binary_patch.shape[0]
                }))
                self.current_state_id = new_state

        return events, binary_patch

class ObjectTracker:
    def __init__(self, min_blob_area=30):
        self.bg_model = None
        self.active_objects = {}
        self.next_id = 1
        self.morph_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.text_refine_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (75, 25))
        self.gap_bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        self.min_blob_area = min_blob_area

    def process_frame(self, curr_bgr, alpha=0.005, threshold=30, cut_threshold=40.0):
        frame_events = []

        curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
        curr_lab = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2LAB)

        if self.bg_model is None:
            self.bg_model = curr_lab.astype(np.float32)
            return [], []

        bg_lab_8u = cv2.convertScaleAbs(self.bg_model)

        mean_full_diff = float(np.mean(cv2.absdiff(curr_lab, bg_lab_8u)))
        if mean_full_diff > cut_threshold:
            self.bg_model = curr_lab.astype(np.float32)
            frame_events.append((0, "SCENE_CUT", f"mean_diff={mean_full_diff:.1f}"))
            return list(self.active_objects.values()), frame_events

        locked_mask = np.zeros_like(curr_gray)
        unlocked_objects = {}
        new_active_objects = {}

        for obj_id, obj in list(self.active_objects.items()):
            if obj.dead: continue
            if obj.locked:
                x, y, w, h = obj.history[-1]['bbox']

                if obj.classification == 'TEXT':
                    pad = 3
                    sx, sy, sw, sh, search_roi = get_safe_roi(curr_gray, x-pad, y-pad, w+2*pad, h+2*pad)
                    if sw > 0 and sh > 0:
                        smoothed_roi = cv2.medianBlur(search_roi, 3) if min(search_roi.shape[:2]) >= 3 else search_roi
                        _, bin_roi = cv2.threshold(smoothed_roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                        bin_roi = fill_holes(bin_roi)
                        bin_roi = cv2.morphologyEx(bin_roi, cv2.MORPH_CLOSE, self.text_refine_close)
                        ro_contours, _ = cv2.findContours(bin_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if ro_contours:
                            areas = [cv2.contourArea(c) for c in ro_contours]
                            max_area = max(areas) if areas else 0.0
                            significant = [c for c, a in zip(ro_contours, areas) if a >= max(2.0, 0.05 * max_area)]
                            if significant:
                                tx, ty, tw, th = cv2.boundingRect(np.vstack(significant))
                                x, y, w, h = sx + tx, sy + ty, tw, th

                cx_safe, cy_safe, cw, ch, curr_patch = get_safe_roi(curr_gray, x, y, w, h)

                if cw == 0 or ch == 0:
                    obj.missing_frames += 1
                    unlocked_objects[obj_id] = obj
                    continue

                curr_color_patch = curr_lab[cy_safe:cy_safe+ch, cx_safe:cx_safe+cw]
                bg_color_patch = bg_lab_8u[cy_safe:cy_safe+ch, cx_safe:cx_safe+cw]
                if np.mean(cv2.absdiff(curr_color_patch, bg_color_patch)) < 10.0:
                    obj.missing_frames += 1
                    unlocked_objects[obj_id] = obj
                    continue

                obj.missing_frames = 0
                if not obj.patch_history:
                    evts = obj.update((cx_safe, cy_safe, cw, ch), obj.history[-1]['centroid'], obj.history[-1]['area'], curr_patch)
                    frame_events.extend(evts)
                    new_active_objects[obj_id] = obj
                    cv2.rectangle(locked_mask, (cx_safe, cy_safe), (cx_safe+cw, cy_safe+ch), 255, -1)
                    continue

                old_patch = obj.patch_history[-1]
                min_h, min_w = min(curr_patch.shape[0], old_patch.shape[0]), min(curr_patch.shape[1], old_patch.shape[1])
                curr_cmp = curr_patch[:min_h, :min_w]
                old_cmp = old_patch[:min_h, :min_w]

                if curr_cmp.shape == old_cmp.shape and np.array_equal(curr_cmp, old_cmp):
                    state_changed = False
                else:
                    _, bin_curr = cv2.threshold(curr_cmp, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    _, bin_old = cv2.threshold(old_cmp, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    xor_mask = cv2.bitwise_xor(bin_curr, bin_old)
                    changed_ratio = np.count_nonzero(xor_mask) / max(1, xor_mask.size)
                    state_changed = changed_ratio > 0.05

                evts, _ = obj.update((cx_safe, cy_safe, cw, ch), (cx_safe + cw//2, cy_safe + ch//2), cw*ch, curr_patch, state_changed)
                frame_events.extend(evts)

                cv2.rectangle(locked_mask, (cx_safe, cy_safe), (cx_safe+cw, cy_safe+ch), 255, -1)
                new_active_objects[obj_id] = obj
            else:
                unlocked_objects[obj_id] = obj

        diff_lab = cv2.absdiff(curr_lab, bg_lab_8u)
        diff_mag = np.max(diff_lab, axis=2)
        diff_mag[locked_mask == 255] = 0

        _, mask = cv2.threshold(diff_mag, threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_open)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, 255, -1)

        safe_bg_mask = cv2.bitwise_not(mask)
        safe_bg_mask[locked_mask == 255] = 0
        cv2.accumulateWeighted(curr_lab, self.bg_model, alpha, mask=safe_bg_mask)

        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        group_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        group_mask = fill_holes(group_mask)
        group_mask = cv2.dilate(group_mask, self.gap_bridge_kernel, iterations=1)
        
        num_labels, labels = cv2.connectedComponents(group_mask, connectivity=8)
        
        merged_rects = []
        for i in range(1, num_labels):
            y_idx, x_idx = np.where((mask > 0) & (labels == i))
            if len(y_idx) > 0:
                xs, ys = x_idx.min(), y_idx.min()
                xe, ye = x_idx.max(), y_idx.max()
                merged_rects.append((int(xs), int(ys), int(xe - xs + 1), int(ye - ys + 1)))

        current_blobs = []
        for x, y, w, h in merged_rects:
            area = w * h
            if area < self.min_blob_area or w < 3 or h < 3: continue
            cx_safe, cy_safe, cw, ch, curr_patch = get_safe_roi(curr_gray, x, y, w, h)
            if cw > 0 and ch > 0:
                current_blobs.append({'bbox': (cx_safe, cy_safe, cw, ch), 'centroid': (x + w//2, y + h//2), 'area': area, 'patch': curr_patch})

        matched_blobs = set()
        matched_objs = set()

        cost_list = []
        for b_idx, blob in enumerate(current_blobs):
            cx, cy = blob['centroid']
            for obj_id, old_obj in unlocked_objects.items():
                if old_obj.dead: continue
                old_cx, old_cy = old_obj.history[-1]['centroid']
                dist = math.hypot(cx - old_cx, cy - old_cy)
                if dist < 50.0: cost_list.append((dist, b_idx, obj_id))

        cost_list.sort(key=lambda x: x[0])

        for dist, b_idx, obj_id in cost_list:
            if b_idx not in matched_blobs and obj_id not in matched_objs:
                p_obj = unlocked_objects[obj_id]
                evts, _ = p_obj.update(current_blobs[b_idx]['bbox'], current_blobs[b_idx]['centroid'], current_blobs[b_idx]['area'], current_blobs[b_idx]['patch'])
                frame_events.extend(evts)
                new_active_objects[obj_id] = p_obj
                matched_blobs.add(b_idx)
                matched_objs.add(obj_id)

        for b_idx, blob in enumerate(current_blobs):
            if b_idx not in matched_blobs:
                new_id = self.next_id
                self.next_id += 1
                p_obj = PersistentObject(new_id)
                evts, _ = p_obj.update(blob['bbox'], blob['centroid'], blob['area'], blob['patch'])
                frame_events.extend(evts)
                new_active_objects[new_id] = p_obj

        for obj_id, obj in unlocked_objects.items():
            if obj_id not in new_active_objects and not obj.dead:
                obj.missing_frames += 1
                if obj.missing_frames > 5:
                    obj.dead = True
                    obj.locked = False
                    obj.classification = 'UNKNOWN'
                    obj.lifetime_bytes += 1
                    frame_events.append((obj_id, "DESPAWN", ""))
                else:
                    new_active_objects[obj_id] = obj
            elif obj_id in new_active_objects:
                new_active_objects[obj_id].missing_frames = 0

        self.active_objects = new_active_objects
        return list(self.active_objects.values()), frame_events

class BinaryEncoder:
    def __init__(self, filepath, width, height):
        self.f = open(filepath, 'wb')
        self.f.write(b'NAM\x01')
        self.f.write(struct.pack('<HH', width, height))
        self.cctx = zstd.ZstdCompressor(level=10)

    def write_spawn(self, obj_id, x, y, w, h):
        self.f.write(struct.pack('<BHHHHH', 0x01, obj_id, x, y, w, h))

    def write_lock(self, obj_id, c_class):
        self.f.write(struct.pack('<BHB', 0x02, obj_id, c_class))

    def write_init_state(self, obj_id, state_id, mask):
        packed = np.packbits(mask > 0).tobytes()
        compressed = self.cctx.compress(packed)
        self.f.write(struct.pack('<BHHII', 0x03, obj_id, state_id, mask.shape[1], mask.shape[0]))
        self.f.write(struct.pack('<I', len(compressed)))
        self.f.write(compressed)

    def write_state_change(self, obj_id, old_state, new_state, xor_mask=None, new_w=0, new_h=0):
        if xor_mask is not None:
            packed = np.packbits(xor_mask > 0).tobytes()
            compressed = self.cctx.compress(packed)
            self.f.write(struct.pack('<BHHHII', 0x04, obj_id, old_state, new_state, new_w, new_h))
            self.f.write(struct.pack('<I', len(compressed)))
            self.f.write(compressed)
        else:
            self.f.write(struct.pack('<BHHHII', 0x04, obj_id, old_state, new_state, new_w, new_h))
            self.f.write(struct.pack('<I', 0))

    def write_move(self, obj_id, dx, dy):
        dx_b = max(-128, min(127, dx))
        dy_b = max(-128, min(127, dy))
        self.f.write(struct.pack('<BHbb', 0x05, obj_id, dx_b, dy_b))

    def write_despawn(self, obj_id):
        self.f.write(struct.pack('<BH', 0x06, obj_id))

    def write_frame_end(self):
        self.f.write(struct.pack('<B', 0xFF))
        
    def close(self):
        self.f.close()


def run_encoder(video_path, out_path, config=None):
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video")
        return 0, {}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    tracker = ObjectTracker()
    encoder = BinaryEncoder(out_path, width, height)
    
    stats = {'spawns': 0, 'moves': 0, 'transitions': 0, 'despawns': 0, 'locks': 0}
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        objects, events = tracker.process_frame(frame)
        
        for evt in events:
            obj_id, evt_type, detail = evt
            if evt_type == "SPAWN":
                encoder.write_spawn(obj_id, detail['x'], detail['y'], detail['w'], detail['h'])
                stats['spawns'] += 1
            elif evt_type == "LOCK":
                encoder.write_lock(obj_id, detail['class'])
                stats['locks'] += 1
            elif evt_type == "INIT_STATE":
                encoder.write_init_state(obj_id, detail['state_id'], detail['mask'])
                stats['transitions'] += 1
            elif evt_type == "STATE_CHANGE":
                encoder.write_state_change(
                    obj_id, 
                    detail['old_state'], 
                    detail['new_state'], 
                    detail.get('xor_mask'),
                    detail.get('new_w', 0),
                    detail.get('new_h', 0)
                )
                stats['transitions'] += 1
            elif evt_type == "MOVE":
                encoder.write_move(obj_id, detail['dx'], detail['dy'])
                stats['moves'] += 1
            elif evt_type == "DESPAWN":
                encoder.write_despawn(obj_id)
                stats['despawns'] += 1
        
        encoder.write_frame_end()
        frame_idx += 1

    encoder.close()
    cap.release()
    enc_time = time.time() - start_time
    return enc_time, stats

def encode_video(video_path, out_path):
    run_encoder(video_path, out_path)
    print(f"Successfully encoded {video_path} to {out_path}!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python encoder.py <input.mp4> <output.nam>")
        sys.exit(1)
    encode_video(sys.argv[1], sys.argv[2])
