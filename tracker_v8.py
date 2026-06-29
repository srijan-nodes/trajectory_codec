import cv2
import sys
import numpy as np
import hashlib
from collections import deque

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
    h, w = binary.shape[:2]
    flood = binary.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    return binary | flood_inv

def vertical_projection_split(binary_patch, min_glyph_width=2, min_gap=1):
    if binary_patch.size == 0: return []
    col_sums = np.count_nonzero(binary_patch, axis=0)
    w = len(col_sums)
    cols_fg = col_sums > 0
    raw_segments = []
    start = None
    for i in range(w):
        if cols_fg[i] and start is None: start = i
        elif not cols_fg[i] and start is not None:
            raw_segments.append((start, i))
            start = None
    if start is not None: raw_segments.append((start, w))
    merged = []
    for seg in raw_segments:
        if merged and seg[0] - merged[-1][1] <= min_gap:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(list(seg))
    return [tuple(s) for s in merged if (s[1] - s[0]) >= min_glyph_width]

class PersistentObject:
    def __init__(self, obj_id):
        self.id = obj_id
        self.history = deque(maxlen=10)
        self.patch_history = deque(maxlen=10)
        self.state_cache = {}
        self.state_masks = {}
        self.next_state_id = 0
        self.current_state_id = None
        self.micro_objects = {}
        self.total_distinct_states = 0
        self.total_transitions = 0
        self.locked = False
        self.classification = 'UNKNOWN'
        self.age = 0
        self.missing_frames = 0
        self.frames_alive = 0
        self.lifetime_bytes = 0
        self.raw_pixels = 0
        self._last_actual_pixels = None 
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

    def update(self, bbox, centroid, area, patch, color_patch, state_changed=False):
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
                evts, _ = self._process_state(patch, color_patch, aspect)
                events.extend(evts)
                self.has_init_state = True
            elif state_changed:
                evts, _ = self._process_state(patch, color_patch, aspect)
                events.extend(evts)
        elif self.locked and self.classification != 'TEXT':
            _, binary_patch = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
            fg_frac = np.count_nonzero(binary_patch) / max(binary_patch.size, 1)
            h_ = self._hash_mask(binary_patch, aspect, fg_frac)
            
            if h_ not in self.state_cache:
                new_state = self.next_state_id
                self.state_cache[h_] = new_state
                self.state_masks[new_state] = binary_patch.copy()
                self.next_state_id += 1
                self.total_distinct_states += 1
                
                if self.current_state_id is None:
                    events.append((self.id, "INIT_STATE", {
                        "state_id": new_state,
                        "mask": binary_patch.copy(),
                        "color_patch": color_patch.copy()
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
                        "new_w": binary_patch.shape[1],
                        "new_h": binary_patch.shape[0]
                    }))
                self.current_state_id = new_state

        if self.locked:
            old_bbox = self.history[-1]['bbox'] if self.history else bbox
            dx, dy = bbox[0] - old_bbox[0], bbox[1] - old_bbox[1]
            if abs(dx) > 0 or abs(dy) > 0:
                self.lifetime_bytes += 3
                events.append((self.id, "MOVE", {"dx": int(dx), "dy": int(dy)}))
                
        if 'binary_patch' not in locals():
            _, binary_patch = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
        return events, binary_patch

    def _evaluate_coherence(self, patch, aspect):
        if len(self.history) < 5: return
        hist_list = list(self.history)
        ious = [bb_iou(h['bbox'], hist_list[-1]['bbox']) for h in hist_list[-5:-1]]
        score = np.mean(ious)
        areas = [h['area'] for h in hist_list]
        aspects = [h['aspect'] for h in hist_list]
        mean_area = max(np.mean(areas), 1e-5)
        centers = [h['centroid'] for h in hist_list[-5:]]
        dx = max([c[0] for c in centers]) - min([c[0] for c in centers])
        dy = max([c[1] for c in centers]) - min([c[1] for c in centers])

        if score > 0.85 or (dx <= 2 and dy <= 2) or self.classification == 'TEXT':
            self.locked = True
            if self.classification != 'TEXT':
                _, binary = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                fg_fraction = np.count_nonzero(binary) / max(patch.size, 1)
                self.classification = 'TEXT' if self._is_text(patch, aspect, fg_fraction) else 'STATIC_SHAPE'
        else:
            self.locked = False
            if self.classification != 'TEXT': self.classification = 'DYNAMIC'

    def _process_state(self, patch, color_patch, aspect):
        smoothed = cv2.medianBlur(patch, 3) if min(patch.shape[:2]) >= 3 else patch
        _, binary_patch = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        binary_patch = fill_holes(binary_patch)
        if min(binary_patch.shape[:2]) > 6:
            binary_patch = cv2.morphologyEx(binary_patch, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        segments = vertical_projection_split(binary_patch, min_glyph_width=2, min_gap=1)
        if not segments:
            segments = [(0, binary_patch.shape[1])]

        events = []
        active_micros = set()

        for m_idx, (sx1, sx2) in enumerate(segments):
            micro_patch = binary_patch[:, sx1:sx2]
            row_sums = np.count_nonzero(micro_patch, axis=1)
            fg_rows = np.where(row_sums > 0)[0]
            if len(fg_rows) == 0: continue
            
            sy1, sy2 = fg_rows[0], fg_rows[-1] + 1
            micro_patch = micro_patch[sy1:sy2, :]
            active_micros.add(m_idx)
            micro_color_patch = color_patch[sy1:sy2, sx1:sx2]

            micro_aspect = micro_patch.shape[1] / max(micro_patch.shape[0], 1)
            fg_frac = np.count_nonzero(micro_patch) / max(micro_patch.size, 1)
            h_ = self._hash_mask(micro_patch, micro_aspect, fg_frac)

            if m_idx not in self.micro_objects:
                self.micro_objects[m_idx] = {'state_cache': {}, 'state_masks': {}, 'next_state_id': 0, 'current_state_id': None}

            mo = self.micro_objects[m_idx]
            if h_ not in mo['state_cache']:
                new_state = mo['next_state_id']
                mo['state_cache'][h_] = new_state
                mo['state_masks'][new_state] = micro_patch.copy()
                mo['next_state_id'] += 1
                self.total_distinct_states += 1

                if mo['current_state_id'] is None:
                    events.append((self.id, "INIT_MICRO_STATE", {
                        "micro_idx": m_idx, "state_id": new_state, "offset_x": sx1, "offset_y": sy1,
                        "mask": micro_patch.copy(), "color_patch": micro_color_patch.copy()
                    }))
                else:
                    old_state = mo['current_state_id']
                    old_mask = mo['state_masks'][old_state]
                    mh, mw = max(old_mask.shape[0], micro_patch.shape[0]), max(old_mask.shape[1], micro_patch.shape[1])
                    
                    padded_old = np.zeros((mh, mw), dtype=np.uint8)
                    padded_old[:old_mask.shape[0], :old_mask.shape[1]] = old_mask
                    padded_new = np.zeros((mh, mw), dtype=np.uint8)
                    padded_new[:micro_patch.shape[0], :micro_patch.shape[1]] = micro_patch
                    
                    xor_mask = cv2.bitwise_xor(padded_old, padded_new)
                    self.total_transitions += 1
                    events.append((self.id, "MICRO_STATE_CHANGE", {
                        "micro_idx": m_idx, "old_state": old_state, "new_state": new_state,
                        "offset_x": sx1, "offset_y": sy1, "xor_mask": xor_mask, "is_new": True,
                        "new_w": micro_patch.shape[1], "new_h": micro_patch.shape[0]
                    }))
                mo['current_state_id'] = new_state
            else:
                new_state = mo['state_cache'][h_]
                if mo['current_state_id'] != new_state:
                    old_state = mo['current_state_id']
                    self.total_transitions += 1
                    events.append((self.id, "MICRO_STATE_CHANGE", {
                        "micro_idx": m_idx, "old_state": old_state, "new_state": new_state,
                        "offset_x": sx1, "offset_y": sy1, "xor_mask": None, "is_new": False,
                        "new_w": micro_patch.shape[1], "new_h": micro_patch.shape[0]
                    }))
                    mo['current_state_id'] = new_state
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
                    margin = 15
                    sx, sy = max(0, int(x - margin)), max(0, int(y - margin))
                    sw, sh = min(curr_gray.shape[1] - sx, int(w + 2*margin)), min(curr_gray.shape[0] - sy, int(h + 2*margin))
                    if sw > 0 and sh > 0:
                        bg_patch = bg_lab_8u[sy:sy+sh, sx:sx+sw]
                        curr_lab_patch = curr_lab[sy:sy+sh, sx:sx+sw]
                        patch_diff = np.max(cv2.absdiff(curr_lab_patch, bg_patch), axis=2)
                        _, bin_search = cv2.threshold(patch_diff, threshold, 255, cv2.THRESH_BINARY)
                        bin_search = cv2.morphologyEx(bin_search, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
                        
                        col_sums = np.sum(bin_search, axis=0)
                        row_sums = np.sum(bin_search, axis=1)
                        fg_cols = np.where(col_sums > 0)[0]
                        fg_rows = np.where(row_sums > 0)[0]
                        
                        if len(fg_cols) > 0 and len(fg_rows) > 0:
                            x = int(sx + fg_cols[0])
                            y = int(sy + fg_rows[0])
                            w = int(fg_cols[-1] - fg_cols[0] + 1)
                            h = int(fg_rows[-1] - fg_rows[0] + 1)

                cx_safe, cy_safe, cw, ch, curr_patch = get_safe_roi(curr_gray, x, y, w, h)

                if cw == 0 or ch == 0:
                    obj.missing_frames += 1
                    unlocked_objects[obj_id] = obj
                    continue

                curr_color_patch = curr_lab[cy_safe:cy_safe+ch, cx_safe:cx_safe+cw]
                curr_bgr_patch = curr_bgr[cy_safe:cy_safe+ch, cx_safe:cx_safe+cw]
                bg_color_patch = bg_lab_8u[cy_safe:cy_safe+ch, cx_safe:cx_safe+cw]
                
                obj.missing_frames = 0
                if not obj.patch_history:
                    evts, _ = obj.update((cx_safe, cy_safe, cw, ch), obj.history[-1]['centroid'], obj.history[-1]['area'], curr_patch, curr_bgr_patch)
                    frame_events.extend(evts)
                    new_active_objects[obj_id] = obj
                    pad_m = 20
                    mx1, my1 = max(0, cx_safe - pad_m), max(0, cy_safe - pad_m)
                    mx2, my2 = min(curr_gray.shape[1], cx_safe + cw + pad_m), min(curr_gray.shape[0], cy_safe + ch + pad_m)
                    cv2.rectangle(locked_mask, (mx1, my1), (mx2, my2), 255, -1)
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

                evts, _ = obj.update((cx_safe, cy_safe, cw, ch), (cx_safe + cw//2, cy_safe + ch//2), cw*ch, curr_patch, curr_bgr_patch, state_changed)
                frame_events.extend(evts)

                pad_m = 20
                mx1, my1 = max(0, cx_safe - pad_m), max(0, cy_safe - pad_m)
                mx2, my2 = min(curr_gray.shape[1], cx_safe + cw + pad_m), min(curr_gray.shape[0], cy_safe + ch + pad_m)
                cv2.rectangle(locked_mask, (mx1, my1), (mx2, my2), 255, -1)
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
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(group_mask, connectivity=8)
        merged_rects = []
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            merged_rects.append((x, y, w, h))

        current_blobs = []
        for x, y, w, h in merged_rects:
            area = w * h
            if area < self.min_blob_area or w < 3 or h < 3: continue
            current_blobs.append({
                'bbox': (x, y, w, h), 
                'centroid': (x + w//2, y + h//2), 
                'area': area, 
                'patch': get_safe_roi(curr_gray, x, y, w, h)[4],
                'color_patch': get_safe_roi(curr_bgr, x, y, w, h)[4]
            })

        matched_blobs = set()
        matched_objs = set()

        cost_list = []
        for b_idx, blob in enumerate(current_blobs):
            for obj_id, old_obj in unlocked_objects.items():
                if old_obj.dead: continue
                dist = np.hypot(blob['centroid'][0] - old_obj.history[-1]['centroid'][0],
                                blob['centroid'][1] - old_obj.history[-1]['centroid'][1])
                if dist < 50: cost_list.append((dist, b_idx, obj_id))

        cost_list.sort(key=lambda x: x[0])

        for iou, b_idx, obj_id in cost_list:
            if b_idx not in matched_blobs and obj_id not in matched_objs:
                p_obj = unlocked_objects[obj_id]
                evts, _ = p_obj.update(
                    current_blobs[b_idx]['bbox'], 
                    current_blobs[b_idx]['centroid'], 
                    current_blobs[b_idx]['area'], 
                    current_blobs[b_idx]['patch'],
                    current_blobs[b_idx]['color_patch']
                )
                frame_events.extend(evts)
                new_active_objects[obj_id] = p_obj
                matched_blobs.add(b_idx)
                matched_objs.add(obj_id)

        for b_idx, blob in enumerate(current_blobs):
            if b_idx not in matched_blobs:
                new_id = self.next_id
                self.next_id += 1
                p_obj = PersistentObject(new_id)
                evts, _ = p_obj.update(blob['bbox'], blob['centroid'], blob['area'], blob['patch'], blob['color_patch'])
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


def visualize_tracker(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    tracker = ObjectTracker()
    print(f"Visualizing tracker on {video_path}...")
    cv2.namedWindow("SOBVC Tracker Live View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SOBVC Tracker Live View", 1280, 720)
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        objects, events = tracker.process_frame(frame)
        
        for obj in objects:
            if obj.dead:
                continue
            
            x, y, w_box, h_box = obj.history[-1]['bbox']
            
            if obj.locked:
                if obj.classification == 'TEXT':
                    color = (0, 255, 255)
                elif obj.classification == 'STATIC_SHAPE':
                    color = (0, 255, 0)
                else:
                    color = (0, 165, 255)
            else:
                color = (255, 0, 0)
                
            cv2.rectangle(frame, (int(x), int(y)), (int(x+w_box), int(y+h_box)), color, 2)
            cv2.putText(frame, f"ID:{obj.id} {obj.classification}", (int(x), int(y)-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow("SOBVC Tracker Live View", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == '__main__':
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    video = filedialog.askopenfilename(title="Select Target Codec Test Sequence")
        
    visualize_tracker(video)