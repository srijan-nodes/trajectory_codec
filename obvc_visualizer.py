import cv2
import numpy as np
import math
import hashlib
import os
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
    """Standard flood-fill hole-filling so a ring-shaped glyph (e.g. '0')
    is treated as one solid blob rather than breaking into fragments."""
    h, w = binary.shape[:2]
    flood = binary.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    return binary | flood_inv

def vertical_projection_split(binary_patch, min_glyph_width=2, min_gap=1):
    """Splits a binary text patch into glyph-sized column segments using a
    vertical projection histogram. A colon's two stacked dots share the same
    column range, so they fall out as a single segment naturally."""
    if binary_patch.size == 0:
        return []
    col_sums = np.count_nonzero(binary_patch, axis=0)
    w = len(col_sums)
    cols_fg = col_sums > 0

    raw_segments = []
    start = None
    for i in range(w):
        if cols_fg[i] and start is None:
            start = i
        elif not cols_fg[i] and start is not None:
            raw_segments.append((start, i))
            start = None
    if start is not None:
        raw_segments.append((start, w))

    merged = []
    for seg in raw_segments:
        if merged and seg[0] - merged[-1][1] <= min_gap:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(list(seg))

    return [tuple(s) for s in merged if (s[1] - s[0]) >= min_glyph_width]

def fallback_split_wide_segments(segments, width_ratio_trigger=1.6):
    """Tightly-kerned synthetic fonts can leave zero-width gaps between glyphs,
    so vertical_projection_split sometimes returns one fused segment for two+
    characters. Detect segments that are anomalously wide relative to the
    median glyph width in this region and force-split them into equal parts."""
    if len(segments) < 2:
        return segments

    widths = [b - a for a, b in segments]
    median_w = float(np.median(widths))
    if median_w <= 0:
        return segments

    refined = []
    for (a, b) in segments:
        w = b - a
        ratio = w / median_w
        if ratio > width_ratio_trigger:
            n = max(2, int(round(ratio)))
            step = w / n
            for k in range(n):
                seg_a = int(round(a + k * step))
                seg_b = int(round(a + (k + 1) * step))
                if seg_b > seg_a:
                    refined.append((seg_a, seg_b))
        else:
            refined.append((a, b))
    return refined

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
        if self.dead: return []
        self.age += 1
        self.frames_alive += 1

        # Exact-match fast path: synthetic renders frequently reproduce byte-identical
        # patches frame to frame for an unchanged glyph. Skip the Otsu threshold and
        # reuse the cached foreground pixel count instead of recomputing it.
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
            events.append((self.id, "SPAWN", ""))

        was_locked = self.locked
        self._evaluate_coherence(patch, aspect)

        if self.locked and not was_locked:
            events.append((self.id, "LOCK", self.classification))

        if self.classification == 'TEXT':
            if not self.has_init_state:
                self._process_state(patch, aspect)
                self.has_init_state = True
                events.append((self.id, "INIT_STATE", f"glyphs={len(self.glyph_slots)}"))
            elif state_changed:
                evts, _ = self._process_state(patch, aspect)
                events.extend(evts)
        elif self.locked and self.classification != 'TEXT':
            dx, dy = centroid[0] - old_centroid[0], centroid[1] - old_centroid[1]
            if abs(dx) > 0.5 or abs(dy) > 0.5:
                self.lifetime_bytes += 3
                events.append((self.id, "MOVE", f"({dx:+.1f}, {dy:+.1f})"))
        return events

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
        """Glyph-level state tracking. Splits the text patch into column
        segments (one per glyph/character) and tracks each segment's state
        independently, so e.g. '00:03' -> '00:02' only reports a change for
        the one glyph slot that actually changed."""
        # Anti-aliasing-robust preprocessing: a median blur removes the dithered/
        # blended fringe pixels synthetic renderers put on glyph edges, before
        # Otsu ever sees them -- this prevents sub-pixel AA jitter from flipping
        # a handful of border pixels and producing a different hash for what is
        # logically the same glyph.
        smoothed = cv2.medianBlur(patch, 3) if min(patch.shape[:2]) >= 3 else patch
        _, binary_patch = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        binary_patch = fill_holes(binary_patch)

        # Light morphological opening as a second AA-robustness pass, but only
        # when the patch is large enough that we won't erase thin strokes or
        # small glyphs (colon dots, periods, thin clock-digit segments).
        if min(binary_patch.shape[:2]) > 6:
            binary_patch = cv2.morphologyEx(binary_patch, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

        segments = vertical_projection_split(binary_patch)
        segments = fallback_split_wide_segments(segments)
        events = []

        if len(segments) != len(self.glyph_slots):
            # Glyph count changed (or first time seeing this object) -- re-init slots.
            self.glyph_slots = [{'cache': {}, 'next_id': 0, 'current': None} for _ in segments]
            for idx, (x0, x1) in enumerate(segments):
                glyph_mask = binary_patch[:, x0:x1]
                g_aspect = (x1 - x0) / max(binary_patch.shape[0], 1)
                g_fg = np.count_nonzero(glyph_mask) / max(glyph_mask.size, 1)
                h_ = self._hash_mask(glyph_mask, g_aspect, g_fg)
                slot = self.glyph_slots[idx]
                slot['cache'][h_] = 0
                slot['next_id'] = 1
                slot['current'] = 0
                self.total_distinct_states += 1
            if self.has_init_state:
                events.append((self.id, "RESEGMENT", f"glyphs={len(segments)}"))
            return events, binary_patch

        for idx, (x0, x1) in enumerate(segments):
            glyph_mask = binary_patch[:, x0:x1]
            g_aspect = (x1 - x0) / max(binary_patch.shape[0], 1)
            g_fg = np.count_nonzero(glyph_mask) / max(glyph_mask.size, 1)
            h_ = self._hash_mask(glyph_mask, g_aspect, g_fg)
            slot = self.glyph_slots[idx]

            if h_ not in slot['cache']:
                slot['cache'][h_] = slot['next_id']
                slot['next_id'] += 1
                self.total_distinct_states += 1

            new_state = slot['cache'][h_]
            if slot['current'] is not None and slot['current'] != new_state:
                self.total_transitions += 1
                self.lifetime_bytes += 3
                events.append((self.id, "GLYPH_STATE", f"slot{idx}:{slot['current']}->{new_state}"))
            slot['current'] = new_state

        return events, binary_patch

class ObjectTracker:
    def __init__(self, min_blob_area=30):
        # Background model is kept in Lab color space (float32) so color-only
        # changes at constant luminance -- common in synthetic UI elements --
        # are actually visible to the diff, not just grayscale brightness shifts.
        self.bg_model = None
        self.active_objects = {}
        self.next_id = 1
        self.morph_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.text_refine_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # Small dilation used only to GROUP disconnected fragments of the same
        # glyph (see Tier 2 below) -- not applied to the real content mask.
        self.gap_bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # Decorative background motion (confetti, sparkle/glitter particles in
        # synthetic NYE-style footage) is often small enough to clear this
        # threshold and floods next_id with hundreds of short-lived objects.
        # This is a separate, deliberate knob -- raise it if you don't care
        # about tracking decorative particles and only want UI/text objects.
        self.min_blob_area = min_blob_area

    def process_frame(self, curr_bgr, alpha=0.005, threshold=30, cut_threshold=40.0):
        frame_events = []

        curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
        curr_lab = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2LAB)

        if self.bg_model is None:
            self.bg_model = curr_lab.astype(np.float32)
            return [], []

        bg_lab_8u = cv2.convertScaleAbs(self.bg_model)

        # --- HARD-CUT DETECTION ---
        # Synthetic video can hard-cut to an entirely new scene with no transition.
        # The accumulateWeighted background model assumes slow drift; without this
        # check, a cut produces a frame-spanning diff that Tier 2 extraction reads
        # as one giant blob (or a "blob explosion" over the next several frames
        # while alpha slowly drags the background model into place). Detect the
        # jump and snap the background to the new frame immediately instead.
        mean_full_diff = float(np.mean(cv2.absdiff(curr_lab, bg_lab_8u)))
        if mean_full_diff > cut_threshold:
            self.bg_model = curr_lab.astype(np.float32)
            frame_events.append((0, "SCENE_CUT", f"mean_diff={mean_full_diff:.1f}"))
            return list(self.active_objects.values()), frame_events

        locked_mask = np.zeros_like(curr_gray)
        unlocked_objects = {}
        new_active_objects = {}

        # --- TIER 1: Locked Object Direct Routing ---
        for obj_id, obj in list(self.active_objects.items()):
            if obj.dead: continue
            if obj.locked:
                x, y, w, h = obj.history[-1]['bbox']

                # Dynamic Bounding Box Refinement for TEXT
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

                # Color-aware "has this object actually disappeared into the
                # background" check, instead of grayscale-only absdiff.
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

                # Exact-match fast path: skip Otsu + XOR entirely when the patch is
                # byte-identical to last frame (extremely common for static synthetic
                # glyphs/UI between logical state changes).
                if curr_cmp.shape == old_cmp.shape and np.array_equal(curr_cmp, old_cmp):
                    state_changed = False
                else:
                    _, bin_curr = cv2.threshold(curr_cmp, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    _, bin_old = cv2.threshold(old_cmp, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    xor_mask = cv2.bitwise_xor(bin_curr, bin_old)
                    changed_ratio = np.count_nonzero(xor_mask) / max(1, xor_mask.size)
                    state_changed = changed_ratio > 0.05

                evts = obj.update((cx_safe, cy_safe, cw, ch), (cx_safe + cw//2, cy_safe + ch//2), cw*ch, curr_patch, state_changed)
                frame_events.extend(evts)

                cv2.rectangle(locked_mask, (cx_safe, cy_safe), (cx_safe+cw, cy_safe+ch), 255, -1)
                new_active_objects[obj_id] = obj
            else:
                unlocked_objects[obj_id] = obj

        # --- TIER 2: Extraction (color-aware) ---
        # Use the max absolute difference across L/a/b channels so a change that
        # only shows up in chroma (a/b) -- e.g. a same-brightness color swap on a
        # UI icon -- still triggers extraction, not just luminance shifts.
        diff_lab = cv2.absdiff(curr_lab, bg_lab_8u)
        diff_mag = np.max(diff_lab, axis=2)
        diff_mag[locked_mask == 255] = 0

        _, mask = cv2.threshold(diff_mag, threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_open)
        # MORPH_CLOSE intentionally omitted here -- closing at this stage bridges
        # gaps between adjacent glyphs/digits and merges them into one blob.

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, 255, -1)

        safe_bg_mask = cv2.bitwise_not(mask)
        safe_bg_mask[locked_mask == 255] = 0
        cv2.accumulateWeighted(curr_lab, self.bg_model, alpha, mask=safe_bg_mask)

        # Fragment grouping: anti-aliased curved strokes (e.g. the top/bottom
        # arcs of a "0") can fall just under the diff threshold and split what
        # is visually one glyph into multiple disconnected components. A small
        # dilation -- applied only to a throwaway copy used for grouping, never
        # to the real content mask -- closes those few-pixel AA gaps so the
        # fragments get reunited into one bounding box. The dilation radius is
        # deliberately small so genuine inter-glyph/inter-word spacing (almost
        # always wider) is not bridged. Tune self.gap_bridge_kernel if your
        # font/render still fragments or if separate glyphs start re-merging.
        raw_rects = [cv2.boundingRect(cnt) for cnt in contours]
        group_mask = cv2.dilate(mask, self.gap_bridge_kernel, iterations=1)
        group_contours, _ = cv2.findContours(group_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        merged_rects = []
        assigned = [False] * len(raw_rects)
        for gcontour in group_contours:
            members = []
            for i, r in enumerate(raw_rects):
                if assigned[i]: continue
                rx, ry, rw, rh = r
                cx, cy = rx + rw / 2.0, ry + rh / 2.0
                if cv2.pointPolygonTest(gcontour, (cx, cy), False) >= 0:
                    members.append(r)
                    assigned[i] = True
            if members:
                xs = [m[0] for m in members]; ys = [m[1] for m in members]
                xe = [m[0] + m[2] for m in members]; ye = [m[1] + m[3] for m in members]
                merged_rects.append((min(xs), min(ys), max(xe) - min(xs), max(ye) - min(ys)))

        # Safety net: any raw rect a grouping contour didn't claim (rounding
        # edge cases) still gets through as its own region instead of vanishing.
        for i, r in enumerate(raw_rects):
            if not assigned[i]:
                merged_rects.append(r)

        current_blobs = []
        for x, y, w, h in merged_rects:
            area = w * h
            if area < self.min_blob_area or w < 3 or h < 3: continue
            cx_safe, cy_safe, cw, ch, curr_patch = get_safe_roi(curr_gray, x, y, w, h)
            if cw > 0 and ch > 0:
                current_blobs.append({'bbox': (cx_safe, cy_safe, cw, ch), 'centroid': (x + w//2, y + h//2), 'area': area, 'patch': curr_patch})

        # --- TIER 3: Global Cost Matrix (Unlocked Objects Only) ---
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
                evts = p_obj.update(current_blobs[b_idx]['bbox'], current_blobs[b_idx]['centroid'], current_blobs[b_idx]['area'], current_blobs[b_idx]['patch'])
                frame_events.extend(evts)
                new_active_objects[obj_id] = p_obj
                matched_blobs.add(b_idx)
                matched_objs.add(obj_id)

        for b_idx, blob in enumerate(current_blobs):
            if b_idx not in matched_blobs:
                new_id = self.next_id
                self.next_id += 1
                p_obj = PersistentObject(new_id)
                frame_events.extend(p_obj.update(blob['bbox'], blob['centroid'], blob['area'], blob['patch']))
                new_active_objects[new_id] = p_obj

        # Manage Despawns & Metrics
        for obj_id, obj in unlocked_objects.items():
            if obj_id not in new_active_objects and not obj.dead:
                obj.missing_frames += 1
                if obj.missing_frames > 5:
                    obj.dead = True
                    obj.locked = False
                    obj.classification = 'UNKNOWN'
                    obj.lifetime_bytes += 1
                    frame_events.append((obj_id, "DESPAWN", ""))

                    compression = max(0, 100.0 * (1.0 - (obj.lifetime_bytes / max(obj.raw_pixels, 1))))
                    frame_events.append((obj_id, "METRICS", f"Alive:{obj.frames_alive}f | States:{obj.total_distinct_states} | Trans:{obj.total_transitions} | Payload:{obj.lifetime_bytes}B | Comp:{compression:.1f}%"))
                else:
                    new_active_objects[obj_id] = obj
            elif obj_id in new_active_objects:
                new_active_objects[obj_id].missing_frames = 0

        self.active_objects = new_active_objects
        return list(self.active_objects.values()), frame_events

def run_visualizer(video_path):
    print(f"\n🔬 Booting V8 OBVC Telemetry Profiler (synthetic-video tuned): {os.path.basename(video_path)}\n" + "="*60)
    cap = cv2.VideoCapture(video_path)
    tracker = ObjectTracker()
    stats = {'frames': 0, 'spawns': 0, 'moves': 0, 'transitions': 0, 'despawns': 0, 'resegments': 0, 'cuts': 0}

    cv2.namedWindow("OBVC State Machine", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret: break
        stats['frames'] += 1

        objects, events = tracker.process_frame(frame)

        if events:
            print(f"\n[FRAME {stats['frames']:04d}]")
            for evt in events:
                obj_id, evt_type, detail = evt
                print(f"  OBJ {obj_id:<3} | {evt_type:<10} | {detail}")
                if evt_type == "SPAWN": stats['spawns'] += 1
                elif evt_type == "MOVE": stats['moves'] += 1
                elif evt_type == "GLYPH_STATE": stats['transitions'] += 1
                elif evt_type == "RESEGMENT": stats['resegments'] += 1
                elif evt_type == "DESPAWN": stats['despawns'] += 1
                elif evt_type == "SCENE_CUT": stats['cuts'] += 1

        for obj in objects:
            if obj.missing_frames > 0 or obj.dead: continue
            x, y, w, h = obj.history[-1]['bbox']
            color = (0, 255, 0) if obj.locked else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"ID:{obj.id}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow("OBVC State Machine", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

    est_payload = (stats['spawns']*15) + (stats['moves']*3) + (stats['transitions']*3) + (stats['despawns']*1)
    print("\n" + "="*60)
    print(f"📊 V8 TELEMETRY REPORT: {os.path.basename(video_path)}")
    print("="*60)
    print(f"Total Frames Processed : {stats['frames']}")
    print(f"Total Scene Cuts        : {stats['cuts']}")
    print(f"Total Object Spawns    : {stats['spawns']}")
    print(f"Total Object Moves     : {stats['moves']}")
    print(f"Total Glyph Transitions: {stats['transitions']}")
    print(f"Total Resegments       : {stats['resegments']}")
    print(f"Total Despawns         : {stats['despawns']}")
    print("-" * 60)
    print(f"Estimated Event Payload: {est_payload / 1024:.2f} KB")
    print(f"Estimated Byte/Frame   : {est_payload / max(1, stats['frames']):.1f} bytes")
    print("="*60 + "\n")

if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    video = filedialog.askopenfilename(title="Select Video", filetypes=[("Video", "*.mp4 *.y4m")])
    if video: run_visualizer(video)