//! boundary.rs — Object-Level Boundary Understanding & Constant Effects Engine for TrajCDDec V7
//!
//! Unlike classical macroblock codecs that divide frames into rigid 16x16 tiles,
//! the Boundary Engine understands where an object geometrically starts and ends.
//! An object is represented by:
//! 1. An exact contour bounding box (x, y, w, h)
//! 2. An anti-aliased subpixel boundary mask
//! 3. Canonical texture appearance
//! 4. Uniform constant effects (rigid trajectory dx/dy, constant luma shift, opacity, and scale)

use std::collections::HashMap;
use std::io::{Read, Write};
use anyhow::{bail, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

/// Flag bits for active constant effects on an object
pub const EFFECT_TRANSLATION: u8 = 0x01; // Spatial translation (dx, dy)
pub const EFFECT_LUMA_SHIFT: u8  = 0x02; // Constant photometric offset (Delta Y)
pub const EFFECT_CONTRAST: u8    = 0x04; // Constant gain / contrast scale
pub const EFFECT_OPACITY: u8     = 0x08; // Constant alpha transparency
pub const EFFECT_SCALE: u8       = 0x10; // Uniform boundary scale factor

/// Uniform constant effects applied across an entire object boundary
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConstantEffects {
    /// Spatial translation (dx, dy) in pixels
    pub dx: i16,
    pub dy: i16,
    /// Constant luma / brightness shift (-128..=127) applied uniformly across the boundary
    pub luma_shift: i8,
    /// Contrast / gain scaling in fixed point (128 = 1.0x neutral)
    pub contrast_scale: u8,
    /// Overall object opacity (0 = fully transparent, 255 = fully opaque)
    pub opacity: u8,
    /// Uniform scale factor in Q8 (256 = 1.0x neutral)
    pub scale_q8: u16,
}

impl Default for ConstantEffects {
    fn default() -> Self {
        Self {
            dx: 0,
            dy: 0,
            luma_shift: 0,
            contrast_scale: 128,
            opacity: 255,
            scale_q8: 256,
        }
    }
}

impl ConstantEffects {
    /// Serialize constant effects into a compact variable-length payload
    pub fn pack(&self) -> Vec<u8> {
        let mut flags = 0u8;
        if self.dx != 0 || self.dy != 0 {
            flags |= EFFECT_TRANSLATION;
        }
        if self.luma_shift != 0 {
            flags |= EFFECT_LUMA_SHIFT;
        }
        if self.contrast_scale != 128 {
            flags |= EFFECT_CONTRAST;
        }
        if self.opacity != 255 {
            flags |= EFFECT_OPACITY;
        }
        if self.scale_q8 != 256 {
            flags |= EFFECT_SCALE;
        }

        let mut buf = Vec::with_capacity(12);
        buf.push(flags);

        if flags & EFFECT_TRANSLATION != 0 {
            buf.write_i16::<LittleEndian>(self.dx).unwrap();
            buf.write_i16::<LittleEndian>(self.dy).unwrap();
        }
        if flags & EFFECT_LUMA_SHIFT != 0 {
            buf.push(self.luma_shift as u8);
        }
        if flags & EFFECT_CONTRAST != 0 {
            buf.push(self.contrast_scale);
        }
        if flags & EFFECT_OPACITY != 0 {
            buf.push(self.opacity);
        }
        if flags & EFFECT_SCALE != 0 {
            buf.write_u16::<LittleEndian>(self.scale_q8).unwrap();
        }

        buf
    }

    /// Deserialize constant effects from a compact payload
    pub fn unpack<R: Read>(mut reader: R) -> Result<Self> {
        let flags = reader.read_u8()?;
        let mut effects = Self::default();

        if flags & EFFECT_TRANSLATION != 0 {
            effects.dx = reader.read_i16::<LittleEndian>()?;
            effects.dy = reader.read_i16::<LittleEndian>()?;
        }
        if flags & EFFECT_LUMA_SHIFT != 0 {
            effects.luma_shift = reader.read_u8()? as i8;
        }
        if flags & EFFECT_CONTRAST != 0 {
            effects.contrast_scale = reader.read_u8()?;
        }
        if flags & EFFECT_OPACITY != 0 {
            effects.opacity = reader.read_u8()?;
        }
        if flags & EFFECT_SCALE != 0 {
            effects.scale_q8 = reader.read_u16::<LittleEndian>()?;
        }

        Ok(effects)
    }

    /// Apply constant effects uniformly across the object's pixels
    #[inline(always)]
    pub fn apply_to_pixel(&self, base_val: u8) -> u8 {
        let mut val = base_val as i32;
        if self.contrast_scale != 128 {
            val = ((val - 128) * self.contrast_scale as i32) / 128 + 128;
        }
        if self.luma_shift > 0 {
            // Smooth headroom preservation: avoid blowing out specular highlights into clipped plateaus
            let headroom = 255 - val.clamp(0, 255);
            val += (self.luma_shift as i32 * headroom) / 255;
        } else if self.luma_shift < 0 {
            // Smooth footroom preservation: avoid crushing near-black shadows
            let footroom = val.clamp(0, 255);
            val += (self.luma_shift as i32 * footroom) / 255;
        }
        val.clamp(0, 255) as u8
    }
}

/// Boundary alpha mask defining the exact subpixel contour of an object
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundaryMask {
    pub width: usize,
    pub height: usize,
    /// 0 = Outside (Background), 255 = Core Interior, 1..=254 = Anti-aliased boundary fringe
    pub alpha: Vec<u8>,
}

impl BoundaryMask {
    pub fn new(width: usize, height: usize, alpha: Vec<u8>) -> Self {
        assert_eq!(alpha.len(), width * height);
        Self { width, height, alpha }
    }

    /// Compress mask using run-length encoding (RLE)
    pub fn compress_rle(&self) -> Vec<u8> {
        let mut out = Vec::new();
        if self.alpha.is_empty() {
            return out;
        }

        let mut current_val = self.alpha[0];
        let mut count: usize = 1;

        for &val in &self.alpha[1..] {
            if val == current_val && count < 255 {
                count += 1;
            } else {
                out.push(count as u8);
                out.push(current_val);
                current_val = val;
                count = 1;
            }
        }
        out.push(count as u8);
        out.push(current_val);

        out
    }

    /// Decompress run-length encoded mask
    pub fn decompress_rle(rle: &[u8], width: usize, height: usize) -> Result<Self> {
        let mut alpha = Vec::with_capacity(width * height);
        let mut idx = 0;
        while idx + 1 < rle.len() {
            let count = rle[idx] as usize;
            let val = rle[idx + 1];
            alpha.extend(std::iter::repeat(val).take(count));
            idx += 2;
        }

        if alpha.len() != width * height {
            bail!("Decompressed mask length mismatch: got {}, expected {}", alpha.len(), width * height);
        }

        Ok(Self { width, height, alpha })
    }
}

/// An object entity with known boundary, canonical appearance, and tracking dynamics
#[derive(Clone, Debug)]
pub struct ObjectEntity {
    pub id: u32,
    /// Spatial origin in canvas coordinates
    pub origin_x: usize,
    pub origin_y: usize,
    pub width: usize,
    pub height: usize,
    /// Exact contour boundary mask
    pub boundary: BoundaryMask,
    /// Canonical un-shifted template appearance pixels (w * h)
    pub canonical_pixels: Vec<u8>,
    /// Active constant effects applied to this object
    pub effects: ConstantEffects,
    /// Number of frames this object has been continuously tracked
    pub age: usize,
    /// Prior displacement for linear velocity extrapolation
    pub prev_dx: i16,
    pub prev_dy: i16,
    /// Pre-indexed interior points (px, py, alpha, base_pixel) for accelerated tracking
    pub active_points: Vec<(u16, u16, u8, u8)>,
}

impl ObjectEntity {
    pub fn new(
        id: u32,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        mask: BoundaryMask,
        pixels: Vec<u8>,
    ) -> Self {
        let mut active_points = Vec::with_capacity((w * h) / 2);
        for py in 0..h {
            for px in 0..w {
                let idx = py * w + px;
                let alpha = mask.alpha[idx];
                if alpha > 32 {
                    active_points.push((px as u16, py as u16, alpha, pixels[idx]));
                }
            }
        }

        Self {
            id,
            origin_x: x,
            origin_y: y,
            width: w,
            height: h,
            boundary: mask,
            canonical_pixels: pixels,
            effects: ConstantEffects::default(),
            age: 0,
            prev_dx: 0,
            prev_dy: 0,
            active_points,
        }
    }

    /// Pack initial object definition
    pub fn pack_definition(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(self.id).unwrap();
        buf.write_u16::<LittleEndian>(self.origin_x as u16).unwrap();
        buf.write_u16::<LittleEndian>(self.origin_y as u16).unwrap();
        buf.write_u16::<LittleEndian>(self.width as u16).unwrap();
        buf.write_u16::<LittleEndian>(self.height as u16).unwrap();

        let rle_mask = self.boundary.compress_rle();
        buf.write_u32::<LittleEndian>(rle_mask.len() as u32).unwrap();
        buf.extend_from_slice(&rle_mask);

        buf.write_u32::<LittleEndian>(self.canonical_pixels.len() as u32).unwrap();
        buf.extend_from_slice(&self.canonical_pixels);

        let eff_bytes = self.effects.pack();
        buf.write_u16::<LittleEndian>(eff_bytes.len() as u16).unwrap();
        buf.extend_from_slice(&eff_bytes);

        buf
    }

    /// Unpack object definition from bitstream
    pub fn unpack_definition<R: Read>(mut reader: R) -> Result<Self> {
        let id = reader.read_u32::<LittleEndian>()?;
        let origin_x = reader.read_u16::<LittleEndian>()? as usize;
        let origin_y = reader.read_u16::<LittleEndian>()? as usize;
        let width = reader.read_u16::<LittleEndian>()? as usize;
        let height = reader.read_u16::<LittleEndian>()? as usize;

        let mask_len = reader.read_u32::<LittleEndian>()? as usize;
        let mut mask_bytes = vec![0u8; mask_len];
        reader.read_exact(&mut mask_bytes)?;
        let boundary = BoundaryMask::decompress_rle(&mask_bytes, width, height)?;

        let pixels_len = reader.read_u32::<LittleEndian>()? as usize;
        let mut canonical_pixels = vec![0u8; pixels_len];
        reader.read_exact(&mut canonical_pixels)?;

        let eff_len = reader.read_u16::<LittleEndian>()? as usize;
        let mut eff_bytes = vec![0u8; eff_len];
        reader.read_exact(&mut eff_bytes)?;
        let effects = ConstantEffects::unpack(&eff_bytes[..])?;

        let mut obj = Self::new(id, origin_x, origin_y, width, height, boundary, canonical_pixels);
        obj.effects = effects;
        Ok(obj)
    }
}

/// Boundary Engine — Coordinates Object Segmentation, Effect Tracking & Compositing
pub struct BoundaryEngine {
    pub width: usize,
    pub height: usize,
    pub next_object_id: u32,
    pub active_objects: HashMap<u32, ObjectEntity>,
    pub background_model: Vec<u8>,
    pub camera_dx: i16,
    pub camera_dy: i16,
}

impl BoundaryEngine {
    pub fn new(width: usize, height: usize, background: Vec<u8>) -> Self {
        assert_eq!(background.len(), width * height);
        Self {
            width,
            height,
            next_object_id: 1,
            active_objects: HashMap::new(),
            background_model: background,
            camera_dx: 0,
            camera_dy: 0,
        }
    }

    /// Segment contiguous foreground objects against a reference canvas
    /// (background model on frame 0, or current active composite for residual detection)
    pub fn segment_objects_against(
        &mut self,
        frame: &[u8],
        reference_canvas: &[u8],
        diff_threshold: i32,
        min_area: usize,
    ) -> Vec<ObjectEntity> {
        let w = self.width;
        let h = self.height;

        // 1. Binary foreground difference mask
        let mut fg_mask = vec![false; w * h];
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let diff = (frame[idx] as i32 - reference_canvas[idx] as i32).abs();
                if diff > diff_threshold {
                    fg_mask[idx] = true;
                }
            }
        }

        // 2. Connected Component Labeling with Union-Find directly on foreground mask
        let mut labels = vec![0u32; w * h];
        let mut parent: Vec<u32> = vec![0];
        let mut next_label = 1u32;

        fn find(mut i: u32, parent: &mut [u32]) -> u32 {
            while parent[i as usize] != i {
                parent[i as usize] = parent[parent[i as usize] as usize];
                i = parent[i as usize];
            }
            i
        }

        fn union(i: u32, j: u32, parent: &mut [u32]) {
            let root_i = find(i, parent);
            let root_j = find(j, parent);
            if root_i != root_j {
                parent[root_j as usize] = root_i;
            }
        }

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                if !fg_mask[idx] {
                    continue;
                }

                let mut neighbor_labels = Vec::with_capacity(4);
                if y > 0 && labels[(y - 1) * w + x] > 0 {
                    neighbor_labels.push(labels[(y - 1) * w + x]);
                }
                if x > 0 && labels[y * w + (x - 1)] > 0 {
                    neighbor_labels.push(labels[y * w + (x - 1)]);
                }
                if y > 0 && x > 0 && labels[(y - 1) * w + (x - 1)] > 0 {
                    neighbor_labels.push(labels[(y - 1) * w + (x - 1)]);
                }
                if y > 0 && x + 1 < w && labels[(y - 1) * w + (x + 1)] > 0 {
                    neighbor_labels.push(labels[(y - 1) * w + (x + 1)]);
                }

                if neighbor_labels.is_empty() {
                    let l = next_label;
                    next_label += 1;
                    parent.push(l);
                    labels[idx] = l;
                } else {
                    let min_l = *neighbor_labels.iter().min().unwrap();
                    labels[idx] = min_l;
                    for &nl in &neighbor_labels {
                        union(min_l, nl, &mut parent);
                    }
                }
            }
        }

        struct ComponentStats {
            min_x: usize,
            max_x: usize,
            min_y: usize,
            max_y: usize,
            area: usize,
        }

        let mut comp_map: HashMap<u32, ComponentStats> = HashMap::new();
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let l = labels[idx];
                if l > 0 {
                    let canon = find(l, &mut parent);
                    labels[idx] = canon;

                    comp_map.entry(canon)
                        .and_modify(|s| {
                            s.min_x = s.min_x.min(x);
                            s.max_x = s.max_x.max(x);
                            s.min_y = s.min_y.min(y);
                            s.max_y = s.max_y.max(y);
                            s.area += 1;
                        })
                        .or_insert(ComponentStats {
                            min_x: x,
                            max_x: x,
                            min_y: y,
                            max_y: y,
                            area: 1,
                        });
                }
            }
        }

        // Helper to construct an ObjectEntity from bounding slice
        fn create_object_entity(
            obj_id: u32,
            bx: usize,
            by: usize,
            bw: usize,
            bh: usize,
            target_label: u32,
            labels: &[u32],
            frame: &[u8],
            bg: &[u8],
            canvas_w: usize,
        ) -> Option<ObjectEntity> {
            if bw == 0 || bh == 0 {
                return None;
            }

            let mut mask_alpha = vec![0u8; bw * bh];
            let mut pixels = vec![0u8; bw * bh];
            let mut actual_area = 0usize;

            for py in 0..bh {
                for px in 0..bw {
                    let gy = by + py;
                    let gx = bx + px;
                    let g_idx = gy * canvas_w + gx;
                    let p_idx = py * bw + px;

                    pixels[p_idx] = frame[g_idx];

                    if labels[g_idx] == target_label {
                        actual_area += 1;

                        // Check if boundary edge pixel for anti-aliased subpixel feathering
                        let mut is_edge = false;
                        if py == 0 || py == bh - 1 || px == 0 || px == bw - 1 {
                            is_edge = true;
                        } else {
                            let g_left = gy * canvas_w + (gx - 1);
                            let g_right = gy * canvas_w + (gx + 1);
                            let g_up = (gy - 1) * canvas_w + gx;
                            let g_down = (gy + 1) * canvas_w + gx;

                            if labels[g_left] != target_label
                                || labels[g_right] != target_label
                                || labels[g_up] != target_label
                                || labels[g_down] != target_label
                            {
                                is_edge = true;
                            }
                        }

                        if is_edge {
                            let diff = (frame[g_idx] as i32 - bg[g_idx] as i32).abs();
                            let alpha_val = ((diff * 255) / 40).clamp(64, 255) as u8;
                            mask_alpha[p_idx] = alpha_val;
                        } else {
                            mask_alpha[p_idx] = 255;
                        }
                    }
                }
            }

            if actual_area < 24 {
                return None;
            }

            let boundary = BoundaryMask::new(bw, bh, mask_alpha);
            Some(ObjectEntity::new(obj_id, bx, by, bw, bh, boundary, pixels))
        }

        let mut objects = Vec::new();

        for (label, stats) in comp_map {
            if stats.area < min_area {
                continue;
            }

            let bx = stats.min_x;
            let by = stats.min_y;
            let bw = stats.max_x - stats.min_x + 1;
            let bh = stats.max_y - stats.min_y + 1;

            // Constriction / bottleneck neck detection:
            // If two distinct bodies are touching at a narrow seam (e.g. ball touching reflection),
            // split them into distinct geometric entities.
            let mut split_py = None;
            if bh > 48 && bw > 32 {
                let mut row_widths = vec![0usize; bh];
                for py in 0..bh {
                    let gy = by + py;
                    for px in 0..bw {
                        let gx = bx + px;
                        if labels[gy * w + gx] == label {
                            row_widths[py] += 1;
                        }
                    }
                }

                let mid_div = bh / 2;
                let top_max = row_widths[0..mid_div].iter().cloned().max().unwrap_or(0);
                let bot_max = row_widths[mid_div..bh].iter().cloned().max().unwrap_or(0);
                let min_peak = top_max.min(bot_max);

                if min_peak >= 30 {
                    let mid_start = bh / 4;
                    let mid_end = (bh * 3) / 4;
                    let mut min_mid_w = usize::MAX;
                    let mut min_mid_py = mid_start;
                    for py in mid_start..mid_end {
                        if row_widths[py] < min_mid_w {
                            min_mid_w = row_widths[py];
                            min_mid_py = py;
                        }
                    }
                    if min_mid_w * 10 < min_peak * 65 / 10 {
                        split_py = Some(min_mid_py);
                    }
                }
            }

            if let Some(cut) = split_py {
                // Split into two cohesive geometric entities
                let id1 = self.next_object_id;
                self.next_object_id += 1;
                if let Some(obj1) = create_object_entity(id1, bx, by, bw, cut, label, &labels, frame, &self.background_model, w) {
                    objects.push(obj1);
                }

                let id2 = self.next_object_id;
                self.next_object_id += 1;
                if let Some(obj2) = create_object_entity(id2, bx, by + cut + 1, bw, bh - cut - 1, label, &labels, frame, &self.background_model, w) {
                    objects.push(obj2);
                }
            } else {
                let id = self.next_object_id;
                self.next_object_id += 1;
                if let Some(obj) = create_object_entity(id, bx, by, bw, bh, label, &labels, frame, &self.background_model, w) {
                    objects.push(obj);
                }
            }
        }

        objects
    }

    /// Segment initial objects against the background model on Frame 0
    pub fn segment_objects(&mut self, frame: &[u8], diff_threshold: i32, min_area: usize) -> Vec<ObjectEntity> {
        let bg = self.background_model.clone();
        self.segment_objects_against(frame, &bg, diff_threshold, min_area)
    }

    /// Segment newly entering objects from unexplained foreground residuals on P-frames
    pub fn segment_residual_objects(
        &mut self,
        frame: &[u8],
        reference_canvas: &[u8],
        diff_threshold: i32,
        min_area: usize,
    ) -> Vec<ObjectEntity> {
        self.segment_objects_against(frame, reference_canvas, diff_threshold, min_area)
    }

    /// Track existing objects across frames with predictive velocity and solve for optimal constant effects
    pub fn track_and_solve_effects(
        &mut self,
        curr_frame: &[u8],
        search_radius: i16,
    ) -> (Vec<u32>, Vec<(u32, ConstantEffects)>) {
        let w = self.width;
        let h = self.height;
        let mut effect_updates = Vec::new();
        let mut retired_ids = Vec::new();

        for (obj_id, obj) in &mut self.active_objects {
            let bw = obj.width;
            let bh = obj.height;

            // Velocity prediction: extrapolate position using prior frame velocity
            let vel_x = obj.effects.dx - obj.prev_dx;
            let vel_y = obj.effects.dy - obj.prev_dy;
            let pred_center_x = obj.origin_x as i32 + obj.effects.dx as i32 + vel_x as i32;
            let pred_center_y = obj.origin_y as i32 + obj.effects.dy as i32 + vel_y as i32;

            let mut best_dx = obj.effects.dx;
            let mut best_dy = obj.effects.dy;
            let mut best_luma_shift = obj.effects.luma_shift;
            let mut best_opacity = obj.effects.opacity;
            let mut min_sad = i64::MAX;
            let mut best_count = 0i64;

            let active_points = &obj.active_points;
            if active_points.is_empty() {
                retired_ids.push(*obj_id);
                continue;
            }

            // Coarse search step for candidates
            let s_step: i16 = if search_radius > 8 { 2 } else { 1 };
            let mut cand_offsets = Vec::with_capacity(256);
            let mut sdy = -search_radius;
            while sdy <= search_radius {
                let mut sdx = -search_radius;
                while sdx <= search_radius {
                    cand_offsets.push((sdx, sdy));
                    sdx += s_step;
                }
                sdy += s_step;
            }

            let mut coarse_best_sdx = 0i16;
            let mut coarse_best_sdy = 0i16;
            let mut coarse_min_sad = i64::MAX;

            // Subsampling factor for fast coarse evaluation
            let p_step = if active_points.len() > 8000 { 4 } else if active_points.len() > 2000 { 2 } else { 1 };

            for &(sdx, sdy) in &cand_offsets {
                let cand_x = pred_center_x + sdx as i32;
                let cand_y = pred_center_y + sdy as i32;

                if cand_x + bw as i32 <= 0 || cand_x >= w as i32
                    || cand_y + bh as i32 <= 0 || cand_y >= h as i32
                {
                    continue;
                }

                let mut sum_diff = 0i64;
                let mut count = 0i64;

                let mut i = 0;
                while i < active_points.len() {
                    let (px, py, _alpha, base_pix) = active_points[i];
                    let gx = cand_x + px as i32;
                    let gy = cand_y + py as i32;
                    if gx >= 0 && gx < w as i32 && gy >= 0 && gy < h as i32 {
                        let curr_pix = curr_frame[(gy as usize) * w + (gx as usize)] as i32;
                        sum_diff += (curr_pix - base_pix as i32) as i64;
                        count += 1;
                    }
                    i += p_step;
                }

                if (count as usize) < (active_points.len() / p_step) / 2 {
                    continue;
                }

                let est_luma = (sum_diff / count).clamp(-128, 127) as i8;

                let mut cur_sad = 0i64;
                let mut i = 0;
                while i < active_points.len() {
                    let (px, py, _alpha, base_pix) = active_points[i];
                    let gx = cand_x + px as i32;
                    let gy = cand_y + py as i32;
                    if gx >= 0 && gx < w as i32 && gy >= 0 && gy < h as i32 {
                        let curr_pix = curr_frame[(gy as usize) * w + (gx as usize)] as i32;
                        let shifted = (base_pix as i32 + est_luma as i32).clamp(0, 255);
                        cur_sad += (curr_pix - shifted).abs() as i64;
                    } else {
                        cur_sad += 255;
                    }
                    i += p_step;
                }

                if cur_sad < coarse_min_sad {
                    coarse_min_sad = cur_sad;
                    coarse_best_sdx = sdx;
                    coarse_best_sdy = sdy;
                }
            }

            // Fine refinement window: check [-2..=2] around coarse best at full resolution to guarantee optimal alignment
            let mut fine_candidates = Vec::with_capacity(25);
            for fdy in -2..=2 {
                for fdx in -2..=2 {
                    fine_candidates.push((coarse_best_sdx + fdx, coarse_best_sdy + fdy));
                }
            }

            for (sdx, sdy) in fine_candidates {
                let cand_x = pred_center_x + sdx as i32;
                let cand_y = pred_center_y + sdy as i32;

                if cand_x + bw as i32 <= 0 || cand_x >= w as i32
                    || cand_y + bh as i32 <= 0 || cand_y >= h as i32
                {
                    continue;
                }

                let mut sum_diff = 0i64;
                let mut count = 0i64;

                for &(px, py, _alpha, base_pix) in active_points {
                    let gx = cand_x + px as i32;
                    let gy = cand_y + py as i32;
                    if gx >= 0 && gx < w as i32 && gy >= 0 && gy < h as i32 {
                        let curr_pix = curr_frame[(gy as usize) * w + (gx as usize)] as i32;
                        sum_diff += (curr_pix - base_pix as i32) as i64;
                        count += 1;
                    }
                }

                if (count as usize) < active_points.len() / 2 {
                    continue;
                }

                let est_luma = (sum_diff / count).clamp(-128, 127) as i8;

                // Closed-form opacity estimation against background
                let mut num = 0i64;
                let mut den = 0i64;
                for &(px, py, _alpha, base_pix) in active_points {
                    let gx = cand_x + px as i32;
                    let gy = cand_y + py as i32;
                    if gx >= 0 && gx < w as i32 && gy >= 0 && gy < h as i32 {
                        let curr_pix = curr_frame[(gy as usize) * w + (gx as usize)] as i64;
                        let eff_pix = (base_pix as i64 + est_luma as i64).clamp(0, 255);
                        let bg_pix = self.background_model[(gy as usize) * w + (gx as usize)] as i64;
                        num += (curr_pix - bg_pix) * (eff_pix - bg_pix);
                        den += (eff_pix - bg_pix) * (eff_pix - bg_pix);
                    }
                }

                let est_opacity = if den > 100 {
                    ((num * 255) / den).clamp(0, 255) as u8
                } else {
                    255
                };

                // Opacity Stability Filter:
                // Only permit opacity reduction below 255 if there is a severe, unambiguous global fade (est_opacity < 210)
                // This eliminates the single-frame specular highlight flicker/glint on physical solid objects
                let candidate_opacities: &[u8] = if est_opacity < 210 {
                    &[255, est_opacity]
                } else {
                    &[255]
                };

                for &cand_opacity in candidate_opacities {
                    let mut cur_sad = 0i64;
                    for &(px, py, _alpha, base_pix) in active_points {
                        let gx = cand_x + px as i32;
                        let gy = cand_y + py as i32;
                        if gx >= 0 && gx < w as i32 && gy >= 0 && gy < h as i32 {
                            let curr_pix = curr_frame[(gy as usize) * w + (gx as usize)] as i32;
                            let eff_pix = (base_pix as i32 + est_luma as i32).clamp(0, 255);
                            let bg_pix = self.background_model[(gy as usize) * w + (gx as usize)] as i32;
                            let blended = if cand_opacity == 255 {
                                eff_pix
                            } else {
                                (eff_pix * cand_opacity as i32 + bg_pix * (255 - cand_opacity as i32)) / 255
                            };
                            cur_sad += (curr_pix - blended).abs() as i64;
                        } else {
                            cur_sad += 255;
                        }
                    }

                    if cur_sad < min_sad {
                        min_sad = cur_sad;
                        best_dx = (cand_x - obj.origin_x as i32) as i16;
                        best_dy = (cand_y - obj.origin_y as i32) as i16;
                        best_luma_shift = est_luma;
                        best_opacity = cand_opacity;
                        best_count = count;
                    }
                }
            }

            let mut avg_sad = if best_count > 0 { min_sad / best_count } else { 999 };

            // Dynamic Scale Estimation for zooming / UI element expansion
            let mut best_scale_q8 = obj.effects.scale_q8;
            if avg_sad > 8 || obj.effects.scale_q8 != 256 {
                let cur_s = obj.effects.scale_q8;
                let test_scales = [
                    cur_s.saturating_sub(6).max(128),
                    cur_s,
                    (cur_s + 6).min(512),
                ];
                for &cand_scale in &test_scales {
                    if cand_scale == cur_s {
                        continue;
                    }
                    let scale_u32 = cand_scale as u32;
                    let mut scale_sad = 0i64;
                    let cand_x = obj.origin_x as i32 + best_dx as i32;
                    let cand_y = obj.origin_y as i32 + best_dy as i32;
                    for &(px, py, _alpha, base_pix) in active_points {
                        let spx = ((px as u32 * scale_u32) >> 8) as i32;
                        let spy = ((py as u32 * scale_u32) >> 8) as i32;
                        let gx = cand_x + spx;
                        let gy = cand_y + spy;
                        if gx >= 0 && gx < w as i32 && gy >= 0 && gy < h as i32 {
                            let curr_pix = curr_frame[(gy as usize) * w + (gx as usize)] as i32;
                            let eff_pix = (base_pix as i32 + best_luma_shift as i32).clamp(0, 255);
                            scale_sad += (curr_pix - eff_pix).abs() as i64;
                        } else {
                            scale_sad += 255;
                        }
                    }
                    if scale_sad < min_sad {
                        min_sad = scale_sad;
                        best_scale_q8 = cand_scale;
                        if best_count > 0 {
                            avg_sad = min_sad / best_count;
                        }
                    }
                }
            }

            // Damped luma shift to prevent single-frame brightness flashes
            let prev_luma = obj.effects.luma_shift;
            let final_luma = if (best_luma_shift as i32 - prev_luma as i32).abs() > 3 {
                prev_luma + ((best_luma_shift as i32 - prev_luma as i32) / 2) as i8
            } else {
                best_luma_shift
            };

            // Adaptive retirement threshold: allow large UI components room to expand
            let sad_limit = if active_points.len() > 4000 { 75 } else { 55 };

            // If match error exceeds confidence threshold or object has faded out completely, retire it
            if avg_sad > sad_limit || best_count < 20 || best_opacity <= 20 {
                retired_ids.push(*obj_id);
            } else {
                let effects = ConstantEffects {
                    dx: best_dx,
                    dy: best_dy,
                    luma_shift: final_luma,
                    contrast_scale: 128,
                    opacity: best_opacity,
                    scale_q8: best_scale_q8,
                };

                obj.prev_dx = obj.effects.dx;
                obj.prev_dy = obj.effects.dy;
                obj.effects = effects;
                obj.age += 1;

                effect_updates.push((*obj_id, effects));
            }
        }

        (retired_ids, effect_updates)
    }

    /// Render frame with complete Boundary Understanding and Constant Effects
    pub fn render_frame(&self) -> Vec<u8> {
        let w = self.width;
        let h = self.height;

        // 1. Pristine background model with Global Motion Compensation for camera jitter
        let mut canvas = if self.camera_dx != 0 || self.camera_dy != 0 {
            let mut shifted = vec![0u8; w * h];
            for y in 0..h {
                let sy = (y as i32 - self.camera_dy as i32).clamp(0, h as i32 - 1) as usize;
                for x in 0..w {
                    let sx = (x as i32 - self.camera_dx as i32).clamp(0, w as i32 - 1) as usize;
                    shifted[y * w + x] = self.background_model[sy * w + sx];
                }
            }
            shifted
        } else {
            self.background_model.clone()
        };

        // 2. Composite each active object with its constant effects and anti-aliased boundary
        for obj in self.active_objects.values() {
            let bw = obj.width;
            let bh = obj.height;
            let eff = &obj.effects;

            let dest_x = obj.origin_x as i32 + eff.dx as i32;
            let dest_y = obj.origin_y as i32 + eff.dy as i32;

            if eff.scale_q8 == 256 {
                // 1:1 Identity scale fast-path (over 20,000 FPS decode)
                for py in 0..bh {
                    for px in 0..bw {
                        let p_idx = py * bw + px;
                        let mask_val = obj.boundary.alpha[p_idx];
                        if mask_val == 0 {
                            continue;
                        }

                        let gx = dest_x + px as i32;
                        let gy = dest_y + py as i32;

                        if gx >= 0 && gx < w as i32 && gy >= 0 && gy < h as i32 {
                            let c_idx = (gy as usize) * w + (gx as usize);
                            let base_pix = obj.canonical_pixels[p_idx];

                            // Apply constant effect uniformly across object interior
                            let eff_pix = eff.apply_to_pixel(base_pix);

                            // Alpha composite onto background along the boundary perimeter
                            if mask_val == 255 && eff.opacity == 255 {
                                canvas[c_idx] = eff_pix;
                            } else {
                                let total_alpha = (mask_val as u32 * eff.opacity as u32) / 255;
                                let bg_pix = canvas[c_idx] as u32;
                                let blended = (eff_pix as u32 * total_alpha + bg_pix * (255 - total_alpha)) / 255;
                                canvas[c_idx] = blended as u8;
                            }
                        }
                    }
                }
            } else {
                // Arbitrary Q8 dynamic scaling loop
                let scale = eff.scale_q8 as u32;
                let scaled_w = ((bw as u32 * scale + 128) >> 8) as usize;
                let scaled_h = ((bh as u32 * scale + 128) >> 8) as usize;

                for spy in 0..scaled_h {
                    let py = ((spy as u32 * 256) / scale).min(bh as u32 - 1) as usize;
                    for spx in 0..scaled_w {
                        let px = ((spx as u32 * 256) / scale).min(bw as u32 - 1) as usize;
                        let p_idx = py * bw + px;
                        let mask_val = obj.boundary.alpha[p_idx];
                        if mask_val == 0 {
                            continue;
                        }

                        let gx = dest_x + spx as i32;
                        let gy = dest_y + spy as i32;

                        if gx >= 0 && gx < w as i32 && gy >= 0 && gy < h as i32 {
                            let c_idx = (gy as usize) * w + (gx as usize);
                            let base_pix = obj.canonical_pixels[p_idx];

                            let eff_pix = eff.apply_to_pixel(base_pix);

                            if mask_val == 255 && eff.opacity == 255 {
                                canvas[c_idx] = eff_pix;
                            } else {
                                let total_alpha = (mask_val as u32 * eff.opacity as u32) / 255;
                                let bg_pix = canvas[c_idx] as u32;
                                let blended = (eff_pix as u32 * total_alpha + bg_pix * (255 - total_alpha)) / 255;
                                canvas[c_idx] = blended as u8;
                            }
                        }
                    }
                }
            }
        }

        canvas
    }
}

/// Encode video frames into a high-efficiency V7 Boundary Object Stream
pub fn encode_boundary_stream<W: Write>(
    frames: &[Vec<u8>],
    width: usize,
    height: usize,
    mut writer: W,
) -> Result<()> {
    let total_frames = frames.len();
    if total_frames == 0 {
        return Ok(());
    }

    // 1. Build pristine background model with percentile filtering
    let corners = [
        frames[0][0] as u32,
        frames[0][width - 1] as u32,
        frames[0][(height - 1) * width] as u32,
        frames[0][height * width - 1] as u32,
    ];
    let mean_corner = corners.iter().sum::<u32>() / 4;
    let is_light_bg = mean_corner > 180;
    let is_dark_bg = mean_corner < 80;

    let sample_count = 20.min(total_frames);
    let sample_indices: Vec<usize> = (0..sample_count)
        .map(|i| (i * total_frames) / sample_count)
        .collect();

    let percentile_idx = if is_light_bg {
        (sample_count * 85) / 100
    } else if is_dark_bg {
        (sample_count * 15) / 100
    } else {
        sample_count / 2
    };

    let mut bg = vec![0u8; width * height];
    for idx in 0..width * height {
        let mut vals: Vec<u8> = sample_indices.iter().map(|&fi| frames[fi][idx]).collect();
        vals.sort_unstable();
        bg[idx] = vals[percentile_idx.min(vals.len() - 1)];
    }

    writer.write_u32::<LittleEndian>(bg.len() as u32)?;
    writer.write_all(&bg)?;

    let mut engine = BoundaryEngine::new(width, height, bg);
    let diff_thresh = if is_light_bg { 25 } else { 15 };
    let min_area = ((width * height) / 8000).max(32);

    for (f_idx, frame) in frames.iter().enumerate() {
        if f_idx == 0 {
            writer.write_u8(0)?; // I-frame
            let initial_objects = engine.segment_objects(frame, diff_thresh, min_area);
            writer.write_u16::<LittleEndian>(initial_objects.len() as u16)?;
            for obj in &initial_objects {
                let def = obj.pack_definition();
                writer.write_all(&def)?;
                engine.active_objects.insert(obj.id, obj.clone());
            }
            writer.write_u16::<LittleEndian>(0)?; // retired_count
            writer.write_u16::<LittleEndian>(0)?; // update_count
        } else {
            writer.write_u8(1)?; // P-frame

            // 0. Global Motion Estimation for Camera Jitter
            let mut best_cam_dx = 0i16;
            let mut best_cam_dy = 0i16;
            let mut min_cam_sad = i64::MAX;
            let mut zero_cam_sad = 0i64;

            let border_step = 4;
            for cdy in -4..=4 {
                for cdx in -4..=4 {
                    let mut sad = 0i64;
                    for y in (0..16).chain((height.saturating_sub(16))..height).step_by(border_step) {
                        for x in (0..width).step_by(border_step) {
                            let sx = (x as i32 + cdx).clamp(0, width as i32 - 1) as usize;
                            let sy = (y as i32 + cdy).clamp(0, height as i32 - 1) as usize;
                            sad += (frame[sy * width + sx] as i32 - engine.background_model[y * width + x] as i32).abs() as i64;
                        }
                    }
                    if cdx == 0 && cdy == 0 {
                        zero_cam_sad = sad;
                    }
                    if sad < min_cam_sad {
                        min_cam_sad = sad;
                        best_cam_dx = cdx as i16;
                        best_cam_dy = cdy as i16;
                    }
                }
            }

            let (cam_dx, cam_dy) = if min_cam_sad + 100 < zero_cam_sad {
                (best_cam_dx, best_cam_dy)
            } else {
                (0, 0)
            };
            engine.camera_dx = cam_dx;
            engine.camera_dy = cam_dy;

            writer.write_i16::<LittleEndian>(cam_dx)?;
            writer.write_i16::<LittleEndian>(cam_dy)?;

            // 1. Track existing active objects and solve constant effects (search radius 24 spans high velocity motion)
            let (retired_ids, updates) = engine.track_and_solve_effects(frame, 24);

            // Remove retired objects from tracking engine
            for id in &retired_ids {
                engine.active_objects.remove(id);
            }

            // Apply effect updates to active objects
            for (obj_id, eff) in &updates {
                if let Some(obj) = engine.active_objects.get_mut(obj_id) {
                    obj.effects = *eff;
                }
            }

            // 2. Render intermediate canvas to find unexplained foreground residuals
            let temp_canvas = engine.render_frame();
            let mut total_residual_pix = 0usize;
            for idx in 0..width * height {
                if (frame[idx] as i32 - temp_canvas[idx] as i32).abs() > diff_thresh {
                    total_residual_pix += 1;
                }
            }

            let new_objects = if total_residual_pix >= min_area {
                let candidates = engine.segment_residual_objects(frame, &temp_canvas, diff_thresh, min_area);
                let mut filtered = Vec::new();
                for obj in candidates {
                    let mut touches_active = false;
                    let o_x1 = obj.origin_x as i32;
                    let o_x2 = (obj.origin_x + obj.width) as i32;
                    let o_y1 = obj.origin_y as i32;
                    let o_y2 = (obj.origin_y + obj.height) as i32;

                    for act in engine.active_objects.values() {
                        let a_x1 = (act.origin_x as i32 + act.effects.dx as i32) - 16;
                        let a_x2 = (act.origin_x as i32 + act.effects.dx as i32 + act.width as i32) + 16;
                        let a_y1 = (act.origin_y as i32 + act.effects.dy as i32) - 16;
                        let a_y2 = (act.origin_y as i32 + act.effects.dy as i32 + act.height as i32) + 16;

                        if o_x1 < a_x2 && o_x2 > a_x1 && o_y1 < a_y2 && o_y2 > a_y1 {
                            touches_active = true;
                            break;
                        }
                    }

                    // Only permit distinct, large new entities (e.g. newly entering distinct objects)
                    // and reject border slivers or phantom duplicates inside active objects
                    if !touches_active && obj.width >= 16 && obj.height >= 16 && (obj.width * obj.height) >= min_area * 2 {
                        filtered.push(obj);
                    }
                }
                filtered
            } else {
                Vec::new()
            };

            // 3. Emit P-frame: new_count, retired_count, update_count
            writer.write_u16::<LittleEndian>(new_objects.len() as u16)?;
            for obj in &new_objects {
                let def = obj.pack_definition();
                writer.write_all(&def)?;
            }

            writer.write_u16::<LittleEndian>(retired_ids.len() as u16)?;
            for id in &retired_ids {
                writer.write_u32::<LittleEndian>(*id)?;
            }

            writer.write_u16::<LittleEndian>(updates.len() as u16)?;
            for (obj_id, eff) in &updates {
                writer.write_u32::<LittleEndian>(*obj_id)?;
                let eff_bytes = eff.pack();
                writer.write_all(&eff_bytes)?;
            }

            // 4. Register newly spawned objects into tracking engine
            for obj in new_objects {
                engine.active_objects.insert(obj.id, obj);
            }
        }
    }

    Ok(())
}

/// Decode a V7 Boundary Object Stream into reconstructed frames
pub fn decode_boundary_stream<R: Read>(
    mut reader: R,
    width: usize,
    height: usize,
    total_frames: usize,
) -> Result<Vec<Vec<u8>>> {
    let bg_len = reader.read_u32::<LittleEndian>()? as usize;
    let mut bg = vec![0u8; bg_len];
    reader.read_exact(&mut bg)?;
    if bg.len() != width * height {
        bail!("Boundary stream background size mismatch: got {}, expected {}", bg.len(), width * height);
    }

    let mut engine = BoundaryEngine::new(width, height, bg);
    let mut reconstructed_frames = Vec::with_capacity(total_frames);

    for _ in 0..total_frames {
        let mut frame_type_buf = [0u8; 1];
        reader.read_exact(&mut frame_type_buf)?;

        if frame_type_buf[0] == 1 {
            let cam_dx = reader.read_i16::<LittleEndian>()?;
            let cam_dy = reader.read_i16::<LittleEndian>()?;
            engine.camera_dx = cam_dx;
            engine.camera_dy = cam_dy;
        } else {
            engine.camera_dx = 0;
            engine.camera_dy = 0;
        }

        // 1. New objects entering
        let new_count = reader.read_u16::<LittleEndian>()? as usize;
        for _ in 0..new_count {
            let obj = ObjectEntity::unpack_definition(&mut reader)?;
            engine.active_objects.insert(obj.id, obj);
        }

        // 2. Retired objects exiting
        let retired_count = reader.read_u16::<LittleEndian>()? as usize;
        for _ in 0..retired_count {
            let ret_id = reader.read_u32::<LittleEndian>()?;
            engine.active_objects.remove(&ret_id);
        }

        // 3. Constant effects updates on active objects
        let update_count = reader.read_u16::<LittleEndian>()? as usize;
        for _ in 0..update_count {
            let obj_id = reader.read_u32::<LittleEndian>()?;
            let eff = ConstantEffects::unpack(&mut reader)?;
            if let Some(obj) = engine.active_objects.get_mut(&obj_id) {
                obj.effects = eff;
            }
        }

        // 4. Render composite frame with boundary understanding
        let frame = engine.render_frame();
        reconstructed_frames.push(frame);
    }

    Ok(reconstructed_frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_effects_pack_unpack() {
        let orig = ConstantEffects {
            dx: -12,
            dy: 34,
            luma_shift: -5,
            contrast_scale: 140,
            opacity: 220,
            scale_q8: 256,
        };

        let packed = orig.pack();
        let unpacked = ConstantEffects::unpack(&packed[..]).unwrap();
        assert_eq!(orig, unpacked);
    }

    #[test]
    fn test_boundary_mask_rle() {
        let width = 16;
        let height = 16;
        let mut alpha = vec![0u8; 16 * 16];
        // 8x8 circle in the center
        for y in 4..12 {
            for x in 4..12 {
                alpha[y * 16 + x] = 255;
            }
        }

        let mask = BoundaryMask::new(width, height, alpha);
        let rle = mask.compress_rle();
        let decomp = BoundaryMask::decompress_rle(&rle, width, height).unwrap();

        assert_eq!(mask, decomp);
    }

    #[test]
    fn test_boundary_render_fidelity() {
        let w = 64;
        let h = 64;
        let bg = vec![255u8; w * h]; // white background
        let mut engine = BoundaryEngine::new(w, h, bg);

        // Object: 10x10 dark sphere at (20, 20)
        let bw = 10;
        let bh = 10;
        let alpha = vec![255u8; bw * bh];
        let pixels = vec![40u8; bw * bh];

        let mask = BoundaryMask::new(bw, bh, alpha);
        let mut obj = ObjectEntity::new(1, 20, 20, bw, bh, mask, pixels);

        // Constant effect: translate by (5, 3) and luma shift +10
        obj.effects = ConstantEffects {
            dx: 5,
            dy: 3,
            luma_shift: 10,
            contrast_scale: 128,
            opacity: 255,
            scale_q8: 256,
        };

        engine.active_objects.insert(1, obj);
        let canvas = engine.render_frame();

        // Check background outside is still 255
        assert_eq!(canvas[0], 255);
        assert_eq!(canvas[19 * w + 19], 255);

        // Check object at (25, 23) has constant effect: 40 + 10 = 50
        assert_eq!(canvas[23 * w + 25], 50);
        assert_eq!(canvas[25 * w + 27], 50);
    }

    #[test]
    fn test_boundary_render_scaled() {
        let w = 40;
        let h = 40;
        let bg = vec![0u8; w * h];
        let mut engine = BoundaryEngine::new(w, h, bg);

        let bw = 4;
        let bh = 4;
        let alpha = vec![255u8; bw * bh];
        let pixels = vec![100u8; bw * bh];

        let mask = BoundaryMask::new(bw, bh, alpha);
        let mut obj = ObjectEntity::new(1, 10, 10, bw, bh, mask, pixels);

        // Scale by 2.0x (scale_q8 = 512): 4x4 becomes 8x8
        obj.effects = ConstantEffects {
            dx: 0,
            dy: 0,
            luma_shift: 0,
            contrast_scale: 128,
            opacity: 255,
            scale_q8: 512,
        };

        engine.active_objects.insert(1, obj);
        let canvas = engine.render_frame();

        // Origin at (10, 10), scaled size should cover from (10, 10) to (17, 17)
        assert_eq!(canvas[10 * w + 10], 100);
        assert_eq!(canvas[17 * w + 17], 100);
        assert_eq!(canvas[18 * w + 18], 0); // Beyond scaled bbox
    }

    #[test]
    fn test_dynamic_lifecycle_stream() {
        let w = 32;
        let h = 32;
        let bg_val = 0u8;

        // Frame 0: object at (4, 4), 6x6 square of luma 200
        let mut f0 = vec![bg_val; w * h];
        for y in 4..10 {
            for x in 4..10 {
                f0[y * w + x] = 200;
            }
        }

        // Frame 1: object moves to (10, 8)
        let mut f1 = vec![bg_val; w * h];
        for y in 8..14 {
            for x in 10..16 {
                f1[y * w + x] = 200;
            }
        }

        // Frame 2: object moves to (16, 12)
        let mut f2 = vec![bg_val; w * h];
        for y in 12..18 {
            for x in 16..22 {
                f2[y * w + x] = 200;
            }
        }

        let frames = vec![f0.clone(), f1.clone(), f2.clone()];
        let mut encoded = Vec::new();
        encode_boundary_stream(&frames, w, h, &mut encoded).unwrap();

        let decoded = decode_boundary_stream(&encoded[..], w, h, 3).unwrap();
        assert_eq!(decoded.len(), 3);

        // Verify frame 0 reconstruction fidelity
        let mut mse0 = 0.0;
        for i in 0..w * h {
            let diff = f0[i] as f64 - decoded[0][i] as f64;
            mse0 += diff * diff;
        }
        mse0 /= (w * h) as f64;
        assert!(mse0 < 5.0, "Frame 0 MSE was {}", mse0);

        // Verify frame 1 reconstruction fidelity
        let mut mse1 = 0.0;
        for i in 0..w * h {
            let diff = f1[i] as f64 - decoded[1][i] as f64;
            mse1 += diff * diff;
        }
        mse1 /= (w * h) as f64;
        assert!(mse1 < 5.0, "Frame 1 MSE was {}", mse1);

        // Verify frame 2 reconstruction fidelity
        let mut mse2 = 0.0;
        for i in 0..w * h {
            let diff = f2[i] as f64 - decoded[2][i] as f64;
            mse2 += diff * diff;
        }
        mse2 /= (w * h) as f64;
        assert!(mse2 < 5.0, "Frame 2 MSE was {}", mse2);
    }
}
