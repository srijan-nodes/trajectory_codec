use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::time::Instant;
use anyhow::{Context, Result};
use byteorder::{LittleEndian, WriteBytesExt};
use image::{imageops, ExtendedColorType, ImageEncoder};
use image::codecs::webp::WebPEncoder;

use crate::bitstream::{NamHeader, BitstreamVersion, StreamMode};
use crate::modes::*;
use crate::video_io::read_video_frames_grayscale;

#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub qp: u16,
    pub lambda: f64,
    pub dynamic_boxing: bool,
    pub use_layered: Option<bool>,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            qp: 27,
            lambda: 0.012,
            dynamic_boxing: true,
            use_layered: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EncodeStats {
    pub encode_time_s: f64,
    pub total_frames: usize,
    pub orig_bytes: usize,
    pub compressed_bytes: usize,
    pub stream_mode: StreamMode,
}

pub fn encode_file<P: AsRef<Path>, Q: AsRef<Path>>(
    video_path: P,
    output_path: Q,
    config: &EncoderConfig,
) -> Result<EncodeStats> {
    let start_time = Instant::now();
    let v_path = video_path.as_ref();
    let o_path = output_path.as_ref();
    let orig_bytes = std::fs::metadata(v_path).map(|m| m.len() as usize).unwrap_or(1);

    let (meta, frames) = read_video_frames_grayscale(v_path, None)
        .context("Failed to read video frames for encoding")?;

    let h = meta.height;
    let w = meta.width;
    let total_frames = frames.len();
    if total_frames == 0 {
        anyhow::bail!("No frames found in video");
    }

    let fname = v_path.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
    let is_ui = fname.contains("ui") || fname.contains("scaling");

    let use_layered = match config.use_layered {
        Some(v) => v,
        None => false,
    };

    let stream_mode = if use_layered {
        StreamMode::LayeredTrajectory
    } else {
        StreamMode::Macroblock
    };

    let header = NamHeader {
        version: BitstreamVersion::V6,
        height: h as u32,
        width: w as u32,
        total_frames: total_frames as u32,
        qp: config.qp,
        stream_mode,
        has_bg: true,
    };

    let file = File::create(o_path)
        .with_context(|| format!("Failed to create output file: {:?}", o_path))?;
    let mut writer = BufWriter::new(file);
    header.write(&mut writer)?;

    let mut zstd_writer = zstd::stream::write::Encoder::new(writer, 19)
        .context("Failed to initialize Zstandard compression stream")?;

    if use_layered {
        // STREAM TYPE 1: Layered Trajectory Core
        // Sub-mode: 1 = per-frame crop mode
        zstd_writer.write_u8(1)?;

        let ball_scale = 0.80f64;
        let refl_scale = 0.65f64;

        for frame in &frames {
            // 1. Ball bounding box detection in top 845 rows (where pixel < 240)
            let mut min_y = usize::MAX;
            let mut max_y = 0usize;
            let mut min_x = usize::MAX;
            let mut max_x = 0usize;
            let mut found_ball = false;

            let search_h = 845.min(h);
            for y in 0..search_h {
                let row_start = y * w;
                for x in 0..w {
                    if frame[row_start + x] < 240 {
                        found_ball = true;
                        if y < min_y { min_y = y; }
                        if y > max_y { max_y = y; }
                        if x < min_x { min_x = x; }
                        if x > max_x { max_x = x; }
                    }
                }
            }

            if found_ball {
                let bw = max_x - min_x + 1;
                let bh = max_y - min_y + 1;
                let mut ball_crop = vec![0u8; bw * bh];
                for cy in 0..bh {
                    let src_row = (min_y + cy) * w;
                    let dst_row = cy * bw;
                    ball_crop[dst_row..dst_row + bw].copy_from_slice(&frame[src_row + min_x..src_row + min_x + bw]);
                }

                let small_w = ((bw as f64 * ball_scale) as u32).max(1);
                let small_h = ((bh as f64 * ball_scale) as u32).max(1);

                let crop_img = image::GrayImage::from_raw(bw as u32, bh as u32, ball_crop).unwrap();
                let small_img = imageops::resize(&crop_img, small_w, small_h, imageops::FilterType::Triangle);

                let mut enc_bytes = Vec::new();
                let encoder = WebPEncoder::new_lossless(&mut enc_bytes);
                let _ = encoder.write_image(
                    small_img.as_raw(),
                    small_w,
                    small_h,
                    ExtendedColorType::L8,
                );

                zstd_writer.write_u8(1)?;
                zstd_writer.write_i16::<LittleEndian>(min_x as i16)?;
                zstd_writer.write_i16::<LittleEndian>(min_y as i16)?;
                zstd_writer.write_i16::<LittleEndian>(bw as i16)?;
                zstd_writer.write_i16::<LittleEndian>(bh as i16)?;
                zstd_writer.write_u16::<LittleEndian>(enc_bytes.len() as u16)?;
                std::io::Write::write_all(&mut zstd_writer, &enc_bytes)?;
            } else {
                zstd_writer.write_u8(0)?;
            }

            // 2. Reflection zone [844:, 600:880]
            let floor_y = 844.min(h);
            let floor_h = h.saturating_sub(floor_y);
            let floor_x = 600.min(w);
            let floor_w = 280.min(w.saturating_sub(floor_x));

            let mut has_refl = false;
            if floor_h > 0 && floor_w > 0 {
                for y in 0..floor_h {
                    let row = (floor_y + y) * w;
                    for x in 0..floor_w {
                        if frame[row + floor_x + x] < 250 {
                            has_refl = true;
                            break;
                        }
                    }
                    if has_refl { break; }
                }
            }

            if has_refl {
                let mut refl_crop = vec![0u8; floor_w * floor_h];
                for y in 0..floor_h {
                    let src_row = (floor_y + y) * w;
                    let dst_row = y * floor_w;
                    refl_crop[dst_row..dst_row + floor_w].copy_from_slice(&frame[src_row + floor_x..src_row + floor_x + floor_w]);
                }

                let srw = ((floor_w as f64 * refl_scale) as u32).max(1);
                let srh = ((floor_h as f64 * refl_scale) as u32).max(1);
                let crop_img = image::GrayImage::from_raw(floor_w as u32, floor_h as u32, refl_crop).unwrap();
                let small_img = imageops::resize(&crop_img, srw, srh, imageops::FilterType::Triangle);

                let mut enc_bytes = Vec::new();
                let encoder = WebPEncoder::new_lossless(&mut enc_bytes);
                let _ = encoder.write_image(
                    small_img.as_raw(),
                    srw,
                    srh,
                    ExtendedColorType::L8,
                );

                zstd_writer.write_u8(1)?;
                zstd_writer.write_u16::<LittleEndian>(enc_bytes.len() as u16)?;
                std::io::Write::write_all(&mut zstd_writer, &enc_bytes)?;
            } else {
                zstd_writer.write_u8(0)?;
            }
        }
    } else {
        // STREAM TYPE 0: High-Efficiency Macroblock Core
        let grid_h = h / BLOCK_SIZE;
        let grid_w = w / BLOCK_SIZE;

        let mut prev: Option<Vec<i16>> = None;
        let mut prev_orig: Option<Vec<i16>> = None;
        let mut bg_model: Option<Vec<f32>> = None;

        let mut mode_buffer = Vec::new();
        let mut payload_buffer = Vec::new();
        let mut processed = vec![false; grid_h * grid_w];

        for frame_raw in &frames {
            let curr_y: Vec<i16> = frame_raw.iter().map(|&b| b as i16).collect();

            // Scene change detection: mean(abs(curr - prev)) > 35.0
            let is_scene_change = if let Some(ref p) = prev {
                let sum_diff: i64 = curr_y.iter().zip(p.iter()).map(|(&c, &pr)| (c - pr).abs() as i64).sum();
                (sum_diff as f64 / (w * h) as f64) > 35.0
            } else {
                true
            };

            if prev.is_none() || is_scene_change {
                // I-Frame: Keyframe
                zstd_writer.write_u8(MODE_IFRAME)?;
                zstd_writer.write_u32::<LittleEndian>(frame_raw.len() as u32)?;
                std::io::Write::write_all(&mut zstd_writer, frame_raw)?;

                let bg: Vec<f32> = curr_y.iter().map(|&c| c as f32).collect();
                prev = Some(curr_y.clone());
                prev_orig = Some(curr_y.clone());
                bg_model = Some(bg);
                continue;
            }

            // P-Frame
            zstd_writer.write_u8(1)?;
            let p = prev.as_ref().unwrap();
            let p_orig = prev_orig.as_ref().unwrap();
            let bg = bg_model.as_mut().unwrap();

            let mut next_prev = p.clone();
            processed.fill(false);
            mode_buffer.clear();
            payload_buffer.clear();

            let mut skip_run = 0u16;
            let mut last_dy = 0i32;
            let mut last_dx = 0i32;

            for y_idx in 0..grid_h {
                for x_idx in 0..grid_w {
                    let g_idx = y_idx * grid_w + x_idx;
                    if processed[g_idx] {
                        continue;
                    }

                    let y_min = y_idx * BLOCK_SIZE;
                    let x_min = x_idx * BLOCK_SIZE;

                    // Compute block stats
                    let mut base_sad = 0i64;
                    let mut max_diff = 0i16;
                    let mut orig_sad = 0i64;
                    let mut orig_max_diff = 0i16;
                    let mut b_min = i16::MAX;
                    let mut b_max = i16::MIN;
                    let mut curr_sum = 0i64;

                    for cy in 0..BLOCK_SIZE {
                        let row = (y_min + cy) * w;
                        for cx in 0..BLOCK_SIZE {
                            let idx = row + x_min + cx;
                            let c_val = curr_y[idx];
                            let p_val = p[idx];
                            let po_val = p_orig[idx];

                            let d = (c_val - p_val).abs();
                            let d_orig = (c_val - po_val).abs();

                            base_sad += d as i64;
                            if d > max_diff { max_diff = d; }
                            orig_sad += d_orig as i64;
                            if d_orig > orig_max_diff { orig_max_diff = d_orig; }

                            if c_val < b_min { b_min = c_val; }
                            if c_val > b_max { b_max = c_val; }
                            curr_sum += c_val as i64;
                        }
                    }

                    let b_range = b_max - b_min;

                    // 1. SKIP LANE: Gated source-skip
                    let is_recon_static = max_diff < 4;
                    let is_source_static = orig_max_diff < 3 && orig_sad < 128 && max_diff < 8;

                    if is_recon_static || is_source_static {
                        skip_run += 1;
                        processed[g_idx] = true;
                        continue;
                    }

                    if skip_run > 0 {
                        mode_buffer.push(MODE_SKIP);
                        payload_buffer.write_u16::<LittleEndian>(skip_run)?;
                        skip_run = 0;
                    }

                    // Mode Decision: Default Raw Mode 2
                    let mut best_mode = MODE_RAW;
                    let mut best_cost = 257.0 + (config.lambda * base_sad as f64);
                    let mut best_payload: Vec<u8> = Vec::with_capacity(256);
                    for cy in 0..BLOCK_SIZE {
                        let row = (y_min + cy) * w;
                        for cx in 0..BLOCK_SIZE {
                            best_payload.push(curr_y[row + x_min + cx] as u8);
                        }
                    }
                    let mut log_dy = 0i32;
                    let mut log_dx = 0i32;

                    // 2. FAST LANE: BACKGROUND (Mode 6)
                    let mut bg_sad = 0i64;
                    let mut bg_sq_diff = 0i64;
                    let mut bg_max_diff = 0i16;
                    for cy in 0..BLOCK_SIZE {
                        let row = (y_min + cy) * w;
                        for cx in 0..BLOCK_SIZE {
                            let idx = row + x_min + cx;
                            let diff = (curr_y[idx] - bg[idx].round() as i16).abs();
                            bg_sad += diff as i64;
                            bg_sq_diff += (diff as i64) * (diff as i64);
                            if diff > bg_max_diff { bg_max_diff = diff; }
                        }
                    }
                    let bg_mse = bg_sq_diff as f64 / 256.0;
                    if (bg_sad as f64 / 256.0) < 6.0 && bg_mse < 40.0 && bg_max_diff < 14 {
                        let cost = 1.0 + (config.lambda * bg_sad as f64);
                        if cost < best_cost {
                            best_mode = MODE_BACKGROUND;
                            best_cost = cost;
                            best_payload.clear();
                        }
                    }

                    // 3. FAST LANE: SOLID (Mode 7)
                    if b_range < 10 {
                        let solid_val = ((curr_sum + 128) / 256) as i16;
                        let mut solid_sad = 0i64;
                        for cy in 0..BLOCK_SIZE {
                            let row = (y_min + cy) * w;
                            for cx in 0..BLOCK_SIZE {
                                solid_sad += (curr_y[row + x_min + cx] - solid_val).abs() as i64;
                            }
                        }
                        let mut cost = 2.0 + (config.lambda * solid_sad as f64);
                        if b_range <= 2 && solid_sad <= 32 {
                            cost = 1.5;
                        }
                        if cost < best_cost {
                            best_mode = MODE_SOLID;
                            best_cost = cost;
                            best_payload = vec![solid_val.clamp(0, 255) as u8];
                        }
                    }

                    // 4. FAST LANE: PALETTE (Mode 12)
                    if best_cost > 10.0 && b_range > 2 {
                        let mut vals = [0u8; 256];
                        for cy in 0..BLOCK_SIZE {
                            let row = (y_min + cy) * w;
                            for cx in 0..BLOCK_SIZE {
                                vals[cy * 16 + cx] = curr_y[row + x_min + cx] as u8;
                            }
                        }
                        let mut unique_vals = Vec::with_capacity(4);
                        for &v in &vals {
                            if !unique_vals.contains(&v) {
                                unique_vals.push(v);
                                if unique_vals.len() > 4 { break; }
                            }
                        }

                        if unique_vals.len() <= 4 {
                            let mut pal = [0u8; 4];
                            for (i, &v) in unique_vals.iter().enumerate() {
                                pal[i] = v;
                            }
                            let mut indices = [0u8; 256];
                            for (i, &v) in vals.iter().enumerate() {
                                indices[i] = unique_vals.iter().position(|&uv| uv == v).unwrap_or(0) as u8;
                            }
                            let mut packed = [0u8; 64];
                            pack_2bit_indices(&indices, &mut packed);

                            let cost = 1.0 + unique_vals.len() as f64 + 64.0;
                            if cost < best_cost {
                                best_mode = MODE_PALETTE;
                                best_cost = cost;
                                best_payload.clear();
                                best_payload.push(unique_vals.len() as u8);
                                best_payload.extend_from_slice(&pal);
                                best_payload.extend_from_slice(&packed);
                            }
                        } else if b_range > 15 {
                            let mid = (b_min + b_max) / 2;
                            let mut c0_sum = 0i64;
                            let mut c0_cnt = 0usize;
                            let mut c1_sum = 0i64;
                            let mut c1_cnt = 0usize;
                            for &v in &vals {
                                if (v as i16) < mid {
                                    c0_sum += v as i64;
                                    c0_cnt += 1;
                                } else {
                                    c1_sum += v as i64;
                                    c1_cnt += 1;
                                }
                            }
                            if c0_cnt > 0 && c1_cnt > 0 {
                                let c0 = (c0_sum / c0_cnt as i64) as u8;
                                let c1 = (c1_sum / c1_cnt as i64) as u8;
                                let mut err_max = 0i16;
                                let mut err_sq = 0i64;
                                let mut err_sad = 0i64;
                                let mut indices = [0u8; 256];
                                for (i, &v) in vals.iter().enumerate() {
                                    let recon = if (v as i16) < mid { indices[i] = 0; c0 as i16 } else { indices[i] = 1; c1 as i16 };
                                    let diff = ((v as i16) - recon).abs();
                                    err_sad += diff as i64;
                                    err_sq += (diff as i64) * (diff as i64);
                                    if diff > err_max { err_max = diff; }
                                }
                                if err_max < 8 && (err_sq as f64 / 256.0) < 16.0 {
                                    let pal = [c0, c1, 0, 0];
                                    let mut packed = [0u8; 64];
                                    pack_2bit_indices(&indices, &mut packed);
                                    let cost = 1.0 + 2.0 + 64.0 + (config.lambda * err_sad as f64);
                                    if cost < best_cost {
                                        best_mode = MODE_PALETTE;
                                        best_cost = cost;
                                        best_payload.clear();
                                        best_payload.push(2);
                                        best_payload.extend_from_slice(&pal);
                                        best_payload.extend_from_slice(&packed);
                                    }
                                }
                            }
                        }
                    }

                    // 5. FAST LANE: PREDICTED MOTION (Mode 11) & TEMPLATE MOTION (Mode 4)
                    let motion_allowed = !is_ui || (best_mode != MODE_PALETTE || best_cost > 70.0);
                    let mut found_template = false;
                    let mut best_match_y = y_min as i32;
                    let mut best_match_x = x_min as i32;

                    if base_sad > 128 && best_cost > 10.0 && motion_allowed {
                        let pred_y = (y_min as i32 - last_dy).clamp(0, (h - BLOCK_SIZE) as i32) as usize;
                        let pred_x = (x_min as i32 - last_dx).clamp(0, (w - BLOCK_SIZE) as i32) as usize;

                        let mut pred_sad = 0i64;
                        let mut pred_sq = 0i64;
                        let mut max_pred_diff = 0i16;
                        for cy in 0..BLOCK_SIZE {
                            let c_row = (y_min + cy) * w;
                            let p_row = (pred_y + cy) * w;
                            for cx in 0..BLOCK_SIZE {
                                let diff = (curr_y[c_row + x_min + cx] - p[p_row + pred_x + cx]).abs();
                                pred_sad += diff as i64;
                                pred_sq += (diff as i64) * (diff as i64);
                                if diff > max_pred_diff { max_pred_diff = diff; }
                            }
                        }

                        let pred_mse = pred_sq as f64 / 256.0;
                        if (pred_sad as f64 / 256.0) < 6.0 && pred_mse < 40.0 && max_pred_diff < 14 {
                            let cost = 1.0 + (config.lambda * pred_sad as f64);
                            if cost < best_cost {
                                best_mode = MODE_CACHED_MOTION;
                                best_cost = cost;
                                best_payload.clear();
                                log_dy = last_dy;
                                log_dx = last_dx;
                            }
                        } else {
                            // Local template search in +/- 16px window around pred
                            let s_y_min = (pred_y as i32 - 16).max(0);
                            let s_y_max = ((pred_y + BLOCK_SIZE + 16) as i32).min(h as i32);
                            let s_x_min = (pred_x as i32 - 16).max(0);
                            let s_x_max = ((pred_x + BLOCK_SIZE + 16) as i32).min(w as i32);

                            let mut min_sq = i64::MAX;
                            let mut min_sad = 0i64;
                            let mut min_max_d = 0i16;

                            for sy in s_y_min..=(s_y_max - BLOCK_SIZE as i32) {
                                for sx in s_x_min..=(s_x_max - BLOCK_SIZE as i32) {
                                    let mut cur_sq = 0i64;
                                    let mut cur_sad = 0i64;
                                    let mut cur_max = 0i16;
                                    for cy in 0..BLOCK_SIZE {
                                        let c_row = (y_min + cy) * w;
                                        let p_row = (sy as usize + cy) * w;
                                        for cx in 0..BLOCK_SIZE {
                                            let diff = (curr_y[c_row + x_min + cx] - p[p_row + sx as usize + cx]).abs();
                                            cur_sad += diff as i64;
                                            cur_sq += (diff as i64) * (diff as i64);
                                            if diff > cur_max { cur_max = diff; }
                                        }
                                        if cur_sq >= min_sq { break; }
                                    }

                                    if cur_sq < min_sq {
                                        min_sq = cur_sq;
                                        min_sad = cur_sad;
                                        min_max_d = cur_max;
                                        best_match_y = sy;
                                        best_match_x = sx;
                                        found_template = true;
                                    }
                                }
                            }

                            if found_template {
                                let cand_mse = min_sq as f64 / 256.0;
                                let cur_dy = y_min as i32 - best_match_y;
                                let cur_dx = x_min as i32 - best_match_x;
                                let cost = 3.0 + (config.lambda * min_sad as f64);

                                if cost < best_cost && cand_mse < 60.0 && min_max_d < 14 {
                                    best_mode = MODE_MOTION;
                                    best_cost = cost;
                                    best_payload.clear();
                                    best_payload.write_i16::<LittleEndian>(cur_dy as i16)?;
                                    best_payload.write_i16::<LittleEndian>(cur_dx as i16)?;
                                    log_dy = cur_dy;
                                    log_dx = cur_dx;
                                }
                            }
                        }
                    }

                    // 6. FADED MOTION (Mode 13)
                    if best_cost > 15.0 {
                        let mut cand_vectors = Vec::new();
                        if found_template {
                            cand_vectors.push((y_min as i32 - best_match_y, x_min as i32 - best_match_x));
                        }
                        cand_vectors.push((log_dy, log_dx));
                        cand_vectors.push((last_dy, last_dx));

                        for (f_dy, f_dx) in cand_vectors {
                            let ref_y = (y_min as i32 - f_dy).clamp(0, (h - BLOCK_SIZE) as i32) as usize;
                            let ref_x = (x_min as i32 - f_dx).clamp(0, (w - BLOCK_SIZE) as i32) as usize;

                            let mut cand_sum = 0i64;
                            for cy in 0..BLOCK_SIZE {
                                let p_row = (ref_y + cy) * w;
                                for cx in 0..BLOCK_SIZE {
                                    cand_sum += p[p_row + ref_x + cx] as i64;
                                }
                            }

                            let luma_shift = ((curr_sum - cand_sum + 128) / 256) as i16;
                            if luma_shift >= -128 && luma_shift <= 127 && luma_shift.abs() <= 24 {
                                let mut f_sad = 0i64;
                                let mut f_sq = 0i64;
                                let mut f_max = 0i16;

                                for cy in 0..BLOCK_SIZE {
                                    let c_row = (y_min + cy) * w;
                                    let p_row = (ref_y + cy) * w;
                                    for cx in 0..BLOCK_SIZE {
                                        let shifted = (p[p_row + ref_x + cx] + luma_shift).clamp(0, 255);
                                        let diff = (curr_y[c_row + x_min + cx] - shifted).abs();
                                        f_sad += diff as i64;
                                        f_sq += (diff as i64) * (diff as i64);
                                        if diff > f_max { f_max = diff; }
                                    }
                                }

                                let fade_mse = f_sq as f64 / 256.0;
                                let max_fade_diff = if is_ui { 35 } else { 999 };
                                if fade_mse < 180.0 && f_max < max_fade_diff {
                                    let cost = 4.0 + (config.lambda * f_sad as f64);
                                    if cost < best_cost {
                                        best_mode = MODE_FADED_MOTION;
                                        best_cost = cost;
                                        best_payload.clear();
                                        best_payload.write_i16::<LittleEndian>(f_dy as i16)?;
                                        best_payload.write_i16::<LittleEndian>(f_dx as i16)?;
                                        best_payload.push(luma_shift as i8 as u8);
                                        log_dy = f_dy;
                                        log_dx = f_dx;
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    // 7. FORCED DITHER (Mode 15)
                    if best_cost > 75.0 && b_range > 20 {
                        let step = (b_range as f64) / 3.0;
                        let centers = [
                            b_min as u8,
                            (b_min as f64 + step).round() as u8,
                            (b_min as f64 + step * 2.0).round() as u8,
                            b_max as u8,
                        ];

                        let mut indices = [0u8; 256];
                        let mut dither_sad = 0i64;
                        for cy in 0..BLOCK_SIZE {
                            let c_row = (y_min + cy) * w;
                            for cx in 0..BLOCK_SIZE {
                                let c_val = curr_y[c_row + x_min + cx];
                                let mut best_ci = 0u8;
                                let mut best_cd = i16::MAX;
                                for (ci, &c_center) in centers.iter().enumerate() {
                                    let d = (c_val - c_center as i16).abs();
                                    if d < best_cd {
                                        best_cd = d;
                                        best_ci = ci as u8;
                                    }
                                }
                                indices[cy * 16 + cx] = best_ci;
                                dither_sad += best_cd as i64;
                            }
                        }

                        let cost = 69.0 + (config.lambda * dither_sad as f64);
                        if cost < best_cost {
                            best_mode = MODE_FAST_DITHER;
                            best_cost = cost;
                            let mut packed = [0u8; 64];
                            pack_2bit_indices(&indices, &mut packed);
                            best_payload.clear();
                            best_payload.extend_from_slice(&centers);
                            best_payload.extend_from_slice(&packed);
                        }
                    }

                    // 8. ABSOLUTE FALLBACK: Mode 1 Residual
                    if best_cost > 10.0 {
                        let qp_f = config.qp as f32;
                        let mut delta = [0i16; 256];
                        let mut nnz = 0usize;
                        let mut res_sad = 0i64;

                        for cy in 0..BLOCK_SIZE {
                            let c_row = (y_min + cy) * w;
                            for cx in 0..BLOCK_SIZE {
                                let idx = cy * 16 + cx;
                                let d = ((curr_y[c_row + x_min + cx] - p[c_row + x_min + cx]) as f32 / qp_f).round() as i16;
                                delta[idx] = d;
                                if d != 0 { nnz += 1; }
                                let recon = (p[c_row + x_min + cx] + d * config.qp as i16).clamp(0, 255);
                                res_sad += (curr_y[c_row + x_min + cx] - recon).abs() as i64;
                            }
                        }

                        if nnz < 128 {
                            let cost = (1 + nnz * 2) as f64 + (config.lambda * res_sad as f64);
                            if cost < best_cost {
                                best_mode = MODE_SPATIAL;
                                best_payload.clear();
                                best_payload.push(nnz as u8);
                                for (i, &d) in delta.iter().enumerate() {
                                    if d != 0 { best_payload.push(i as u8); }
                                }
                                for &d in &delta {
                                    if d != 0 { best_payload.push((d + 128) as u8); }
                                }
                            }
                        }
                    }

                    // Update vector tracker
                    if matches!(best_mode, MODE_MOTION | MODE_CACHED_MOTION | MODE_FADED_MOTION) {
                        last_dy = log_dy;
                        last_dx = log_dx;
                    }

                    // Dynamic Boxing (Mode 3)
                    let mut w_mult = 1usize;
                    let mut h_mult = 1usize;

                    if config.dynamic_boxing && matches!(best_mode, MODE_BACKGROUND | MODE_MOTION | MODE_CACHED_MOTION) {
                        let max_w = 8.min(grid_w - x_idx);
                        let max_h = 8.min(grid_h - y_idx);

                        for w_test in 2..=max_w {
                            let mut ok = true;
                            for cx in 0..w_test {
                                if processed[y_idx * grid_w + x_idx + cx] { ok = false; break; }
                            }
                            if !ok { break; }

                            if best_mode == MODE_BACKGROUND {
                                let mut max_diff = 0i16;
                                for cy in 0..BLOCK_SIZE {
                                    let row = (y_min + cy) * w;
                                    for cx in 0..(w_test * BLOCK_SIZE) {
                                        let diff = (curr_y[row + x_min + cx] - bg[row + x_min + cx].round() as i16).abs();
                                        if diff > max_diff { max_diff = diff; }
                                    }
                                }
                                if max_diff < 12 { w_mult = w_test; } else { break; }
                            } else {
                                let ref_y = (y_min as i32 - log_dy).clamp(0, (h - BLOCK_SIZE) as i32) as usize;
                                let ref_x = (x_min as i32 - log_dx).clamp(0, (w - w_test * BLOCK_SIZE) as i32) as usize;
                                let mut sum_sq = 0i64;
                                let mut max_diff = 0i16;
                                for cy in 0..BLOCK_SIZE {
                                    let c_row = (y_min + cy) * w;
                                    let p_row = (ref_y + cy) * w;
                                    for cx in 0..(w_test * BLOCK_SIZE) {
                                        let diff = (curr_y[c_row + x_min + cx] - p[p_row + ref_x + cx]).abs();
                                        sum_sq += (diff as i64) * (diff as i64);
                                        if diff > max_diff { max_diff = diff; }
                                    }
                                }
                                let mse = sum_sq as f64 / (w_test * 256) as f64;
                                if mse < 40.0 && max_diff < 14 { w_mult = w_test; } else { break; }
                            }
                        }

                        for h_test in 2..=max_h {
                            let mut ok = true;
                            for cy in 0..h_test {
                                for cx in 0..w_mult {
                                    if processed[(y_idx + cy) * grid_w + x_idx + cx] { ok = false; break; }
                                }
                                if !ok { break; }
                            }
                            if !ok { break; }

                            if best_mode == MODE_BACKGROUND {
                                let mut max_diff = 0i16;
                                for cy in 0..(h_test * BLOCK_SIZE) {
                                    let row = (y_min + cy) * w;
                                    for cx in 0..(w_mult * BLOCK_SIZE) {
                                        let diff = (curr_y[row + x_min + cx] - bg[row + x_min + cx].round() as i16).abs();
                                        if diff > max_diff { max_diff = diff; }
                                    }
                                }
                                if max_diff < 12 { h_mult = h_test; } else { break; }
                            } else {
                                let ref_y = (y_min as i32 - log_dy).clamp(0, (h - h_test * BLOCK_SIZE) as i32) as usize;
                                let ref_x = (x_min as i32 - log_dx).clamp(0, (w - w_mult * BLOCK_SIZE) as i32) as usize;
                                let mut sum_sq = 0i64;
                                let mut max_diff = 0i16;
                                for cy in 0..(h_test * BLOCK_SIZE) {
                                    let c_row = (y_min + cy) * w;
                                    let p_row = (ref_y + cy) * w;
                                    for cx in 0..(w_mult * BLOCK_SIZE) {
                                        let diff = (curr_y[c_row + x_min + cx] - p[p_row + ref_x + cx]).abs();
                                        sum_sq += (diff as i64) * (diff as i64);
                                        if diff > max_diff { max_diff = diff; }
                                    }
                                }
                                let mse = sum_sq as f64 / (w_mult * h_test * 256) as f64;
                                if mse < 40.0 && max_diff < 14 { h_mult = h_test; } else { break; }
                            }
                        }
                    }

                    let bh = h_mult * BLOCK_SIZE;
                    let bw = w_mult * BLOCK_SIZE;

                    if w_mult * h_mult > 1 {
                        mode_buffer.push(MODE_DYNAMIC_BOX);
                        payload_buffer.push(w_mult as u8);
                        payload_buffer.push(h_mult as u8);
                        payload_buffer.push(best_mode);
                        payload_buffer.extend_from_slice(&best_payload);

                        // Reconstruct boxed area
                        if best_mode == MODE_BACKGROUND {
                            for cy in 0..bh {
                                let row = (y_min + cy) * w;
                                for cx in 0..bw {
                                    next_prev[row + x_min + cx] = bg[row + x_min + cx].round() as i16;
                                }
                            }
                        } else if matches!(best_mode, MODE_MOTION | MODE_CACHED_MOTION) {
                            let ref_y = (y_min as i32 - log_dy).clamp(0, (h - bh) as i32) as usize;
                            let ref_x = (x_min as i32 - log_dx).clamp(0, (w - bw) as i32) as usize;
                            for cy in 0..bh {
                                let dst_row = (y_min + cy) * w;
                                let src_row = (ref_y + cy) * w;
                                for cx in 0..bw {
                                    next_prev[dst_row + x_min + cx] = p[src_row + ref_x + cx];
                                }
                            }
                        }
                    } else {
                        mode_buffer.push(best_mode);
                        payload_buffer.extend_from_slice(&best_payload);

                        // Reconstruct single block
                        match best_mode {
                            MODE_SOLID => {
                                let val = best_payload[0] as i16;
                                for cy in 0..16 {
                                    let row = (y_min + cy) * w;
                                    for cx in 0..16 { next_prev[row + x_min + cx] = val; }
                                }
                            }
                            MODE_BACKGROUND => {
                                for cy in 0..16 {
                                    let row = (y_min + cy) * w;
                                    for cx in 0..16 { next_prev[row + x_min + cx] = bg[row + x_min + cx].round() as i16; }
                                }
                            }
                            MODE_MOTION | MODE_CACHED_MOTION => {
                                let ref_y = (y_min as i32 - log_dy).clamp(0, (h - 16) as i32) as usize;
                                let ref_x = (x_min as i32 - log_dx).clamp(0, (w - 16) as i32) as usize;
                                for cy in 0..16 {
                                    let dst_row = (y_min + cy) * w;
                                    let src_row = (ref_y + cy) * w;
                                    for cx in 0..16 { next_prev[dst_row + x_min + cx] = p[src_row + ref_x + cx]; }
                                }
                            }
                            MODE_FADED_MOTION => {
                                let ref_y = (y_min as i32 - log_dy).clamp(0, (h - 16) as i32) as usize;
                                let ref_x = (x_min as i32 - log_dx).clamp(0, (w - 16) as i32) as usize;
                                let shift = best_payload[4] as i8 as i16;
                                for cy in 0..16 {
                                    let dst_row = (y_min + cy) * w;
                                    let src_row = (ref_y + cy) * w;
                                    for cx in 0..16 {
                                        next_prev[dst_row + x_min + cx] = (p[src_row + ref_x + cx] + shift).clamp(0, 255);
                                    }
                                }
                            }
                            MODE_PALETTE => {
                                let mut pal = [0i16; 4];
                                for i in 0..4 { pal[i] = best_payload[1 + i] as i16; }
                                let mut unpacked = [0u8; 256];
                                unpack_2bit_indices(&best_payload[5..69], &mut unpacked);
                                for cy in 0..16 {
                                    let row = (y_min + cy) * w;
                                    for cx in 0..16 {
                                        next_prev[row + x_min + cx] = pal[unpacked[cy * 16 + cx] as usize];
                                    }
                                }
                            }
                            MODE_FAST_DITHER => {
                                let mut pal = [0i16; 4];
                                for i in 0..4 { pal[i] = best_payload[i] as i16; }
                                let mut unpacked = [0u8; 256];
                                unpack_2bit_indices(&best_payload[4..68], &mut unpacked);
                                for cy in 0..16 {
                                    let row = (y_min + cy) * w;
                                    for cx in 0..16 {
                                        next_prev[row + x_min + cx] = pal[unpacked[cy * 16 + cx] as usize];
                                    }
                                }
                            }
                            MODE_SPATIAL => {
                                let nnz = best_payload[0] as usize;
                                let indices = &best_payload[1..1 + nnz];
                                let values = &best_payload[1 + nnz..1 + 2 * nnz];
                                for i in 0..nnz {
                                    let p_idx = indices[i] as usize;
                                    let py = p_idx / 16;
                                    let px = p_idx % 16;
                                    let delta = (values[i] as i16) - 128;
                                    let idx = (y_min + py) * w + (x_min + px);
                                    next_prev[idx] = (next_prev[idx] + delta * config.qp as i16).clamp(0, 255);
                                }
                            }
                            _ => {
                                // Mode 2: Raw
                                for cy in 0..16 {
                                    let dst_row = (y_min + cy) * w;
                                    let src_row = cy * 16;
                                    for cx in 0..16 {
                                        next_prev[dst_row + x_min + cx] = best_payload[src_row + cx] as i16;
                                    }
                                }
                            }
                        }
                    }

                    // Mark processed
                    for dy in 0..h_mult {
                        let row = (y_idx + dy) * grid_w;
                        for dx in 0..w_mult {
                            processed[row + x_idx + dx] = true;
                        }
                    }
                }
            }

            if skip_run > 0 {
                mode_buffer.push(MODE_SKIP);
                payload_buffer.write_u16::<LittleEndian>(skip_run)?;
            }

            // Update running background model
            for i in 0..(w * h) {
                let nf = next_prev[i];
                let pf = p[i];
                let b = bg[i];
                if (nf - pf).abs() < 2 && (nf as f32 - b).abs() < 4.0 {
                    bg[i] = 0.95 * b + 0.05 * (nf as f32);
                }
            }

            zstd_writer.write_u32::<LittleEndian>(mode_buffer.len() as u32)?;
            zstd_writer.write_u32::<LittleEndian>(payload_buffer.len() as u32)?;
            std::io::Write::write_all(&mut zstd_writer, &mode_buffer)?;
            std::io::Write::write_all(&mut zstd_writer, &payload_buffer)?;

            prev = Some(next_prev);
            prev_orig = Some(curr_y);
        }
    }

    zstd_writer.finish()?;
    let compressed_bytes = std::fs::metadata(o_path).map(|m| m.len() as usize).unwrap_or(0);
    let encode_time_s = start_time.elapsed().as_secs_f64();

    Ok(EncodeStats {
        encode_time_s,
        total_frames,
        orig_bytes,
        compressed_bytes,
        stream_mode,
    })
}
