use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::time::Instant;
use anyhow::{bail, Context, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use image::{imageops, ImageFormat};

use crate::bitstream::{NamHeader, StreamMode};
use crate::boundary::decode_boundary_stream;
use crate::modes::*;

#[derive(Debug, Clone)]
pub struct DecodedVideo {
    pub width: usize,
    pub height: usize,
    pub total_frames: usize,
    pub is_rgb: bool,
    pub frames: Vec<Vec<u8>>,
    pub dec_time_s: f64,
}

// Convert 16-bit half precision float to 32-bit single precision float
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            let mut m = mant;
            let mut e = 0;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            let exp_f32 = 127 - 15 - e + 1;
            let mant_f32 = (m & 0x3FF) << 13;
            f32::from_bits((sign << 31) | (exp_f32 << 23) | mant_f32)
        }
    } else if exp == 31 {
        let exp_f32 = 255;
        let mant_f32 = if mant != 0 { 0x400000 } else { 0 };
        f32::from_bits((sign << 31) | (exp_f32 << 23) | mant_f32)
    } else {
        let exp_f32 = exp + 127 - 15;
        let mant_f32 = mant << 13;
        f32::from_bits((sign << 31) | (exp_f32 << 23) | mant_f32)
    }
}

pub fn decode_file<P: AsRef<Path>>(encoded_path: P) -> Result<DecodedVideo> {
    let start_time = Instant::now();
    let file = File::open(encoded_path.as_ref())
        .with_context(|| format!("Failed to open encoded file: {:?}", encoded_path.as_ref()))?;
    let mut reader = BufReader::new(file);

    let header = NamHeader::read(&mut reader)?;
    let h = header.height as usize;
    let w = header.width as usize;
    let total_frames = header.total_frames as usize;
    let qp = header.qp as i16;
    let has_bg = header.has_bg;

    let mut stream = zstd::stream::read::Decoder::new(reader)
        .context("Failed to initialize Zstandard decompression stream")?;

    let dct = DctEngine::new();
    let mut reconstructed_frames = Vec::with_capacity(total_frames);
    let mut is_rgb = false;

    match header.stream_mode {
        StreamMode::LayeredTrajectory => {
            is_rgb = true;
            let mut sub_mode_buf = [0u8; 1];
            stream.read_exact(&mut sub_mode_buf)?;
            let sub_mode = sub_mode_buf[0];

            if sub_mode == 1 {
                // Per-frame downscaled crop mode
                for _ in 0..total_frames {
                    // Start with solid white RGB (255, 255, 255)
                    let mut frame_rgb = vec![255u8; w * h * 3];

                    let mut has_b_buf = [0u8; 1];
                    stream.read_exact(&mut has_b_buf)?;
                    let has_b = has_b_buf[0];

                    if has_b > 0 {
                        let bx = stream.read_i16::<LittleEndian>()? as i32;
                        let by = stream.read_i16::<LittleEndian>()? as i32;
                        let bw = stream.read_i16::<LittleEndian>()? as usize;
                        let bh = stream.read_i16::<LittleEndian>()? as usize;
                        let enc_len = stream.read_u16::<LittleEndian>()? as usize;

                        let mut raw_b = vec![0u8; enc_len];
                        stream.read_exact(&mut raw_b)?;

                        if let Ok(img) = image::load_from_memory_with_format(&raw_b, ImageFormat::WebP) {
                            let mut gray_img = img.to_luma8();
                            if gray_img.width() as usize != bw || gray_img.height() as usize != bh {
                                gray_img = imageops::resize(
                                    &gray_img,
                                    bw as u32,
                                    bh as u32,
                                    imageops::FilterType::CatmullRom, // Bicubic equivalent
                                );
                            }

                            for cy in 0..bh {
                                for cx in 0..bw {
                                    let py = (by + cy as i32) as usize;
                                    let px = (bx + cx as i32) as usize;
                                    if py < h && px < w {
                                        let pixel_val = gray_img.get_pixel(cx as u32, cy as u32)[0];
                                        if pixel_val < 240 {
                                            let idx = (py * w + px) * 3;
                                            frame_rgb[idx] = pixel_val;
                                            frame_rgb[idx + 1] = pixel_val;
                                            frame_rgb[idx + 2] = pixel_val;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    let mut has_r_buf = [0u8; 1];
                    stream.read_exact(&mut has_r_buf)?;
                    let has_r = has_r_buf[0];

                    if has_r > 0 {
                        let enc_len = stream.read_u16::<LittleEndian>()? as usize;
                        let mut raw_r = vec![0u8; enc_len];
                        stream.read_exact(&mut raw_r)?;

                        if let Ok(img) = image::load_from_memory_with_format(&raw_r, ImageFormat::WebP) {
                            let floor_h = h.saturating_sub(844);
                            let floor_w = 280.min(w.saturating_sub(600));

                            let mut rgb_img = img.to_rgb8();
                            if rgb_img.width() as usize != floor_w || rgb_img.height() as usize != floor_h {
                                rgb_img = imageops::resize(
                                    &rgb_img,
                                    floor_w as u32,
                                    floor_h as u32,
                                    imageops::FilterType::Triangle, // Bilinear
                                );
                            }

                            for cy in 0..floor_h {
                                for cx in 0..floor_w {
                                    let py = 844 + cy;
                                    let px = 600 + cx;
                                    if py < h && px < w {
                                        let p = rgb_img.get_pixel(cx as u32, cy as u32);
                                        let idx = (py * w + px) * 3;
                                        frame_rgb[idx] = p[0];
                                        frame_rgb[idx + 1] = p[1];
                                        frame_rgb[idx + 2] = p[2];
                                    }
                                }
                            }
                        }
                    }

                    reconstructed_frames.push(frame_rgb);
                }
            } else {
                // Legacy template mode
                let num_templates = sub_mode as usize;
                let mut templates = Vec::with_capacity(num_templates);
                for _ in 0..num_templates {
                    let w_sz = stream.read_u32::<LittleEndian>()? as usize;
                    let mut w_bytes = vec![0u8; w_sz];
                    stream.read_exact(&mut w_bytes)?;
                    let img = image::load_from_memory_with_format(&w_bytes, ImageFormat::WebP)?.to_luma8();
                    templates.push(img);
                }

                for _ in 0..total_frames {
                    let mut frame_rgb = vec![255u8; w * h * 3];
                    let mut has_b_buf = [0u8; 1];
                    stream.read_exact(&mut has_b_buf)?;
                    let has_b = has_b_buf[0];

                    if has_b > 0 {
                        let bx = stream.read_i16::<LittleEndian>()? as i32;
                        let by = stream.read_i16::<LittleEndian>()? as i32;
                        let bw = stream.read_i16::<LittleEndian>()? as usize;
                        let bh = stream.read_i16::<LittleEndian>()? as usize;
                        let t_idx = (has_b as usize - 1).min(num_templates - 1);
                        let templ = &templates[t_idx];
                        let scaled = imageops::resize(templ, bw as u32, bh as u32, imageops::FilterType::Triangle);

                        for cy in 0..bh {
                            for cx in 0..bw {
                                let py = (by + cy as i32) as usize;
                                let px = (bx + cx as i32) as usize;
                                if py < h && px < w {
                                    let pixel_val = scaled.get_pixel(cx as u32, cy as u32)[0];
                                    if pixel_val < 240 {
                                        let idx = (py * w + px) * 3;
                                        frame_rgb[idx] = pixel_val;
                                        frame_rgb[idx + 1] = pixel_val;
                                        frame_rgb[idx + 2] = pixel_val;
                                    }
                                }
                            }
                        }
                    }

                    let mut has_r_buf = [0u8; 1];
                    stream.read_exact(&mut has_r_buf)?;
                    let has_r = has_r_buf[0];
                    if has_r > 0 {
                        let enc_len = stream.read_u16::<LittleEndian>()? as usize;
                        let mut raw_r = vec![0u8; enc_len];
                        stream.read_exact(&mut raw_r)?;
                        if let Ok(img) = image::load_from_memory_with_format(&raw_r, ImageFormat::WebP) {
                            let rgb_img = img.to_rgb8();
                            let rh = rgb_img.height() as usize;
                            let rw = rgb_img.width() as usize;
                            for cy in 0..rh {
                                for cx in 0..rw {
                                    let py = 844 + cy;
                                    let px = 600 + cx;
                                    if py < h && px < w {
                                        let p = rgb_img.get_pixel(cx as u32, cy as u32);
                                        let idx = (py * w + px) * 3;
                                        frame_rgb[idx] = p[0];
                                        frame_rgb[idx + 1] = p[1];
                                        frame_rgb[idx + 2] = p[2];
                                    }
                                }
                            }
                        }
                    }
                    reconstructed_frames.push(frame_rgb);
                }
            }
        }

        StreamMode::BoundaryObject => {
            is_rgb = false;
            let boundary_frames = decode_boundary_stream(&mut stream, w, h, total_frames)?;
            reconstructed_frames.extend(boundary_frames);
        }

        StreamMode::Macroblock => {
            let grid_h = h / BLOCK_SIZE;
            let grid_w = w / BLOCK_SIZE;
            let mut prev = vec![0i16; w * h];
            let mut bg_model = vec![0.0f32; w * h];
            let mut processed = vec![false; grid_h * grid_w];
            let mut frame_type_buf = [0u8; 1];

            loop {
                match stream.read_exact(&mut frame_type_buf) {
                    Ok(()) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                    Err(e) => return Err(e.into()),
                }

                let frame_type = frame_type_buf[0];
                if frame_type == 0 {
                    // I-frame (Keyframe)
                    let size = stream.read_u32::<LittleEndian>()? as usize;
                    let mut raw_pixels = vec![0u8; size];
                    stream.read_exact(&mut raw_pixels)?;

                    for (i, &b) in raw_pixels.iter().enumerate().take(w * h) {
                        prev[i] = b as i16;
                        bg_model[i] = b as f32;
                    }

                    let out_frame: Vec<u8> = prev.iter().map(|&v| v.clamp(0, 255) as u8).collect();
                    reconstructed_frames.push(out_frame);
                } else if frame_type == 1 {
                    // P-frame
                    let mut next_frame = prev.clone();
                    processed.fill(false);
                    let mut last_dy = 0i32;
                    let mut last_dx = 0i32;
                    let mut skip_remaining = 0usize;

                    let len_modes = stream.read_u32::<LittleEndian>()? as usize;
                    let len_payloads = stream.read_u32::<LittleEndian>()? as usize;

                    let mut mode_data = vec![0u8; len_modes];
                    stream.read_exact(&mut mode_data)?;

                    let mut payload_data = vec![0u8; len_payloads];
                    stream.read_exact(&mut payload_data)?;

                    let mut mode_idx = 0usize;
                    let mut pay_idx = 0usize;

                    for y_idx in 0..grid_h {
                        for x_idx in 0..grid_w {
                            let g_idx = y_idx * grid_w + x_idx;
                            if processed[g_idx] {
                                continue;
                            }
                            if skip_remaining > 0 {
                                skip_remaining -= 1;
                                if skip_remaining == 0 {
                                    continue;
                                }
                                processed[g_idx] = true;
                                continue;
                            }

                            let y_min = y_idx * BLOCK_SIZE;
                            let x_min = x_idx * BLOCK_SIZE;

                            if mode_idx >= mode_data.len() {
                                break;
                            }
                            let flag = mode_data[mode_idx];
                            mode_idx += 1;

                            if flag == MODE_SKIP {
                                skip_remaining = u16::from_le_bytes([
                                    payload_data[pay_idx],
                                    payload_data[pay_idx + 1],
                                ]) as usize;
                                pay_idx += 2;
                                skip_remaining -= 1;
                                processed[g_idx] = true;
                                continue;
                            }

                            let mut w_mult = 1usize;
                            let mut h_mult = 1usize;
                            let mut current_mode = flag;

                            if flag == MODE_DYNAMIC_BOX {
                                w_mult = payload_data[pay_idx] as usize;
                                h_mult = payload_data[pay_idx + 1] as usize;
                                current_mode = payload_data[pay_idx + 2];
                                pay_idx += 3;
                            }

                            let bh = h_mult * BLOCK_SIZE;
                            let bw = w_mult * BLOCK_SIZE;
                            let y_max = y_min + bh;
                            let x_max = x_min + bw;

                            match current_mode {
                                MODE_SOLID => {
                                    let val = payload_data[pay_idx] as i16;
                                    pay_idx += 1;
                                    for cy in y_min..y_max {
                                        let row_start = cy * w;
                                        for cx in x_min..x_max {
                                            next_frame[row_start + cx] = val;
                                        }
                                    }
                                }
                                MODE_BACKGROUND => {
                                    for cy in y_min..y_max {
                                        let row_start = cy * w;
                                        for cx in x_min..x_max {
                                            next_frame[row_start + cx] = bg_model[row_start + cx] as i16;
                                        }
                                    }
                                }
                                MODE_MOTION => {
                                    let dy = i16::from_le_bytes([
                                        payload_data[pay_idx],
                                        payload_data[pay_idx + 1],
                                    ]) as i32;
                                    let dx = i16::from_le_bytes([
                                        payload_data[pay_idx + 2],
                                        payload_data[pay_idx + 3],
                                    ]) as i32;
                                    pay_idx += 4;

                                    let src_y = (y_min as i32 - dy).clamp(0, (h.saturating_sub(bh)) as i32) as usize;
                                    let src_x = (x_min as i32 - dx).clamp(0, (w.saturating_sub(bw)) as i32) as usize;

                                    for cy in 0..bh {
                                        let dst_row = (y_min + cy) * w;
                                        let src_row = (src_y + cy) * w;
                                        for cx in 0..bw {
                                            next_frame[dst_row + x_min + cx] = prev[src_row + src_x + cx];
                                        }
                                    }
                                    last_dy = dy;
                                    last_dx = dx;
                                }
                                MODE_CACHED_MOTION => {
                                    let src_y = (y_min as i32 - last_dy).clamp(0, (h.saturating_sub(bh)) as i32) as usize;
                                    let src_x = (x_min as i32 - last_dx).clamp(0, (w.saturating_sub(bw)) as i32) as usize;

                                    for cy in 0..bh {
                                        let dst_row = (y_min + cy) * w;
                                        let src_row = (src_y + cy) * w;
                                        for cx in 0..bw {
                                            next_frame[dst_row + x_min + cx] = prev[src_row + src_x + cx];
                                        }
                                    }
                                }
                                MODE_FADED_MOTION => {
                                    let dy = i16::from_le_bytes([
                                        payload_data[pay_idx],
                                        payload_data[pay_idx + 1],
                                    ]) as i32;
                                    let dx = i16::from_le_bytes([
                                        payload_data[pay_idx + 2],
                                        payload_data[pay_idx + 3],
                                    ]) as i32;
                                    let shift = payload_data[pay_idx + 4] as i8 as i16;
                                    pay_idx += 5;

                                    let src_y = (y_min as i32 - dy).clamp(0, (h.saturating_sub(bh)) as i32) as usize;
                                    let src_x = (x_min as i32 - dx).clamp(0, (w.saturating_sub(bw)) as i32) as usize;

                                    for cy in 0..bh {
                                        let dst_row = (y_min + cy) * w;
                                        let src_row = (src_y + cy) * w;
                                        for cx in 0..bw {
                                            let val = prev[src_row + src_x + cx] + shift;
                                            next_frame[dst_row + x_min + cx] = val.clamp(0, 255);
                                        }
                                    }
                                    last_dy = dy;
                                    last_dx = dx;
                                }
                                MODE_DCT_LOW => {
                                    let mut coeffs_4x4 = [[0.0f32; 4]; 4];
                                    for i in 0..4 {
                                        for j in 0..4 {
                                            let raw_u16 = u16::from_le_bytes([
                                                payload_data[pay_idx],
                                                payload_data[pay_idx + 1],
                                            ]);
                                            pay_idx += 2;
                                            coeffs_4x4[i][j] = f16_to_f32(raw_u16);
                                        }
                                    }
                                    let idct_block = dct.idct_low_4x4(&coeffs_4x4);
                                    for cy in 0..16 {
                                        let dst_row = (y_min + cy) * w;
                                        for cx in 0..16 {
                                            next_frame[dst_row + x_min + cx] = idct_block[cy][cx].clamp(0.0, 255.0).round() as i16;
                                        }
                                    }
                                }
                                MODE_DCT_MID => {
                                    let mut coeffs_8x8 = [[0.0f32; 8]; 8];
                                    for i in 0..8 {
                                        for j in 0..8 {
                                            let raw_u16 = u16::from_le_bytes([
                                                payload_data[pay_idx],
                                                payload_data[pay_idx + 1],
                                            ]);
                                            pay_idx += 2;
                                            coeffs_8x8[i][j] = f16_to_f32(raw_u16);
                                        }
                                    }
                                    let idct_block = dct.idct_mid_8x8(&coeffs_8x8);
                                    for cy in 0..16 {
                                        let dst_row = (y_min + cy) * w;
                                        for cx in 0..16 {
                                            next_frame[dst_row + x_min + cx] = idct_block[cy][cx].clamp(0.0, 255.0).round() as i16;
                                        }
                                    }
                                }
                                MODE_FAST_DITHER => {
                                    let palette = [
                                        payload_data[pay_idx] as i16,
                                        payload_data[pay_idx + 1] as i16,
                                        payload_data[pay_idx + 2] as i16,
                                        payload_data[pay_idx + 3] as i16,
                                    ];
                                    pay_idx += 4;
                                    let packed = &payload_data[pay_idx..pay_idx + 64];
                                    pay_idx += 64;

                                    let mut unpacked = [0u8; 256];
                                    unpack_2bit_indices(packed, &mut unpacked);

                                    for cy in 0..16 {
                                        let dst_row = (y_min + cy) * w;
                                        let p_row = cy * 16;
                                        for cx in 0..16 {
                                            let idx = unpacked[p_row + cx] as usize;
                                            next_frame[dst_row + x_min + cx] = palette[idx];
                                        }
                                    }
                                }
                                MODE_PALETTE => {
                                    let _num_colors = payload_data[pay_idx];
                                    pay_idx += 1;
                                    let palette = [
                                        payload_data[pay_idx] as i16,
                                        payload_data[pay_idx + 1] as i16,
                                        payload_data[pay_idx + 2] as i16,
                                        payload_data[pay_idx + 3] as i16,
                                    ];
                                    pay_idx += 4;
                                    let packed = &payload_data[pay_idx..pay_idx + 64];
                                    pay_idx += 64;

                                    let mut unpacked = [0u8; 256];
                                    unpack_2bit_indices(packed, &mut unpacked);

                                    for cy in 0..16 {
                                        let dst_row = (y_min + cy) * w;
                                        let p_row = cy * 16;
                                        for cx in 0..16 {
                                            let idx = unpacked[p_row + cx] as usize;
                                            next_frame[dst_row + x_min + cx] = palette[idx];
                                        }
                                    }
                                }
                                MODE_SPATIAL => {
                                    let nnz = payload_data[pay_idx] as usize;
                                    pay_idx += 1;

                                    if nnz < 128 {
                                        let indices = &payload_data[pay_idx..pay_idx + nnz];
                                        pay_idx += nnz;
                                        let values = &payload_data[pay_idx..pay_idx + nnz];
                                        pay_idx += nnz;

                                        for i in 0..nnz {
                                            let p_idx = indices[i] as usize;
                                            let py = p_idx / 16;
                                            let px = p_idx % 16;
                                            let delta = (values[i] as i16) - 128;
                                            let dst_idx = (y_min + py) * w + (x_min + px);
                                            next_frame[dst_idx] = (next_frame[dst_idx] + delta * qp).clamp(0, 255);
                                        }
                                    } else {
                                        let data = &payload_data[pay_idx..pay_idx + 256];
                                        pay_idx += 256;
                                        for py in 0..16 {
                                            let dst_row = (y_min + py) * w;
                                            let p_row = py * 16;
                                            for px in 0..16 {
                                                let delta = (data[p_row + px] as i16) - 128;
                                                let dst_idx = dst_row + (x_min + px);
                                                next_frame[dst_idx] = (next_frame[dst_idx] + delta * qp).clamp(0, 255);
                                            }
                                        }
                                    }
                                }
                                MODE_RAW => {
                                    let len = bh * bw;
                                    let raw = &payload_data[pay_idx..pay_idx + len];
                                    pay_idx += len;

                                    for cy in 0..bh {
                                        let dst_row = (y_min + cy) * w;
                                        let src_row = cy * bw;
                                        for cx in 0..bw {
                                            next_frame[dst_row + x_min + cx] = raw[src_row + cx] as i16;
                                        }
                                    }
                                }
                                other => {
                                    bail!("Unknown macroblock mode encountered: {}", other);
                                }
                            }

                            // Mark processed for boxing
                            for dy in 0..h_mult {
                                let row = (y_idx + dy) * grid_w;
                                for dx in 0..w_mult {
                                    processed[row + x_idx + dx] = true;
                                }
                            }
                        }
                    }

                    // Background model update
                    if has_bg {
                        for i in 0..(w * h) {
                            let nf = next_frame[i];
                            let pf = prev[i];
                            let bg = bg_model[i];
                            if (nf - pf).abs() < 2 && (nf as f32 - bg).abs() < 4.0 {
                                bg_model[i] = 0.95 * bg + 0.05 * (nf as f32);
                            }
                        }
                    }

                    prev = next_frame;
                    let out_frame: Vec<u8> = prev.iter().map(|&v| v.clamp(0, 255) as u8).collect();
                    reconstructed_frames.push(out_frame);
                }
            }
        }
    }

    let dec_time_s = start_time.elapsed().as_secs_f64();

    Ok(DecodedVideo {
        width: w,
        height: h,
        total_frames: reconstructed_frames.len(),
        is_rgb,
        frames: reconstructed_frames,
        dec_time_s,
    })
}
