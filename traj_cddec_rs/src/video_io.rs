use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use anyhow::{bail, Context, Result};

#[derive(Debug, Clone)]
pub struct Frame {
    pub width: usize,
    pub height: usize,
    pub channels: usize, // 1 for grayscale, 3 for RGB
    pub data: Vec<u8>,
}

impl Frame {
    pub fn new_gray(width: usize, height: usize, fill: u8) -> Self {
        Self {
            width,
            height,
            channels: 1,
            data: vec![fill; width * height],
        }
    }

    pub fn new_rgb(width: usize, height: usize, fill_r: u8, fill_g: u8, fill_b: u8) -> Self {
        let mut data = Vec::with_capacity(width * height * 3);
        for _ in 0..(width * height) {
            data.push(fill_r);
            data.push(fill_g);
            data.push(fill_b);
        }
        Self {
            width,
            height,
            channels: 3,
            data,
        }
    }

    pub fn to_grayscale(&self) -> Vec<u8> {
        if self.channels == 1 {
            self.data.clone()
        } else if self.channels == 3 {
            let mut gray = Vec::with_capacity(self.width * self.height);
            for chunk in self.data.chunks_exact(3) {
                // Standard ITU-R BT.601 luma conversion: Y = 0.299 R + 0.587 G + 0.114 B
                let y = ((chunk[0] as u32 * 77 + chunk[1] as u32 * 150 + chunk[2] as u32 * 29) >> 8) as u8;
                gray.push(y);
            }
            gray
        } else {
            self.data.clone()
        }
    }
}

#[derive(Debug, Clone)]
pub struct VideoMetadata {
    pub width: usize,
    pub height: usize,
    pub total_frames: usize,
    pub fps: f64,
}

pub fn get_video_metadata<P: AsRef<Path>>(path: P) -> Result<VideoMetadata> {
    let p_str = path.as_ref().to_string_lossy();
    let output = Command::new("ffprobe")
        .args([
            "-v", "error",
            "-select_streams", "v:0",
            "-count_packets",
            "-show_entries", "stream=width,height,nb_read_packets,r_frame_rate",
            "-of", "csv=p=0",
            &p_str,
        ])
        .output()
        .context("Failed to execute ffprobe. Make sure FFmpeg is in PATH.")?;

    if !output.status.success() {
        bail!("ffprobe failed on {}: {}", p_str, String::from_utf8_lossy(&output.stderr));
    }

    let out_str = String::from_utf8_lossy(&output.stdout);
    let parts: Vec<&str> = out_str.trim().split(',').collect();
    if parts.len() < 3 {
        bail!("Unexpected ffprobe output: {}", out_str);
    }

    let width: usize = parts[0].parse().unwrap_or(0);
    let height: usize = parts[1].parse().unwrap_or(0);

    // Rate could be "30/1" or "29.97"
    let fps = if parts[2].contains('/') {
        let frac: Vec<&str> = parts[2].split('/').collect();
        let num: f64 = frac[0].parse().unwrap_or(30.0);
        let den: f64 = frac.get(1).and_then(|d| d.parse().ok()).unwrap_or(1.0);
        if den > 0.0 { num / den } else { 30.0 }
    } else {
        parts[2].parse().unwrap_or(30.0)
    };

    let total_frames: usize = parts.get(3).and_then(|n| n.parse().ok()).unwrap_or(0);

    Ok(VideoMetadata {
        width,
        height,
        total_frames,
        fps,
    })
}

pub fn read_video_frames_grayscale<P: AsRef<Path>>(
    path: P,
    max_frames: Option<usize>,
) -> Result<(VideoMetadata, Vec<Vec<u8>>)> {
    let meta = get_video_metadata(path.as_ref())?;
    let p_str = path.as_ref().to_string_lossy();

    // Round height and width to multiples of 16
    let aligned_w = meta.width - (meta.width % 16);
    let aligned_h = meta.height - (meta.height % 16);

    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-i", &p_str]);
    if let Some(mf) = max_frames {
        cmd.args(["-vframes", &mf.to_string()]);
    }
    // Crop if needed to align to 16x16
    if aligned_w != meta.width || aligned_h != meta.height {
        cmd.args(["-vf", &format!("crop={}:{}:0:0", aligned_w, aligned_h)]);
    }
    cmd.args(["-f", "rawvideo", "-pix_fmt", "gray", "-"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null());

    let mut child = cmd.spawn().context("Failed to spawn ffmpeg child process")?;
    let mut stdout = child.stdout.take().context("Failed to capture ffmpeg stdout")?;

    let frame_size = aligned_w * aligned_h;
    let mut frames = Vec::new();
    let mut buf = vec![0u8; frame_size];

    loop {
        match stdout.read_exact(&mut buf) {
            Ok(()) => frames.push(buf.clone()),
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
    }

    let _ = child.wait();

    let final_meta = VideoMetadata {
        width: aligned_w,
        height: aligned_h,
        total_frames: frames.len(),
        fps: meta.fps,
    };

    Ok((final_meta, frames))
}

pub fn read_video_frames_rgb<P: AsRef<Path>>(
    path: P,
    max_frames: Option<usize>,
) -> Result<(VideoMetadata, Vec<Vec<u8>>)> {
    let meta = get_video_metadata(path.as_ref())?;
    let p_str = path.as_ref().to_string_lossy();

    let aligned_w = meta.width - (meta.width % 16);
    let aligned_h = meta.height - (meta.height % 16);

    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-i", &p_str]);
    if let Some(mf) = max_frames {
        cmd.args(["-vframes", &mf.to_string()]);
    }
    if aligned_w != meta.width || aligned_h != meta.height {
        cmd.args(["-vf", &format!("crop={}:{}:0:0", aligned_w, aligned_h)]);
    }
    cmd.args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null());

    let mut child = cmd.spawn().context("Failed to spawn ffmpeg child process")?;
    let mut stdout = child.stdout.take().context("Failed to capture ffmpeg stdout")?;

    let frame_size = aligned_w * aligned_h * 3;
    let mut frames = Vec::new();
    let mut buf = vec![0u8; frame_size];

    loop {
        match stdout.read_exact(&mut buf) {
            Ok(()) => frames.push(buf.clone()),
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
    }

    let _ = child.wait();

    let final_meta = VideoMetadata {
        width: aligned_w,
        height: aligned_h,
        total_frames: frames.len(),
        fps: meta.fps,
    };

    Ok((final_meta, frames))
}

pub fn save_frames_to_mp4<P: AsRef<Path>>(
    frames: &[Vec<u8>],
    width: usize,
    height: usize,
    is_rgb: bool,
    fps: f64,
    out_path: P,
) -> Result<()> {
    if let Some(parent) = out_path.as_ref().parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let p_str = out_path.as_ref().to_string_lossy();
    let pix_fmt_in = if is_rgb { "rgb24" } else { "gray" };

    let mut child = Command::new("ffmpeg")
        .args([
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", &format!("{}x{}", width, height),
            "-pix_fmt", pix_fmt_in,
            "-r", &fps.to_string(),
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            &p_str,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("Failed to spawn ffmpeg to write video")?;

    {
        let mut stdin = child.stdin.take().context("Failed to open stdin for ffmpeg")?;
        for frame in frames {
            stdin.write_all(frame)?;
        }
    }

    let status = child.wait()?;
    if !status.success() {
        bail!("ffmpeg failed with status: {}", status);
    }

    Ok(())
}

pub fn colorize_luma_with_orig_rgb(dec_luma: &[u8], orig_rgb: &[u8]) -> Vec<u8> {
    let mut out_rgb = vec![0u8; orig_rgb.len()];
    for (i, (&y_dec, chunk)) in dec_luma.iter().zip(orig_rgb.chunks_exact(3)).enumerate() {
        let r = chunk[0] as i32;
        let g = chunk[1] as i32;
        let b = chunk[2] as i32;
        let cb = 128 + ((-43 * r - 85 * g + 128 * b) >> 8);
        let cr = 128 + ((128 * r - 107 * g - 21 * b) >> 8);

        let y = y_dec as i32;
        let r_out = (y + ((359 * (cr - 128)) >> 8)).clamp(0, 255) as u8;
        let g_out = (y - ((88 * (cb - 128) + 183 * (cr - 128)) >> 8)).clamp(0, 255) as u8;
        let b_out = (y + ((454 * (cb - 128)) >> 8)).clamp(0, 255) as u8;

        let o_idx = i * 3;
        out_rgb[o_idx] = r_out;
        out_rgb[o_idx + 1] = g_out;
        out_rgb[o_idx + 2] = b_out;
    }
    out_rgb
}

pub fn save_side_by_side_video<P: AsRef<Path>, Q: AsRef<Path>>(
    orig_path: P,
    dec_frames: &[Vec<u8>],
    width: usize,
    height: usize,
    is_rgb: bool,
    fps: f64,
    out_path: Q,
) -> Result<()> {
    if let Some(parent) = out_path.as_ref().parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let p_str = out_path.as_ref().to_string_lossy();
    let divider_w = 4usize;
    let canvas_w = width * 2 + divider_w;

    // Try reading original in RGB so we can produce full color side-by-side comparison
    if let Ok((_, orig_frames_rgb)) = read_video_frames_rgb(orig_path.as_ref(), Some(dec_frames.len())) {
        let mut child = Command::new("ffmpeg")
            .args([
                "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", &format!("{}x{}", canvas_w, height),
                "-pix_fmt", "rgb24",
                "-r", &fps.to_string(),
                "-i", "-",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                &p_str,
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .context("Failed to spawn ffmpeg for side-by-side video")?;

        {
            let mut stdin = child.stdin.take().context("Failed to open stdin for ffmpeg")?;
            let mut canvas = vec![0u8; canvas_w * height * 3];

            for (of, df) in orig_frames_rgb.iter().zip(dec_frames.iter()) {
                // If decoded frame is grayscale, colorize with original chroma
                let df_rgb = if is_rgb {
                    df.clone()
                } else {
                    colorize_luma_with_orig_rgb(df, of)
                };

                for y in 0..height {
                    let c_row = y * canvas_w * 3;
                    let o_row = y * width * 3;
                    let d_row = y * width * 3;

                    // Left: Original
                    canvas[c_row..c_row + width * 3].copy_from_slice(&of[o_row..o_row + width * 3]);

                    // Divider: bright cyan [0, 255, 255]
                    for dx in 0..divider_w {
                        let idx = c_row + (width + dx) * 3;
                        canvas[idx] = 0;
                        canvas[idx + 1] = 255;
                        canvas[idx + 2] = 255;
                    }

                    // Right: Decoded (color-restored)
                    let r_start = c_row + (width + divider_w) * 3;
                    canvas[r_start..r_start + width * 3].copy_from_slice(&df_rgb[d_row..d_row + width * 3]);
                }
                stdin.write_all(&canvas)?;
            }
        }
        let status = child.wait()?;
        if !status.success() {
            bail!("ffmpeg side-by-side failed with status: {}", status);
        }
        return Ok(());
    }

    // Fallback to pure grayscale if RGB extraction failed
    let (_, orig_frames) = read_video_frames_grayscale(orig_path, Some(dec_frames.len()))?;
    let mut child = Command::new("ffmpeg")
        .args([
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", &format!("{}x{}", canvas_w, height),
            "-pix_fmt", "gray",
            "-r", &fps.to_string(),
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            &p_str,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("Failed to spawn ffmpeg for side-by-side video")?;

    {
        let mut stdin = child.stdin.take().context("Failed to open stdin for ffmpeg")?;
        let mut canvas = vec![0u8; canvas_w * height];

        for (of, df) in orig_frames.iter().zip(dec_frames.iter()) {
            for y in 0..height {
                let c_row = y * canvas_w;
                let o_row = y * width;
                let d_row = y * width;

                // Left: Original
                canvas[c_row..c_row + width].copy_from_slice(&of[o_row..o_row + width]);

                // Divider: white 255
                for dx in 0..divider_w {
                    canvas[c_row + width + dx] = 255;
                }

                // Right: Decoded
                canvas[c_row + width + divider_w..c_row + canvas_w].copy_from_slice(&df[d_row..d_row + width]);
            }
            stdin.write_all(&canvas)?;
        }
    }
    let status = child.wait()?;
    if !status.success() {
        bail!("ffmpeg side-by-side grayscale failed with status: {}", status);
    }
    Ok(())
}
