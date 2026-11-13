#[derive(Debug, Clone, Copy)]
pub struct QualityMetrics {
    pub mse: f64,
    pub psnr: f64,
}

pub fn calculate_frame_mse(orig: &[u8], dec: &[u8]) -> f64 {
    if orig.is_empty() || orig.len() != dec.len() {
        return 0.0;
    }
    let sum_sq: f64 = orig
        .iter()
        .zip(dec.iter())
        .map(|(&a, &b)| {
            let diff = a as f64 - b as f64;
            diff * diff
        })
        .sum();
    sum_sq / orig.len() as f64
}

pub fn calculate_psnr(mse: f64) -> f64 {
    if mse <= 1e-12 {
        f64::INFINITY
    } else {
        10.0 * ((255.0 * 255.0) / mse).log10()
    }
}

pub fn calculate_metrics(orig: &[u8], dec: &[u8]) -> QualityMetrics {
    let mse = calculate_frame_mse(orig, dec);
    let psnr = calculate_psnr(mse);
    QualityMetrics { mse, psnr }
}

#[derive(Debug, Clone)]
pub struct VideoBenchmarkSummary {
    pub name: String,
    pub orig_bytes: usize,
    pub comp_bytes: usize,
    pub reduction_pct: f64,
    pub enc_fps: f64,
    pub dec_fps: f64,
    pub avg_mse: f64,
    pub psnr: f64,
}

impl VideoBenchmarkSummary {
    pub fn new(
        name: String,
        orig_bytes: usize,
        comp_bytes: usize,
        total_frames: usize,
        enc_time_s: f64,
        dec_time_s: f64,
        avg_mse: f64,
    ) -> Self {
        let reduction_pct = if orig_bytes > 0 {
            (1.0 - (comp_bytes as f64 / orig_bytes as f64)) * 100.0
        } else {
            0.0
        };
        let enc_fps = if enc_time_s > 0.0 {
            total_frames as f64 / enc_time_s
        } else {
            0.0
        };
        let dec_fps = if dec_time_s > 0.0 {
            total_frames as f64 / dec_time_s
        } else {
            0.0
        };
        let psnr = calculate_psnr(avg_mse);

        Self {
            name,
            orig_bytes,
            comp_bytes,
            reduction_pct,
            enc_fps,
            dec_fps,
            avg_mse,
            psnr,
        }
    }
}
