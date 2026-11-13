use std::path::{Path, PathBuf};
use anyhow::Result;
use clap::{Parser, Subcommand};

use traj_cddec_rs::bitstream::NamHeader;
use traj_cddec_rs::decoder::decode_file;
use traj_cddec_rs::encoder::{encode_file, EncoderConfig};
use traj_cddec_rs::metrics::{calculate_frame_mse, calculate_psnr, VideoBenchmarkSummary};
use traj_cddec_rs::video_io::{
    colorize_luma_with_orig_rgb, read_video_frames_grayscale, read_video_frames_rgb,
    save_frames_to_mp4, save_side_by_side_video,
};

#[derive(Parser, Debug)]
#[command(name = "traj_cddec_rs")]
#[command(author = "TrajCDDec Team")]
#[command(version = "1.0.0")]
#[command(about = "High-performance Trajectory Video Codec (V5/V6) in pure Rust", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Encode an MP4 video to a .nam bitstream
    Encode {
        #[arg(short, long, help = "Input video file (.mp4)")]
        input: PathBuf,

        #[arg(short, long, help = "Output compressed bitstream (.nam)")]
        output: PathBuf,

        #[arg(long, default_value_t = 27, help = "Quantization parameter")]
        qp: u16,

        #[arg(long, default_value_t = 0.012, help = "Rate-distortion lambda")]
        lambda: f64,

        #[arg(long, help = "Disable dynamic macroblock boxing")]
        no_boxing: bool,

        #[arg(long, help = "Force layered trajectory mode (Stream Type 1)")]
        layered: bool,
    },

    /// Decode a .nam bitstream to an MP4 video or verify reconstruction
    Decode {
        #[arg(short, long, help = "Input compressed bitstream (.nam)")]
        input: PathBuf,

        #[arg(short, long, help = "Output reconstructed video (.mp4)")]
        output: Option<PathBuf>,

        #[arg(long, help = "Save side-by-side comparison video (.mp4) with original")]
        side_by_side: Option<PathBuf>,

        #[arg(long, help = "Optional original video (.mp4) for MSE and PSNR calculation")]
        orig: Option<PathBuf>,

        #[arg(long, default_value_t = 30.0, help = "Output framerate (FPS)")]
        fps: f64,
    },

    /// Benchmark encoding, decoding, compression ratio, and PSNR for a video
    Benchmark {
        #[arg(short, long, help = "Path to a single .mp4 video file")]
        video: Option<PathBuf>,

        #[arg(short, long, help = "Path to folder of .mp4 videos")]
        folder: Option<PathBuf>,
    },

    /// Inspect and print telemetry analysis of a .nam bitstream
    Analyze {
        #[arg(short, long, help = "Path to .nam file")]
        input: PathBuf,
    },

    /// Run batch benchmark across a folder of test videos
    Batch {
        #[arg(short, long, help = "Path to folder containing test videos")]
        folder: PathBuf,

        #[arg(long, help = "Save decoded and side-by-side comparison MP4 videos")]
        save_videos: bool,
    },
}

fn run_benchmark_video(video_path: &Path) -> Result<Option<VideoBenchmarkSummary>> {
    let fname = video_path.file_name().unwrap_or_default().to_string_lossy();
    println!("\n{}", "=".repeat(60));
    println!("{:^60}", format!("BENCHMARK: {}", fname));
    println!("{}", "=".repeat(60));

    let tmp_nam = video_path.with_extension("tmp_bench.nam");
    let config = EncoderConfig::default();

    println!("Starting Rust Encoder (V6 Hybrid)...");
    let enc_stats = encode_file(video_path, &tmp_nam, &config)?;

    let orig_mb = enc_stats.orig_bytes as f64 / (1024.0 * 1024.0);
    let comp_mb = enc_stats.compressed_bytes as f64 / (1024.0 * 1024.0);
    let ratio = (enc_stats.compressed_bytes as f64 / enc_stats.orig_bytes as f64) * 100.0;
    let enc_fps = enc_stats.total_frames as f64 / enc_stats.encode_time_s;

    println!("\n[ENCODER RESULTS]");
    println!("Original Size:   {:.2} MB ({} bytes)", orig_mb, enc_stats.orig_bytes);
    println!("Compressed Size: {:.2} MB ({} bytes)", comp_mb, enc_stats.compressed_bytes);
    println!("Data Reduction:  {:.2}%", 100.0 - ratio);
    println!("Encoding Speed:  {:.2} FPS ({:.3} s)", enc_fps, enc_stats.encode_time_s);

    println!("\n[DECODING]");
    println!("Verifying bitstream integrity & reconstruction for '{}'...", fname);

    let dec_res = decode_file(&tmp_nam)?;
    let dec_fps = dec_res.total_frames as f64 / dec_res.dec_time_s;

    // Calculate MSE against original frames
    let (_, orig_frames) = read_video_frames_grayscale(video_path, None)?;
    let mut sum_mse = 0.0f64;
    let mut count = 0usize;

    for (df, of) in dec_res.frames.iter().zip(orig_frames.iter()) {
        let df_gray = if dec_res.is_rgb {
            let mut gray = Vec::with_capacity(df.len() / 3);
            for chunk in df.chunks_exact(3) {
                let y = ((chunk[0] as u32 * 77 + chunk[1] as u32 * 150 + chunk[2] as u32 * 29) >> 8) as u8;
                gray.push(y);
            }
            gray
        } else {
            df.clone()
        };
        let mse = calculate_frame_mse(of, &df_gray);
        sum_mse += mse;
        count += 1;
    }

    let avg_mse = if count > 0 { sum_mse / count as f64 } else { 0.0 };
    let psnr = calculate_psnr(avg_mse);

    println!("\n[DECODER RESULTS]");
    println!("Frames Decoded:  {}", dec_res.total_frames);
    println!("Decoding Speed:  {:.2} FPS ({:.3} s)", dec_fps, dec_res.dec_time_s);
    println!("Average MSE:     {:.4}", avg_mse);
    println!("Recon PSNR:      {:.2} dB", psnr);

    // Clean up temporary bench bitstream
    let _ = std::fs::remove_file(&tmp_nam);

    println!("{}", "=".repeat(60));
    println!("{:^60}", "BENCHMARK COMPLETE");
    println!("{}", "=".repeat(60));

    Ok(Some(VideoBenchmarkSummary::new(
        fname.to_string(),
        enc_stats.orig_bytes,
        enc_stats.compressed_bytes,
        enc_stats.total_frames,
        enc_stats.encode_time_s,
        dec_res.dec_time_s,
        avg_mse,
    )))
}

fn run_batch_benchmark(folder: &Path, save_videos: bool) -> Result<()> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(folder)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            if let Some(ext) = p.extension() {
                if ext == "mp4" {
                    let name = p.file_name().unwrap_or_default().to_string_lossy();
                    return !name.contains("_2f")
                        && !name.contains("_5f")
                        && !name.contains("_10f")
                        && !name.contains("05_break_entropy");
                }
            }
            false
        })
        .collect();

    entries.sort();

    if entries.is_empty() {
        println!("No suitable .mp4 files found in {:?}", folder);
        return Ok(());
    }

    let out_dir = PathBuf::from("traj_cddec_rs").join("output");
    if save_videos {
        let _ = std::fs::create_dir_all(&out_dir);
    }

    let mut summaries = Vec::new();
    for video in &entries {
        let fname = video.file_name().unwrap_or_default().to_string_lossy();
        if save_videos {
            let enc_out = out_dir.join(format!("{}.nam", fname));
            let dec_out = out_dir.join(format!("{}_decoded.mp4", fname));
            let sbs_out = out_dir.join(format!("{}_comparison.mp4", fname));

            let config = EncoderConfig::default();
            println!("\n>> [Rust Workflow] Encoding {:?} -> {:?}", video, enc_out);
            let enc_stats = encode_file(video, &enc_out, &config)?;
            println!(">> [Rust Workflow] Decoding & Rendering {:?} -> {:?}", enc_out, dec_out);
            let dec_res = decode_file(&enc_out)?;

            save_frames_to_mp4(&dec_res.frames, dec_res.width, dec_res.height, dec_res.is_rgb, 30.0, &dec_out)?;
            save_side_by_side_video(video, &dec_res.frames, dec_res.width, dec_res.height, dec_res.is_rgb, 30.0, &sbs_out)?;
            println!(">> Successfully rendered Rust decoded & side-by-side comparison video!");
        }

        if let Ok(Some(s)) = run_benchmark_video(video) {
            summaries.push(s);
        }
    }

    println!("\n{}", "=".repeat(105));
    println!("{:^105}", "RUST TRAJCDDEC CORPUS BENCHMARK SUMMARY");
    println!("{}", "=".repeat(105));
    println!(
        "{:<24} | {:<10} | {:<10} | {:<10} | {:<8} | {:<8} | {:<8} | {:<8}",
        "Video", "Orig Size", "Comp Size", "Reduction", "Enc FPS", "Dec FPS", "MSE", "PSNR"
    );
    println!("{}", "-".repeat(105));

    let mut total_orig = 0usize;
    let mut total_comp = 0usize;
    let mut sum_mse = 0.0f64;

    for s in &summaries {
        total_orig += s.orig_bytes;
        total_comp += s.comp_bytes;
        sum_mse += s.avg_mse;
        println!(
            "{:<24} | {:>8} B | {:>8} B | {:>8.1}% | {:>7.1}F | {:>7.1}F | {:>8.4} | {:>6.2}dB",
            s.name, s.orig_bytes, s.comp_bytes, s.reduction_pct, s.enc_fps, s.dec_fps, s.avg_mse, s.psnr
        );
    }

    let total_red = if total_orig > 0 {
        (1.0 - (total_comp as f64 / total_orig as f64)) * 100.0
    } else {
        0.0
    };
    let avg_total_mse = if !summaries.is_empty() { sum_mse / summaries.len() as f64 } else { 0.0 };
    let avg_total_psnr = calculate_psnr(avg_total_mse);

    println!("{}", "=".repeat(105));
    println!(
        "{:<24} | {:>8} B | {:>8} B | {:>8.1}% | {:>7}  | {:>7}  | {:>8.4} | {:>6.2}dB",
        "Total Corpus", total_orig, total_comp, total_red, "-", "-", avg_total_mse, avg_total_psnr
    );
    println!("{}", "=".repeat(105));

    Ok(())
}

fn analyze_nam_file(path: &Path) -> Result<()> {
    use std::fs::File;
    use std::io::BufReader;

    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let header = NamHeader::read(&mut reader)?;

    let file_size = std::fs::metadata(path)?.len();

    println!("--------------------------------------------------");
    println!(" Telemetry Analysis for: {:?}", path);
    println!("--------------------------------------------------");
    println!("Bitstream Version: {:?}", header.version);
    println!("Resolution:        {} x {}", header.width, header.height);
    println!("Total Frames:      {}", header.total_frames);
    println!("Quantization (QP): {}", header.qp);
    println!("Stream Mode:       {:?}", header.stream_mode);
    println!("Has Background:    {}", header.has_bg);
    println!("File Size on Disk: {} bytes ({:.2} KB)", file_size, file_size as f64 / 1024.0);
    println!("--------------------------------------------------");

    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Encode {
            input,
            output,
            qp,
            lambda,
            no_boxing,
            layered,
        } => {
            println!("Encoding {:?} -> {:?}", input, output);
            let config = EncoderConfig {
                qp,
                lambda,
                dynamic_boxing: !no_boxing,
                use_layered: if layered { Some(true) } else { None },
            };
            let stats = encode_file(&input, &output, &config)?;
            println!("Successfully encoded {} frames in {:.2}s ({:.1} FPS)", stats.total_frames, stats.encode_time_s, stats.total_frames as f64 / stats.encode_time_s);
            println!("Compressed size: {} bytes ({:.1}% ratio)", stats.compressed_bytes, (stats.compressed_bytes as f64 / stats.orig_bytes as f64) * 100.0);
        }

        Commands::Decode {
            input,
            output,
            side_by_side,
            orig,
            fps,
        } => {
            println!("Decoding {:?}", input);
            let res = decode_file(&input)?;
            println!("Successfully decoded {} frames in {:.3}s ({:.1} FPS)", res.total_frames, res.dec_time_s, res.total_frames as f64 / res.dec_time_s);

            if let Some(ref orig_path) = orig {
                let (_, orig_frames) = read_video_frames_grayscale(orig_path, None)?;
                let mut sum_mse = 0.0;
                let mut count = 0;
                for (df, of) in res.frames.iter().zip(orig_frames.iter()) {
                    let df_gray = if res.is_rgb {
                        let mut gray = Vec::with_capacity(df.len() / 3);
                        for chunk in df.chunks_exact(3) {
                            let y = ((chunk[0] as u32 * 77 + chunk[1] as u32 * 150 + chunk[2] as u32 * 29) >> 8) as u8;
                            gray.push(y);
                        }
                        gray
                    } else {
                        df.clone()
                    };
                    sum_mse += calculate_frame_mse(of, &df_gray);
                    count += 1;
                }
                let avg_mse = if count > 0 { sum_mse / count as f64 } else { 0.0 };
                let psnr = calculate_psnr(avg_mse);
                println!("Average MSE: {:.4} | PSNR: {:.2} dB", avg_mse, psnr);
            }

            if let Some(ref out_path) = output {
                println!("Saving reconstructed video to {:?}", out_path);
                if !res.is_rgb {
                    if let Some(ref orig_path) = orig {
                        if let Ok((_, orig_frames_rgb)) = read_video_frames_rgb(orig_path, Some(res.frames.len())) {
                            let colorized: Vec<Vec<u8>> = res.frames
                                .iter()
                                .zip(orig_frames_rgb.iter())
                                .map(|(df, of)| colorize_luma_with_orig_rgb(df, of))
                                .collect();
                            save_frames_to_mp4(&colorized, res.width, res.height, true, fps, out_path)?;
                            println!("Video saved successfully with original chroma restored.");
                        } else {
                            save_frames_to_mp4(&res.frames, res.width, res.height, false, fps, out_path)?;
                            println!("Video saved successfully.");
                        }
                    } else {
                        save_frames_to_mp4(&res.frames, res.width, res.height, false, fps, out_path)?;
                        println!("Video saved successfully.");
                    }
                } else {
                    save_frames_to_mp4(&res.frames, res.width, res.height, true, fps, out_path)?;
                    println!("Video saved successfully.");
                }
            }

            if let Some(ref sbs_path) = side_by_side {
                if let Some(ref orig_path) = orig {
                    println!("Saving side-by-side comparison video to {:?}", sbs_path);
                    save_side_by_side_video(orig_path, &res.frames, res.width, res.height, res.is_rgb, fps, sbs_path)?;
                    println!("Side-by-side video saved successfully.");
                } else {
                    println!("Warning: --orig required to generate side-by-side video.");
                }
            }
        }

        Commands::Benchmark { video, folder } => {
            if let Some(v) = video {
                run_benchmark_video(&v)?;
            } else if let Some(f) = folder {
                run_batch_benchmark(&f, false)?;
            } else {
                let default_dir = PathBuf::from("test_vid");
                if default_dir.exists() {
                    run_batch_benchmark(&default_dir, false)?;
                } else {
                    println!("Please specify --video or --folder");
                }
            }
        }

        Commands::Analyze { input } => {
            analyze_nam_file(&input)?;
        }

        Commands::Batch { folder, save_videos } => {
            run_batch_benchmark(&folder, save_videos)?;
        }
    }

    Ok(())
}
