use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::*;
use std::path::PathBuf;
use std::time::Instant;
use videoretalking_core::{
    EnhancerModel, MelProcessor, VideoReTalkingConfig, VideoReTalkingPipeline, WeightsManager,
};

#[derive(Parser, Debug)]
#[command(
    name = "videoretalking",
    author = "VideoReTalking Contributors",
    version = "0.1.0",
    about = "High-performance Rust CLI for VideoReTalking: Audio-based Lip Synchronization for Video Editing"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Synchronize mouth movements in input video to match driving audio
    Retalk {
        /// Source video file
        #[arg(short = 'v', long = "face-video", required = true)]
        face_video: PathBuf,

        /// Driving audio WAV file
        #[arg(short = 'a', long = "audio", required = true)]
        audio: PathBuf,

        /// Destination path for retalked video output
        #[arg(short = 'o', long = "output", default_value = "output_retalked.mp4")]
        output: PathBuf,

        /// Canonical face resolution (default: 256)
        #[arg(long = "face-size", default_value_t = 256)]
        face_size: u32,

        /// Preserve upper-half face natural motion
        #[arg(long = "up-half", default_value_t = true)]
        up_half: bool,
    },

    /// Inspect video, audio features, or SafeTensors checkpoints
    Inspect {
        /// Target file to inspect
        #[arg(required = true)]
        file: PathBuf,
    },

    /// Benchmark audio Mel-spectrogram processing and lip sync planning
    Bench {
        /// Optional audio WAV file
        #[arg(short = 'a', long = "audio")]
        audio: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Retalk {
            face_video,
            audio,
            output,
            face_size,
            up_half,
        } => {
            println!("{}", "=== VideoReTalking Native Rust Pipeline ===".bold().cyan());
            println!("  Input Video:  {}", face_video.display().to_string().yellow());
            println!("  Driven Audio: {}", audio.display().to_string().yellow());
            println!("  Face Size:    {}x{}", face_size, face_size);
            println!("  Up-Half Mode: {}", up_half);

            let config = VideoReTalkingConfig {
                face_size,
                fps: 25.0,
                up_half,
                smooth_window: 5,
                enhancer: EnhancerModel::None,
                batch_size: 4,
                checkpoint_dir: None,
            };

            let start = Instant::now();
            let pipeline = VideoReTalkingPipeline::new(config);
            let out_file = pipeline.run(&face_video, &audio, &output)?;

            println!(
                "{} Video retalking complete in {:.2?} -> {}",
                "✓".green(),
                start.elapsed(),
                out_file.display().to_string().bold()
            );
        }

        Commands::Inspect { file } => {
            println!("{} {}", "Inspecting:".bold().cyan(), file.display());
            let ext = file.extension().and_then(|s| s.to_str()).unwrap_or("");

            match ext.to_lowercase().as_str() {
                "wav" => {
                    let samples = MelProcessor::load_wav(&file)?;
                    let info = MelProcessor::process_lip_sync_windows(&samples, 25.0);
                    println!("  Type:          WAV Audio Track");
                    println!("  Sample Rate:   {} Hz", info.sample_rate);
                    println!("  Duration:      {:.2} seconds", info.duration_seconds);
                    println!("  Total Samples: {}", info.total_samples);
                    println!("  Sync Windows:  {} (@ 25 FPS)", info.sync_windows.len());
                }
                "safetensors" => {
                    let tensors = WeightsManager::inspect_checkpoint(&file)?;
                    println!("  Type:          SafeTensors Checkpoint");
                    println!("  Tensors Count: {}", tensors.len());
                    for t in tensors.iter().take(10) {
                        println!("    • {:<35} {:?} ({})", t.name, t.shape, t.dtype);
                    }
                    if tensors.len() > 10 {
                        println!("    ... and {} more tensors", tensors.len() - 10);
                    }
                }
                _ => {
                    println!("File extension .{} not supported for detailed inspection", ext);
                }
            }
        }

        Commands::Bench { audio } => {
            println!("{}", "=== Benchmarking VideoReTalking Core ===".bold().cyan());
            let samples = if let Some(p) = audio {
                MelProcessor::load_wav(&p)?
            } else {
                (0..160_000).map(|i| (i as f32 * 0.05).sin() * 0.4).collect()
            };

            let duration = samples.len() as f32 / 16000.0;
            println!("Testing on {:.1} seconds of 16kHz audio...", duration);

            let t0 = Instant::now();
            let track_info = MelProcessor::process_lip_sync_windows(&samples, 25.0);
            let mel_dur = t0.elapsed();

            let engine = videoretalking_core::LipSyncEngine::new(true);
            let t1 = Instant::now();
            let plans = engine.plan_sequence(&track_info.sync_windows);
            let plan_dur = t1.elapsed();

            println!("\nResults:");
            println!("  Mel-Spectrogram Processing: {:.2?} ({} frames, {:.1} fps throughput)",
                mel_dur, track_info.total_frames, track_info.total_frames as f64 / mel_dur.as_secs_f64());
            println!("  Lip Sync Plan Generation:   {:.2?} ({} frames, {:.1} fps throughput)",
                plan_dur, plans.len(), plans.len() as f64 / plan_dur.as_secs_f64());
        }
    }

    Ok(())
}
