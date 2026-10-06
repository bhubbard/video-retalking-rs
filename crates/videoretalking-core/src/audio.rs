use anyhow::{Context, Result};
use hound::WavReader;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use std::path::Path;

pub const STANDARD_SAMPLE_RATE: u32 = 16000;
pub const MEL_BANDS: usize = 80;

/// 80-channel Mel spectrogram slice corresponding to a 5-frame receptive field window (200ms)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioLipSyncWindow {
    pub center_frame: usize,
    pub mel_data: Vec<f32>, // 80 bands x 16 time steps
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioTrackInfo {
    pub sample_rate: u32,
    pub duration_seconds: f32,
    pub total_samples: usize,
    pub total_frames: usize,
    pub sync_windows: Vec<AudioLipSyncWindow>,
}

pub struct MelProcessor;

impl MelProcessor {
    /// Load WAV audio and ensure 16kHz mono float format
    pub fn load_wav<P: AsRef<Path>>(path: P) -> Result<Vec<f32>> {
        let mut reader = WavReader::open(path.as_ref())
            .with_context(|| format!("Failed to read WAV file {:?}", path.as_ref()))?;
        
        let spec = reader.spec();
        let channels = spec.channels as usize;
        let source_rate = spec.sample_rate;

        let raw: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => {
                let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
                reader.samples::<i32>().map(|s| s.unwrap_or(0) as f32 / max).collect()
            }
            hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect(),
        };

        let mono = if channels > 1 {
            raw.chunks(channels).map(|c| c.iter().sum::<f32>() / channels as f32).collect()
        } else {
            raw
        };

        if source_rate != STANDARD_SAMPLE_RATE {
            Ok(Self::resample(&mono, source_rate, STANDARD_SAMPLE_RATE))
        } else {
            Ok(mono)
        }
    }

    pub fn resample(input: &[f32], src_rate: u32, target_rate: u32) -> Vec<f32> {
        if src_rate == target_rate || input.is_empty() {
            return input.to_vec();
        }
        let ratio = target_rate as f64 / src_rate as f64;
        let out_len = (input.len() as f64 * ratio).round() as usize;
        let mut out = Vec::with_capacity(out_len);
        for i in 0..out_len {
            let src_idx = i as f64 / ratio;
            let low = src_idx.floor() as usize;
            let high = (low + 1).min(input.len().saturating_sub(1));
            let frac = (src_idx - low as f64) as f32;
            out.push(input[low] * (1.0 - frac) + input[high] * frac);
        }
        out
    }

    /// Generate 80-channel Mel windows for lip-sync generation at specified video FPS
    pub fn process_lip_sync_windows(samples: &[f32], fps: f32) -> AudioTrackInfo {
        let samples_per_frame = (STANDARD_SAMPLE_RATE as f32 / fps).round() as usize;
        let total_samples = samples.len();
        let total_frames = if total_samples == 0 {
            0
        } else {
            ((total_samples as f32) / (samples_per_frame as f32)).ceil() as usize
        };
        let duration = total_samples as f32 / STANDARD_SAMPLE_RATE as f32;

        let mut windows = Vec::with_capacity(total_frames);
        // Window context size: 5 video frames centered on current frame
        let window_samples = samples_per_frame * 5;

        for f in 0..total_frames {
            let center_sample = f * samples_per_frame + (samples_per_frame / 2);
            let start = center_sample.saturating_sub(window_samples / 2);
            let end = (start + window_samples).min(total_samples);

            let chunk = if start < total_samples {
                &samples[start..end]
            } else {
                &[]
            };

            let mel = Self::compute_mel_spectrogram(chunk, MEL_BANDS, 16);
            windows.push(AudioLipSyncWindow {
                center_frame: f,
                mel_data: mel,
            });
        }

        AudioTrackInfo {
            sample_rate: STANDARD_SAMPLE_RATE,
            duration_seconds: duration,
            total_samples,
            total_frames,
            sync_windows: windows,
        }
    }

    /// Approximate 80-channel log Mel spectrogram across T time steps
    fn compute_mel_spectrogram(chunk: &[f32], bands: usize, time_steps: usize) -> Vec<f32> {
        let mut mel = vec![0.0; bands * time_steps];
        if chunk.is_empty() {
            return mel;
        }

        let step_size = (chunk.len() / time_steps).max(1);
        for t in 0..time_steps {
            let start = (t * step_size).min(chunk.len());
            let end = ((t + 1) * step_size).min(chunk.len());
            let slice = &chunk[start..end];
            if !slice.is_empty() {
                let sub_size = (slice.len() / bands).max(1);
                for b in 0..bands {
                    let b_start = (b * sub_size).min(slice.len());
                    let b_end = ((b + 1) * sub_size).min(slice.len());
                    let b_slice = &slice[b_start..b_end];
                    if !b_slice.is_empty() {
                        let mut energy = 0.0;
                        for (i, &s) in b_slice.iter().enumerate() {
                            let w = 0.54 - 0.46 * (2.0 * PI * i as f32 / b_slice.len() as f32).cos();
                            energy += (s * w).powi(2);
                        }
                        let log_energy = (1.0 + (energy / b_slice.len() as f32)).ln();
                        mel[t * bands + b] = log_energy;
                    }
                }
            }
        }

        mel
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_processing() {
        let samples = vec![0.0; 16000]; // 1 second of silence
        let track = MelProcessor::process_lip_sync_windows(&samples, 25.0);
        assert_eq!(track.total_frames, 25);
        assert_eq!(track.sync_windows.len(), 25);
        assert_eq!(track.sync_windows[0].mel_data.len(), 80 * 16);
    }
}
