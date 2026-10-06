use crate::audio::AudioLipSyncWindow;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LipSyncFramePlan {
    pub frame_idx: usize,
    pub has_audio_context: bool,
    pub mouth_open_ratio: f32,
    pub mouth_spread_ratio: f32,
}

pub struct LipSyncEngine {
    pub up_half: bool,
}

impl LipSyncEngine {
    pub fn new(up_half: bool) -> Self {
        Self { up_half }
    }

    /// Plan frame-by-frame lip generation parameters based on Mel audio windows
    pub fn plan_sequence(&self, windows: &[AudioLipSyncWindow]) -> Vec<LipSyncFramePlan> {
        let mut plans = Vec::with_capacity(windows.len());

        for (i, win) in windows.iter().enumerate() {
            // Calculate phonemic energy from Mel bands 0..20 (formants F1, F2)
            let speech_energy = win.mel_data.iter().take(40).sum::<f32>() / 40.0;
            let high_energy = win.mel_data.iter().skip(40).take(40).sum::<f32>() / 40.0;

            let mouth_open = (speech_energy * 0.8).clamp(0.0, 1.2);
            let mouth_spread = (high_energy * 0.6).clamp(0.0, 1.0);

            plans.push(LipSyncFramePlan {
                frame_idx: i,
                has_audio_context: !win.mel_data.is_empty(),
                mouth_open_ratio: mouth_open,
                mouth_spread_ratio: mouth_spread,
            });
        }

        plans
    }
}
