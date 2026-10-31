use crate::audio::AudioLipSyncWindow;
use crate::face::{FaceLandmarks68, Point2D};
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

    /// Stage 1 (ENet): Neutralize source expression mouth landmarks (indices 48..68) to prevent source lip motion leakage
    pub fn neutralize_expression_delta(&self, landmarks: &FaceLandmarks68) -> FaceLandmarks68 {
        if landmarks.points.len() < 68 {
            return landmarks.clone();
        }
        let mut neutralized = landmarks.clone();
        // Compute mouth centroid across landmarks 48..68
        let mouth_slice = &landmarks.points[48..68];
        let mean_x = mouth_slice.iter().map(|p| p.x).sum::<f32>() / 20.0;
        let mean_y = mouth_slice.iter().map(|p| p.y).sum::<f32>() / 20.0;

        // Collapse vertical mouth opening toward canonical neutral lip line while preserving horizontal width
        for pt in &mut neutralized.points[48..68] {
            pt.x = mean_x + (pt.x - mean_x) * 0.92;
            pt.y = mean_y + (pt.y - mean_y) * 0.15;
        }
        if !self.up_half {
            for pt in &mut neutralized.points[17..48] {
                *pt = Point2D {
                    x: pt.x,
                    y: pt.y,
                };
            }
        }
        neutralized
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
