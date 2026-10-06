use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EnhancerModel {
    None,
    Gfpgan,
    RestoreFormer,
    CodeFormer,
}

impl Default for EnhancerModel {
    fn default() -> Self {
        Self::None
    }
}

/// Execution parameters for VideoReTalking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoReTalkingConfig {
    /// Canonical face crop resolution (standard: 256)
    pub face_size: u32,
    /// Video processing frame rate (default 25.0)
    pub fps: f32,
    /// Up-half face preservation (keeps natural eyes and upper expression)
    pub up_half: bool,
    /// Temporal smoothing window for face bounding box (in frames)
    pub smooth_window: usize,
    /// Face enhancement post-processor
    pub enhancer: EnhancerModel,
    /// Batch size for neural inference
    pub batch_size: usize,
    /// Checkpoint directory override
    pub checkpoint_dir: Option<PathBuf>,
}

impl Default for VideoReTalkingConfig {
    fn default() -> Self {
        Self {
            face_size: 256,
            fps: 25.0,
            up_half: true,
            smooth_window: 5,
            enhancer: EnhancerModel::None,
            batch_size: 4,
            checkpoint_dir: None,
        }
    }
}
