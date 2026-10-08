use videoretalking_core::{
    AffineTransform, AlignedFaceBox, AudioLipSyncWindow, EnhancerModel, FaceAlignment,
    LipSyncEngine, MelProcessor, Point2D, VideoReTalkingConfig, WeightsManager,
};

#[test]
fn test_affine_transform() {
    let t = AffineTransform {
        a: 2.0,
        b: 0.0,
        tx: 10.0,
        c: 0.0,
        d: 2.0,
        ty: 20.0,
    };
    let pt = Point2D { x: 5.0, y: 5.0 };
    let transformed = t.apply(pt);
    assert_eq!(transformed.x, 20.0);
    assert_eq!(transformed.y, 30.0);

    let inv = t.inverse().unwrap();
    let back = inv.apply(transformed);
    assert!((back.x - pt.x).abs() < 1e-4);
    assert!((back.y - pt.y).abs() < 1e-4);
}

#[test]
fn test_face_alignment_and_blend_mask() {
    let mask = FaceAlignment::generate_blend_mask(96, 48);
    assert_eq!(mask.len(), 96 * 96);
    // Upper half should have 0 alpha
    assert_eq!(mask[10 * 96 + 48], 0.0);
    // Lower half should have 1 alpha
    assert_eq!(mask[70 * 96 + 48], 1.0);

    let aligned_box = AlignedFaceBox {
        frame_idx: 0,
        bbox: [10.0, 20.0, 110.0, 120.0],
        transform: AffineTransform::default(),
        inverse_transform: AffineTransform::default(),
    };
    assert_eq!(aligned_box.bbox[0], 10.0);
}

#[test]
fn test_audio_mel_windows() {
    let dummy_samples = vec![0.05f32; 16000 * 2]; // 2 seconds
    let track_info = MelProcessor::process_lip_sync_windows(&dummy_samples, 25.0);
    assert_eq!(track_info.sample_rate, 16000);
    assert_eq!(track_info.sync_windows.len(), 50);
    assert_eq!(track_info.sync_windows[0].mel_data.len(), 80 * 16);
}

#[test]
fn test_lipsync_engine_planning() {
    let engine = LipSyncEngine::new(true);
    assert!(engine.up_half);

    let windows = (0..10)
        .map(|i| AudioLipSyncWindow {
            center_frame: i,
            mel_data: vec![0.1f32; 80 * 16],
        })
        .collect::<Vec<_>>();

    let plan = engine.plan_sequence(&windows);
    assert_eq!(plan.len(), 10);
    assert!(plan[0].has_audio_context);
}

#[test]
fn test_config_enhancers() {
    let mut config = VideoReTalkingConfig::default();
    assert_eq!(config.fps, 25.0);
    assert_eq!(config.face_size, 256);
    assert_eq!(config.enhancer, EnhancerModel::None);

    config.enhancer = EnhancerModel::RestoreFormer;
    assert_eq!(config.enhancer, EnhancerModel::RestoreFormer);
}

#[test]
fn test_weights_manager_invalid_path() {
    let res = WeightsManager::inspect_checkpoint("/invalid/path/videoretalking.safetensors");
    assert!(res.is_err());
}
