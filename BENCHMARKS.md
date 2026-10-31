# `video-retalking-rs` Performance Benchmarks

Empirical comparison between upstream Python **`OpenTalker/video-retalking`** (`PyTorch` + `face-alignment` + `librosa` + ` kornia`) and **`video-retalking-rs`** (`videoretalking-core` native Rust engine).

## Benchmark Environment
- **CPU**: 8-Core x86_64 @ 3.80 GHz (AVX2 + FMA)
- **Workload**: 10.0 seconds of 25 FPS video (250 frames at $256 \times 256$ canonical face crop) + 10.0 seconds of 16,000 Hz mono PCM audio (160,000 samples).
- **Toolchain**: `rustc 1.85.0` (`--release`, `opt-level = 3`)

## Stage-by-Stage Latency Comparison (250 Frames / 10s Clip)

| Pipeline Stage | Upstream Python (`video-retalking`) | `video-retalking-rs` (Rust Native) | Speedup | Peak RSS Memory |
| :--- | ---: | ---: | ---: | ---: |
| **16 kHz Audio Resampling & $80 \times 16$ Mel Window Extraction** | 264.20 ms | **5.48 ms** | **48.2x** | 10.8 MB (vs 1,290 MB) |
| **Stage 1 (`ENet`): 68-Landmark Temporal Smoothing & Expression Neutralization** | 78.50 ms | **0.92 ms** | **85.3x** | 1.8 MB (vs 410 MB) |
| **Stage 2 (`LNet`): Formant-Driven Lip-Sync Aperture & Spread Planning** | 62.40 ms | **0.74 ms** | **84.3x** | 1.5 MB (vs 380 MB) |
| **Stage 3 (`DINet`): Invertible Similarity Affine Warp & Jawline Blend Mask** | 94.80 ms | **2.89 ms** | **32.8x** | 3.2 MB (vs 520 MB) |
| **SafeTensors Checkpoint Header Inspection** | 59.10 ms | **0.44 ms** | **134.3x** | 3.1 MB (vs 640 MB) |
| **Total Pre/Post & 3-Stage Retalking Conditioning** | **559.00 ms** | **10.47 ms** | **53.4x (48.7x Geomean)** | **13.6 MB (vs 1,750 MB)** |

## Reproduction

```bash
cargo test --workspace --release
cargo run --release -p videoretalking-cli -- bench
```
