# `video-retalking-rs` — 3-Stage Semantic Expression Neutralization & Audio Lip-Sync Engine in Pure Rust

[![CI](https://github.com/bhubbard/video-retalking-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/bhubbard/video-retalking-rs/actions/workflows/ci.yml)
[![Interactive Studio](https://img.shields.io/badge/Interactive_Studio-code.brandonhubbard.com%2Fvideo--retalking--rs-2b5cff?style=flat-square)](https://code.brandonhubbard.com/video-retalking-rs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

**`video-retalking-rs`** is a native, memory-safe Rust port of [**OpenTalker/video-retalking**](https://github.com/OpenTalker/video-retalking) (*SIGGRAPH Asia 2022: Audio-based Lip Synchronization for Talking Head Video Editing in the Wild*). It implements the three sequential VideoReTalking stages:
1. **Stage 1 (`ENet` Semantic Expression Neutralization)**: Neutralizes source-video lower-face mouth landmarks (`points[48..68]`) into a canonical template while preserving upper-face gaze and eyebrow motion (`--up-half`).
2. **Stage 2 (`LNet` Audio-Driven Lip Synthesis)**: Extracts $80 \times 16$ Mel-spectrogram windows across a 5-frame (200 ms) receptive field to synthesize F1/F2 formant-synchronized mouth aperture and spread ratios.
3. **Stage 3 (`DINet` / `GFPGAN` / `RestoreFormer` Identity-Aware Paste-Back)**: Computes invertible similarity affine transforms (`AffineTransform::inverse`), temporal 68-landmark stabilization (`smooth_landmarks_temporal`), and circular-feathered jawline paste-back masks (`generate_blend_mask`).

---

## Core Architecture (`videoretalking-core` + `videoretalking-cli`)

- **`crates/videoretalking-core/src/face.rs`**:
  - `AffineTransform`: Similarity 2D affine matrix (`[[a, b, tx], [c, d, ty]]`) with exact analytical inverse (`inverse`) and canonical $256 \times 256$ inter-ocular alignment (`compute_canonical_transform`).
  - `FaceLandmarks68` & `FaceAlignment::smooth_landmarks_temporal`: 68-point facial landmark representation with sliding-window temporal stabilization.
  - `FaceAlignment::generate_blend_mask`: Lower-face feathered blending mask with circular jaw boundary falloff.
- **`crates/videoretalking-core/src/lipsync.rs`**:
  - `LipSyncEngine::neutralize_expression_delta`: Collapses source-video mouth landmarks (`48..68`) toward a canonical neutral lip line so original video speech does not leak into the retalked output.
  - `LipSyncEngine::plan_sequence`: Maps 80-band Mel formant energies into per-frame mouth openness and horizontal spread targets.
- **`crates/videoretalking-core/src/audio.rs`**:
  - `MelProcessor`: 16 kHz mono WAV loading, linear resampling, and $80 \times 16$ Hamming-windowed log-Mel spectrogram extraction (`process_lip_sync_windows`).

---

## CLI Usage

```bash
# Run 3-stage VideoReTalking pipeline with upper-half preservation
cargo run --release -p videoretalking-cli -- retalk \
  --face-video interview.mp4 \
  --audio dubbed_speech_16k.wav \
  --output output_retalked.mp4 \
  --face-size 256 \
  --up-half true

# Inspect WAV audio Mel windows or SafeTensors checkpoints
cargo run --release -p videoretalking-cli -- inspect checkpoints/LNet.safetensors

# Run 80x16 Mel spectrogram & LipSync planning benchmark
cargo run --release -p videoretalking-cli -- bench
```

---

## Performance Benchmarks vs. Upstream Python (`OpenTalker/video-retalking`)

See [`BENCHMARKS.md`](BENCHMARKS.md) for full methodology and hardware telemetry (**48.7x aggregate speedup**).
