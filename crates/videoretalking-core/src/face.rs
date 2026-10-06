use serde::{Deserialize, Serialize};

/// 2D landmark point (x, y)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Point2D {
    pub x: f32,
    pub y: f32,
}

/// 68-point facial landmark layout
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceLandmarks68 {
    pub points: Vec<Point2D>, // 68 points
}

/// 2D Affine Transformation Matrix [[a, b, tx], [c, d, ty]]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AffineTransform {
    pub a: f32,
    pub b: f32,
    pub tx: f32,
    pub c: f32,
    pub d: f32,
    pub ty: f32,
}

impl Default for AffineTransform {
    fn default() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            tx: 0.0,
            c: 0.0,
            d: 1.0,
            ty: 0.0,
        }
    }
}

impl AffineTransform {
    pub fn inverse(&self) -> Option<Self> {
        let det = self.a * self.d - self.b * self.c;
        if det.abs() < 1e-6 {
            return None;
        }
        let inv_det = 1.0 / det;
        let a = self.d * inv_det;
        let b = -self.b * inv_det;
        let c = -self.c * inv_det;
        let d = self.a * inv_det;
        let tx = -(a * self.tx + b * self.ty);
        let ty = -(c * self.tx + d * self.ty);

        Some(Self { a, b, tx, c, d, ty })
    }

    pub fn apply(&self, pt: Point2D) -> Point2D {
        Point2D {
            x: self.a * pt.x + self.b * pt.y + self.tx,
            y: self.c * pt.x + self.d * pt.y + self.ty,
        }
    }
}

/// Face bounding box and canonical alignment metadata for a single frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignedFaceBox {
    pub frame_idx: usize,
    pub bbox: [f32; 4], // [x1, y1, x2, y2]
    pub transform: AffineTransform,
    pub inverse_transform: AffineTransform,
}

pub struct FaceAlignment;

impl FaceAlignment {
    /// Compute similarity affine transform from 5 key face landmarks to canonical 256x256
    pub fn compute_canonical_transform(
        left_eye: Point2D,
        right_eye: Point2D,
        _nose: Point2D,
        target_size: f32,
    ) -> AffineTransform {
        let dx = right_eye.x - left_eye.x;
        let dy = right_eye.y - left_eye.y;
        let dist = (dx * dx + dy * dy).sqrt().max(1.0);

        // Desired inter-ocular distance in 256x256 canonical space: ~70px
        let target_dist = target_size * 0.28;
        let scale = target_dist / dist;
        let angle = dy.atan2(dx);

        let cos_a = angle.cos() * scale;
        let sin_a = angle.sin() * scale;

        // Center between eyes placed at (128, 85)
        let eye_center_x = (left_eye.x + right_eye.x) / 2.0;
        let eye_center_y = (left_eye.y + right_eye.y) / 2.0;

        let tx = (target_size * 0.5) - (cos_a * eye_center_x - sin_a * eye_center_y);
        let ty = (target_size * 0.35) - (sin_a * eye_center_x + cos_a * eye_center_y);

        AffineTransform {
            a: cos_a,
            b: -sin_a,
            tx,
            c: sin_a,
            d: cos_a,
            ty,
        }
    }

    /// Generate lower-face blending mask (feathered gradient from mouth region to jaw)
    pub fn generate_blend_mask(size: usize, up_half_cut: usize) -> Vec<f32> {
        let mut mask = vec![0.0; size * size];
        let feather = 16.0f32;

        for y in 0..size {
            let alpha = if y < up_half_cut {
                0.0
            } else if (y as f32) < (up_half_cut as f32 + feather) {
                (y - up_half_cut) as f32 / feather
            } else {
                1.0
            };

            for x in 0..size {
                // Circular mask cutoff around jaw
                let center_x = size as f32 / 2.0;
                let center_y = size as f32 * 0.7;
                let r = ((x as f32 - center_x).powi(2) + (y as f32 - center_y).powi(2)).sqrt();
                let edge_alpha = if r > (size as f32 * 0.45) {
                    0.0
                } else {
                    1.0
                };

                mask[y * size + x] = alpha * edge_alpha;
            }
        }

        mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_affine_inverse() {
        let t = AffineTransform {
            a: 2.0,
            b: 0.0,
            tx: 10.0,
            c: 0.0,
            d: 2.0,
            ty: 20.0,
        };
        let inv = t.inverse().unwrap();
        let pt = Point2D { x: 5.0, y: 5.0 };
        let mapped = t.apply(pt);
        let restored = inv.apply(mapped);
        assert!((restored.x - pt.x).abs() < 1e-4);
        assert!((restored.y - pt.y).abs() < 1e-4);
    }

    #[test]
    fn test_blend_mask() {
        let mask = FaceAlignment::generate_blend_mask(256, 128);
        assert_eq!(mask.len(), 256 * 256);
        // Top half should be 0.0 (unmodified upper face)
        assert_eq!(mask[50 * 256 + 128], 0.0);
        // Lower center should be blended
        assert!(mask[180 * 256 + 128] > 0.8);
    }
}
