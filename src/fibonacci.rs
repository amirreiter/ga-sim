use glam::Vec3A;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::f32::consts::{GOLDEN_RATIO, PI};

const GOLDEN_ANGLE: f32 = 2.39996322973;

pub struct FibonacciSphere {
    index: u64,
    samples: u64,
    inv_delta: f32,
}

impl FibonacciSphere {
    #[must_use]
    #[inline]
    pub fn new(samples: u64) -> Self {
        debug_assert!(samples > 1, "samples must be > 1");
        Self {
            index: 0,
            samples,
            inv_delta: 2.0 / (samples - 1) as f32,
        }
    }

    #[inline]
    fn compute(index: u64, samples: u64, inv_delta: f32) -> Vec3A {
        let i = index as f32;
        let y = 1.0 - i * inv_delta;
        let r = (1.0 - y * y).sqrt();
        let theta = GOLDEN_RATIO * i;

        Vec3A::new(theta.cos() * r, y, theta.sin() * r).normalize()
    }

    pub fn into_par_iter(self) -> impl ParallelIterator<Item = Vec3A> {
        let (samples, inv_delta) = (self.samples, self.inv_delta);
        (0..samples)
            .into_par_iter()
            .map(move |i| Self::compute(i, samples, inv_delta))
    }
}

impl Iterator for FibonacciSphere {
    type Item = Vec3A;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        (self.index < self.samples).then(|| {
            let pt = Self::compute(self.index, self.samples, self.inv_delta);
            self.index += 1;
            pt
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = (self.samples - self.index) as usize;
        (rem, Some(rem))
    }
}

pub struct FibonacciHemisphere {
    index: u64,
    samples: u64,
    basis: [Vec3A; 3],
    inv_samples: f32,
}

impl FibonacciHemisphere {
    #[must_use]
    pub fn new(samples: u64, normal: Vec3A) -> Self {
        debug_assert!(samples > 0, "samples must be > 0");
        let n = normal.normalize();

        let t = if n.x.abs() > 0.9 {
            n.cross(Vec3A::Z).normalize()
        } else {
            n.cross(Vec3A::X).normalize()
        };
        let b = n.cross(t);

        Self {
            index: 0,
            samples,
            basis: [t, b, n],
            inv_samples: 1.0 / samples as f32,
        }
    }

    #[inline]
    fn compute(i: u64, basis: &[Vec3A; 3], inv_n: f32) -> Vec3A {
        let idx = i as f32;
        let θ = idx * GOLDEN_ANGLE;
        let y = 1.0 - (idx + 0.5) * inv_n;
        let r = (1.0 - y * y).max(0.0).sqrt();

        let local = Vec3A::new(θ.cos() * r, y, θ.sin() * r);

        basis[0] * local.x + basis[1] * local.z + basis[2] * local.y
    }

    pub fn into_par_iter(self) -> impl ParallelIterator<Item = Vec3A> {
        let (samples, basis, inv_n) = (self.samples, self.basis, self.inv_samples);
        (0..samples)
            .into_par_iter()
            .map(move |i| Self::compute(i, &basis, inv_n))
    }
}

impl Iterator for FibonacciHemisphere {
    type Item = Vec3A;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        (self.index < self.samples).then(|| {
            let pt = Self::compute(self.index, &self.basis, self.inv_samples);
            self.index += 1;
            pt
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = (self.samples - self.index) as usize;
        (rem, Some(rem))
    }
}
