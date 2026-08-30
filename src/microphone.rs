use glam::Vec3A;

/// A microphone directivity pattern.
///
/// All returned values are linear amplitude multipliers:
/// - 1.0 =   0 dB
/// - 0.5 ≈  -6 dB
/// - 0.0 = -∞ dB
#[derive(Clone, Debug)]
pub enum DirectivityPattern {
    /// Equal sensitivity in every direction.
    Omni,

    /// Standard cardioid:
    ///
    ///     gain = 0.5 * (1 + cos(theta))
    ///
    /// 0°   -> 1.0
    /// 90°  -> 0.5
    /// 180° -> 0.0
    Cardioid,

    /// General first-order polar pattern:
    ///
    ///     gain = a + b * cos(theta)
    ///
    /// Examples:
    /// - omni:        a=1.0,   b=0.0
    /// - cardioid:    a=0.5,   b=0.5
    /// - subcardioid: a=0.75,  b=0.25
    /// - hypercardioid:
    ///                a=0.25,  b=0.75
    ///
    /// Negative pressure response is converted to magnitude because this
    /// API returns an amplitude multiplier.
    FirstOrder { a: f32, b: f32 },

    /// Arbitrary rotationally-symmetric polar response.
    ///
    /// Samples span 0..=180 degrees:
    /// - samples[0]   = directly in front
    /// - samples[last] = directly behind
    ///
    /// Linear interpolation is used between samples.
    PolarLut(Box<[f32]>),
}

impl DirectivityPattern {
    #[inline]
    pub fn gain_from_cos(&self, cos_theta: f32) -> f32 {
        let cos_theta = cos_theta.clamp(-1.0, 1.0);

        match self {
            Self::Omni => 1.0,

            Self::Cardioid => 0.5 * (1.0 + cos_theta),

            Self::FirstOrder { a, b } => (a + b * cos_theta).abs(),

            Self::PolarLut(samples) => {
                if samples.is_empty() {
                    return 1.0;
                }

                if samples.len() == 1 {
                    return samples[0];
                }

                // Convert cos(theta) to theta in [0, pi].
                //
                // LUTs are intended for arbitrary patterns, so the acos here
                // is unavoidable unless you store the LUT in cosine-space.
                let theta = cos_theta.acos();
                let t = theta * ((samples.len() - 1) as f32 / std::f32::consts::PI);

                let i0 = t as usize;
                let i1 = (i0 + 1).min(samples.len() - 1);
                let frac = t - i0 as f32;

                samples[i0] + (samples[i1] - samples[i0]) * frac
            }
        }
    }

    /// Creates an arbitrary pattern sampled uniformly in cosine-space.
    ///
    /// This is faster than `PolarLut` because evaluation requires no `acos`.
    pub fn first_order(a: f32, b: f32) -> Self {
        Self::FirstOrder { a, b }
    }

    pub const fn cardioid() -> Self {
        Self::Cardioid
    }

    pub const fn omni() -> Self {
        Self::Omni
    }
}

pub struct Microphone {
    pub position: Vec3A,

    /// Unit vector pointing out of the front of the microphone.
    ///
    /// For example:
    ///
    ///     Vec3A::X
    ///
    /// means the microphone points toward +X.
    pub forward: Vec3A,

    pub pattern: DirectivityPattern,
}

impl Microphone {
    pub fn new(position: Vec3A, forward: Vec3A, pattern: DirectivityPattern) -> Self {
        Self {
            position,
            forward: forward.normalize_or_zero(),
            pattern,
        }
    }

    /// Returns the microphone's linear amplitude response to an incoming ray.
    ///
    /// `ray_direction` is the direction the sound ray is travelling.
    ///
    /// Therefore, if a microphone points toward +X, a sound arriving directly
    /// from the front travels toward the microphone along -X:
    ///
    /// ```text
    /// sound -----> microphone
    ///
    /// ray_direction = +X
    /// microphone.forward = -X
    /// ```
    ///
    /// In other words, a ray coming directly from the microphone's front has:
    ///
    ///     ray_direction == -self.forward
    ///
    /// This function accepts a non-normalized ray direction.
    #[inline]
    pub fn amplitude_multiplier(&self, ray_direction: Vec3A) -> f32 {
        let ray_direction = ray_direction.normalize_or_zero();

        if ray_direction == Vec3A::ZERO {
            return 0.0;
        }

        self.amplitude_multiplier_normalized(ray_direction)
    }

    /// Faster version for when `ray_direction` is already normalized.
    #[inline]
    pub fn amplitude_multiplier_normalized(&self, ray_direction: Vec3A) -> f32 {
        // Sound arriving from the front propagates opposite to the
        // microphone's forward vector.
        //
        // Front:
        //   ray = -forward
        //   (-ray) dot forward = 1
        //
        // Rear:
        //   ray = forward
        //   (-ray) dot forward = -1
        let cos_theta = (-ray_direction).dot(self.forward);

        let g = self.pattern.gain_from_cos(cos_theta);

        g * g
    }
}
