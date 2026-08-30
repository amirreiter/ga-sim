use glam::Vec3A;
use obvhs::ray::Ray;
use rand::{
    RngExt,
    rngs::{StdRng, ThreadRng},
};

pub mod singleband;

pub const SPEED_OF_SOUND: f32 = 343.0;
pub const ENERGY_CUTOFF: f32 = f32::EPSILON;

pub fn intersect_ray_sphere(r: &Ray, center: &Vec3A, radius_sq: f32) -> f32 {
    let p = r.origin;
    let d = r.direction;

    let m = p - *center;
    let b = m.dot(d);
    let c = m.dot(m) - radius_sq;

    if c > 0.0 && b > 0.0 {
        return f32::INFINITY;
    }

    let discr = b * b - c;
    if discr < 0.0 {
        return f32::INFINITY;
    }

    let sqrt_discr = discr.sqrt();
    let t0 = -b - sqrt_discr;
    let t1 = -b + sqrt_discr;

    if t0 > 0.001 {
        return t0;
    } else if t1 > 0.001 {
        return t1;
    }

    f32::INFINITY
}

pub fn sample_cosine_weighted_hemisphere(normal: Vec3A, rng: &mut StdRng) -> Vec3A {
    let theta = rng.random::<f32>() * std::f32::consts::TAU;
    let cos_phi = rng.random::<f32>().sqrt(); // cosine-weighted; remove sqrt for uniform
    let sin_phi = (1.0 - cos_phi * cos_phi).sqrt();
    let (sin_theta, cos_theta) = theta.sin_cos();

    // Revised Pixar/Frisvad — safe at both poles
    let sign = if normal.z >= 0.0 { 1.0_f32 } else { -1.0_f32 };
    let a = -1.0 / (sign + normal.z);
    let b = normal.x * normal.y * a;
    let tangent = Vec3A::new(
        1.0 + sign * normal.x * normal.x * a,
        sign * b,
        -sign * normal.x,
    );
    let bitangent = Vec3A::new(b, sign + normal.y * normal.y * a, -normal.y);

    tangent * (sin_phi * cos_theta) + bitangent * (sin_phi * sin_theta) + normal * cos_phi
}
