use glam::Vec3A;
use rand::random;

pub fn sample_hemisphere(normal: Vec3A) -> Vec3A {
    let u1: f32 = random();
    let u2: f32 = random();

    let z = u1 * 2.0 - 1.0;
    let phi = u2 * 2.0 * std::f32::consts::PI;
    let r = (1.0 - z * z).sqrt();
    let v = Vec3A::new(r * phi.cos(), r * phi.sin(), z);

    if v.dot(normal) < 0.0 { -v } else { v }
}
