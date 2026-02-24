use std::time::Instant;

use glam::Vec3A;

use crate::{frequency::{F_63_HZ}, microphone::Microphone, simulation::rt_specular_cpu};

mod frequency;
mod material;
mod scenes;
mod simulation;
mod fibonacci;
mod microphone;

fn main() {
    let scene = scenes::cr3::load_bras_cr3();
    let microphone = Microphone {
        position: Vec3A::new(0.0, 5.0, 0.0),
    };
    let sample_rate = 48_000.0;
    let ray_count = 10_000;
    let histogram_bin_count = sample_rate as usize * 3;

    let start = Instant::now();

    rt_specular_cpu::<F_63_HZ>(true, scene, microphone, sample_rate, ray_count, histogram_bin_count);

    println!("{:?}", start.elapsed())
}
