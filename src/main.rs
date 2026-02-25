use std::time::Instant;

use glam::Vec3A;
use hound;

use crate::{frequency::F_63_HZ, microphone::Microphone, simulation::rt_specular_cpu};

mod fibonacci;
mod frequency;
mod material;
mod microphone;
mod scenes;
mod simulation;

fn main() {
    let scene = scenes::cr3::load_bras_cr3();

    let microphone = Microphone {
        position: Vec3A::new(3.62673, 8.95161, 0.0),
    };
    let sample_rate = 48_000.0;
    let ray_count = 1_000_000;
    let histogram_bin_count = sample_rate as usize * 3;

    let start = Instant::now();

    let energy = rt_specular_cpu::<F_63_HZ>(
        true,
        scene,
        microphone,
        Vec3A::new(-1.75653, 5.00912, 0.0),
        sample_rate,
        ray_count,
        histogram_bin_count,
    );

    println!("{:?}", start.elapsed());

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create("data.wav", spec).unwrap();
    for (i, s) in energy.inner.iter().enumerate() {
        writer.write_sample(*s).unwrap();
    }
    writer.finalize().unwrap();
}
