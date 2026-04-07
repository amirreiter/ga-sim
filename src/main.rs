use std::time::Instant;

use glam::Vec3A;
use hound;

use crate::{frequency::*, microphone::Microphone, scenes::Scene, simulation::{EnergyHistogram, rt_diffuse_cpu, rt_specular_cpu}};

mod fibonacci;
mod frequency;
mod material;
mod microphone;
mod scenes;
mod simulation;
mod render;
mod random;

fn main() {
    let scene = scenes::cr3::load_bras_cr3();

    let mic = Microphone {
        position: Vec3A::new(3.62673, 8.95161, 0.0),
    };
    let sample_rate = 48_000.0 / 48.0;
    let rays = 1_000_000u64;
    let bins = sample_rate as usize;

    // let mut energy_histograms = Vec::new();

    // Manually call for each required frequency band
    // energy_histograms.push(get_energy::<F_63_HZ>(&scene, &mic, sample_rate, rays, bins));
    // energy_histograms.push(get_energy::<F_125_HZ>(&scene, &mic, sample_rate, rays, bins));
    // energy_histograms.push(get_energy::<F_250_HZ>(&scene, &mic, sample_rate, rays, bins));
    // energy_histograms.push(get_energy::<F_500_HZ>(&scene, &mic, sample_rate, rays, bins));
    // energy_histograms.push(get_energy::<F_1000_HZ>(&scene, &mic, sample_rate, rays, bins));
    // energy_histograms.push(get_energy::<F_2000_HZ>(&scene, &mic, sample_rate, rays, bins));
    // energy_histograms.push(get_energy::<F_4000_HZ>(&scene, &mic, sample_rate, rays, bins));
    // energy_histograms.push(get_energy::<F_8000_HZ>(&scene, &mic, sample_rate, rays, bins));
    // energy_histograms.push(get_energy::<F_16000_HZ>(&scene, &mic, sample_rate, rays, bins));
    // let rendered = render::render(48_000.0, energy_histograms);

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    // let mut writer = hound::WavWriter::create("ir.wav", spec).unwrap();
    // for s in rendered.iter() {
    //     writer.write_sample(*s).unwrap();
    // }
    // writer.finalize().unwrap();

    let start = Instant::now();
    let mut specular = rt_specular_cpu::<F_63_HZ>(
        true,
        &scene,
        &mic,
        Vec3A::new(-1.75653, 5.00912, 0.0),
        sample_rate,
        rays,
        bins,
    );
    println!("{:?}", start.elapsed());
    // let start = Instant::now();
    // let mut diffuse = rt_specular_cpu::<F_16000_HZ>(
    //     true,
    //     &scene,
    //     &mic,
    //     Vec3A::new(-1.75653, 5.00912, 0.0),
    //     sample_rate,
    //     rays,
    //     bins,
    // );
    // println!("{:?}", start.elapsed());

    specular.resample_linear(48_000.0);
    // diffuse.resample_linear(48_000.0);

    let mut writer = hound::WavWriter::create("specular.wav", spec).unwrap();
    for s in specular.inner.iter() {
        writer.write_sample(*s).unwrap();
    }
    writer.finalize().unwrap();

    // let mut writer = hound::WavWriter::create("diffuse.wav", spec).unwrap();
    // for s in diffuse.inner.iter() {
    //     writer.write_sample(*s).unwrap();
    // }
    // writer.finalize().unwrap();
}

fn get_energy<F>(
    scene: &Scene,
    microphone: &Microphone,
    sample_rate: f32,
    ray_count: u64,
    bin_count: usize
) -> (f32, EnergyHistogram)
where F: SimulationFrequency
{
    let freq_hz = F::hz() as f32;

    let start = Instant::now();

    let mut specular = rt_specular_cpu::<F>(
        true,
        scene,
        microphone,
        Vec3A::new(-1.75653, 5.00912, 0.0),
        sample_rate,
        ray_count,
        bin_count,
    );

    let diffuse = rt_diffuse_cpu::<F, 1>(
        true,
        scene,
        microphone,
        Vec3A::new(-1.75653, 5.00912, 0.0),
        sample_rate,
        ray_count,
        bin_count,
    );

    println!("F {} - {:?}", freq_hz, Instant::now().duration_since(start));

    specular.add(&diffuse);

    (freq_hz, specular)
}
