#![feature(iter_advance_by)]
use std::{sync::atomic::Ordering, time::Instant};

use glam::Vec3A;
use hound;

use crate::{
    frequency::*, microphone::Microphone, render::render, scenes::Scene, simulation::{
        DEBUG_LEAK_COUNTER, DEBUG_MIC_HITS, DEBUG_MIC_HITS_OUT_OF_BOUNDS, EnergyHistogram,
        cpu_stochastic_rt,
    }
};

mod fibonacci;
mod frequency;
mod material;
mod microphone;
mod random;
mod render;
mod scenes;
mod simulation;

fn estimate_scene_aabb_volume(scene: &Scene) -> f32 {
    let mut min = Vec3A::splat(f32::INFINITY);
    let mut max = Vec3A::splat(f32::NEG_INFINITY);

    for tri in &scene.triangles {
        let v0 = tri.v0;
        let v1 = tri.v0 - tri.e1;
        let v2 = tri.v0 + tri.e2;

        min = min.min(v0).min(v1).min(v2);
        max = max.max(v0).max(v1).max(v2);
    }

    let d = (max - min).max(Vec3A::ZERO);
    d.x * d.y * d.z
}

fn main() {
    let scene = scenes::cr3::load_bras_cr3();

    println!("{}", scene.gpu_triangles[0].unpack().0);

    let mic = Microphone {
        position: Vec3A::new(4.52566, -2.92411, 0.333065),
    };
    let sample_rate = 48_000.0 / 48.0;
    let rays = 100_000u64;
    let bins = sample_rate as usize;

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
    let mut energy_histograms = cpu_stochastic_rt(
        false,
        &scene,
        &mic,
        Vec3A::new(4.52566, -2.92411, 0.333065),
        sample_rate,
        rays,
        bins,
    ).to_vec();
    println!("{:?}", start.elapsed());

    println!(
        "leaks        : {}    /    {}",
        DEBUG_LEAK_COUNTER.load(Ordering::SeqCst),
        rays
    );
    println!(
        "    {:.2}% success rate",
        100.0 - (DEBUG_LEAK_COUNTER.load(Ordering::SeqCst) as f32 / rays as f32) * 100.0
    );
    println!(
        "histogram oob: {}    /    {}",
        DEBUG_MIC_HITS_OUT_OF_BOUNDS.load(Ordering::SeqCst),
        DEBUG_MIC_HITS.load(Ordering::SeqCst)
    );
    println!(
        "    {:.2}%",
        (DEBUG_MIC_HITS_OUT_OF_BOUNDS.load(Ordering::SeqCst) as f32
            / DEBUG_MIC_HITS.load(Ordering::SeqCst) as f32)
            * 100.0
    );

    let samples = render(48_000.0, vec![
        (125.0, energy_histograms.remove(0)),
        (250.0, energy_histograms.remove(0)),
        (500.0, energy_histograms.remove(0)),
        (1000.0, energy_histograms.remove(0)),
        (2000.0, energy_histograms.remove(0)),
        (4000.0, energy_histograms.remove(0)),
        (8000.0, energy_histograms.remove(0)),
        (16000.0, energy_histograms.remove(0)),
    ], estimate_scene_aabb_volume(&scene), 343.0, 400.0);

    let mut writer = hound::WavWriter::create("125hz.wav", spec).unwrap();
    for s in samples.iter() {
        writer.write_sample(*s).unwrap();
    }
    writer.finalize().unwrap();

    // let mut writer = hound::WavWriter::create("diffuse.wav", spec).unwrap();
    // for s in diffuse.inner.iter() {
    //     writer.write_sample(*s).unwrap();
    // }
    // writer.finalize().unwrap();
}

// fn get_energy<F>(
//     scene: &Scene,
//     microphone: &Microphone,
//     sample_rate: f32,
//     ray_count: u64,
//     bin_count: usize,
// ) -> (f32, EnergyHistogram)
// where
//     F: SimulationFrequency,
// {
//     let freq_hz = F::hz() as f32;

//     let start = Instant::now();

//     let mut specular = rt_specular_cpu::<F>(
//         true,
//         scene,
//         microphone,
//         Vec3A::new(-1.75653, 5.00912, 0.0),
//         sample_rate,
//         ray_count,
//         bin_count,
//     );

//     let diffuse = rt_diffuse_cpu::<F, 1>(
//         true,
//         scene,
//         microphone,
//         Vec3A::new(-1.75653, 5.00912, 0.0),
//         sample_rate,
//         ray_count,
//         bin_count,
//     );

//     println!("F {} - {:?}", freq_hz, Instant::now().duration_since(start));

//     specular.add(&diffuse);

//     (freq_hz, specular)
// }
