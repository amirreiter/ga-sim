use std::{time::Instant};

use glam::Vec3A;
use hound::{self, WavReader};

use crate::{
    analysis::{
        analyze_and_plot_energy_deviation,
    },
    microphone::Microphone,
    render::render,
    scenes::Scene,
    simulation::{
        cpu_rt_stochastic_diffuse,
    },
};

mod analysis;
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

    let mic = Microphone {
        position: Vec3A::new(4.52566, -2.92411, 0.333065),
    };

    let sample_rate = 48_000.0 / 48.0;
    let rays = 1_000_000u64;
    let bins = (sample_rate * 3.5) as usize;

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    // let start = Instant::now();
    // let mut specular = cpu_rt_stochastic_specular::<true>(
    //     true,
    //     &scene,
    //     &mic,
    //     Vec3A::new(4.52566, -2.92411, 0.333065),
    //     sample_rate,
    //     rays,
    //     bins,
    // )
    // .to_vec();
    // println!("Specular: {:?}", start.elapsed());

    let start = Instant::now();
    let diffuse = cpu_rt_stochastic_diffuse::<2>(
        true,
        &scene,
        &mic,
        Vec3A::new(4.52566, -2.92411, 0.333065),
        sample_rate,
        rays,
        bins,
    )
    .to_vec();
    println!("Diffuse: {:?}", start.elapsed());

    // specular.iter_mut().zip(diffuse.iter()).for_each(|(s, d)| {
    //     s.add(d);
    // });

    // let mut energy_histograms = specular;
    let mut energy_histograms = diffuse;

    // println!(
    //     "leaks        : {}    /    {}",
    //     DEBUG_LEAK_COUNTER.load(Ordering::SeqCst),
    //     rays
    // );
    // println!(
    //     "    {:.2}% success rate",
    //     100.0 - (DEBUG_LEAK_COUNTER.load(Ordering::SeqCst) as f32 / rays as f32) * 100.0
    // );
    // println!(
    //     "histogram oob: {}    /    {}",
    //     DEBUG_MIC_HITS_OUT_OF_BOUNDS.load(Ordering::SeqCst),
    //     DEBUG_MIC_HITS.load(Ordering::SeqCst)
    // );
    // println!(
    //     "    {:.2}%",
    //     (DEBUG_MIC_HITS_OUT_OF_BOUNDS.load(Ordering::SeqCst) as f32
    //         / DEBUG_MIC_HITS.load(Ordering::SeqCst) as f32)
    //         * 100.0
    // );

    let ehc = energy_histograms.clone();

    let samples = render(
        48_000.0,
        vec![
            (62.5, energy_histograms.remove(0)),
            (125.0, energy_histograms.remove(0)),
            (250.0, energy_histograms.remove(0)),
            (500.0, energy_histograms.remove(0)),
            (1000.0, energy_histograms.remove(0)),
            (2000.0, energy_histograms.remove(0)),
            (4000.0, energy_histograms.remove(0)),
            (8000.0, energy_histograms.remove(0)),
            (16000.0, energy_histograms.remove(0)),
        ],
        estimate_scene_aabb_volume(&scene),
        343.0,
    );

    let mut energy_histograms = ehc;

    let mut writer = hound::WavWriter::create("simulated_ir.wav", spec).unwrap();
    for s in samples.iter() {
        writer.write_sample(*s).unwrap();
    }
    writer.finalize().unwrap();

    let benchmark: Vec<f32> = WavReader::open("/../../_TU_BERLIN_ACOUSTIC_BENCHES/1_scene_descriptions-CR3/1 Scene descriptions/CR3 medium room (chamber music hall)/RIRs/wav/CR3_RIR_LS1_MP1_Dodecahedron.wav")
        .unwrap()
        .samples()
        .map(|s| s.unwrap())
        .collect();

    analyze_and_plot_energy_deviation(&samples, &benchmark, 48_000.0, 8192, 2048).unwrap();

    // analyze_and_plot_energy_deviation_from_histograms(
    //     &vec![
    //         (62.5, energy_histograms.remove(0)),
    //         (125.0, energy_histograms.remove(0)),
    //         (250.0, energy_histograms.remove(0)),
    //         (500.0, energy_histograms.remove(0)),
    //         (1000.0, energy_histograms.remove(0)),
    //         (2000.0, energy_histograms.remove(0)),
    //         (4000.0, energy_histograms.remove(0)),
    //         (8000.0, energy_histograms.remove(0)),
    //         (16000.0, energy_histograms.remove(0)),
    //     ],
    //     &benchmark,
    //     sample_rate,
    //     8192,
    //     2048,
    // )
    // .unwrap();

    // let mut writer = hound::WavWriter::create("diffuse.wav", spec).unwrap();
    // for s in diffuse.inner.iter() {
    //     writer.write_sample(*s).unwrap();
    // }
    // writer.finalize().unwrap();
}
