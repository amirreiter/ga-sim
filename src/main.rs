use std::{sync::atomic::Ordering, time::Instant};

use glam::Vec3A;
use hound::{self, WavReader};

use crate::{
    ambisonics::render_ambisonics, analysis::analyze_and_plot_energy_deviation,
    caviar::ambisonic_b_to_caviar_14x2, frequency::*, microphone::Microphone, render::render,
    scenes::Scene, simulation::*,
};

mod ambisonics;
mod analysis;
mod caviar;
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
    ambisonic_b_to_caviar_14x2(
        "/Users/amirreiter/Github/amirreiter/ga-sim/AmbiX_B_ACN_SN3D.wav".into(),
        "/Users/amirreiter/Github/amirreiter/ga-sim/caviar".into(),
    )
    .unwrap();
    // render_ambisonics(Vec3A::new(-6.33102, 0.0, 2.83734), Vec3A::new(-6.33102, 0.0, 2.83734));

    println!(
        "leaks        : {}",
        DEBUG_LEAK_COUNTER.load(Ordering::SeqCst),
    );
    // println!(
    //     "    {:.2}% success rate",
    //     100.0 - (DEBUG_LEAK_COUNTER.load(Ordering::SeqCst) as f32 / (rays as f32 * 9.0)) * 100.0
    // );
    println!(
        "histogram oob: {}    /    {}",
        DEBUG_MIC_HITS_OUT_OF_BOUNDS.load(Ordering::SeqCst),
        DEBUG_MIC_HITS.load(Ordering::SeqCst)
    );
    // println!(
    //     "    {:.2}%",
    //     (DEBUG_MIC_HITS_OUT_OF_BOUNDS.load(Ordering::SeqCst) as f32
    //         / DEBUG_MIC_HITS.load(Ordering::SeqCst) as f32)
    //         * 100.0
    // );

    return;

    let scene = scenes::cr3::load_bras_cr3();

    let mic = Microphone::new(
        Vec3A::new(6.08988, 0.0, 1.43685),
        Vec3A::ZERO,
        microphone::DirectivityPattern::Omni,
    );

    let sample_rate = 44_100.0 / (1.0); //(44.1 / 4.0);
    let rays = 2_000_000u64;
    let bins = (sample_rate * 3.5) as usize;

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    macro_rules! cpu_rt_ssb {
        ($F:ty) => {
            cpu_rt_stochastic_singleband::<$F>(
                true,
                &scene,
                &mic,
                Vec3A::new(-1.94418, 0.0, 2.22308),
                sample_rate,
                rays,
                bins,
            )
        };
    }

    let start = Instant::now();
    let mut combined = Vec::new();
    combined.push(cpu_rt_ssb!(F_63_HZ));
    println!("     --- 63 --- {:?}", start.elapsed());
    combined.push(cpu_rt_ssb!(F_125_HZ));
    println!("     --- 125 --- {:?}", start.elapsed());
    combined.push(cpu_rt_ssb!(F_250_HZ));
    println!("     --- 250 --- {:?}", start.elapsed());
    combined.push(cpu_rt_ssb!(F_500_HZ));
    println!("     --- 500 --- {:?}", start.elapsed());
    combined.push(cpu_rt_ssb!(F_1000_HZ));
    println!("     --- 1000 --- {:?}", start.elapsed());
    combined.push(cpu_rt_ssb!(F_2000_HZ));
    println!("     --- 2000 --- {:?}", start.elapsed());
    combined.push(cpu_rt_ssb!(F_4000_HZ));
    println!("     --- 4000 --- {:?}", start.elapsed());
    combined.push(cpu_rt_ssb!(F_8000_HZ));
    println!("     --- 8000 --- {:?}", start.elapsed());
    combined.push(cpu_rt_ssb!(F_16000_HZ));
    println!("     --- 16000 --- {:?}", start.elapsed());
    println!("New Combined: {:?}", start.elapsed());

    // specular.iter_mut().zip(diffuse.iter()).for_each(|(s, d)| {
    //     s.add(d);
    // });

    let mut energy_histograms = combined;

    // let ehc = energy_histograms.clone();

    let samples = render(
        44_100.0,
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
        true,
    );

    // let mut energy_histograms = ehc;

    let mut writer = hound::WavWriter::create("simulated_ir.wav", spec).unwrap();
    for s in samples.iter() {
        writer.write_sample(*s).unwrap();
    }
    writer.finalize().unwrap();

    let benchmark: Vec<f32> = WavReader::open("/Users/amirreiter/Github/amirreiter/_TU_BERLIN_ACOUSTIC_BENCHES/1_scene_descriptions-CR3/1 Scene descriptions/CR3 medium room (chamber music hall)/RIRs/wav/CR3_RIR_LS1_MP1_Dodecahedron.wav")
        .unwrap()
        .samples()
        .map(|s| s.unwrap())
        .collect();

    analyze_and_plot_energy_deviation(&samples, &benchmark, 44_100.0, 8192 * 2, 2048 * 2).unwrap();

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
