use std::time::Instant;

use glam::Vec3A;

use crate::{
    estimate_scene_aabb_volume,
    frequency::*,
    microphone::{DirectivityPattern::Cardioid, Microphone},
    render::render,
    scenes::{self, Scene},
    simulation::cpu_rt_stochastic_singleband,
};

pub fn ambisonic_a_format(position: Vec3A) -> [Microphone; 4] {
    const S: f32 = 0.5773502691896258; // 1 / sqrt(3)

    [
        // Channel 1: LF — Left, Front, Up
        Microphone {
            position,
            forward: Vec3A::new(S, S, S),
            pattern: Cardioid,
        },
        // Channel 2: RF — Right, Front, Down
        Microphone {
            position,
            forward: Vec3A::new(S, -S, -S),
            pattern: Cardioid,
        },
        // Channel 3: LB — Left, Back, Down
        Microphone {
            position,
            forward: Vec3A::new(-S, S, -S),
            pattern: Cardioid,
        },
        // Channel 4: RB — Right, Back, Up
        Microphone {
            position,
            forward: Vec3A::new(-S, -S, S),
            pattern: Cardioid,
        },
    ]
}

pub fn generate_ir_for_microphone(
    scene: &Scene,
    mic: &Microphone,
    speaker: Vec3A,
    sample_rate: f32,
    rays: u64,
    bins: usize,
) -> Vec<f32> {
    macro_rules! cpu_rt_ssb {
        ($F:ty) => {
            cpu_rt_stochastic_singleband::<$F>(true, &scene, &mic, speaker, sample_rate, rays, bins)
        };
    }

    let start = Instant::now();
    let mut combined = Vec::new();
    println!("Starting RT...");
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
    println!("     DONE! --- {:?}", start.elapsed());

    let samples = render(
        44_100.0,
        vec![
            (62.5, combined.remove(0)),
            (125.0, combined.remove(0)),
            (250.0, combined.remove(0)),
            (500.0, combined.remove(0)),
            (1000.0, combined.remove(0)),
            (2000.0, combined.remove(0)),
            (4000.0, combined.remove(0)),
            (8000.0, combined.remove(0)),
            (16000.0, combined.remove(0)),
        ],
        estimate_scene_aabb_volume(&scene),
        343.0,
        false,
    );

    samples
}

pub fn render_ambisonics(mic: Vec3A, speaker: Vec3A) {
    let scene = scenes::cr3::load_bras_cr3();

    let sample_rate = 44_100.0 / (1.0); //(44.1 / 4.0);
    let rays = 1_000_000u64;
    let bins = (sample_rate * 3.5) as usize;

    let microphones = ambisonic_a_format(mic); // Vec3A::new(6.08988, 0.0, 1.43685)

    let mut ambi_a: Vec<Vec<f32>> = microphones
        .iter()
        .map(|mic| {
            generate_ir_for_microphone(
                &scene, mic, speaker, //Vec3A::new(-1.94418, 0.0, 2.22308),
                44_100.0, rays, bins,
            )
        })
        .collect();

    let peak = ambi_a[0]
        .iter()
        .chain(ambi_a[1].iter())
        .chain(ambi_a[2].iter())
        .chain(ambi_a[3].iter())
        .map(|s| s.abs())
        .fold(0.0f32, f32::max);

    if peak > 1.0 {
        ambi_a[0].iter_mut().for_each(|s| *s /= peak);
        ambi_a[1].iter_mut().for_each(|s| *s /= peak);
        ambi_a[2].iter_mut().for_each(|s| *s /= peak);
        ambi_a[3].iter_mut().for_each(|s| *s /= peak);
    }

    let ambi_b = a_format_to_ambix_b_format(&ambi_a[0], &ambi_a[1], &ambi_a[2], &ambi_a[3]);

    let spec = hound::WavSpec {
        channels: 4,
        sample_rate: 44100,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = hound::WavWriter::create("AmbiX_B_ACN_SN3D.wav", spec).unwrap();

    for i in 0..ambi_b[0].len() {
        // WAV samples are interleaved by frame
        writer.write_sample(ambi_b[0][i]).unwrap();
        writer.write_sample(ambi_b[1][i]).unwrap();
        writer.write_sample(ambi_b[2][i]).unwrap();
        writer.write_sample(ambi_b[3][i]).unwrap();
    }

    writer.finalize().unwrap();
}

/// Converts ideal coincident tetrahedral cardioid A-format
/// [LF, RF, LB, RB]
/// into AmbiX / ACN / SN3D:
///
/// [W, Y, Z, X]
///
/// Coordinate convention:
/// +X = Front
/// +Y = Left
/// +Z = Up
pub fn a_format_to_ambix_b_format(lf: &[f32], rf: &[f32], lb: &[f32], rb: &[f32]) -> [Vec<f32>; 4] {
    assert_eq!(lf.len(), rf.len());
    assert_eq!(lf.len(), lb.len());
    assert_eq!(lf.len(), rb.len());

    let sample_count = lf.len();

    let mut w = Vec::with_capacity(sample_count);
    let mut y = Vec::with_capacity(sample_count);
    let mut z = Vec::with_capacity(sample_count);
    let mut x = Vec::with_capacity(sample_count);

    const W_SCALE: f32 = 0.5;
    const XYZ_SCALE: f32 = 0.8660254037844386; // sqrt(3) / 2

    for i in 0..sample_count {
        let lf = lf[i];
        let rf = rf[i];
        let lb = lb[i];
        let rb = rb[i];

        // ACN 0: W
        w.push(W_SCALE * (lf + rf + lb + rb));

        // ACN 1: Y — Left / Right
        y.push(XYZ_SCALE * (lf - rf + lb - rb));

        // ACN 2: Z — Up / Down
        z.push(XYZ_SCALE * (lf - rf - lb + rb));

        // ACN 3: X — Front / Back
        x.push(XYZ_SCALE * (lf + rf - lb - rb));
    }

    [w, y, z, x]
}
