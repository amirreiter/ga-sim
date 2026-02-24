use std::{cell::RefCell, mem::ManuallyDrop, ops::Mul};

use glam::Vec3A;
use obvhs::{
    ray::{Ray, RayHit},
    rt_triangle::RtTriangle,
};
use rayon::iter::ParallelIterator;

use crate::{
    fibonacci::FibonacciSphere,
    frequency::SimulationFrequency,
    material::SurfaceMaterial,
    microphone::Microphone,
    scenes::Scene,
    simulation::{shared::EnergyHistogram, trace::trace},
};

const SPEED_OF_SOUND: f32 = 343.0;

/// A specular raytracing simulation that runs on the CPU, either single-threaded
/// or multi-threaded.
pub fn rt_specular_cpu<F: SimulationFrequency>(
    multithread: bool,
    scene: Scene,
    microphone: Microphone,
    sample_rate: f32,
    ray_count: u64,
    histogram_bin_count: usize,
) {
    let iter = FibonacciSphere::new(ray_count);

    if multithread {
        // Multi threaded implementation, when we want to run a singular specular
        // simulation in parallel.
        iter.into_par_iter()
            .fold(
                || EnergyHistogram::new(histogram_bin_count, sample_rate),
                |mut local_histogram, seed_direction: Vec3A| {
                    let seed_ray = Ray::new_inf(
                        unsafe { std::mem::transmute(microphone.position) },
                        unsafe { std::mem::transmute(seed_direction) },
                    );
                    trace::<F, _>(
                        &scene,
                        &microphone,
                        seed_ray,
                        &mut local_histogram,
                        1.0,
                        specular_procedure::<F>,
                    );
                    local_histogram
                },
            )
            .reduce(
                || EnergyHistogram::new(histogram_bin_count, sample_rate),
                |mut a, b| {
                    for (ai, bi) in a.inner.iter_mut().zip(b.inner.iter()) {
                        *ai += bi;
                    }
                    a
                },
            );
    } else {
        // Single threaded implementation, when we want to run multiple specular
        // simulations in parallel.
        let mut histogram = EnergyHistogram::new(histogram_bin_count, sample_rate);

        iter.into_iter().for_each(|seed_direction: Vec3A| {
            let seed_ray = Ray::new_inf(
                unsafe { std::mem::transmute(microphone.position) },
                unsafe { std::mem::transmute(seed_direction) },
            );

            trace::<F, _>(
                &scene,
                &microphone,
                seed_ray,
                &mut histogram,
                1.0,
                specular_procedure::<F>,
            );
        });
    }
}

/// The core procedure for a specular pass.
pub fn specular_procedure<F: SimulationFrequency>(
    scene: &Scene,
    microphone: &Microphone,
    histogram: &mut EnergyHistogram,
    ray: Ray,
    distance_travelled: f32,
    triangle: &RtTriangle,
    material: &SurfaceMaterial,
    in_energy: f32,
) -> (Ray, f32) {
    let current_pos = ray.origin + ray.direction.mul(distance_travelled);
    let energy_after_travel = F::air_alpha().powf(distance_travelled) * in_energy;

    microphone_traceback::<F>(
        scene,
        microphone,
        histogram,
        unsafe { std::mem::transmute(current_pos) },
        in_energy,
    );

    let out_dir = ray.direction.reflect(triangle.compute_normal());
    let out_ray = Ray::new_inf(current_pos, out_dir);
    // TODO: Make 1-ac const
    let out_energy = energy_after_travel * (1.0 - F::ac(material));

    (out_ray, out_energy)
}

/// A sub-procedure for microphone traceback from location `current_pos`.
pub fn microphone_traceback<F: SimulationFrequency>(
    scene: &Scene,
    microphone: &Microphone,
    histogram: &mut EnergyHistogram,
    current_pos: Vec3A,
    in_energy: f32,
) {
    let microphone_pos = microphone.position;

    let mic_ray = Ray::new_inf(unsafe { std::mem::transmute(current_pos) }, unsafe {
        std::mem::transmute((microphone_pos - current_pos).normalize())
    });

    let mut hit = RayHit::none();
    if !scene
        .accelerator
        .ray_traverse(mic_ray, &mut hit, |ray, id| {
            scene.triangles[scene.accelerator.primitive_indices[id] as usize].intersect(ray)
        })
    {
        return;
    }

    // We have line-of-sight to the microphone.
    if hit.t < current_pos.distance(microphone_pos) {
        let bucket = ((hit.t / SPEED_OF_SOUND) * histogram.sample_rate).round() as usize;
        if bucket < histogram.inner.len() {
            // TODO: Multiply by microphone bias.
            let contribution = in_energy * F::air_alpha().powf(hit.t);

            histogram.inner[bucket] += contribution;
        }
    }
}
