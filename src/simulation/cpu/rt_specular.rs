use std::{cell::RefCell, f32, sync::Arc};

use glam::Vec3A;
use obvhs::ray::{Ray, RayHit};
use rayon::iter::{ParallelBridge, ParallelIterator};
use thread_local::ThreadLocal;

use crate::{
    fibonacci::FibonacciSphere,
    microphone::Microphone,
    scenes::Scene,
    simulation::{
        EnergyHistogram,
        cpu::{ENERGY_CUTOFF, SPEED_OF_SOUND, intersect_ray_sphere},
    },
};

/// A specular raytracing simulation that runs on the CPU, either single-threaded
/// or multi-threaded.
pub fn cpu_rt_stochastic_specular<const DIFFUSE_RETURN: bool>(
    multithread: bool,
    scene: &Scene,
    microphone: &Microphone,
    emitter: Vec3A,
    sample_rate: f32,
    ray_count: u64,
    histogram_bin_count: usize,
) -> [EnergyHistogram; 9] {
    let iter = FibonacciSphere::new(ray_count);

    if multithread {
        // Multi threaded implementation, when we want to run a singular specular
        // simulation in parallel.

        let tl_histogram: Arc<ThreadLocal<RefCell<[EnergyHistogram; 9]>>> =
            Arc::new(ThreadLocal::new());
        iter.into_par_iter().for_each_init(
            || tl_histogram.clone(),
            |histogram_handle, seed_direction: Vec3A| {
                let local_histogram = histogram_handle.get_or(|| {
                    RefCell::new(core::array::from_fn(|_| {
                        EnergyHistogram::new(histogram_bin_count, sample_rate)
                    }))
                });

                let seed_dir = seed_direction.normalize();
                if seed_dir.length_squared() == 0.0 {
                    return;
                }

                let seed_ray = Ray::new_inf(emitter, seed_dir);

                kernel_rt_cpu_stochastic_specular::<DIFFUSE_RETURN>(
                    &scene,
                    &microphone,
                    &mut *local_histogram.borrow_mut(),
                    seed_ray,
                    [1.0; 9],
                    0.0,
                );
            },
        );

        let histogram_inner = Arc::try_unwrap(tl_histogram).unwrap();

        let final_histograms = histogram_inner
            .into_iter()
            .map(|r| r.into_inner())
            .par_bridge()
            .fold(
                || core::array::from_fn(|_| EnergyHistogram::new(histogram_bin_count, sample_rate)),
                |mut acc_array, local_array| {
                    for (acc_h, local_h) in acc_array.iter_mut().zip(local_array.iter()) {
                        acc_h
                            .inner
                            .iter_mut()
                            .zip(local_h.inner.iter())
                            .for_each(|(a, b)| *a += b);
                    }
                    acc_array
                },
            )
            .reduce(
                || core::array::from_fn(|_| EnergyHistogram::new(histogram_bin_count, sample_rate)),
                |mut a_array, b_array| {
                    for (a_h, b_h) in a_array.iter_mut().zip(b_array.iter()) {
                        a_h.inner
                            .iter_mut()
                            .zip(b_h.inner.iter())
                            .for_each(|(a, b)| *a += b);
                    }
                    a_array
                },
            );

        final_histograms
    } else {
        // Single threaded implementation, when we want to run multiple specular
        // simulations in parallel.
        let mut histogram =
            core::array::from_fn(|_| EnergyHistogram::new(histogram_bin_count, sample_rate));

        iter.into_iter().for_each(|seed_direction: Vec3A| {
            let seed_dir = seed_direction.normalize_or_zero();
            if seed_dir.length_squared() == 0.0 {
                return;
            }

            let seed_ray = Ray::new_inf(emitter, seed_dir);

            kernel_rt_cpu_stochastic_specular::<DIFFUSE_RETURN>(
                &scene,
                &microphone,
                &mut histogram,
                seed_ray,
                [1.0; 9],
                0.0,
            );
        });

        histogram
    }
}

/// This implements a stochastic raytracing simulation for specular reflections,
/// with optional diffuse returns from impacts.
///
/// https://reuk.github.io/wayverb/ray_tracer.html
///
fn kernel_rt_cpu_stochastic_specular<const DIFFUSE_RETURN: bool>(
    scene: &Scene,
    microphone: &Microphone,
    histograms: &mut [EnergyHistogram; 9],
    in_ray: Ray,
    in_energy: [f32; 9],
    in_distance: f32,
) {
    // 1. Determine if we hit the microphone on our way to wherever we land.

    let distance_to_mic = intersect_ray_sphere(&in_ray, &microphone.position, 0.1 * 0.1);

    // 2. Determine where the ray impacts in the scene.

    let mut hit = RayHit::none();
    if !scene.accelerator.ray_traverse(in_ray, &mut hit, |ray, id| {
        scene.accelerator_id_to_tri[id].intersect(ray)
    }) {
        return;
    }

    // 3. If we hit a microphone before ray-scene impact, log the energy.

    if distance_to_mic < hit.t {
        let bucket = (((in_distance + distance_to_mic) / SPEED_OF_SOUND)
            * histograms[0].sample_rate)
            .round() as usize;

        if bucket < histograms[0].inner.len() {
            histograms[0].inner[bucket] += in_energy[0];
            histograms[1].inner[bucket] += in_energy[1];
            histograms[2].inner[bucket] += in_energy[2];
            histograms[3].inner[bucket] += in_energy[3];
            histograms[4].inner[bucket] += in_energy[4];
            histograms[5].inner[bucket] += in_energy[5];
            histograms[6].inner[bucket] += in_energy[6];
            histograms[7].inner[bucket] += in_energy[7];
            histograms[8].inner[bucket] += in_energy[8];
        }
    }

    // 4. Determine if we should continue tracing.

    let material = &scene.accelerator_id_to_material[hit.primitive_id as usize];

    let mut out_energy = [0.0f32; 9];
    out_energy[0] = in_energy[0] * (1.0 - material.ac_63hz);
    out_energy[1] = in_energy[1] * (1.0 - material.ac_125hz);
    out_energy[2] = in_energy[2] * (1.0 - material.ac_250hz);
    out_energy[3] = in_energy[3] * (1.0 - material.ac_500hz);
    out_energy[4] = in_energy[4] * (1.0 - material.ac_1000hz);
    out_energy[5] = in_energy[5] * (1.0 - material.ac_2000hz);
    out_energy[6] = in_energy[6] * (1.0 - material.ac_4000hz);
    out_energy[7] = in_energy[7] * (1.0 - material.ac_8000hz);
    out_energy[8] = in_energy[8] * (1.0 - material.ac_16000hz);

    let should_continue = out_energy.iter().any(|&e| e > ENERGY_CUTOFF);

    if !should_continue {
        return; // End recursive trace
    }

    let out_distance = in_distance + hit.t;

    // 5. Reflect the ray back into the scene.

    let triangle = &scene.accelerator_id_to_tri[hit.primitive_id as usize];
    let geometric_normal = triangle.compute_normal();

    // Flip normal to face the incoming ray (the side the ray came from)
    let normal = if geometric_normal.dot(-in_ray.direction) < 0.0 {
        -geometric_normal
    } else {
        geometric_normal
    };

    let out_ray_direction = in_ray.direction.normalize().reflect(normal).normalize();

    // Offset ray to avoid leaking through triangles.
    let out_ray_origin =
        in_ray.origin + in_ray.direction.normalize() * (hit.t - 1e-4) + normal * 1e-4;

    let out_ray = Ray::new_inf(out_ray_origin, out_ray_direction);

    if DIFFUSE_RETURN {
        let return_distance = out_ray_origin.distance(microphone.position) - 0.1;
        let return_dir = (microphone.position - out_ray_origin).normalize();
        let return_ray = Ray::new(out_ray_origin, return_dir, 0.0, return_distance);

        let mut hit = RayHit::none();
        if !scene
            .accelerator
            .ray_traverse(return_ray, &mut hit, |ray, id| {
                scene.accelerator_id_to_tri[id].intersect(ray)
            })
        {
            hit.t = return_distance + 1e-3; // epsilon value
        }

        if return_distance < hit.t {
            let bucket = (((out_distance + return_distance) / SPEED_OF_SOUND)
                * histograms[0].sample_rate)
                .round() as usize;

            let lambert_factor = normal.angle_between(return_dir).cos().abs();

            if bucket < histograms[0].inner.len() {
                histograms[0].inner[bucket] += out_energy[0] * lambert_factor * material.sc_63hz;
                histograms[1].inner[bucket] += out_energy[1] * lambert_factor * material.sc_125hz;
                histograms[2].inner[bucket] += out_energy[2] * lambert_factor * material.sc_250hz;
                histograms[3].inner[bucket] += out_energy[3] * lambert_factor * material.sc_500hz;
                histograms[4].inner[bucket] += out_energy[4] * lambert_factor * material.sc_1000hz;
                histograms[5].inner[bucket] += out_energy[5] * lambert_factor * material.sc_2000hz;
                histograms[6].inner[bucket] += out_energy[6] * lambert_factor * material.sc_4000hz;
                histograms[7].inner[bucket] += out_energy[7] * lambert_factor * material.sc_8000hz;
                histograms[8].inner[bucket] += out_energy[8] * lambert_factor * material.sc_16000hz;
            }
        }
    }

    out_energy[0] *= 1.0 - material.sc_63hz;
    out_energy[1] *= 1.0 - material.sc_125hz;
    out_energy[2] *= 1.0 - material.sc_250hz;
    out_energy[3] *= 1.0 - material.sc_500hz;
    out_energy[4] *= 1.0 - material.sc_1000hz;
    out_energy[5] *= 1.0 - material.sc_2000hz;
    out_energy[6] *= 1.0 - material.sc_4000hz;
    out_energy[7] *= 1.0 - material.sc_8000hz;
    out_energy[8] *= 1.0 - material.sc_16000hz;

    kernel_rt_cpu_stochastic_specular::<DIFFUSE_RETURN>(
        scene,
        microphone,
        histograms,
        out_ray,
        out_energy,
        out_distance,
    );
}
