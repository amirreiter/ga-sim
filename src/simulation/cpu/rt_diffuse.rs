use std::{cell::RefCell, sync::Arc};

use glam::Vec3A;
use obvhs::ray::{Ray, RayHit};
use rand::rngs::ThreadRng;
use rayon::iter::{ParallelBridge, ParallelIterator};
use thread_local::ThreadLocal;

use crate::{
    fibonacci::FibonacciSphere,
    microphone::Microphone,
    scenes::Scene,
    simulation::{
        DEBUG_LEAK_COUNTER, DEBUG_MIC_HITS, EnergyHistogram, cpu::{ENERGY_CUTOFF, SPEED_OF_SOUND, intersect_ray_sphere, random_vector_off_normal}
    },
};

pub fn cpu_rt_stochastic_diffuse<const BRANCH_COUNT: usize>(
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
            || (tl_histogram.clone(), ThreadRng::default()),
            |(histogram_handle, rng), seed_direction: Vec3A| {
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

                kernel_cpu_rt_stochastic_diffuse::<BRANCH_COUNT>(
                    rng,
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

        let mut rng = ThreadRng::default();

        iter.into_iter().for_each(|seed_direction: Vec3A| {
            let seed_dir = seed_direction.normalize_or_zero();
            if seed_dir.length_squared() == 0.0 {
                return;
            }

            let seed_ray = Ray::new_inf(emitter, seed_dir);

            kernel_cpu_rt_stochastic_diffuse::<BRANCH_COUNT>(
                &mut rng,
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

/// This implements a stochastic raytracing simulation for diffuse reflections,
/// with the capability of branching on impact to sample multiple directions
/// from an impact.
///
/// https://reuk.github.io/wayverb/ray_tracer.html
///
fn kernel_cpu_rt_stochastic_diffuse<const BRANCH_COUNT: usize>(
    rng: &mut ThreadRng,
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
        // DEBUG_LEAK_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return;
    }

    // 3. If we hit a microphone before ray-scene impact, log the energy.

    if distance_to_mic < hit.t {
        let bucket = (((in_distance + distance_to_mic) / SPEED_OF_SOUND)
            * histograms[0].sample_rate)
            .round() as usize;

        // DEBUG_MIC_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

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

    // 4. Preload information for later calculations.

    let triangle = &scene.accelerator_id_to_tri[hit.primitive_id as usize];
    let geometric_normal = triangle.compute_normal();

    // Flip normal to face the incoming ray (i.e., the side the ray came from)
    let normal = if geometric_normal.dot(-in_ray.direction) < 0.0 {
        -geometric_normal
    } else {
        geometric_normal
    };

    let out_distance = in_distance + hit.t;

    unroll_lite::unroll!(_ in 0..BRANCH_COUNT => {
        // 5. Determine if we should continue tracing.

        let material = &scene.accelerator_id_to_material[hit.primitive_id as usize];

        let mut out_energy = [0.0f32; 9];
        out_energy[0] = in_energy[0] * (1.0 - material.ac_63hz) * (material.sc_63hz);
        out_energy[1] = in_energy[1] * (1.0 - material.ac_125hz) * (material.sc_125hz);
        out_energy[2] = in_energy[2] * (1.0 - material.ac_250hz) * (material.sc_250hz);
        out_energy[3] = in_energy[3] * (1.0 - material.ac_500hz) * (material.sc_500hz);
        out_energy[4] = in_energy[4] * (1.0 - material.ac_1000hz) * (material.sc_1000hz);
        out_energy[5] = in_energy[5] * (1.0 - material.ac_2000hz) * (material.sc_2000hz);
        out_energy[6] = in_energy[6] * (1.0 - material.ac_4000hz) * (material.sc_4000hz);
        out_energy[7] = in_energy[7] * (1.0 - material.ac_8000hz) * (material.sc_8000hz);
        out_energy[8] = in_energy[8] * (1.0 - material.ac_16000hz) * (material.sc_16000hz);

        let out_ray_direction = random_vector_off_normal(normal, rng);
        let lambert_factor = normal.angle_between(out_ray_direction).cos().abs();

        unroll_lite::unroll!(i in 0..9 => {
            out_energy[i] = out_energy[i] * lambert_factor;
        });

        let should_continue = out_energy.iter().any(|&e| e > ENERGY_CUTOFF * 0.1);

        if !should_continue {
            return; // End recursive trace
        }

        // Offset ray to avoid leaking through triangles.
        let out_ray_origin =
            in_ray.origin + in_ray.direction.normalize() * (hit.t - 1e-4) + normal * 1e-4;

        let out_ray = Ray::new_inf(out_ray_origin, out_ray_direction);

        kernel_cpu_rt_stochastic_diffuse::<BRANCH_COUNT>(
            rng,
            scene,
            microphone,
            histograms,
            out_ray,
            out_energy,
            out_distance,
        );
    });
}
