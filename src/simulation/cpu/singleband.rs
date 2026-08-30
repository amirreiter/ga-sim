use std::{
    cell::RefCell,
    sync::{Arc, atomic::Ordering::Relaxed},
};

use glam::Vec3A;
use obvhs::ray::{Ray, RayHit};
use rand::{
    RngExt, SeedableRng,
    rngs::{StdRng, ThreadRng},
};
use rayon::iter::{ParallelBridge, ParallelIterator};
use thread_local::ThreadLocal;

use crate::{
    fibonacci::FibonacciSphere,
    frequency::SimulationFrequency,
    microphone::Microphone,
    scenes::Scene,
    simulation::{
        DEBUG_LEAK_COUNTER, DEBUG_MIC_HITS_OUT_OF_BOUNDS, EnergyHistogram,
        cpu::{
            ENERGY_CUTOFF, SPEED_OF_SOUND, intersect_ray_sphere, sample_cosine_weighted_hemisphere,
        },
    },
};

pub fn cpu_rt_stochastic_singleband<F: SimulationFrequency>(
    multithread: bool,
    scene: &Scene,
    microphone: &Microphone,
    emitter: Vec3A,
    sample_rate: f32,
    ray_count: u64,
    histogram_bin_count: usize,
) -> EnergyHistogram {
    let iter = FibonacciSphere::new(ray_count);

    if multithread {
        // Multi threaded implementation, when we want to run a singular specular
        // simulation in parallel.

        let tl_histogram: Arc<ThreadLocal<RefCell<EnergyHistogram>>> = Arc::new(ThreadLocal::new());
        iter.into_iter().enumerate().par_bridge().for_each_init(
            || tl_histogram.clone(),
            |histogram_handle, (ray_index, seed_direction)| {
                let local_histogram = histogram_handle.get_or(|| {
                    RefCell::new(EnergyHistogram::new(histogram_bin_count, sample_rate))
                });

                let seed_dir = seed_direction.normalize();
                let seed_ray = Ray::new_inf(emitter, seed_dir);

                let mut rng = StdRng::seed_from_u64(123u64.wrapping_add(ray_index as u64));

                kernel_rt_stochastic_single_band::<F, false>(
                    scene,
                    microphone,
                    &mut *local_histogram.borrow_mut(),
                    seed_ray,
                    1.0,
                    0.0,
                    &mut rng,
                );
            },
        );

        let histogram_inner = Arc::try_unwrap(tl_histogram).unwrap();

        let final_histograms = histogram_inner
            .into_iter()
            .map(|r| r.into_inner())
            .par_bridge()
            .fold(
                || EnergyHistogram::new(histogram_bin_count, sample_rate),
                |mut acc, local| {
                    acc.inner
                        .iter_mut()
                        .zip(local.inner.iter())
                        .for_each(|(a, b)| *a += b);
                    acc
                },
            )
            .reduce(
                || EnergyHistogram::new(histogram_bin_count, sample_rate),
                |mut a_array, b_array| {
                    a_array
                        .inner
                        .iter_mut()
                        .zip(b_array.inner.iter())
                        .for_each(|(a, b)| *a += b);
                    a_array
                },
            );

        final_histograms
    } else {
        // Single threaded implementation, when we want to run multiple specular
        // simulations in parallel.
        let mut histogram = EnergyHistogram::new(histogram_bin_count, sample_rate);

        iter.into_iter().for_each(|seed_direction: Vec3A| {
            let seed_dir = seed_direction.normalize_or_zero();
            if seed_dir.length_squared() == 0.0 {
                return;
            }

            let seed_ray = Ray::new_inf(emitter, seed_dir);

            kernel_rt_stochastic_single_band::<F, false>(
                &scene,
                &microphone,
                &mut histogram,
                seed_ray,
                1.0,
                0.0,
                &mut StdRng::seed_from_u64(123),
            );
        });

        histogram
    }
}

fn kernel_rt_stochastic_single_band<F: SimulationFrequency, const MICROPHONE_COLLISION: bool>(
    scene: &Scene,
    microphone: &Microphone,
    histogram: &mut EnergyHistogram,
    in_ray: Ray,
    in_energy: f32,
    in_distance: f32,
    rng: &mut StdRng,
) {
    // 1. Determine if we hit the microphone on our way to wherever we land.

    const RECEIVER_RADIUS: f32 = 0.05;

    let distance_to_mic = intersect_ray_sphere(
        &in_ray,
        &microphone.position,
        RECEIVER_RADIUS * RECEIVER_RADIUS,
    );

    // 2. Determine where the ray impacts in the scene.

    let mut hit = RayHit::none();
    if !scene.accelerator.ray_traverse(in_ray, &mut hit, |ray, id| {
        scene.accelerator_id_to_tri[id].intersect(ray)
    }) {
        // println!("{:?}", in_ray);
        DEBUG_LEAK_COUNTER.fetch_add(1, Relaxed);
        return;
    }

    // 3. If we hit a microphone before ray-scene impact, log the energy.
    // This can be disabled for rays that are children of diffuse rays, who
    // already have diffuse rain. This prevents double counting.
    if MICROPHONE_COLLISION && distance_to_mic < hit.t {
        let bucket = (((in_distance + distance_to_mic) / SPEED_OF_SOUND) * histogram.sample_rate)
            .round() as usize;

        if bucket < histogram.inner.len() {
            histogram.inner[bucket] += in_energy
                * F::air_decay_multiplier(in_distance + distance_to_mic)
                * microphone.amplitude_multiplier_normalized(in_ray.direction);
        } else {
            DEBUG_MIC_HITS_OUT_OF_BOUNDS.fetch_add(1, Relaxed);
        }
    }

    // 4. Apply absorption.

    let material = &scene.accelerator_id_to_material[hit.primitive_id as usize];

    let out_energy = in_energy * (1.0 - F::ac(material));

    if out_energy < ENERGY_CUTOFF {
        return;
    }

    let out_distance = in_distance + hit.t;

    // 5. Reflect the ray back into the scene.

    let triangle = &scene.accelerator_id_to_tri[hit.primitive_id as usize];
    let geometric_normal = triangle.compute_normal();

    // Flip normal to face the incoming ray (the side the ray came from).
    let normal = if geometric_normal.dot(-in_ray.direction) < 0.0 {
        -geometric_normal
    } else {
        geometric_normal
    };

    // Offset ray to avoid leaking through triangles.
    let out_ray_origin = in_ray.origin + in_ray.direction * (hit.t - 1e-4) + normal * 1e-4;

    // Determine what species the next ray should be
    if F::sc(material) > rng.random::<f32>() {
        // The next ray should be a scattered ray.

        // Diffuse rain
        if true {
            let receiver_center_distance = out_ray_origin.distance(microphone.position);

            if receiver_center_distance > RECEIVER_RADIUS {
                let return_dir = (microphone.position - out_ray_origin).normalize();

                // Diffuse scattering only exists in the outward hemisphere.
                let cos_theta = normal.dot(return_dir).max(0.0);

                if cos_theta > 0.0 {
                    let return_distance = receiver_center_distance - RECEIVER_RADIUS;

                    let return_ray = Ray::new(out_ray_origin, return_dir, 0.0, return_distance);

                    let mut return_hit = RayHit::none();

                    if !scene
                        .accelerator
                        .ray_traverse(return_ray, &mut return_hit, |ray, id| {
                            scene.accelerator_id_to_tri[id].intersect(ray)
                        })
                    {
                        return_hit.t = return_distance + 1e-4;
                    }

                    if return_distance < return_hit.t {
                        let bucket = (((out_distance + return_distance) / SPEED_OF_SOUND)
                            * histogram.sample_rate)
                            .round() as usize;

                        if bucket < histogram.inner.len() {
                            // Fraction of the cosine-weighted diffuse hemisphere
                            // occupied by the spherical receiver.
                            //
                            // sin(gamma) = a / r
                            //
                            // Integral of cos(theta)/pi over the receiver's
                            // solid-angle cap gives approximately:
                            //
                            //     cos(theta) * (a/r)^2
                            //
                            let radius_ratio = RECEIVER_RADIUS / receiver_center_distance;

                            let receiver_factor = cos_theta * radius_ratio * radius_ratio;

                            histogram.inner[bucket] += out_energy
                                * receiver_factor
                                * F::air_decay_multiplier(out_distance + return_distance)
                                * microphone.amplitude_multiplier_normalized(return_dir);
                        } else {
                            DEBUG_MIC_HITS_OUT_OF_BOUNDS.fetch_add(1, Relaxed);
                        }
                    }
                }
            }
        }

        let out_ray_direction = sample_cosine_weighted_hemisphere(normal, rng);

        let out_ray = Ray::new_inf(out_ray_origin, out_ray_direction);

        kernel_rt_stochastic_single_band::<F, false>(
            scene,
            microphone,
            histogram,
            out_ray,
            out_energy,
            out_distance,
            rng,
        );
    } else {
        // The next ray should be a specular ray

        let out_ray_direction = in_ray.direction.normalize().reflect(normal).normalize();
        let out_ray = Ray::new_inf(out_ray_origin, out_ray_direction);

        kernel_rt_stochastic_single_band::<F, true>(
            scene,
            microphone,
            histogram,
            out_ray,
            out_energy,
            out_distance,
            rng,
        );
    }
}
