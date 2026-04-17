use std::{
    cell::RefCell,
    sync::{Arc, atomic::Ordering},
};

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
        shared::{DEBUG_LEAK_COUNTER, DEBUG_MIC_HITS, DEBUG_MIC_HITS_OUT_OF_BOUNDS},
    },
};

pub const SPEED_OF_SOUND: f32 = 343.0;

pub const ENERGY_CUTOFF: f32 = 0.000001;
pub const HIT_EPSILON: f32 = 0.01;

/// This implements a stochastic raytracing simulation.
/// https://reuk.github.io/wayverb/ray_tracer.html
///
/// This means the histogram is populated by
fn cpu_stochastic_rt_specular(
    scene: &Scene,
    microphone: &Microphone,
    histograms: &mut [EnergyHistogram; 8],
    in_ray: Ray,
    in_energy: [f32; 8],
    in_distance: f32,
) {
    // println!("e: {}       d: {}", in_energy[0], in_distance);
    // println!("{:?}", in_ray);
    // println!("\n");

    // 1. Determine if we hit the microphone on our way to wherever we land.

    let distance_to_mic = intersect_ray_sphere(&in_ray, &microphone.position);

    // 2. Determine where the ray impacts in the scene.

    let mut hit = RayHit::none();
    if !scene.accelerator.ray_traverse(in_ray, &mut hit, |ray, id| {
        // scene.accelerator_id_to_tri[id].intersect(ray)
        scene.triangles[scene.accelerator.primitive_indices[id] as usize].intersect(ray)
    }) {
        if true {
            DEBUG_LEAK_COUNTER.fetch_add(1, Ordering::Relaxed);
        }
        // println!("--A");
        // println!("      {:?}", in_ray);
        return;
    }

    if hit.t > 1000000.0 {
        if true {
            DEBUG_LEAK_COUNTER.fetch_add(1, Ordering::Relaxed);
        }
        // println!("--B");
        return;
    }

    if hit.t < HIT_EPSILON {
        DEBUG_LEAK_COUNTER.fetch_add(1, Ordering::Relaxed);
        // println!("--C");
        return;
    }

    // 3. If we hit a microphone before ray-scene impact, log the energy.

    if distance_to_mic < hit.t {
        if true {
            DEBUG_MIC_HITS.fetch_add(1, Ordering::Relaxed);
        }

        let bucket = (((in_distance + distance_to_mic) / SPEED_OF_SOUND)
            * histograms[0].sample_rate)
            .round() as usize;

        if bucket < histograms[0].inner.len() {
            // let contribution = in_energy; // * (distance_to_microphone) / SPEED_OF_SOUND;// * F::air_alpha().powf(distance_to_microphone);
            histograms[0].inner[bucket] += in_energy[0];
            histograms[1].inner[bucket] += in_energy[1];
            histograms[2].inner[bucket] += in_energy[2];
            histograms[3].inner[bucket] += in_energy[3];
            histograms[4].inner[bucket] += in_energy[4];
            histograms[5].inner[bucket] += in_energy[5];
            histograms[6].inner[bucket] += in_energy[6];
            histograms[7].inner[bucket] += in_energy[7];
        } else {
            if true {
                DEBUG_MIC_HITS_OUT_OF_BOUNDS.fetch_add(1, Ordering::Relaxed);
                // println!("--D");
                return;
            }
        }
    }

    // 4. Reflect the ray back into the scene.
    // TODO: Some calculations can be gated on the `if should_continue` block.

    // let triangle = &scene.accelerator_id_to_tri[hit.primitive_id as usize];
    let triangle =
        &scene.triangles[scene.accelerator.primitive_indices[hit.primitive_id as usize] as usize];

    // let material = &scene.accelerator_id_to_material[hit.primitive_id as usize];
    let material = &scene.materials[scene.material_indicies[hit.primitive_id as usize] as usize];

    let geometric_normal = triangle.compute_normal();

    // Flip normal to face the incoming ray (i.e., the side the ray came from)
    let normal = if geometric_normal.dot(-in_ray.direction) < 0.0 {
        -geometric_normal
    } else {
        geometric_normal
    };

    // Now reflect — result is guaranteed to be on the correct side
    let out_ray_direction = in_ray.direction.normalize().reflect(normal).normalize();

    // Offset origin along the normal to avoid self-intersection
    let out_ray_origin = in_ray.origin
        + in_ray.direction.normalize() * (hit.t - 1e-4)
        + normal * 1e-4;  // epsilon offset along the outward normal

    let out_ray = Ray::new_inf(out_ray_origin, out_ray_direction);

    let mut out_energy = [0.0f32; 8];
    out_energy[0] = in_energy[0] * (1.0 - material.ac_125hz);
    out_energy[1] = in_energy[1] * (1.0 - material.ac_250hz);
    out_energy[2] = in_energy[2] * (1.0 - material.ac_500hz);
    out_energy[3] = in_energy[3] * (1.0 - material.ac_1000hz);
    out_energy[4] = in_energy[4] * (1.0 - material.ac_2000hz);
    out_energy[5] = in_energy[5] * (1.0 - material.ac_4000hz);
    out_energy[6] = in_energy[6] * (1.0 - material.ac_8000hz);
    out_energy[7] = in_energy[7] * (1.0 - material.ac_16000hz);

    let out_distance = in_distance + hit.t;

    let should_continue = out_energy.iter().any(|&e| e > ENERGY_CUTOFF);

    if should_continue {
        cpu_stochastic_rt_specular(
            scene,
            microphone,
            histograms,
            out_ray,
            out_energy,
            out_distance,
        )
    } else {
        // println!("--E");
    }
}

pub fn intersect_ray_sphere(r: &Ray, center: &Vec3A) -> f32 {
    // TODO: un-hardcode radius
    const RADIUS: f32 = 0.5;
    const RADIUS_SQ: f32 = RADIUS * RADIUS;

    let p = r.origin;
    let d = r.direction;

    let m = p - *center;
    let b = m.dot(d);
    let c = m.dot(m) - RADIUS_SQ;

    // If we are outside (c > 0) and pointing away (b > 0), we'll never hit
    if c > 0.0 && b > 0.0 {
        return f32::INFINITY;
    }

    let discr = b * b - c;
    if discr < 0.0 {
        return f32::INFINITY;
    }

    let sqrt_discr = discr.sqrt();
    let t0 = -b - sqrt_discr;
    let t1 = -b + sqrt_discr;

    // Logic for returning the correct hit:
    if t0 > 0.001 {
        // Small epsilon to avoid self-hits at the boundary
        return t0;
    } else if t1 > 0.001 {
        // If t0 is behind us or 0, we check the exit point t1
        return t1;
    }

    f32::INFINITY
}

/// A specular raytracing simulation that runs on the CPU, either single-threaded
/// or multi-threaded.
pub fn cpu_stochastic_rt(
    multithread: bool,
    scene: &Scene,
    microphone: &Microphone,
    emitter: Vec3A,
    sample_rate: f32,
    ray_count: u64,
    histogram_bin_count: usize,
) -> [EnergyHistogram; 8] {
    let iter = FibonacciSphere::new(ray_count);

    if multithread {
        // Multi threaded implementation, when we want to run a singular specular
        // simulation in parallel.

        let tl_histogram: Arc<ThreadLocal<RefCell<[EnergyHistogram; 8]>>> =
            Arc::new(ThreadLocal::new());
        iter.into_par_iter().for_each_init(
            || tl_histogram.clone(),
            |histogram_handle, seed_direction: Vec3A| {
                let local_histogram = histogram_handle.get_or(|| {
                    RefCell::new(core::array::from_fn(|_| EnergyHistogram::new(histogram_bin_count, sample_rate)))
                });

                let seed_dir = seed_direction.normalize();
                if seed_dir.length_squared() == 0.0 {
                    return;
                }

                let seed_ray = Ray::new_inf(emitter, seed_dir);

                cpu_stochastic_rt_specular(
                    &scene,
                    &microphone,
                    &mut *local_histogram.borrow_mut(),
                    seed_ray,
                    [1.0; 8],
                    0.0,
                );
            },
        );

        let histogram_inner = Arc::try_unwrap(tl_histogram).unwrap();

        // Final result is now an array of 8 histograms
        let final_histograms = histogram_inner
            .into_iter()
            .map(|r| r.into_inner())
            .par_bridge()
            .fold(
                // Identity: Create a blank array of 8 histograms
                || core::array::from_fn(|_| EnergyHistogram::new(histogram_bin_count, sample_rate)),
                |mut acc_array, local_array| {
                    // Zip the arrays (length 8)
                    for (acc_h, local_h) in acc_array.iter_mut().zip(local_array.iter()) {
                        // Zip the bins within each histogram
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
        let mut histogram = core::array::from_fn(|_| EnergyHistogram::new(histogram_bin_count, sample_rate));

        iter.into_iter().for_each(|seed_direction: Vec3A| {
            let seed_dir = seed_direction.normalize_or_zero();
            if seed_dir.length_squared() == 0.0 {
                return;
            }

            let seed_ray = Ray::new_inf(emitter, seed_dir);

            cpu_stochastic_rt_specular(
                &scene,
                &microphone,
                &mut histogram,
                seed_ray,
                [1.0; 8],
                0.0,
            );
        });

        histogram
    }
}
