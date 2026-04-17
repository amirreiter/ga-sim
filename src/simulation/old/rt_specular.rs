use std::{cell::RefCell, sync::Arc};

use glam::Vec3A;
use obvhs::{ray::Ray, rt_triangle::RtTriangle};
use rayon::iter::{ParallelBridge, ParallelIterator};
use thread_local::ThreadLocal;

use crate::{
    fibonacci::FibonacciSphere,
    frequency::SimulationFrequency,
    material::SurfaceMaterial,
    microphone::Microphone,
    scenes::Scene,
    simulation::{shared::EnergyHistogram, trace::{microphone_traceback, trace}},
};

/// A specular raytracing simulation that runs on the CPU, either single-threaded
/// or multi-threaded.
pub fn rt_specular_cpu<F: SimulationFrequency>(
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
        iter.into_par_iter().for_each_init(
            || tl_histogram.clone(),
            |histogram_handle, seed_direction: Vec3A| {
                let local_histogram = histogram_handle.get_or(|| {
                    RefCell::new(EnergyHistogram::new(histogram_bin_count, sample_rate))
                });

                let seed_dir = seed_direction.normalize_or_zero();
                if seed_dir.length_squared() == 0.0 {
                    return;
                }

                let seed_ray = Ray::new_inf(emitter, seed_dir);

                trace::<F, _, 1>(
                    &scene,
                    &microphone,
                    seed_ray,
                    &mut *local_histogram.borrow_mut(),
                    1.0,
                    0.0,
                    specular_procedure::<F>,
                );
            },
        );

        let histogram_inner = Arc::try_unwrap(tl_histogram).unwrap();

        let final_histogram = histogram_inner
            .into_iter()
            .map(|r| r.into_inner())
            .par_bridge()
            .fold(
                || EnergyHistogram::new(histogram_bin_count, sample_rate),
                |mut acc, h| {
                    acc.inner
                        .iter_mut()
                        .zip(h.inner.iter())
                        .for_each(|(a, b)| *a += b);
                    acc
                },
            )
            .reduce(
                || EnergyHistogram::new(histogram_bin_count, sample_rate),
                |mut a, b| {
                    a.inner
                        .iter_mut()
                        .zip(b.inner.iter())
                        .for_each(|(a, b)| *a += b);
                    a
                },
            );

        final_histogram
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

            trace::<F, _, 1>(
                &scene,
                &microphone,
                seed_ray,
                &mut histogram,
                1.0,
                0.0,
                specular_procedure::<F>,
            );
        });

        histogram
    }
}

/// The core procedure for a specular pass.
pub fn specular_procedure<F: SimulationFrequency>(
    scene: &Scene,
    microphone: &Microphone,
    histogram: &mut EnergyHistogram,
    ray: Ray,
    ray_t: f32,
    distance_travelled: f32,
    triangle: &RtTriangle,
    material: &SurfaceMaterial,
    in_energy: f32,
) -> (Ray, f32, f32) {
    const EPSILON: f32 = 0.001;

    let in_dir = ray.direction.normalize_or_zero();
    if in_dir.length_squared() == 0.0 {
        println!(
            "[SPECULAR_SANITY] zero in_dir; origin=({:.6}, {:.6}, {:.6}), direction=({:.6}, {:.6}, {:.6}), ray_t={:.6}, distance_travelled={:.6}, in_energy={:.6}",
            ray.origin.x,
            ray.origin.y,
            ray.origin.z,
            ray.direction.x,
            ray.direction.y,
            ray.direction.z,
            ray_t,
            distance_travelled,
            in_energy
        );
        return (ray, 0.0, distance_travelled);
    }

    let hit_pos = ray.origin + in_dir * ray_t;
    let energy_after_travel = in_energy; // * distance_travelled / SPEED_OF_SOUND;// * F::air_alpha().powf(distance_travelled);

    let normalized_ray = Ray::new_inf(ray.origin, in_dir);
    microphone_traceback::<F>(
        scene,
        microphone,
        histogram,
        normalized_ray,
        in_energy,
        distance_travelled,
    );

    let mut n = triangle.compute_normal().normalize_or_zero();
    if n.length_squared() == 0.0 {
        let raw_n = triangle.compute_normal();
        println!(
            "[SPECULAR_SANITY] zero surface normal; raw_normal=({:.6}, {:.6}, {:.6}), hit_pos=({:.6}, {:.6}, {:.6}), ray_t={:.6}, distance_travelled={:.6}, in_energy={:.6}",
            raw_n.x,
            raw_n.y,
            raw_n.z,
            hit_pos.x,
            hit_pos.y,
            hit_pos.z,
            ray_t,
            distance_travelled,
            in_energy
        );
        return (ray, 0.0, distance_travelled + ray_t);
    }
    if in_dir.dot(n) > 0.0 {
        n = -n;
    }

    let out_dir = in_dir.reflect(n).normalize_or_zero();
    if out_dir.length_squared() == 0.0 {
        println!(
            "[SPECULAR_SANITY] zero out_dir; in_dir=({:.6}, {:.6}, {:.6}), normal=({:.6}, {:.6}, {:.6}), hit_pos=({:.6}, {:.6}, {:.6}), ray_t={:.6}, distance_travelled={:.6}, in_energy={:.6}",
            in_dir.x,
            in_dir.y,
            in_dir.z,
            n.x,
            n.y,
            n.z,
            hit_pos.x,
            hit_pos.y,
            hit_pos.z,
            ray_t,
            distance_travelled,
            in_energy
        );
        return (ray, 0.0, distance_travelled + ray_t);
    }

    let out_origin = hit_pos + out_dir * EPSILON;
    let out_ray = Ray::new_inf(out_origin, out_dir);

    // TODO: Make 1-ac const
    let out_energy = energy_after_travel * (1.0 - F::ac(material)) * (1.0 - F::sc(material));

    (out_ray, out_energy, distance_travelled + ray_t)
}
