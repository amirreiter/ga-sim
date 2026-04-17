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
    random::sample_hemisphere,
    scenes::Scene,
    simulation::{EnergyHistogram, trace::{SPEED_OF_SOUND, microphone_traceback, trace}},
};

/// A diffuse raytracing simulation that runs on the CPU, either single-threaded
/// or multi-threaded.
pub fn rt_diffuse_cpu<F: SimulationFrequency, const BranchCount: u32>(
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

                let seed_ray = Ray::new_inf(emitter, seed_direction);

                trace::<F, _, BranchCount>(
                    &scene,
                    &microphone,
                    seed_ray,
                    &mut *local_histogram.borrow_mut(),
                    1.0,
                    0.0,
                    diffuse_procedure::<F>,
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
            let seed_ray = Ray::new_inf(emitter, seed_direction);

            trace::<F, _, BranchCount>(
                &scene,
                &microphone,
                seed_ray,
                &mut histogram,
                1.0,
                0.0,
                diffuse_procedure::<F>,
            );
        });

        histogram
    }
}

/// The core procedure for a diffuse pass.
pub fn diffuse_procedure<F: SimulationFrequency>(
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
    let current_pos = ray.origin + ray.direction * ray_t;
    let energy_after_travel = in_energy;// * distance_travelled / SPEED_OF_SOUND; // F::air_alpha().powf(distance_travelled);

    microphone_traceback::<F>(
        scene,
        microphone,
        histogram,
        ray,
        in_energy,
        distance_travelled,
    );

    let tri_normal = triangle.compute_normal();
    let out_dir = ray.direction.reflect(tri_normal);
    let out_ray = Ray::new_inf(current_pos, sample_hemisphere(tri_normal));

    // TODO: Make 1-ac const
    let out_energy =
        energy_after_travel * (1.0 - F::ac(material)) * F::sc(material) * tri_normal.angle_between(out_dir).cos();

    (out_ray, out_energy, distance_travelled)
}
