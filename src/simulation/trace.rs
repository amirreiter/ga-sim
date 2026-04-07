use std::{f32, time::Instant};

use glam::Vec3A;
use obvhs::{
    ray::{Ray, RayHit},
    rt_triangle::RtTriangle,
};
use unroll_lite::unroll;

use crate::{
    frequency::SimulationFrequency, material::SurfaceMaterial, microphone::Microphone,
    scenes::Scene, simulation::shared::EnergyHistogram,
};

const ENERGY_CUTOFF: f32 = 0.00001;
pub const SPEED_OF_SOUND: f32 = 343.0;

/// A generic trace function skeletion which raytraces a scene, running a procedure
/// at every intersection.
/// Procedure `P` must follow `let (new_ray, new_energy) = procedure(histogram, in_ray, *triangle, in_energy);`
pub fn trace<F: SimulationFrequency, P, const BranchCount: u32>(
    scene: &Scene,
    microphone: &Microphone,
    in_ray: Ray,
    histogram: &mut EnergyHistogram,
    in_energy: f32,
    distance_travelled: f32,
    procedure: P,
) where
    P: Copy
        + Fn(
            &Scene,
            &Microphone,
            &mut EnergyHistogram,
            Ray,
            f32,
            f32,
            &RtTriangle,
            &SurfaceMaterial,
            f32,
        ) -> (Ray, f32, f32),
{
    let mut hit = RayHit::none();
    if !scene.accelerator.ray_traverse(in_ray, &mut hit, |ray, id| {
        scene.accelerator_id_to_tri[id].intersect(ray)
        // scene.triangles[scene.accelerator.primitive_indices[id] as usize].intersect(ray)
    }) {
        println!("LEAK!");
        return;
    }

    let primitive_id_usize = hit.primitive_id as usize;

    let triangle = &scene.accelerator_id_to_tri[primitive_id_usize];
    // let triangle =
    //     &scene.triangles[scene.accelerator.primitive_indices[primitive_id_usize] as usize];

    let material = &scene.accelerator_id_to_material[primitive_id_usize];
    // let material = &scene.materials[scene.material_indicies[primitive_id_usize] as usize];

    unroll!(_ in 0..BranchCount => {
        let (new_ray, new_energy, distance_travelled) = procedure(
            scene,
            microphone,
            histogram,
            in_ray,
            hit.t,
            distance_travelled,
            triangle,
            material,
            in_energy,
        );

        if new_energy > ENERGY_CUTOFF {
            trace::<F, P, BranchCount>(
                scene,
                microphone,
                new_ray,
                histogram,
                new_energy,// / BranchCount as f32,
                distance_travelled,
                procedure,
            )
        }
    });
}

/// A sub-procedure for microphone traceback from location `current_pos`.
pub fn microphone_traceback<F: SimulationFrequency>(
    scene: &Scene,
    microphone: &Microphone,
    histogram: &mut EnergyHistogram,
    last_ray: Ray,
    in_energy: f32,
    distance_travelled: f32,
) {
    // let microphone_pos = microphone.position;

    let distance_to_microphone = intersect_ray_sphere(&last_ray, &microphone.position);
    // let distance_to_microphone = last_ray.origin.distance(microphone_pos); //intersect_ray_sphere(&last_ray, &microphone.position);

    if distance_to_microphone == f32::INFINITY {
        return;
    }

    let mut hit = RayHit::none();
    if !scene
        .accelerator
        .ray_traverse(last_ray, &mut hit, |ray, id| {
            scene.accelerator_id_to_tri[id].intersect(ray)
            // scene.triangles[scene.accelerator.primitive_indices[id] as usize].intersect(ray)
        })
    {
        return;
    }

    // We have line-of-sight to the microphone.
    if hit.t > distance_to_microphone {
        let bucket = (((distance_travelled + distance_to_microphone) / SPEED_OF_SOUND) * histogram.sample_rate)
            .round() as usize;
        if bucket < histogram.inner.len() {
            // TODO: Multiply by microphone bias.
            let contribution = in_energy; // * (distance_to_microphone) / SPEED_OF_SOUND;// * F::air_alpha().powf(distance_to_microphone);
            // println!("{}", contribution);
            histogram.inner[bucket] += contribution;
        }
    }
}

pub fn intersect_ray_sphere(r: &Ray, center: &Vec3A) -> f32 {
    // TODO: un-hardcode radius;
    const RADIUS: f32 = 1.5;
    const RADIUS_SQ: f32 = RADIUS * RADIUS;

    let p = r.origin;
    let d = r.direction;

    let m = p - center;
    let b = m.dot(d);
    let c = m.dot(m) - RADIUS_SQ;

    if c > 0.0 && b > 0.0 {
        return f32::INFINITY;
    }

    let discr = b * b - c;

    if discr < 0.0 {
        return f32::INFINITY;
    }

    let mut t = -b - discr.sqrt();

    if t < 0.0 {
        t = 0.0;
    }

    t
}
