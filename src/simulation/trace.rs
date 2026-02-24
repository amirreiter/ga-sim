use obvhs::{
    ray::{Ray, RayHit},
    rt_triangle::RtTriangle,
};

use crate::{
    frequency::SimulationFrequency, material::SurfaceMaterial, microphone::Microphone,
    scenes::Scene, simulation::shared::EnergyHistogram,
};

const ENERGY_CUTOFF: f32 = 0.1;

/// A generic trace function skeletion which raytraces a scene, running a procedure
/// at every intersection.
/// Procedure `P` must follow `let (new_ray, new_energy) = procedure(histogram, in_ray, *triangle, in_energy);`
pub fn trace<F: SimulationFrequency, P>(
    scene: &Scene,
    microphone: &Microphone,
    in_ray: Ray,
    histogram: &mut EnergyHistogram,
    in_energy: f32,
    procedure: P,
) where
    P: Fn(
        &Scene,
        &Microphone,
        &mut EnergyHistogram,
        Ray,
        f32,
        &RtTriangle,
        &SurfaceMaterial,
        f32,
    ) -> (Ray, f32),
{
    let mut hit = RayHit::none();
    if !scene.accelerator.ray_traverse(in_ray, &mut hit, |ray, id| {
        scene.triangles[scene.accelerator.primitive_indices[id] as usize].intersect(ray)
    }) {
        return;
    }

    let primitive_id_usize = hit.primitive_id as usize;

    let triangle =
        &scene.triangles[scene.accelerator.primitive_indices[primitive_id_usize] as usize];

    let material = &scene.materials[scene.material_indicies[primitive_id_usize] as usize];

    let (new_ray, new_energy) = procedure(
        scene, microphone, histogram, in_ray, hit.t, triangle, material, in_energy,
    );

    if new_energy > ENERGY_CUTOFF {
        trace::<F, P>(scene, microphone, new_ray, histogram, new_energy, procedure)
    }
}
