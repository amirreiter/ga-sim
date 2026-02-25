pub mod cr3;

use std::time::Duration;

use glam::Vec3A;
use obvhs::{
    BvhBuildParams,
    cwbvh::{CwBvh, builder::build_cwbvh},
    rt_triangle::{RtCompressedTriangle, RtTriangle},
};
use tobj;

use crate::material::SurfaceMaterial;

pub struct Scene {
    pub triangles: Vec<RtTriangle>,
    pub gpu_triangles: Vec<RtCompressedTriangle>,
    pub material_indicies: Vec<u8>,
    pub materials: Vec<SurfaceMaterial>,
    pub accelerator: CwBvh,
    pub accelerator_id_to_tri: Vec<RtTriangle>,
    pub accelerator_id_to_material: Vec<SurfaceMaterial>,
}

impl Scene {
    pub fn from_obj(obj: Vec<tobj::Model>, materials: Vec<SurfaceMaterial>) -> Scene {
        let material_indicies: Vec<u8> = obj
            .iter()
            .map(|model| {
                model
                    .mesh
                    .indices
                    .chunks_exact(3)
                    .enumerate()
                    .filter_map(|(_chunk_index, _indicies)| match model.mesh.material_id {
                        Some(i) => Some(i as u8),
                        None => None,
                    })
                    .collect()
            })
            .collect::<Vec<Vec<u8>>>()
            .into_iter()
            .flatten()
            .collect();

        let gpu_triangles: Vec<RtCompressedTriangle> = obj
            .iter()
            .map(|model| {
                let positions = &model.mesh.positions;

                let mesh_tris = model
                    .mesh
                    .indices
                    .chunks_exact(3)
                    .enumerate()
                    .filter_map(|(_chunk_index, indicies)| {
                        // Materials are saved in a separate iterator.

                        let i0 = (indicies[0] as usize) * 3;
                        let i1 = (indicies[1] as usize) * 3;
                        let i2 = (indicies[2] as usize) * 3;

                        let v0 = Vec3A::new(positions[i0], positions[i0 + 1], positions[i0 + 2]);
                        let v1 = Vec3A::new(positions[i1], positions[i1 + 1], positions[i1 + 2]);
                        let v2 = Vec3A::new(positions[i2], positions[i2 + 1], positions[i2 + 2]);

                        Some(RtCompressedTriangle::new(v0, v1, v2))
                    })
                    .collect::<Vec<RtCompressedTriangle>>();

                mesh_tris
            })
            .collect::<Vec<Vec<RtCompressedTriangle>>>()
            .into_iter()
            .flatten()
            .collect();

        let triangles: Vec<RtTriangle> = gpu_triangles
            .iter()
            .map(|gpu_tri| {
                let unpack = gpu_tri.unpack();
                RtTriangle::new(unpack.0, unpack.1, unpack.2)
            })
            .collect();

        let accelerator = build_cwbvh(
            &triangles,
            BvhBuildParams::very_slow_build(),
            &mut Duration::ZERO.clone(), // Not using Cwbvh build profiling
        );

        // Accelerate OBVHS lookups.
        let accelerator_id_to_tri: Vec<RtTriangle> = accelerator
            .primitive_indices
            .iter()
            .map(|&original_idx| triangles[original_idx as usize].clone())
            .collect();

        let accelerator_id_to_material: Vec<SurfaceMaterial> = accelerator
            .primitive_indices
            .iter()
            .map(|&original_idx| {
                materials[material_indicies[original_idx as usize] as usize].clone()
            })
            .collect();

        Scene {
            triangles,
            gpu_triangles,
            material_indicies,
            materials,
            accelerator,
            accelerator_id_to_tri,
            accelerator_id_to_material,
        }
    }
}
