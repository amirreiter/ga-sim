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
                        Some(0..3) => None,
                        Some(i) => Some((i - 3) as u8),
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
                        let material_id: Option<usize> = match model.mesh.material_id {
                            Some(0..3) => None,
                            Some(i) => Some(i - 3),
                            None => None,
                        };

                        if material_id.is_none() {
                            return None;
                        }
                        // Materials are saved in a separate iterator.

                        let v0 = Vec3A::new(
                            positions[indicies[0] as usize + 0],
                            positions[indicies[0] as usize + 1],
                            positions[indicies[0] as usize + 2],
                        );
                        let v1 = Vec3A::new(
                            positions[indicies[1] as usize + 0],
                            positions[indicies[1] as usize + 1],
                            positions[indicies[1] as usize + 2],
                        );
                        let v2 = Vec3A::new(
                            positions[indicies[2] as usize + 0],
                            positions[indicies[2] as usize + 1],
                            positions[indicies[2] as usize + 2],
                        );

                        Some(RtCompressedTriangle::new(
                            unsafe { std::mem::transmute(v0) },
                            unsafe { std::mem::transmute(v1) },
                            unsafe { std::mem::transmute(v2) },
                        ))
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

        Scene {
            triangles,
            gpu_triangles,
            material_indicies,
            materials,
            accelerator,
        }
    }
}
