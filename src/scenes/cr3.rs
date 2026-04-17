use std::path::PathBuf;

use crate::{material::SurfaceMaterial, scenes::Scene};

#[allow(dead_code)]
pub fn load_bras_cr3() -> Scene {
    fn root(ext: &str) -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push(PathBuf::from(
            "../../_TU_BERLIN_ACOUSTIC_BENCHES/1_scene_descriptions-CR3/_WORKING_DIR/",
        ));
        path.push(PathBuf::from(ext));
        path
    }

    // These are identified visually matching the Virtual Sketchup viewer in the browser
    // with the material colors in blender.

    // Materials 0, 1, and 2 are dedicated to an X,Y,Z axis placed in the center of the scene.
    // TODO: This is not the case anymore.

    // mat_CR3_plaster
    let mat3 = SurfaceMaterial::from_csv(&root("mat_CR3_plaster.csv"));

    // mat_CR3_stagePanels
    let mat4 = SurfaceMaterial::from_csv(&root("mat_CR3_stagePanels.csv"));

    // mat_CR3_structuredPlaster
    let mat5 = SurfaceMaterial::from_csv(&root("mat_CR3_structuredPlaster.csv"));

    // mat_CR3_floor
    let mat6 = SurfaceMaterial::from_csv(&root("mat_CR3_floor.csv"));

    // mat_CR3_ceiling
    let mat7 = SurfaceMaterial::from_csv(&root("mat_CR3_ceiling.csv"));

    // mat_CR3_windows
    let mat8 = SurfaceMaterial::from_csv(&root("mat_CR3_windows.csv"));

    // mat_CR3_seating
    let mat9 = SurfaceMaterial::from_csv(&root("mat_CR3_seating.csv"));

    let (obj, _materials) = tobj::load_obj(
        &root("CR3_BRIR.obj"),
        &tobj::LoadOptions {
            single_index: false,
            triangulate: true,
            ignore_points: true,
            ignore_lines: true,
        },
    )
    .expect("Failed to load CR3_BRIR model");

    let scene = Scene::from_obj(obj, vec![mat3, mat4, mat5, mat6, mat7, mat8, mat9]);

    scene.save_to_obj("/Users/amirreiter/Downloads/test.obj").unwrap();

    scene
}
