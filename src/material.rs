use std::path::Path;

#[derive(Clone)]
pub struct SurfaceMaterial {
    pub ac_63hz: f32,
    pub sc_63hz: f32,

    pub ac_125hz: f32,
    pub sc_125hz: f32,

    pub ac_250hz: f32,
    pub sc_250hz: f32,

    pub ac_500hz: f32,
    pub sc_500hz: f32,

    pub ac_1000hz: f32,
    pub sc_1000hz: f32,

    pub ac_2000hz: f32,
    pub sc_2000hz: f32,

    pub ac_4000hz: f32,
    pub sc_4000hz: f32,

    pub ac_8000hz: f32,
    pub sc_8000hz: f32,

    pub ac_16000hz: f32,
    pub sc_16000hz: f32,
}

impl SurfaceMaterial {
    pub fn from_csv(path: &Path) -> Self {
        let mut rdr = csv::Reader::from_path(path).expect("Could not read surface material file");
        let mut records = rdr.records();

        // CSV reader already excludes header when reading records

        // Read absorption coefficients row
        let ac_row = records
            .next()
            .expect("Missing absorption coefficients row")
            .expect("Error reading absorption coefficients");

        // Read scattering coefficients row
        let sc_row = records
            .next()
            .expect("Missing scattering coefficients row")
            .expect("Error reading scattering coefficients");

        // Parse the values from columns
        let ac_63hz: f32 = ac_row[5].trim().parse().unwrap();
        let sc_63hz: f32 = sc_row[5].trim().parse().unwrap();

        let ac_125hz: f32 = ac_row[8].trim().parse().unwrap();
        let sc_125hz: f32 = sc_row[8].trim().parse().unwrap();

        let ac_250hz: f32 = ac_row[11].trim().parse().unwrap();
        let sc_250hz: f32 = sc_row[11].trim().parse().unwrap();

        let ac_500hz: f32 = ac_row[14].trim().parse().unwrap();
        let sc_500hz: f32 = sc_row[14].trim().parse().unwrap();

        let ac_1000hz: f32 = ac_row[17].trim().parse().unwrap();
        let sc_1000hz: f32 = sc_row[17].trim().parse().unwrap();

        let ac_2000hz: f32 = ac_row[20].trim().parse().unwrap();
        let sc_2000hz: f32 = sc_row[20].trim().parse().unwrap();

        let ac_4000hz: f32 = ac_row[23].trim().parse().unwrap();
        let sc_4000hz: f32 = sc_row[23].trim().parse().unwrap();

        let ac_8000hz: f32 = ac_row[26].trim().parse().unwrap();
        let sc_8000hz: f32 = sc_row[26].trim().parse().unwrap();

        let ac_16000hz: f32 = ac_row[29].trim().parse().unwrap();
        let sc_16000hz: f32 = sc_row[29].trim().parse().unwrap();

        SurfaceMaterial {
            ac_63hz,
            sc_63hz,
            ac_125hz,
            sc_125hz,
            ac_250hz,
            sc_250hz,
            ac_500hz,
            sc_500hz,
            ac_1000hz,
            sc_1000hz,
            ac_2000hz,
            sc_2000hz,
            ac_4000hz,
            sc_4000hz,
            ac_8000hz,
            sc_8000hz,
            ac_16000hz,
            sc_16000hz,
        }
    }
}
