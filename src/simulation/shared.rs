pub struct EnergyHistogram {
    pub inner: Vec<f32>,
    pub sample_rate: f32,
}

impl EnergyHistogram {
    pub fn new(bin_count: usize, sample_rate: f32) -> Self {
        Self {
            inner: vec![0.0_f32; bin_count],
            sample_rate,
        }
    }
}
