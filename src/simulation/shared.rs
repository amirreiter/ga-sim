#[derive(Debug)]
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

    pub fn add(&mut self, rhs: &Self) {
        if self.sample_rate == rhs.sample_rate {
            self.inner.iter_mut().zip(rhs.inner.iter()).for_each(|(s, rhs)| {
                *s += rhs;
            });
        } else {
            todo!()
        }
    }

    pub fn normalize(&mut self) {
        let peak = self.inner.iter().map(|&x| x.abs()).fold(0.0_f32, f32::max);

        if peak > 0.0 {
            let scale = 1.0 / peak;
            for x in self.inner.iter_mut() {
                *x *= scale;
            }
        }
    }

    pub fn scale(&mut self, scale: f32) {
        let peak = self.inner.iter().map(|&x| x.abs()).fold(0.0_f32, f32::max);

        if peak > 0.0 {
            for x in self.inner.iter_mut() {
                *x *= scale;
            }
        }
    }

    pub fn resample_linear(&mut self, target_sample_rate: f32) {
        if (self.sample_rate - target_sample_rate).abs() < f32::EPSILON || self.inner.is_empty() {
            return;
        }

        let ratio = self.sample_rate / target_sample_rate;
        let target_length =
            ((self.inner.len() as f32) * (target_sample_rate / self.sample_rate)).ceil() as usize;
        let mut output = Vec::with_capacity(target_length);

        for i in 0..target_length {
            let source_pos = i as f32 * ratio;
            let index_floor = source_pos as usize;
            let fract = source_pos - index_floor as f32;

            if index_floor + 1 < self.inner.len() {
                let sample_a = self.inner[index_floor];
                let sample_b = self.inner[index_floor + 1];

                let interpolated = sample_a + fract * (sample_b - sample_a);
                output.push(interpolated);
            } else {
                output.push(self.inner[self.inner.len() - 1]);
            }
        }

        self.inner = output;
        self.sample_rate = target_sample_rate;
    }
}
