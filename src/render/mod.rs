use crate::simulation::EnergyHistogram;

use fundsp::prelude32::*;
use rand::{RngExt, rngs::ThreadRng};

pub fn render(sample_rate: f32, mut energy_histograms: Vec<(f32, EnergyHistogram)>) -> Vec<f32> {
    energy_histograms.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let max_value = 1.0
        / energy_histograms
            .iter()
            .map(|(_, h)| {
                h.inner
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap()
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

    energy_histograms.iter_mut().for_each(|(_, histogram)| {
        histogram.scale(max_value);
        histogram.resample_linear(sample_rate);
    });

    let render_samples = energy_histograms
        .iter()
        .map(|(_, h)| h.inner.len())
        .min()
        .unwrap();

    if render_samples == 0 {
        return Vec::new();
    }

    let uniform =
        rand::distr::Uniform::new_inclusive(-1.0, 1.0).expect("Failed to create distribution");

    let mut result: Vec<f32> = vec![0.0; render_samples];

    let num_histograms = energy_histograms.len();

    for (index, (freq, histogram)) in energy_histograms.iter_mut().enumerate() {
        let mut filter: An<Unit<U1, U1>> = {
            if index == 0 {
                // println!("first filter");
                unit(Box::new(lowpass_hz(*freq, 1.414)))
            } else if index == (num_histograms - 1) {
                // println!("last filter");
                unit(Box::new(highpass_hz(*freq, 1.414)))
            } else {
                // println!("middle filter");
                unit(Box::new(bandpass_hz(*freq, 1.414)))
            }
        };
        filter.set_sample_rate(sample_rate as f64);
        filter.allocate();

        for i in 0..render_samples {
            let filtered_sample = filter.filter_mono(ThreadRng::default().sample(uniform));
            let envelope = histogram.inner[i];

            result[i] += filtered_sample * envelope;
        }
    }

    let peak = result.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 1.0 {
        result.iter_mut().for_each(|s| *s /= peak);
    }

    result
}
