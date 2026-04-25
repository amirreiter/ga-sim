use crate::simulation::EnergyHistogram;

use rand::{Rng, RngExt};
use rustfft::{Fft, FftPlanner, num_complex::Complex32};
use std::sync::Arc;

const MU_MAX_HZ: f64 = 10_000.0;
const CROSSOVER_OVERLAP: f32 = 1.0;
const CROSSOVER_STEEPNESS: usize = 0;

fn constant_mean_event_occurrence(speed_of_sound: f64, room_volume: f64) -> f64 {
    4.0 * std::f64::consts::PI * speed_of_sound.powi(3) / room_volume
}

fn mean_event_occurrence(constant: f64, t: f64) -> f64 {
    (constant * t.powi(2)).min(MU_MAX_HZ)
}

fn first_event_time(constant: f64) -> f64 {
    (2.0 * std::f64::consts::LN_2 / constant).powf(1.0 / 3.0)
}

fn generate_dirac_sequence(
    sample_rate: f32,
    max_time: f32,
    speed_of_sound: f32,
    room_volume: f32,
) -> Vec<f32> {
    if sample_rate <= 0.0 || max_time <= 0.0 || speed_of_sound <= 0.0 || room_volume <= 0.0 {
        return Vec::new();
    }

    let sequence_len = (max_time * sample_rate).ceil() as usize;
    let mut sequence = vec![0.0f32; sequence_len];

    let constant = constant_mean_event_occurrence(speed_of_sound as f64, room_volume as f64);
    let mut t = first_event_time(constant);
    let mut rng = rand::rng();

    while t < max_time as f64 {
        let sample_pos = t * sample_rate as f64;
        let sample_index = sample_pos as usize;

        if sample_index >= sequence.len() {
            break;
        }

        let twice = (2.0 * sample_pos) as usize;
        let negative = (twice % 2) != 0;
        if sequence[sample_index] == 0.0 {
            sequence[sample_index] = if negative { -1.0 } else { 1.0 };
        }

        let mu = mean_event_occurrence(constant, t);
        let z: f64 = rng.random_range(f64::MIN_POSITIVE..=1.0);
        let delta_t = (1.0 / z).ln() / mu;
        t += delta_t;
    }

    sequence
}

fn crossover_width_factor(lowest_hz: f32, highest_hz: f32, band_count: usize) -> f32 {
    if lowest_hz <= 0.0 || highest_hz <= lowest_hz || band_count == 0 {
        return 0.0;
    }

    let x = (highest_hz / lowest_hz).powf(1.0 / band_count as f32);
    if x <= 1.0 {
        0.0
    } else {
        (x - 1.0) / (x + 1.0)
    }
}

fn crossover_phase(p: f32, width_hz: f32, steepness: usize) -> f32 {
    if width_hz <= 0.0 {
        return 0.0;
    }

    let mut phase = (0.5 * (p / width_hz + 1.0)).clamp(0.0, 1.0);
    for _ in 0..steepness {
        phase = (std::f32::consts::FRAC_PI_2 * phase).sin();
    }

    phase
}

fn build_filter_bank_gains(
    sample_rate: f32,
    render_samples: usize,
    band_frequencies: &[f32],
    overlap: f32,
    steepness: usize,
) -> Vec<Vec<f32>> {
    if render_samples == 0 || band_frequencies.is_empty() {
        return Vec::new();
    }

    let half_bin_count = render_samples / 2 + 1;
    let band_count = band_frequencies.len();

    if band_count == 1 {
        return vec![vec![1.0; half_bin_count]];
    }

    let nyquist = 0.5 * sample_rate;
    if nyquist <= 0.0 {
        return vec![vec![0.0; half_bin_count]; band_count];
    }

    let lowest_hz = band_frequencies[0].max(f32::MIN_POSITIVE);
    let highest_hz = band_frequencies[band_count - 1].max(lowest_hz);
    let max_width_factor = crossover_width_factor(lowest_hz, highest_hz, band_count);

    let edge_frequencies: Vec<f32> = band_frequencies
        .windows(2)
        .map(|pair| (pair[0].max(f32::MIN_POSITIVE) * pair[1].max(f32::MIN_POSITIVE)).sqrt())
        .collect();

    let overlap = overlap.clamp(0.0, 1.0);
    let edge_widths: Vec<f32> = edge_frequencies
        .iter()
        .map(|edge| edge * overlap * max_width_factor)
        .collect();

    let mut gains = vec![vec![0.0f32; half_bin_count]; band_count];

    for bin in 0..half_bin_count {
        let freq_hz = bin as f32 * sample_rate / render_samples as f32;
        let mut assigned = false;

        for edge_idx in 0..edge_frequencies.len() {
            let edge_hz = edge_frequencies[edge_idx];
            let width_hz = edge_widths[edge_idx];
            let low_hz = (edge_hz - width_hz).max(0.0);
            let high_hz = (edge_hz + width_hz).min(nyquist);

            if width_hz <= f32::EPSILON || high_hz <= low_hz {
                if freq_hz < edge_hz {
                    gains[edge_idx][bin] = 1.0;
                } else if edge_idx == edge_frequencies.len() - 1 {
                    gains[edge_idx + 1][bin] = 1.0;
                }
                assigned = true;
                break;
            }

            if freq_hz < low_hz {
                gains[edge_idx][bin] = 1.0;
                assigned = true;
                break;
            }

            if freq_hz <= high_hz {
                let p = freq_hz - edge_hz;
                let phase = crossover_phase(p, width_hz, steepness);

                // Complementary low/high gains preserve reconstruction when adjacent bands are equal.
                let theta = std::f32::consts::FRAC_PI_2 * phase;
                gains[edge_idx][bin] = theta.cos().powi(2);
                gains[edge_idx + 1][bin] = theta.sin().powi(2);
                assigned = true;
                break;
            }
        }

        if !assigned {
            gains[band_count - 1][bin] = 1.0;
        }
    }

    gains
}

fn apply_zero_phase_filter(
    signal: &[f32],
    gains_half: &[f32],
    fft: &Arc<dyn Fft<f32>>,
    ifft: &Arc<dyn Fft<f32>>,
) -> Vec<f32> {
    if signal.is_empty() {
        return Vec::new();
    }

    let n = signal.len();
    let expected_half_bins = n / 2 + 1;
    if gains_half.len() != expected_half_bins {
        return signal.to_vec();
    }

    let mut spectrum: Vec<Complex32> = signal.iter().map(|&s| Complex32::new(s, 0.0)).collect();

    fft.process(&mut spectrum);

    for k in 0..expected_half_bins {
        let gain = gains_half[k];
        spectrum[k] *= gain;

        if k == 0 {
            continue;
        }

        if n % 2 == 0 && k == (n / 2) {
            continue;
        }

        let mirror_k = n - k;
        spectrum[mirror_k] *= gain;
    }

    ifft.process(&mut spectrum);

    let scale = 1.0 / n as f32;
    spectrum.into_iter().map(|c| c.re * scale).collect()
}

fn convert_index(hist_index: usize, out_sample_rate: f32, hist_sample_rate: f32) -> usize {
    ((hist_index as f64) * (out_sample_rate as f64) / (hist_sample_rate as f64)) as usize
}

fn weight_sequence_for_band(
    histogram: &EnergyHistogram,
    poisson_sequence: &[f32],
    out_sample_rate: f32,
    acoustic_impedance: f32,
) -> Vec<f32> {
    let ideal_len = convert_index(
        histogram.inner.len(),
        out_sample_rate,
        histogram.sample_rate,
    );
    let out_len = core::cmp::min(ideal_len, poisson_sequence.len());
    let mut weighted = poisson_sequence[..out_len].to_vec();

    for i in 0..histogram.inner.len() {
        let beg = core::cmp::min(
            convert_index(i, out_sample_rate, histogram.sample_rate),
            weighted.len(),
        );
        let end = core::cmp::min(
            convert_index(i + 1, out_sample_rate, histogram.sample_rate),
            weighted.len(),
        );

        if end <= beg {
            continue;
        }

        let squared_sum: f32 = poisson_sequence[beg..end].iter().map(|x| x * x).sum();
        if squared_sum <= 0.0 {
            for sample in &mut weighted[beg..end] {
                *sample = 0.0;
            }
            continue;
        }

        let energy = histogram.inner[i].max(0.0);
        let intensity = energy / squared_sum;
        let pressure = (intensity * acoustic_impedance).sqrt();

        for sample in &mut weighted[beg..end] {
            *sample *= pressure;
        }
    }

    weighted
}

pub fn render(
    sample_rate: f32,
    mut energy_histograms: Vec<(f32, EnergyHistogram)>,
    room_volume: f32,
    speed_of_sound: f32,
) -> Vec<f32> {
    if energy_histograms.is_empty() || sample_rate <= 0.0 {
        return Vec::new();
    }

    energy_histograms.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let histogram_seconds = energy_histograms
        .iter()
        .map(|(_, h)| h.inner.len() as f32 / h.sample_rate)
        .reduce(f32::min)
        .unwrap_or(0.0);

    if histogram_seconds <= 0.0 {
        return Vec::new();
    }

    let dirac_sequence = generate_dirac_sequence(
        sample_rate,
        histogram_seconds,
        speed_of_sound,
        room_volume,
    );

    let render_samples = core::cmp::min(
        energy_histograms
            .iter()
            .map(|(_, h)| convert_index(h.inner.len(), sample_rate, h.sample_rate))
            .min()
            .unwrap_or(0),
        dirac_sequence.len(),
    );

    if render_samples == 0 {
        return Vec::new();
    }

    let mut result: Vec<f32> = vec![0.0; render_samples];
    let band_frequencies: Vec<f32> = energy_histograms.iter().map(|(f, _)| *f).collect();
    let gains = build_filter_bank_gains(
        sample_rate,
        render_samples,
        &band_frequencies,
        CROSSOVER_OVERLAP,
        CROSSOVER_STEEPNESS,
    );
    if gains.len() != energy_histograms.len() {
        return Vec::new();
    }

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(render_samples);
    let ifft = planner.plan_fft_inverse(render_samples);

    for (index, (freq, histogram)) in energy_histograms.iter().enumerate() {
        let weighted = weight_sequence_for_band(
            histogram,
            &dirac_sequence,
            sample_rate,
            400.0
        );

        let mut band_signal = vec![0.0f32; render_samples];
        let copy_len = core::cmp::min(render_samples, weighted.len());
        band_signal[..copy_len].copy_from_slice(&weighted[..copy_len]);

        let filtered = apply_zero_phase_filter(&band_signal, &gains[index], &fft, &ifft);

        for i in 0..render_samples {
            result[i] += filtered[i];
        }
    }

    let peak = result.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 1.0 {
        result.iter_mut().for_each(|s| *s /= peak);
    }

    result
}
