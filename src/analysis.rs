use plotters::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f32::consts::PI;

use crate::simulation::EnergyHistogram;

const OCTAVE_CENTERS: [f32; 9] = [
    62.5, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0,
];

pub fn analyze_and_plot_energy_deviation(
    sim_pcm: &[f32],
    bench_pcm: &[f32],
    sample_rate: f32,
    window_size: usize,
    hop_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // --- 1. Peak Normalization ---
    let normalize = |data: &[f32]| -> Vec<f32> {
        let peak = data.iter().map(|&x| x.abs()).fold(0.0, f32::max);
        if peak > 0.0 {
            data.iter().map(|&x| x / peak).collect()
        } else {
            data.to_vec()
        }
    };

    let sim_norm = normalize(sim_pcm);
    let bench_norm = normalize(bench_pcm);

    let max_samples = sim_norm.len().max(bench_norm.len());
    if max_samples < window_size {
        return Err("Input data is shorter than window size".into());
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(window_size);

    let mut band_deviations: Vec<Vec<f32>> = vec![Vec::new(); OCTAVE_CENTERS.len()];
    let mut total_deviations: Vec<f32> = Vec::new();
    let mut time_axis: Vec<f32> = Vec::new();

    let window: Vec<f32> = (0..window_size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (window_size - 1) as f32).cos()))
        .collect();

    // --- 2. Processing Loop ---
    let mut start = 0;
    while start + window_size <= max_samples {
        time_axis.push(start as f32 / sample_rate);

        let mut sim_complex = vec![Complex { re: 0.0, im: 0.0 }; window_size];
        let mut bench_complex = vec![Complex { re: 0.0, im: 0.0 }; window_size];

        for i in 0..window_size {
            let idx = start + i;
            let w = window[i];
            if idx < sim_norm.len() {
                sim_complex[i].re = sim_norm[idx] * w;
            }
            if idx < bench_norm.len() {
                bench_complex[i].re = bench_norm[idx] * w;
            }
        }

        fft.process(&mut sim_complex);
        fft.process(&mut bench_complex);

        let mut sim_total_en = 0.0;
        let mut bench_total_en = 0.0;
        let mut sim_band_en = vec![0.0; OCTAVE_CENTERS.len()];
        let mut bench_band_en = vec![0.0; OCTAVE_CENTERS.len()];

        for bin in 0..(window_size / 2) {
            let freq = bin as f32 * sample_rate / window_size as f32;
            let s_p = sim_complex[bin].norm_sqr();
            let b_p = bench_complex[bin].norm_sqr();

            sim_total_en += s_p;
            bench_total_en += b_p;

            for (i, &center) in OCTAVE_CENTERS.iter().enumerate() {
                if freq >= center / 1.414 && freq < center * 1.414 {
                    sim_band_en[i] += s_p;
                    bench_band_en[i] += b_p;
                }
            }
        }

        // dB Deviation (Simulation - Benchmark)
        for i in 0..OCTAVE_CENTERS.len() {
            let dev =
                10.0 * (sim_band_en[i] + 1e-12).log10() - 10.0 * (bench_band_en[i] + 1e-12).log10();
            band_deviations[i].push(dev);
        }
        total_deviations
            .push(10.0 * (sim_total_en + 1e-12).log10() - 10.0 * (bench_total_en + 1e-12).log10());

        start += hop_size;
    }

    // --- 3. Dynamic Y-Axis Bounds ---
    let all_values: Vec<f32> = band_deviations
        .iter()
        .flatten()
        .chain(total_deviations.iter())
        .cloned()
        .collect();
    let mut min_y = all_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let mut max_y = all_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    // Force a minimum visual range if the simulation is near-perfect
    if (max_y - min_y).abs() < 1.0 {
        min_y = -1.0;
        max_y = 1.0;
    } else {
        min_y -= 1.0; // Margin
        max_y += 1.0;
    }

    // --- 4. Plotting ---
    let root =
        BitMapBackend::new("Simulation_Energy_Deviation.png", (1920, 1080)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Simulation Energy Deviation (Normalized)",
            ("serif", 30).into_font(),
        )
        .margin(50)
        .x_label_area_size(80)
        .y_label_area_size(100)
        .build_cartesian_2d(0f32..*time_axis.last().unwrap_or(&1.0), min_y..max_y)?;

    chart
        .configure_mesh()
        .x_desc("Time (seconds)")
        .y_desc("Deviation (dB)")
        .x_label_style(("serif", 24))
        .y_label_style(("serif", 24))
        .axis_desc_style(("serif", 30))
        .draw()?;

    let colors = [
        RGBColor(230, 50, 50),
        RGBColor(243, 114, 44),
        RGBColor(248, 150, 30),
        RGBColor(255, 208, 67),
        RGBColor(127, 201, 107),
        RGBColor(67, 170, 139),
        RGBColor(39, 125, 161),
        RGBColor(59, 73, 142),
        RGBColor(102, 65, 138),
    ];

    for (i, &center) in OCTAVE_CENTERS.iter().enumerate() {
        chart
            .draw_series(LineSeries::new(
                time_axis
                    .iter()
                    .cloned()
                    .zip(band_deviations[i].iter().cloned()),
                colors[i % colors.len()].stroke_width(2),
            ))?
            .label(format!("{} Hz", center))
            .legend({
                let col = colors[i % colors.len()].clone();
                move |(x, y)| PathElement::new(vec![(x, y), (x + 30, y)], col.stroke_width(2))
            });
    }

    chart
        .draw_series(LineSeries::new(
            time_axis
                .iter()
                .cloned()
                .zip(total_deviations.iter().cloned()),
            BLACK.stroke_width(5),
        ))?
        .label("Total Wideband")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 30, y)], BLACK.stroke_width(5)));

    chart
        .configure_series_labels()
        .label_font(("serif", 30))
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

pub fn analyze_and_plot_energy_deviation_from_histograms(
    sim_histograms: &[(f32, EnergyHistogram)],
    bench_pcm: &[f32],
    sample_rate: f32,
    window_size: usize,
    hop_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // --- 1. Validate inputs ---
    if sim_histograms.is_empty() {
        return Err("Sim energy histograms cannot be empty".into());
    }

    let num_sim_frames = sim_histograms[0].1.inner.len();

    // --- 2. Normalize bench PCM by peak amplitude ---
    let normalize = |data: &[f32]| -> Vec<f32> {
        let peak = data.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        if peak > 0.0 {
            data.iter().map(|&x| x / peak).collect()
        } else {
            data.to_vec()
        }
    };
    let bench_norm = normalize(bench_pcm);

    if bench_norm.len() < window_size {
        return Err("Bench PCM is shorter than window size".into());
    }

    // --- 3. Normalize sim histograms globally (single peak across all bands and frames),
    //        so inter-band amplitude relationships are preserved — matching how bench PCM
    //        is normalized by its single global peak before FFT analysis. ---
    let sim_global_peak = sim_histograms
        .iter()
        .flat_map(|(_, h)| h.inner.iter())
        .map(|&x| x.abs())
        .fold(0.0f32, f32::max);

    let sim_norm_hists: Vec<(f32, Vec<f32>)> = sim_histograms
        .iter()
        .map(|(freq, h)| {
            let normed = if sim_global_peak > 0.0 {
                h.inner.iter().map(|&x| x / sim_global_peak).collect()
            } else {
                h.inner.clone()
            };
            (*freq, normed)
        })
        .collect();

    // --- 4. FFT setup for bench ---
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(window_size);

    let window: Vec<f32> = (0..window_size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (window_size - 1) as f32).cos()))
        .collect();

    let mut band_deviations: Vec<Vec<f32>> = vec![Vec::new(); OCTAVE_CENTERS.len()];
    let mut total_deviations: Vec<f32> = Vec::new();
    let mut time_axis: Vec<f32> = Vec::new();

    // --- 5. Processing loop — driven by bench PCM windows ---
    let mut start = 0;
    let mut frame = 0;
    while start + window_size <= bench_norm.len() && frame < num_sim_frames {
        time_axis.push(start as f32 / sample_rate);

        // FFT bench window
        let mut bench_complex: Vec<Complex<f32>> = (0..window_size)
            .map(|i| Complex {
                re: bench_norm[start + i] * window[i],
                im: 0.0,
            })
            .collect();
        fft.process(&mut bench_complex);

        // Accumulate bench energy per band and total
        let mut bench_band_en = vec![0.0f32; OCTAVE_CENTERS.len()];
        let mut bench_total_en = 0.0f32;
        for bin in 0..(window_size / 2) {
            let freq = bin as f32 * sample_rate / window_size as f32;
            let power = bench_complex[bin].norm_sqr();
            bench_total_en += power;
            for (i, &center) in OCTAVE_CENTERS.iter().enumerate() {
                if freq >= center / 1.414 && freq < center * 1.414 {
                    bench_band_en[i] += power;
                }
            }
        }

        // Accumulate sim energy per band and total from histogram frame.
        //
        // Key fixes vs. original:
        //   • Frequency matching uses a ±5 % relative tolerance so high-frequency
        //     bands (kHz range) are found even when the stored freq label is slightly off.
        //   • sim_energy is the energy value directly from the (globally-normalized)
        //     histogram — it is NOT squared again, because it is already an energy
        //     quantity, not an amplitude.
        let mut sim_total_en = 0.0f32;
        for (band_idx, &center) in OCTAVE_CENTERS.iter().enumerate() {
            let sim_energy = sim_norm_hists
                .iter()
                .find(|(freq, _)| (*freq - center).abs() / center < 0.05)
                .and_then(|(_, hist)| hist.get(frame).copied())
                .unwrap_or(0.0);

            // sim_energy is already an energy (power) value; use it directly.
            sim_total_en += sim_energy;

            let dev = 10.0 * (sim_energy + 1e-12).log10()
                - 10.0 * (bench_band_en[band_idx] + 1e-12).log10();
            band_deviations[band_idx].push(dev);
        }

        total_deviations
            .push(10.0 * (sim_total_en + 1e-12).log10() - 10.0 * (bench_total_en + 1e-12).log10());

        start += hop_size;
        frame += 1;
    }

    // --- 6. Dynamic Y-Axis Bounds ---
    let all_values: Vec<f32> = band_deviations
        .iter()
        .flatten()
        .chain(total_deviations.iter())
        .cloned()
        .collect();

    let mut min_y = all_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let mut max_y = all_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    if (max_y - min_y).abs() < 1.0 {
        min_y = -1.0;
        max_y = 1.0;
    } else {
        min_y -= 1.0;
        max_y += 1.0;
    }

    // --- 7. Plotting ---
    let root = BitMapBackend::new("Simulation_Energy_Histogram_Deviation.png", (1920, 1080))
        .into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Simulation Energy Deviation (Normalized)",
            ("serif", 30).into_font(),
        )
        .margin(50)
        .x_label_area_size(80)
        .y_label_area_size(100)
        .build_cartesian_2d(0f32..*time_axis.last().unwrap_or(&1.0), min_y..max_y)?;

    chart
        .configure_mesh()
        .x_desc("Time (seconds)")
        .y_desc("Deviation (dB)")
        .x_label_style(("serif", 24))
        .y_label_style(("serif", 24))
        .axis_desc_style(("serif", 30))
        .draw()?;

    let colors = [
        RGBColor(230, 50, 50),
        RGBColor(243, 114, 44),
        RGBColor(248, 150, 30),
        RGBColor(255, 208, 67),
        RGBColor(127, 201, 107),
        RGBColor(67, 170, 139),
        RGBColor(39, 125, 161),
        RGBColor(59, 73, 142),
        RGBColor(102, 65, 138),
    ];

    for (i, &center) in OCTAVE_CENTERS.iter().enumerate() {
        chart
            .draw_series(LineSeries::new(
                time_axis
                    .iter()
                    .cloned()
                    .zip(band_deviations[i].iter().cloned()),
                colors[i % colors.len()].stroke_width(2),
            ))?
            .label(format!("{} Hz", center))
            .legend({
                let col = colors[i % colors.len()].clone();
                move |(x, y)| PathElement::new(vec![(x, y), (x + 30, y)], col.stroke_width(2))
            });
    }

    chart
        .draw_series(LineSeries::new(
            time_axis
                .iter()
                .cloned()
                .zip(total_deviations.iter().cloned()),
            BLACK.stroke_width(5),
        ))?
        .label("Total Wideband")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 30, y)], BLACK.stroke_width(5)));

    chart
        .configure_series_labels()
        .label_font(("serif", 30))
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}
