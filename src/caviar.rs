// Generate Impulse Responses for Stanford CCRMA's CAVIAR system in the
// downstairs listening room (as of 2026).

use std::env;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

// First-order AmbiX / ACN / SN3D input channel order:
//   0 = W (ACN 0)
//   1 = Y (ACN 1)
//   2 = Z (ACN 2)
//   3 = X (ACN 3)
// Coordinate convention:
//   +X = Front, +Y = Left, +Z = Up
//
// The 14 loudspeaker directions are channels 1..=14 from the supplied
// CCRMA layout. Set A uses the directions exactly as listed. Set B applies
// +1 degree roll, +1 degree pitch, +1 degree yaw to every direction, using
// the fixed-axis composition Rz(yaw) * Ry(pitch) * Rx(roll).
//
// Decoder construction:
//   For each set, form the first-order spherical-harmonic sampling matrix Y
//   whose row for speaker i is [1, y_i, z_i, x_i]. Then build the
//   mode-matching / least-norm decoder
//
//       D = Y * inverse(Y^T * Y)
//
//   so each speaker output is s_i = D_i * [W, Y, Z, X]^T.
//
// Usage:
//   rustc -O ambisonic_decoder.rs -o ambisonic_decoder
//   ./ambisonic_decoder AmbiX_B_ACN_SN3D.wav output_directory
//
// Output:
//   Amir_SetA_Speaker_1.wav  ... Amir_SetA_Speaker_14.wav
//   Amir_SetB_Speaker_1.wav  ... Amir_SetB_Speaker_14.wav
//
// This is intentionally dependency-free and accepts 4-channel, 32-bit
// floating-point WAV input (including WAVE_FORMAT_EXTENSIBLE float WAV).

const NUM_SPEAKERS: usize = 14;
const NUM_AMBI_CHANNELS: usize = 4;
const OFF_AXIS_DEGREES: f64 = 1.0;

#[derive(Clone, Copy, Debug)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    fn normalized(self) -> Self {
        let len = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        assert!(len > 0.0);
        Self::new(self.x / len, self.y / len, self.z / len)
    }
}

#[derive(Clone, Copy, Debug)]
struct SpeakerDirection {
    azimuth_deg: f64,
    elevation_deg: f64,
}

// Channels 1..=14, in loudspeaker playback order.
const SPEAKERS: [SpeakerDirection; NUM_SPEAKERS] = [
    SpeakerDirection {
        azimuth_deg: 22.5,
        elevation_deg: 0.0,
    }, // 1
    SpeakerDirection {
        azimuth_deg: -22.5,
        elevation_deg: 0.0,
    }, // 2
    SpeakerDirection {
        azimuth_deg: 67.5,
        elevation_deg: 0.0,
    }, // 3
    SpeakerDirection {
        azimuth_deg: -67.5,
        elevation_deg: 0.0,
    }, // 4
    SpeakerDirection {
        azimuth_deg: 112.5,
        elevation_deg: 0.0,
    }, // 5
    SpeakerDirection {
        azimuth_deg: -112.5,
        elevation_deg: 0.0,
    }, // 6
    SpeakerDirection {
        azimuth_deg: 157.5,
        elevation_deg: 0.0,
    }, // 7
    SpeakerDirection {
        azimuth_deg: -157.5,
        elevation_deg: 0.0,
    }, // 8
    SpeakerDirection {
        azimuth_deg: 30.0,
        elevation_deg: 40.0,
    }, // 9
    SpeakerDirection {
        azimuth_deg: -30.0,
        elevation_deg: 40.0,
    }, // 10
    SpeakerDirection {
        azimuth_deg: 90.0,
        elevation_deg: 40.0,
    }, // 11
    SpeakerDirection {
        azimuth_deg: -90.0,
        elevation_deg: 40.0,
    }, // 12
    SpeakerDirection {
        azimuth_deg: 150.0,
        elevation_deg: 40.0,
    }, // 13
    SpeakerDirection {
        azimuth_deg: -150.0,
        elevation_deg: 40.0,
    }, // 14
];

fn speaker_to_cartesian(s: SpeakerDirection) -> Vec3 {
    let az = s.azimuth_deg.to_radians();
    let el = s.elevation_deg.to_radians();
    let cos_el = el.cos();

    // Ambisonic axes used by the encoder:
    // x = front/back, y = left/right, z = up/down.
    Vec3::new(cos_el * az.cos(), cos_el * az.sin(), el.sin()).normalized()
}

fn rotate_roll_pitch_yaw(v: Vec3, roll_deg: f64, pitch_deg: f64, yaw_deg: f64) -> Vec3 {
    let r = roll_deg.to_radians();
    let p = pitch_deg.to_radians();
    let y = yaw_deg.to_radians();

    // Roll: +X axis.
    let (sr, cr) = r.sin_cos();
    let v1 = Vec3::new(v.x, cr * v.y - sr * v.z, sr * v.y + cr * v.z);

    // Pitch: +Y axis.
    let (sp, cp) = p.sin_cos();
    let v2 = Vec3::new(cp * v1.x + sp * v1.z, v1.y, -sp * v1.x + cp * v1.z);

    // Yaw: +Z axis.
    let (sy, cy) = y.sin_cos();
    Vec3::new(cy * v2.x - sy * v2.y, sy * v2.x + cy * v2.y, v2.z).normalized()
}

fn harmonic_row(v: Vec3) -> [f64; NUM_AMBI_CHANNELS] {
    // AmbiX ACN/SN3D order expected from the user's encoder: [W, Y, Z, X].
    [1.0, v.y, v.z, v.x]
}

fn invert_4x4(mut a: [[f64; 4]; 4]) -> Result<[[f64; 4]; 4], String> {
    let mut inv = [[0.0f64; 4]; 4];
    for i in 0..4 {
        inv[i][i] = 1.0;
    }

    for col in 0..4 {
        let mut pivot_row = col;
        let mut pivot_abs = a[col][col].abs();
        for row in (col + 1)..4 {
            let candidate = a[row][col].abs();
            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }

        if pivot_abs < 1.0e-12 {
            return Err("decoder matrix is singular or numerically unstable".to_string());
        }

        if pivot_row != col {
            a.swap(col, pivot_row);
            inv.swap(col, pivot_row);
        }

        let pivot = a[col][col];
        for j in 0..4 {
            a[col][j] /= pivot;
            inv[col][j] /= pivot;
        }

        for row in 0..4 {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            if factor == 0.0 {
                continue;
            }
            for j in 0..4 {
                a[row][j] -= factor * a[col][j];
                inv[row][j] -= factor * inv[col][j];
            }
        }
    }

    Ok(inv)
}

fn build_decoder(directions: &[Vec3; NUM_SPEAKERS]) -> Result<[[f64; 4]; NUM_SPEAKERS], String> {
    let mut y = [[0.0f64; 4]; NUM_SPEAKERS];
    for i in 0..NUM_SPEAKERS {
        y[i] = harmonic_row(directions[i]);
    }

    // gram = Y^T * Y
    let mut gram = [[0.0f64; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            let mut sum = 0.0;
            for i in 0..NUM_SPEAKERS {
                sum += y[i][r] * y[i][c];
            }
            gram[r][c] = sum;
        }
    }

    let inv_gram = invert_4x4(gram)?;

    // D = Y * inverse(Y^T * Y)
    let mut d = [[0.0f64; 4]; NUM_SPEAKERS];
    for i in 0..NUM_SPEAKERS {
        for c in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += y[i][k] * inv_gram[k][c];
            }
            d[i][c] = sum;
        }
    }

    Ok(d)
}

#[derive(Debug)]
struct Wav4F32 {
    sample_rate: u32,
    frames: Vec<[f32; 4]>,
}

fn read_u16_le(bytes: &[u8], offset: usize) -> io::Result<u16> {
    if offset + 2 > bytes.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "unexpected end of WAV",
        ));
    }
    Ok(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]))
}

fn read_u32_le(bytes: &[u8], offset: usize) -> io::Result<u32> {
    if offset + 4 > bytes.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "unexpected end of WAV",
        ));
    }
    Ok(u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ]))
}

fn read_wav_4ch_f32(path: &Path) -> Result<Wav4F32, String> {
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(|e| format!("failed to open {}: {e}", path.display()))?
        .read_to_end(&mut bytes)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;

    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("input is not a RIFF/WAVE file".to_string());
    }

    let mut fmt: Option<Vec<u8>> = None;
    let mut data: Option<Vec<u8>> = None;
    let mut pos = 12usize;

    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = read_u32_le(&bytes, pos + 4).map_err(|e| e.to_string())? as usize;
        let start = pos + 8;
        let end = start
            .checked_add(size)
            .ok_or_else(|| "WAV chunk size overflow".to_string())?;
        if end > bytes.len() {
            return Err("truncated WAV chunk".to_string());
        }

        if id == b"fmt " {
            fmt = Some(bytes[start..end].to_vec());
        } else if id == b"data" {
            data = Some(bytes[start..end].to_vec());
        }

        pos = end + (size & 1); // RIFF chunks are word-aligned.
    }

    let fmt = fmt.ok_or_else(|| "WAV has no fmt chunk".to_string())?;
    let data = data.ok_or_else(|| "WAV has no data chunk".to_string())?;

    if fmt.len() < 16 {
        return Err("WAV fmt chunk is too short".to_string());
    }

    let format_tag = read_u16_le(&fmt, 0).map_err(|e| e.to_string())?;
    let channels = read_u16_le(&fmt, 2).map_err(|e| e.to_string())?;
    let sample_rate = read_u32_le(&fmt, 4).map_err(|e| e.to_string())?;
    let block_align = read_u16_le(&fmt, 12).map_err(|e| e.to_string())?;
    let bits_per_sample = read_u16_le(&fmt, 14).map_err(|e| e.to_string())?;

    if channels != 4 {
        return Err(format!(
            "expected exactly 4 input channels, found {channels}"
        ));
    }
    if bits_per_sample != 32 {
        return Err(format!(
            "expected 32-bit floating-point WAV, found {bits_per_sample} bits/sample"
        ));
    }
    if block_align != 16 {
        return Err(format!(
            "expected 16-byte frames for 4 x f32 channels, found block_align={block_align}"
        ));
    }

    let is_ieee_float = if format_tag == 3 {
        true
    } else if format_tag == 0xFFFE {
        // WAVE_FORMAT_EXTENSIBLE. SubFormat GUID starts at byte 24; its first
        // 32-bit value is 0x00000003 for KSDATAFORMAT_SUBTYPE_IEEE_FLOAT.
        if fmt.len() < 40 {
            return Err("WAVE_FORMAT_EXTENSIBLE fmt chunk is too short".to_string());
        }
        read_u32_le(&fmt, 24).map_err(|e| e.to_string())? == 3
    } else {
        false
    };

    if !is_ieee_float {
        return Err(format!(
            "expected IEEE float WAV (format 3 or extensible float), found format tag 0x{format_tag:04x}"
        ));
    }

    if data.len() % 16 != 0 {
        return Err("WAV data size is not an integer number of 4-channel f32 frames".to_string());
    }

    let frame_count = data.len() / 16;
    let mut frames = Vec::with_capacity(frame_count);
    for frame in data.chunks_exact(16) {
        let mut out = [0.0f32; 4];
        for ch in 0..4 {
            let o = ch * 4;
            out[ch] = f32::from_le_bytes([frame[o], frame[o + 1], frame[o + 2], frame[o + 3]]);
        }
        frames.push(out);
    }

    Ok(Wav4F32 {
        sample_rate,
        frames,
    })
}

fn write_mono_f32_wav(path: &Path, sample_rate: u32, samples: &[f32]) -> Result<(), String> {
    let data_bytes_u64 = (samples.len() as u64)
        .checked_mul(4)
        .ok_or_else(|| "output WAV is too large".to_string())?;
    if data_bytes_u64 > u32::MAX as u64 {
        return Err(format!(
            "{} exceeds classic RIFF/WAV size limits",
            path.display()
        ));
    }
    let data_bytes = data_bytes_u64 as u32;
    let riff_size = 36u32
        .checked_add(data_bytes)
        .ok_or_else(|| "output WAV RIFF size overflow".to_string())?;

    let mut f =
        File::create(path).map_err(|e| format!("failed to create {}: {e}", path.display()))?;

    f.write_all(b"RIFF").map_err(|e| e.to_string())?;
    f.write_all(&riff_size.to_le_bytes())
        .map_err(|e| e.to_string())?;
    f.write_all(b"WAVE").map_err(|e| e.to_string())?;

    f.write_all(b"fmt ").map_err(|e| e.to_string())?;
    f.write_all(&16u32.to_le_bytes())
        .map_err(|e| e.to_string())?;
    f.write_all(&3u16.to_le_bytes())
        .map_err(|e| e.to_string())?; // IEEE float
    f.write_all(&1u16.to_le_bytes())
        .map_err(|e| e.to_string())?; // mono
    f.write_all(&sample_rate.to_le_bytes())
        .map_err(|e| e.to_string())?;
    let byte_rate = sample_rate
        .checked_mul(4)
        .ok_or_else(|| "sample rate is too large".to_string())?;
    f.write_all(&byte_rate.to_le_bytes())
        .map_err(|e| e.to_string())?;
    f.write_all(&4u16.to_le_bytes())
        .map_err(|e| e.to_string())?; // block align
    f.write_all(&32u16.to_le_bytes())
        .map_err(|e| e.to_string())?;

    f.write_all(b"data").map_err(|e| e.to_string())?;
    f.write_all(&data_bytes.to_le_bytes())
        .map_err(|e| e.to_string())?;
    for &sample in samples {
        f.write_all(&sample.to_le_bytes())
            .map_err(|e| e.to_string())?;
    }

    Ok(())
}

fn decode_set(input: &Wav4F32, decoder: &[[f64; 4]; NUM_SPEAKERS]) -> [Vec<f32>; NUM_SPEAKERS] {
    let mut outputs: [Vec<f32>; NUM_SPEAKERS] =
        std::array::from_fn(|_| Vec::with_capacity(input.frames.len()));

    for frame in &input.frames {
        // Input order is exactly [W, Y, Z, X], matching the encoder.
        let b = [
            frame[0] as f64,
            frame[1] as f64,
            frame[2] as f64,
            frame[3] as f64,
        ];

        for speaker in 0..NUM_SPEAKERS {
            let row = decoder[speaker];
            let sample = row[0] * b[0] + row[1] * b[1] + row[2] * b[2] + row[3] * b[3];
            outputs[speaker].push(sample as f32);
        }
    }

    outputs
}

fn write_set(
    output_dir: &Path,
    set_name: &str,
    sample_rate: u32,
    outputs: &[Vec<f32>; NUM_SPEAKERS],
) -> Result<(), String> {
    for speaker_index in 0..NUM_SPEAKERS {
        let file_name = format!("Amir_Set{}_Speaker_{}.wav", set_name, speaker_index + 1);
        let path = output_dir.join(file_name);
        write_mono_f32_wav(&path, sample_rate, &outputs[speaker_index])?;
    }
    Ok(())
}

fn print_decoder(name: &str, d: &[[f64; 4]; NUM_SPEAKERS]) {
    println!("{name} decoder rows [W, Y, Z, X]:");
    for (i, row) in d.iter().enumerate() {
        println!(
            "  Speaker {:2}: [{:+.9}, {:+.9}, {:+.9}, {:+.9}]",
            i + 1,
            row[0],
            row[1],
            row[2],
            row[3]
        );
    }
}

pub fn ambisonic_b_to_caviar_14x2(input_path: PathBuf, output_dir: PathBuf) -> Result<(), String> {
    let input = read_wav_4ch_f32(&input_path)?;

    let set_a_directions: [Vec3; NUM_SPEAKERS] =
        std::array::from_fn(|i| speaker_to_cartesian(SPEAKERS[i]));

    let set_b_directions: [Vec3; NUM_SPEAKERS] =
        std::array::from_fn(|i| rotate_roll_pitch_yaw(set_a_directions[i], 7.0, 2.0, 2.0));

    let decoder_a = build_decoder(&set_a_directions)?;
    let decoder_b = build_decoder(&set_b_directions)?;

    print_decoder("Set A (on-axis)", &decoder_a);
    print_decoder("Set B (+1 deg roll/pitch/yaw)", &decoder_b);

    let outputs_a = decode_set(&input, &decoder_a);
    let outputs_b = decode_set(&input, &decoder_b);

    fs::create_dir_all(&output_dir)
        .map_err(|e| format!("failed to create {}: {e}", output_dir.display()))?;

    write_set(&output_dir, "A", input.sample_rate, &outputs_a)?;
    write_set(&output_dir, "B", input.sample_rate, &outputs_b)?;

    println!();
    println!(
        "Decoded {} input frames at {} Hz.",
        input.frames.len(),
        input.sample_rate
    );
    println!("Wrote 28 mono IR WAV files to {}.", output_dir.display());
    println!("Speaker numbering is preserved exactly as 1..14 in both sets.");

    Ok(())
}

fn main() {
    let mut args = env::args_os();
    let exe = args.next().unwrap_or_default();
    let input = args.next();
    let output = args.next();

    if input.is_none() || output.is_none() || args.next().is_some() {
        eprintln!(
            "Usage: {} <4ch_AmbiX_ACN_SN3D_float32.wav> <output_directory>",
            Path::new(&exe)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("ambisonic_decoder")
        );
        std::process::exit(2);
    }

    if let Err(e) = ambisonic_b_to_caviar_14x2(
        PathBuf::from(input.unwrap()),
        PathBuf::from(output.unwrap()),
    ) {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
