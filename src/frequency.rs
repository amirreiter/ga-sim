use crate::material::SurfaceMaterial;

pub trait SimulationFrequency {
    fn hz() -> f32;

    fn air_alpha() -> f32;

    fn air_decay_multiplier(distance_m: f32) -> f32 {
        (-Self::air_alpha() * distance_m).exp()
    }

    fn ac(material: &SurfaceMaterial) -> f32;

    fn sc(material: &SurfaceMaterial) -> f32;
}

macro_rules! define_simulation_frequency {
    ($hz:literal) => {
        paste::paste! {
            #[allow(non_camel_case_types)]
            pub struct [<F_ $hz _HZ>](());

            impl SimulationFrequency for [<F_ $hz _HZ>] {
                #[inline(always)]
                fn hz() -> f32 {
                    $hz as f32
                }

                #[inline(always)]
                fn air_alpha() -> f32 {
                    #[cfg(not(test))]
                    {
                        crabtime::eval! {{
                            // ISO 9613-1:1993 atmospheric absorption.
                            //
                            // Returns an exponential AMPLITUDE attenuation coefficient [1/m],
                            // suitable for:
                            //
                            //     gain *= exp(-air_alpha * distance_m)
                            //
                            // A gain of 0.5 corresponds to approximately -6.02 dB.

                            let f: f32 = $hz as f32; // Hz
                            let temp: f32 = 20.0;    // °C
                            let hum: f32 = 0.5;      // relative humidity: 0.0 .. 1.0

                            const P_REF: f32 = 101.325; // kPa
                            const T_REF: f32 = 293.15;  // K
                            const T_TRIPLE: f32 = 273.16; // K

                            let pressure = 101.325; // kPa
                            let t = temp + 273.15;
                            let tr = t / T_REF;
                            let pr = pressure / P_REF;

                            // Saturation vapour pressure divided by reference pressure.
                            let psat_over_pref =
                                10.0f32.powf(
                                    -6.8346 * (T_TRIPLE / t).powf(1.261)
                                    + 4.6151
                                );

                            // Molar concentration of water vapour, in percent.
                            //
                            // hum is 0..1, so multiply by 100 to obtain RH percent.
                            let h =
                                hum * 100.0 * psat_over_pref / pr;

                            // Oxygen relaxation frequency.
                            let f_ro =
                                pr
                                * (
                                    24.0
                                    + 4.04e4
                                        * h
                                        * (0.02 + h)
                                        / (0.391 + h)
                                );

                            // Nitrogen relaxation frequency.
                            let f_rn =
                                pr
                                * tr.powf(-0.5)
                                * (
                                    9.0
                                    + 280.0
                                        * h
                                        * (
                                            -4.17
                                                * (
                                                    tr.powf(-1.0 / 3.0)
                                                    - 1.0
                                                )
                                        ).exp()
                                );

                            let f2 = f * f;

                            // Atmospheric attenuation in dB/m.
                            let alpha_db =
                                8.686
                                * f2
                                * (
                                    1.84e-11
                                        * pr.recip()
                                        * tr.sqrt()
                                    +
                                    tr.powf(-2.5)
                                        * (
                                            0.01275
                                                * (-2239.1 / t).exp()
                                                / (f_ro + f2 / f_ro)
                                            +
                                            0.1068
                                                * (-3352.0 / t).exp()
                                                / (f_rn + f2 / f_rn)
                                        )
                                );

                            // Convert dB/m to exponential amplitude attenuation [1/m].
                            //
                            // exp(-alpha * d)
                            // =
                            // 10^(-alpha_db * d / 20)
                            alpha_db * std::f32::consts::LN_10 / 20.0
                        }}
                    }

                    #[cfg(test)]
                    {
                        -1.0
                    }
                }

                #[inline(always)]
                fn ac(material: &SurfaceMaterial) -> f32 {
                    material.[<ac_ $hz hz>]
                }

                #[inline(always)]
                fn sc(material: &SurfaceMaterial) -> f32 {
                    material.[<sc_ $hz hz>]
                }
            }
        }
    };
}

define_simulation_frequency!(63);
define_simulation_frequency!(125);
define_simulation_frequency!(250);
define_simulation_frequency!(500);
define_simulation_frequency!(1000);
define_simulation_frequency!(2000);
define_simulation_frequency!(4000);
define_simulation_frequency!(8000);
define_simulation_frequency!(16000);

#[test]
fn frequency_crabtime() {
    // Ensure our crabtime macro is working correctly since rust-analyzer

    assert!(F_63_HZ::air_alpha() > 0.0);
    assert!(F_125_HZ::air_alpha() > 0.0);
    assert!(F_250_HZ::air_alpha() > 0.0);
    assert!(F_500_HZ::air_alpha() > 0.0);
    assert!(F_1000_HZ::air_alpha() > 0.0);
    assert!(F_2000_HZ::air_alpha() > 0.0);
    assert!(F_4000_HZ::air_alpha() > 0.0);
    assert!(F_8000_HZ::air_alpha() > 0.0);
    assert!(F_16000_HZ::air_alpha() > 0.0);
}
