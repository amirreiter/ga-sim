use crate::material::SurfaceMaterial;

pub trait SimulationFrequency {
    fn hz() -> f32;

    fn air_alpha() -> f32;

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
                    // rust-analyzer does not handle this kind of macro stuff well,
                    // but we must satisfy the skeleton of the function so as a workaround
                    // we make rust_analyzer think the frequency alpha value is always -1.0
                    //
                    // The test uses this to check if crabtime is working or not.
                    #[cfg(not(rust_analyzer))]
                    {
                        // This is from ISO 9613-1:1993
                        crabtime::eval! {{
                            // primary inputs
                            let f: f32 = $hz as f32; // hz
                            let temp: f32 = 20.0; // celsius
                            let hum: f32 = 0.5; // percentage

                            let pr = 101.325;
                            let t_0 = 293.15;
                            let pa = 101.325;
                            let t = temp + 273.15;
                            let tr_ratio = t / t_0;
                            let psat_pr = 10.0f32.powf(-6.8346 * (273.16 / t).powf(1.261) + 4.6151);
                            let h = hum * 100.0 * psat_pr / (pa / pr);
                            let f_ro = (pa / pr) * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h));
                            let f_rn = (pa / pr)
                                * tr_ratio.sqrt().recip()
                                * (9.0 + 280.0 * h * (-4.17 * (tr_ratio.powf(-1.0 / 3.0) - 1.0)).exp());
                            let alpha_db = 8.686
                                * f.powi(2)
                                * ((1.84e-11 * (pa / pr).recip() * tr_ratio.sqrt())
                                    + tr_ratio.powf(-2.5)
                                        * ((0.01278 * (-2239.1 / t).exp()) / (f_ro + (f.powi(2) / f_ro))
                                            + (0.1068 * (-3352.0 / t).exp()) / (f_rn + (f.powi(2) / f_rn))));

                            // Convert dB energy loss to relative energy per meter
                            10.0f32.powf(-alpha_db / 20.0f32)
                        }}
                    }

                    #[cfg(rust_analyzer)]
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
