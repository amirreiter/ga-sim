// mod rt_specular;
mod shared;
mod cpu;
// mod trace;
// mod rt_diffuse;

pub use shared::EnergyHistogram;

pub use cpu::rt_specular::cpu_rt_stochastic_specular;
pub use cpu::rt_diffuse::cpu_rt_stochastic_diffuse;

pub use shared::DEBUG_LEAK_COUNTER;
pub use shared::DEBUG_MIC_HITS;
pub use shared::DEBUG_MIC_HITS_OUT_OF_BOUNDS;
// pub use rt_specular::rt_specular_cpu;
// pub use rt_diffuse::rt_diffuse_cpu;
