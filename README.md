# GA Simulation

This repository houses an in-progress rewrite of my 2025 RACA-CON presentation
on geometrical acoustics. The presentation showcased a prototype research
simulator that used spatial caching and lookups to turn the diffuse raytracing
phase from an O(a^n) to O(n) by caching reflection-order passes.

The rewrite was necessary to quantify accuracy and speed, and time complexity of
the simulation. In addition, rather than use toy virtual models like in the
RACA-CON presentation, this rewrite uses the BRAS standard acoustical dataset
from the Technical University of Berlin, which has higher resolution
measurements than the toy models that were approximated in the RACA-CON version.

![](/DeviationNormalized.png)

## Research Objectives and Status

The rewrite is currently incomplete. Here are the following steps in working
towards a paper release:

- [x] 1. Correctly load the new BRAS dataset from TU Berlin
- [x] 2. Write a specular-trace simulation
  - [ ] 2b. (Optionally) write a GPGPU version
- [x] 3. Write a naive diffuse-trace simulation for benchmarking purposes
  - [ ] 3b. (Optionally) write a GPGPU version
- [x] 4. Quantify the simulation statistics before introducing the spatial cache
- [ ] 5. Introduce the spatial cache diffuse-trace technique
- [ ] 6. Quantify the new technique
- [ ] 7. Write the paper

## Other Technologies in Use

From the get-go, a quasi-objective was to determine the computational and
accuracy performance of a geometrical-acoustics simulator built with modern
tools and techniques.

The simulation itself is written in Rust, which makes expressing
performance-minded code extremely easy and erganomic. Not to mention, the
borrow-checker in the rust compiler makes it trivially easy to reason about
allocaiton lifetimes, even in complex multithreaded code, and thus encouraging
their use.

As a result, the simulation is (optionally) fully multithreaded. The raytracing
itself uses *obvhs* under the hood, an open-source implementation of an NVIDIA
paper on using wide bounding volume hierarchies and wide SIMD intrinsics to
quickly prune large portions of a scene during ray intersection tests. The
implementation is only slightly slower than Intel's *Embree*, but is cross
platform. Obvhs is also the fastest software accelerator for GPUs, being
surpassed only RTX hardware raytracing.

For scientific computing on GPU, *wgpu*, a Rust implementation of the WebGPU
spec was chosen for it's cross platform compute shader support and good
integration into the Rust ecosystem over toolchains like CUDA. Moreover,
vendor lock-in is an anti-goal of this project, and *wgpu* is sufficiently
powerful in describing high-performance compute shaders.
