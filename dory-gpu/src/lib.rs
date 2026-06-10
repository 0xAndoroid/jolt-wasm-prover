//! WebGPU implementation of the Dory polynomial commitment scheme (BN254).
//!
//! The protocol logic mirrors `dory-pcs` exactly (same transcript, same
//! messages, byte-identical proofs); the heavy arithmetic — MSMs, vector
//! folds, Miller loops, the vector-matrix product — runs in WGSL compute
//! shaders. Final exponentiations, masks and transcript work stay on the CPU,
//! where they are cheap and inherently sequential.

pub mod context;
pub mod msm;
pub mod repr;
pub mod shader;

pub use context::GpuContext;
