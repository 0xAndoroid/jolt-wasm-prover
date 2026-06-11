//! CPU-side parallelism helpers that degrade to serial execution on WASM.
//!
//! In the e2e bridge the GPU jobs execute on a dedicated worker while every
//! rayon pool thread may be blocked waiting on the engine — engine-side
//! rayon would deadlock. The amounts of CPU work involved (a handful of
//! final exponentiations per pass) are milliseconds, so serial execution on
//! WASM costs noise.

use ark_bn254::Fq12;
use ark_ec::CurveGroup;
use dory_pcs::backends::arkworks::ArkGT;

use crate::pairing::final_exponentiation;

/// `CurveGroup::normalize_batch` routes through rayon for large inputs; on
/// WASM the engine runs on the GPU worker while every pool thread may be
/// parked waiting on it, so setup-time normalization must stay serial
/// (per-point inversions; setup-time only, small inputs).
pub fn normalize_batch<G: CurveGroup>(points: &[G]) -> Vec<G::Affine> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        G::normalize_batch(points)
    }
    #[cfg(target_arch = "wasm32")]
    {
        // Serial fallback: per-chunk calls below any parallel threshold
        // would still be ark-internal; do the straightforward thing instead
        // and normalize one by one (setup-time only, small inputs).
        points.iter().map(|p| (*p).into_affine()).collect()
    }
}

pub fn join2<A, B>(a: impl FnOnce() -> A + Send, b: impl FnOnce() -> B + Send) -> (A, B)
where
    A: Send,
    B: Send,
{
    #[cfg(not(target_arch = "wasm32"))]
    {
        rayon::join(a, b)
    }
    #[cfg(target_arch = "wasm32")]
    {
        (a(), b())
    }
}

pub fn final_exps(fs: Vec<Fq12>) -> Vec<ArkGT> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        fs.into_par_iter()
            .map(|f| ArkGT(final_exponentiation(f)))
            .collect()
    }
    #[cfg(target_arch = "wasm32")]
    {
        fs.into_iter()
            .map(|f| ArkGT(final_exponentiation(f)))
            .collect()
    }
}
