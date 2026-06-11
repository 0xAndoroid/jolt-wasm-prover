//! CPU-side parallelism helpers that degrade to serial execution on WASM.
//!
//! In the e2e bridge the GPU jobs execute on a dedicated worker while every
//! rayon pool thread may be blocked waiting on the engine — engine-side
//! rayon would deadlock. The amounts of CPU work involved (a handful of
//! final exponentiations per pass) are milliseconds, so serial execution on
//! WASM costs noise.

use ark_bn254::Fq12;
use dory_pcs::backends::arkworks::ArkGT;

use crate::pairing::final_exponentiation;

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
