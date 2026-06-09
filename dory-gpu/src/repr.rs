//! Host <-> GPU representations.
//!
//! Field elements cross the boundary in Montgomery form as 8 little-endian u32
//! words — bit-identical to arkworks' internal `BigInt<4>` layout, so all
//! conversions are pure repacking (no Montgomery conversions on the host).
//!
//! Points use homogeneous projective coordinates on the GPU (the complete
//! formulas need them); arkworks uses Jacobian, so point conversions go
//! through cheap coordinate maps documented on each function.

use ark_bn254::{Fq, Fq2, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ff::{AdditiveGroup, BigInt, Field, Fp, Fp2, PrimeField};
use ark_std::Zero;

pub const FQ_WORDS: usize = 8;
pub const FQ2_WORDS: usize = 16;
pub const G1_AFFINE_WORDS: usize = 2 * FQ_WORDS;
pub const G1_PROJ_WORDS: usize = 3 * FQ_WORDS;
pub const G2_AFFINE_WORDS: usize = 2 * FQ2_WORDS;
pub const G2_PROJ_WORDS: usize = 3 * FQ2_WORDS;
pub const FR_WORDS: usize = 8;

#[inline]
pub fn bigint_to_words(v: &BigInt<4>) -> [u32; 8] {
    let mut out = [0u32; 8];
    for (i, w) in v.0.iter().enumerate() {
        out[2 * i] = *w as u32;
        out[2 * i + 1] = (*w >> 32) as u32;
    }
    out
}

#[inline]
pub fn words_to_bigint(w: &[u32]) -> BigInt<4> {
    let mut limbs = [0u64; 4];
    for i in 0..4 {
        limbs[i] = (w[2 * i] as u64) | ((w[2 * i + 1] as u64) << 32);
    }
    BigInt(limbs)
}

#[inline]
pub fn fq_to_words(x: &Fq) -> [u32; 8] {
    bigint_to_words(&x.0)
}

#[inline]
pub fn fq_from_words(w: &[u32]) -> Fq {
    Fp::new_unchecked(words_to_bigint(w))
}

#[inline]
pub fn fr_to_words(x: &Fr) -> [u32; 8] {
    bigint_to_words(&x.0)
}

#[inline]
pub fn fr_from_words(w: &[u32]) -> Fr {
    Fp::new_unchecked(words_to_bigint(w))
}

#[inline]
pub fn fq2_to_words(x: &Fq2) -> [u32; 16] {
    let mut out = [0u32; 16];
    out[..8].copy_from_slice(&fq_to_words(&x.c0));
    out[8..].copy_from_slice(&fq_to_words(&x.c1));
    out
}

#[inline]
pub fn fq2_from_words(w: &[u32]) -> Fq2 {
    Fp2::new(fq_from_words(&w[..8]), fq_from_words(&w[8..16]))
}

/// Affine -> packed (x, y). The caller must not pass the point at infinity;
/// Dory setup generators and lookup-table entries are never the identity.
pub fn g1_affine_to_words(p: &G1Affine) -> [u32; G1_AFFINE_WORDS] {
    debug_assert!(!p.infinity, "cannot upload G1 point at infinity as affine");
    let mut out = [0u32; G1_AFFINE_WORDS];
    out[..8].copy_from_slice(&fq_to_words(&p.x));
    out[8..].copy_from_slice(&fq_to_words(&p.y));
    out
}

/// Homogeneous projective (X, Y, Z) with x = X/Z, y = Y/Z.
/// From an affine point this is just (x, y, 1).
pub fn g1_affine_to_proj_words(p: &G1Affine) -> [u32; G1_PROJ_WORDS] {
    let mut out = [0u32; G1_PROJ_WORDS];
    if p.infinity {
        out[8..16].copy_from_slice(&fq_to_words(&Fq::ONE));
        return out;
    }
    out[..8].copy_from_slice(&fq_to_words(&p.x));
    out[8..16].copy_from_slice(&fq_to_words(&p.y));
    out[16..].copy_from_slice(&fq_to_words(&Fq::ONE));
    out
}

/// arkworks Jacobian (X, Y, Z): x = X/Z^2, y = Y/Z^3 -> homogeneous (X*Z, Y, Z^3).
pub fn g1_proj_to_words(p: &G1Projective) -> [u32; G1_PROJ_WORDS] {
    let z2 = p.z * p.z;
    let mut out = [0u32; G1_PROJ_WORDS];
    out[..8].copy_from_slice(&fq_to_words(&(p.x * p.z)));
    out[8..16].copy_from_slice(&fq_to_words(&p.y));
    out[16..].copy_from_slice(&fq_to_words(&(z2 * p.z)));
    out
}

/// Homogeneous (X, Y, Z) -> arkworks Jacobian (X*Z, Y*Z^2, Z).
pub fn g1_proj_from_words(w: &[u32]) -> G1Projective {
    let x = fq_from_words(&w[..8]);
    let y = fq_from_words(&w[8..16]);
    let z = fq_from_words(&w[16..24]);
    if z.is_zero() {
        return G1Projective::zero();
    }
    let z2 = z * z;
    G1Projective::new_unchecked(x * z, y * z2, z)
}

pub fn g2_affine_to_words(p: &G2Affine) -> [u32; G2_AFFINE_WORDS] {
    debug_assert!(!p.infinity, "cannot upload G2 point at infinity as affine");
    let mut out = [0u32; G2_AFFINE_WORDS];
    out[..16].copy_from_slice(&fq2_to_words(&p.x));
    out[16..].copy_from_slice(&fq2_to_words(&p.y));
    out
}

pub fn g2_affine_to_proj_words(p: &G2Affine) -> [u32; G2_PROJ_WORDS] {
    let mut out = [0u32; G2_PROJ_WORDS];
    if p.infinity {
        out[16..32].copy_from_slice(&fq2_to_words(&Fq2::new(Fq::ONE, Fq::ZERO)));
        return out;
    }
    out[..16].copy_from_slice(&fq2_to_words(&p.x));
    out[16..32].copy_from_slice(&fq2_to_words(&p.y));
    out[32..].copy_from_slice(&fq2_to_words(&Fq2::new(Fq::ONE, Fq::ZERO)));
    out
}

pub fn g2_proj_to_words(p: &G2Projective) -> [u32; G2_PROJ_WORDS] {
    let z2 = p.z * p.z;
    let mut out = [0u32; G2_PROJ_WORDS];
    out[..16].copy_from_slice(&fq2_to_words(&(p.x * p.z)));
    out[16..32].copy_from_slice(&fq2_to_words(&p.y));
    out[32..].copy_from_slice(&fq2_to_words(&(z2 * p.z)));
    out
}

pub fn g2_proj_from_words(w: &[u32]) -> G2Projective {
    let x = fq2_from_words(&w[..16]);
    let y = fq2_from_words(&w[16..32]);
    let z = fq2_from_words(&w[32..48]);
    if z.is_zero() {
        return G2Projective::zero();
    }
    let z2 = z * z;
    G2Projective::new_unchecked(x * z, y * z2, z)
}

/// Packs a slice of values into a flat word vector.
pub fn pack_slice<T, const N: usize>(items: &[T], f: impl Fn(&T) -> [u32; N]) -> Vec<u32> {
    let mut out = Vec::with_capacity(items.len() * N);
    for item in items {
        out.extend_from_slice(&f(item));
    }
    out
}

/// Canonical (non-Montgomery) scalar words, for host-side digit logic.
pub fn fr_canonical_words(x: &Fr) -> [u32; 8] {
    bigint_to_words(&x.into_bigint())
}
