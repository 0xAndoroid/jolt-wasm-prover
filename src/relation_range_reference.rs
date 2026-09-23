//! CPU reference implementation of the stage-2 relation-range device ABI
//! (`akita_prover::RelationRangeDevice`), structured the way the WebGPU
//! kernels compute it: per round fold the witness and the flat weights by the
//! previous challenge, then accumulate the six round terms over live pairs.
//! It exists to prove the seam (proof bytes must not change with it
//! installed) and to be the bit-exact oracle for the kernels; it is not fast.

use akita_prover::{
    RelationRangeDevice, RelationRangeJob, RelationRangeSession, RelationRangeShape,
    RELATION_RANGE_FIELD_BYTES as FIELD_BYTES, RELATION_RANGE_MESSAGE_BYTES as MESSAGE_BYTES,
};
use jolt_akita::{AkitaError, AkitaField as F};
use jolt_field::{CanonicalEncoding, Ring};
use rayon::prelude::*;

/// Hand the CPU a `2^CPU_TAIL_BITS`-entry weight table.
const CPU_TAIL_BITS: u32 = 12;

pub struct CpuReferenceDevice;

impl RelationRangeDevice for CpuReferenceDevice {
    fn rounds_for(&self, shape: &RelationRangeShape) -> usize {
        shape.domain_len.ilog2().saturating_sub(CPU_TAIL_BITS) as usize
    }

    fn open(
        &self,
        job: &RelationRangeJob<'_>,
        rounds: usize,
    ) -> Result<Box<dyn RelationRangeSession>, AkitaError> {
        let digits = decode_digits(job.digits, job.shape.bit_width, job.shape.len);
        let weights = fields_from_bytes(job.weights)?;
        if weights.len() != job.shape.domain_len {
            return Err(AkitaError::InvalidSize {
                expected: job.shape.domain_len,
                actual: weights.len(),
            });
        }
        Ok(Box::new(Session {
            digits,
            witness: None,
            weights,
            rounds,
        }))
    }
}

/// `PackedSignedDigits` codec: digit `k` occupies bits `[k*bw, (k+1)*bw)`
/// LSB-first, two's complement.
pub fn decode_digits(bytes: &[u8], bit_width: u8, len: usize) -> Vec<i8> {
    let bw = u32::from(bit_width);
    let mask = (1u32 << bw) - 1;
    (0..len)
        .map(|k| {
            let bit = k * bw as usize;
            let byte = bit / 8;
            let word = u32::from(bytes[byte])
                | bytes.get(byte + 1).map_or(0, |&b| u32::from(b) << 8)
                | bytes.get(byte + 2).map_or(0, |&b| u32::from(b) << 16);
            let raw = (word >> (bit % 8)) & mask;
            if raw >> (bw - 1) & 1 == 1 {
                (raw as i32 - (1i32 << bw)) as i8
            } else {
                raw as i8
            }
        })
        .collect()
}

pub fn fields_from_bytes(bytes: &[u8]) -> Result<Vec<F>, AkitaError> {
    if !bytes.len().is_multiple_of(FIELD_BYTES) {
        return Err(AkitaError::InvalidSize {
            expected: bytes.len().div_ceil(FIELD_BYTES) * FIELD_BYTES,
            actual: bytes.len(),
        });
    }
    Ok(bytes
        .chunks_exact(FIELD_BYTES)
        .map(|c| F::from_u128_reduced(u128::from_le_bytes(c.try_into().unwrap())))
        .collect())
}

pub fn fields_to_bytes(values: &[F]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| v.to_u128_checked().expect("canonical").to_le_bytes())
        .collect()
}

fn field_from_bytes(bytes: &[u8; FIELD_BYTES]) -> F {
    F::from_u128_reduced(u128::from_le_bytes(*bytes))
}

/// `left + r * (right - left)` over adjacent pairs, zero-padded (the CPU's
/// `fold_evals_in_place`).
pub fn fold(table: &[F], r: F) -> Vec<F> {
    (0..table.len().div_ceil(2))
        .into_par_iter()
        .map(|j| {
            let left = table[2 * j];
            let right = table.get(2 * j + 1).copied().unwrap_or_else(F::zero);
            left + r * (right - left)
        })
        .collect()
}

struct Session {
    digits: Vec<i8>,
    /// `None` while the witness is still the compact digit table.
    witness: Option<Vec<F>>,
    weights: Vec<F>,
    rounds: usize,
}

/// The six round terms over `live_pairs` pairs: `w0, w1` from `witness`,
/// `p0, p1` from `weights`, `eq(j) = e_first[j & mask] * e_second[j >> bits]`
/// on the norm terms only.
pub fn round_terms(
    witness: impl Fn(usize) -> F + Sync,
    live_pairs: usize,
    weights: &[F],
    e_first: &[F],
    e_second: &[F],
) -> [F; 6] {
    let mask = e_first.len() - 1;
    let bits = e_first.len().trailing_zeros();
    (0..live_pairs)
        .into_par_iter()
        .fold(
            || [F::zero(); 6],
            |mut acc, j| {
                let w0 = witness(2 * j);
                let dw = witness(2 * j + 1) - w0;
                let (p0, p1) = (weights[2 * j], weights[2 * j + 1]);
                let dp = p1 - p0;
                let e = e_first[j & mask] * e_second[j >> bits];
                acc[0] += e * (w0 * (w0 + F::one()));
                acc[1] += e * (dw * (w0 + w0 + F::one()));
                acc[2] += e * (dw * dw);
                acc[3] += w0 * p0;
                acc[4] += w0 * dp + dw * p0;
                acc[5] += dw * dp;
                acc
            },
        )
        .reduce(
            || [F::zero(); 6],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
                a
            },
        )
}

impl RelationRangeSession for Session {
    fn round(
        &mut self,
        round: usize,
        prev: Option<&[u8; FIELD_BYTES]>,
        e_first: &[u8],
        e_second: &[u8],
        out: &mut [u8; MESSAGE_BYTES],
    ) -> Result<(), AkitaError> {
        if round >= self.rounds {
            return Err(AkitaError::InvalidInput(format!(
                "relation-range reference: round {round} past {}",
                self.rounds
            )));
        }
        if let Some(prev) = prev {
            let r = field_from_bytes(prev);
            self.witness = Some(match self.witness.take() {
                Some(w) => fold(&w, r),
                None => {
                    let d = &self.digits;
                    (0..d.len().div_ceil(2))
                        .into_par_iter()
                        .map(|j| {
                            let left = F::from_i64(i64::from(d[2 * j]));
                            let right = d
                                .get(2 * j + 1)
                                .map_or_else(F::zero, |&v| F::from_i64(i64::from(v)));
                            left + r * (right - left)
                        })
                        .collect()
                }
            });
            self.weights = fold(&self.weights, r);
        }
        let e_first = fields_from_bytes(e_first)?;
        let e_second = fields_from_bytes(e_second)?;
        let terms = match &self.witness {
            Some(w) => round_terms(
                |i| w.get(i).copied().unwrap_or_else(F::zero),
                w.len().div_ceil(2),
                &self.weights,
                &e_first,
                &e_second,
            ),
            None => round_terms(
                |i| {
                    self.digits
                        .get(i)
                        .map_or_else(F::zero, |&v| F::from_i64(i64::from(v)))
                },
                self.digits.len().div_ceil(2),
                &self.weights,
                &e_first,
                &e_second,
            ),
        };
        out.copy_from_slice(&fields_to_bytes(&terms));
        Ok(())
    }

    fn tables(self: Box<Self>) -> Result<(Vec<u8>, Vec<u8>), AkitaError> {
        let witness = self.witness.ok_or_else(|| {
            AkitaError::InvalidInput("relation-range reference: tables before round 1".into())
        })?;
        Ok((fields_to_bytes(&witness), fields_to_bytes(&self.weights)))
    }
}
