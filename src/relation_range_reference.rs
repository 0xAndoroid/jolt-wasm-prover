//! CPU reference implementation of the stage-2 relation-range device ABI
//! (`akita_prover::RelationRangeDevice`), structured the way the WebGPU
//! kernels compute it: per round fold the witness and (dense) weights by the
//! previous challenge, evaluate the factored or dense relation weights per
//! pair, then accumulate the ten round terms. It exists to prove the seam
//! (proof bytes must not change with it installed) and as the bit-exact
//! oracle for the kernels; it is not fast.

use akita_prover::{
    RelationRangeDevice, RelationRangeJob, RelationRangeRoundInput as RoundInput,
    RelationRangeSession, RelationRangeShape, RelationRangeWeightsKind as WeightsKind,
    RELATION_RANGE_FIELD_BYTES as FIELD_BYTES, RELATION_RANGE_MESSAGE_BYTES as MESSAGE_BYTES,
    RELATION_RANGE_PAIR_BYTES as PAIR_BYTES, RELATION_RANGE_SEGMENT_BYTES as SEGMENT_BYTES,
};
use jolt_akita::{AkitaError, AkitaField as F};
use jolt_field::{CanonicalEncoding, One, Ring, Zero};
use rayon::prelude::*;

/// Hand the CPU a `2^CPU_TAIL_BITS`-entry domain.
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
        let lane_weights = fields_from_bytes(job.lane_weights)?;
        if job.shape.weights == WeightsKind::Dense && weights.len() != job.shape.domain_len {
            return Err(AkitaError::InvalidSize {
                expected: job.shape.domain_len,
                actual: weights.len(),
            });
        }
        let segments = job
            .segments
            .chunks_exact(SEGMENT_BYTES)
            .map(|c| Segment {
                source_lane: u32_at(c, 0) as usize,
                source_index: u32_at(c, 4) as usize,
                factor: field_from_bytes(&c[16..32]),
            })
            .collect();
        Ok(Box::new(Session {
            digits,
            witness: None,
            weights: (job.shape.weights == WeightsKind::Dense).then_some(weights),
            lane_weights,
            segments,
            lane_map: job.lane_map.to_vec(),
            source_lanes: job.source_lanes.to_vec(),
            coefficient_bits: job.shape.coefficient_bits,
            kind: job.shape.weights,
            rounds,
        }))
    }
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

struct Segment {
    source_lane: usize,
    source_index: usize,
    factor: F,
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
        .map(field_from_bytes)
        .collect())
}

pub fn fields_to_bytes(values: &[F]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| v.to_u128_checked().expect("canonical").to_le_bytes())
        .collect()
}

fn field_from_bytes(bytes: &[u8]) -> F {
    F::from_u128_reduced(u128::from_le_bytes(bytes.try_into().unwrap()))
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
    /// Flat weights (`Dense` from the start, `Factored` from the first lane round).
    weights: Option<Vec<F>>,
    lane_weights: Vec<F>,
    segments: Vec<Segment>,
    lane_map: Vec<u32>,
    source_lanes: Vec<u32>,
    coefficient_bits: usize,
    kind: WeightsKind,
    rounds: usize,
}

impl Session {
    /// Factored weight at flat index `i` with the coefficient width of this
    /// round: `alpha[c] * lane_w[l] + sum over segments containing l`.
    fn factored_weight(
        &self,
        i: usize,
        cw: usize,
        alpha: &[F],
        sources: &[F],
        source_offsets: &[usize],
    ) -> F {
        let cc = 1usize << cw;
        let c = i & (cc - 1);
        let lane = i >> cw;
        let mut p = alpha[c] * self.lane_weights[lane];
        let map = self.lane_map.get(lane).copied().unwrap_or(0);
        let start = (map >> 8) as usize;
        for seg in &self.segments[start..start + (map & 0xFF) as usize] {
            p += seg.factor * sources[source_offsets[seg.source_index] + seg.source_lane * cc + c];
        }
        p
    }
}

/// The ten round terms over `live_pairs` pairs: `w0, w1` from `witness`,
/// `p0, p1` from `weight`, `eq(j) = e_first[j & mask] * e_second[j >> bits]`
/// on the norm terms only, plus the additional cubic over `pairs`.
#[allow(clippy::too_many_arguments)]
pub fn round_terms(
    witness: impl Fn(usize) -> F + Sync,
    live_pairs: usize,
    weight: impl Fn(usize) -> F + Sync,
    e_first: &[F],
    e_second: &[F],
    pairs: &[u8],
) -> [F; 10] {
    let mask = e_first.len() - 1;
    let bits = e_first.len().trailing_zeros();
    let mut acc = (0..live_pairs)
        .into_par_iter()
        .fold(
            || [F::zero(); 10],
            |mut acc, j| {
                let w0 = witness(2 * j);
                let dw = witness(2 * j + 1) - w0;
                let (p0, p1) = (weight(2 * j), weight(2 * j + 1));
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
            || [F::zero(); 10],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
                a
            },
        );
    for pair in pairs.chunks_exact(PAIR_BYTES) {
        let m = u32_at(pair, 0) as usize;
        let (l0, l1, b0, b1) = (
            field_from_bytes(&pair[16..32]),
            field_from_bytes(&pair[32..48]),
            field_from_bytes(&pair[48..64]),
            field_from_bytes(&pair[64..80]),
        );
        let w0 = witness(2 * m);
        let dw = witness(2 * m + 1) - w0;
        let (dl, db) = (l1 - l0, b1 - b0);
        let (q0, q1, q2) = (w0 * (w0 + F::one()), dw * (w0 + w0 + F::one()), dw * dw);
        acc[6] += w0 * l0 + b0 * q0;
        acc[7] += w0 * dl + dw * l0 + b0 * q1 + db * q0;
        acc[8] += dw * dl + b0 * q2 + db * q1;
        acc[9] += db * q2;
    }
    acc
}

impl RelationRangeSession for Session {
    fn round(
        &mut self,
        input: &RoundInput<'_>,
        out: &mut [u8; MESSAGE_BYTES],
    ) -> Result<(), AkitaError> {
        let round = input.round;
        if round >= self.rounds {
            return Err(AkitaError::InvalidInput(format!(
                "relation-range reference: round {round} past {}",
                self.rounds
            )));
        }
        if let Some(prev) = input.prev {
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
            if let Some(weights) = &self.weights {
                self.weights = Some(fold(weights, r));
            }
        }
        let e_first = fields_from_bytes(input.e_first)?;
        let e_second = fields_from_bytes(input.e_second)?;
        let alpha = fields_from_bytes(input.alpha)?;
        let sources = fields_from_bytes(input.sources)?;
        let cw = self.coefficient_bits.saturating_sub(round);
        let mut source_offsets = Vec::with_capacity(self.source_lanes.len());
        let mut off = 0usize;
        for lanes in &self.source_lanes {
            source_offsets.push(off);
            off += (*lanes as usize) << cw;
        }
        let digits = &self.digits;
        let witness_at = |i: usize| match &self.witness {
            Some(w) => w.get(i).copied().unwrap_or_else(F::zero),
            None => digits
                .get(i)
                .map_or_else(F::zero, |&v| F::from_i64(i64::from(v))),
        };
        let live_pairs = match &self.witness {
            Some(w) => w.len().div_ceil(2),
            None => digits.len().div_ceil(2),
        };
        let factored_phase = self.kind == WeightsKind::Factored && round <= self.coefficient_bits;
        let terms = if factored_phase {
            let weight = |i: usize| self.factored_weight(i, cw, &alpha, &sources, &source_offsets);
            let terms = round_terms(
                witness_at,
                live_pairs,
                weight,
                &e_first,
                &e_second,
                input.pairs,
            );
            if round == self.coefficient_bits {
                // First lane round: the factored weights become a flat lane table.
                let domain = self.lane_weights.len();
                self.weights = Some((0..domain).into_par_iter().map(weight).collect());
            }
            terms
        } else {
            let weights = self.weights.as_ref().expect("dense weights present");
            round_terms(
                witness_at,
                live_pairs,
                |i| weights[i],
                &e_first,
                &e_second,
                input.pairs,
            )
        };
        out.copy_from_slice(&fields_to_bytes(&terms));
        Ok(())
    }

    fn tables(self: Box<Self>) -> Result<(Vec<u8>, Vec<u8>), AkitaError> {
        let witness = self.witness.ok_or_else(|| {
            AkitaError::InvalidInput("relation-range reference: tables before round 1".into())
        })?;
        let weights = self
            .weights
            .map(|w| fields_to_bytes(&w))
            .unwrap_or_default();
        Ok((fields_to_bytes(&witness), weights))
    }
}
