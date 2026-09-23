//! CPU reference implementation of the trace-commit device ABI, structured
//! the way a GPU kernel computes it: one work item per output ring, exact
//! signed accumulation (positive and negative 2^128-wrapping sums with a wrap
//! counter), one canonical fold at the end. It exists to prove the seam and
//! to be the bit-exact oracle for the WebGPU kernel; it is not fast.

use jolt_akita::{AkitaError, TraceCommitDevice, TraceCommitJob};
use rayon::prelude::*;

/// p = 2^128 − C, the Akita field modulus.
const C: u128 = 0xFFFF_A7F7;
const P: u128 = 0u128.wrapping_sub(C);
const K: usize = 16;
const D: usize = 512;

/// Qualifies K=16, D=512, n_a=1, num_digits_inner=1 (every shipped sha2-chain
/// shape); declines everything else so the CPU kernels run.
pub struct CpuReferenceDevice;

impl TraceCommitDevice for CpuReferenceDevice {
    fn commit_accumulate(&self, job: &TraceCommitJob<'_>) -> Option<Result<Vec<u32>, AkitaError>> {
        let shape = job.shape;
        (shape.one_hot_k == K
            && shape.ring_dimension == D
            && shape.n_a == 1
            && shape.num_digits_inner == 1)
            .then(|| Ok(commit(job)))
    }
}

/// `lo + wraps * 2^128`, a sum of canonical field elements.
#[derive(Clone, Copy, Default)]
struct Wide {
    lo: u128,
    wraps: u32,
}

impl Wide {
    #[inline(always)]
    fn add(&mut self, value: u128) {
        let (lo, carry) = self.lo.overflowing_add(value);
        self.lo = lo;
        self.wraps += u32::from(carry);
    }

    /// Canonical value: 2^128 ≡ C (mod p), so each wrap folds to `+C`.
    fn fold(self) -> u128 {
        let mut value = self.lo;
        let mut carry = u128::from(self.wraps) * C;
        loop {
            let (sum, overflow) = value.overflowing_add(carry);
            value = sum;
            if !overflow {
                break;
            }
            carry = C;
        }
        if value >= P {
            value - P
        } else {
            value
        }
    }
}

/// `pos − neg (mod p)` for canonical inputs.
fn sub_mod(pos: u128, neg: u128) -> u128 {
    if pos >= neg {
        pos - neg
    } else {
        pos.wrapping_sub(neg).wrapping_sub(C)
    }
}

fn read_u128(limbs: &[u32]) -> u128 {
    u128::from(limbs[0])
        | u128::from(limbs[1]) << 32
        | u128::from(limbs[2]) << 64
        | u128::from(limbs[3]) << 96
}

fn commit(job: &TraceCommitJob<'_>) -> Vec<u32> {
    let shape = job.shape;
    let positions = shape.positions_per_block;
    let columns = shape.num_columns;
    let rows_per_ring = D / K;
    let a_plane = job
        .a_plane
        .chunks_exact(4)
        .map(read_u128)
        .collect::<Vec<_>>();
    let mut out = vec![0u32; shape.num_blocks * D * 4];
    out.par_chunks_exact_mut(D * 4)
        .enumerate()
        .for_each(|(block, out)| {
            let column = block / shape.blocks_per_column;
            let trace_block = block % shape.blocks_per_column;
            if column >= columns {
                return;
            }
            let mut pos = vec![Wide::default(); D];
            let mut neg = vec![Wide::default(); D];
            for (position, a_ring) in a_plane.chunks_exact(D).take(positions).enumerate() {
                for row_in_ring in 0..rows_per_ring {
                    let row = (trace_block * positions + position) * rows_per_ring + row_in_ring;
                    let hot = job.hot[row * columns + column];
                    if hot == 0 && job.masks[row] >> column & 1 == 0 {
                        continue;
                    }
                    let shift = row_in_ring * K + usize::from(hot);
                    for (dst, &value) in pos[shift..].iter_mut().zip(a_ring) {
                        dst.add(value);
                    }
                    for (dst, &value) in neg[..shift].iter_mut().zip(&a_ring[D - shift..]) {
                        dst.add(value);
                    }
                }
            }
            for ((out, pos), neg) in out.chunks_exact_mut(4).zip(&pos).zip(&neg) {
                let value = sub_mod(pos.fold(), neg.fold());
                out[0] = value as u32;
                out[1] = (value >> 32) as u32;
                out[2] = (value >> 64) as u32;
                out[3] = (value >> 96) as u32;
            }
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_akita::TraceCommitShape;

    fn add_mod(a: u128, b: u128) -> u128 {
        let (sum, overflow) = a.overflowing_add(b);
        let sum = if overflow { sum.wrapping_add(C) } else { sum };
        if sum >= P {
            sum - P
        } else {
            sum
        }
    }

    fn write_u128(out: &mut [u32], value: u128) {
        for (limb, out) in out.iter_mut().enumerate() {
            *out = (value >> (32 * limb)) as u32;
        }
    }

    #[test]
    fn matches_naive_field_arithmetic() {
        let shape = TraceCommitShape {
            one_hot_k: K,
            ring_dimension: D,
            num_rows: 1024,
            num_columns: 5,
            column_capacity: 8,
            n_a: 1,
            positions_per_block: 16,
            num_digits_inner: 1,
            segment_rings: 32,
            blocks_per_column: 2,
            num_blocks: 16,
        };
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let hot = (0..shape.num_rows * shape.num_columns)
            .map(|_| (next() % 16) as u8)
            .collect::<Vec<_>>();
        let masks = (0..shape.num_rows).map(|_| next() % 32).collect::<Vec<_>>();
        let mut a_plane = vec![0u32; shape.positions_per_block * D * 4];
        for limbs in a_plane.chunks_exact_mut(4) {
            let value = u128::from(next()) << 64 | u128::from(next());
            write_u128(limbs, if value >= P { value - P } else { value });
        }
        let job = TraceCommitJob {
            shape,
            hot: &hot,
            masks: &masks,
            a_plane: &a_plane,
        };

        let mut expected = vec![0u128; shape.num_blocks * D];
        for row in 0..shape.num_rows {
            let ring = row / (D / K);
            let trace_block = ring / shape.positions_per_block;
            let position = ring % shape.positions_per_block;
            for column in 0..shape.num_columns {
                let hot = hot[row * shape.num_columns + column];
                if hot == 0 && masks[row] >> column & 1 == 0 {
                    continue;
                }
                let shift = (row % (D / K)) * K + usize::from(hot);
                let block = column * shape.blocks_per_column + trace_block;
                for i in 0..D {
                    let value =
                        read_u128(&a_plane[(position * D + (i + D - shift) % D) * 4..][..4]);
                    let dst = &mut expected[block * D + i];
                    *dst = if i >= shift {
                        add_mod(*dst, value)
                    } else {
                        add_mod(*dst, P - value)
                    };
                }
            }
        }
        let mut expected_limbs = vec![0u32; expected.len() * 4];
        for (limbs, &value) in expected_limbs.chunks_exact_mut(4).zip(&expected) {
            write_u128(limbs, value);
        }

        let actual = CpuReferenceDevice.commit_accumulate(&job).unwrap().unwrap();
        assert_eq!(actual, expected_limbs);
        assert!(expected[..shape.num_columns * shape.blocks_per_column * D]
            .iter()
            .any(|&value| value != 0));
    }

    #[test]
    fn declines_other_shapes() {
        let shape = TraceCommitShape {
            one_hot_k: 256,
            ring_dimension: D,
            num_rows: 2,
            num_columns: 1,
            column_capacity: 1,
            n_a: 1,
            positions_per_block: 1,
            num_digits_inner: 1,
            segment_rings: 1,
            blocks_per_column: 1,
            num_blocks: 1,
        };
        let job = TraceCommitJob {
            shape,
            hot: &[0; 2],
            masks: &[0; 2],
            a_plane: &[],
        };
        assert!(CpuReferenceDevice.commit_accumulate(&job).is_none());
    }
}
