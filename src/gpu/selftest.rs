//! GPU correctness oracle: `a·b + c mod p` over 2^20 random (partly
//! non-canonical) inputs, checked against `AkitaField` on the CPU, plus a
//! NOP round-trip latency probe.

use jolt_akita::AkitaField;
use jolt_field::CanonicalEncoding;

use super::mailbox::{
    self, GpuError, Region, OP_CREATE_BUFFER, OP_DESTROY, OP_NOP, OP_RUN, OP_UPLOAD,
};
use super::now_ms;
use super::GpuReport;

pub const SHADER_FP128_OPS: u32 = 0;
pub const FP128_OP_MULADD: u32 = 3;
pub const WORKGROUP_SIZE: u32 = 256;

const N: usize = 1 << 20;
const NOP_ROUNDS: u32 = 200;
/// W1's largest persistent buffer; `a` is uploaded into it so every prove
/// exercises CREATE_BUFFER / UPLOAD / handle binding / DESTROY at that size.
const PERSISTENT_BYTES: u32 = 256 << 20;

/// `args` for `OP_RUN`: shader, workgroups xyz, param count, ≤16 params,
/// binding count. Params land in the uniform at binding 0; regions bind
/// from 1 in order.
pub fn run_args(shader: u32, workgroups: [u32; 3], params: &[u32], bindings: u32) -> Vec<u32> {
    assert!(params.len() <= 16);
    let mut args = vec![
        shader,
        workgroups[0],
        workgroups[1],
        workgroups[2],
        params.len() as u32,
    ];
    args.extend_from_slice(params);
    args.resize(21, 0);
    args.push(bindings);
    args
}

struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Raw u128: about 1/2^32 of the draws are ≥ p, so the shader also sees
    /// non-canonical inputs on the edge cases seeded below.
    fn u128(&mut self) -> u128 {
        (self.next() as u128) << 64 | self.next() as u128
    }
}

fn to_le_bytes(values: &[u128]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub fn run() -> Result<GpuReport, GpuError> {
    let t0 = now_ms();
    for _ in 0..NOP_ROUNDS {
        mailbox::call(OP_NOP, &[], &[])?;
    }
    let roundtrip_us = (now_ms() - t0) * 1000.0 / NOP_ROUNDS as f64;

    let mut rng = SplitMix(0x5EED_A7F7);
    let p = u128::MAX - 0xFFFF_A7F6;
    let edges = [0u128, 1, p - 1, p, p + 1, u128::MAX, 0xFFFF_A7F7];
    let mut a: Vec<u128> = Vec::with_capacity(N);
    let mut b: Vec<u128> = Vec::with_capacity(N);
    let mut c: Vec<u128> = Vec::with_capacity(N);
    for &x in &edges {
        for &y in &edges {
            for &z in &edges {
                a.push(x);
                b.push(y);
                c.push(z);
            }
        }
    }
    while a.len() < N {
        a.push(rng.u128());
        b.push(rng.u128());
        c.push(rng.u128());
    }

    let t1 = now_ms();
    let (a_bytes, b_bytes, c_bytes) = (to_le_bytes(&a), to_le_bytes(&b), to_le_bytes(&c));
    let mut out = vec![0u8; N * 16];
    let handle = mailbox::call(OP_CREATE_BUFFER, &[PERSISTENT_BYTES], &[])?[0];
    mailbox::call(OP_UPLOAD, &[handle, 0], &[Region::upload(&a_bytes)])?;
    let workgroups = [(N as u32).div_ceil(WORKGROUP_SIZE), 1, 1];
    let run = mailbox::call(
        OP_RUN,
        &run_args(
            SHADER_FP128_OPS,
            workgroups,
            &[N as u32, FP128_OP_MULADD],
            4,
        ),
        &[
            Region::handle(handle),
            Region::upload(&b_bytes),
            Region::upload(&c_bytes),
            Region::readback(&mut out),
        ],
    );
    let destroy = mailbox::call(OP_DESTROY, &[handle], &[]);
    if let Err(e) = run.and(destroy) {
        if mailbox::is_dead() {
            // The proxy may still write `out`; leaking it keeps that write harmless.
            std::mem::forget(out);
        }
        return Err(e);
    }
    let selftest_ms = now_ms() - t1;

    let mut mismatches = 0u32;
    for i in 0..N {
        let expected = AkitaField::from_u128_reduced(a[i]) * AkitaField::from_u128_reduced(b[i])
            + AkitaField::from_u128_reduced(c[i]);
        let got = u128::from_le_bytes(out[i * 16..i * 16 + 16].try_into().unwrap());
        if got != expected.to_canonical_u128() {
            mismatches += 1;
        }
    }

    Ok(GpuReport {
        status: if mismatches == 0 {
            "ok".into()
        } else {
            format!("error: selftest {mismatches} mismatches of {N}")
        },
        selftest_ms,
        selftest_mismatches: mismatches,
        roundtrip_us,
        commit: String::new(),
        stage2: String::new(),
        digit_range: String::new(),
    })
}
