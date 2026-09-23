//! Validates the exact browser path natively: loads the artifacts written by
//! `generate-preprocessing` from `frontend/public/`, then runs the Akita
//! prover (`JoltAkitaBackend::optimized()`) and verifier from those bytes,
//! including the per-proof setup derivation the browser performs.

use std::path::{Path, PathBuf};
use std::time::Instant;

// The lib is cdylib-only (rlib + `-C lto=fat` can't coexist for the wasm
// build), so the shared engine is included at the source level.
#[path = "../src/engine.rs"]
mod engine;

fn load(dir: &Path, name: &str) -> Vec<u8> {
    let path = dir.join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest;
    sha2::Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn roundtrip(dir: &Path, schedules: &[u8], name: &str, inputs: &[u8]) {
    println!("[{name}] loading artifacts...");
    let program_bytes = load(dir, &format!("{name}_program.bin"));
    let elf = load(dir, &format!("{name}.elf"));

    let start = Instant::now();
    let ctx = engine::ProverContext::new(schedules, &program_bytes, &elf).expect("prover context");
    println!(
        "[{name}] artifacts decoded in {:.2}s",
        start.elapsed().as_secs_f64()
    );

    let out = engine::prove(&ctx, inputs).expect("prove");
    let t = out.timings;
    let prove_secs = t.prove_ms / 1000.0;
    println!(
        "[{name}] trace {:.2}s, setup {:.2}s, prove {prove_secs:.2}s ({} cycles, padded {}, {:.1} kHz padded, proof {} bytes, verifier preprocessing {} bytes)",
        t.trace_ms / 1000.0,
        t.setup_ms / 1000.0,
        out.unpadded_cycles,
        out.padded_cycles,
        out.padded_cycles as f64 / prove_secs / 1000.0,
        out.proof_bytes.len(),
        out.verifier_preprocessing_bytes.len(),
    );

    println!(
        "[{name}] sha256 proof {} verifier-preprocessing {}",
        sha256_hex(&out.proof_bytes),
        sha256_hex(&out.verifier_preprocessing_bytes)
    );

    let verifier_prep = engine::decode_verifier_preprocessing(&out.verifier_preprocessing_bytes)
        .expect("verifier prep");
    let start = Instant::now();
    engine::verify(&verifier_prep, &out.proof_bytes, &out.io_bytes).expect("verify");
    println!("[{name}] verified in {:.3}s", start.elapsed().as_secs_f64());
}

// Inline registration is inventory-based (link-time); keeping the crates
// linked is all the tracer needs.
use jolt_inlines_keccak256 as _;
use jolt_inlines_secp256k1 as _;
use jolt_inlines_sha2 as _;

fn main() {
    #[cfg(feature = "trace-commit-device")]
    engine::install_trace_commit_device_from_env().expect("trace commit device");
    if std::env::var_os("RUST_LOG").is_some() {
        use tracing_subscriber::fmt::format::FmtSpan;
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .with_span_events(FmtSpan::CLOSE)
            .with_writer(std::io::stderr)
            .init();
    }
    let public_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("frontend/public");
    let schedules = load(&public_dir, "akita_schedules.bin");

    let sha2_input: &[u8] = b"jolt wasm prover roundtrip test input";
    let inputs = postcard::to_allocvec(&sha2_input).expect("serialize");
    roundtrip(&public_dir, &schedules, "sha2", &inputs);

    // Fixed valid signature over SHA-256("hello world"); limbs are
    // little-endian u64 (limb 0 least significant), Q = (x limbs 0..4, y 4..8).
    // Input bytes match `WasmProver::prove_ecdsa`.
    let z: [u64; 4] = [
        0x9088f7ace2efcde9,
        0xc484efe37a5380ee,
        0xa52e52d7da7dabfa,
        0xb94d27b9934d3e08,
    ];
    let r: [u64; 4] = [
        0xb8fc413b4b967ed8,
        0x248d4b0b2829ab00,
        0x587f69296af3cd88,
        0x3a5d6a386e6cf7c0,
    ];
    let s: [u64; 4] = [
        0x66a82f274e3dcafc,
        0x299a02486be40321,
        0x6212d714118f617e,
        0x9d452f63cf91018d,
    ];
    let q: [u64; 8] = [
        0x0012563f32ed0216,
        0xee00716af6a73670,
        0x91fc70e34e00e6c8,
        0xeeb6be8b9e68868b,
        0x4780de3d5fda972d,
        0xcb1b42d72491e47f,
        0xdc7f31262e4ba2b7,
        0xdc7b004d3bb2800d,
    ];
    let mut inputs = postcard::to_allocvec(&z).expect("serialize");
    inputs.extend_from_slice(&postcard::to_allocvec(&r).expect("serialize"));
    inputs.extend_from_slice(&postcard::to_allocvec(&s).expect("serialize"));
    inputs.extend_from_slice(&postcard::to_allocvec(&q).expect("serialize"));
    roundtrip(&public_dir, &schedules, "ecdsa", &inputs);

    let mut inputs = postcard::to_allocvec(&[5u8; 32]).expect("serialize");
    // 17 → 2^16, 69 → 2^18 (default 100), 278 → 2^20, 556 → 2^21 padded cycles.
    let iters =
        std::env::var("SHA2_CHAIN_ITERS").map_or(100u32, |v| v.parse().expect("SHA2_CHAIN_ITERS"));
    inputs.extend_from_slice(&postcard::to_allocvec(&iters).expect("serialize"));
    roundtrip(&public_dir, &schedules, "sha2_chain", &inputs);

    println!("All roundtrips passed!");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A malformed input (an overlong postcard varint) makes the sha2
    /// guest's argument decode `unwrap` → `jolt_panic`. The proof of
    /// that execution verifies cryptographically (the panic flag is public
    /// IO), so the engine must reject it explicitly on both ends.
    #[test]
    fn rejects_panicked_guest_execution() {
        let public_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("frontend/public");
        let ctx = engine::ProverContext::new(
            &load(&public_dir, "akita_schedules.bin"),
            &load(&public_dir, "sha2_program.bin"),
            &load(&public_dir, "sha2.elf"),
        )
        .expect("prover context");
        let inputs = [0xff; 11];

        let out = engine::prove_unchecked(&ctx, &inputs).expect("prove");
        assert!(out.panicked, "malformed input must panic the guest");
        let prep = engine::decode_verifier_preprocessing(&out.verifier_preprocessing_bytes)
            .expect("verifier prep");
        let err = engine::verify(&prep, &out.proof_bytes, &out.io_bytes)
            .expect_err("panicked execution must not verify");
        assert!(err.contains("panicked"), "{err}");

        let err = engine::prove(&ctx, &inputs)
            .err()
            .expect("prove must refuse a panicked trace");
        assert!(err.contains("panicked"), "{err}");
    }
}
