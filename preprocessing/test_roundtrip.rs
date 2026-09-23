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
    #[cfg(feature = "relation-range-device")]
    engine::install_relation_range_device_from_env().expect("relation range device");
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

    let mut inputs = postcard::to_allocvec(&[5u8; 32]).expect("serialize");
    // 17 → 2^16, 69 → 2^18 (default 100), 278 → 2^20, 556 → 2^21 padded cycles.
    let iters =
        std::env::var("SHA2_CHAIN_ITERS").map_or(100u32, |v| v.parse().expect("SHA2_CHAIN_ITERS"));
    inputs.extend_from_slice(&postcard::to_allocvec(&iters).expect("serialize"));
    roundtrip(&public_dir, &schedules, "sha2_chain", &inputs);

    println!("All roundtrips passed!");
}
