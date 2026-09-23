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
    let public_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("frontend/public");
    let schedules = load(&public_dir, "akita_schedules.bin");

    let sha2_input: &[u8] = b"jolt wasm prover roundtrip test input";
    let inputs = postcard::to_allocvec(&sha2_input).expect("serialize");
    roundtrip(&public_dir, &schedules, "sha2", &inputs);

    let mut inputs = postcard::to_allocvec(&[5u8; 32]).expect("serialize");
    inputs.extend_from_slice(&postcard::to_allocvec(&100u32).expect("serialize"));
    roundtrip(&public_dir, &schedules, "sha2_chain", &inputs);

    println!("All roundtrips passed!");
}
