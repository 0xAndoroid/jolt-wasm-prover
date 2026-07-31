//! Validates the exact browser path natively: loads the artifacts written by
//! `generate-preprocessing` from `frontend/public/`, then runs the modular
//! prover (`JoltBackend::optimized()`) and verifier from those bytes.

use std::path::PathBuf;
use std::time::Instant;

// The lib is cdylib-only (rlib + `-C lto=fat` can't coexist for the wasm
// build), so the shared engine is included at the source level.
#[path = "../src/engine.rs"]
mod engine;

fn load(dir: &PathBuf, name: &str) -> Vec<u8> {
    let path = dir.join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"))
}

fn roundtrip(dir: &PathBuf, name: &str, inputs: &[u8]) {
    println!("[{name}] loading artifacts...");
    let srs = load(dir, &format!("{name}_prover.bin"));
    let verifier_bytes = load(dir, &format!("{name}_verifier.bin"));
    let elf = load(dir, &format!("{name}.elf"));

    let start = Instant::now();
    let prep = engine::build_prover_preprocessing(&srs, &verifier_bytes).expect("prover prep");
    println!(
        "[{name}] prover preprocessing deserialized in {:.2}s",
        start.elapsed().as_secs_f64()
    );

    let start = Instant::now();
    let out = engine::prove(&prep, &elf, inputs).expect("prove");
    let prove_secs = start.elapsed().as_secs_f64();
    println!(
        "[{name}] proved in {prove_secs:.2}s ({} cycles, padded {}, {:.1} kHz padded, proof {} bytes)",
        out.unpadded_cycles,
        out.padded_cycles,
        out.padded_cycles as f64 / prove_secs / 1000.0,
        out.proof_bytes.len(),
    );

    let verifier_prep =
        engine::decode_verifier_preprocessing(&verifier_bytes).expect("verifier prep");
    let start = Instant::now();
    engine::verify(&verifier_prep, &out.proof_bytes, &out.io_bytes).expect("verify");
    println!("[{name}] verified in {:.3}s", start.elapsed().as_secs_f64());
}

// Inline registration is inventory-based (link-time); keeping the crates
// linked is all the tracer needs.
use jolt_inlines_keccak256 as _;
use jolt_inlines_secp256k1 as _;
use jolt_inlines_sha2 as _;

fn run_all() {
    let public_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("frontend/public");

    let sha2_input: &[u8] = b"jolt wasm prover roundtrip test input";
    let inputs = postcard::to_allocvec(&sha2_input).expect("serialize");
    roundtrip(&public_dir, "sha2", &inputs);

    let mut inputs = postcard::to_allocvec(&[5u8; 32]).expect("serialize");
    inputs.extend_from_slice(&postcard::to_allocvec(&100u32).expect("serialize"));
    roundtrip(&public_dir, "sha2_chain", &inputs);

    println!("All roundtrips passed!");
}

fn main() {
    // BlindFold verification (and the prover's replay of it) recurses over a
    // large folded R1CS — run on a dedicated wide stack like jolt's own ZK
    // e2e suite does.
    std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(run_all)
        .expect("spawn roundtrip thread")
        .join()
        .expect("roundtrip thread panicked");
}
