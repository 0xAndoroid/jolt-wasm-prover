//! Validates the exact browser path natively: loads the artifacts written by
//! `generate-preprocessing` from `frontend/public/`, then runs the Akita
//! prover (`JoltAkitaBackend::optimized()`) and verifier from those bytes,
//! including the per-proof setup derivation the browser performs.

#![expect(clippy::print_stdout, reason = "CLI output")]

use sha2::{Digest, Sha256};
use std::error::Error;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

// The lib is cdylib-only (rlib + `-C lto=fat` can't coexist for the wasm
// build), so the shared engine is included at the source level.
#[path = "../src/engine.rs"]
mod engine;

fn load(dir: &Path, name: &str) -> io::Result<Vec<u8>> {
    let path = dir.join(name);
    std::fs::read(&path)
        .map_err(|e| io::Error::new(e.kind(), format!("read {}: {e}", path.display())))
}

fn sha256_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    Sha256::digest(bytes)
        .iter()
        .flat_map(|byte| {
            [
                char::from(HEX[usize::from(byte >> 4)]),
                char::from(HEX[usize::from(byte & 15)]),
            ]
        })
        .collect()
}

fn roundtrip(
    dir: &Path,
    schedules: &[u8],
    name: &str,
    inputs: &[u8],
) -> Result<(), Box<dyn Error>> {
    println!("[{name}] loading artifacts...");
    let program_bytes = load(dir, &format!("{name}_program.bin"))?;
    let elf = load(dir, &format!("{name}.elf"))?;

    let start = Instant::now();
    let ctx = engine::ProverContext::new(schedules, &program_bytes, &elf)?;
    println!(
        "[{name}] artifacts decoded in {:.2}s",
        start.elapsed().as_secs_f64()
    );

    let out = engine::prove(&ctx, inputs)?;
    let t = out.timings;
    let prove_secs = t.prove / 1000.0;
    println!(
        "[{name}] trace {:.2}s, setup {:.2}s, prove {prove_secs:.2}s ({} cycles, padded {}, {:.1} kHz padded, proof {} bytes, verifier preprocessing {} bytes)",
        t.trace / 1000.0,
        t.setup / 1000.0,
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

    if let Some(dir) = std::env::var_os("JOLT_ROUNDTRIP_DUMP_DIR") {
        let dir = Path::new(&dir);
        for (suffix, bytes) in [
            ("proof", &out.proof_bytes),
            ("io", &out.io_bytes),
            ("verifier_preprocessing", &out.verifier_preprocessing_bytes),
        ] {
            std::fs::write(dir.join(format!("{name}_{suffix}.bin")), bytes)?;
        }
    }

    let verifier_prep = engine::decode_verifier_preprocessing(&out.verifier_preprocessing_bytes)?;
    let start = Instant::now();
    engine::verify(&verifier_prep, &out.proof_bytes, &out.io_bytes)?;
    println!("[{name}] verified in {:.3}s", start.elapsed().as_secs_f64());

    if let Some(dir) = std::env::var_os("JOLT_ROUNDTRIP_VERIFY_DIR") {
        cross_verify(Path::new(&dir), name, &out)?;
    }
    Ok(())
}

/// Compares `{name}_{proof,io,verifier_preprocessing}.bin` from `dir` (a
/// browser proof dumped by `bench/dump_browser_proof.py`) with the native
/// bytes and runs it through the native verifier.
fn cross_verify(
    dir: &Path,
    name: &str,
    native: &engine::ProveOutput,
) -> Result<(), Box<dyn Error>> {
    let proof = load(dir, &format!("{name}_proof.bin"))?;
    let io = load(dir, &format!("{name}_io.bin"))?;
    let prep_bytes = load(dir, &format!("{name}_verifier_preprocessing.bin"))?;
    let same = |a: &[u8], b: &[u8]| if a == b { "identical" } else { "DIFFERENT" };
    println!(
        "[{name}] foreign proof {} bytes: proof {}, io {}, verifier preprocessing {}",
        proof.len(),
        same(&proof, &native.proof_bytes),
        same(&io, &native.io_bytes),
        same(&prep_bytes, &native.verifier_preprocessing_bytes),
    );
    let prep = engine::decode_verifier_preprocessing(&prep_bytes)?;
    engine::verify(&prep, &proof, &io)?;
    println!("[{name}] foreign proof verified natively");
    Ok(())
}

// Inline registration is inventory-based (link-time); keeping the crates
// linked is all the tracer needs.
use jolt_inlines_keccak256 as _;
use jolt_inlines_secp256k1 as _;
use jolt_inlines_sha2 as _;

fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(feature = "trace-commit-device")]
    engine::install_trace_commit_device_from_env()?;
    #[cfg(feature = "relation-range-device")]
    engine::install_relation_range_device_from_env()?;
    if std::env::var_os("RUST_LOG").is_some() {
        use tracing_subscriber::fmt::format::FmtSpan;
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .with_span_events(FmtSpan::CLOSE)
            .with_writer(std::io::stderr)
            .init();
    }
    let public_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("frontend/public");
    let schedules = load(&public_dir, "akita_schedules.bin")?;

    let sha2_input: &[u8] = b"jolt wasm prover roundtrip test input";
    let inputs = postcard::to_allocvec(&sha2_input)?;
    roundtrip(&public_dir, &schedules, "sha2", &inputs)?;

    // Fixed valid signature over SHA-256("hello world"); limbs are
    // little-endian u64 (limb 0 least significant), Q = (x limbs 0..4, y 4..8).
    // Input bytes match `WasmProver::prove_ecdsa`.
    let z: [u64; 4] = [
        0x9088_f7ac_e2ef_cde9,
        0xc484_efe3_7a53_80ee,
        0xa52e_52d7_da7d_abfa,
        0xb94d_27b9_934d_3e08,
    ];
    let r: [u64; 4] = [
        0xb8fc_413b_4b96_7ed8,
        0x248d_4b0b_2829_ab00,
        0x587f_6929_6af3_cd88,
        0x3a5d_6a38_6e6c_f7c0,
    ];
    let s: [u64; 4] = [
        0x66a8_2f27_4e3d_cafc,
        0x299a_0248_6be4_0321,
        0x6212_d714_118f_617e,
        0x9d45_2f63_cf91_018d,
    ];
    let q: [u64; 8] = [
        0x0012_563f_32ed_0216,
        0xee00_716a_f6a7_3670,
        0x91fc_70e3_4e00_e6c8,
        0xeeb6_be8b_9e68_868b,
        0x4780_de3d_5fda_972d,
        0xcb1b_42d7_2491_e47f,
        0xdc7f_3126_2e4b_a2b7,
        0xdc7b_004d_3bb2_800d,
    ];
    let mut inputs = postcard::to_allocvec(&z)?;
    inputs.extend_from_slice(&postcard::to_allocvec(&r)?);
    inputs.extend_from_slice(&postcard::to_allocvec(&s)?);
    inputs.extend_from_slice(&postcard::to_allocvec(&q)?);
    roundtrip(&public_dir, &schedules, "ecdsa", &inputs)?;

    let mut inputs = postcard::to_allocvec(&[5u8; 32])?;
    let iters = std::env::var("SHA2_CHAIN_ITERS").map_or(Ok(100u32), |v| v.parse())?;
    inputs.extend_from_slice(&postcard::to_allocvec(&iters)?);
    roundtrip(&public_dir, &schedules, "sha2_chain", &inputs)?;

    println!("All roundtrips passed!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha2_ctx() -> Result<engine::ProverContext, Box<dyn Error>> {
        let public_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("frontend/public");
        Ok(engine::ProverContext::new(
            &load(&public_dir, "akita_schedules.bin")?,
            &load(&public_dir, "sha2_program.bin")?,
            &load(&public_dir, "sha2.elf")?,
        )?)
    }

    /// A malformed input (an overlong postcard varint) makes the sha2
    /// guest's argument decode `unwrap` → `jolt_panic`. The proof of
    /// that execution verifies cryptographically (the panic flag is public
    /// IO), so the engine must reject it explicitly on both ends.
    #[test]
    fn rejects_panicked_guest_execution() -> Result<(), Box<dyn Error>> {
        let ctx = sha2_ctx()?;
        let inputs = [0xff; 11];

        let out = engine::prove_inner(&ctx, &inputs, false)?;
        let prep = engine::decode_verifier_preprocessing(&out.verifier_preprocessing_bytes)?;
        let err = engine::verify(&prep, &out.proof_bytes, &out.io_bytes)
            .err()
            .ok_or("panicked execution must not verify")?;
        assert!(err.contains("panicked"), "{err}");

        let err = engine::prove(&ctx, &inputs)
            .err()
            .ok_or("prove must refuse a panicked trace")?;
        assert!(err.contains("panicked"), "{err}");
        Ok(())
    }

    /// Browser (wasm32, GPU off) proof of the sha2 roundtrip input, dumped
    /// with `bench/dump_browser_proof.py` (WebKit and Chromium agree). Native
    /// and wasm32 proofs are byte-identical since spongefish 4ee5f2b2
    /// (a16z/jolt #1924: the Blake2b512 sponge behind `AkitaTranscript`
    /// hashed its squeeze counters at pointer width, so the batched opening
    /// diverged across targets and the native verifier rejected browser
    /// proofs). `bench/dump_browser_proof.py` prints the digest; regenerate
    /// if the transcript or the sha2 artifacts change.
    const WASM_SHA2_PROOF_SHA256: &str =
        "afe5b6cced775f1663e7c047ec27ccd0a1593cd9d8d717427018ec2312f5ed1d";

    #[test]
    fn sha2_proof_matches_browser_digest() -> Result<(), Box<dyn Error>> {
        let input: &[u8] = b"jolt wasm prover roundtrip test input";
        let out = engine::prove(&sha2_ctx()?, &postcard::to_allocvec(&input)?)?;
        assert_eq!(sha256_hex(&out.proof_bytes), WASM_SHA2_PROOF_SHA256);
        Ok(())
    }
}
