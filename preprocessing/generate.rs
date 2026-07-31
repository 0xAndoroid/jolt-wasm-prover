//! Compiles the guests and writes browser-served preprocessing artifacts:
//! `{name}_prover.bin` — the Dory SRS (`ArkworksProverSetup`, arkworks
//! uncompressed for fast wasm deserialization), `{name}_verifier.bin` — the
//! modular `JoltVerifierPreprocessing` (bincode/serde), and `{name}.elf`.
//! Prover- and verifier-side generators come from the same legacy
//! `JoltProverPreprocessing`, so the SRS pair is consistent by construction.

use std::path::{Path, PathBuf};

type LegacyProverPrep = jolt::JoltProverPreprocessing<jolt::F, jolt::Curve, jolt::PCS>;

fn emit(
    name: &str,
    program: jolt::host::Program,
    shared: jolt::JoltSharedPreprocessing,
    public_dir: &Path,
) {
    println!("[{name}] Generating prover preprocessing (SRS)...");
    let prover_prep = LegacyProverPrep::new(shared);

    println!("[{name}] Deriving modular verifier preprocessing...");
    let verifier_prep: jolt::JoltVerifierPreprocessing =
        jolt::jolt_prover_legacy::zkvm::proof::verifier_preprocessing_from_prover(&prover_prep);

    let srs_bytes = serialize_uncompressed(&prover_prep.generators);
    write_file(
        public_dir,
        &format!("{name}_prover.bin"),
        name,
        "SRS",
        &srs_bytes,
    );

    let verifier_bytes = bincode::serde::encode_to_vec(&verifier_prep, bincode::config::standard())
        .expect("verifier preprocessing encode");
    write_file(
        public_dir,
        &format!("{name}_verifier.bin"),
        name,
        "Verifier",
        &verifier_bytes,
    );

    let elf_contents = program.get_elf_contents().expect("ELF contents");
    let elf_path = public_dir.join(format!("{name}.elf"));
    std::fs::write(&elf_path, &elf_contents).expect("write ELF");
    println!("[{name}] ELF: {} bytes -> {elf_path:?}", elf_contents.len());
}

fn serialize_uncompressed<T: ark_serialize::CanonicalSerialize>(value: &T) -> Vec<u8> {
    let mut buf = Vec::with_capacity(value.serialized_size(ark_serialize::Compress::No));
    value.serialize_uncompressed(&mut buf).expect("serialize");
    buf
}

fn write_file(public_dir: &Path, filename: &str, program: &str, kind: &str, bytes: &[u8]) {
    let path = public_dir.join(filename);
    std::fs::write(&path, bytes).expect("write file");
    println!("[{program}] {kind}: {} bytes -> {path:?}", bytes.len());
}

// Inline registration is inventory-based (link-time); keeping the crates
// linked is all the tracer needs.
use jolt_inlines_keccak256 as _;
use jolt_inlines_secp256k1 as _;
use jolt_inlines_sha2 as _;

fn main() {
    let public_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("frontend/public");
    std::fs::create_dir_all(&public_dir).expect("create frontend/public");

    {
        let target_dir = "/tmp/jolt-wasm-sha2-guest";
        println!("[sha2] Compiling guest...");
        let mut program = sha2_guest::compile_sha2(target_dir);
        let shared = sha2_guest::preprocess_shared_sha2(&mut program).expect("preprocess");
        emit("sha2", program, shared, &public_dir);
    }
    {
        let target_dir = "/tmp/jolt-wasm-ecdsa-guest";
        println!("[ecdsa] Compiling guest...");
        let mut program = secp256k1_ecdsa_verify_guest::compile_secp256k1_ecdsa_verify(target_dir);
        let shared =
            secp256k1_ecdsa_verify_guest::preprocess_shared_secp256k1_ecdsa_verify(&mut program)
                .expect("preprocess");
        emit("ecdsa", program, shared, &public_dir);
    }
    {
        let target_dir = "/tmp/jolt-wasm-keccak-guest";
        println!("[keccak] Compiling guest...");
        let mut program = sha3_chain_guest::compile_sha3_chain(target_dir);
        let shared =
            sha3_chain_guest::preprocess_shared_sha3_chain(&mut program).expect("preprocess");
        emit("keccak", program, shared, &public_dir);
    }
    {
        let target_dir = "/tmp/jolt-wasm-sha2-chain-guest";
        println!("[sha2_chain] Compiling guest...");
        let mut program = sha2_chain_guest::compile_sha2_chain(target_dir);
        let shared =
            sha2_chain_guest::preprocess_shared_sha2_chain(&mut program).expect("preprocess");
        emit("sha2_chain", program, shared, &public_dir);
    }

    println!("Done!");
}
