//! Modular-prover engine shared by the wasm bindings and the native
//! roundtrip test: deserializes shipped preprocessing, traces the guest ELF,
//! and runs `jolt_prover::prove` / `jolt_verifier::verify`.

use std::io::Cursor;

use common::jolt_device::{JoltDevice, MemoryConfig};
use dory::backends::arkworks::ArkworksProverSetup;
use jolt_crypto::{Bn254G1, Pedersen};
use jolt_dory::{DoryProverSetup, DoryScheme};
use jolt_field::Fr;
use jolt_program::execution::{
    ExecutionBackend, JoltProgram, OwnedTrace, TraceInputs, TraceOutput, TraceRow,
};
use jolt_prover::{JoltBackend, JoltProverPreprocessing, ProverConfig};
use jolt_transcript::LegacyBlake2bTranscript;
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
use tracer::execution_backend::TracerBackend;

pub type Pcs = DoryScheme;
pub type Vc = Pedersen<Bn254G1>;
pub type Transcript = LegacyBlake2bTranscript<Fr>;
pub type VerifierPrep = jolt_verifier::JoltVerifierPreprocessing<Pcs, Vc>;
pub type ProverPrep = JoltProverPreprocessing<Pcs, Vc>;
pub type Proof = jolt_verifier::JoltProof<Pcs, Vc>;

/// Uncompressed, unvalidated on purpose: the SRS ships with the app and is
/// megabytes of curve points — validation would dominate startup time.
pub fn load_prover_setup(bytes: &[u8]) -> Result<DoryProverSetup, String> {
    use ark_serialize::CanonicalDeserialize;
    let inner = ArkworksProverSetup::deserialize_with_mode(
        &mut Cursor::new(bytes),
        ark_serialize::Compress::No,
        ark_serialize::Validate::No,
    )
    .map_err(|e| format!("SRS deserialize error: {e}"))?;
    Ok(DoryProverSetup(inner))
}

pub fn decode_verifier_preprocessing(bytes: &[u8]) -> Result<VerifierPrep, String> {
    let (prep, consumed) = bincode::serde::decode_from_slice(bytes, bincode::config::standard())
        .map_err(|e| format!("verifier preprocessing decode error: {e}"))?;
    if consumed != bytes.len() {
        return Err("trailing bytes in verifier preprocessing".to_string());
    }
    Ok(prep)
}

pub fn build_prover_preprocessing(
    srs_bytes: &[u8],
    verifier_prep_bytes: &[u8],
) -> Result<ProverPrep, String> {
    Ok(ProverPrep {
        verifier: decode_verifier_preprocessing(verifier_prep_bytes)?,
        pcs_setup: load_prover_setup(srs_bytes)?,
        committed_program: None,
    })
}

pub struct ProveOutput {
    pub proof_bytes: Vec<u8>,
    pub io_bytes: Vec<u8>,
    pub unpadded_cycles: usize,
    pub padded_cycles: usize,
}

#[tracing::instrument(skip_all, name = "engine::prove")]
pub fn prove(prep: &ProverPrep, elf: &[u8], inputs: &[u8]) -> Result<ProveOutput, String> {
    let program = JoltProgram::from_elf_bytes(elf.to_vec());
    let program_preprocessing = prep
        .verifier
        .program
        .as_full()
        .ok_or("full (non-committed) program preprocessing required")?;
    let layout = prep.verifier.program.memory_layout().clone();
    let memory_config = MemoryConfig {
        max_untrusted_advice_size: layout.max_untrusted_advice_size,
        max_trusted_advice_size: layout.max_trusted_advice_size,
        max_input_size: layout.max_input_size,
        max_output_size: layout.max_output_size,
        stack_size: layout.stack_size,
        heap_size: layout.heap_size,
        program_size: Some(layout.program_size),
    };

    let trace_output = TracerBackend::new()
        .trace(
            &program,
            TraceInputs {
                inputs: inputs.to_vec(),
                untrusted_advice: Vec::new(),
                trusted_advice: Vec::new(),
                memory_config,
            },
        )
        .map_err(|e| format!("trace error: {e:?}"))?;
    let unpadded_cycles = trace_output.trace.rows().len();

    let config = ProverConfig::derive::<Fr>(
        trace_output.trace.rows(),
        &layout,
        prep.verifier.program.min_bytecode_address(),
        prep.verifier.program.program_image_len_words(),
        prep.verifier.program.max_padded_trace_length(),
    )
    .map_err(|e| format!("config derive error: {e}"))?;

    let public_io = trace_output.device.clone();
    let padded = pad_trace(trace_output, config.trace_length);

    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(
            config.trace_length.ilog2() as usize,
            config.ram_K,
            config.one_hot_config,
        ),
        JoltVmWitnessInputs::new(&program, program_preprocessing, padded),
    );

    let backend = build_backend();
    let proof = jolt_prover::prove::<Fr, Pcs, Vc, Transcript, _>(
        &backend, prep, &config, None, &witness, &public_io,
    )
    .map_err(|e| format!("prove error: {e}"))?;

    let proof_bytes = bincode::serde::encode_to_vec(&proof, bincode::config::standard())
        .map_err(|e| format!("proof encode error: {e}"))?;
    let io_bytes = bincode::serde::encode_to_vec(&public_io, bincode::config::standard())
        .map_err(|e| format!("program io encode error: {e}"))?;

    Ok(ProveOutput {
        proof_bytes,
        io_bytes,
        unpadded_cycles,
        padded_cycles: config.trace_length,
    })
}

/// The webgpu arm when its engine was explicitly brought up (browser:
/// `webgpu_warmup` after the GPU-worker handshake; native: `JOLT_WEBGPU=1`
/// on the test binaries), the optimized arm otherwise. Fail-closed: any
/// webgpu construction error degrades to the optimized arm, and with no
/// warmup this is byte-for-byte the Phase-1 code path.
fn build_backend() -> JoltBackend<Fr, Pcs> {
    #[cfg(not(target_arch = "wasm32"))]
    if std::env::var("JOLT_WEBGPU").is_ok_and(|v| !v.is_empty() && v != "0") {
        if let Err(error) = jolt_kernels::webgpu::warmup() {
            tracing::warn!(%error, "JOLT_WEBGPU=1 but warmup failed");
        }
    }
    if jolt_kernels::webgpu::WebGpuEngine::get().is_some() {
        match JoltBackend::<Fr, Pcs>::webgpu() {
            Ok(backend) => {
                tracing::info!("proving with the webgpu arm");
                #[cfg(not(target_arch = "wasm32"))]
                eprintln!("[engine] webgpu arm active");
                return backend;
            }
            Err(error) => tracing::warn!(%error, "webgpu arm unavailable; optimized arm"),
        }
    }
    JoltBackend::<Fr, Pcs>::optimized()
}

#[tracing::instrument(skip_all, name = "engine::verify")]
pub fn verify(prep: &VerifierPrep, proof_bytes: &[u8], io_bytes: &[u8]) -> Result<(), String> {
    let (proof, consumed): (Proof, usize) =
        bincode::serde::decode_from_slice(proof_bytes, bincode::config::standard())
            .map_err(|e| format!("proof decode error: {e}"))?;
    if consumed != proof_bytes.len() {
        return Err("trailing bytes in proof".to_string());
    }
    let (public_io, consumed): (JoltDevice, usize) =
        bincode::serde::decode_from_slice(io_bytes, bincode::config::standard())
            .map_err(|e| format!("program io decode error: {e}"))?;
    if consumed != io_bytes.len() {
        return Err("trailing bytes in program io".to_string());
    }

    jolt_verifier::verify::<Fr, Pcs, Vc, Transcript>(prep, &public_io, &proof, None)
        .map_err(|e| format!("verification failed: {e}"))
}

/// Pad to the padded trace length with no-op rows. Exact capacity up front to
/// avoid an amortized-growth realloc of the whole trace.
fn pad_trace(
    trace_output: TraceOutput<OwnedTrace>,
    trace_length: usize,
) -> TraceOutput<OwnedTrace> {
    let source = trace_output.trace.rows();
    let mut rows = Vec::with_capacity(trace_length.max(source.len()));
    rows.extend_from_slice(source);
    rows.resize(trace_length, TraceRow::default());
    TraceOutput::new(
        OwnedTrace::new(rows),
        trace_output.device,
        trace_output.final_memory,
    )
}
