//! Modular-prover engine shared by the wasm bindings and the native
//! roundtrip test, on the packed Akita (lattice) protocol: deserializes the
//! shipped program preprocessing and schedule catalogs, traces the guest
//! ELF, builds the shape-exact Akita setup for that trace, and runs
//! `jolt_prover::prove` / `jolt_verifier::verify`.
//!
//! Compiled with `jolt-prover/akita`. The Akita commitment setup is exact in
//! the proof shape (padded trace length, RAM size, bytecode size) and its
//! prover half is not serializable, so unlike the Dory SRS it cannot ship as
//! a static artifact: every prove derives the config from the trace and runs
//! `preprocess_full` in-process (the "setup" phase). The verifier
//! preprocessing that comes out of the same call is the only thing a
//! verifier needs, and `prove` returns it serialized alongside the proof.

use std::sync::Arc;

use common::jolt_device::{JoltDevice, MemoryConfig};
use jolt_akita::{AkitaField, AkitaScheduleArtifacts, AkitaScheme};
use jolt_program::execution::{JoltProgram, OwnedTrace, TraceInputs};
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_prover::akita::preprocessing::{self as akita_preprocessing, AkitaTranscript, AkitaVc};
use jolt_prover::akita::JoltAkitaBackend;
use jolt_prover::ProverConfig;
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
use tracer::execution_backend::TracerBackend;

#[cfg(feature = "relation-range-device")]
#[path = "relation_range_reference.rs"]
pub mod relation_range_reference;
#[cfg(feature = "trace-commit-device")]
#[path = "trace_commit_reference.rs"]
pub mod trace_commit_reference;

pub type F = AkitaField;
pub type Pcs = AkitaScheme;
pub type Vc = AkitaVc;
pub type Transcript = AkitaTranscript;
pub type VerifierPrep = jolt_verifier::JoltVerifierPreprocessing<Pcs, Vc>;
pub type ProverPrep = jolt_prover::JoltProverPreprocessing<Pcs, Vc>;
pub type Proof = jolt_verifier::JoltProof<Pcs, Vc>;

fn decode<T: serde::de::DeserializeOwned>(bytes: &[u8], what: &str) -> Result<T, String> {
    let (value, consumed) = bincode::serde::decode_from_slice(bytes, bincode::config::standard())
        .map_err(|e| format!("{what} decode error: {e}"))?;
    if consumed != bytes.len() {
        return Err(format!("trailing bytes in {what}"));
    }
    Ok(value)
}

fn encode<T: serde::Serialize>(value: &T, what: &str) -> Result<Vec<u8>, String> {
    bincode::serde::encode_to_vec(value, bincode::config::standard())
        .map_err(|e| format!("{what} encode error: {e}"))
}

/// The three base schedule catalogs (`jolt-akita/schedules/*.aks`) bundled
/// into one bincode blob by `generate-preprocessing`.
pub fn decode_schedule_artifacts(bytes: &[u8]) -> Result<Arc<AkitaScheduleArtifacts>, String> {
    decode(bytes, "Akita schedule artifacts").map(Arc::new)
}

pub fn decode_program_preprocessing(bytes: &[u8]) -> Result<JoltProgramPreprocessing, String> {
    decode(bytes, "program preprocessing")
}

pub fn decode_verifier_preprocessing(bytes: &[u8]) -> Result<VerifierPrep, String> {
    decode(bytes, "verifier preprocessing")
}

/// Routes every later stage-0 trace commit in this process through `device`
/// (`jolt_akita::set_trace_commit_device`); the CPU kernels remain the
/// fallback for shapes the device declines.
#[cfg(feature = "trace-commit-device")]
pub fn install_trace_commit_device(
    device: Arc<dyn jolt_akita::TraceCommitDevice>,
) -> Result<(), String> {
    jolt_akita::set_trace_commit_device(device).map_err(|e| format!("trace commit device: {e}"))
}

/// Routes the y-phase rounds of every later stage-1 digit-range sumcheck
/// through `device` (`akita_prover::set_digit_range_device`); instances the
/// device declines stay on the CPU prover. Only the browser installs one.
#[cfg(all(feature = "digit-range-device", target_arch = "wasm32"))]
pub fn install_digit_range_device(
    device: Arc<dyn akita_prover::DigitRangeDevice>,
) -> Result<(), String> {
    akita_prover::set_digit_range_device(device).map_err(|e| format!("digit range device: {e}"))
}

/// Routes the leading rounds of every stage-2 relation-range sumcheck
/// (quotient-factored and reduced-dense levels) through `device`
/// (`akita_prover::set_relation_range_device`); instances the device
/// declines stay on the CPU prover.
#[cfg(feature = "relation-range-device")]
pub fn install_relation_range_device(
    device: Arc<dyn akita_prover::RelationRangeDevice>,
) -> Result<(), String> {
    akita_prover::set_relation_range_device(device)
        .map_err(|e| format!("relation range device: {e}"))
}

/// `JOLT_RELATION_RANGE_DEVICE=cpu-ref` installs the CPU reference device.
#[cfg(all(feature = "relation-range-device", not(target_arch = "wasm32")))]
pub fn install_relation_range_device_from_env() -> Result<(), String> {
    match std::env::var("JOLT_RELATION_RANGE_DEVICE").as_deref() {
        Ok("cpu-ref") => {
            install_relation_range_device(Arc::new(relation_range_reference::CpuReferenceDevice))
        }
        Ok(other) => Err(format!("unknown JOLT_RELATION_RANGE_DEVICE={other:?}")),
        Err(_) => Ok(()),
    }
}

/// `JOLT_TRACE_COMMIT_DEVICE=cpu-ref` installs the CPU reference device.
#[cfg(all(feature = "trace-commit-device", not(target_arch = "wasm32")))]
pub fn install_trace_commit_device_from_env() -> Result<(), String> {
    match std::env::var("JOLT_TRACE_COMMIT_DEVICE").as_deref() {
        Ok("cpu-ref") => {
            install_trace_commit_device(Arc::new(trace_commit_reference::CpuReferenceDevice))
        }
        Ok(other) => Err(format!("unknown JOLT_TRACE_COMMIT_DEVICE={other:?}")),
        Err(_) => Ok(()),
    }
}

/// Everything a prover needs that is independent of the concrete trace.
pub struct ProverContext {
    pub schedule_artifacts: Arc<AkitaScheduleArtifacts>,
    pub program_preprocessing: JoltProgramPreprocessing,
    pub program: Arc<JoltProgram>,
}

impl ProverContext {
    pub fn new(
        schedule_artifacts_bytes: &[u8],
        program_preprocessing_bytes: &[u8],
        elf: &[u8],
    ) -> Result<Self, String> {
        Ok(Self {
            schedule_artifacts: decode_schedule_artifacts(schedule_artifacts_bytes)?,
            program_preprocessing: decode_program_preprocessing(program_preprocessing_bytes)?,
            program: Arc::new(JoltProgram::from_elf_bytes(elf.to_vec())),
        })
    }
}

/// Wall-clock milliseconds per phase, measured inside the engine so the
/// browser and the native roundtrip report the same split.
#[derive(Clone, Copy, Debug, Default)]
pub struct PhaseTimings {
    pub trace_ms: f64,
    pub setup_ms: f64,
    pub prove_ms: f64,
}

pub struct ProveOutput {
    pub proof_bytes: Vec<u8>,
    pub io_bytes: Vec<u8>,
    /// `JoltVerifierPreprocessing` for exactly this proof shape (bincode).
    pub verifier_preprocessing_bytes: Vec<u8>,
    pub unpadded_cycles: usize,
    pub padded_cycles: usize,
    pub timings: PhaseTimings,
    #[cfg(target_arch = "wasm32")]
    pub gpu: crate::gpu::GpuReport,
}

#[tracing::instrument(skip_all, name = "engine::prove")]
pub fn prove(ctx: &ProverContext, inputs: &[u8]) -> Result<ProveOutput, String> {
    let program_preprocessing = &ctx.program_preprocessing;
    let layout = &program_preprocessing.memory_layout;
    let memory_config = MemoryConfig {
        max_untrusted_advice_size: layout.max_untrusted_advice_size,
        max_trusted_advice_size: layout.max_trusted_advice_size,
        max_input_size: layout.max_input_size,
        max_output_size: layout.max_output_size,
        stack_size: layout.stack_size,
        heap_size: layout.heap_size,
        program_size: Some(layout.program_size),
    };

    #[cfg(target_arch = "wasm32")]
    let mut gpu = crate::gpu::status_report();

    let mut clock = Clock::start();
    let trace_output = TracerBackend::new()
        .trace_compact(
            &ctx.program,
            TraceInputs::new(inputs.to_vec(), Vec::new(), Vec::new(), memory_config),
            &program_preprocessing.bytecode,
        )
        .map_err(|e| format!("trace error: {e:?}"))?;
    let unpadded_cycles = trace_output.trace.len();
    let trace_ms = clock.lap();

    let config = ProverConfig::derive_compact::<F>(
        trace_output.trace.as_slice(),
        layout,
        program_preprocessing.ram.min_bytecode_address,
        program_preprocessing.ram.bytecode_words.len(),
        program_preprocessing.max_padded_trace_length,
    )
    .map_err(|e| format!("config derive error: {e}"))?;

    let prep: ProverPrep = akita_preprocessing::preprocess_full(
        &ctx.schedule_artifacts,
        program_preprocessing.clone(),
        &config,
    )
    .map_err(|e| format!("Akita preprocessing error: {e}"))?;
    let setup_ms = clock.lap();

    let program_preprocessing_arc = prep
        .program_arc()
        .ok_or("full (non-committed) program preprocessing required")?;
    let public_io = trace_output.device.clone();
    let witness = TraceBackend::<OwnedTrace>::from_compact(
        JoltVmWitnessConfig::new(
            config.trace_length.ilog2() as usize,
            config.ram_K,
            config.one_hot_config,
        ),
        JoltVmWitnessInputs::new(&ctx.program, &program_preprocessing_arc, trace_output),
    );

    let backend = JoltAkitaBackend::<F, Pcs>::optimized();
    let proof = jolt_prover::prove::<F, Pcs, Vc, Transcript, _>(
        &backend, &prep, &config, None, &witness, &public_io,
    )
    .map_err(|e| format!("prove error: {e}"))?;
    let prove_ms = clock.lap();
    #[cfg(target_arch = "wasm32")]
    {
        gpu.commit = crate::gpu::take_commit_report();
        gpu.digit_range = crate::gpu::take_digit_range_report();
        gpu.stage2 = crate::gpu::take_stage2_report();
    }

    Ok(ProveOutput {
        proof_bytes: encode(&proof, "proof")?,
        io_bytes: encode(&public_io, "program io")?,
        verifier_preprocessing_bytes: encode(&prep.verifier, "verifier preprocessing")?,
        unpadded_cycles,
        padded_cycles: config.trace_length,
        timings: PhaseTimings {
            trace_ms,
            setup_ms,
            prove_ms,
        },
        #[cfg(target_arch = "wasm32")]
        gpu,
    })
}

#[tracing::instrument(skip_all, name = "engine::verify")]
pub fn verify(prep: &VerifierPrep, proof_bytes: &[u8], io_bytes: &[u8]) -> Result<(), String> {
    let proof: Proof = decode(proof_bytes, "proof")?;
    let public_io: JoltDevice = decode(io_bytes, "program io")?;
    jolt_verifier::verify::<F, Pcs, Vc, Transcript>(prep, &public_io, &proof, None)
        .map_err(|e| format!("verification failed: {e}"))
}

/// `std::time::Instant` panics on wasm32-unknown-unknown; the browser gets
/// its wall clock from JS instead.
struct Clock {
    last: f64,
}

impl Clock {
    fn start() -> Self {
        Self { last: now_ms() }
    }

    /// Milliseconds since the previous lap (or start).
    fn lap(&mut self) -> f64 {
        let now = now_ms();
        let elapsed = now - self.last;
        self.last = now;
        elapsed
    }
}

#[cfg(target_arch = "wasm32")]
fn now_ms() -> f64 {
    js_sys::Date::now()
}

#[cfg(not(target_arch = "wasm32"))]
fn now_ms() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64() * 1000.0)
        .unwrap_or(0.0)
}
