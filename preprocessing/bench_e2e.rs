//! Native e2e prover benchmark: full Jolt prove with the stock CPU Dory PCS
//! vs the full-GPU Dory PCS, with per-span phase breakdown. Every proof is
//! serialized and verified with the stock verifier; in `--check` mode the
//! CPU and GPU runs must produce byte-identical commitments (commits are
//! Transparent and deterministic; the ZK opening is not).
//!
//! Usage:
//!   bench-e2e --guest keccak --iters 2400 --pcs gpu --runs 3
//!   bench-e2e --guest keccak --iters 2400 --check
//!   bench-e2e --guest keccak --iters 2400 --calibrate

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use ark_bn254::Fr;
use ark_serialize::CanonicalDeserialize;
use common::jolt_device::{JoltDevice, MemoryConfig};
use jolt_core::curve::Bn254Curve;
use jolt_core::poly::commitment::commitment_scheme::{
    CommitmentScheme, StreamingCommitmentScheme, ZkEvalCommitment,
};
use jolt_core::poly::commitment::dory::{ArkworksProverSetup, DoryCommitmentScheme};
use jolt_core::transcripts::Blake2bTranscript;
use jolt_core::zkvm::proof_serialization::JoltProof;
use jolt_core::zkvm::prover::{JoltCpuProver, JoltProverPreprocessing};
use jolt_core::zkvm::verifier::{JoltSharedPreprocessing, JoltVerifierPreprocessing};
use jolt_core::zkvm::{RV64IMACVerifier, Serializable};
use jolt_wasm_prover::gpu_pcs::GpuDoryCommitmentScheme;

// ---------------------------------------------------------------------------
// Span timing layer: aggregates busy time per span name across threads.
// ---------------------------------------------------------------------------

static SPAN_ACC: Mutex<Option<HashMap<&'static str, (Duration, u64)>>> = Mutex::new(None);

thread_local! {
    static OPEN_SPANS: std::cell::RefCell<HashMap<u64, Instant>> =
        std::cell::RefCell::new(HashMap::new());
}

struct TimingLayer;

impl<S> tracing_subscriber::Layer<S> for TimingLayer
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    fn on_enter(&self, id: &tracing::span::Id, _ctx: tracing_subscriber::layer::Context<'_, S>) {
        OPEN_SPANS.with(|spans| {
            spans.borrow_mut().insert(id.into_u64(), Instant::now());
        });
    }

    fn on_exit(&self, id: &tracing::span::Id, ctx: tracing_subscriber::layer::Context<'_, S>) {
        let started = OPEN_SPANS.with(|spans| spans.borrow_mut().remove(&id.into_u64()));
        let (Some(started), Some(span)) = (started, ctx.span(id)) else {
            return;
        };
        let elapsed = started.elapsed();
        let mut acc = SPAN_ACC.lock().unwrap();
        if let Some(map) = acc.as_mut() {
            let entry = map.entry(span.name()).or_insert((Duration::ZERO, 0));
            entry.0 += elapsed;
            entry.1 += 1;
        }
    }
}

fn spans_start() {
    *SPAN_ACC.lock().unwrap() = Some(HashMap::new());
}

fn spans_take() -> Vec<(&'static str, Duration, u64)> {
    let map = SPAN_ACC.lock().unwrap().take().unwrap_or_default();
    let mut rows: Vec<_> = map.into_iter().map(|(k, (d, n))| (k, d, n)).collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1));
    rows
}

// ---------------------------------------------------------------------------

struct Args {
    guest: String,
    iters: u32,
    pcs: String,
    runs: usize,
    calibrate: bool,
    check: bool,
}

fn parse_args() -> Args {
    let mut args = Args {
        guest: "keccak".to_string(),
        iters: 2400,
        pcs: "cpu".to_string(),
        runs: 3,
        calibrate: false,
        check: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--guest" => args.guest = it.next().expect("--guest value"),
            "--iters" => args.iters = it.next().expect("--iters value").parse().unwrap(),
            "--pcs" => args.pcs = it.next().expect("--pcs value"),
            "--runs" => args.runs = it.next().expect("--runs value").parse().unwrap(),
            "--calibrate" => args.calibrate = true,
            "--check" => args.check = true,
            other => panic!("unknown flag {other}"),
        }
    }
    args
}

fn load_artifacts(guest: &str) -> (Vec<u8>, ArkworksProverSetup, JoltSharedPreprocessing) {
    let public = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("frontend/public");
    let elf = std::fs::read(public.join(format!("{guest}.elf"))).expect("guest elf");
    let prep_bytes =
        std::fs::read(public.join(format!("{guest}_prover.bin"))).expect("prover preprocessing");

    let mut cursor = std::io::Cursor::new(&prep_bytes[..]);
    let generators = ArkworksProverSetup::deserialize_with_mode(
        &mut cursor,
        ark_serialize::Compress::No,
        ark_serialize::Validate::No,
    )
    .expect("setup deserialize");
    let shared = JoltSharedPreprocessing::deserialize_with_mode(
        &mut cursor,
        ark_serialize::Compress::No,
        ark_serialize::Validate::No,
    )
    .expect("shared deserialize");
    (elf, generators, shared)
}

fn guest_inputs(guest: &str, iters: u32) -> Vec<u8> {
    match guest {
        "keccak" => {
            let input = [7u8; 32];
            let mut inputs = Vec::new();
            inputs.extend_from_slice(&postcard::to_allocvec(&input).unwrap());
            inputs.extend_from_slice(&postcard::to_allocvec(&iters).unwrap());
            inputs
        }
        "sha2" => {
            let input = vec![7u8; iters as usize];
            postcard::to_allocvec(&input.as_slice()).unwrap()
        }
        other => panic!("unknown guest {other}"),
    }
}

#[allow(clippy::type_complexity)]
fn trace_guest(
    elf: &[u8],
    shared: &JoltSharedPreprocessing,
    inputs: &[u8],
) -> (
    tracer::LazyTraceIterator,
    Vec<tracer::instruction::Cycle>,
    tracer::emulator::memory::Memory,
    JoltDevice,
) {
    let layout = &shared.memory_layout;
    let memory_config = MemoryConfig {
        max_untrusted_advice_size: layout.max_untrusted_advice_size,
        max_trusted_advice_size: layout.max_trusted_advice_size,
        max_input_size: layout.max_input_size,
        max_output_size: layout.max_output_size,
        stack_size: layout.stack_size,
        heap_size: layout.heap_size,
        program_size: Some(layout.program_size),
    };
    let (lazy_trace, trace, final_memory, program_io, _advice) =
        jolt_core::guest::program::trace(elf, None, inputs, &[], &[], &memory_config, None);
    (lazy_trace, trace, final_memory, program_io)
}

struct RunResult {
    trace_s: f64,
    prove_s: f64,
    proof_bytes: Vec<u8>,
    io_bytes: Vec<u8>,
    num_cycles: usize,
    commitment_bytes: Vec<u8>,
    spans: Vec<(&'static str, Duration, u64)>,
}

fn run_once<PCS>(
    elf: &[u8],
    generators: &ArkworksProverSetup,
    shared: &JoltSharedPreprocessing,
    inputs: &[u8],
) -> RunResult
where
    PCS: StreamingCommitmentScheme<Field = Fr, ProverSetup = ArkworksProverSetup>
        + ZkEvalCommitment<Bn254Curve>
        + CommitmentScheme<Proof = jolt_core::poly::commitment::dory::ArkDoryProof>
        + CommitmentScheme<Commitment = jolt_core::poly::commitment::dory::ArkGT>,
{
    let preprocessing: JoltProverPreprocessing<Fr, PCS> = JoltProverPreprocessing {
        generators: generators.clone(),
        shared: shared.clone(),
    };

    let t = Instant::now();
    let (lazy_trace, trace, final_memory, program_io) = trace_guest(elf, shared, inputs);
    let trace_s = t.elapsed().as_secs_f64();
    let num_cycles = trace.len();

    spans_start();
    let t = Instant::now();
    let prover: JoltCpuProver<'_, Fr, Bn254Curve, PCS, Blake2bTranscript> =
        JoltCpuProver::gen_from_trace(
            &preprocessing,
            lazy_trace,
            trace,
            program_io.clone(),
            None,
            None,
            final_memory,
        );
    let (proof, _) = prover.prove();
    let prove_s = t.elapsed().as_secs_f64();
    let spans = spans_take();

    let mut commitment_bytes = Vec::new();
    ark_serialize::CanonicalSerialize::serialize_compressed(
        &proof.commitments,
        &mut commitment_bytes,
    )
    .expect("commitment serialization");

    // `Serializable` is only implemented for the stock-PCS proof alias;
    // the GPU-PCS proof has identical field types, so CanonicalSerialize
    // produces the same byte format.
    let mut proof_bytes = Vec::new();
    ark_serialize::CanonicalSerialize::serialize_compressed(&proof, &mut proof_bytes)
        .expect("proof serialization");

    RunResult {
        trace_s,
        prove_s,
        proof_bytes,
        io_bytes: program_io.serialize_to_bytes().expect("io serialization"),
        num_cycles,
        commitment_bytes,
        spans,
    }
}

/// Every proof — whichever PCS produced it — is checked by the stock
/// verifier type.
fn verify_stock(
    generators: &ArkworksProverSetup,
    shared: &JoltSharedPreprocessing,
    proof_bytes: &[u8],
    io_bytes: &[u8],
) {
    let prover_prep: JoltProverPreprocessing<Fr, DoryCommitmentScheme> = JoltProverPreprocessing {
        generators: generators.clone(),
        shared: shared.clone(),
    };
    let verifier_prep = JoltVerifierPreprocessing::from(&prover_prep);

    let proof: JoltProof<Fr, Bn254Curve, DoryCommitmentScheme, Blake2bTranscript> =
        JoltProof::deserialize_compressed(&proof_bytes[..]).expect("proof deserialize");
    let program_io = JoltDevice::deserialize_from_bytes(io_bytes).expect("io deserialize");

    let verifier = RV64IMACVerifier::new(&verifier_prep, proof, program_io, None, None)
        .expect("verifier init");
    verifier.verify().expect("proof must verify");
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn span_table(spans: &[(&'static str, Duration, u64)], top: usize) -> String {
    spans
        .iter()
        .take(top)
        .map(|(name, d, n)| format!("    {:<60} {:>9.3}s  x{n}", name, d.as_secs_f64()))
        .collect::<Vec<_>>()
        .join("\n")
}

fn main() {
    use tracing_subscriber::layer::SubscriberExt;
    let subscriber = tracing_subscriber::registry().with(TimingLayer);
    tracing::subscriber::set_global_default(subscriber).expect("set subscriber");

    let _ = jolt_inlines_sha2::init_inlines();
    let _ = jolt_inlines_secp256k1::init_inlines();
    let _ = jolt_inlines_keccak256::init_inlines();

    let args = parse_args();
    let (elf, generators, shared) = load_artifacts(&args.guest);
    let inputs = guest_inputs(&args.guest, args.iters);

    if args.calibrate {
        let t = Instant::now();
        let (_, trace, _, _) = trace_guest(&elf, &shared, &inputs);
        println!(
            "{{\"guest\":\"{}\",\"iters\":{},\"cycles\":{},\"padded\":{},\"log2\":{},\"trace_s\":{:.2}}}",
            args.guest,
            args.iters,
            trace.len(),
            trace.len().next_power_of_two(),
            trace.len().next_power_of_two().trailing_zeros(),
            t.elapsed().as_secs_f64(),
        );
        return;
    }

    if args.check {
        eprintln!("[check] CPU run…");
        let cpu = run_once::<DoryCommitmentScheme>(&elf, &generators, &shared, &inputs);
        verify_stock(&generators, &shared, &cpu.proof_bytes, &cpu.io_bytes);
        eprintln!("[check] CPU proof verified ({} cycles)", cpu.num_cycles);
        eprintln!("[check] GPU run…");
        let gpu = run_once::<GpuDoryCommitmentScheme>(&elf, &generators, &shared, &inputs);
        verify_stock(&generators, &shared, &gpu.proof_bytes, &gpu.io_bytes);
        eprintln!("[check] GPU proof verified ({} cycles)", gpu.num_cycles);
        assert_eq!(
            cpu.commitment_bytes, gpu.commitment_bytes,
            "CPU and GPU witness commitments must be byte-identical"
        );
        println!(
            "{{\"check\":\"ok\",\"cycles\":{},\"commitments\":\"byte-identical\",\"both_verified\":true}}",
            cpu.num_cycles
        );
        return;
    }

    let is_gpu = match args.pcs.as_str() {
        "cpu" => false,
        "gpu" => true,
        other => panic!("unknown pcs {other}"),
    };

    eprintln!("[bench] guest={} pcs={} warmup…", args.guest, args.pcs);
    // Warmup run (engine init, pipeline compilation, page cache) — discarded.
    let warm = if is_gpu {
        run_once::<GpuDoryCommitmentScheme>(&elf, &generators, &shared, &inputs)
    } else {
        run_once::<DoryCommitmentScheme>(&elf, &generators, &shared, &inputs)
    };
    verify_stock(&generators, &shared, &warm.proof_bytes, &warm.io_bytes);
    eprintln!(
        "[bench] warmup ok: {} cycles (2^{}), prove {:.2}s",
        warm.num_cycles,
        warm.num_cycles.next_power_of_two().trailing_zeros(),
        warm.prove_s
    );

    let mut trace_times = Vec::new();
    let mut prove_times = Vec::new();
    for run in 0..args.runs {
        let result = if is_gpu {
            run_once::<GpuDoryCommitmentScheme>(&elf, &generators, &shared, &inputs)
        } else {
            run_once::<DoryCommitmentScheme>(&elf, &generators, &shared, &inputs)
        };
        verify_stock(&generators, &shared, &result.proof_bytes, &result.io_bytes);
        eprintln!(
            "[bench] run {}: trace {:.2}s prove {:.2}s (verified)",
            run, result.trace_s, result.prove_s
        );
        eprintln!("{}", span_table(&result.spans, 24));
        trace_times.push(result.trace_s);
        prove_times.push(result.prove_s);
    }

    let trace_median = median(trace_times);
    let prove_median = median(prove_times.clone());
    println!(
        "{{\"guest\":\"{}\",\"pcs\":\"{}\",\"cycles\":{},\"runs\":{},\"trace_median_s\":{:.3},\"prove_median_s\":{:.3},\"prove_all\":{:?},\"verified\":true}}",
        args.guest, args.pcs, warm.num_cycles, args.runs, trace_median, prove_median, prove_times,
    );
}
