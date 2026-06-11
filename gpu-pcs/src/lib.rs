#![cfg_attr(target_arch = "wasm32", feature(stdarch_wasm_atomic_wait))]
//! Full-GPU Dory commitment scheme for the Jolt prover (benchmark path).
//!
//! Drop-in replacement for `jolt_core`'s `DoryCommitmentScheme` with the
//! same commitment/proof types, so the stock verifier checks the proofs
//! unchanged. Selecting this type as the prover's PCS parameter switches
//! the build from the streamed CPU path to the materialized GPU path; the
//! default `DoryCommitmentScheme` path is untouched.
//!
//! Unfused commit: jolt-core's streaming witness generation still calls
//! `process_chunk` per matrix row, but rows are buffered (`RowData`) instead
//! of MSM'd — `aggregate_chunks` then runs tier-1 (dense row MSMs, one-hot
//! gather-adds) and tier-2 (batched multipairing) on the GPU, and the raw
//! polynomial data stays GPU-resident. The opening reuses it: the joint RLC
//! matrix, the L^T·M product, the row-commitment RLC, and all reduce rounds
//! run on the GPU (ZK mode, matching jolt-core's `zk` build). The CPU keeps
//! only the transcript, blinds/masks, and final exponentiations.
//!
//! Memory: this materializes the witness (~T*4 bytes per one-hot poly,
//! ~T*32 per dense poly, plus the 2^(nu+sigma)*32-byte joint matrix on the
//! GPU) — the price of unfusing the streaming design for the benchmark.
//!
//! Limitation: advice polynomials commit on the CPU in their own Dory
//! contexts; openings that include advice claims are rejected (the
//! benchmark guests use no advice).

use std::sync::mpsc::SyncSender;
use std::sync::{Mutex, OnceLock};

use ark_bn254::Fr;
use rayon::prelude::*;

use dory_gpu::jolt::{CommitOut, JoltGpuDory, PolyUpload};
use dory_gpu::prove::GpuDory;
use dory_gpu::GpuContext;
use dory_pcs::backends::arkworks::{ArkFr, ArkG1, ArkGT, BN254};
use dory_pcs::primitives::arithmetic::Group as DoryGroup;
use dory_pcs::primitives::poly::Polynomial as DoryPolynomial;
use dory_pcs::primitives::transcript::Transcript as DoryTranscript;
use dory_pcs::primitives::DorySerialize;
use dory_pcs::ProverSetup;

use jolt_core::curve::JoltCurve;
use jolt_core::field::JoltField;
use jolt_core::poly::commitment::commitment_scheme::{
    CommitmentScheme, StreamingCommitmentScheme, ZkEvalCommitment,
};
use jolt_core::poly::commitment::dory::{
    ArkDoryProof, ArkworksProverSetup, ArkworksVerifierSetup, DoryCommitmentScheme, DoryGlobals,
    DoryLayout, JoltG1Routines,
};
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::transcripts::Transcript;
use jolt_core::utils::{errors::ProofVerifyError, math::Math, small_scalar::SmallScalar};
use tracing::trace_span;

#[inline]
fn jolt_to_ark(f: &Fr) -> ArkFr {
    // SAFETY: ArkFr is a transparent-layout newtype over Fr (same as
    // jolt-core's private wrappers::jolt_to_ark).
    unsafe { std::mem::transmute_copy(f) }
}

#[inline]
fn ark_to_jolt(ark: &ArkFr) -> Fr {
    // SAFETY: see jolt_to_ark.
    unsafe { std::mem::transmute_copy(ark) }
}

// ---------------------------------------------------------------------------
// GPU engine: a dedicated thread owning the wgpu device, the Dory setup
// buffers, and the per-proof polynomial registry. Jobs are closures executed
// in submission order; callers block on a response channel, so jobs may
// borrow the caller's stack.
// ---------------------------------------------------------------------------

/// A queued engine job: receives the (lazily initialized) engine state and
/// returns a future driven by the engine's executor — `pollster` on the
/// native engine thread, the GPU worker's event loop on WASM.
///
/// The state is a raw pointer and the lifetimes are erased (`'static`):
/// soundness rests on two invariants — the executor drives each job future
/// to completion before starting the next (no state aliasing), and the
/// submitting caller blocks until its job completed (captured borrows
/// outlive execution).
type JobFuture = std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'static>>;
type Job = Box<dyn FnOnce(*mut Option<JoltGpuDory>) -> JobFuture + Send + 'static>;

pub struct GpuEngine {
    transport: Transport,
    init: std::sync::Once,
    commit_pending: Mutex<Vec<(PolyUpload, SyncSender<CommitOut>)>>,
    commit_flush: Mutex<()>,
}

static ENGINE: OnceLock<GpuEngine> = OnceLock::new();

#[cfg(not(target_arch = "wasm32"))]
struct Transport {
    tx: std::sync::mpsc::Sender<Job>,
}

#[cfg(not(target_arch = "wasm32"))]
impl Transport {
    fn start() -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<Job>();
        std::thread::Builder::new()
            .name("dory-gpu-engine".into())
            .spawn(move || {
                // Jobs run inside a private rayon pool: callers block
                // global-pool threads while waiting on the engine, so any
                // engine-side parallel work routed to the global pool (ark
                // normalize_batch during init, final exponentiations during
                // passes) would deadlock.
                let pool = rayon::ThreadPoolBuilder::new()
                    .thread_name(|i| format!("dory-gpu-cpu-{i}"))
                    .build()
                    .expect("engine rayon pool");
                let mut state: Option<JoltGpuDory> = None;
                let state_addr = &mut state as *mut Option<JoltGpuDory> as usize;
                while let Ok(job) = rx.recv() {
                    // The pointer travels as usize: pool.install moves the
                    // closure to a (Send-requiring) pool thread.
                    pool.install(move || {
                        pollster::block_on(job(state_addr as *mut Option<JoltGpuDory>))
                    });
                }
            })
            .expect("failed to spawn GPU engine thread");
        Self { tx }
    }

    fn submit(&self, job: Job) {
        self.tx.send(job).expect("GPU engine thread died");
    }
}

/// WASM transport: jobs queue in shared memory; an atomic signal wakes the
/// dedicated GPU worker (`Atomics.waitAsync` in the GPU worker's JS loop),
/// which drains the queue through the async [`gpu_engine_drain`] export.
/// Callers run on worker threads and may block; the GPU worker never blocks
/// (its event loop drives the WebGPU callbacks).
#[cfg(target_arch = "wasm32")]
struct Transport;

#[cfg(target_arch = "wasm32")]
static WASM_JOBS: Mutex<std::collections::VecDeque<Job>> =
    Mutex::new(std::collections::VecDeque::new());
#[cfg(target_arch = "wasm32")]
static WASM_SIGNAL: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

#[cfg(target_arch = "wasm32")]
impl Transport {
    fn start() -> Self {
        Self
    }

    fn submit(&self, job: Job) {
        WASM_JOBS.lock().unwrap().push_back(job);
        WASM_SIGNAL.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        // SAFETY: notifying a valid i32 in shared linear memory.
        unsafe {
            core::arch::wasm32::memory_atomic_notify(
                &WASM_SIGNAL as *const _ as *mut i32,
                u32::MAX,
            );
        }
    }
}

/// Byte address of the GPU job signal word inside the WASM linear memory,
/// for `Atomics.waitAsync` in the GPU worker's JS loop.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn gpu_engine_signal_addr() -> u32 {
    &WASM_SIGNAL as *const _ as u32
}

/// Executes every queued GPU job. Must be called from the dedicated GPU
/// worker (the thread that owns the WebGPU device); the JS loop awaits each
/// call, so jobs run sequentially to completion.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn gpu_engine_drain() {
    thread_local! {
        static STATE: std::cell::RefCell<Option<JoltGpuDory>> =
            const { std::cell::RefCell::new(None) };
    }
    loop {
        let job = WASM_JOBS.lock().unwrap().pop_front();
        let Some(job) = job else { break };
        let state_ptr = STATE.with(|s| s.as_ptr());
        job(state_ptr).await;
    }
}

impl GpuEngine {
    /// Returns the process-global engine, initializing the device and the
    /// setup buffers on first use with the setup sliced to the current Dory
    /// context's column count (Jolt's balanced layout guarantees
    /// rows <= columns).
    pub fn get(setup: &ArkworksProverSetup) -> &'static GpuEngine {
        let engine = ENGINE.get_or_init(|| GpuEngine {
            transport: Transport::start(),
            init: std::sync::Once::new(),
            commit_pending: Mutex::new(Vec::new()),
            commit_flush: Mutex::new(()),
        });
        engine.init.call_once(|| {
            let n = DoryGlobals::get_num_columns();
            assert!(n.is_power_of_two() && n > 1, "Dory context not initialized");
            let sliced = ProverSetup::<BN254> {
                g1_vec: setup.g1_vec[..n].to_vec(),
                g2_vec: setup.g2_vec[..n].to_vec(),
                h1: setup.h1,
                h2: setup.h2,
                ht: setup.ht,
            };
            let (rtx, rrx) = std::sync::mpsc::sync_channel::<()>(1);
            engine.transport.submit(Box::new(move |state| {
                Box::pin(async move {
                    let ctx = GpuContext::new().await.expect("WebGPU init failed");
                    let gpu = GpuDory::new(std::sync::Arc::new(ctx), sliced);
                    // SAFETY: jobs run sequentially to completion on the
                    // engine's executor; the state pointer is valid for the
                    // executor's lifetime.
                    unsafe { *state = Some(JoltGpuDory::new(gpu, n as u32)) };
                    let _ = rtx.send(());
                })
            }));
            rrx.recv().expect("GPU engine init failed");
        });
        engine
    }

    /// Runs `f` against the engine state and blocks until it completes.
    /// `f` may borrow from the caller's stack.
    pub fn run<'env, R, F>(&self, f: F) -> R
    where
        R: Send + 'env,
        F: FnOnce(
                &'env mut JoltGpuDory,
            ) -> std::pin::Pin<Box<dyn std::future::Future<Output = R> + 'env>>
            + Send
            + 'env,
    {
        type JobIn<'e> = Box<
            dyn FnOnce(
                    *mut Option<JoltGpuDory>,
                )
                    -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'e>>
                + Send
                + 'e,
        >;

        let (rtx, rrx) = std::sync::mpsc::sync_channel::<R>(1);
        let job: JobIn<'env> = Box::new(move |state| {
            // SAFETY: jobs run sequentially to completion on the engine's
            // executor — the state is never aliased — and the caller blocks
            // until completion, so 'env covers the execution.
            let gpu: &'env mut JoltGpuDory =
                unsafe { (*state).as_mut().expect("GPU engine not initialized") };
            let fut = f(gpu);
            Box::pin(async move {
                let _ = rtx.send(fut.await);
            })
        });
        // SAFETY: the environment lifetime is erased, but we block on `rrx`
        // until the job has run to completion (or the engine died), so every
        // borrow captured by `f` outlives its use — the same argument as
        // scoped threads.
        let job: Job = unsafe { std::mem::transmute(job) };
        self.transport.submit(job);
        rrx.recv().expect("GPU engine dropped job (engine panic)")
    }

    /// Unfused commit with cross-caller coalescing: requests that arrive
    /// while a batch is in flight merge into the next batched GPU pass
    /// (Jolt commits ~45 polynomials from a rayon `par_iter`). A short
    /// collection window lets concurrently flattening callers land in the
    /// same batch — each batched pass costs a near-constant ~450 Miller
    /// dispatches, so batch COUNT dominates tier-2 wall time.
    fn commit(&self, upload: PolyUpload) -> CommitOut {
        let (rtx, rrx) = std::sync::mpsc::sync_channel::<CommitOut>(1);
        self.commit_pending.lock().unwrap().push((upload, rtx));

        let guard = self.commit_flush.lock().unwrap();
        if let Ok(result) = rrx.try_recv() {
            return result; // a previous flush holder served us
        }
        let batch: Vec<_> = std::mem::take(&mut *self.commit_pending.lock().unwrap());
        debug_assert!(!batch.is_empty());
        let (uploads, senders): (Vec<_>, Vec<_>) = batch.into_iter().unzip();
        let outs = self.run(move |gpu| {
            Box::pin(async move { gpu.commit_batch(uploads).await })
                as std::pin::Pin<Box<dyn std::future::Future<Output = Vec<CommitOut>> + '_>>
        });
        for (sender, out) in senders.into_iter().zip(outs) {
            let _ = sender.send(out);
        }
        drop(guard);
        rrx.recv()
            .expect("commit batch did not include own request")
    }
}

/// Initializes the GPU engine (device, setup buffers, prepared G2 lines)
/// outside any timed region. Requires an active Dory context.
pub fn warmup_gpu_engine(setup: &ArkworksProverSetup) {
    GpuEngine::get(setup).run(|_| {
        Box::pin(async {}) as std::pin::Pin<Box<dyn std::future::Future<Output = ()> + '_>>
    });
}

// ---------------------------------------------------------------------------
// Transcript bridge (replica of jolt-core's private JoltToDoryTranscript —
// byte-compatible with the verifier's adapter).
// ---------------------------------------------------------------------------

struct JoltDoryTranscript<'a, T: Transcript> {
    transcript: &'a mut T,
}

impl<'a, T: Transcript> DoryTranscript for JoltDoryTranscript<'a, T> {
    type Curve = BN254;

    fn append_bytes(&mut self, _label: &[u8], bytes: &[u8]) {
        self.transcript.append_bytes(b"dory_bytes", bytes);
    }

    fn append_field(&mut self, _label: &[u8], x: &ArkFr) {
        self.transcript
            .append_scalar(b"dory_field", &ark_to_jolt(x));
    }

    fn append_group<G: DoryGroup>(&mut self, _label: &[u8], g: &G) {
        let mut buffer = Vec::new();
        g.serialize_compressed(&mut buffer)
            .expect("DorySerialize serialization should not fail");
        self.transcript.append_bytes(b"dory_group", &buffer);
    }

    fn append_serde<S: DorySerialize>(&mut self, _label: &[u8], s: &S) {
        let mut buffer = Vec::new();
        s.serialize_compressed(&mut buffer)
            .expect("DorySerialize serialization should not fail");
        self.transcript.append_bytes(b"dory_serde", &buffer);
    }

    fn challenge_scalar(&mut self, _label: &[u8]) -> ArkFr {
        jolt_to_ark(&self.transcript.challenge_scalar::<Fr>())
    }

    fn reset(&mut self, _domain_label: &[u8]) {
        panic!("reset not supported")
    }
}

/// Replica of jolt-core's private `reorder_opening_point_for_layout`.
fn reorder_opening_point_for_layout<F: JoltField>(
    opening_point: &[F::Challenge],
) -> Vec<F::Challenge> {
    if DoryGlobals::get_layout() == DoryLayout::AddressMajor {
        let log_t = DoryGlobals::get_T().log_2();
        let log_k = opening_point.len().saturating_sub(log_t);
        let (r_address, r_cycle) = opening_point.split_at(log_k);
        [r_cycle, r_address].concat()
    } else {
        opening_point.to_vec()
    }
}

// ---------------------------------------------------------------------------
// The commitment scheme
// ---------------------------------------------------------------------------

/// Opening hint: the row commitments plus the engine registry handle of the
/// GPU-resident polynomial data (absent for CPU-committed advice polys).
#[derive(Clone, Debug, PartialEq)]
pub struct GpuDoryHint {
    gpu_id: Option<u64>,
    rows: Vec<ArkG1>,
}

/// Sentinel id of a combined (RLC) hint whose joint state is staged in the
/// engine.
const JOINT_HINT: u64 = u64::MAX;

/// A buffered matrix row, produced by the streaming witness generation and
/// consumed by the GPU tier-1 pass.
#[derive(Clone, Debug, PartialEq)]
pub enum RowData {
    /// One dense row (Montgomery form), `num_columns` entries.
    Dense(Vec<Fr>),
    /// One chunk's bucket index per column (`ONEHOT_NONE` for none).
    OneHot(Vec<u32>),
}

#[derive(Clone)]
pub struct GpuDoryCommitmentScheme;

impl CommitmentScheme for GpuDoryCommitmentScheme {
    type Field = Fr;
    type ProverSetup = ArkworksProverSetup;
    type VerifierSetup = ArkworksVerifierSetup;
    type Commitment = ArkGT;
    type Proof = ArkDoryProof;
    type BatchedProof = Vec<ArkDoryProof>;
    type OpeningProofHint = GpuDoryHint;

    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup {
        DoryCommitmentScheme::setup_prover(max_num_vars)
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        DoryCommitmentScheme::setup_verifier(setup)
    }

    /// Whole-matrix commit. Only used for the (tiny) advice polynomials,
    /// which live in their own Dory contexts — stays on the CPU.
    fn commit(
        poly: &MultilinearPolynomial<Fr>,
        setup: &Self::ProverSetup,
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        let _span = trace_span!("GpuDoryCommitmentScheme::commit").entered();

        let sigma = DoryGlobals::get_num_columns().log_2();
        let nu = DoryGlobals::get_max_num_rows().log_2();

        let (tier_2, rows, _blind) =
            <MultilinearPolynomial<Fr> as DoryPolynomial<ArkFr>>::commit::<
                BN254,
                dory_pcs::Transparent,
                JoltG1Routines,
            >(poly, nu, sigma, setup)
            .expect("commitment should succeed");

        (tier_2, GpuDoryHint { gpu_id: None, rows })
    }

    fn batch_commit<U>(
        polys: &[U],
        gens: &Self::ProverSetup,
    ) -> Vec<(Self::Commitment, Self::OpeningProofHint)>
    where
        U: std::borrow::Borrow<MultilinearPolynomial<Fr>> + Sync,
    {
        polys
            .par_iter()
            .map(|poly| Self::commit(poly.borrow(), gens))
            .collect()
    }

    /// Batched Dory opening, entirely on the GPU: the joint RLC matrix is
    /// rebuilt on-device from the commit-time polynomial data, the L^T·M
    /// product runs as a GPU kernel, and the reduce rounds follow. ZK mode
    /// matches jolt-core's `zk` build (CPU-side OsRng blinds and masks).
    /// The joint polynomial argument is never touched.
    fn prove<ProofTranscript: Transcript>(
        setup: &Self::ProverSetup,
        _poly: &MultilinearPolynomial<Fr>,
        opening_point: &[<Fr as JoltField>::Challenge],
        hint: Option<Self::OpeningProofHint>,
        transcript: &mut ProofTranscript,
    ) -> (Self::Proof, Option<Self::Field>) {
        let _span = trace_span!("GpuDoryCommitmentScheme::prove").entered();

        let hint = hint.expect("GPU Dory opening requires the combined hint");
        assert_eq!(
            hint.gpu_id,
            Some(JOINT_HINT),
            "GPU Dory opening requires a combine_hints-produced hint"
        );

        let sigma = DoryGlobals::get_num_columns().log_2();
        let nu = DoryGlobals::get_max_num_rows().log_2();

        let reordered_point = reorder_opening_point_for_layout::<Fr>(opening_point);
        let ark_point: Vec<ArkFr> = reordered_point
            .iter()
            .rev()
            .map(|p| {
                let f_val: Fr = (*p).into();
                jolt_to_ark(&f_val)
            })
            .collect();

        let (left, right) =
            dory_pcs::primitives::poly::compute_left_right_vectors(&ark_point, nu, sigma);
        let left_fr: Vec<Fr> = left.iter().map(|v| v.0).collect();
        let right_fr: Vec<Fr> = right.iter().map(|v| v.0).collect();

        let mut dory_transcript = JoltDoryTranscript { transcript };

        let engine = GpuEngine::get(setup);
        let (proof, y_blinding) = {
            let _span = trace_span!("gpu_opening").entered();
            engine.run(move |gpu| {
                Box::pin(async move {
                    gpu.open::<dory_pcs::ZK, _>(
                        &left_fr,
                        &right_fr,
                        nu,
                        sigma,
                        &mut dory_transcript,
                    )
                    .await
                })
            })
        };

        (proof, y_blinding.map(|b| ark_to_jolt(&b)))
    }

    fn verify<ProofTranscript: Transcript>(
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        opening_point: &[<Fr as JoltField>::Challenge],
        opening: &Fr,
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        DoryCommitmentScheme::verify(proof, setup, transcript, opening_point, opening, commitment)
    }

    fn protocol_name() -> &'static [u8] {
        b"Dory"
    }

    // jolt-core builds with its `zk` feature in this workspace, so the
    // cfg-gated trait method always exists.
    fn zk_generators_raw(
        setup: &Self::ProverSetup,
        count: usize,
    ) -> Option<(Vec<jolt_core::curve::Bn254G1>, jolt_core::curve::Bn254G1)> {
        DoryCommitmentScheme::zk_generators_raw(setup, count)
    }

    /// Homomorphic RLC of the row-commitment hints, on the GPU; also stages
    /// the joint-matrix recipe for the opening.
    #[tracing::instrument(skip_all, name = "GpuDoryCommitmentScheme::combine_hints")]
    fn combine_hints(
        hints: Vec<Self::OpeningProofHint>,
        coeffs: &[Self::Field],
    ) -> Self::OpeningProofHint {
        let num_rows = DoryGlobals::get_max_num_rows();

        let parts: Vec<(u64, Fr)> = hints
            .iter()
            .zip(coeffs)
            .map(|(hint, coeff)| {
                let id = hint
                    .gpu_id
                    .expect("advice polynomials are not supported by the GPU opening");
                assert_ne!(id, JOINT_HINT, "combine_hints called on a combined hint");
                (id, *coeff)
            })
            .collect();

        let engine = ENGINE.get().expect("GPU engine not initialized");
        let combined = engine.run(move |gpu| {
            Box::pin(async move { gpu.combine_rows(&parts, num_rows as u32).await })
        });

        GpuDoryHint {
            gpu_id: Some(JOINT_HINT),
            rows: combined.into_iter().map(ArkG1).collect(),
        }
    }

    #[tracing::instrument(skip_all, name = "GpuDoryCommitmentScheme::combine_commitments")]
    fn combine_commitments<C: std::borrow::Borrow<Self::Commitment>>(
        commitments: &[C],
        coeffs: &[Self::Field],
    ) -> Self::Commitment {
        let commitments_vec: Vec<&ArkGT> = commitments.iter().map(|c| c.borrow()).collect();
        coeffs
            .par_iter()
            .zip(commitments_vec.par_iter())
            .map(|(coeff, commitment)| jolt_to_ark(coeff) * **commitment)
            .reduce(ArkGT::identity, |a, b| a + b)
    }
}

impl StreamingCommitmentScheme for GpuDoryCommitmentScheme {
    type ChunkState = RowData;

    /// Buffers one dense matrix row (converted to Fr) — the MSM runs later
    /// on the GPU in `aggregate_chunks`.
    fn process_chunk<T: SmallScalar>(setup: &Self::ProverSetup, chunk: &[T]) -> Self::ChunkState {
        let _ = setup;
        debug_assert_eq!(chunk.len(), DoryGlobals::get_num_columns());
        RowData::Dense(chunk.iter().map(|s| s.to_field::<Fr>()).collect())
    }

    /// Buffers one chunk's bucket indices — the gather-add runs later on
    /// the GPU in `aggregate_chunks`.
    fn process_chunk_onehot(
        setup: &Self::ProverSetup,
        onehot_k: usize,
        chunk: &[Option<usize>],
    ) -> Self::ChunkState {
        let _ = (setup, onehot_k);
        RowData::OneHot(
            chunk
                .iter()
                .map(|k| match k {
                    Some(k) => *k as u32,
                    None => dory_gpu::commit::ONEHOT_NONE,
                })
                .collect(),
        )
    }

    /// Tier-1 + tier-2 on the GPU: row MSMs / gather-adds plus the batched
    /// multipairing, coalesced across concurrently committing polynomials.
    #[tracing::instrument(skip_all, name = "GpuDoryCommitmentScheme::compute_tier2_commitment")]
    fn aggregate_chunks(
        setup: &Self::ProverSetup,
        onehot_k: Option<usize>,
        chunks: &[Self::ChunkState],
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        let cols = DoryGlobals::get_num_columns();

        let upload = if let Some(k) = onehot_k {
            let rows_per_k = chunks.len() as u32;
            debug_assert_eq!(rows_per_k as usize, DoryGlobals::get_T() / cols);
            let mut indices = Vec::with_capacity(chunks.len() * cols);
            for chunk in chunks {
                match chunk {
                    RowData::OneHot(idx) => indices.extend_from_slice(idx),
                    RowData::Dense(_) => panic!("dense row in one-hot polynomial"),
                }
            }
            PolyUpload::OneHot {
                indices,
                k: k as u32,
                rows_per_k,
            }
        } else {
            let mut matrix = Vec::with_capacity(chunks.len() * cols);
            for chunk in chunks {
                match chunk {
                    RowData::Dense(row) => matrix.extend_from_slice(row),
                    RowData::OneHot(_) => panic!("one-hot row in dense polynomial"),
                }
            }
            PolyUpload::dense_from_mont(matrix, chunks.len() as u32)
        };

        let out = GpuEngine::get(setup).commit(upload);
        (
            out.tier2,
            GpuDoryHint {
                gpu_id: Some(out.id),
                rows: out.rows.into_iter().map(ArkG1).collect(),
            },
        )
    }
}

impl<C: JoltCurve> ZkEvalCommitment<C> for GpuDoryCommitmentScheme
where
    C::G1: From<ArkG1>,
{
    fn eval_commitment(proof: &Self::Proof) -> Option<C::G1> {
        proof.y_com.as_ref().copied().map(C::G1::from)
    }

    fn eval_commitment_gens(setup: &Self::ProverSetup) -> Option<(C::G1, C::G1)> {
        let g1_0 = setup.0.g1_vec.first().copied().map(C::G1::from)?;
        let h1 = C::G1::from(setup.0.h1);
        Some((g1_0, h1))
    }

    fn eval_commitment_gens_verifier(setup: &Self::VerifierSetup) -> Option<(C::G1, C::G1)> {
        let g1_0 = C::G1::from(setup.0.g1_0);
        let h1 = C::G1::from(setup.0.h1);
        Some((g1_0, h1))
    }
}
