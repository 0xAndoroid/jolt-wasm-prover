//! WebGPU offload harness (cargo feature `webgpu`, wasm32 only).
//!
//! The prover runs synchronously on Web Workers that share one
//! `WebAssembly.Memory`; a blocked thread cannot service WebGPU promises, so
//! the `GPUDevice` lives in a separate dedicated Worker (`gpu-proxy.js`).
//! Both sides talk through [`mailbox::Mailbox`], a `#[repr(C)]` static in
//! wasm memory: Rust fills op/args/regions, bumps the doorbell and blocks in
//! `Atomics.wait` until the proxy flips `status`. Without the feature (or
//! natively) every entry point reports the GPU as disabled and nothing else
//! changes.

#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
pub mod mailbox;
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
mod selftest;
#[cfg(all(
    target_arch = "wasm32",
    feature = "webgpu",
    feature = "trace-commit-device"
))]
pub mod trace_commit;

use std::sync::atomic::{AtomicU32, Ordering};

const STATE_DISABLED: u32 = 0;
const STATE_UNAVAILABLE: u32 = 1;
const STATE_ENABLED: u32 = 2;

static STATE: AtomicU32 = AtomicU32::new(STATE_DISABLED);

/// The proxy reported `ready`: `preflight` may enter the mailbox.
pub fn set_enabled() {
    STATE.store(STATE_ENABLED, Ordering::SeqCst);
}

/// JS asked for the GPU but the proxy found no adapter (or the feature is
/// compiled out): proving stays on the CPU and reports `unavailable`.
pub fn set_unavailable() {
    STATE.store(STATE_UNAVAILABLE, Ordering::SeqCst);
}

/// The installed `TraceCommitDevice` is a process-wide `OnceLock`, but the GPU
/// toggle is per prove call: the device consults this on every job so a
/// `gpu=false` run after a `gpu=true` run really stays on the CPU.
pub fn is_enabled() -> bool {
    STATE.load(Ordering::SeqCst) == STATE_ENABLED
}

/// What a prove run learned about the GPU before doing any field work.
#[derive(Clone, Debug, Default)]
pub struct GpuReport {
    /// `disabled` | `unavailable` | `ok` | `error: …`
    pub status: String,
    pub selftest_ms: f64,
    pub selftest_mismatches: u32,
    /// Mean NOP mailbox round trip over 200 calls, microseconds (browser
    /// clocks are ~1 ms coarse, so a per-call median is meaningless).
    pub roundtrip_us: f64,
    /// JSON stage breakdown of the GPU trace commits in this prove
    /// (`trace_commit::CommitBreakdown`); empty when none ran.
    pub commit: String,
}

/// Self-test + NOP latency probe; only touches the mailbox when enabled.
pub fn preflight() -> GpuReport {
    match STATE.load(Ordering::SeqCst) {
        STATE_UNAVAILABLE => GpuReport {
            status: "unavailable".into(),
            ..Default::default()
        },
        STATE_ENABLED => preflight_enabled(),
        _ => GpuReport {
            status: "disabled".into(),
            ..Default::default()
        },
    }
}

#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
fn preflight_enabled() -> GpuReport {
    match selftest::run() {
        Ok(report) => {
            #[cfg(feature = "trace-commit-device")]
            if report.status == "ok" {
                trace_commit::install_once();
            }
            report
        }
        Err(e) => GpuReport {
            status: format!("error: {e}"),
            ..Default::default()
        },
    }
}

#[cfg(not(all(target_arch = "wasm32", feature = "webgpu")))]
fn preflight_enabled() -> GpuReport {
    GpuReport {
        status: "unavailable".into(),
        ..Default::default()
    }
}

/// Stage breakdown of the GPU trace commits since the last call (JSON);
/// empty when the device is compiled out or never ran.
pub fn take_commit_report() -> String {
    #[cfg(all(
        target_arch = "wasm32",
        feature = "webgpu",
        feature = "trace-commit-device"
    ))]
    {
        let b = trace_commit::take_breakdown();
        if b.calls > 0 {
            return b.to_json();
        }
    }
    String::new()
}

/// Address of the mailbox for `gpu-proxy.js`; 0 when compiled out.
pub fn mailbox_ptr() -> u32 {
    #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
    {
        mailbox::ptr()
    }
    #[cfg(not(all(target_arch = "wasm32", feature = "webgpu")))]
    {
        0
    }
}
