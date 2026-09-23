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

#[cfg(all(
    target_arch = "wasm32",
    feature = "webgpu",
    feature = "digit-range-device"
))]
pub mod digit_range;
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
pub mod mailbox;
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
mod selftest;
#[cfg(all(
    target_arch = "wasm32",
    feature = "webgpu",
    feature = "relation-range-device"
))]
pub mod stage2;
#[cfg(all(
    target_arch = "wasm32",
    feature = "webgpu",
    feature = "trace-commit-device"
))]
pub mod trace_commit;

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Mutex;

/// `performance.now()` (falls back to `Date.now()`), shared by the drivers.
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
fn now_ms() -> f64 {
    use wasm_bindgen::JsCast;
    js_sys::Reflect::get(&js_sys::global(), &"performance".into())
        .map(|p| p.unchecked_into::<web_sys::Performance>().now())
        .unwrap_or_else(|_| js_sys::Date::now())
}

const STATE_DISABLED: u32 = 0;
const STATE_UNAVAILABLE: u32 = 1;
const STATE_ENABLED: u32 = 2;

static STATE: AtomicU32 = AtomicU32::new(STATE_DISABLED);

/// The proxy reported `ready`: `selftest` may enter the mailbox.
pub fn set_enabled() {
    STATE.store(STATE_ENABLED, Ordering::SeqCst);
}

/// User chose CPU: the proxy stays alive, the devices decline every job.
pub fn set_disabled() {
    STATE.store(STATE_DISABLED, Ordering::SeqCst);
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

static COMMIT_ENABLED: AtomicBool = AtomicBool::new(true);

/// Bench knob: GPU on but the stage-0 trace commit stays on the CPU.
pub fn set_commit_enabled(enabled: bool) {
    COMMIT_ENABLED.store(enabled, Ordering::SeqCst);
}

pub fn commit_enabled() -> bool {
    COMMIT_ENABLED.load(Ordering::SeqCst)
}

static DIGIT_RANGE_ENABLED: AtomicBool = AtomicBool::new(true);

/// Bench knob: GPU on but the stage-1 digit-range rounds stay on the CPU.
pub fn set_digit_range_enabled(enabled: bool) {
    DIGIT_RANGE_ENABLED.store(enabled, Ordering::SeqCst);
}

pub fn digit_range_enabled() -> bool {
    DIGIT_RANGE_ENABLED.load(Ordering::SeqCst)
}

static STAGE2_ENABLED: AtomicBool = AtomicBool::new(true);

/// Bench knob (`s2off`): GPU on but the stage-2 relation-range rounds stay on the CPU.
pub fn set_stage2_enabled(enabled: bool) {
    STAGE2_ENABLED.store(enabled, Ordering::SeqCst);
}

pub fn stage2_enabled() -> bool {
    STAGE2_ENABLED.load(Ordering::SeqCst)
}

/// What a prove run learned about the GPU before doing any field work.
#[derive(Clone, Debug, Default, serde::Serialize)]
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
    /// JSON breakdown of the GPU digit-range instances in this prove
    /// (`digit_range::Breakdown`); empty when none ran.
    pub digit_range: String,
    /// JSON breakdown of the GPU stage-2 instances in this prove
    /// (`stage2::Breakdown`); empty when none ran.
    pub stage2: String,
}

/// The session's one self-test result (`selftest`), reused by every prove.
static LAST: Mutex<Option<GpuReport>> = Mutex::new(None);

fn state_report() -> Option<GpuReport> {
    let status = match STATE.load(Ordering::SeqCst) {
        STATE_UNAVAILABLE => "unavailable",
        STATE_ENABLED => return None,
        _ => "disabled",
    };
    Some(GpuReport {
        status: status.into(),
        ..Default::default()
    })
}

/// Self-test + NOP latency probe, run once per session (the first call
/// while enabled enters the mailbox, later calls return the cached report).
/// The devices are installed on the first passing run.
pub fn selftest() -> GpuReport {
    if let Some(report) = state_report() {
        return report;
    }
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(report) = last.as_ref() {
        return report.clone();
    }
    let report = selftest_enabled();
    *last = Some(report.clone());
    report
}

/// What a prove reports about the GPU; never touches the mailbox.
pub fn status_report() -> GpuReport {
    if let Some(report) = state_report() {
        return report;
    }
    if mailbox_is_dead() {
        return GpuReport {
            status: "error: gpu proxy dead".into(),
            ..Default::default()
        };
    }
    LAST.lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
        .unwrap_or_else(|| GpuReport {
            status: "error: selftest not run".into(),
            ..Default::default()
        })
}

/// An op timed out this session: every later mailbox call is refused.
pub fn mailbox_is_dead() -> bool {
    #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
    {
        mailbox::is_dead()
    }
    #[cfg(not(all(target_arch = "wasm32", feature = "webgpu")))]
    {
        false
    }
}

pub fn set_op_timeout_ms(ms: u32) {
    #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
    mailbox::set_timeout_ms(ms);
    #[cfg(not(all(target_arch = "wasm32", feature = "webgpu")))]
    let _ = ms;
}

#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
fn selftest_enabled() -> GpuReport {
    match selftest::run() {
        Ok(report) => {
            #[cfg(feature = "trace-commit-device")]
            if report.status == "ok" {
                trace_commit::install_once();
            }
            #[cfg(feature = "digit-range-device")]
            if report.status == "ok" {
                digit_range::install_once();
            }
            #[cfg(feature = "relation-range-device")]
            if report.status == "ok" {
                stage2::install_once();
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
fn selftest_enabled() -> GpuReport {
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
            return serde_json::to_string(&b).expect("plain struct");
        }
    }
    String::new()
}

/// Breakdown of the GPU digit-range instances since the last call (JSON);
/// empty when the device is compiled out or never ran.
pub fn take_digit_range_report() -> String {
    #[cfg(all(
        target_arch = "wasm32",
        feature = "webgpu",
        feature = "digit-range-device"
    ))]
    {
        let b = digit_range::take_breakdown();
        if b.instances > 0 {
            return serde_json::to_string(&b).expect("plain struct");
        }
    }
    String::new()
}

/// Mean ms per dependent RUN_SEQ trip: one dispatch of the digit-range
/// reduce kernel over zero partials plus a 96 B inline readback — the shape
/// of one stage-2 sumcheck round without its compute.
pub fn trip_probe(n: u32) -> f64 {
    #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
    {
        use mailbox::{Region, OP_RUN_SEQ};
        if STATE.load(Ordering::SeqCst) != STATE_ENABLED || n == 0 {
            return f64::NAN;
        }
        let mut params = [0u32; 16];
        params[4] = 1;
        let param_bytes: Vec<u8> = params.iter().flat_map(|w| w.to_le_bytes()).collect();
        let partials = [0u8; 80];
        let mut out = [0u8; 96];
        // reduce kernel (shader 8): partials -> binding 3, out -> binding 4.
        let args = [1, 8, 1, 2, 3, 2, 4, 1];
        let t0 = now_ms();
        for _ in 0..n {
            let regions = [
                Region::upload(&param_bytes),
                Region::readback(&mut out),
                Region::upload(&partials),
            ];
            if mailbox::call(OP_RUN_SEQ, &args, &regions).is_err() {
                return f64::NAN;
            }
        }
        (now_ms() - t0) / f64::from(n)
    }
    #[cfg(not(all(target_arch = "wasm32", feature = "webgpu")))]
    {
        let _ = n;
        f64::NAN
    }
}

/// Breakdown of the GPU stage-2 instances since the last call (JSON);
/// empty when the device is compiled out or never ran.
pub fn take_stage2_report() -> String {
    #[cfg(all(
        target_arch = "wasm32",
        feature = "webgpu",
        feature = "relation-range-device"
    ))]
    {
        let b = stage2::take_breakdown();
        if b.instances > 0 {
            return serde_json::to_string(&b).expect("plain struct");
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
