//! Shared-memory mailbox between the prover threads and `gpu-proxy.js`.
//! Layout is mirrored by hand in the proxy (u32 word offsets) — keep both in
//! sync.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Mutex;

use js_sys::{Atomics, Int32Array, WebAssembly};
use wasm_bindgen::JsCast;

pub const OP_NOP: u32 = 1;
pub const OP_CREATE_BUFFER: u32 = 2;
pub const OP_UPLOAD: u32 = 3;
pub const OP_DESTROY: u32 = 4;
pub const OP_RUN: u32 = 5;
/// Several RUN passes in one trip: `args = [npasses, {shader, wx, nbind, region_idx…}…]`,
/// region 0 = the packed per-pass 64 B uniform blocks. `args[RUN_SEQ_COPY..]`
/// = `[src_region, dst_region, dst_byte_offset, byte_len]` appends a
/// buffer-to-buffer copy after the passes (`byte_len` 0 = none).
pub const OP_RUN_SEQ: u32 = 6;
/// Copy a device buffer (`args = [handle, byte_len]`) into the READBACK region.
pub const OP_DOWNLOAD: u32 = 7;
/// Destroy then create handles in one trip: `args = [n_destroy, handles…, n_create, sizes…]`,
/// the new handles come back in RET (≤ 8).
pub const OP_ALLOC: u32 = 8;
/// Several uploads in one trip: `args = [n, {handle, byte_offset, region}…]`.
pub const OP_UPLOAD_MULTI: u32 = 9;
pub const RUN_SEQ_COPY: usize = 28;

pub const STATUS_IDLE: u32 = 0;
pub const STATUS_BUSY: u32 = 1;
pub const STATUS_DONE: u32 = 2;
pub const STATUS_ERROR: u32 = 3;

/// Region flags: the proxy copies `upload` regions into a fresh storage
/// buffer before the dispatch and writes `readback` regions back into wasm
/// memory after it; `handle` means `ptr` is a persistent buffer handle.
pub const REGION_UPLOAD: u32 = 1;
pub const REGION_READBACK: u32 = 2;
pub const REGION_HANDLE: u32 = 4;

pub const MAX_ARGS: usize = 32;
pub const MAX_REGIONS: usize = 8;
const ERROR_WORDS: usize = 64;

const WORD_DOORBELL: u32 = 0;
const WORD_STATUS: u32 = 1;

#[repr(C)]
pub struct Mailbox {
    doorbell: AtomicU32,
    status: AtomicU32,
    op: AtomicU32,
    _pad: AtomicU32,
    args: [AtomicU32; MAX_ARGS],
    regions: [AtomicU32; MAX_REGIONS * 3],
    ret: [AtomicU32; 8],
    error_len: AtomicU32,
    error: [AtomicU32; ERROR_WORDS],
}

static MAILBOX: Mailbox = Mailbox {
    doorbell: AtomicU32::new(0),
    status: AtomicU32::new(0),
    op: AtomicU32::new(0),
    _pad: AtomicU32::new(0),
    args: [const { AtomicU32::new(0) }; MAX_ARGS],
    regions: [const { AtomicU32::new(0) }; MAX_REGIONS * 3],
    ret: [const { AtomicU32::new(0) }; 8],
    error_len: AtomicU32::new(0),
    error: [const { AtomicU32::new(0) }; ERROR_WORDS],
};

/// One in-flight op at a time: the mailbox has a single set of slots.
static LOCK: Mutex<()> = Mutex::new(());
/// Per-op deadline; W1's accumulate at 2^21 is the longest op.
static TIMEOUT_MS: AtomicU32 = AtomicU32::new(30_000);
/// Set once an op timed out: the proxy may still write into the abandoned
/// regions, so every later call is refused for the rest of the session.
static DEAD: AtomicBool = AtomicBool::new(false);

pub fn set_timeout_ms(ms: u32) {
    TIMEOUT_MS.store(ms.max(1), Ordering::SeqCst);
}

pub fn is_dead() -> bool {
    DEAD.load(Ordering::SeqCst)
}

pub fn ptr() -> u32 {
    &MAILBOX as *const Mailbox as u32
}

#[derive(Clone, Copy, Debug)]
pub struct Region {
    pub ptr: u32,
    pub len: u32,
    pub flags: u32,
}

impl Region {
    pub fn upload(bytes: &[u8]) -> Self {
        Self {
            ptr: bytes.as_ptr() as u32,
            len: bytes.len() as u32,
            flags: REGION_UPLOAD,
        }
    }

    pub fn readback(bytes: &mut [u8]) -> Self {
        Self {
            ptr: bytes.as_mut_ptr() as u32,
            len: bytes.len() as u32,
            flags: REGION_READBACK,
        }
    }

    pub fn handle(handle: u32) -> Self {
        Self {
            ptr: handle,
            len: 0,
            flags: REGION_HANDLE,
        }
    }
}

#[derive(Debug)]
pub struct GpuError(pub String);

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

fn mailbox_view() -> Int32Array {
    let memory = wasm_bindgen::memory().unchecked_into::<WebAssembly::Memory>();
    Int32Array::new(&memory.buffer())
}

/// Fill the mailbox, ring the proxy and block this thread until it answers.
/// `regions` are read/written by the proxy while we wait, so `readback`
/// buffers must stay alive and untouched until this returns.
pub fn call(op: u32, args: &[u32], regions: &[Region]) -> Result<[u32; 8], GpuError> {
    assert!(args.len() <= MAX_ARGS && regions.len() <= MAX_REGIONS);
    let _guard = LOCK
        .lock()
        .map_err(|_| GpuError("mailbox lock poisoned".into()))?;
    // Checked under the lock: a caller queued behind the op that timed out
    // must not ring a mailbox the proxy may still be servicing.
    if is_dead() {
        return Err(GpuError("gpu proxy dead".into()));
    }
    let m = &MAILBOX;
    for (slot, v) in m.args.iter().zip(args.iter().chain(std::iter::repeat(&0))) {
        slot.store(*v, Ordering::Relaxed);
    }
    for (i, slot) in m.regions.chunks(3).enumerate() {
        let r = regions.get(i).copied().unwrap_or(Region {
            ptr: 0,
            len: 0,
            flags: 0,
        });
        slot[0].store(r.ptr, Ordering::Relaxed);
        slot[1].store(r.len, Ordering::Relaxed);
        slot[2].store(r.flags, Ordering::Relaxed);
    }
    m.op.store(op, Ordering::Relaxed);
    m.status.store(STATUS_BUSY, Ordering::SeqCst);
    m.doorbell.fetch_add(1, Ordering::SeqCst);

    let view = mailbox_view();
    let base = ptr() / 4;
    Atomics::notify(&view, base + WORD_DOORBELL)
        .map_err(|e| GpuError(format!("Atomics.notify failed: {e:?}")))?;
    // A timed-out op is abandoned, not cancelled: the proxy may still write
    // the `readback` regions later, so callers must leak them when
    // `is_dead()` and the session never enters the mailbox again.
    let timeout = f64::from(TIMEOUT_MS.load(Ordering::SeqCst));
    let deadline = super::now_ms() + timeout;
    while m.status.load(Ordering::SeqCst) == STATUS_BUSY {
        let remaining = deadline - super::now_ms();
        if remaining <= 0.0 {
            DEAD.store(true, Ordering::SeqCst);
            return Err(GpuError(format!(
                "gpu proxy unresponsive after {timeout:.0} ms (op {op})"
            )));
        }
        // Returns "ok" | "not-equal" | "timed-out"; all re-check.
        Atomics::wait_with_timeout(&view, base + WORD_STATUS, STATUS_BUSY as i32, remaining)
            .map_err(|e| GpuError(format!("Atomics.wait failed: {e:?}")))?;
    }
    let status = m.status.swap(STATUS_IDLE, Ordering::SeqCst);
    if status == STATUS_DONE {
        let mut ret = [0u32; 8];
        for (dst, slot) in ret.iter_mut().zip(m.ret.iter()) {
            *dst = slot.load(Ordering::Relaxed);
        }
        return Ok(ret);
    }
    let len = (m.error_len.load(Ordering::Relaxed) as usize).min(ERROR_WORDS * 4);
    let mut bytes = Vec::with_capacity(len);
    for slot in m.error.iter().take(len.div_ceil(4)) {
        bytes.extend_from_slice(&slot.load(Ordering::Relaxed).to_le_bytes());
    }
    bytes.truncate(len);
    Err(GpuError(format!(
        "gpu op {op} failed (status {status}): {}",
        String::from_utf8_lossy(&bytes)
    )))
}
