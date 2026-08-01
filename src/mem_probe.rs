//! W5-U3 heap probe: counting global allocator + OOM scream.
//!
//! Wraps `System` (dlmalloc on wasm) with relaxed live/peak counters the
//! harness reads straight out of shared linear memory while the prove
//! worker is blocked, and logs the failing request size on allocation
//! failure. The wasm watermark (`memory.size`) never shrinks, so
//! live-vs-watermark is the only way to tell real residency from
//! allocator retention.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// `[0]` = live bytes, `[1]` = peak live bytes. One array so the harness
/// reads both through one exported linear-memory offset.
pub static COUNTERS: [AtomicUsize; 2] = [AtomicUsize::new(0), AtomicUsize::new(0)];

pub fn live_bytes() -> usize {
    COUNTERS[0].load(Ordering::Relaxed)
}

pub fn counters_ptr() -> u32 {
    COUNTERS.as_ptr() as u32
}

#[inline]
fn on_alloc(size: usize) {
    let live = COUNTERS[0].fetch_add(size, Ordering::Relaxed) + size;
    let _ = COUNTERS[1].fetch_max(live, Ordering::Relaxed);
}

#[inline]
fn on_dealloc(size: usize) {
    let _ = COUNTERS[0].fetch_sub(size, Ordering::Relaxed);
}

pub struct CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if p.is_null() {
            scream_oom("alloc", layout.size());
        } else {
            on_alloc(layout.size());
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if p.is_null() {
            scream_oom("alloc_zeroed", layout.size());
        } else {
            on_alloc(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        on_dealloc(layout.size());
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if p.is_null() {
            scream_oom("realloc", new_size);
        } else {
            on_dealloc(layout.size());
            on_alloc(new_size);
        }
        p
    }
}

/// No-heap OOM report: stack-buffer formatting, and `JsValue::from_str`
/// hands the wasm ptr/len to `__wbindgen_string_new` which copies on the
/// JS side — nothing here allocates from the exhausted wasm heap.
#[cold]
fn scream_oom(kind: &str, size: usize) {
    use core::fmt::Write;
    let mut buf = StackStr::<160>::new();
    let pages = core::arch::wasm32::memory_size::<0>();
    let _ = write!(
        buf,
        "[memprobe] OOM {kind} failed size={size} live={} wm_pages={pages}",
        live_bytes(),
    );
    web_sys::console::error_1(&wasm_bindgen::JsValue::from_str(buf.as_str()));
}

struct StackStr<const N: usize> {
    buf: [u8; N],
    len: usize,
}

impl<const N: usize> StackStr<N> {
    fn new() -> Self {
        Self {
            buf: [0; N],
            len: 0,
        }
    }

    fn as_str(&self) -> &str {
        core::str::from_utf8(&self.buf[..self.len]).unwrap_or("")
    }
}

impl<const N: usize> core::fmt::Write for StackStr<N> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let take = s.len().min(N - self.len);
        self.buf[self.len..self.len + take].copy_from_slice(&s.as_bytes()[..take]);
        self.len += take;
        Ok(())
    }
}
