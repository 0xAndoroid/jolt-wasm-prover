//! Streaming-bandwidth microbench exports (W2c boundary lane): measures what
//! the wasm-bindgen-rayon arm actually moves on fold-shaped work, to place
//! the GPU/CPU boundary against W1d's device numbers (0.40 Gmul/s at ~97GB/s
//! device streaming).
//!
//! All kernels report *logical* bytes per pass (semantic reads + writes).

use rayon::prelude::*;
use std::hint::black_box;
use wasm_bindgen::prelude::*;

use jolt_field::Fr;

/// Sequential-loop grain per rayon task; large enough that task overhead is
/// invisible, small enough to load-balance across P+E cores.
const CHUNK_U64: usize = 1 << 16;
const CHUNK_FR: usize = 1 << 13;

fn now_ms() -> f64 {
    use wasm_bindgen::JsCast;
    let global = js_sys::global();
    js_sys::Reflect::get(&global, &"performance".into())
        .ok()
        .and_then(|p| p.dyn_into::<web_sys::Performance>().ok())
        .map(|p| p.now())
        .unwrap_or(0.0)
}

struct PassResult {
    bytes_per_pass: u64,
    /// Field muls per pass (0 for integer kernels).
    fr_muls_per_pass: u64,
    times_ms: Vec<f64>,
    checksum: u64,
}

fn time_passes<F: FnMut() -> u64>(passes: u32, mut f: F) -> (Vec<f64>, u64) {
    let mut checksum = 0u64;
    // Warmup: first touch after allocation pays wasm memory growth + page
    // faults; keep it out of the timed window.
    checksum ^= f();
    let mut times = Vec::with_capacity(passes as usize);
    for _ in 0..passes {
        let t0 = now_ms();
        checksum ^= f();
        times.push(now_ms() - t0);
    }
    (times, checksum)
}

fn bench_copy_u64(len: usize, passes: u32) -> PassResult {
    let src: Vec<u64> = (0..len as u64).collect();
    let mut dst: Vec<u64> = vec![0u64; len];
    let (times_ms, checksum) = time_passes(passes, || {
        dst.par_chunks_mut(CHUNK_U64)
            .zip(src.par_chunks(CHUNK_U64))
            .for_each(|(d, s)| d.copy_from_slice(s));
        black_box(dst[len / 2])
    });
    PassResult {
        bytes_per_pass: (len as u64) * 16,
        fr_muls_per_pass: 0,
        times_ms,
        checksum,
    }
}

fn bench_sum_u64(len: usize, passes: u32) -> PassResult {
    let src: Vec<u64> = (0..len as u64)
        .map(|i| i.wrapping_mul(0x9E37_79B9))
        .collect();
    let (times_ms, checksum) = time_passes(passes, || {
        src.par_chunks(CHUNK_U64)
            .map(|c| c.iter().fold(0u64, |s, &x| s.wrapping_add(x)))
            .reduce(|| 0u64, u64::wrapping_add)
    });
    PassResult {
        bytes_per_pass: (len as u64) * 8,
        fr_muls_per_pass: 0,
        times_ms,
        checksum,
    }
}

fn bench_scale_u64(len: usize, passes: u32) -> PassResult {
    let mut v: Vec<u64> = (0..len as u64).collect();
    let k = black_box(0x9E37_79B9_7F4A_7C15u64);
    let (times_ms, checksum) = time_passes(passes, || {
        v.par_chunks_mut(CHUNK_U64).for_each(|c| {
            c.iter_mut()
                .for_each(|x| *x = x.wrapping_mul(k).wrapping_add(1))
        });
        black_box(v[len / 2])
    });
    PassResult {
        bytes_per_pass: (len as u64) * 16,
        fr_muls_per_pass: 0,
        times_ms,
        checksum,
    }
}

fn bench_triad_u64(len: usize, passes: u32) -> PassResult {
    let b: Vec<u64> = (0..len as u64).collect();
    let c: Vec<u64> = (0..len as u64).map(|i| i ^ 0xA5A5).collect();
    let mut a: Vec<u64> = vec![0u64; len];
    let k = black_box(3u64);
    let (times_ms, checksum) = time_passes(passes, || {
        a.par_chunks_mut(CHUNK_U64)
            .zip(b.par_chunks(CHUNK_U64).zip(c.par_chunks(CHUNK_U64)))
            .for_each(|(av, (bv, cv))| {
                for ((x, &y), &z) in av.iter_mut().zip(bv).zip(cv) {
                    *x = y.wrapping_mul(k).wrapping_add(z);
                }
            });
        black_box(a[len / 2])
    });
    PassResult {
        bytes_per_pass: (len as u64) * 24,
        fr_muls_per_pass: 0,
        times_ms,
        checksum,
    }
}

fn fr_material(len: usize) -> Vec<Fr> {
    (0..len)
        .into_par_iter()
        .map(|i| Fr::from(0x1234_5678_9ABC_DEF0u64 ^ (i as u64 + 1)))
        .collect()
}

fn fr_word(x: &Fr) -> u64 {
    // First limb of the Montgomery representation — cheap checksum probe.
    unsafe { *(x as *const Fr as *const u64) }
}

/// bound_poly-style fold: dst[i] = src[2i] + r*(src[2i+1] - src[2i]).
/// Per output element: 64B read + 32B write, 1 Fr mul.
fn bench_fold_fr(len: usize, passes: u32) -> PassResult {
    let src = fr_material(len);
    let half = len / 2;
    let mut dst: Vec<Fr> = vec![Fr::from(0u64); half];
    let r = black_box(Fr::from(0xDEAD_BEEF_1234_5678u64));
    let (times_ms, checksum) = time_passes(passes, || {
        dst.par_chunks_mut(CHUNK_FR)
            .enumerate()
            .for_each(|(ci, dch)| {
                let base = ci * CHUNK_FR;
                for (j, d) in dch.iter_mut().enumerate() {
                    let lo = src[2 * (base + j)];
                    let hi = src[2 * (base + j) + 1];
                    *d = lo + r * (hi - lo);
                }
            });
        fr_word(&dst[half / 2])
    });
    PassResult {
        bytes_per_pass: (half as u64) * 96,
        fr_muls_per_pass: half as u64,
        times_ms,
        checksum,
    }
}

/// Pure streaming scale: v[i] = r*v[i]. 32B read + 32B write, 1 Fr mul per
/// element — the closest CPU analogue of W1d's streaming mont-mul shape.
fn bench_scale_fr(len: usize, passes: u32) -> PassResult {
    let mut v = fr_material(len);
    let r = black_box(Fr::from(0xC0FF_EE00_DDBA_11ADu64));
    let (times_ms, checksum) = time_passes(passes, || {
        v.par_chunks_mut(CHUNK_FR)
            .for_each(|c| c.iter_mut().for_each(|x| *x *= r));
        fr_word(&v[len / 2])
    });
    PassResult {
        bytes_per_pass: (len as u64) * 64,
        fr_muls_per_pass: len as u64,
        times_ms,
        checksum,
    }
}

/// ALU-dense anchor: same array, but a chain of DEPTH dependent muls per
/// element — measures the wasm arm's mul throughput when not bus-bound
/// (CPU-side twin of W1d's dependent-chain kernel).
fn bench_mulchain_fr(len: usize, passes: u32) -> PassResult {
    const DEPTH: u64 = 64;
    let mut v = fr_material(len);
    let r = black_box(Fr::from(0x0123_4567_89AB_CDEFu64));
    let (times_ms, checksum) = time_passes(passes, || {
        v.par_chunks_mut(CHUNK_FR).for_each(|c| {
            c.iter_mut().for_each(|x| {
                let mut acc = *x;
                for _ in 0..DEPTH {
                    acc *= r;
                }
                *x = acc;
            })
        });
        fr_word(&v[len / 2])
    });
    PassResult {
        bytes_per_pass: (len as u64) * 64,
        fr_muls_per_pass: (len as u64) * DEPTH,
        times_ms,
        checksum,
    }
}

/// Runs one kernel and returns a JSON report. `log2_len` is the element
/// count of the primary array (u64 or Fr elements depending on kind).
#[wasm_bindgen]
pub fn bench_stream(kind: &str, log2_len: u32, passes: u32) -> String {
    let len = 1usize << log2_len;
    let result = match kind {
        "copy_u64" => bench_copy_u64(len, passes),
        "sum_u64" => bench_sum_u64(len, passes),
        "scale_u64" => bench_scale_u64(len, passes),
        "triad_u64" => bench_triad_u64(len, passes),
        "fold_fr" => bench_fold_fr(len, passes),
        "scale_fr" => bench_scale_fr(len, passes),
        "mulchain_fr" => bench_mulchain_fr(len, passes),
        _ => return format!("{{\"error\":\"unknown kind {kind}\"}}"),
    };
    let json = serde_json::json!({
        "kind": kind,
        "len": len,
        "threads": rayon::current_num_threads(),
        "bytes_per_pass": result.bytes_per_pass,
        "fr_muls_per_pass": result.fr_muls_per_pass,
        "times_ms": result.times_ms,
        "checksum": format!("{:016x}", result.checksum),
    });
    json.to_string()
}
