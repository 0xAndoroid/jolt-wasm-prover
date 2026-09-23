//! Stage-1 digit-range sumcheck rounds on the GPU: `akita_prover::DigitRangeDevice`
//! backed by the kernels in `frontend/public/wgsl/digit_range/` (layout
//! contract in `bench/proto-w2/README.md` of the kernel prototype).
//!
//! One instance = one UPLOAD (digits, plus lut0 on fresh handles) → per
//! round one RUN_SEQ (params + this round's Gruen tables as inline uploads,
//! the 80 B message as inline readback; the last round also copies the
//! folded table for the CPU tail into that readback). Digits, the two
//! ping-pong tables, the three LUTs and the partials scratch are persistent
//! handles (re)allocated in one ALLOC trip and owned by the live session.

use serde::Serialize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Once};

use akita_prover::{
    DigitRangeDevice, DigitRangeJob, DigitRangeSession, DIGIT_RANGE_FIELD_BYTES as FIELD_BYTES,
    DIGIT_RANGE_MESSAGE_BYTES as MESSAGE_BYTES,
};
use jolt_akita::AkitaError;

use super::mailbox::{
    self, GpuError, Region, OP_ALLOC, OP_DOWNLOAD, OP_RUN_SEQ, OP_UPLOAD_MULTI, RUN_SEQ_COPY,
};
use super::now_ms;

/// Shader ids: index into `SHADERS` in `gpu-proxy.js`.
const SHADER_ROUND0: u32 = 4;
const SHADER_LUT: u32 = 5;
const SHADER_ROUND1: u32 = 6;
const SHADER_FIELD: u32 = 7;
const SHADER_REDUCE: u32 = 8;
const WG: u32 = 256;
/// Round-1 histogram segment (`blk` in round1.wgsl).
const ROUND1_BLK: u32 = 2048;
/// Per-thread work saturates the GPU at 2^18 units per dispatch.
const UNITS_PER_THREAD_TARGET: u32 = WG * 1024;
const PARAMS_BYTES: usize = 64;
/// Smallest instance worth the trips; the CPU tail keeps `2^CPU_TAIL_BITS` entries.
const MIN_LEN_BITS: u32 = 16;
const CPU_TAIL_BITS: u32 = 12;
/// Rounds 0..3 read the packed digits; a table only exists from round 3 on.
const MIN_ROUNDS: usize = 4;
const RANGE_V: [i64; 4] = [0, 2, 6, 12];

/// Region slots of one round's RUN_SEQ; slot 0 is the params block the proxy
/// reads directly.
const R_EQ: u32 = 1;
const R_OUT: u32 = 2;
const R_DIGITS: u32 = 3;
const R_PARTIALS: u32 = 4;
const R_LUT: u32 = 5;
const R_TA: u32 = 6;
const R_TB: u32 = 7;

struct Handles {
    digits_cap: u32,
    /// Capacity in digits of the tables/partials.
    n_cap: u32,
    /// Workgroups the partials scratch can hold.
    max_wg: u32,
    digits: u32,
    ta: u32,
    tb: u32,
    lut0: u32,
    lut1: u32,
    lut2f: u32,
    partials: u32,
}

impl Handles {
    fn all(&self) -> [u32; 7] {
        [
            self.digits,
            self.ta,
            self.tb,
            self.lut0,
            self.lut1,
            self.lut2f,
            self.partials,
        ]
    }
}

pub struct WebGpuDigitRange;

/// Persistent handles between sessions; the live session owns them meanwhile.
static HANDLES: Mutex<Option<Handles>> = Mutex::new(None);
static SESSION_OPEN: AtomicBool = AtomicBool::new(false);

/// Summed over every instance of one prove; `take()` resets.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Breakdown {
    pub instances: u32,
    pub gpu_rounds: u32,
    pub ops: u32,
    pub upload_ms: f64,
    pub rounds_ms: f64,
    pub download_ms: f64,
    pub total_ms: f64,
}

static BREAKDOWN: Mutex<Breakdown> = Mutex::new(Breakdown {
    instances: 0,
    gpu_rounds: 0,
    ops: 0,
    upload_ms: 0.0,
    rounds_ms: 0.0,
    download_ms: 0.0,
    total_ms: 0.0,
});

pub fn take_breakdown() -> Breakdown {
    std::mem::take(&mut *BREAKDOWN.lock().unwrap())
}

static INSTALL: Once = Once::new();

/// Installs the device into akita-prover's process-global slot exactly once
/// per worker; gpu on/off per prove is honoured inside `open`.
pub fn install_once() {
    INSTALL.call_once(|| {
        let message = match crate::engine::install_digit_range_device(Arc::new(WebGpuDigitRange)) {
            Ok(()) => "[gpu] digit range device installed".to_string(),
            Err(e) => format!("[gpu] digit range device not installed: {e}"),
        };
        web_sys::console::log_1(&message.into());
    });
}

/// Coefficients of `Q(L + (R - L) X)` for `L = V[a]`, `R = V[b]`, as i32 in
/// the round-0 layout `lut0[c * 16 + (a | b << 2)]`.
fn lut0_bytes() -> Vec<u8> {
    let mut out = Vec::with_capacity(80 * 4);
    for c in 0..5 {
        for cp in 0..16 {
            let l = RANGE_V[cp & 3];
            let d = RANGE_V[cp >> 2] - l;
            let (fq, sq) = (l * l - 2 * l, l * l - 18 * l + 72);
            let (fl, sl, d2) = (d * (2 * l - 2), d * (2 * l - 18), d * d);
            let coeff = [
                fq * sq,
                fq * sl + fl * sq,
                fq * d2 + fl * sl + d2 * sq,
                d2 * (fl + sl),
                d2 * d2,
            ][c];
            out.extend_from_slice(&(i32::try_from(coeff).unwrap()).to_le_bytes());
        }
    }
    out
}

/// Largest power of two ≤ 8 keeping ≥ `UNITS_PER_THREAD_TARGET` threads busy.
fn units_per_thread(units: u32) -> u32 {
    let mut ppt = 8;
    while ppt > 1 && units < UNITS_PER_THREAD_TARGET * ppt {
        ppt /= 2;
    }
    ppt
}

/// `true` when the handles were just allocated (lut0 still needs uploading).
fn handles_for(
    slot: &mut Option<Handles>,
    digits_bytes: u32,
    n: u32,
) -> Result<(Handles, bool), GpuError> {
    let stale = slot.take_if(|h| h.digits_cap < digits_bytes || h.n_cap < n);
    if let Some(h) = slot.take() {
        return Ok((h, false));
    }
    // Round 1 needs n/4/blk workgroups where blk = min(|E_first|, 2048) and
    // |E_first| ≥ 2^((num_vars-1)/2 - 1); round 0 needs at most n/8/256.
    let inner1_bits = (n.ilog2().saturating_sub(1)) / 2 - 1;
    let blk1 = (1u32 << inner1_bits).min(ROUND1_BLK);
    let max_wg = (n / 4).div_ceil(blk1).max((n / 8).div_ceil(WG)) + 1;
    let sizes = [
        // +4: the decoder may read one word past the last digit.
        digits_bytes + 4,
        n / 8 * FIELD_BYTES as u32,
        n / 16 * FIELD_BYTES as u32,
        80 * 4,
        256 * MESSAGE_BYTES as u32,
        256 * FIELD_BYTES as u32,
        max_wg * MESSAGE_BYTES as u32,
    ];
    let old = stale.map(|h| h.all().to_vec()).unwrap_or_default();
    let mut args = vec![old.len() as u32];
    args.extend_from_slice(&old);
    args.push(sizes.len() as u32);
    args.extend_from_slice(&sizes);
    let created = mailbox::call(OP_ALLOC, &args, &[])?;
    let [digits, ta, tb, lut0, lut1, lut2f, partials] = created[..sizes.len()].try_into().unwrap();
    Ok((
        Handles {
            digits_cap: digits_bytes,
            n_cap: n,
            max_wg,
            digits,
            ta,
            tb,
            lut0,
            lut1,
            lut2f,
            partials,
        },
        true,
    ))
}

/// Best effort: nothing is left to recover when the destroy itself fails.
fn destroy(handles: &[u32]) {
    let mut args = vec![handles.len() as u32];
    args.extend_from_slice(handles);
    args.push(0);
    let _ = mailbox::call(OP_ALLOC, &args, &[]);
}

struct Session {
    handles: Option<Handles>,
    n: u32,
    rounds: usize,
    bit_width: u32,
    r0: [u8; FIELD_BYTES],
    r1: [u8; FIELD_BYTES],
    /// Table handle written by the last round (`None` before round 3).
    last_dst: Option<u32>,
    /// Folded table read back with the last round's message.
    table: Option<Vec<u8>>,
    ops: u32,
    upload_ms: f64,
    rounds_ms: f64,
    t_open: f64,
}

/// One RUN_SEQ pass: `[shader, workgroups, nbind, (binding, region)…]`.
struct Pass {
    shader: u32,
    wgs: u32,
    binds: &'static [(u32, u32)],
    params: [u32; 16],
}

#[allow(clippy::too_many_arguments)]
fn params(
    n_units: u32,
    inner_bits: u32,
    off_second: u32,
    ppt: u32,
    bit_width: u32,
    src_mode: u32,
    case_c: u32,
    r: &[u8; FIELD_BYTES],
    r_aux: &[u8; FIELD_BYTES],
) -> [u32; 16] {
    let mut p = [
        n_units, inner_bits, 0, off_second, ppt, bit_width, src_mode, case_c, 0, 0, 0, 0, 0, 0, 0,
        0,
    ];
    for (i, limb) in r.chunks_exact(4).enumerate() {
        p[8 + i] = u32::from_le_bytes(limb.try_into().unwrap());
    }
    for (i, limb) in r_aux.chunks_exact(4).enumerate() {
        p[12 + i] = u32::from_le_bytes(limb.try_into().unwrap());
    }
    p
}

/// round0 / round1: digits, eq, partials, lut.
const BINDS_HIST: &[(u32, u32)] = &[(1, R_DIGITS), (2, R_EQ), (3, R_PARTIALS), (4, R_LUT)];
const BINDS_LUT: &[(u32, u32)] = &[(4, R_LUT)];
const BINDS_FIELD_TA_DST: &[(u32, u32)] = &[
    (1, R_DIGITS),
    (2, R_EQ),
    (3, R_PARTIALS),
    (4, R_TB),
    (5, R_TA),
    (6, R_LUT),
];
const BINDS_FIELD_TB_DST: &[(u32, u32)] = &[
    (1, R_DIGITS),
    (2, R_EQ),
    (3, R_PARTIALS),
    (4, R_TA),
    (5, R_TB),
    (6, R_LUT),
];
const BINDS_REDUCE: &[(u32, u32)] = &[(3, R_PARTIALS), (4, R_OUT)];

/// Every exit — `table()`, a failed round or download, an abandoned prove —
/// hands the buffers back so the next qualifying instance can open.
impl Drop for Session {
    fn drop(&mut self) {
        *HANDLES.lock().unwrap() = self.handles.take();
        SESSION_OPEN.store(false, Ordering::SeqCst);
    }
}

impl Session {
    fn handles(&self) -> &Handles {
        self.handles
            .as_ref()
            .expect("session owns the handles until table()")
    }

    fn run_round(
        &mut self,
        round: usize,
        prev: Option<&[u8; FIELD_BYTES]>,
        e_first: &[u8],
        e_second: &[u8],
        out: &mut [u8; MESSAGE_BYTES],
    ) -> Result<(), GpuError> {
        let t0 = now_ms();
        let last = round + 1 == self.rounds;
        let zero = [0u8; FIELD_BYTES];
        let prev = prev.unwrap_or(&zero);
        let n = self.n;
        let bw = self.bit_width;
        let inner = (e_first.len() / FIELD_BYTES) as u32;
        let inner_bits = inner.ilog2();
        let off_second = inner;
        let mut passes: Vec<Pass> = Vec::with_capacity(3);
        let (main_wgs, lut_handle, table_dst, table_region);
        match round {
            0 => {
                if inner_bits < 2 {
                    return Err(GpuError("round 0 needs |E_first| >= 4".into()));
                }
                let units = n / 8;
                let ppt = units_per_thread(units).min(inner / 4);
                main_wgs = units.div_ceil(WG * ppt);
                lut_handle = self.handles().lut0;
                (table_dst, table_region) = (None, R_TA);
                passes.push(Pass {
                    shader: SHADER_ROUND0,
                    wgs: main_wgs,
                    binds: BINDS_HIST,
                    params: params(units, inner_bits, off_second, ppt, bw, 0, 0, prev, &zero),
                });
            }
            1 => {
                self.r0 = *prev;
                let units = n / 4;
                let blk = inner.min(ROUND1_BLK);
                main_wgs = units.div_ceil(blk);
                lut_handle = self.handles().lut1;
                (table_dst, table_region) = (None, R_TA);
                passes.push(Pass {
                    shader: SHADER_LUT,
                    wgs: 1,
                    binds: BINDS_LUT,
                    params: params(256, 0, 0, 1, bw, 0, 0, &self.r0, &zero),
                });
                passes.push(Pass {
                    shader: SHADER_ROUND1,
                    wgs: main_wgs,
                    binds: BINDS_HIST,
                    params: params(units, inner_bits, off_second, 1, bw, 0, 0, prev, &zero),
                });
            }
            2 => {
                self.r1 = *prev;
                let units = n / 8;
                let ppt = units_per_thread(units);
                main_wgs = units.div_ceil(WG * ppt);
                lut_handle = self.handles().lut2f;
                (table_dst, table_region) = (None, R_TA);
                passes.push(Pass {
                    shader: SHADER_LUT,
                    wgs: 1,
                    binds: BINDS_LUT,
                    params: params(256, 0, 0, 1, bw, 1, 0, &self.r0, &self.r1),
                });
                passes.push(Pass {
                    shader: SHADER_FIELD,
                    wgs: main_wgs,
                    binds: BINDS_FIELD_TA_DST,
                    params: params(
                        units,
                        inner_bits,
                        off_second,
                        ppt,
                        bw,
                        2,
                        (inner < ppt) as u32,
                        prev,
                        &zero,
                    ),
                });
            }
            k => {
                let units = n >> (k + 1);
                let ppt = units_per_thread(units);
                main_wgs = units.div_ceil(WG * ppt);
                lut_handle = self.handles().lut2f;
                let (src_mode, binds, dst, region) = match k {
                    3 => (1, BINDS_FIELD_TA_DST, self.handles().ta, R_TA),
                    k if k % 2 == 0 => (0, BINDS_FIELD_TB_DST, self.handles().tb, R_TB),
                    _ => (0, BINDS_FIELD_TA_DST, self.handles().ta, R_TA),
                };
                (table_dst, table_region) = (Some(dst), region);
                passes.push(Pass {
                    shader: SHADER_FIELD,
                    wgs: main_wgs,
                    binds,
                    params: params(
                        units,
                        inner_bits,
                        off_second,
                        ppt,
                        bw,
                        src_mode,
                        (inner < ppt) as u32,
                        prev,
                        &zero,
                    ),
                });
            }
        }
        if main_wgs > self.handles().max_wg {
            return Err(GpuError(format!(
                "round {round} needs {main_wgs} workgroups, partials hold {}",
                self.handles().max_wg
            )));
        }
        passes.push(Pass {
            shader: SHADER_REDUCE,
            wgs: 1,
            binds: BINDS_REDUCE,
            params: params(main_wgs, 0, 0, 1, bw, 0, 0, &zero, &zero),
        });

        let mut param_bytes = Vec::with_capacity(passes.len() * PARAMS_BYTES);
        let mut args = vec![passes.len() as u32];
        for pass in &passes {
            for word in pass.params {
                param_bytes.extend_from_slice(&word.to_le_bytes());
            }
            args.extend_from_slice(&[pass.shader, pass.wgs, pass.binds.len() as u32]);
            for &(binding, region) in pass.binds {
                args.extend_from_slice(&[binding, region]);
            }
        }
        // The last round's readback also carries the folded table (copied
        // after the passes), which saves the DOWNLOAD trip.
        let table_len = if last && table_dst.is_some() {
            (self.n >> (self.rounds - 1)) as usize * FIELD_BYTES
        } else {
            0
        };
        if table_len > 0 {
            assert!(args.len() <= RUN_SEQ_COPY);
            args.resize(RUN_SEQ_COPY, 0);
            args.extend_from_slice(&[table_region, R_OUT, MESSAGE_BYTES as u32, table_len as u32]);
        }
        let mut eq = Vec::with_capacity(e_first.len() + e_second.len());
        eq.extend_from_slice(e_first);
        eq.extend_from_slice(e_second);
        let mut buf = vec![0u8; MESSAGE_BYTES + table_len];
        let h = self.handles();
        let regions = [
            Region::upload(&param_bytes),
            Region::upload(&eq),
            Region::readback(&mut buf),
            Region::handle(h.digits),
            Region::handle(h.partials),
            Region::handle(lut_handle),
            Region::handle(h.ta),
            Region::handle(h.tb),
        ];
        if let Err(e) = mailbox::call(OP_RUN_SEQ, &args, &regions) {
            if mailbox::is_dead() {
                // The proxy may still write `buf`; leaking it keeps that write harmless.
                std::mem::forget(buf);
            }
            return Err(e);
        }
        out.copy_from_slice(&buf[..MESSAGE_BYTES]);
        if table_len > 0 {
            buf.drain(..MESSAGE_BYTES);
            self.table = Some(buf);
        }
        if table_dst.is_some() {
            self.last_dst = table_dst;
        }
        self.ops += 1;
        self.rounds_ms += now_ms() - t0;
        Ok(())
    }

    /// Only enters the mailbox when `table()` comes before the last round.
    fn download_table(&mut self) -> Result<Vec<u8>, GpuError> {
        if let Some(table) = self.table.take() {
            return Ok(table);
        }
        let dst = self
            .last_dst
            .ok_or_else(|| GpuError("table requested before round 3".into()))?;
        let entries = self.n >> (self.rounds - 1);
        let mut table = vec![0u8; entries as usize * FIELD_BYTES];
        if let Err(e) = mailbox::call(
            OP_DOWNLOAD,
            &[dst, table.len() as u32],
            &[Region::readback(&mut table)],
        ) {
            if mailbox::is_dead() {
                std::mem::forget(table);
            }
            return Err(e);
        }
        self.ops += 1;
        Ok(table)
    }
}

impl DigitRangeSession for Session {
    fn round(
        &mut self,
        round: usize,
        prev: Option<&[u8; FIELD_BYTES]>,
        e_first: &[u8],
        e_second: &[u8],
        out: &mut [u8; MESSAGE_BYTES],
    ) -> Result<(), AkitaError> {
        self.run_round(round, prev, e_first, e_second, out)
            .map_err(|e| AkitaError::InvalidInput(format!("webgpu digit range round {round}: {e}")))
    }

    fn table(mut self: Box<Self>) -> Result<Vec<u8>, AkitaError> {
        let t0 = now_ms();
        let table = self
            .download_table()
            .map_err(|e| AkitaError::InvalidInput(format!("webgpu digit range table: {e}")))?;
        let download_ms = now_ms() - t0;
        let total_ms = now_ms() - self.t_open;
        tracing::info!(
            n = self.n,
            rounds = self.rounds,
            upload_ms = self.upload_ms,
            rounds_ms = self.rounds_ms,
            download_ms,
            ops = self.ops,
            "digit range rounds on gpu"
        );
        web_sys::console::log_1(
            &format!(
                "[gpu] digit range n={} rounds={}: upload {:.1} + rounds {:.1} ({} ops) + download {:.1} = {total_ms:.1} ms",
                self.n, self.rounds, self.upload_ms, self.rounds_ms, self.ops, download_ms
            )
            .into(),
        );
        let mut b = BREAKDOWN.lock().unwrap();
        b.instances += 1;
        b.gpu_rounds += self.rounds as u32;
        b.ops += self.ops;
        b.upload_ms += self.upload_ms;
        b.rounds_ms += self.rounds_ms;
        b.download_ms += download_ms;
        b.total_ms += total_ms;
        drop(b);
        Ok(table)
    }
}

/// GPU rounds: leave a `2^CPU_TAIL_BITS`-entry table to the CPU, but never
/// cross into the x-phase (`ring_bits`).
fn gpu_rounds(job: &DigitRangeJob<'_>) -> usize {
    (job.len.ilog2().saturating_sub(CPU_TAIL_BITS) as usize).min(job.ring_bits)
}

fn open_session(job: &DigitRangeJob<'_>, rounds: usize) -> Result<Session, GpuError> {
    let t_open = now_ms();
    let n = u32::try_from(job.len).map_err(|_| GpuError("instance exceeds u32".into()))?;
    // Padded to a whole number of words for writeBuffer.
    let mut digits = job.digits.to_vec();
    digits.resize(digits.len().div_ceil(4) * 4, 0);
    let (handles, fresh) = handles_for(&mut HANDLES.lock().unwrap(), digits.len() as u32, n)?;
    let mut session = Session {
        handles: Some(handles),
        n,
        rounds,
        bit_width: u32::from(job.bit_width),
        r0: [0; FIELD_BYTES],
        r1: [0; FIELD_BYTES],
        last_dst: None,
        table: None,
        ops: 1 + u32::from(fresh),
        upload_ms: 0.0,
        rounds_ms: 0.0,
        t_open,
    };
    let lut0 = lut0_bytes();
    let h = session.handles();
    let mut args = vec![1, h.digits, 0, 0];
    let mut regions = vec![Region::upload(&digits)];
    if fresh {
        args[0] = 2;
        args.extend_from_slice(&[h.lut0, 0, 1]);
        regions.push(Region::upload(&lut0));
    }
    if let Err(e) = mailbox::call(OP_UPLOAD_MULTI, &args, &regions) {
        // Never hand back handles whose lut0 may be missing.
        if let Some(h) = session.handles.take() {
            destroy(&h.all());
        }
        return Err(e);
    }
    session.upload_ms = now_ms() - t_open;
    Ok(session)
}

impl DigitRangeDevice for WebGpuDigitRange {
    fn open(
        &self,
        job: &DigitRangeJob<'_>,
    ) -> Option<Result<(Box<dyn DigitRangeSession>, usize), AkitaError>> {
        let rounds = gpu_rounds(job);
        let qualifies = super::is_enabled()
            && super::digit_range_enabled()
            && job.len >= 1 << MIN_LEN_BITS
            && job.bit_width >= 1
            && rounds >= MIN_ROUNDS;
        qualifies.then(|| {
            if SESSION_OPEN.swap(true, Ordering::SeqCst) {
                return Err(AkitaError::InvalidInput(
                    "webgpu digit range: a session is already open".into(),
                ));
            }
            open_session(job, rounds)
                .map(|s| (Box::new(s) as Box<dyn DigitRangeSession>, rounds))
                .map_err(|e| {
                    // handles_for failed before a Session (and its Drop) existed.
                    SESSION_OPEN.store(false, Ordering::SeqCst);
                    AkitaError::InvalidInput(format!("webgpu digit range: {e}"))
                })
        })
    }
}
