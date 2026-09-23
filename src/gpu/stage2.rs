//! Stage-2 relation-range sumcheck rounds on the GPU:
//! `akita_prover::RelationRangeDevice` backed by `frontend/public/wgsl/stage2/`.
//!
//! One instance = one ALLOC (when the cached handles are too small) + one
//! UPLOAD (digits plus the dense weights or the lane weights) → per round one
//! RUN_SEQ: the fused fold + evaluate pass over all pairs, the sparse
//! additional-terms pass, the reduce; params, the Gruen tables, the folded
//! alpha / linear sources, the segments and the additional pairs travel
//! inline, the 160 B message comes back inline and the last round also
//! copies the folded `[P | W]` tables for the CPU tail into that readback.

use serde::Serialize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Once};

use akita_prover::{
    RelationRangeDevice, RelationRangeJob, RelationRangeRoundInput as RoundInput,
    RelationRangeSession, RelationRangeShape, RelationRangeWeightsKind as WeightsKind,
    RELATION_RANGE_FIELD_BYTES as FIELD_BYTES, RELATION_RANGE_MESSAGE_BYTES as MESSAGE_BYTES,
    RELATION_RANGE_PAIR_BYTES as PAIR_BYTES, RELATION_RANGE_SEGMENT_BYTES as SEGMENT_BYTES,
};
use jolt_akita::AkitaError;

use super::mailbox::{self, GpuError, Region, OP_ALLOC, OP_RUN_SEQ, OP_UPLOAD_MULTI, RUN_SEQ_COPY};
use super::now_ms;

const SHADER_ROUND: u32 = 9;
const SHADER_ADDITIONAL: u32 = 10;
const SHADER_REDUCE: u32 = 11;
const WG: u32 = 256;
const NT: u32 = 6;
const UNITS_PER_THREAD_TARGET: u32 = WG * 1024;
const PARAMS_BYTES: usize = 64;
/// Smallest relation domain worth the trips; the CPU tail keeps a `2^CPU_TAIL_BITS` domain.
const MIN_DOMAIN_BITS: u32 = 16;
const CPU_TAIL_BITS: u32 = 12;
/// vec4 slots of the aux header: `[cw, alpha_off, seg_off, nseg, src_p_off, dst_p_off, pairs_off, n_pairs,
/// lane_map_off, n_sources, live_lanes, 0]` then up to 16 source value offsets (elements of aux).
const AUX_HDR: u32 = 7;
const MAX_SOURCES: usize = 16;

const R_AUX: u32 = 1;
const R_OUT: u32 = 2;
const R_DIGITS: u32 = 3;
const R_PARTIALS: u32 = 4;
const R_BASE: u32 = 5;
const R_TA: u32 = 6;
const R_TB: u32 = 7;

struct Handles {
    digits_cap: u32,
    base_cap: u32,
    ta_cap: u32,
    tb_cap: u32,
    part_cap: u32,
    digits: u32,
    base: u32,
    ta: u32,
    tb: u32,
    partials: u32,
}

impl Handles {
    fn all(&self) -> [u32; 5] {
        [self.digits, self.base, self.ta, self.tb, self.partials]
    }
}

pub struct WebGpuRelationRange;

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
    pub total_ms: f64,
    pub upload_bytes: u64,
    pub inline_bytes: u64,
}

static BREAKDOWN: Mutex<Breakdown> = Mutex::new(Breakdown {
    instances: 0,
    gpu_rounds: 0,
    ops: 0,
    upload_ms: 0.0,
    rounds_ms: 0.0,
    total_ms: 0.0,
    upload_bytes: 0,
    inline_bytes: 0,
});

pub fn take_breakdown() -> Breakdown {
    std::mem::take(&mut *BREAKDOWN.lock().unwrap())
}

static INSTALL: Once = Once::new();

/// Installs the device into akita-prover's process-global slot exactly once
/// per worker; gpu on/off per prove is honoured inside `rounds_for`.
pub fn install_once() {
    INSTALL.call_once(|| {
        let message =
            match crate::engine::install_relation_range_device(Arc::new(WebGpuRelationRange)) {
                Ok(()) => "[gpu] stage-2 relation-range device installed".to_string(),
                Err(e) => format!("[gpu] stage-2 relation-range device not installed: {e}"),
            };
        web_sys::console::log_1(&message.into());
    });
}

/// Largest power of two ≤ 8 keeping ≥ `UNITS_PER_THREAD_TARGET` threads busy.
fn units_per_thread(units: u32) -> u32 {
    let mut ppt = 8;
    while ppt > 1 && units < UNITS_PER_THREAD_TARGET * ppt {
        ppt /= 2;
    }
    ppt
}

/// Table sizes of one session, in field elements.
struct Layout {
    domain: u32,
    len: u32,
    kind: WeightsKind,
    coefficient_bits: u32,
    rounds: u32,
}

impl Layout {
    /// Does round `k`'s destination table carry a flat weight region?
    fn has_p(&self, k: u32) -> bool {
        match self.kind {
            WeightsKind::Dense => k >= 1,
            WeightsKind::Factored => k >= self.coefficient_bits,
        }
    }

    fn p_len(&self, k: u32) -> u32 {
        if self.has_p(k) {
            self.domain >> k
        } else {
            0
        }
    }

    fn w_len(&self, k: u32) -> u32 {
        self.len.div_ceil(1 << k)
    }

    /// Entries of round `k`'s destination table (`k >= 1`).
    fn table_len(&self, k: u32) -> u32 {
        self.p_len(k) + self.w_len(k)
    }

    fn dst_is_ta(k: u32) -> bool {
        k % 2 == 1
    }

    fn ta_cap(&self) -> u32 {
        (1..self.rounds)
            .filter(|&k| Self::dst_is_ta(k))
            .map(|k| self.table_len(k))
            .max()
            .unwrap_or(1)
    }

    fn tb_cap(&self) -> u32 {
        (1..self.rounds)
            .filter(|&k| !Self::dst_is_ta(k))
            .map(|k| self.table_len(k))
            .max()
            .unwrap_or(1)
    }

    /// `Factored`: element offset of the lane map words after the lane weights in `base`.
    fn lane_map_off(&self) -> u32 {
        match self.kind {
            WeightsKind::Factored => self.domain >> self.coefficient_bits,
            WeightsKind::Dense => 0,
        }
    }

    fn max_wgs(&self) -> u32 {
        let units = self.domain / 2;
        units.div_ceil(WG * units_per_thread(units)) + 1
    }
}

fn handles_for(
    slot: &mut Option<Handles>,
    digits_bytes: u32,
    base_bytes: u32,
    layout: &Layout,
    max_pairs: u32,
) -> Result<(Handles, bool), GpuError> {
    let ta = layout.ta_cap() * FIELD_BYTES as u32;
    let tb = layout.tb_cap() * FIELD_BYTES as u32;
    let part = (layout.max_wgs() + max_pairs.div_ceil(WG) + 1) * NT * FIELD_BYTES as u32;
    let stale = slot.take_if(|h| {
        h.digits_cap < digits_bytes
            || h.base_cap < base_bytes
            || h.ta_cap < ta
            || h.tb_cap < tb
            || h.part_cap < part
    });
    if let Some(h) = slot.take() {
        return Ok((h, false));
    }
    let sizes = [digits_bytes + 4, base_bytes.max(16), ta, tb, part];
    let old = stale.map(|h| h.all().to_vec()).unwrap_or_default();
    let mut args = vec![old.len() as u32];
    args.extend_from_slice(&old);
    args.push(sizes.len() as u32);
    args.extend_from_slice(&sizes);
    let created = mailbox::call(OP_ALLOC, &args, &[])?;
    let [digits, base, ta_h, tb_h, partials] = created[..sizes.len()].try_into().unwrap();
    Ok((
        Handles {
            digits_cap: digits_bytes,
            base_cap: base_bytes,
            ta_cap: ta,
            tb_cap: tb,
            part_cap: part,
            digits,
            base,
            ta: ta_h,
            tb: tb_h,
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
    layout: Layout,
    bit_width: u32,
    segments: Vec<u8>,
    n_segments: u32,
    source_lanes: Vec<u32>,
    /// Folded `[P | W]` tables read back with the last round's message.
    tables: Option<Vec<u8>>,
    ops: u32,
    upload_ms: f64,
    rounds_ms: f64,
    inline_bytes: u64,
    t_open: f64,
}

impl Drop for Session {
    fn drop(&mut self) {
        *HANDLES.lock().unwrap() = self.handles.take();
        SESSION_OPEN.store(false, Ordering::SeqCst);
    }
}

struct Pass {
    shader: u32,
    wgs: u32,
    binds: Vec<(u32, u32)>,
    params: [u32; 16],
}

fn push_u32s(out: &mut Vec<u8>, words: &[u32]) {
    for w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
}

impl Session {
    fn handles(&self) -> &Handles {
        self.handles
            .as_ref()
            .expect("session owns the handles until tables()")
    }

    fn run_round(
        &mut self,
        input: &RoundInput<'_>,
        out: &mut [u8; MESSAGE_BYTES],
    ) -> Result<(), GpuError> {
        let t0 = now_ms();
        let k = u32::try_from(input.round).map_err(|_| GpuError("round exceeds u32".into()))?;
        let lay = &self.layout;
        if k >= lay.rounds {
            return Err(GpuError(format!(
                "round {k} past the {} device rounds",
                lay.rounds
            )));
        }
        let last = k + 1 == lay.rounds;
        let n_units = (lay.domain >> k) / 2;
        let inner = (input.e_first.len() / FIELD_BYTES) as u32;
        let outer = (input.e_second.len() / FIELD_BYTES) as u32;
        if inner == 0 || inner * outer != n_units {
            return Err(GpuError(format!(
                "round {k}: eq tables {inner} x {outer} do not cover {n_units} pairs"
            )));
        }
        let inner_bits = inner.ilog2();
        let n_pairs = (input.pairs.len() / PAIR_BYTES) as u32;
        let alpha_len = (input.alpha.len() / FIELD_BYTES) as u32;
        let src_len = (input.sources.len() / FIELD_BYTES) as u32;

        // aux blob: header | e_first | e_second | alpha | sources | segments | pairs
        let off_first = AUX_HDR;
        let off_second = off_first + inner;
        let alpha_off = off_second + outer;
        let src_off = alpha_off + alpha_len;
        let seg_off = src_off + src_len;
        let pairs_off = seg_off + 2 * self.n_segments;
        let factored_phase = lay.kind == WeightsKind::Factored && k <= lay.coefficient_bits;
        let cw = lay.coefficient_bits.saturating_sub(k);
        let mut aux = Vec::with_capacity((pairs_off as usize + 5 * n_pairs as usize) * FIELD_BYTES);
        // Source value offsets (elements of aux) for this round's coefficient width.
        let mut source_offsets = [0u32; MAX_SOURCES];
        let mut off = src_off;
        for (slot, lanes) in source_offsets.iter_mut().zip(&self.source_lanes) {
            *slot = off;
            off += lanes << cw;
        }
        if factored_phase && off != seg_off {
            return Err(GpuError(format!(
                "round {k}: linear sources hold {src_len} values, expected {}",
                off - src_off
            )));
        }
        let lane_map_off = self.layout.lane_map_off();
        push_u32s(
            &mut aux,
            &[
                cw,
                alpha_off,
                seg_off,
                self.n_segments,
                0,
                0,
                pairs_off,
                n_pairs,
                lane_map_off,
                self.source_lanes.len() as u32,
                lay.len >> lay.coefficient_bits,
                0,
            ],
        );
        push_u32s(&mut aux, &source_offsets);
        aux.extend_from_slice(input.e_first);
        aux.extend_from_slice(input.e_second);
        aux.extend_from_slice(input.alpha);
        aux.extend_from_slice(input.sources);
        aux.extend_from_slice(&self.segments);
        aux.extend_from_slice(input.pairs);

        let (src_region, dst_region, dst_handle) = match k {
            0 => (R_BASE, R_TA, None),
            1 => (R_BASE, R_TA, Some(self.handles().ta)),
            k if Layout::dst_is_ta(k) => (R_TB, R_TA, Some(self.handles().ta)),
            _ => (R_TA, R_TB, Some(self.handles().tb)),
        };
        let src_mode = k.min(2);
        let live_len = if k <= 1 { lay.len } else { lay.w_len(k - 1) };
        let src_w_off = if k >= 2 { lay.p_len(k - 1) } else { 0 };
        let dst_w_off = lay.p_len(k);
        let wflags = match lay.kind {
            WeightsKind::Dense => {
                if k == 0 {
                    0
                } else {
                    4 | 2
                }
            }
            WeightsKind::Factored => {
                if k < lay.coefficient_bits {
                    1
                } else if k == lay.coefficient_bits {
                    1 | 2
                } else {
                    4 | 2
                }
            }
        };
        let ppt = units_per_thread(n_units);
        let case_c = u32::from(inner < ppt);
        let wgs = n_units.div_ceil(WG * ppt);
        let zero = [0u8; FIELD_BYTES];
        let r = input.prev.map_or(&zero, |p| p);
        let mut params = [
            n_units,
            inner_bits,
            off_first,
            off_second,
            ppt,
            self.bit_width,
            src_mode,
            case_c,
            0,
            0,
            0,
            0,
            live_len,
            src_w_off,
            dst_w_off,
            wflags,
        ];
        for (i, limb) in r.chunks_exact(4).enumerate() {
            params[8 + i] = u32::from_le_bytes(limb.try_into().unwrap());
        }
        // Factored rounds bind the lane weights at 6 and need any live buffer as src.
        let round_binds = vec![
            (1, R_DIGITS),
            (2, R_AUX),
            (3, R_PARTIALS),
            (4, src_region),
            (5, dst_region),
            (6, R_BASE),
        ];
        let mut passes = vec![Pass {
            shader: SHADER_ROUND,
            wgs,
            binds: round_binds,
            params,
        }];
        let add_off = NT * wgs;
        let add_wgs = n_pairs.div_ceil(WG);
        let add_live = if k == 0 { lay.len } else { lay.w_len(k) };
        if n_pairs > 0 {
            passes.push(Pass {
                shader: SHADER_ADDITIONAL,
                wgs: add_wgs,
                binds: vec![(1, R_DIGITS), (2, R_AUX), (3, R_PARTIALS), (5, dst_region)],
                params: [
                    n_pairs,
                    0,
                    add_off,
                    0,
                    1,
                    self.bit_width,
                    src_mode,
                    0,
                    0,
                    0,
                    0,
                    0,
                    add_live,
                    0,
                    dst_w_off,
                    0,
                ],
            });
        }
        let part_needed = (wgs + add_wgs) * NT * FIELD_BYTES as u32;
        if part_needed > self.handles().part_cap {
            return Err(GpuError(format!(
                "round {k} needs {part_needed} B of partials, allocated {}",
                self.handles().part_cap
            )));
        }
        passes.push(Pass {
            shader: SHADER_REDUCE,
            wgs: 1,
            binds: vec![(3, R_PARTIALS), (4, R_OUT)],
            params: [wgs, add_wgs, add_off, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        });

        let mut param_bytes = Vec::with_capacity(passes.len() * PARAMS_BYTES);
        let mut args = vec![passes.len() as u32];
        for pass in &passes {
            push_u32s(&mut param_bytes, &pass.params);
            args.extend_from_slice(&[pass.shader, pass.wgs, pass.binds.len() as u32]);
            for &(binding, region) in &pass.binds {
                args.extend_from_slice(&[binding, region]);
            }
        }
        let table_len = if last && dst_handle.is_some() {
            lay.table_len(k) as usize * FIELD_BYTES
        } else {
            0
        };
        if table_len > 0 {
            if args.len() > RUN_SEQ_COPY {
                return Err(GpuError(format!(
                    "RUN_SEQ args overflow ({} words)",
                    args.len()
                )));
            }
            args.resize(RUN_SEQ_COPY, 0);
            args.extend_from_slice(&[dst_region, R_OUT, MESSAGE_BYTES as u32, table_len as u32]);
        }
        let mut buf = vec![0u8; MESSAGE_BYTES + table_len];
        let h = self.handles();
        let regions = [
            Region::upload(&param_bytes),
            Region::upload(&aux),
            Region::readback(&mut buf),
            Region::handle(h.digits),
            Region::handle(h.partials),
            Region::handle(h.base),
            Region::handle(h.ta),
            Region::handle(h.tb),
        ];
        let inline = (param_bytes.len() + aux.len()) as u64;
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
            self.tables = Some(buf);
        }
        self.ops += 1;
        self.inline_bytes += inline;
        self.rounds_ms += now_ms() - t0;
        Ok(())
    }
}

impl RelationRangeSession for Session {
    fn round(
        &mut self,
        input: &RoundInput<'_>,
        out: &mut [u8; MESSAGE_BYTES],
    ) -> Result<(), AkitaError> {
        self.run_round(input, out).map_err(|e| {
            AkitaError::InvalidInput(format!("webgpu stage-2 round {}: {e}", input.round))
        })
    }

    fn tables(mut self: Box<Self>) -> Result<(Vec<u8>, Vec<u8>), AkitaError> {
        let last = self.layout.rounds - 1;
        let mut blob = self.tables.take().ok_or_else(|| {
            AkitaError::InvalidInput(
                "webgpu stage-2: tables requested before the last round".into(),
            )
        })?;
        let p_bytes = self.layout.p_len(last) as usize * FIELD_BYTES;
        let w_bytes = self.layout.w_len(last) as usize * FIELD_BYTES;
        if blob.len() != p_bytes + w_bytes {
            return Err(AkitaError::InvalidSize {
                expected: p_bytes + w_bytes,
                actual: blob.len(),
            });
        }
        let witness = blob.split_off(p_bytes);
        let total_ms = now_ms() - self.t_open;
        tracing::info!(
            domain = self.layout.domain,
            len = self.layout.len,
            rounds = self.layout.rounds,
            upload_ms = self.upload_ms,
            rounds_ms = self.rounds_ms,
            ops = self.ops,
            inline_bytes = self.inline_bytes,
            "stage-2 rounds on gpu"
        );
        web_sys::console::log_1(
            &format!(
                "[gpu] stage 2 domain=2^{} len={} {:?} rounds={}: upload {:.1} + rounds {:.1} ({} ops, {:.1} MiB inline) = {total_ms:.1} ms",
                self.layout.domain.ilog2(),
                self.layout.len,
                self.layout.kind,
                self.layout.rounds,
                self.upload_ms,
                self.rounds_ms,
                self.ops,
                self.inline_bytes as f64 / (1024.0 * 1024.0)
            )
            .into(),
        );
        let mut b = BREAKDOWN.lock().unwrap();
        b.instances += 1;
        b.gpu_rounds += self.layout.rounds;
        b.ops += self.ops;
        b.upload_ms += self.upload_ms;
        b.rounds_ms += self.rounds_ms;
        b.total_ms += total_ms;
        b.inline_bytes += self.inline_bytes;
        drop(b);
        Ok((witness, blob))
    }
}

/// Device rounds: leave a `2^CPU_TAIL_BITS` domain to the CPU.
fn gpu_rounds(shape: &RelationRangeShape) -> usize {
    shape.domain_len.ilog2().saturating_sub(CPU_TAIL_BITS) as usize
}

fn open_session(job: &RelationRangeJob<'_>, rounds: usize) -> Result<Session, GpuError> {
    let t_open = now_ms();
    let shape = &job.shape;
    let domain =
        u32::try_from(shape.domain_len).map_err(|_| GpuError("domain exceeds u32".into()))?;
    let len = u32::try_from(shape.len).map_err(|_| GpuError("witness exceeds u32".into()))?;
    let layout = Layout {
        domain,
        len,
        kind: shape.weights,
        coefficient_bits: shape.coefficient_bits as u32,
        rounds: rounds as u32,
    };
    let mut digits = job.digits.to_vec();
    digits.resize(digits.len().div_ceil(4) * 4, 0);
    if job.source_lanes.len() > MAX_SOURCES {
        return Err(GpuError(format!(
            "{} linear sources exceed {MAX_SOURCES}",
            job.source_lanes.len()
        )));
    }
    // `base` = the dense weights, or the lane weights followed by the lane map words.
    let mut base_vec = Vec::new();
    let base: &[u8] = match shape.weights {
        WeightsKind::Dense => job.weights,
        WeightsKind::Factored => {
            base_vec.reserve(job.lane_weights.len() + job.lane_map.len() * 4 + 16);
            base_vec.extend_from_slice(job.lane_weights);
            for w in job.lane_map {
                base_vec.extend_from_slice(&w.to_le_bytes());
            }
            base_vec.resize(base_vec.len().div_ceil(16) * 16, 0);
            &base_vec
        }
    };
    // Additional pairs are at most one per witness pair (upper bound for the partials scratch).
    let max_pairs = (len / 2).min(1 << 22);
    let (handles, fresh) = handles_for(
        &mut HANDLES.lock().unwrap(),
        digits.len() as u32,
        base.len() as u32,
        &layout,
        max_pairs,
    )?;
    let mut session = Session {
        handles: Some(handles),
        layout,
        bit_width: u32::from(shape.bit_width),
        segments: job.segments.to_vec(),
        n_segments: (job.segments.len() / SEGMENT_BYTES) as u32,
        source_lanes: job.source_lanes.to_vec(),
        tables: None,
        ops: 1 + u32::from(fresh),
        upload_ms: 0.0,
        rounds_ms: 0.0,
        inline_bytes: 0,
        t_open,
    };
    let h = session.handles();
    let args = [2, h.digits, 0, 0, h.base, 0, 1];
    let regions = [Region::upload(&digits), Region::upload(base)];
    if let Err(e) = mailbox::call(OP_UPLOAD_MULTI, &args, &regions) {
        if let Some(h) = session.handles.take() {
            destroy(&h.all());
        }
        return Err(e);
    }
    session.upload_ms = now_ms() - t_open;
    BREAKDOWN.lock().unwrap().upload_bytes += (digits.len() + base.len()) as u64;
    Ok(session)
}

impl RelationRangeDevice for WebGpuRelationRange {
    fn rounds_for(&self, shape: &RelationRangeShape) -> usize {
        let qualifies = super::is_enabled()
            && super::stage2_enabled()
            && shape.domain_len.is_power_of_two()
            && shape.domain_len >= 1 << MIN_DOMAIN_BITS
            && shape.bit_width >= 1
            && shape.bit_width <= 8
            && !SESSION_OPEN.load(Ordering::SeqCst);
        if qualifies {
            gpu_rounds(shape)
        } else {
            0
        }
    }

    fn open(
        &self,
        job: &RelationRangeJob<'_>,
        rounds: usize,
    ) -> Result<Box<dyn RelationRangeSession>, AkitaError> {
        if SESSION_OPEN.swap(true, Ordering::SeqCst) {
            return Err(AkitaError::InvalidInput(
                "webgpu stage-2: a session is already open".into(),
            ));
        }
        open_session(job, rounds)
            .map(|s| Box::new(s) as Box<dyn RelationRangeSession>)
            .map_err(|e| {
                // handles_for failed before a Session (and its Drop) existed.
                SESSION_OPEN.store(false, Ordering::SeqCst);
                AkitaError::InvalidInput(format!("webgpu stage-2: {e}"))
            })
    }
}
