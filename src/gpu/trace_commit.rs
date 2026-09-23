//! Stage-0 one-hot trace commit on the GPU: `jolt_akita::TraceCommitDevice`
//! backed by the kernels in `frontend/public/wgsl/commit/` (layout contract in
//! `bench/proto/README.md`, ABI in `docs/trace-commit-device.md`).
//!
//! One commit = pack codes → UPLOAD A, codes → RUN prep → RUN accumulate →
//! RUN reduce (RES read back inline). A, A2, codes and PART are persistent
//! handles cached per shape; RES is a per-call temp buffer. The seam validates
//! the result (length, canonical limbs) before converting it to rings.

use std::sync::{Arc, Mutex, Once};

use jolt_akita::{AkitaError, TraceCommitDevice, TraceCommitJob, TraceCommitShape};
use rayon::prelude::*;
use serde::Serialize;
use wasm_bindgen::JsCast;

use super::mailbox::{
    self, GpuError, Region, OP_CREATE_BUFFER, OP_DESTROY, OP_RUN, OP_UPLOAD, REGION_READBACK,
    REGION_UPLOAD,
};
use super::selftest::run_args;

/// Shader ids: index into `SHADERS` in `gpu-proxy.js`.
const SHADER_PREP: u32 = 1;
const SHADER_ACCUMULATE: u32 = 2;
const SHADER_REDUCE: u32 = 3;
const PREP_WORKGROUP: u32 = 256;
const REDUCE_WORKGROUP: u32 = 64;
/// The accumulate kernel handles 2 columns per workgroup (`COLS` in the WGSL).
const ACCUMULATE_COLUMNS_PER_WORKGROUP: u32 = 2;

const K: usize = 16;
const D: usize = 512;
const COLUMN_CAPACITY: usize = 64;
const ROWS_PER_RING: usize = D / K;
/// Byte for an uncommitted (row, column) in the packed code table.
const UNCOMMITTED: u8 = 0xFF;
/// Largest chunk whose 16-bit digit accumulators cannot overflow:
/// 2048 positions × 32 rows = 65536 terms, 65536 · 0xFFFF < 2^32.
const MAX_CHUNK: u32 = 2048;
const MIN_CHUNK: u32 = 64;
const PART_BUDGET_BYTES: u64 = 256 << 20;

/// Per-shape persistent buffers (one shape cached at a time).
struct Buffers {
    positions: u32,
    blocks: u32,
    a: u32,
    a2: u32,
    codes: u32,
    part: u32,
}

impl Buffers {
    fn handles(&self) -> [u32; 4] {
        [self.a, self.a2, self.codes, self.part]
    }
}

#[derive(Default)]
pub struct WebGpuTraceCommit {
    buffers: Mutex<Option<Buffers>>,
}

/// Summed over every device call of one prove; `take()` resets.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct CommitBreakdown {
    pub calls: u32,
    pub pack_ms: f64,
    pub upload_ms: f64,
    /// prep + accumulate.
    pub gpu_ms: f64,
    /// reduce + RES readback.
    pub readback_ms: f64,
    pub total_ms: f64,
}

static BREAKDOWN: Mutex<CommitBreakdown> = Mutex::new(CommitBreakdown {
    calls: 0,
    pack_ms: 0.0,
    upload_ms: 0.0,
    gpu_ms: 0.0,
    readback_ms: 0.0,
    total_ms: 0.0,
});

pub fn take_breakdown() -> CommitBreakdown {
    std::mem::take(&mut *BREAKDOWN.lock().unwrap())
}

static INSTALL: Once = Once::new();

/// Installs the device into jolt-akita's process-global slot exactly once per
/// worker; gpu on/off per prove is honoured inside `commit_accumulate`.
pub fn install_once() {
    INSTALL.call_once(|| {
        let message = match crate::engine::install_trace_commit_device(Arc::new(
            WebGpuTraceCommit::default(),
        )) {
            Ok(()) => "[gpu] trace commit device installed".to_string(),
            Err(e) => format!("[gpu] trace commit device not installed: {e}"),
        };
        web_sys::console::log_1(&message.into());
    });
}

fn now_ms() -> f64 {
    js_sys::Reflect::get(&js_sys::global(), &"performance".into())
        .map(|p| p.unchecked_into::<web_sys::Performance>().now())
        .unwrap_or_else(|_| js_sys::Date::now())
}

/// Smallest chunk (fewest accumulator passes per position, most partials)
/// whose PART scratch fits the budget; `None` when even MAX_CHUNK does not.
fn chunk_for(shape: &TraceCommitShape) -> Option<u32> {
    let positions = shape.positions_per_block as u64;
    let part_bytes_per_chunk =
        COLUMN_CAPACITY as u64 * shape.blocks_per_column as u64 * D as u64 * 32;
    let mut chunk = MIN_CHUNK;
    while chunk <= MAX_CHUNK {
        if positions.is_multiple_of(u64::from(chunk))
            && positions / u64::from(chunk) * part_bytes_per_chunk <= PART_BUDGET_BYTES
        {
            return Some(chunk);
        }
        chunk *= 2;
    }
    None
}

fn qualifies(shape: &TraceCommitShape) -> bool {
    shape.one_hot_k == K
        && shape.ring_dimension == D
        && shape.n_a == 1
        && shape.num_digits_inner == 1
        && shape.column_capacity == COLUMN_CAPACITY
        && shape.num_columns <= COLUMN_CAPACITY
        && shape.num_rows == shape.blocks_per_column * shape.positions_per_block * ROWS_PER_RING
        && shape.num_blocks == COLUMN_CAPACITY * shape.blocks_per_column
        && chunk_for(shape).is_some()
}

/// Kernel code table: one byte per (row, column) with row stride
/// `COLUMN_CAPACITY` = hot if committed else 0xFF; padding columns 0xFF.
fn pack_codes(job: &TraceCommitJob<'_>) -> Vec<u8> {
    let columns = job.shape.num_columns;
    let mut codes = vec![UNCOMMITTED; job.shape.num_rows * COLUMN_CAPACITY];
    codes
        .par_chunks_exact_mut(COLUMN_CAPACITY)
        .zip(job.hot.par_chunks_exact(columns))
        .zip(job.masks.par_iter())
        .for_each(|((row, hot), &mask)| {
            for (column, (code, &hot)) in row.iter_mut().zip(hot).enumerate() {
                if hot != 0 || mask >> column & 1 == 1 {
                    *code = hot;
                }
            }
        });
    codes
}

fn create_buffer(bytes: u64) -> Result<u32, GpuError> {
    let bytes =
        u32::try_from(bytes).map_err(|_| GpuError(format!("buffer of {bytes} B exceeds u32")))?;
    Ok(mailbox::call(OP_CREATE_BUFFER, &[bytes], &[])?[0])
}

fn buffers_for<'a>(
    slot: &'a mut Option<Buffers>,
    shape: &TraceCommitShape,
    chunk: u32,
) -> Result<&'a Buffers, GpuError> {
    let positions = shape.positions_per_block as u32;
    let blocks = shape.blocks_per_column as u32;
    if let Some(b) = slot.take_if(|b| (b.positions, b.blocks) != (positions, blocks)) {
        for handle in b.handles() {
            mailbox::call(OP_DESTROY, &[handle], &[])?;
        }
    }
    if slot.is_none() {
        let a_bytes = u64::from(positions) * (D * 16) as u64;
        let part_bytes = u64::from(positions / chunk)
            * COLUMN_CAPACITY as u64
            * u64::from(blocks)
            * (D * 32) as u64;
        *slot = Some(Buffers {
            positions,
            blocks,
            a: create_buffer(a_bytes)?,
            a2: create_buffer(2 * a_bytes)?,
            codes: create_buffer(shape.num_rows as u64 * COLUMN_CAPACITY as u64)?,
            part: create_buffer(part_bytes)?,
        });
    }
    Ok(slot.as_ref().unwrap())
}

impl WebGpuTraceCommit {
    fn commit(&self, job: &TraceCommitJob<'_>) -> Result<Vec<u32>, GpuError> {
        let _span = tracing::info_span!("trace_onehot_commit_gpu").entered();
        let shape = &job.shape;
        let chunk = chunk_for(shape).expect("qualified shape");
        let t0 = now_ms();
        let codes = pack_codes(job);
        let t1 = now_ms();

        let mut slot = self
            .buffers
            .lock()
            .map_err(|_| GpuError("buffer cache poisoned".into()))?;
        let bufs = buffers_for(&mut slot, shape, chunk)?;
        let positions = bufs.positions;
        let blocks = bufs.blocks;
        let num_chunks = positions / chunk;
        let a_plane = Region {
            ptr: job.a_plane.as_ptr() as u32,
            len: (job.a_plane.len() * 4) as u32,
            flags: REGION_UPLOAD,
        };
        mailbox::call(OP_UPLOAD, &[bufs.a, 0], &[a_plane])?;
        mailbox::call(OP_UPLOAD, &[bufs.codes, 0], &[Region::upload(&codes)])?;
        let t2 = now_ms();

        let params = [positions, num_chunks, blocks, COLUMN_CAPACITY as u32, 0];
        mailbox::call(
            OP_RUN,
            &run_args(
                SHADER_PREP,
                [positions * 1024 / PREP_WORKGROUP, 1, 1],
                &params,
                2,
            ),
            &[Region::handle(bufs.a), Region::handle(bufs.a2)],
        )?;
        let mut accumulate = run_args(
            SHADER_ACCUMULATE,
            [
                num_chunks,
                COLUMN_CAPACITY as u32 / ACCUMULATE_COLUMNS_PER_WORKGROUP,
                blocks,
            ],
            &params,
            3,
        );
        accumulate.push(chunk);
        mailbox::call(
            OP_RUN,
            &accumulate,
            &[
                Region::handle(bufs.a2),
                Region::handle(bufs.codes),
                Region::handle(bufs.part),
            ],
        )?;
        let t3 = now_ms();

        let mut out = vec![0u32; shape.num_blocks * D * 4];
        let res = Region {
            ptr: out.as_mut_ptr() as u32,
            len: (out.len() * 4) as u32,
            flags: REGION_READBACK,
        };
        mailbox::call(
            OP_RUN,
            &run_args(
                SHADER_REDUCE,
                [shape.num_blocks as u32 * D as u32 / REDUCE_WORKGROUP, 1, 1],
                &params,
                2,
            ),
            &[Region::handle(bufs.part), res],
        )?;
        drop(slot);
        let t4 = now_ms();

        let (pack_ms, upload_ms, gpu_ms, readback_ms) = (t1 - t0, t2 - t1, t3 - t2, t4 - t3);
        tracing::info!(
            pack_ms,
            upload_ms,
            gpu_ms,
            readback_ms,
            chunk,
            "trace commit on gpu"
        );
        web_sys::console::log_1(
            &format!(
                "[gpu] trace commit P={positions} blocks={blocks} chunk={chunk}: pack {pack_ms:.1} + upload {upload_ms:.1} + gpu {gpu_ms:.1} + readback {readback_ms:.1} = {:.1} ms",
                t4 - t0
            )
            .into(),
        );
        let mut b = BREAKDOWN.lock().unwrap();
        b.calls += 1;
        b.pack_ms += pack_ms;
        b.upload_ms += upload_ms;
        b.gpu_ms += gpu_ms;
        b.readback_ms += readback_ms;
        b.total_ms += t4 - t0;
        Ok(out)
    }
}

impl TraceCommitDevice for WebGpuTraceCommit {
    fn commit_accumulate(&self, job: &TraceCommitJob<'_>) -> Option<Result<Vec<u32>, AkitaError>> {
        (super::is_enabled() && super::commit_enabled() && qualifies(&job.shape)).then(|| {
            self.commit(job)
                .map_err(|e| AkitaError::InvalidInput(format!("webgpu trace commit: {e}")))
        })
    }
}
