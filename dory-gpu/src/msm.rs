//! MSM orchestration: scalar prep (Montgomery -> canonical + carry masks),
//! projective -> affine normalization, and the bucket MSM over G1 or G2.
//!
//! Both curves use the split pipeline (clear / accumulate into global
//! buckets / weight / sum / combine): every kernel stays small enough for
//! the Apple Metal compiler and uses the unrolled (register-resident) field
//! ops. Multi-row batches are row-chunked to cap bucket scratch memory.
//! All MSM bases are affine — projective inputs are normalized on the GPU
//! first (identity becomes (0,0) and is skipped during accumulation).

use crate::context::GpuContext;
use crate::shader::{
    fq_header, fr_header, g2_3b_header, msm_header, msm_windows, ShaderBuilder, CURVE_WGSL,
    FIELD_WGSL, FQ2_WGSL, G1_GLUE, G1_SUBST, G1_WINDOW, G2_GLUE, G2_SUBST, G2_WINDOW, MSM_CHUNK,
    MSM_COMMON_WGSL, MSM_PREP_WGSL, MSM_SPLIT_WGSL, NORMALIZE_WGSL,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Curve {
    G1,
    G2,
}

impl Curve {
    pub fn window(self) -> u32 {
        match self {
            Curve::G1 => G1_WINDOW,
            Curve::G2 => G2_WINDOW,
        }
    }

    pub fn fe_words(self) -> u32 {
        match self {
            Curve::G1 => 8,
            Curve::G2 => 16,
        }
    }

    pub fn point_words(self) -> u32 {
        3 * self.fe_words()
    }

    pub fn affine_words(self) -> u32 {
        2 * self.fe_words()
    }

    pub fn windows(self) -> u32 {
        msm_windows(self.window())
    }

    pub fn buckets(self) -> u32 {
        1 << (self.window() - 1)
    }
}

fn msm_module_source(curve: Curve) -> String {
    match curve {
        Curve::G1 => ShaderBuilder::new()
            .push(&fq_header())
            .push(&FIELD_WGSL)
            .push(G1_GLUE)
            .push_subst(CURVE_WGSL, G1_SUBST)
            .push(&msm_header(G1_WINDOW))
            .push_subst(MSM_COMMON_WGSL, G1_SUBST)
            .push_subst(MSM_SPLIT_WGSL, G1_SUBST)
            .build(),
        Curve::G2 => ShaderBuilder::new()
            .push(&fq_header())
            .push(&g2_3b_header())
            .push(&FIELD_WGSL)
            .push(FQ2_WGSL)
            .push(G2_GLUE)
            .push_subst(CURVE_WGSL, G2_SUBST)
            .push(&msm_header(G2_WINDOW))
            .push_subst(MSM_COMMON_WGSL, G2_SUBST)
            .push_subst(MSM_SPLIT_WGSL, G2_SUBST)
            .build(),
    }
}

fn normalize_module_source(curve: Curve) -> String {
    match curve {
        Curve::G1 => ShaderBuilder::new()
            .push(&fq_header())
            .push(&FIELD_WGSL)
            .push(G1_GLUE)
            .push_subst(NORMALIZE_WGSL, G1_SUBST)
            .build(),
        Curve::G2 => ShaderBuilder::new()
            .push(&fq_header())
            .push(&g2_3b_header())
            .push(&FIELD_WGSL)
            .push(FQ2_WGSL)
            .push(G2_GLUE)
            .push_subst(NORMALIZE_WGSL, G2_SUBST)
            .build(),
    }
}

fn prep_module_source(c: u32) -> String {
    ShaderBuilder::new()
        .push(&fr_header())
        .push(&FIELD_WGSL)
        .push(&msm_header(c))
        .push(MSM_PREP_WGSL)
        .build()
}

fn msm_module_key(curve: Curve) -> &'static str {
    match curve {
        Curve::G1 => "msm_g1",
        Curve::G2 => "msm_g2",
    }
}

/// Prepared scalars: canonical form plus signed-digit carry masks for one
/// window size. Scalars enter in Montgomery form (bit-identical upload from
/// arkworks or produced by an earlier GPU pass).
pub struct PreparedScalars {
    pub canon: wgpu::Buffer,
    pub masks: wgpu::Buffer,
    pub count: u32,
    pub window: u32,
}

pub fn encode_prep_scalars(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    curve: Curve,
    mont_scalars: &wgpu::Buffer,
    count: u32,
) -> PreparedScalars {
    let c = curve.window();
    let (module_key, source): (&'static str, fn() -> String) = match curve {
        Curve::G1 => ("msm_prep_c8", || prep_module_source(G1_WINDOW)),
        Curve::G2 => ("msm_prep_c7", || prep_module_source(G2_WINDOW)),
    };
    let pipeline = ctx.pipeline(module_key, "msm_prep", source);
    let canon = ctx.empty_buffer("msm-canon", count as u64 * 32, wgpu::BufferUsages::COPY_SRC);
    let masks = ctx.empty_buffer("msm-masks", count as u64 * 8, wgpu::BufferUsages::COPY_SRC);
    let params = ctx.buffer_from(
        "prep-params",
        bytemuck::cast_slice(&[count, 0u32, 0, 0]),
        wgpu::BufferUsages::UNIFORM,
    );
    ctx.encode_pass_indexed(
        encoder,
        &pipeline,
        &[(0, &params), (1, mont_scalars), (2, &canon), (3, &masks)],
        (count.div_ceil(64), 1, 1),
    );
    PreparedScalars {
        canon,
        masks,
        count,
        window: c,
    }
}

/// Normalizes `count` projective points into a fresh affine buffer.
/// The identity maps to (0, 0), which the MSM kernels treat as a skip.
pub fn encode_normalize(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    curve: Curve,
    proj: &wgpu::Buffer,
    count: u32,
) -> wgpu::Buffer {
    let (module_key, source): (&'static str, fn() -> String) = match curve {
        Curve::G1 => ("normalize_g1", || normalize_module_source(Curve::G1)),
        Curve::G2 => ("normalize_g2", || normalize_module_source(Curve::G2)),
    };
    let pipeline = ctx.pipeline(module_key, "normalize_points", source);
    let affine = ctx.empty_buffer(
        "normalized-affine",
        count as u64 * curve.affine_words() as u64 * 4,
        wgpu::BufferUsages::COPY_SRC,
    );
    let params = ctx.buffer_from(
        "norm-params",
        bytemuck::cast_slice(&[count, 0u32, 0, 0]),
        wgpu::BufferUsages::UNIFORM,
    );
    ctx.encode_pass_indexed(
        encoder,
        &pipeline,
        &[(0, &params), (1, proj), (2, &affine)],
        (count.div_ceil(64), 1, 1),
    );
    affine
}

/// One MSM batch: `rows` independent MSMs of `n` points each, sharing one
/// affine bases buffer. Results are written as projective points at
/// `out_offset + row` into `results`.
pub struct MsmCall<'a> {
    pub curve: Curve,
    /// Affine bases ((0,0) = identity, skipped).
    pub bases: &'a wgpu::Buffer,
    pub scalars: &'a PreparedScalars,
    pub rows: u32,
    pub n: u32,
    /// Offset into `bases`, in points.
    pub base_offset: u32,
    /// Offset into the prepared scalars, in scalars.
    pub scalar_offset: u32,
    /// Stride between rows, in scalars (use `n` for dense batches).
    pub scalar_stride: u32,
    pub results: &'a wgpu::Buffer,
    /// Offset into `results`, in points.
    pub out_offset: u32,
}

/// Bucket scratch cap; multi-row batches that would exceed it are encoded in
/// row chunks.
const BUCKET_BYTES_CAP: u64 = 256 << 20;

pub fn encode_msm(ctx: &GpuContext, encoder: &mut wgpu::CommandEncoder, call: &MsmCall) {
    debug_assert_eq!(call.scalars.window, call.curve.window());
    let n_chunks = call.n.div_ceil(MSM_CHUNK).max(1);
    let nw = call.curve.windows();
    let pw = call.curve.point_words();
    let bucket_bytes_per_row = (nw * n_chunks * call.curve.buckets() * pw) as u64 * 4;
    let rows_per_batch =
        (BUCKET_BYTES_CAP / bucket_bytes_per_row).clamp(1, call.rows as u64) as u32;

    let mut row_start = 0u32;
    while row_start < call.rows {
        let batch_rows = rows_per_batch.min(call.rows - row_start);
        encode_msm_batch(ctx, encoder, call, row_start, batch_rows, n_chunks);
        row_start += batch_rows;
    }
}

fn encode_msm_batch(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    call: &MsmCall,
    row_start: u32,
    rows: u32,
    n_chunks: u32,
) {
    let curve = call.curve;
    let nw = curve.windows();
    let pw = curve.point_words();

    let partials = ctx.empty_buffer(
        "msm-partials",
        (rows * nw * n_chunks * pw) as u64 * 4,
        wgpu::BufferUsages::empty(),
    );
    let params = ctx.buffer_from(
        "msm-params",
        bytemuck::cast_slice(&[
            rows,
            call.n,
            n_chunks,
            call.base_offset,
            call.scalar_offset + row_start * call.scalar_stride,
            call.scalar_stride,
            call.out_offset + row_start,
            0u32,
        ]),
        wgpu::BufferUsages::UNIFORM,
    );

    let buckets = ctx.empty_buffer(
        "msm-buckets",
        (rows * nw * n_chunks * curve.buckets() * pw) as u64 * 4,
        wgpu::BufferUsages::empty(),
    );
    let total_buckets = rows * nw * n_chunks * curve.buckets();
    let module_key = msm_module_key(curve);

    let clear = ctx.pipeline(module_key, "msm_clear_buckets", move || {
        msm_module_source(curve)
    });
    ctx.encode_pass_indexed(
        encoder,
        &clear,
        &[(0, &params), (6, &buckets)],
        (total_buckets.div_ceil(64), 1, 1),
    );
    let acc = ctx.pipeline(module_key, "msm_acc_global", move || {
        msm_module_source(curve)
    });
    ctx.encode_pass_indexed(
        encoder,
        &acc,
        &[
            (0, &params),
            (1, call.bases),
            (2, &call.scalars.canon),
            (3, &call.scalars.masks),
            (6, &buckets),
        ],
        (n_chunks, nw, rows),
    );
    let weight = ctx.pipeline(module_key, "msm_weight_global", move || {
        msm_module_source(curve)
    });
    ctx.encode_pass_indexed(
        encoder,
        &weight,
        &[(0, &params), (6, &buckets)],
        (total_buckets.div_ceil(64), 1, 1),
    );
    let sum = ctx.pipeline(module_key, "msm_sum_global", move || {
        msm_module_source(curve)
    });
    ctx.encode_pass_indexed(
        encoder,
        &sum,
        &[(0, &params), (4, &partials), (6, &buckets)],
        (n_chunks, nw, rows),
    );

    let combine = ctx.pipeline(module_key, "msm_combine", move || msm_module_source(curve));
    ctx.encode_pass_indexed(
        encoder,
        &combine,
        &[(0, &params), (4, &partials), (5, call.results)],
        (rows, 1, 1),
    );
}
