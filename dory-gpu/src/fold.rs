//! Dory vector operations: G1/G2 folds, the fixed-base G2 batch scalar mul
//! (v2 construction), the Fr vector-matrix product and Fr folds.

use ark_bn254::G2Projective;

use crate::context::GpuContext;
use crate::msm::{Curve, PreparedScalars};
use crate::repr::g2_affine_to_words;
use crate::shader::{
    fq_header, fr_header, g2_3b_header, msm_header, ShaderBuilder, CURVE_WGSL, FIELD_WGSL,
    FOLD_WGSL, FQ2_WGSL, G1_GLUE, G1_SUBST, G2_GLUE, G2_SUBST, VMV_WGSL,
};

/// The fixed-base path always uses c = 8 digits regardless of curve, so the
/// same scalar prep feeds G1 MSMs and the G2 fixed-base mul.
pub const FIXED_BASE_WINDOW: u32 = 8;

fn fold_module_source(curve: Curve) -> String {
    match curve {
        Curve::G1 => ShaderBuilder::new()
            .push(&fq_header())
            .push(&FIELD_WGSL)
            .push(G1_GLUE)
            .push_subst(CURVE_WGSL, G1_SUBST)
            .push(&msm_header(FIXED_BASE_WINDOW))
            .push_subst(FOLD_WGSL, G1_SUBST)
            .build(),
        Curve::G2 => ShaderBuilder::new()
            .push(&fq_header())
            .push(&g2_3b_header())
            .push(&FIELD_WGSL)
            .push(FQ2_WGSL)
            .push(G2_GLUE)
            .push_subst(CURVE_WGSL, G2_SUBST)
            .push(&msm_header(FIXED_BASE_WINDOW))
            .push_subst(FOLD_WGSL, G2_SUBST)
            .build(),
    }
}

fn fr_ops_module_source() -> String {
    ShaderBuilder::new()
        .push(&fr_header())
        .push(&FIELD_WGSL)
        .push(VMV_WGSL)
        .build()
}

fn fold_module_key(curve: Curve) -> &'static str {
    match curve {
        Curve::G1 => "fold_g1",
        Curve::G2 => "fold_g2",
    }
}

fn fold_params(ctx: &GpuContext, n: u32, v_off: u32, a_off: u32, out_off: u32) -> wgpu::Buffer {
    ctx.buffer_from(
        "fold-params",
        bytemuck::cast_slice(&[n, v_off, a_off, out_off]),
        wgpu::BufferUsages::UNIFORM,
    )
}

/// v[out_offset + i] = k * v[v_offset + i] + v[a_offset + i], projective.
/// `k` is canonical (non-Montgomery) scalar words.
#[allow(clippy::too_many_arguments)]
pub fn encode_fold_scale_add(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    curve: Curve,
    v: &wgpu::Buffer,
    k_canonical: &[u32; 8],
    n: u32,
    v_offset: u32,
    a_offset: u32,
    out_offset: u32,
) {
    let key = fold_module_key(curve);
    let pipeline = ctx.pipeline(key, "fold_scale_add", move || fold_module_source(curve));
    let params = fold_params(ctx, n, v_offset, a_offset, out_offset);
    let scalar = ctx.buffer_from(
        "fold-scalar",
        bytemuck::cast_slice(k_canonical),
        wgpu::BufferUsages::empty(),
    );
    ctx.encode_pass_indexed(
        encoder,
        &pipeline,
        &[(0, &params), (1, &scalar), (2, v)],
        (n.div_ceil(64), 1, 1),
    );
}

/// v[out_offset + i] = v[v_offset + i] + k * bases[base_offset + i], with
/// affine bases (setup generators; never the identity).
#[allow(clippy::too_many_arguments)]
pub fn encode_fold_add_scaled_base(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    curve: Curve,
    v: &wgpu::Buffer,
    bases: &wgpu::Buffer,
    k_canonical: &[u32; 8],
    n: u32,
    v_offset: u32,
    base_offset: u32,
    out_offset: u32,
) {
    let key = fold_module_key(curve);
    let pipeline = ctx.pipeline(key, "fold_add_scaled_base", move || {
        fold_module_source(curve)
    });
    let params = fold_params(ctx, n, v_offset, base_offset, out_offset);
    let scalar = ctx.buffer_from(
        "fold-scalar",
        bytemuck::cast_slice(k_canonical),
        wgpu::BufferUsages::empty(),
    );
    ctx.encode_pass_indexed(
        encoder,
        &pipeline,
        &[(0, &params), (1, &scalar), (2, v), (3, bases)],
        (n.div_ceil(64), 1, 1),
    );
}

/// out[out_offset + i] = scalars[scalar_offset + i] * G for the fixed base
/// encoded in `table` (built by [`build_fixed_base_table_g2`]). Scalars come
/// from a c=8 [`PreparedScalars`].
#[allow(clippy::too_many_arguments)]
pub fn encode_fixed_base_mul(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    curve: Curve,
    out: &wgpu::Buffer,
    table: &wgpu::Buffer,
    scalars: &PreparedScalars,
    n: u32,
    scalar_offset: u32,
    out_offset: u32,
) {
    debug_assert_eq!(scalars.window, FIXED_BASE_WINDOW);
    let key = fold_module_key(curve);
    let pipeline = ctx.pipeline(key, "fixed_base_mul", move || fold_module_source(curve));
    let params = fold_params(ctx, n, 0, scalar_offset, out_offset);
    ctx.encode_pass_indexed(
        encoder,
        &pipeline,
        &[
            (0, &params),
            (2, out),
            (4, table),
            (5, &scalars.canon),
            (6, &scalars.masks),
        ],
        (n.div_ceil(64), 1, 1),
    );
}

/// Window/entry table for the fixed-base kernel: for each c=8 window w and
/// digit d in [1, 128], the affine point d * 2^(8w) * base.
pub fn build_fixed_base_table_g2(base: &G2Projective) -> Vec<u32> {
    let windows = 254u32.div_ceil(FIXED_BASE_WINDOW) as usize;
    let entries = 1usize << (FIXED_BASE_WINDOW - 1);
    let mut all = Vec::with_capacity(windows * entries);
    let mut window_base = *base;
    for _ in 0..windows {
        let mut acc = window_base;
        for _ in 0..entries {
            all.push(acc);
            acc += window_base;
        }
        // acc after the loop is (entries + 1) * window_base; the next window
        // base is 2^c * window_base = 2 * entries * window_base.
        window_base = acc + acc - window_base - window_base;
    }
    let affine = crate::par::normalize_batch(&all);
    let mut words = Vec::with_capacity(affine.len() * 32);
    for p in &affine {
        words.extend_from_slice(&g2_affine_to_words(p));
    }
    words
}

/// out[out_offset + j] = sum_i left[left_offset + i] * matrix[i * cols + j]
/// over Fr (all Montgomery form).
#[allow(clippy::too_many_arguments)]
pub fn encode_vmv(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    matrix: &wgpu::Buffer,
    left: &wgpu::Buffer,
    out: &wgpu::Buffer,
    rows: u32,
    cols: u32,
    left_offset: u32,
    out_offset: u32,
) {
    let pipeline = ctx.pipeline("fr_ops", "vmv_columns", fr_ops_module_source);
    let params = ctx.buffer_from(
        "vmv-params",
        bytemuck::cast_slice(&[rows, cols, left_offset, out_offset]),
        wgpu::BufferUsages::UNIFORM,
    );
    ctx.encode_pass_indexed(
        encoder,
        &pipeline,
        &[(0, &params), (1, matrix), (2, left), (3, out)],
        (cols.div_ceil(64), 1, 1),
    );
}

/// s[out_offset + i] = k * s[s_offset + i] + s[a_offset + i] over Fr;
/// `k_mont` is the Montgomery representation words.
#[allow(clippy::too_many_arguments)]
pub fn encode_fold_scalars(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    s: &wgpu::Buffer,
    k_mont: &[u32; 8],
    n: u32,
    s_offset: u32,
    a_offset: u32,
    out_offset: u32,
) {
    let pipeline = ctx.pipeline("fr_ops", "fold_scalars", fr_ops_module_source);
    let params = ctx.buffer_from(
        "fold-scalars-params",
        bytemuck::cast_slice(&[n, s_offset, a_offset, out_offset]),
        wgpu::BufferUsages::UNIFORM,
    );
    let k = ctx.buffer_from(
        "fold-scalars-k",
        bytemuck::cast_slice(k_mont),
        wgpu::BufferUsages::empty(),
    );
    ctx.encode_pass_indexed(
        encoder,
        &pipeline,
        &[(4, &params), (5, &k), (6, s)],
        (n.div_ceil(64), 1, 1),
    );
}
