//! Encoders for the unfused commit path: one-hot tier-1 row commitments and
//! the joint RLC matrix combine.

use crate::context::GpuContext;
use crate::shader::{
    fq_header, fr_header, ShaderBuilder, CURVE_WGSL, FIELD_WGSL, G1_GLUE, G1_SUBST, ONEHOT_WGSL,
    RLC_WGSL,
};

/// Sentinel for "no bucket" in one-hot index uploads (`Option::None`).
pub const ONEHOT_NONE: u32 = u32::MAX;

/// Words per poly entry in the RLC meta buffer.
pub const RLC_META_STRIDE: usize = 12;

fn onehot_module_source() -> String {
    ShaderBuilder::new()
        .push(&fq_header())
        .push(&FIELD_WGSL)
        .push(G1_GLUE)
        .push_subst(CURVE_WGSL, G1_SUBST)
        .push_subst(ONEHOT_WGSL, G1_SUBST)
        .build()
}

fn rlc_module_source() -> String {
    ShaderBuilder::new()
        .push(&fr_header())
        .push(&FIELD_WGSL)
        .push(RLC_WGSL)
        .build()
}

/// Stripe count of the one-hot gather (occupancy multiplier).
pub const ONEHOT_STRIPES: u32 = 32;

/// Tier-1 row commitments for a one-hot polynomial: out[out_offset + r] is
/// the sum of `bases[col]` over columns where `indices[c * cols + col] == k`
/// with `r = k * rows_per_k + c` (jolt-core's scatter layout).
///
/// Two dispatches recorded into `pass` (striped partials, then a per-row
/// reduce); WebGPU orders storage writes between dispatches of one pass, so
/// callers batch many polynomials into a single pass with one shared
/// `partials` scratch of `num_rows * ONEHOT_STRIPES` projective points.
#[allow(clippy::too_many_arguments)]
pub fn encode_onehot_rows(
    ctx: &GpuContext,
    pass: &mut wgpu::ComputePass<'_>,
    indices: &wgpu::Buffer,
    bases: &wgpu::Buffer,
    partials: &wgpu::Buffer,
    out: &wgpu::Buffer,
    cols: u32,
    rows_per_k: u32,
    num_rows: u32,
    out_offset: u32,
) {
    let params = ctx.buffer_from(
        "onehot-params",
        bytemuck::cast_slice(&[cols, rows_per_k, num_rows, out_offset]),
        wgpu::BufferUsages::UNIFORM,
    );
    let phase_a = ctx.pipeline("onehot_g1", "onehot_partials", onehot_module_source);
    ctx.dispatch_in_pass(
        pass,
        &phase_a,
        &[(0, &params), (1, indices), (2, bases), (4, partials)],
        (num_rows.div_ceil(64), ONEHOT_STRIPES, 1),
    );
    let phase_b = ctx.pipeline("onehot_g1", "onehot_reduce", onehot_module_source);
    ctx.dispatch_in_pass(
        pass,
        &phase_b,
        &[(0, &params), (3, out), (4, partials)],
        (num_rows.div_ceil(64), 1, 1),
    );
}

/// Joint RLC matrix: out[r * cols + col] = sum_p coeff_p * M_p[r][col] over
/// the dense and one-hot polynomials described by `meta`
/// (see `wgsl/rlc.wgsl` for the meta layout).
#[allow(clippy::too_many_arguments)]
pub fn encode_rlc_combine(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    meta: &wgpu::Buffer,
    dense: &wgpu::Buffer,
    onehot: &wgpu::Buffer,
    out: &wgpu::Buffer,
    num_rows: u32,
    cols: u32,
    rows_per_k: u32,
    n_polys: u32,
) {
    let pipeline = ctx.pipeline("rlc_fr", "rlc_combine", rlc_module_source);
    let params = ctx.buffer_from(
        "rlc-params",
        bytemuck::cast_slice(&[num_rows, cols, rows_per_k, n_polys]),
        wgpu::BufferUsages::UNIFORM,
    );
    ctx.encode_pass_indexed(
        encoder,
        &pipeline,
        &[(0, &params), (1, meta), (2, dense), (3, onehot), (4, out)],
        (cols.div_ceil(64), num_rows, 1),
    );
}
