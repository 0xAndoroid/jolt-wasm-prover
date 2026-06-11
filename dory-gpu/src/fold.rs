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

// ---------------------------------------------------------------------------
// GLV folds (jolt-core parity): the shared challenge scalar is decomposed on
// the CPU (jolt_optimizations decomp_2d/decomp_4d — the exact routines the
// CPU folds use) and the kernels run 2-point (G1) / 4-point (G2) Shamir
// ladders over the endomorphism orbits.
// ---------------------------------------------------------------------------

/// CPU-decomposed GLV scalar: up to four |k_i| magnitudes (LSB-first u32
/// limbs), a negate mask (bit i set = subtract orbit point i), and the
/// ladder length.
pub struct GlvDecomp {
    pub k: [[u32; 4]; 4],
    pub negate: u32,
    pub max_bits: u32,
}

/// Decomposes a challenge scalar for the given curve's fold kernels. The
/// fork's sign conventions (2D: true = positive; 4D: true = negative) are
/// normalized to the kernel's negate mask here.
pub fn glv_decompose(curve: Curve, scalar: &ark_bn254::Fr) -> GlvDecomp {
    use ark_ff::BigInteger;
    let (ks, negs): (Vec<_>, Vec<bool>) = match curve {
        Curve::G1 => {
            let (ks, signs) = jolt_optimizations::decomp_2d::decompose_scalar_2d(*scalar);
            (ks.to_vec(), vec![!signs[0], !signs[1]])
        }
        Curve::G2 => {
            let (ks, signs) = jolt_optimizations::decomp_4d::decompose_scalar_4d(*scalar);
            (ks.to_vec(), signs.to_vec())
        }
    };
    let mut k = [[0u32; 4]; 4];
    let mut negate = 0u32;
    let mut max_bits = 0u32;
    for (i, big) in ks.iter().enumerate() {
        assert_eq!(big.0[2], 0, "GLV sub-scalar exceeds 128 bits");
        assert_eq!(big.0[3], 0, "GLV sub-scalar exceeds 128 bits");
        k[i] = [
            big.0[0] as u32,
            (big.0[0] >> 32) as u32,
            big.0[1] as u32,
            (big.0[1] >> 32) as u32,
        ];
        max_bits = max_bits.max(big.num_bits());
        if negs[i] {
            negate |= 1 << i;
        }
    }
    GlvDecomp {
        k,
        negate,
        max_bits,
    }
}

fn glv_module_source(curve: Curve) -> String {
    use crate::shader::{fe_const_decl, GLV_FOLD_G1_WGSL, GLV_FOLD_G2_WGSL};
    match curve {
        Curve::G1 => {
            use ark_ec::PrimeGroup;
            use ark_ff::Field;
            let gen = ark_bn254::G1Projective::generator();
            let beta = jolt_optimizations::decomp_2d::glv_endomorphism(&gen).x
                * gen.x.inverse().expect("generator x nonzero");
            ShaderBuilder::new()
                .push(&fq_header())
                .push(&fe_const_decl("GLV_BETA", &beta.0))
                .push(&FIELD_WGSL)
                .push(G1_GLUE)
                .push_subst(CURVE_WGSL, G1_SUBST)
                .push(GLV_FOLD_G1_WGSL)
                .build()
        }
        Curve::G2 => {
            let c = jolt_optimizations::constants::get_frobenius_coefficients();
            let mut consts = String::new();
            for (name, fe2) in [
                ("PSI1X", c.psi1_coef2),
                ("PSI1Y", c.psi1_coef3),
                ("PSI2X", c.psi2_coef2),
                ("PSI2Y", c.psi2_coef3),
                ("PSI3X", c.psi3_coef2),
                ("PSI3Y", c.psi3_coef3),
            ] {
                consts.push_str(&fe_const_decl(&format!("{name}_C0"), &fe2.c0.0));
                consts.push_str(&fe_const_decl(&format!("{name}_C1"), &fe2.c1.0));
            }
            ShaderBuilder::new()
                .push(&fq_header())
                .push(&g2_3b_header())
                .push(&consts)
                .push(&FIELD_WGSL)
                .push(FQ2_WGSL)
                .push(G2_GLUE)
                .push_subst(CURVE_WGSL, G2_SUBST)
                .push(GLV_FOLD_G2_WGSL)
                .build()
        }
    }
}

fn glv_module_key(curve: Curve) -> &'static str {
    match curve {
        Curve::G1 => "glv_fold_g1",
        Curve::G2 => "glv_fold_g2",
    }
}

fn glv_params_buffer(
    ctx: &GpuContext,
    d: &GlvDecomp,
    n: u32,
    v_offset: u32,
    a_offset: u32,
    out_offset: u32,
) -> wgpu::Buffer {
    let mut words = vec![
        n, v_offset, a_offset, out_offset, d.max_bits, d.negate, 0, 0,
    ];
    for k in &d.k {
        words.extend_from_slice(k);
    }
    ctx.buffer_from(
        "glv-params",
        bytemuck::cast_slice(&words),
        wgpu::BufferUsages::UNIFORM,
    )
}

/// GLV variant of [`encode_fold_scale_add`]:
/// v[out_offset + i] = k * v[v_offset + i] + v[a_offset + i].
#[allow(clippy::too_many_arguments)]
pub fn encode_glv_fold_scale_add(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    curve: Curve,
    v: &wgpu::Buffer,
    decomp: &GlvDecomp,
    n: u32,
    v_offset: u32,
    a_offset: u32,
    out_offset: u32,
) {
    let key = glv_module_key(curve);
    let pipeline = ctx.pipeline(key, "glv_scale_add", move || glv_module_source(curve));
    let params = glv_params_buffer(ctx, decomp, n, v_offset, a_offset, out_offset);
    ctx.encode_pass_indexed(
        encoder,
        &pipeline,
        &[(0, &params), (1, v)],
        (n.div_ceil(64), 1, 1),
    );
}

/// GLV variant of [`encode_fold_add_scaled_base`]:
/// v[out_offset + i] = v[v_offset + i] + k * bases[base_offset + i].
#[allow(clippy::too_many_arguments)]
pub fn encode_glv_fold_add_scaled_base(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    curve: Curve,
    v: &wgpu::Buffer,
    bases: &wgpu::Buffer,
    decomp: &GlvDecomp,
    n: u32,
    v_offset: u32,
    base_offset: u32,
    out_offset: u32,
) {
    let key = glv_module_key(curve);
    let pipeline = ctx.pipeline(key, "glv_add_scaled_base", move || glv_module_source(curve));
    let params = glv_params_buffer(ctx, decomp, n, v_offset, base_offset, out_offset);
    ctx.encode_pass_indexed(
        encoder,
        &pipeline,
        &[(0, &params), (1, v), (2, bases)],
        (n.div_ceil(64), 1, 1),
    );
}
