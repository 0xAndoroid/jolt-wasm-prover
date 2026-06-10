//! GPU multi-Miller loops. The ate digit schedule is a compile-time constant,
//! so an entire Miller loop encodes as a fixed sequence of small dispatches
//! in one command buffer; the final exponentiation runs on the CPU over the
//! tiny readback.
//!
//! Two variants:
//! - computed: variable G2 points (line coefficients computed on the fly);
//! - prepared: fixed setup G2 points with arkworks `G2Prepared` line
//!   coefficients uploaded once.
//!
//! P-side inputs are affine G1 ((0,0) = identity masks the pair to f = 1, as
//! does an identity Q), so the per-product f equals arkworks
//! `multi_miller_loop` exactly.

use ark_bn254::{Fq12, G2Affine};
use ark_ec::bn::{BnConfig, G2Prepared};
use ark_ec::pairing::{MillerLoopOutput, Pairing};

use crate::context::GpuContext;
use crate::repr::{fq12_from_words, fq2_to_words, FQ12_WORDS};
use crate::shader::{
    fq_header, g2_3b_header, pairing_header, ShaderBuilder, FIELD_WGSL, FQ12_WGSL, FQ2_WGSL,
    PAIRING_FQ12_WGSL, PAIRING_WGSL,
};

fn ate_loop_count() -> &'static [i8] {
    <ark_bn254::Config as BnConfig>::ATE_LOOP_COUNT
}

/// Number of line coefficients per prepared G2 point (the dispatch schedule
/// and arkworks' `G2Prepared::ell_coeffs` length agree by construction).
pub fn prepared_line_count() -> u32 {
    let ate = ate_loop_count();
    let mut n = 0u32;
    for i in (1..ate.len()).rev() {
        n += 1;
        if ate[i - 1] != 0 {
            n += 1;
        }
    }
    n + 2
}

fn g2step_module_source() -> String {
    ShaderBuilder::new()
        .push(&fq_header())
        .push(&g2_3b_header())
        .push(&pairing_header())
        .push(&FIELD_WGSL)
        .push(FQ2_WGSL)
        .push(PAIRING_WGSL)
        .build()
}

fn fq12_module_source() -> String {
    ShaderBuilder::new()
        .push(&fq_header())
        .push(&FIELD_WGSL)
        .push(FQ2_WGSL)
        .push(FQ12_WGSL)
        .push(PAIRING_FQ12_WGSL)
        .build()
}

#[derive(Clone, Copy)]
struct PairParams {
    n_pairs: u32,
    line_source: u32,
    step: u32,
    prep_stride: u32,
    q_source: u32,
    segment: u32,
    stride: u32,
    check_q: u32,
    prep_mod: u32,
}

impl PairParams {
    fn buffer(&self, ctx: &GpuContext) -> wgpu::Buffer {
        ctx.buffer_from(
            "pair-params",
            bytemuck::cast_slice(&[
                self.n_pairs,
                self.line_source,
                self.step,
                self.prep_stride,
                self.q_source,
                self.segment,
                self.stride,
                self.check_q,
                self.prep_mod,
                0,
                0,
                0,
            ]),
            wgpu::BufferUsages::UNIFORM,
        )
    }
}

/// Buffers holding the f-state of one Miller batch (`n_pairs` pairs laid out
/// product-major in contiguous equal segments).
pub struct MillerState {
    pub f: wgpu::Buffer,
    pub n_pairs: u32,
    mask: wgpu::Buffer,
    scratch: wgpu::Buffer,
}

/// Records every Miller dispatch into ONE compute pass: pass boundaries are
/// Metal encoder switches (~tens of ms each), in-pass barriers are cheap.
struct Dispatcher<'c, 'p> {
    ctx: &'c GpuContext,
    pass: wgpu::ComputePass<'p>,
    n_pairs: u32,
}

impl<'c, 'p> Dispatcher<'c, 'p> {
    fn begin(ctx: &'c GpuContext, encoder: &'p mut wgpu::CommandEncoder, n_pairs: u32) -> Self {
        let pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        Self { ctx, pass, n_pairs }
    }

    fn run(
        &mut self,
        module: &'static str,
        entry: &'static str,
        params: PairParams,
        buffers: &[(u32, &wgpu::Buffer)],
    ) {
        let source: fn() -> String = match module {
            "pair_g2" => g2step_module_source,
            _ => fq12_module_source,
        };
        let pipeline = self.ctx.pipeline(module, entry, source);
        let params_buf = params.buffer(self.ctx);
        let mut all = vec![(0u32, &params_buf)];
        all.extend_from_slice(buffers);
        self.ctx.dispatch_in_pass(
            &mut self.pass,
            &pipeline,
            &all,
            (self.n_pairs.div_ceil(64), 1, 1),
        );
    }
}

/// Encodes a full Miller loop over pairs (p[i], q[i]) with on-the-fly line
/// computation. `p` is affine G1 (16 words each), `q` affine G2 (32 words).
pub fn encode_miller_computed(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    p: &wgpu::Buffer,
    q: &wgpu::Buffer,
    n_pairs: u32,
) -> MillerState {
    let f = ctx.empty_buffer(
        "miller-f",
        n_pairs as u64 * 96 * 4,
        wgpu::BufferUsages::COPY_SRC,
    );
    let mask = ctx.empty_buffer(
        "miller-mask",
        n_pairs as u64 * 4,
        wgpu::BufferUsages::empty(),
    );
    let scratch = ctx.empty_buffer(
        "miller-scratch",
        n_pairs as u64 * 144 * 4,
        wgpu::BufferUsages::empty(),
    );
    let t = ctx.empty_buffer(
        "miller-t",
        n_pairs as u64 * 48 * 4,
        wgpu::BufferUsages::empty(),
    );
    let lines = ctx.empty_buffer(
        "miller-lines",
        n_pairs as u64 * 48 * 4,
        wgpu::BufferUsages::empty(),
    );
    let q12 = ctx.empty_buffer(
        "miller-q12",
        n_pairs as u64 * 64 * 4,
        wgpu::BufferUsages::empty(),
    );
    // f_apply_a statically references the prepared-lines binding even on the
    // computed path (uniform branch); satisfy the layout with a stub.
    let dummy_prepared = ctx.empty_buffer("miller-dummy-prep", 4, wgpu::BufferUsages::empty());

    let base = PairParams {
        n_pairs,
        line_source: 0,
        step: 0,
        prep_stride: 0,
        q_source: 0,
        segment: 0,
        stride: 0,
        check_q: 1,
        prep_mod: 1,
    };

    let mut d = Dispatcher::begin(ctx, encoder, n_pairs);

    d.run(
        "pair_fq12",
        "f_init",
        base,
        &[(1, &f), (2, &mask), (3, q), (5, p)],
    );
    d.run("pair_g2", "pair_init", base, &[(2, &t), (3, q)]);

    fn sqr(
        d: &mut Dispatcher,
        base: PairParams,
        f: &wgpu::Buffer,
        mask: &wgpu::Buffer,
        scratch: &wgpu::Buffer,
    ) {
        d.run(
            "pair_fq12",
            "f_sqr_a",
            base,
            &[(1, f), (2, mask), (6, scratch)],
        );
        d.run(
            "pair_fq12",
            "f_sqr_b",
            base,
            &[(1, f), (2, mask), (6, scratch)],
        );
    }
    #[allow(clippy::too_many_arguments)]
    fn apply(
        d: &mut Dispatcher,
        base: PairParams,
        f: &wgpu::Buffer,
        mask: &wgpu::Buffer,
        lines: &wgpu::Buffer,
        p: &wgpu::Buffer,
        scratch: &wgpu::Buffer,
        dummy_prepared: &wgpu::Buffer,
    ) {
        d.run(
            "pair_fq12",
            "f_apply_a",
            base,
            &[
                (1, f),
                (2, mask),
                (4, lines),
                (5, p),
                (6, scratch),
                (7, dummy_prepared),
            ],
        );
        d.run(
            "pair_fq12",
            "f_apply_b",
            base,
            &[(1, f), (2, mask), (6, scratch)],
        );
    }

    let ate = ate_loop_count();
    for i in (1..ate.len()).rev() {
        if i != ate.len() - 1 {
            sqr(&mut d, base, &f, &mask, &scratch);
        }
        d.run("pair_g2", "pair_dbl_step", base, &[(2, &t), (4, &lines)]);
        apply(
            &mut d,
            base,
            &f,
            &mask,
            &lines,
            p,
            &scratch,
            &dummy_prepared,
        );
        let bit = ate[i - 1];
        if bit == 1 || bit == -1 {
            d.run(
                "pair_g2",
                "pair_add_step",
                PairParams {
                    q_source: if bit == 1 { 0 } else { 1 },
                    ..base
                },
                &[(2, &t), (3, q), (4, &lines), (7, &q12)],
            );
            apply(
                &mut d,
                base,
                &f,
                &mask,
                &lines,
                p,
                &scratch,
                &dummy_prepared,
            );
        }
    }

    d.run("pair_g2", "pair_frob", base, &[(3, q), (7, &q12)]);
    for q_source in [2u32, 3] {
        d.run(
            "pair_g2",
            "pair_add_step",
            PairParams { q_source, ..base },
            &[(2, &t), (3, q), (4, &lines), (7, &q12)],
        );
        apply(
            &mut d,
            base,
            &f,
            &mask,
            &lines,
            p,
            &scratch,
            &dummy_prepared,
        );
    }
    drop(d);

    MillerState {
        f,
        n_pairs,
        mask,
        scratch,
    }
}

/// Encodes a Miller loop over pairs (p[i], prepared_q[i % prep_mod]) using
/// uploaded prepared line coefficients.
#[allow(clippy::too_many_arguments)]
pub fn encode_miller_prepared(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    p: &wgpu::Buffer,
    prepared: &wgpu::Buffer,
    prep_mod: u32,
    n_pairs: u32,
) -> MillerState {
    let f = ctx.empty_buffer(
        "miller-f",
        n_pairs as u64 * 96 * 4,
        wgpu::BufferUsages::COPY_SRC,
    );
    let mask = ctx.empty_buffer(
        "miller-mask",
        n_pairs as u64 * 4,
        wgpu::BufferUsages::empty(),
    );
    let scratch = ctx.empty_buffer(
        "miller-scratch",
        n_pairs as u64 * 144 * 4,
        wgpu::BufferUsages::empty(),
    );
    let dummy_q = ctx.empty_buffer("miller-dummy-q", 4, wgpu::BufferUsages::empty());

    let stride = prepared_line_count();
    let mut step = 0u32;
    let base = PairParams {
        n_pairs,
        line_source: 1,
        step: 0,
        prep_stride: stride,
        q_source: 0,
        segment: 0,
        stride: 0,
        check_q: 0,
        prep_mod,
    };

    let mut d = Dispatcher::begin(ctx, encoder, n_pairs);

    d.run(
        "pair_fq12",
        "f_init",
        base,
        &[(1, &f), (2, &mask), (3, &dummy_q), (5, p)],
    );

    #[allow(clippy::too_many_arguments)]
    fn apply(
        d: &mut Dispatcher,
        base: PairParams,
        step: u32,
        f: &wgpu::Buffer,
        mask: &wgpu::Buffer,
        dummy_q: &wgpu::Buffer,
        p: &wgpu::Buffer,
        scratch: &wgpu::Buffer,
        prepared: &wgpu::Buffer,
    ) {
        let params = PairParams { step, ..base };
        d.run(
            "pair_fq12",
            "f_apply_a",
            params,
            &[
                (1, f),
                (2, mask),
                (4, dummy_q),
                (5, p),
                (6, scratch),
                (7, prepared),
            ],
        );
        d.run(
            "pair_fq12",
            "f_apply_b",
            params,
            &[(1, f), (2, mask), (6, scratch)],
        );
    }

    let ate = ate_loop_count();
    for i in (1..ate.len()).rev() {
        if i != ate.len() - 1 {
            d.run(
                "pair_fq12",
                "f_sqr_a",
                base,
                &[(1, &f), (2, &mask), (6, &scratch)],
            );
            d.run(
                "pair_fq12",
                "f_sqr_b",
                base,
                &[(1, &f), (2, &mask), (6, &scratch)],
            );
        }
        apply(
            &mut d, base, step, &f, &mask, &dummy_q, p, &scratch, prepared,
        );
        step += 1;
        if ate[i - 1] != 0 {
            apply(
                &mut d, base, step, &f, &mask, &dummy_q, p, &scratch, prepared,
            );
            step += 1;
        }
    }
    apply(
        &mut d, base, step, &f, &mask, &dummy_q, p, &scratch, prepared,
    );
    apply(
        &mut d,
        base,
        step + 1,
        &f,
        &mask,
        &dummy_q,
        p,
        &scratch,
        prepared,
    );
    drop(d);

    MillerState {
        f,
        n_pairs,
        mask,
        scratch,
    }
}

/// Reduces each contiguous `segment`-sized product of pairs into its first f
/// slot via repeated halving (segment must be a power of two).
pub fn encode_product_reduce(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    state: &MillerState,
    segment: u32,
    n_products: u32,
) {
    debug_assert!(segment.is_power_of_two());
    debug_assert_eq!(segment * n_products, state.n_pairs);
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
    let mut stride = segment / 2;
    while stride > 0 {
        let params = PairParams {
            n_pairs: state.n_pairs,
            line_source: 0,
            step: 0,
            prep_stride: 0,
            q_source: 0,
            segment,
            stride,
            check_q: 0,
            prep_mod: 1,
        };
        for entry in ["f_red_a", "f_red_b", "f_red_c"] {
            let pipeline = ctx.pipeline("pair_fq12", entry, fq12_module_source);
            let params_buf = params.buffer(ctx);
            ctx.dispatch_in_pass(
                &mut pass,
                &pipeline,
                &[(0, &params_buf), (1, &state.f), (6, &state.scratch)],
                (stride.div_ceil(64), n_products, 1),
            );
        }
        stride /= 2;
    }
    let _ = &state.mask;
}

/// Reads back the reduced product Miller outputs (one Fq12 per product at
/// slot `product * segment`).
pub async fn read_miller_products(
    ctx: &GpuContext,
    state: &MillerState,
    segment: u32,
    n_products: u32,
) -> Vec<Fq12> {
    let mut out = Vec::with_capacity(n_products as usize);
    for p in 0..n_products {
        let offset = (p * segment) as u64 * 96 * 4;
        let data = ctx
            .read_buffer(&state.f, offset, FQ12_WORDS as u64 * 4)
            .await;
        out.push(fq12_from_words(bytemuck::cast_slice(&data)));
    }
    out
}

/// CPU final exponentiation of a GPU Miller output.
pub fn final_exponentiation(f: Fq12) -> ark_bn254::Fq12 {
    ark_bn254::Bn254::final_exponentiation(MillerLoopOutput(f))
        .expect("nonzero miller output")
        .0
}

/// Packs arkworks prepared line coefficients for upload; all points must be
/// non-identity (Dory setup generators).
pub fn pack_prepared_g2(points: &[G2Affine]) -> Vec<u32> {
    let stride = prepared_line_count() as usize;
    let mut words = Vec::with_capacity(points.len() * stride * 48);
    for p in points {
        let prep: G2Prepared<ark_bn254::Config> = (*p).into();
        assert!(!prep.infinity, "prepared G2 setup point cannot be identity");
        assert_eq!(prep.ell_coeffs.len(), stride, "line schedule mismatch");
        for (c0, c1, c2) in &prep.ell_coeffs {
            words.extend_from_slice(&fq2_to_words(c0));
            words.extend_from_slice(&fq2_to_words(c1));
            words.extend_from_slice(&fq2_to_words(c2));
        }
    }
    words
}
