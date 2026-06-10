#![cfg(not(target_arch = "wasm32"))]

use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{CurveGroup, PrimeGroup};
use ark_ff::{PrimeField, UniformRand};
use ark_std::Zero;
use dory_gpu::repr::{
    bigint_to_words, g1_affine_to_words, g1_proj_from_words, g1_proj_to_words, g2_affine_to_words,
    g2_proj_from_words, g2_proj_to_words,
};
use dory_gpu::shader::{
    fq_header, g2_3b_header, ShaderBuilder, CURVE_TEST_WGSL, CURVE_WGSL, FIELD_WGSL, FQ2_WGSL,
    G1_GLUE, G1_SUBST, G2_GLUE, G2_SUBST,
};
use dory_gpu::GpuContext;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn g1_module() -> String {
    ShaderBuilder::new()
        .push(&fq_header())
        .push(FIELD_WGSL)
        .push(G1_GLUE)
        .push_subst(CURVE_WGSL, G1_SUBST)
        .push_subst(CURVE_TEST_WGSL, G1_SUBST)
        .build()
}

fn g2_module() -> String {
    ShaderBuilder::new()
        .push(&fq_header())
        .push(&g2_3b_header())
        .push(FIELD_WGSL)
        .push(FQ2_WGSL)
        .push(G2_GLUE)
        .push_subst(CURVE_WGSL, G2_SUBST)
        .push_subst(CURVE_TEST_WGSL, G2_SUBST)
        .build()
}

async fn run_kernel(
    ctx: &GpuContext,
    module: (&'static str, fn() -> String),
    entry: &'static str,
    a_words: &[u32],
    b_words: &[u32],
    out_words: usize,
) -> Vec<u32> {
    let buf_a = ctx.buffer_from(
        "a",
        bytemuck::cast_slice(a_words),
        wgpu::BufferUsages::empty(),
    );
    let buf_b = ctx.buffer_from(
        "b",
        bytemuck::cast_slice(b_words),
        wgpu::BufferUsages::empty(),
    );
    let buf_out = ctx.empty_buffer("out", (out_words * 4) as u64, wgpu::BufferUsages::COPY_SRC);
    let pipeline = ctx.pipeline(module.0, entry, module.1);
    let n = (out_words / 24) as u32;
    ctx.run_pass(
        &pipeline,
        &[&buf_a, &buf_b, &buf_out],
        (n.div_ceil(64), 1, 1),
    );
    let data = ctx.read_buffer(&buf_out, 0, (out_words * 4) as u64).await;
    bytemuck::cast_slice(&data).to_vec()
}

#[test]
fn g1_ops_match_arkworks() {
    pollster::block_on(async {
        let ctx = GpuContext::new().await.expect("gpu context");
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 256usize;

        let mut pa: Vec<G1Projective> = (0..n).map(|_| G1Projective::rand(&mut rng)).collect();
        let mut pb: Vec<G1Projective> = (0..n).map(|_| G1Projective::rand(&mut rng)).collect();
        // Edge cases: identity on either side, doubling via add, P + (-P).
        pa[0] = G1Projective::zero();
        pb[1] = G1Projective::zero();
        pa[2] = G1Projective::zero();
        pb[2] = G1Projective::zero();
        pb[3] = pa[3];
        pb[4] = -pa[4];

        let a_words: Vec<u32> = pa.iter().flat_map(g1_proj_to_words).collect();
        let b_words: Vec<u32> = pb.iter().flat_map(g1_proj_to_words).collect();

        let out = run_kernel(
            &ctx,
            ("g1_curve_test", g1_module),
            "t_point_add",
            &a_words,
            &b_words,
            n * 24,
        )
        .await;
        for i in 0..n {
            let got = g1_proj_from_words(&out[i * 24..(i + 1) * 24]);
            assert_eq!(got, pa[i] + pb[i], "g1 add mismatch at {i}");
        }

        let out = run_kernel(
            &ctx,
            ("g1_curve_test", g1_module),
            "t_point_double",
            &a_words,
            &b_words,
            n * 24,
        )
        .await;
        for i in 0..n {
            let got = g1_proj_from_words(&out[i * 24..(i + 1) * 24]);
            assert_eq!(got, pa[i] + pa[i], "g1 double mismatch at {i}");
        }

        // Mixed addition: affine second operand (no identity allowed there).
        let pb_aff: Vec<G1Affine> = pb
            .iter()
            .map(|p| {
                if p.is_zero() {
                    G1Affine::rand(&mut rng)
                } else {
                    p.into_affine()
                }
            })
            .collect();
        let b_aff_words: Vec<u32> = pb_aff.iter().flat_map(g1_affine_to_words).collect();
        let out = run_kernel(
            &ctx,
            ("g1_curve_test", g1_module),
            "t_point_madd",
            &a_words,
            &b_aff_words,
            n * 24,
        )
        .await;
        for i in 0..n {
            let got = g1_proj_from_words(&out[i * 24..(i + 1) * 24]);
            assert_eq!(got, pa[i] + pb_aff[i], "g1 madd mismatch at {i}");
        }
    });
}

#[test]
fn g1_scalar_mul_matches() {
    pollster::block_on(async {
        let ctx = GpuContext::new().await.expect("gpu context");
        let mut rng = ChaCha20Rng::seed_from_u64(43);
        let n = 64usize;
        let pts: Vec<G1Projective> = (0..n).map(|_| G1Projective::rand(&mut rng)).collect();
        let mut scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        scalars[0] = Fr::zero();
        scalars[1] = Fr::from(1u64);
        scalars[2] = -Fr::from(1u64);

        let a_words: Vec<u32> = pts.iter().flat_map(g1_proj_to_words).collect();
        let s_words: Vec<u32> = scalars
            .iter()
            .flat_map(|s| bigint_to_words(&s.into_bigint()))
            .collect();

        let out = run_kernel(
            &ctx,
            ("g1_curve_test", g1_module),
            "t_point_mul",
            &a_words,
            &s_words,
            n * 24,
        )
        .await;
        for i in 0..n {
            let got = g1_proj_from_words(&out[i * 24..(i + 1) * 24]);
            assert_eq!(got, pts[i] * scalars[i], "g1 scalar mul mismatch at {i}");
        }
    });
}

#[test]
fn g2_ops_match_arkworks() {
    pollster::block_on(async {
        let ctx = GpuContext::new().await.expect("gpu context");
        let mut rng = ChaCha20Rng::seed_from_u64(44);
        let n = 128usize;

        let mut pa: Vec<G2Projective> = (0..n).map(|_| G2Projective::rand(&mut rng)).collect();
        let mut pb: Vec<G2Projective> = (0..n).map(|_| G2Projective::rand(&mut rng)).collect();
        pa[0] = G2Projective::zero();
        pb[1] = G2Projective::zero();
        pb[2] = pa[2];
        pb[3] = -pa[3];
        pa[4] = G2Projective::generator();

        let a_words: Vec<u32> = pa.iter().flat_map(g2_proj_to_words).collect();
        let b_words: Vec<u32> = pb.iter().flat_map(g2_proj_to_words).collect();

        let out = run_kernel(
            &ctx,
            ("g2_curve_test", g2_module),
            "t_point_add",
            &a_words,
            &b_words,
            n * 48,
        )
        .await;
        for i in 0..n {
            let got = g2_proj_from_words(&out[i * 48..(i + 1) * 48]);
            assert_eq!(got, pa[i] + pb[i], "g2 add mismatch at {i}");
        }

        let out = run_kernel(
            &ctx,
            ("g2_curve_test", g2_module),
            "t_point_double",
            &a_words,
            &b_words,
            n * 48,
        )
        .await;
        for i in 0..n {
            let got = g2_proj_from_words(&out[i * 48..(i + 1) * 48]);
            assert_eq!(got, pa[i] + pa[i], "g2 double mismatch at {i}");
        }

        let pb_aff: Vec<G2Affine> = pb
            .iter()
            .map(|p| {
                if p.is_zero() {
                    G2Affine::rand(&mut rng)
                } else {
                    p.into_affine()
                }
            })
            .collect();
        let b_aff_words: Vec<u32> = pb_aff.iter().flat_map(g2_affine_to_words).collect();
        let out = run_kernel(
            &ctx,
            ("g2_curve_test", g2_module),
            "t_point_madd",
            &a_words,
            &b_aff_words,
            n * 48,
        )
        .await;
        for i in 0..n {
            let got = g2_proj_from_words(&out[i * 48..(i + 1) * 48]);
            assert_eq!(got, pa[i] + pb_aff[i], "g2 madd mismatch at {i}");
        }
    });
}

#[test]
fn g2_scalar_mul_matches() {
    pollster::block_on(async {
        let ctx = GpuContext::new().await.expect("gpu context");
        let mut rng = ChaCha20Rng::seed_from_u64(45);
        let n = 32usize;
        let pts: Vec<G2Projective> = (0..n).map(|_| G2Projective::rand(&mut rng)).collect();
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();

        let a_words: Vec<u32> = pts.iter().flat_map(g2_proj_to_words).collect();
        let s_words: Vec<u32> = scalars
            .iter()
            .flat_map(|s| bigint_to_words(&s.into_bigint()))
            .collect();

        let out = run_kernel(
            &ctx,
            ("g2_curve_test", g2_module),
            "t_point_mul",
            &a_words,
            &s_words,
            n * 48,
        )
        .await;
        for i in 0..n {
            let got = g2_proj_from_words(&out[i * 48..(i + 1) * 48]);
            assert_eq!(got, pts[i] * scalars[i], "g2 scalar mul mismatch at {i}");
        }
    });
}
