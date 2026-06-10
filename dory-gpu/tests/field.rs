#![cfg(not(target_arch = "wasm32"))]

use ark_bn254::{Fq, Fq2};
use ark_ff::{AdditiveGroup, Field, PrimeField, UniformRand};
use ark_std::Zero;
use dory_gpu::repr::{fq2_from_words, fq2_to_words, fq_from_words, fq_to_words};
use dory_gpu::shader::{fq_header, FIELD_TEST_WGSL, FIELD_WGSL, FQ2_WGSL};
use dory_gpu::GpuContext;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn test_inputs() -> Vec<Fq> {
    let mut rng = ChaCha20Rng::seed_from_u64(7);
    let mut v: Vec<Fq> = (0..4096).map(|_| Fq::rand(&mut rng)).collect();
    v[0] = Fq::ZERO;
    v[1] = Fq::ONE;
    v[2] = -Fq::ONE;
    v[3] = -Fq::ONE - Fq::ONE;
    v[4] = Fq::from(2u64);
    v[5] = Fq::from(u64::MAX);
    v
}

fn module_source() -> String {
    dory_gpu::shader::ShaderBuilder::new()
        .push(&fq_header())
        .push(&FIELD_WGSL)
        .push(FQ2_WGSL)
        .push(FIELD_TEST_WGSL)
        .build()
}

async fn run_fq_kernel(ctx: &GpuContext, entry: &'static str, a: &[Fq], b: &[Fq]) -> Vec<Fq> {
    let a_words: Vec<u32> = a.iter().flat_map(fq_to_words).collect();
    let b_words: Vec<u32> = b.iter().flat_map(fq_to_words).collect();
    let buf_a = ctx.buffer_from(
        "a",
        bytemuck::cast_slice(&a_words),
        wgpu::BufferUsages::empty(),
    );
    let buf_b = ctx.buffer_from(
        "b",
        bytemuck::cast_slice(&b_words),
        wgpu::BufferUsages::empty(),
    );
    let buf_out = ctx.empty_buffer(
        "out",
        (a.len() * 8 * 4) as u64,
        wgpu::BufferUsages::COPY_SRC,
    );
    let pipeline = ctx.pipeline("field_test", entry, module_source);
    ctx.run_pass(
        &pipeline,
        &[&buf_a, &buf_b, &buf_out],
        ((a.len() as u32).div_ceil(64), 1, 1),
    );
    let data = ctx.read_buffer(&buf_out, 0, (a.len() * 8 * 4) as u64).await;
    let words: &[u32] = bytemuck::cast_slice(&data);
    words.chunks(8).map(fq_from_words).collect()
}

async fn run_fq2_kernel(ctx: &GpuContext, entry: &'static str, a: &[Fq2], b: &[Fq2]) -> Vec<Fq2> {
    let a_words: Vec<u32> = a.iter().flat_map(fq2_to_words).collect();
    let b_words: Vec<u32> = b.iter().flat_map(fq2_to_words).collect();
    let buf_a = ctx.buffer_from(
        "a",
        bytemuck::cast_slice(&a_words),
        wgpu::BufferUsages::empty(),
    );
    let buf_b = ctx.buffer_from(
        "b",
        bytemuck::cast_slice(&b_words),
        wgpu::BufferUsages::empty(),
    );
    let buf_out = ctx.empty_buffer(
        "out",
        (a.len() * 16 * 4) as u64,
        wgpu::BufferUsages::COPY_SRC,
    );
    let pipeline = ctx.pipeline("field_test", entry, module_source);
    ctx.run_pass(
        &pipeline,
        &[&buf_a, &buf_b, &buf_out],
        ((a.len() as u32).div_ceil(64), 1, 1),
    );
    let data = ctx
        .read_buffer(&buf_out, 0, (a.len() * 16 * 4) as u64)
        .await;
    let words: &[u32] = bytemuck::cast_slice(&data);
    words.chunks(16).map(fq2_from_words).collect()
}

#[test]
fn fq_ops_match_arkworks() {
    pollster::block_on(async {
        let ctx = GpuContext::new().await.expect("gpu context");
        let a = test_inputs();
        let mut b = test_inputs();
        b.rotate_left(17);

        let mul = run_fq_kernel(&ctx, "t_mul", &a, &b).await;
        let add = run_fq_kernel(&ctx, "t_add", &a, &b).await;
        let sub = run_fq_kernel(&ctx, "t_sub", &a, &b).await;
        let neg = run_fq_kernel(&ctx, "t_neg", &a, &b).await;
        let sqr_chain = run_fq_kernel(&ctx, "t_sqr_chain", &a, &b).await;

        for i in 0..a.len() {
            assert_eq!(mul[i], a[i] * b[i], "mul mismatch at {i}");
            assert_eq!(add[i], a[i] + b[i], "add mismatch at {i}");
            assert_eq!(sub[i], a[i] - b[i], "sub mismatch at {i}");
            assert_eq!(neg[i], -a[i], "neg mismatch at {i}");
            let mut expect = a[i];
            for _ in 0..100 {
                expect = expect * expect;
            }
            assert_eq!(sqr_chain[i], expect, "sqr chain mismatch at {i}");
        }
    });
}

#[test]
fn fq_from_mont_matches() {
    pollster::block_on(async {
        let ctx = GpuContext::new().await.expect("gpu context");
        let a = test_inputs();
        let out = run_fq_kernel(&ctx, "t_from_mont", &a, &a).await;
        for i in 0..a.len() {
            // fe_from_mont returns the canonical residue as a raw bigint; the
            // readback path reinterprets it as a Montgomery value, so compare
            // against the bigint of the input.
            let expect = fq_from_words(&dory_gpu::repr::bigint_to_words(&a[i].into_bigint()));
            assert_eq!(out[i], expect, "from_mont mismatch at {i}");
        }
    });
}

#[test]
fn fq2_ops_match_arkworks() {
    pollster::block_on(async {
        let ctx = GpuContext::new().await.expect("gpu context");
        let mut rng = ChaCha20Rng::seed_from_u64(11);
        let mut a: Vec<Fq2> = (0..2048).map(|_| Fq2::rand(&mut rng)).collect();
        let b: Vec<Fq2> = (0..2048).map(|_| Fq2::rand(&mut rng)).collect();
        a[0] = Fq2::ZERO;
        a[1] = Fq2::ONE;

        let mul = run_fq2_kernel(&ctx, "t_fq2_mul", &a, &b).await;
        let sqr = run_fq2_kernel(&ctx, "t_fq2_sqr", &a, &b).await;
        let nr = run_fq2_kernel(&ctx, "t_fq2_mul_by_nonresidue", &a, &b).await;

        let xi = Fq2::new(Fq::from(9u64), Fq::ONE);
        for i in 0..a.len() {
            assert_eq!(mul[i], a[i] * b[i], "fq2 mul mismatch at {i}");
            assert_eq!(sqr[i], a[i] * a[i], "fq2 sqr mismatch at {i}");
            assert_eq!(nr[i], a[i] * xi, "fq2 nonresidue mismatch at {i}");
        }
    });
}

#[test]
fn fq_is_zero_edge() {
    pollster::block_on(async {
        let ctx = GpuContext::new().await.expect("gpu context");
        // 0 * x == 0 and 0 + 0 == 0 sanity through the GPU path
        let zeros = vec![Fq::ZERO; 64];
        let mut rng = ChaCha20Rng::seed_from_u64(3);
        let xs: Vec<Fq> = (0..64).map(|_| Fq::rand(&mut rng)).collect();
        let out = run_fq_kernel(&ctx, "t_mul", &zeros, &xs).await;
        assert!(out.iter().all(|v| v.is_zero()));
    });
}
