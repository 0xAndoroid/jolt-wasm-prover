#![cfg(not(target_arch = "wasm32"))]

use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{CurveGroup, VariableBaseMSM};
use ark_ff::UniformRand;
use ark_std::Zero;
use dory_gpu::msm::{encode_msm, encode_normalize, encode_prep_scalars, Curve, MsmCall};
use dory_gpu::repr::{
    fr_to_words, g1_affine_to_words, g1_proj_from_words, g1_proj_to_words, g2_affine_to_words,
    g2_proj_from_words,
};
use dory_gpu::GpuContext;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn upload_scalars(ctx: &GpuContext, scalars: &[Fr]) -> wgpu::Buffer {
    let words: Vec<u32> = scalars.iter().flat_map(fr_to_words).collect();
    ctx.buffer_from(
        "scalars",
        bytemuck::cast_slice(&words),
        wgpu::BufferUsages::empty(),
    )
}

async fn gpu_msm_g1_affine(ctx: &GpuContext, bases: &[G1Affine], scalars: &[Fr]) -> G1Projective {
    let base_words: Vec<u32> = bases.iter().flat_map(g1_affine_to_words).collect();
    let bases_buf = ctx.buffer_from(
        "bases",
        bytemuck::cast_slice(&base_words),
        wgpu::BufferUsages::empty(),
    );
    let mont = upload_scalars(ctx, scalars);
    let results = ctx.empty_buffer("results", 24 * 4, wgpu::BufferUsages::COPY_SRC);

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let prepared = encode_prep_scalars(ctx, &mut encoder, Curve::G1, &mont, scalars.len() as u32);
    encode_msm(
        ctx,
        &mut encoder,
        &MsmCall {
            curve: Curve::G1,
            bases: &bases_buf,
            scalars: &prepared,
            rows: 1,
            n: scalars.len() as u32,
            base_offset: 0,
            scalar_offset: 0,
            scalar_stride: 0,
            results: &results,
            out_offset: 0,
        },
    );
    ctx.queue.submit([encoder.finish()]);
    let data = ctx.read_buffer(&results, 0, 24 * 4).await;
    g1_proj_from_words(bytemuck::cast_slice(&data))
}

#[test]
fn g1_msm_matches_arkworks() {
    pollster::block_on(async {
        let ctx = GpuContext::new().await.expect("gpu context");
        let mut rng = ChaCha20Rng::seed_from_u64(100);

        for n in [1usize, 5, 64, 513, 1000, 1024, 2048] {
            let bases: Vec<G1Affine> = (0..n).map(|_| G1Affine::rand(&mut rng)).collect();
            let mut scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
            scalars[0] = Fr::zero();
            if n > 2 {
                scalars[2] = -Fr::from(1u64);
            }
            let expect: G1Projective = VariableBaseMSM::msm(&bases, &scalars).unwrap();
            let got = gpu_msm_g1_affine(&ctx, &bases, &scalars).await;
            assert_eq!(got, expect, "g1 msm mismatch at n={n}");
        }
    });
}

#[test]
fn g1_msm_projective_bases_matches() {
    pollster::block_on(async {
        let ctx = GpuContext::new().await.expect("gpu context");
        let mut rng = ChaCha20Rng::seed_from_u64(101);
        for n in [1usize, 4, 64, 777] {
            let bases_proj: Vec<G1Projective> =
                (0..n).map(|_| G1Projective::rand(&mut rng)).collect();
            let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
            let bases_affine = G1Projective::normalize_batch(&bases_proj);
            let expect: G1Projective = VariableBaseMSM::msm(&bases_affine, &scalars).unwrap();

            let base_words: Vec<u32> = bases_proj.iter().flat_map(g1_proj_to_words).collect();
            let proj_buf = ctx.buffer_from(
                "bases-proj",
                bytemuck::cast_slice(&base_words),
                wgpu::BufferUsages::empty(),
            );
            let mont = upload_scalars(&ctx, &scalars);
            let results = ctx.empty_buffer("results", 24 * 4, wgpu::BufferUsages::COPY_SRC);

            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            let bases_buf = encode_normalize(&ctx, &mut encoder, Curve::G1, &proj_buf, n as u32);
            let prepared = encode_prep_scalars(&ctx, &mut encoder, Curve::G1, &mont, n as u32);
            encode_msm(
                &ctx,
                &mut encoder,
                &MsmCall {
                    curve: Curve::G1,
                    bases: &bases_buf,
                    scalars: &prepared,
                    rows: 1,
                    n: n as u32,
                    base_offset: 0,
                    scalar_offset: 0,
                    scalar_stride: 0,
                    results: &results,
                    out_offset: 0,
                },
            );
            ctx.queue.submit([encoder.finish()]);
            let data = ctx.read_buffer(&results, 0, 24 * 4).await;
            let got = g1_proj_from_words(bytemuck::cast_slice(&data));
            assert_eq!(got, expect, "g1 msm (projective bases) mismatch at n={n}");
        }
    });
}

#[test]
fn g1_msm_batched_rows_match() {
    pollster::block_on(async {
        let ctx = GpuContext::new().await.expect("gpu context");
        let mut rng = ChaCha20Rng::seed_from_u64(102);
        let rows = 8u32;
        let n = 640usize;

        let bases: Vec<G1Affine> = (0..n).map(|_| G1Affine::rand(&mut rng)).collect();
        let scalars: Vec<Fr> = (0..rows as usize * n).map(|_| Fr::rand(&mut rng)).collect();

        let base_words: Vec<u32> = bases.iter().flat_map(g1_affine_to_words).collect();
        let bases_buf = ctx.buffer_from(
            "bases",
            bytemuck::cast_slice(&base_words),
            wgpu::BufferUsages::empty(),
        );
        let mont = upload_scalars(&ctx, &scalars);
        let results = ctx.empty_buffer(
            "results",
            rows as u64 * 24 * 4,
            wgpu::BufferUsages::COPY_SRC,
        );

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let prepared =
            encode_prep_scalars(&ctx, &mut encoder, Curve::G1, &mont, scalars.len() as u32);
        encode_msm(
            &ctx,
            &mut encoder,
            &MsmCall {
                curve: Curve::G1,
                bases: &bases_buf,
                scalars: &prepared,
                rows,
                n: n as u32,
                base_offset: 0,
                scalar_offset: 0,
                scalar_stride: n as u32,
                results: &results,
                out_offset: 0,
            },
        );
        ctx.queue.submit([encoder.finish()]);
        let data = ctx.read_buffer(&results, 0, rows as u64 * 24 * 4).await;
        let words: &[u32] = bytemuck::cast_slice(&data);
        for r in 0..rows as usize {
            let expect: G1Projective =
                VariableBaseMSM::msm(&bases, &scalars[r * n..(r + 1) * n]).unwrap();
            let got = g1_proj_from_words(&words[r * 24..(r + 1) * 24]);
            assert_eq!(got, expect, "row {r} mismatch");
        }
    });
}

#[test]
fn g2_msm_matches_arkworks() {
    pollster::block_on(async {
        let ctx = GpuContext::new().await.expect("gpu context");
        let mut rng = ChaCha20Rng::seed_from_u64(103);

        for n in [1usize, 100, 1024] {
            let bases: Vec<G2Affine> = (0..n).map(|_| G2Affine::rand(&mut rng)).collect();
            let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
            let expect: G2Projective = VariableBaseMSM::msm(&bases, &scalars).unwrap();

            let base_words: Vec<u32> = bases.iter().flat_map(g2_affine_to_words).collect();
            let bases_buf = ctx.buffer_from(
                "bases",
                bytemuck::cast_slice(&base_words),
                wgpu::BufferUsages::empty(),
            );
            let mont = upload_scalars(&ctx, &scalars);
            let results = ctx.empty_buffer("results", 48 * 4, wgpu::BufferUsages::COPY_SRC);

            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            let prepared = encode_prep_scalars(&ctx, &mut encoder, Curve::G2, &mont, n as u32);
            encode_msm(
                &ctx,
                &mut encoder,
                &MsmCall {
                    curve: Curve::G2,
                    bases: &bases_buf,
                    scalars: &prepared,
                    rows: 1,
                    n: n as u32,
                    base_offset: 0,
                    scalar_offset: 0,
                    scalar_stride: 0,
                    results: &results,
                    out_offset: 0,
                },
            );
            ctx.queue.submit([encoder.finish()]);
            let data = ctx.read_buffer(&results, 0, 48 * 4).await;
            let got = g2_proj_from_words(bytemuck::cast_slice(&data));
            assert_eq!(got, expect, "g2 msm mismatch at n={n}");
        }
    });
}
