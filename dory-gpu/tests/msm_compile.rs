#![cfg(not(target_arch = "wasm32"))]

use ark_bn254::{Fr, G1Affine, G1Projective};
use ark_ec::VariableBaseMSM;
use ark_ff::UniformRand;
use dory_gpu::msm::{encode_msm, encode_prep_scalars, Curve, MsmCall};
use dory_gpu::repr::{fr_to_words, g1_affine_to_words, g1_proj_from_words};
use dory_gpu::GpuContext;
use rand::SeedableRng;

#[test]
fn msm_tiny_executes() {
    pollster::block_on(async {
        let ctx = GpuContext::new().await.expect("gpu context");
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(1);
        let n = 4usize;
        let bases: Vec<G1Affine> = (0..n).map(|_| G1Affine::rand(&mut rng)).collect();
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();

        let base_words: Vec<u32> = bases.iter().flat_map(g1_affine_to_words).collect();
        let bases_buf = ctx.buffer_from(
            "bases",
            bytemuck::cast_slice(&base_words),
            wgpu::BufferUsages::empty(),
        );
        let s_words: Vec<u32> = scalars.iter().flat_map(fr_to_words).collect();
        let mont = ctx.buffer_from(
            "scalars",
            bytemuck::cast_slice(&s_words),
            wgpu::BufferUsages::empty(),
        );
        let results = ctx.empty_buffer("results", 24 * 4, wgpu::BufferUsages::COPY_SRC);

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
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
        let expect: G1Projective = VariableBaseMSM::msm(&bases, &scalars).unwrap();
        assert_eq!(got, expect, "tiny msm mismatch");
    });
}
