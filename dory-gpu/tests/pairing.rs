#![cfg(not(target_arch = "wasm32"))]

use ark_bn254::{Bn254, Fq12, G1Affine, G2Affine};
use ark_ec::pairing::Pairing;
use ark_ec::AffineRepr;
use ark_ff::UniformRand;
use dory_gpu::pairing::{
    encode_miller_computed, encode_miller_prepared, encode_product_reduce, final_exponentiation,
    pack_prepared_g2, read_miller_products,
};
use dory_gpu::repr::{g1_affine_to_words, g2_affine_to_words, pack_slice};
use dory_gpu::GpuContext;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn upload_g1(ctx: &GpuContext, ps: &[G1Affine]) -> wgpu::Buffer {
    let words: Vec<u32> = ps
        .iter()
        .flat_map(|p| {
            if p.is_zero() {
                [0u32; 16]
            } else {
                g1_affine_to_words(p)
            }
        })
        .collect();
    ctx.buffer_from(
        "p",
        bytemuck::cast_slice(&words),
        wgpu::BufferUsages::empty(),
    )
}

fn upload_g2(ctx: &GpuContext, qs: &[G2Affine]) -> wgpu::Buffer {
    let words: Vec<u32> = qs
        .iter()
        .flat_map(|q| {
            if q.is_zero() {
                [0u32; 32]
            } else {
                g2_affine_to_words(q)
            }
        })
        .collect();
    ctx.buffer_from(
        "q",
        bytemuck::cast_slice(&words),
        wgpu::BufferUsages::empty(),
    )
}

fn cpu_miller(ps: &[G1Affine], qs: &[G2Affine]) -> Fq12 {
    Bn254::multi_miller_loop(ps.to_vec(), qs.to_vec()).0
}

#[test]
fn miller_computed_matches_arkworks() {
    pollster::block_on(async {
        let ctx = GpuContext::new().await.expect("gpu context");
        let mut rng = ChaCha20Rng::seed_from_u64(300);

        for n in [1usize, 4, 16] {
            let ps: Vec<G1Affine> = (0..n).map(|_| G1Affine::rand(&mut rng)).collect();
            let qs: Vec<G2Affine> = (0..n).map(|_| G2Affine::rand(&mut rng)).collect();

            let p_buf = upload_g1(&ctx, &ps);
            let q_buf = upload_g2(&ctx, &qs);
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            let state = encode_miller_computed(&ctx, &mut encoder, &p_buf, &q_buf, n as u32);
            encode_product_reduce(&ctx, &mut encoder, &state, n as u32, 1);
            ctx.queue.submit([encoder.finish()]);

            let gpu_f = read_miller_products(&ctx, &state, n as u32, 1).await[0];
            let cpu_f = cpu_miller(&ps, &qs);
            assert_eq!(gpu_f, cpu_f, "miller output mismatch at n={n}");

            let gpu_gt = final_exponentiation(gpu_f);
            let cpu_gt = Bn254::multi_pairing(ps.clone(), qs.clone()).0;
            assert_eq!(gpu_gt, cpu_gt, "pairing mismatch at n={n}");
        }
    });
}

#[test]
fn miller_computed_handles_identity_pairs() {
    pollster::block_on(async {
        let ctx = GpuContext::new().await.expect("gpu context");
        let mut rng = ChaCha20Rng::seed_from_u64(301);
        let n = 8usize;
        let mut ps: Vec<G1Affine> = (0..n).map(|_| G1Affine::rand(&mut rng)).collect();
        let mut qs: Vec<G2Affine> = (0..n).map(|_| G2Affine::rand(&mut rng)).collect();
        ps[2] = G1Affine::zero();
        qs[5] = G2Affine::zero();

        let p_buf = upload_g1(&ctx, &ps);
        let q_buf = upload_g2(&ctx, &qs);
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let state = encode_miller_computed(&ctx, &mut encoder, &p_buf, &q_buf, n as u32);
        encode_product_reduce(&ctx, &mut encoder, &state, n as u32, 1);
        ctx.queue.submit([encoder.finish()]);

        let gpu_f = read_miller_products(&ctx, &state, n as u32, 1).await[0];
        let cpu_f = cpu_miller(&ps, &qs);
        assert_eq!(gpu_f, cpu_f, "miller output mismatch with identities");
    });
}

#[test]
fn miller_prepared_matches_arkworks() {
    pollster::block_on(async {
        let ctx = GpuContext::new().await.expect("gpu context");
        let mut rng = ChaCha20Rng::seed_from_u64(302);
        let n = 8usize;

        // Two products sharing the same prepared generators (Dory D1L/D1R).
        let gens: Vec<G2Affine> = (0..n).map(|_| G2Affine::rand(&mut rng)).collect();
        let ps_l: Vec<G1Affine> = (0..n).map(|_| G1Affine::rand(&mut rng)).collect();
        let ps_r: Vec<G1Affine> = (0..n).map(|_| G1Affine::rand(&mut rng)).collect();

        let prepared_words = pack_prepared_g2(&gens);
        let prepared = ctx.buffer_from(
            "prepared",
            bytemuck::cast_slice(&prepared_words),
            wgpu::BufferUsages::empty(),
        );
        let all_p: Vec<G1Affine> = ps_l.iter().chain(ps_r.iter()).copied().collect();
        let p_buf = upload_g1(&ctx, &all_p);

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let state = encode_miller_prepared(
            &ctx,
            &mut encoder,
            &p_buf,
            &prepared,
            n as u32,
            2 * n as u32,
        );
        encode_product_reduce(&ctx, &mut encoder, &state, n as u32, 2);
        ctx.queue.submit([encoder.finish()]);

        let products = read_miller_products(&ctx, &state, n as u32, 2).await;
        assert_eq!(products[0], cpu_miller(&ps_l, &gens), "D1L mismatch");
        assert_eq!(products[1], cpu_miller(&ps_r, &gens), "D1R mismatch");
        let _ = pack_slice(&[0u32], |_| [0u32; 1]);
    });
}
