#![cfg(not(target_arch = "wasm32"))]

use ark_bn254::{Fr, G1Affine, G1Projective, G2Projective};
use ark_ff::{PrimeField, UniformRand};
use dory_gpu::fold::{
    build_fixed_base_table_g2, encode_fixed_base_mul, encode_fold_add_scaled_base,
    encode_fold_scalars, encode_fold_scale_add, encode_vmv,
};
use dory_gpu::msm::{encode_prep_scalars, Curve};
use dory_gpu::repr::{
    bigint_to_words, fr_from_words, fr_to_words, g1_affine_to_words, g1_proj_from_words,
    g1_proj_to_words, g2_proj_from_words, g2_proj_to_words,
};
use dory_gpu::GpuContext;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

#[test]
fn fold_scale_add_matches() {
    pollster::block_on(async {
        let ctx = GpuContext::new().await.expect("gpu context");
        let mut rng = ChaCha20Rng::seed_from_u64(200);
        let n2 = 65usize;
        let k = Fr::rand(&mut rng);

        // G1: v <- k*v_L + v_R in place over one buffer of 2*n2 points.
        let v: Vec<G1Projective> = (0..2 * n2).map(|_| G1Projective::rand(&mut rng)).collect();
        let v_words: Vec<u32> = v.iter().flat_map(g1_proj_to_words).collect();
        let buf = ctx.buffer_from(
            "v",
            bytemuck::cast_slice(&v_words),
            wgpu::BufferUsages::COPY_SRC,
        );
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encode_fold_scale_add(
            &ctx,
            &mut encoder,
            Curve::G1,
            &buf,
            &bigint_to_words(&k.into_bigint()),
            n2 as u32,
            0,
            n2 as u32,
            0,
        );
        ctx.queue.submit([encoder.finish()]);
        let data = ctx.read_buffer(&buf, 0, (n2 * 24 * 4) as u64).await;
        let words: &[u32] = bytemuck::cast_slice(&data);
        for i in 0..n2 {
            let got = g1_proj_from_words(&words[i * 24..(i + 1) * 24]);
            assert_eq!(got, v[i] * k + v[n2 + i], "g1 fold mismatch at {i}");
        }

        // G2 variant.
        let v2: Vec<G2Projective> = (0..2 * n2).map(|_| G2Projective::rand(&mut rng)).collect();
        let v2_words: Vec<u32> = v2.iter().flat_map(g2_proj_to_words).collect();
        let buf2 = ctx.buffer_from(
            "v2",
            bytemuck::cast_slice(&v2_words),
            wgpu::BufferUsages::COPY_SRC,
        );
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encode_fold_scale_add(
            &ctx,
            &mut encoder,
            Curve::G2,
            &buf2,
            &bigint_to_words(&k.into_bigint()),
            n2 as u32,
            0,
            n2 as u32,
            0,
        );
        ctx.queue.submit([encoder.finish()]);
        let data = ctx.read_buffer(&buf2, 0, (n2 * 48 * 4) as u64).await;
        let words: &[u32] = bytemuck::cast_slice(&data);
        for i in 0..n2 {
            let got = g2_proj_from_words(&words[i * 48..(i + 1) * 48]);
            assert_eq!(got, v2[i] * k + v2[n2 + i], "g2 fold mismatch at {i}");
        }
    });
}

#[test]
fn fold_add_scaled_base_matches() {
    pollster::block_on(async {
        let ctx = GpuContext::new().await.expect("gpu context");
        let mut rng = ChaCha20Rng::seed_from_u64(201);
        let n = 70usize;
        let k = Fr::rand(&mut rng);

        let v: Vec<G1Projective> = (0..n).map(|_| G1Projective::rand(&mut rng)).collect();
        let g: Vec<G1Affine> = (0..n).map(|_| G1Affine::rand(&mut rng)).collect();
        let v_words: Vec<u32> = v.iter().flat_map(g1_proj_to_words).collect();
        let g_words: Vec<u32> = g.iter().flat_map(g1_affine_to_words).collect();
        let v_buf = ctx.buffer_from(
            "v",
            bytemuck::cast_slice(&v_words),
            wgpu::BufferUsages::COPY_SRC,
        );
        let g_buf = ctx.buffer_from(
            "g",
            bytemuck::cast_slice(&g_words),
            wgpu::BufferUsages::empty(),
        );
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encode_fold_add_scaled_base(
            &ctx,
            &mut encoder,
            Curve::G1,
            &v_buf,
            &g_buf,
            &bigint_to_words(&k.into_bigint()),
            n as u32,
            0,
            0,
            0,
        );
        ctx.queue.submit([encoder.finish()]);
        let data = ctx.read_buffer(&v_buf, 0, (n * 24 * 4) as u64).await;
        let words: &[u32] = bytemuck::cast_slice(&data);
        for i in 0..n {
            let got = g1_proj_from_words(&words[i * 24..(i + 1) * 24]);
            assert_eq!(got, v[i] + g[i] * k, "fold add scaled base mismatch at {i}");
        }
    });
}

#[test]
fn fixed_base_mul_matches() {
    pollster::block_on(async {
        let ctx = GpuContext::new().await.expect("gpu context");
        let mut rng = ChaCha20Rng::seed_from_u64(202);
        let n = 100usize;
        let base = G2Projective::rand(&mut rng);
        let mut scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        scalars[0] = Fr::from(0u64);
        scalars[1] = Fr::from(1u64);

        let table_words = build_fixed_base_table_g2(&base);
        let table = ctx.buffer_from(
            "table",
            bytemuck::cast_slice(&table_words),
            wgpu::BufferUsages::empty(),
        );
        let s_words: Vec<u32> = scalars.iter().flat_map(fr_to_words).collect();
        let mont = ctx.buffer_from(
            "scalars",
            bytemuck::cast_slice(&s_words),
            wgpu::BufferUsages::empty(),
        );
        let out = ctx.empty_buffer("out", (n * 48 * 4) as u64, wgpu::BufferUsages::COPY_SRC);

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let prepared = encode_prep_scalars(&ctx, &mut encoder, Curve::G1, &mont, n as u32);
        encode_fixed_base_mul(
            &ctx,
            &mut encoder,
            Curve::G2,
            &out,
            &table,
            &prepared,
            n as u32,
            0,
            0,
        );
        ctx.queue.submit([encoder.finish()]);
        let data = ctx.read_buffer(&out, 0, (n * 48 * 4) as u64).await;
        let words: &[u32] = bytemuck::cast_slice(&data);
        for i in 0..n {
            let got = g2_proj_from_words(&words[i * 48..(i + 1) * 48]);
            assert_eq!(got, base * scalars[i], "fixed base mul mismatch at {i}");
        }
    });
}

#[test]
fn vmv_and_fold_scalars_match() {
    pollster::block_on(async {
        let ctx = GpuContext::new().await.expect("gpu context");
        let mut rng = ChaCha20Rng::seed_from_u64(203);
        let rows = 32usize;
        let cols = 48usize;

        let matrix: Vec<Fr> = (0..rows * cols).map(|_| Fr::rand(&mut rng)).collect();
        let left: Vec<Fr> = (0..rows).map(|_| Fr::rand(&mut rng)).collect();
        let m_words: Vec<u32> = matrix.iter().flat_map(fr_to_words).collect();
        let l_words: Vec<u32> = left.iter().flat_map(fr_to_words).collect();
        let m_buf = ctx.buffer_from(
            "m",
            bytemuck::cast_slice(&m_words),
            wgpu::BufferUsages::empty(),
        );
        let l_buf = ctx.buffer_from(
            "l",
            bytemuck::cast_slice(&l_words),
            wgpu::BufferUsages::empty(),
        );
        let out = ctx.empty_buffer("out", (cols * 32) as u64, wgpu::BufferUsages::COPY_SRC);

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encode_vmv(
            &ctx,
            &mut encoder,
            &m_buf,
            &l_buf,
            &out,
            rows as u32,
            cols as u32,
            0,
            0,
        );
        ctx.queue.submit([encoder.finish()]);
        let data = ctx.read_buffer(&out, 0, (cols * 32) as u64).await;
        let words: &[u32] = bytemuck::cast_slice(&data);
        for j in 0..cols {
            let got = fr_from_words(&words[j * 8..(j + 1) * 8]);
            let expect: Fr = (0..rows).map(|i| left[i] * matrix[i * cols + j]).sum();
            assert_eq!(got, expect, "vmv mismatch at column {j}");
        }

        // fold_scalars: s <- k*s_L + s_R
        let n2 = 33usize;
        let k = Fr::rand(&mut rng);
        let s: Vec<Fr> = (0..2 * n2).map(|_| Fr::rand(&mut rng)).collect();
        let s_words: Vec<u32> = s.iter().flat_map(fr_to_words).collect();
        let s_buf = ctx.buffer_from(
            "s",
            bytemuck::cast_slice(&s_words),
            wgpu::BufferUsages::COPY_SRC,
        );
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encode_fold_scalars(
            &ctx,
            &mut encoder,
            &s_buf,
            &fr_to_words(&k),
            n2 as u32,
            0,
            n2 as u32,
            0,
        );
        ctx.queue.submit([encoder.finish()]);
        let data = ctx.read_buffer(&s_buf, 0, (n2 * 32) as u64).await;
        let words: &[u32] = bytemuck::cast_slice(&data);
        for i in 0..n2 {
            let got = fr_from_words(&words[i * 8..(i + 1) * 8]);
            assert_eq!(got, k * s[i] + s[n2 + i], "fold scalars mismatch at {i}");
        }
    });
}

#[test]
fn glv_folds_match_cpu() {
    use ark_ec::CurveGroup;
    use dory_gpu::fold::{
        encode_glv_fold_add_scaled_base, encode_glv_fold_scale_add, glv_decompose,
    };
    use dory_gpu::msm::Curve;
    use dory_gpu::repr::{
        g1_affine_to_words, g1_proj_from_words, g1_proj_to_words, g2_affine_to_words,
        g2_proj_from_words, g2_proj_to_words, pack_slice, G1_PROJ_WORDS, G2_PROJ_WORDS,
    };

    pollster::block_on(async {
        let ctx = GpuContext::new().await.expect("gpu context");
        let mut rng = ChaCha20Rng::seed_from_u64(500);
        let n = 64usize;

        for trial in 0..3 {
            let k = Fr::rand(&mut rng);

            // --- G1 scale_add: v[i] = k*v[i] + v[n + i] ---
            let mut pts: Vec<ark_bn254::G1Projective> = (0..2 * n)
                .map(|_| ark_bn254::G1Projective::rand(&mut rng))
                .collect();
            pts[5] = ark_bn254::G1Projective::default(); // identity input
            let buf = ctx.buffer_from(
                "glv-v",
                bytemuck::cast_slice(&pack_slice(&pts, g1_proj_to_words)),
                wgpu::BufferUsages::COPY_SRC,
            );
            let d = glv_decompose(Curve::G1, &k);
            let mut enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            encode_glv_fold_scale_add(
                &ctx,
                &mut enc,
                Curve::G1,
                &buf,
                &d,
                n as u32,
                0,
                n as u32,
                0,
            );
            ctx.queue.submit([enc.finish()]);
            let bytes = ctx
                .read_buffer(&buf, 0, n as u64 * G1_PROJ_WORDS as u64 * 4)
                .await;
            let words: &[u32] = bytemuck::cast_slice(&bytes);
            for (i, chunk) in words.chunks_exact(G1_PROJ_WORDS).enumerate() {
                let got = g1_proj_from_words(chunk);
                let want = pts[i] * k + pts[n + i];
                assert_eq!(
                    got, want,
                    "G1 glv scale_add mismatch at {i} (trial {trial})"
                );
            }

            // --- G1 add_scaled_base: v[i] = v[i] + k*base[i] ---
            let bases: Vec<ark_bn254::G1Affine> = (0..n)
                .map(|_| ark_bn254::G1Projective::rand(&mut rng).into_affine())
                .collect();
            let vs: Vec<ark_bn254::G1Projective> = (0..n)
                .map(|_| ark_bn254::G1Projective::rand(&mut rng))
                .collect();
            let vbuf = ctx.buffer_from(
                "glv-v2",
                bytemuck::cast_slice(&pack_slice(&vs, g1_proj_to_words)),
                wgpu::BufferUsages::COPY_SRC,
            );
            let bbuf = ctx.buffer_from(
                "glv-b",
                bytemuck::cast_slice(&pack_slice(&bases, g1_affine_to_words)),
                wgpu::BufferUsages::empty(),
            );
            let mut enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            encode_glv_fold_add_scaled_base(
                &ctx,
                &mut enc,
                Curve::G1,
                &vbuf,
                &bbuf,
                &d,
                n as u32,
                0,
                0,
                0,
            );
            ctx.queue.submit([enc.finish()]);
            let bytes = ctx
                .read_buffer(&vbuf, 0, n as u64 * G1_PROJ_WORDS as u64 * 4)
                .await;
            let words: &[u32] = bytemuck::cast_slice(&bytes);
            for (i, chunk) in words.chunks_exact(G1_PROJ_WORDS).enumerate() {
                let got = g1_proj_from_words(chunk);
                let want = vs[i] + bases[i] * k;
                assert_eq!(got, want, "G1 glv add_scaled_base mismatch at {i}");
            }

            // --- G2 scale_add ---
            let pts2: Vec<ark_bn254::G2Projective> = (0..2 * n)
                .map(|_| ark_bn254::G2Projective::rand(&mut rng))
                .collect();
            let buf2 = ctx.buffer_from(
                "glv-v-g2",
                bytemuck::cast_slice(&pack_slice(&pts2, g2_proj_to_words)),
                wgpu::BufferUsages::COPY_SRC,
            );
            let d2 = glv_decompose(Curve::G2, &k);
            let mut enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            encode_glv_fold_scale_add(
                &ctx,
                &mut enc,
                Curve::G2,
                &buf2,
                &d2,
                n as u32,
                0,
                n as u32,
                0,
            );
            ctx.queue.submit([enc.finish()]);
            let bytes = ctx
                .read_buffer(&buf2, 0, n as u64 * G2_PROJ_WORDS as u64 * 4)
                .await;
            let words: &[u32] = bytemuck::cast_slice(&bytes);
            for (i, chunk) in words.chunks_exact(G2_PROJ_WORDS).enumerate() {
                let got = g2_proj_from_words(chunk);
                let want = pts2[i] * k + pts2[n + i];
                assert_eq!(
                    got, want,
                    "G2 glv scale_add mismatch at {i} (trial {trial})"
                );
            }

            // --- G2 add_scaled_base ---
            let bases2: Vec<ark_bn254::G2Affine> = (0..n)
                .map(|_| ark_bn254::G2Projective::rand(&mut rng).into_affine())
                .collect();
            let vs2: Vec<ark_bn254::G2Projective> = (0..n)
                .map(|_| ark_bn254::G2Projective::rand(&mut rng))
                .collect();
            let vbuf2 = ctx.buffer_from(
                "glv-v2-g2",
                bytemuck::cast_slice(&pack_slice(&vs2, g2_proj_to_words)),
                wgpu::BufferUsages::COPY_SRC,
            );
            let bbuf2 = ctx.buffer_from(
                "glv-b-g2",
                bytemuck::cast_slice(&pack_slice(&bases2, g2_affine_to_words)),
                wgpu::BufferUsages::empty(),
            );
            let mut enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            encode_glv_fold_add_scaled_base(
                &ctx,
                &mut enc,
                Curve::G2,
                &vbuf2,
                &bbuf2,
                &d2,
                n as u32,
                0,
                0,
                0,
            );
            ctx.queue.submit([enc.finish()]);
            let bytes = ctx
                .read_buffer(&vbuf2, 0, n as u64 * G2_PROJ_WORDS as u64 * 4)
                .await;
            let words: &[u32] = bytemuck::cast_slice(&bytes);
            for (i, chunk) in words.chunks_exact(G2_PROJ_WORDS).enumerate() {
                let got = g2_proj_from_words(chunk);
                let want = vs2[i] + bases2[i] * k;
                assert_eq!(got, want, "G2 glv add_scaled_base mismatch at {i}");
            }
        }
    });
}
