//! GPU Dory prover: commit and evaluation proof, mirroring
//! `dory_pcs::create_evaluation_proof` (Transparent mode) message for
//! message so proofs are byte-identical to the CPU implementation and verify
//! with `dory_pcs::verify`.
//!
//! GPU: row-commitment MSMs, tier-2 and per-round multipairing Miller loops,
//! the vector-matrix product, v2 construction, all vector folds, and every
//! protocol MSM. CPU: transcript, challenges, final exponentiations, the two
//! single pairings of the VMV message, and the length-1 final message.

use ark_bn254::{Bn254, Fr, G1Projective, G2Projective};
use ark_ec::pairing::Pairing;
use ark_ec::CurveGroup;
use ark_ff::Field;
use dory_pcs::backends::arkworks::{ArkFr, ArkG1, ArkG2, ArkGT, Blake2bTranscript, BN254};
use dory_pcs::messages::{
    FirstReduceMessage, ScalarProductMessage, SecondReduceMessage, VMVMessage,
};
use dory_pcs::primitives::poly::compute_left_right_vectors;
use dory_pcs::primitives::transcript::Transcript;
use dory_pcs::{DoryProof, ProverSetup};

use crate::context::GpuContext;
use crate::fold::{
    build_fixed_base_table_g2, encode_fixed_base_mul, encode_fold_add_scaled_base,
    encode_fold_scalars, encode_fold_scale_add, encode_vmv,
};
use crate::msm::{encode_msm, encode_normalize, encode_prep_scalars, Curve, MsmCall};
use crate::pairing::{
    encode_miller_computed, encode_miller_prepared, encode_product_reduce, final_exponentiation,
    pack_prepared_g2, read_miller_products,
};
use crate::repr::{
    fr_from_words, fr_to_words, g1_affine_to_words, g1_proj_from_words, g2_affine_to_words,
    g2_proj_from_words, pack_slice, G1_PROJ_WORDS, G2_PROJ_WORDS,
};

type Proof = DoryProof<ArkG1, ArkG2, ArkGT>;

pub struct GpuDory {
    pub ctx: std::rc::Rc<GpuContext>,
    setup: ProverSetup<BN254>,
    g1_affine: wgpu::Buffer,
    g2_affine: wgpu::Buffer,
    prepared_g2: wgpu::Buffer,
    fb_table: wgpu::Buffer,
}

pub struct GpuCommitment {
    pub tier2: ArkGT,
    pub rows_proj: wgpu::Buffer,
    pub nu: usize,
    pub sigma: usize,
}

impl GpuDory {
    pub fn new(ctx: std::rc::Rc<GpuContext>, setup: ProverSetup<BN254>) -> Self {
        let g1_proj: Vec<G1Projective> = setup.g1_vec.iter().map(|g| g.0).collect();
        let g2_proj: Vec<G2Projective> = setup.g2_vec.iter().map(|g| g.0).collect();
        let g1_aff = G1Projective::normalize_batch(&g1_proj);
        let g2_aff = G2Projective::normalize_batch(&g2_proj);

        let g1_words = pack_slice(&g1_aff, g1_affine_to_words);
        let g2_words = pack_slice(&g2_aff, g2_affine_to_words);
        let prepared_words = pack_prepared_g2(&g2_aff);
        let fb_words = build_fixed_base_table_g2(&g2_proj[0]);

        let g1_affine = ctx.buffer_from(
            "setup-g1",
            bytemuck::cast_slice(&g1_words),
            wgpu::BufferUsages::COPY_SRC,
        );
        let g2_affine = ctx.buffer_from(
            "setup-g2",
            bytemuck::cast_slice(&g2_words),
            wgpu::BufferUsages::empty(),
        );
        let prepared_g2 = ctx.buffer_from(
            "setup-g2-prepared",
            bytemuck::cast_slice(&prepared_words),
            wgpu::BufferUsages::empty(),
        );
        let fb_table = ctx.buffer_from(
            "setup-fb-table",
            bytemuck::cast_slice(&fb_words),
            wgpu::BufferUsages::empty(),
        );

        Self {
            ctx,
            setup,
            g1_affine,
            g2_affine,
            prepared_g2,
            fb_table,
        }
    }

    pub fn setup(&self) -> &ProverSetup<BN254> {
        &self.setup
    }

    /// Uploads polynomial coefficients (row-major matrix, Montgomery form).
    pub fn upload_matrix(&self, coeffs: &[Fr]) -> wgpu::Buffer {
        let words = pack_slice(coeffs, fr_to_words);
        self.ctx.buffer_from(
            "matrix",
            bytemuck::cast_slice(&words),
            wgpu::BufferUsages::empty(),
        )
    }

    fn encoder(&self) -> wgpu::CommandEncoder {
        self.ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default())
    }

    async fn read_g1(&self, buf: &wgpu::Buffer, idx: u32) -> G1Projective {
        let data = self
            .ctx
            .read_buffer(
                buf,
                idx as u64 * G1_PROJ_WORDS as u64 * 4,
                G1_PROJ_WORDS as u64 * 4,
            )
            .await;
        g1_proj_from_words(bytemuck::cast_slice(&data))
    }

    async fn read_g2(&self, buf: &wgpu::Buffer, idx: u32) -> G2Projective {
        let data = self
            .ctx
            .read_buffer(
                buf,
                idx as u64 * G2_PROJ_WORDS as u64 * 4,
                G2_PROJ_WORDS as u64 * 4,
            )
            .await;
        g2_proj_from_words(bytemuck::cast_slice(&data))
    }

    async fn read_fr(&self, buf: &wgpu::Buffer, idx: u32) -> Fr {
        let data = self.ctx.read_buffer(buf, idx as u64 * 32, 32).await;
        fr_from_words(bytemuck::cast_slice(&data))
    }

    /// Tier-1 row MSMs + tier-2 multipairing. Row commitments stay on the GPU
    /// for the opening proof.
    pub async fn commit(&self, matrix: &wgpu::Buffer, nu: usize, sigma: usize) -> GpuCommitment {
        let rows = 1u32 << nu;
        let cols = 1u32 << sigma;

        let rows_proj = self.ctx.empty_buffer(
            "row-commitments",
            rows as u64 * G1_PROJ_WORDS as u64 * 4,
            wgpu::BufferUsages::COPY_SRC,
        );

        let t0 = std::time::Instant::now();
        let mut enc = self.encoder();
        let prepared = encode_prep_scalars(&self.ctx, &mut enc, Curve::G1, matrix, rows * cols);
        encode_msm(
            &self.ctx,
            &mut enc,
            &MsmCall {
                curve: Curve::G1,
                bases: &self.g1_affine,
                scalars: &prepared,
                rows,
                n: cols,
                base_offset: 0,
                scalar_offset: 0,
                scalar_stride: cols,
                results: &rows_proj,
                out_offset: 0,
            },
        );
        self.ctx.queue.submit([enc.finish()]);
        self.ctx.poll_wait();
        tracing::info!(ms = t0.elapsed().as_millis() as u64, "commit: row msms");

        let mut enc = self.encoder();
        let rows_affine = encode_normalize(&self.ctx, &mut enc, Curve::G1, &rows_proj, rows);
        self.ctx.queue.submit([enc.finish()]);
        self.ctx.poll_wait();
        tracing::info!(ms = t0.elapsed().as_millis() as u64, "commit: normalize");

        let mut enc = self.encoder();
        let state = encode_miller_prepared(
            &self.ctx,
            &mut enc,
            &rows_affine,
            &self.prepared_g2,
            rows,
            rows,
        );
        encode_product_reduce(&self.ctx, &mut enc, &state, rows, 1);
        self.ctx.queue.submit([enc.finish()]);
        self.ctx.poll_wait();
        tracing::info!(ms = t0.elapsed().as_millis() as u64, "commit: tier2 miller");

        let f = read_miller_products(&self.ctx, &state, rows, 1).await[0];
        let tier2 = ArkGT(final_exponentiation(f));
        tracing::info!(ms = t0.elapsed().as_millis() as u64, "commit: done");

        GpuCommitment {
            tier2,
            rows_proj,
            nu,
            sigma,
        }
    }

    /// Mirror of `dory_pcs::create_evaluation_proof` in Transparent mode.
    /// Square matrices only (nu == sigma), which covers every benchmark size
    /// and Jolt's usage for even log sizes.
    pub async fn prove(
        &self,
        matrix: &wgpu::Buffer,
        commitment: &GpuCommitment,
        point: &[Fr],
        transcript: &mut Blake2bTranscript<BN254>,
    ) -> Proof {
        let (nu, sigma) = (commitment.nu, commitment.sigma);
        assert_eq!(nu, sigma, "GPU prover currently supports square matrices");
        assert_eq!(point.len(), nu + sigma, "point dimension mismatch");
        let n_full = 1u32 << sigma;
        let num_rounds = sigma;

        let ctx = &self.ctx;
        let h1 = self.setup.h1.0;
        let h2 = self.setup.h2.0;
        let g2_fin = self.setup.g2_vec[0].0.into_affine();

        // Evaluation vectors (CPU, tiny) and their uploads.
        let point_ark: Vec<ArkFr> = point.iter().map(|p| ArkFr(*p)).collect();
        let (left_vec, right_vec) = compute_left_right_vectors(&point_ark, nu, sigma);
        let left_fr: Vec<Fr> = left_vec.iter().map(|x| x.0).collect();
        let right_fr: Vec<Fr> = right_vec.iter().map(|x| x.0).collect();

        let left_buf = ctx.buffer_from(
            "left-vec",
            bytemuck::cast_slice(&pack_slice(&left_fr, fr_to_words)),
            wgpu::BufferUsages::empty(),
        );

        // s1 = right_vec, s2 = left_vec, folded in place across rounds.
        let s1 = ctx.buffer_from(
            "s1",
            bytemuck::cast_slice(&pack_slice(&right_fr, fr_to_words)),
            wgpu::BufferUsages::COPY_SRC,
        );
        let s2 = ctx.buffer_from(
            "s2",
            bytemuck::cast_slice(&pack_slice(&left_fr, fr_to_words)),
            wgpu::BufferUsages::COPY_SRC,
        );

        // Phase 1: v_vec = L^T M, VMV MSMs, v2 = v_vec * g2_fin.
        let v_vec = ctx.empty_buffer("v-vec", n_full as u64 * 32, wgpu::BufferUsages::empty());
        let vmv_points = ctx.empty_buffer(
            "vmv-points",
            3 * G1_PROJ_WORDS as u64 * 4,
            wgpu::BufferUsages::COPY_SRC,
        );
        let v1 = ctx.empty_buffer(
            "v1",
            n_full as u64 * G1_PROJ_WORDS as u64 * 4,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        );
        let v2 = ctx.empty_buffer(
            "v2",
            n_full as u64 * G2_PROJ_WORDS as u64 * 4,
            wgpu::BufferUsages::COPY_SRC,
        );

        let mut enc = self.encoder();
        // v1 = row commitments (square: no padding).
        enc.copy_buffer_to_buffer(
            &commitment.rows_proj,
            0,
            &v1,
            0,
            n_full as u64 * G1_PROJ_WORDS as u64 * 4,
        );
        encode_vmv(
            ctx, &mut enc, matrix, &left_buf, &v_vec, n_full, n_full, 0, 0,
        );
        let prep_vvec = encode_prep_scalars(ctx, &mut enc, Curve::G1, &v_vec, n_full);
        let prep_left = encode_prep_scalars(ctx, &mut enc, Curve::G1, &left_buf, n_full);
        let v1_affine0 = encode_normalize(ctx, &mut enc, Curve::G1, &v1, n_full);

        // C numerator, D2 numerator, E1 (slots 0..3 of vmv_points).
        encode_msm(
            ctx,
            &mut enc,
            &MsmCall {
                curve: Curve::G1,
                bases: &v1_affine0,
                scalars: &prep_vvec,
                rows: 1,
                n: n_full,
                base_offset: 0,
                scalar_offset: 0,
                scalar_stride: 0,
                results: &vmv_points,
                out_offset: 0,
            },
        );
        encode_msm(
            ctx,
            &mut enc,
            &MsmCall {
                curve: Curve::G1,
                bases: &self.g1_affine,
                scalars: &prep_vvec,
                rows: 1,
                n: n_full,
                base_offset: 0,
                scalar_offset: 0,
                scalar_stride: 0,
                results: &vmv_points,
                out_offset: 1,
            },
        );
        encode_msm(
            ctx,
            &mut enc,
            &MsmCall {
                curve: Curve::G1,
                bases: &v1_affine0,
                scalars: &prep_left,
                rows: 1,
                n: n_full,
                base_offset: 0,
                scalar_offset: 0,
                scalar_stride: 0,
                results: &vmv_points,
                out_offset: 2,
            },
        );
        encode_fixed_base_mul(
            ctx,
            &mut enc,
            Curve::G2,
            &v2,
            &self.fb_table,
            &prep_vvec,
            n_full,
            0,
            0,
        );
        ctx.queue.submit([enc.finish()]);

        tracing::info!("vmv submitted, reading back");
        let t_vec_v = self.read_g1(&vmv_points, 0).await;
        let d2_num = self.read_g1(&vmv_points, 1).await;
        let e1_vmv = self.read_g1(&vmv_points, 2).await;

        let vmv_c = ArkGT(Bn254::pairing(t_vec_v.into_affine(), g2_fin).0);
        let vmv_d2 = ArkGT(Bn254::pairing(d2_num.into_affine(), g2_fin).0);
        let vmv_e1 = ArkG1(e1_vmv);

        transcript.append_serde(b"vmv_c", &vmv_c);
        transcript.append_serde(b"vmv_d2", &vmv_d2);
        transcript.append_serde(b"vmv_e1", &vmv_e1);

        let vmv_message = VMVMessage {
            c: vmv_c,
            d2: vmv_d2,
            e1: vmv_e1,
        };

        let mut first_messages = Vec::with_capacity(num_rounds);
        let mut second_messages = Vec::with_capacity(num_rounds);

        for round in 0..num_rounds {
            let t_round = std::time::Instant::now();
            tracing::info!(round, "round start");
            let n = 1u32 << (num_rounds - round);
            let n2 = n / 2;

            // --- First message ---
            let mut enc = self.encoder();
            let v1_aff = encode_normalize(ctx, &mut enc, Curve::G1, &v1, n);
            let d1_state = encode_miller_prepared(ctx, &mut enc, &v1_aff, &self.prepared_g2, n2, n);
            encode_product_reduce(ctx, &mut enc, &d1_state, n2, 2);

            let prep_s2 = encode_prep_scalars(ctx, &mut enc, Curve::G1, &s2, n);
            let prep_s1 = encode_prep_scalars(ctx, &mut enc, Curve::G2, &s1, n);
            let beta_points = ctx.empty_buffer(
                "beta-points",
                (G1_PROJ_WORDS + G2_PROJ_WORDS) as u64 * 4,
                wgpu::BufferUsages::COPY_SRC,
            );
            let e2_beta_buf = ctx.empty_buffer(
                "e2-beta",
                G2_PROJ_WORDS as u64 * 4,
                wgpu::BufferUsages::COPY_SRC,
            );
            encode_msm(
                ctx,
                &mut enc,
                &MsmCall {
                    curve: Curve::G1,
                    bases: &self.g1_affine,
                    scalars: &prep_s2,
                    rows: 1,
                    n,
                    base_offset: 0,
                    scalar_offset: 0,
                    scalar_stride: 0,
                    results: &beta_points,
                    out_offset: 0,
                },
            );
            encode_msm(
                ctx,
                &mut enc,
                &MsmCall {
                    curve: Curve::G2,
                    bases: &self.g2_affine,
                    scalars: &prep_s1,
                    rows: 1,
                    n,
                    base_offset: 0,
                    scalar_offset: 0,
                    scalar_stride: 0,
                    results: &e2_beta_buf,
                    out_offset: 0,
                },
            );

            // D2: first round uses the v2-scalars MSM shortcut; later rounds
            // pair Gamma1' against the folded v2.
            enum D2Path {
                Msm(wgpu::Buffer),
                Miller(crate::pairing::MillerState),
            }
            let d2_path = if round == 0 {
                let d2_points = ctx.empty_buffer(
                    "d2-points",
                    2 * G1_PROJ_WORDS as u64 * 4,
                    wgpu::BufferUsages::COPY_SRC,
                );
                for (slot, scalar_offset) in [(0u32, 0u32), (1, n2)] {
                    encode_msm(
                        ctx,
                        &mut enc,
                        &MsmCall {
                            curve: Curve::G1,
                            bases: &self.g1_affine,
                            scalars: &prep_vvec,
                            rows: 1,
                            n: n2,
                            base_offset: 0,
                            scalar_offset,
                            scalar_stride: 0,
                            results: &d2_points,
                            out_offset: slot,
                        },
                    );
                }
                D2Path::Msm(d2_points)
            } else {
                let v2_aff = encode_normalize(ctx, &mut enc, Curve::G2, &v2, n);
                // P batch: Gamma1' twice; Q batch: [v2L | v2R] = v2_aff as-is.
                let g1_dup =
                    ctx.empty_buffer("g1-dup", n as u64 * 16 * 4, wgpu::BufferUsages::COPY_DST);
                let half_bytes = n2 as u64 * 16 * 4;
                enc.copy_buffer_to_buffer(&self.g1_affine, 0, &g1_dup, 0, half_bytes);
                enc.copy_buffer_to_buffer(&self.g1_affine, 0, &g1_dup, half_bytes, half_bytes);
                let state = encode_miller_computed(ctx, &mut enc, &g1_dup, &v2_aff, n);
                encode_product_reduce(ctx, &mut enc, &state, n2, 2);
                D2Path::Miller(state)
            };
            ctx.queue.submit([enc.finish()]);

            tracing::info!(
                round,
                elapsed_ms = t_round.elapsed().as_millis() as u64,
                "first msg encoded+submitted, reading"
            );
            let d1_fs = read_miller_products(ctx, &d1_state, n2, 2).await;
            let (d1_left, d1_right) = {
                let (l, r) = rayon::join(
                    || final_exponentiation(d1_fs[0]),
                    || final_exponentiation(d1_fs[1]),
                );
                (ArkGT(l), ArkGT(r))
            };
            let (d2_left, d2_right) = match &d2_path {
                D2Path::Msm(points) => {
                    let l = self.read_g1(points, 0).await;
                    let r = self.read_g1(points, 1).await;
                    (
                        ArkGT(Bn254::pairing(l.into_affine(), g2_fin).0),
                        ArkGT(Bn254::pairing(r.into_affine(), g2_fin).0),
                    )
                }
                D2Path::Miller(state) => {
                    let fs = read_miller_products(ctx, state, n2, 2).await;
                    let (l, r) = rayon::join(
                        || final_exponentiation(fs[0]),
                        || final_exponentiation(fs[1]),
                    );
                    (ArkGT(l), ArkGT(r))
                }
            };
            let e1_beta = ArkG1(self.read_g1(&beta_points, 0).await);
            let e2_beta = ArkG2(self.read_g2(&e2_beta_buf, 0).await);

            tracing::info!(
                round,
                elapsed_ms = t_round.elapsed().as_millis() as u64,
                "first msg readbacks done"
            );
            transcript.append_serde(b"d1_left", &d1_left);
            transcript.append_serde(b"d1_right", &d1_right);
            transcript.append_serde(b"d2_left", &d2_left);
            transcript.append_serde(b"d2_right", &d2_right);
            transcript.append_serde(b"e1_beta", &e1_beta);
            transcript.append_serde(b"e2_beta", &e2_beta);
            first_messages.push(FirstReduceMessage {
                d1_left,
                d1_right,
                d2_left,
                d2_right,
                e1_beta,
                e2_beta,
            });

            let beta: ArkFr = transcript.challenge_scalar(b"beta");
            let beta_inv = ArkFr(beta.0.inverse().expect("beta nonzero"));

            // --- Apply first challenge + second message ---
            let mut enc = self.encoder();
            encode_fold_add_scaled_base(
                ctx,
                &mut enc,
                Curve::G1,
                &v1,
                &self.g1_affine,
                &crate::repr::fr_canonical_words(&beta.0),
                n,
                0,
                0,
                0,
            );
            encode_fold_add_scaled_base(
                ctx,
                &mut enc,
                Curve::G2,
                &v2,
                &self.g2_affine,
                &crate::repr::fr_canonical_words(&beta_inv.0),
                n,
                0,
                0,
                0,
            );

            let v1_aff2 = encode_normalize(ctx, &mut enc, Curve::G1, &v1, n);
            let v2_aff2 = encode_normalize(ctx, &mut enc, Curve::G2, &v2, n);
            // C+: (v1L, v2R); C-: (v1R, v2L). P batch = v1_aff2 as-is,
            // Q batch = [v2R | v2L].
            let q_swap =
                ctx.empty_buffer("q-swap", n as u64 * 32 * 4, wgpu::BufferUsages::COPY_DST);
            let half_q = n2 as u64 * 32 * 4;
            enc.copy_buffer_to_buffer(&v2_aff2, half_q, &q_swap, 0, half_q);
            enc.copy_buffer_to_buffer(&v2_aff2, 0, &q_swap, half_q, half_q);
            let c_state = encode_miller_computed(ctx, &mut enc, &v1_aff2, &q_swap, n);
            encode_product_reduce(ctx, &mut enc, &c_state, n2, 2);

            // E1± over v1 bases, E2± over v2 bases (cross scalar halves).
            let e_points_g1 = ctx.empty_buffer(
                "e1-pm",
                2 * G1_PROJ_WORDS as u64 * 4,
                wgpu::BufferUsages::COPY_SRC,
            );
            let e_points_g2 = ctx.empty_buffer(
                "e2-pm",
                2 * G2_PROJ_WORDS as u64 * 4,
                wgpu::BufferUsages::COPY_SRC,
            );
            for (slot, base_offset, scalar_offset) in [(0u32, 0u32, n2), (1, n2, 0)] {
                encode_msm(
                    ctx,
                    &mut enc,
                    &MsmCall {
                        curve: Curve::G1,
                        bases: &v1_aff2,
                        scalars: &prep_s2,
                        rows: 1,
                        n: n2,
                        base_offset,
                        scalar_offset,
                        scalar_stride: 0,
                        results: &e_points_g1,
                        out_offset: slot,
                    },
                );
            }
            for (slot, base_offset, scalar_offset) in [(0u32, n2, 0u32), (1, 0, n2)] {
                encode_msm(
                    ctx,
                    &mut enc,
                    &MsmCall {
                        curve: Curve::G2,
                        bases: &v2_aff2,
                        scalars: &prep_s1,
                        rows: 1,
                        n: n2,
                        base_offset,
                        scalar_offset,
                        scalar_stride: 0,
                        results: &e_points_g2,
                        out_offset: slot,
                    },
                );
            }
            ctx.queue.submit([enc.finish()]);

            tracing::info!(
                round,
                elapsed_ms = t_round.elapsed().as_millis() as u64,
                "second msg submitted, reading"
            );
            let c_fs = read_miller_products(ctx, &c_state, n2, 2).await;
            let (c_plus, c_minus) = {
                let (p, m) = rayon::join(
                    || final_exponentiation(c_fs[0]),
                    || final_exponentiation(c_fs[1]),
                );
                (ArkGT(p), ArkGT(m))
            };
            let e1_plus = ArkG1(self.read_g1(&e_points_g1, 0).await);
            let e1_minus = ArkG1(self.read_g1(&e_points_g1, 1).await);
            let e2_plus = ArkG2(self.read_g2(&e_points_g2, 0).await);
            let e2_minus = ArkG2(self.read_g2(&e_points_g2, 1).await);

            tracing::info!(
                round,
                elapsed_ms = t_round.elapsed().as_millis() as u64,
                "second msg readbacks done"
            );
            transcript.append_serde(b"c_plus", &c_plus);
            transcript.append_serde(b"c_minus", &c_minus);
            transcript.append_serde(b"e1_plus", &e1_plus);
            transcript.append_serde(b"e1_minus", &e1_minus);
            transcript.append_serde(b"e2_plus", &e2_plus);
            transcript.append_serde(b"e2_minus", &e2_minus);
            second_messages.push(SecondReduceMessage {
                c_plus,
                c_minus,
                e1_plus,
                e1_minus,
                e2_plus,
                e2_minus,
            });

            let alpha: ArkFr = transcript.challenge_scalar(b"alpha");
            let alpha_inv = ArkFr(alpha.0.inverse().expect("alpha nonzero"));

            // --- Apply second challenge (fold to n/2) ---
            let mut enc = self.encoder();
            encode_fold_scale_add(
                ctx,
                &mut enc,
                Curve::G1,
                &v1,
                &crate::repr::fr_canonical_words(&alpha.0),
                n2,
                0,
                n2,
                0,
            );
            encode_fold_scale_add(
                ctx,
                &mut enc,
                Curve::G2,
                &v2,
                &crate::repr::fr_canonical_words(&alpha_inv.0),
                n2,
                0,
                n2,
                0,
            );
            encode_fold_scalars(ctx, &mut enc, &s1, &fr_to_words(&alpha.0), n2, 0, n2, 0);
            encode_fold_scalars(ctx, &mut enc, &s2, &fr_to_words(&alpha_inv.0), n2, 0, n2, 0);
            ctx.queue.submit([enc.finish()]);
            tracing::info!(
                round,
                elapsed_ms = t_round.elapsed().as_millis() as u64,
                "alpha folds submitted"
            );
        }

        // --- Final scalar product message ---
        let gamma: ArkFr = transcript.challenge_scalar(b"gamma");
        let gamma_inv = gamma.0.inverse().expect("gamma nonzero");

        let v1_0 = self.read_g1(&v1, 0).await;
        let v2_0 = self.read_g2(&v2, 0).await;
        let s1_0 = self.read_fr(&s1, 0).await;
        let s2_0 = self.read_fr(&s2, 0).await;

        let e1 = ArkG1(v1_0 + h1 * (gamma.0 * s1_0));
        let e2 = ArkG2(v2_0 + h2 * (gamma_inv * s2_0));

        transcript.append_serde(b"final_e1", &e1);
        transcript.append_serde(b"final_e2", &e2);
        let _d: ArkFr = transcript.challenge_scalar(b"d");

        DoryProof {
            vmv_message,
            first_messages,
            second_messages,
            final_message: ScalarProductMessage { e1, e2 },
            nu,
            sigma,
            e2: None,
            y_com: None,
            sigma1_proof: None,
            sigma2_proof: None,
            scalar_product_proof: None,
        }
    }
}
