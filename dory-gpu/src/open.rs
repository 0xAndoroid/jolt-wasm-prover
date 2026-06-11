//! Virtual-matrix Dory opening and batched tier-2 commitment for Jolt.
//!
//! Jolt streams its witness matrix from the execution trace, so the
//! coefficient matrix never exists in memory. The caller computes the
//! matrix-dependent vectors on the CPU (`v_vec = L^T M`, the evaluation
//! vectors, the row commitments) and this module runs every later phase on
//! the GPU: the VMV-message MSMs, the v2 construction, and all reduce
//! rounds (multipairings, MSMs, folds).
//!
//! Mode-generic like `dory_pcs::create_evaluation_proof`: `Transparent`
//! proofs are byte-identical to dory-pcs; `ZK` proofs sample OsRng blinds
//! CPU-side with the exact dory-pcs formulas and verify with its stock
//! verifier. Heavy arithmetic is identical in both modes — masks are single
//! GT/G1/G2 scale-adds applied to GPU results.

use ark_bn254::{Bn254, Fr, G1Projective};
use ark_ec::pairing::Pairing;
use ark_ec::CurveGroup;
use ark_ff::Field;
use ark_std::Zero;
use dory_pcs::backends::arkworks::{ArkFr, ArkG1, ArkG2, ArkGT, BN254};
use dory_pcs::messages::{
    FirstReduceMessage, ScalarProductMessage, ScalarProductProof, SecondReduceMessage, VMVMessage,
};
use dory_pcs::primitives::arithmetic::{Field as DoryField, Group, PairingCurve};
use dory_pcs::primitives::transcript::Transcript;
use dory_pcs::reduce_and_fold::{generate_sigma1_proof, generate_sigma2_proof};
use dory_pcs::{DoryProof, Mode};

use crate::coop::{encode_miller_computed_coop, encode_miller_prepared_coop};
use crate::fold::{
    encode_fixed_base_mul, encode_fold_add_scaled_base, encode_fold_scalars, encode_fold_scale_add,
};
use crate::msm::{encode_msm, encode_normalize, encode_prep_scalars, Curve, MsmCall};
use crate::pairing::{encode_product_reduce, final_exponentiation, read_miller_products};
use crate::prove::GpuDory;
use crate::repr::{
    fr_canonical_words, fr_to_words, g1_proj_to_words, pack_slice, G1_PROJ_WORDS, G2_PROJ_WORDS,
};

type Proof = DoryProof<ArkG1, ArkG2, ArkGT>;

/// CPU-computed inputs for an opening over a virtual matrix.
pub struct OpeningInputs<'a> {
    /// `L^T M`, length `2^sigma` (Montgomery form).
    pub v_vec: &'a [Fr],
    /// Unpadded row commitments, length `2^nu`.
    pub rows: &'a [G1Projective],
    /// Left evaluation vector, length `2^nu`.
    pub left: &'a [Fr],
    /// Right evaluation vector, length `2^sigma`.
    pub right: &'a [Fr],
    pub nu: usize,
    pub sigma: usize,
}

/// GPU-resident opening inputs: `v1` and `v_vec` already live on the device
/// (produced by the unfused commit path and the GPU VMV).
pub struct OpeningBuffers<'a> {
    /// Row commitments, `2^sigma` projective points, identity-padded past
    /// `2^nu`. Usage must include COPY_SRC (final read) and COPY_DST.
    pub v1: &'a wgpu::Buffer,
    /// `L^T M`, `2^sigma` Fr values (Montgomery form).
    pub v_vec: &'a wgpu::Buffer,
    /// Left evaluation vector, length `2^nu`.
    pub left: &'a [Fr],
    /// Right evaluation vector, length `2^sigma`.
    pub right: &'a [Fr],
    pub nu: usize,
    pub sigma: usize,
}

/// ZK blind accumulators, folded with the dory-pcs formulas. All zero (and
/// all masks identity) in Transparent mode.
struct Blinds {
    r_c: Fr,
    r_d1: Fr,
    r_d2: Fr,
    r_e1: Fr,
    r_e2: Fr,
}

fn sample<Mo: Mode>() -> Fr {
    Mo::sample::<ArkFr>().0
}

impl GpuDory {
    /// Batched tier-2 commitment: for each group `g`, computes
    /// `prod_j e(rows[g][j], Gamma2[j])` (one multipairing per group) in a
    /// single GPU pass. Groups are padded to a common power-of-two stride
    /// with the identity, which the pairing kernels mask to `f = 1`.
    pub async fn tier2_batch(&self, groups: &[Vec<G1Projective>]) -> Vec<ArkGT> {
        assert!(!groups.is_empty());
        let stride = groups
            .iter()
            .map(|g| g.len())
            .max()
            .unwrap()
            .next_power_of_two() as u32;
        assert!(
            stride as usize <= self.setup().g2_vec.len(),
            "tier-2 rows exceed setup generators"
        );
        let n_groups = groups.len() as u32;
        let total = stride * n_groups;

        let mut words = Vec::with_capacity(total as usize * G1_PROJ_WORDS);
        let zero = g1_proj_to_words(&G1Projective::zero());
        for g in groups {
            for p in g {
                words.extend_from_slice(&g1_proj_to_words(p));
            }
            for _ in g.len()..stride as usize {
                words.extend_from_slice(&zero);
            }
        }
        let rows_proj = self.ctx.buffer_from(
            "tier2-rows",
            bytemuck::cast_slice(&words),
            wgpu::BufferUsages::empty(),
        );

        let mut enc = self.encoder();
        let rows_affine = encode_normalize(&self.ctx, &mut enc, Curve::G1, &rows_proj, total);
        self.ctx.queue.submit([enc.finish()]);
        self.ctx.poll_wait();

        let mut enc = self.encoder();
        let state = encode_miller_prepared_coop(
            &self.ctx,
            &mut enc,
            &rows_affine,
            self.prepared_g2(),
            stride,
            total,
        );
        encode_product_reduce(&self.ctx, &mut enc, &state, stride, n_groups);
        self.ctx.queue.submit([enc.finish()]);
        self.ctx.poll_wait();

        let fs = read_miller_products(&self.ctx, &state, stride, n_groups).await;
        crate::par::final_exps(fs)
    }

    /// Mirror of `dory_pcs::create_evaluation_proof` over a virtual matrix:
    /// the VMV product and evaluation vectors come from the caller, all
    /// later phases run on the GPU. Supports `nu <= sigma` (rectangular
    /// matrices are padded to `2^sigma` exactly like dory-pcs).
    ///
    /// `eval` is only invoked in ZK mode (for the `e2`/`y_com` messages).
    /// Returns the proof and, in ZK mode, the `y` blinding scalar.
    pub async fn prove_virtual<Mo: Mode, T: Transcript<Curve = BN254>>(
        &self,
        inputs: &OpeningInputs<'_>,
        eval: impl FnOnce() -> Fr,
        transcript: &mut T,
    ) -> (Proof, Option<ArkFr>) {
        let (nu, sigma) = (inputs.nu, inputs.sigma);
        assert_eq!(inputs.v_vec.len(), 1 << sigma, "v_vec dimension mismatch");
        assert_eq!(inputs.rows.len(), 1 << nu, "row count mismatch");
        let n_full = 1u32 << sigma;

        // Padded upload: v1 = rows padded with the identity.
        let mut v1_words = Vec::with_capacity(n_full as usize * G1_PROJ_WORDS);
        for p in inputs.rows {
            v1_words.extend_from_slice(&g1_proj_to_words(p));
        }
        let zero = g1_proj_to_words(&G1Projective::zero());
        for _ in inputs.rows.len()..n_full as usize {
            v1_words.extend_from_slice(&zero);
        }
        let v1 = self.ctx.buffer_from(
            "v1",
            bytemuck::cast_slice(&v1_words),
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        );
        let v_buf = self.ctx.buffer_from(
            "v-vec",
            bytemuck::cast_slice(&pack_slice(inputs.v_vec, fr_to_words)),
            wgpu::BufferUsages::empty(),
        );

        let buffers = OpeningBuffers {
            v1: &v1,
            v_vec: &v_buf,
            left: inputs.left,
            right: inputs.right,
            nu,
            sigma,
        };
        self.prove_from_buffers::<Mo, T>(&buffers, eval, transcript)
            .await
    }

    /// Opening over GPU-resident `v1` and `v_vec` buffers (see
    /// [`OpeningBuffers`]). This is the full protocol driver: VMV-message
    /// MSMs, v2 construction, all reduce rounds, and the final message —
    /// transcript, blinds, masks and final exponentiations on the CPU.
    pub async fn prove_from_buffers<Mo: Mode, T: Transcript<Curve = BN254>>(
        &self,
        inputs: &OpeningBuffers<'_>,
        eval: impl FnOnce() -> Fr,
        transcript: &mut T,
    ) -> (Proof, Option<ArkFr>) {
        let (nu, sigma) = (inputs.nu, inputs.sigma);
        assert!(nu <= sigma, "Dory requires nu <= sigma");
        assert_eq!(inputs.left.len(), 1 << nu, "left vector dimension mismatch");
        assert_eq!(
            inputs.right.len(),
            1 << sigma,
            "right vector dimension mismatch"
        );
        let n_full = 1u32 << sigma;
        let num_rounds = sigma;

        let ctx = &self.ctx;
        let setup = self.setup();
        let h1 = setup.h1.0;
        let h2 = setup.h2.0;
        let ht = &setup.ht;
        let g2_fin = setup.g2_vec[0].0.into_affine();

        let v1 = inputs.v1;
        let v_buf = inputs.v_vec;

        // s2 = left padded with zeros (zero scalars contribute nothing to
        // MSMs, so the padded E1 MSM equals dory-pcs's unpadded one).
        // s1 = right.
        let mut left_padded = inputs.left.to_vec();
        left_padded.resize(n_full as usize, Fr::zero());

        let s1 = ctx.buffer_from(
            "s1",
            bytemuck::cast_slice(&pack_slice(inputs.right, fr_to_words)),
            wgpu::BufferUsages::COPY_SRC,
        );
        let s2 = ctx.buffer_from(
            "s2",
            bytemuck::cast_slice(&pack_slice(&left_padded, fr_to_words)),
            wgpu::BufferUsages::COPY_SRC,
        );

        let v2 = ctx.empty_buffer(
            "v2",
            n_full as u64 * G2_PROJ_WORDS as u64 * 4,
            wgpu::BufferUsages::COPY_SRC,
        );
        let vmv_points = ctx.empty_buffer(
            "vmv-points",
            3 * G1_PROJ_WORDS as u64 * 4,
            wgpu::BufferUsages::COPY_SRC,
        );

        // Phase 1: C/D2 numerators, E1, and v2 = v_vec * g2_fin.
        let mut enc = self.encoder();
        let prep_vvec = encode_prep_scalars(ctx, &mut enc, Curve::G1, v_buf, n_full);
        let prep_left = encode_prep_scalars(ctx, &mut enc, Curve::G1, &s2, n_full);
        let v1_affine0 = encode_normalize(ctx, &mut enc, Curve::G1, v1, n_full);
        for (slot, bases, scalars) in [
            (0u32, &v1_affine0, &prep_vvec),
            (1, self.g1_affine(), &prep_vvec),
            (2, &v1_affine0, &prep_left),
        ] {
            encode_msm(
                ctx,
                &mut enc,
                &MsmCall {
                    curve: Curve::G1,
                    bases,
                    scalars,
                    rows: 1,
                    n: n_full,
                    base_offset: 0,
                    scalar_offset: 0,
                    scalar_stride: 0,
                    results: &vmv_points,
                    out_offset: slot,
                },
            );
        }
        encode_fixed_base_mul(
            ctx,
            &mut enc,
            Curve::G2,
            &v2,
            self.fb_table(),
            &prep_vvec,
            n_full,
            0,
            0,
        );
        ctx.queue.submit([enc.finish()]);

        let t_vec_v = self.read_g1(&vmv_points, 0).await;
        let d2_num = self.read_g1(&vmv_points, 1).await;
        let e1_vmv = self.read_g1(&vmv_points, 2).await;

        // VMV blinds, sampled in dory-pcs order.
        let (r_c0, r_d20, r_e10, r_e20) = (
            sample::<Mo>(),
            sample::<Mo>(),
            sample::<Mo>(),
            sample::<Mo>(),
        );

        let vmv_c = Mo::mask(
            ArkGT(Bn254::pairing(t_vec_v.into_affine(), g2_fin).0),
            ht,
            &ArkFr(r_c0),
        );
        let vmv_d2 = Mo::mask(
            ArkGT(Bn254::pairing(d2_num.into_affine(), g2_fin).0),
            ht,
            &ArkFr(r_d20),
        );
        let vmv_e1 = Mo::mask(ArkG1(e1_vmv), &setup.h1, &ArkFr(r_e10));

        transcript.append_serde(b"vmv_c", &vmv_c);
        transcript.append_serde(b"vmv_d2", &vmv_d2);
        transcript.append_serde(b"vmv_e1", &vmv_e1);

        let vmv_message = VMVMessage {
            c: vmv_c,
            d2: vmv_d2,
            e1: vmv_e1,
        };

        let (zk_e2, zk_y_com, zk_sigma1, zk_sigma2, zk_r_y) = if Mo::BLINDING {
            let y = ArkFr(eval());
            let r_y: ArkFr = Mo::sample();
            let e2 = Mo::mask(setup.g2_vec[0].scale(&y), &setup.h2, &ArkFr(r_e20));
            let y_com = setup.g1_vec[0].scale(&y) + setup.h1.scale(&r_y);
            transcript.append_serde(b"vmv_e2", &e2);
            transcript.append_serde(b"vmv_y_com", &y_com);
            let s1_proof =
                generate_sigma1_proof::<BN254, T>(&y, &ArkFr(r_e20), &r_y, setup, transcript);
            let s2_proof =
                generate_sigma2_proof::<BN254, T>(&ArkFr(r_e10), &ArkFr(-r_d20), setup, transcript);
            (
                Some(e2),
                Some(y_com),
                Some(s1_proof),
                Some(s2_proof),
                Some(r_y),
            )
        } else {
            (None, None, None, None, None)
        };

        let mut blinds = Blinds {
            r_c: r_c0,
            r_d1: Fr::zero(),
            r_d2: r_d20,
            r_e1: r_e10,
            r_e2: r_e20,
        };

        let mut first_messages = Vec::with_capacity(num_rounds);
        let mut second_messages = Vec::with_capacity(num_rounds);

        for round in 0..num_rounds {
            let n = 1u32 << (num_rounds - round);
            let n2 = n / 2;

            // --- First message ---
            let mut enc = self.encoder();
            let v1_aff = encode_normalize(ctx, &mut enc, Curve::G1, v1, n);
            let d1_state =
                encode_miller_prepared_coop(ctx, &mut enc, &v1_aff, self.prepared_g2(), n2, n);
            encode_product_reduce(ctx, &mut enc, &d1_state, n2, 2);

            let prep_s2 = encode_prep_scalars(ctx, &mut enc, Curve::G1, &s2, n);
            let prep_s1 = encode_prep_scalars(ctx, &mut enc, Curve::G2, &s1, n);
            let beta_points = ctx.empty_buffer(
                "beta-points",
                G1_PROJ_WORDS as u64 * 4,
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
                    bases: self.g1_affine(),
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
                    bases: self.g2_affine(),
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

            // D2: round 0 uses the v2-scalars MSM shortcut; later rounds pair
            // Gamma1' against the folded v2.
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
                            bases: self.g1_affine(),
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
                enc.copy_buffer_to_buffer(self.g1_affine(), 0, &g1_dup, 0, half_bytes);
                enc.copy_buffer_to_buffer(self.g1_affine(), 0, &g1_dup, half_bytes, half_bytes);
                let state = encode_miller_computed_coop(ctx, &mut enc, &g1_dup, &v2_aff, n);
                encode_product_reduce(ctx, &mut enc, &state, n2, 2);
                D2Path::Miller(state)
            };
            ctx.queue.submit([enc.finish()]);

            let round_d1 = [sample::<Mo>(), sample::<Mo>()];
            let round_d2 = [sample::<Mo>(), sample::<Mo>()];

            let d1_fs = read_miller_products(ctx, &d1_state, n2, 2).await;
            let (d1l, d1r) = crate::par::join2(
                || final_exponentiation(d1_fs[0]),
                || final_exponentiation(d1_fs[1]),
            );
            let d1_left = Mo::mask(ArkGT(d1l), ht, &ArkFr(round_d1[0]));
            let d1_right = Mo::mask(ArkGT(d1r), ht, &ArkFr(round_d1[1]));
            let (d2l, d2r) = match &d2_path {
                D2Path::Msm(points) => {
                    let l = self.read_g1(points, 0).await;
                    let r = self.read_g1(points, 1).await;
                    (
                        Bn254::pairing(l.into_affine(), g2_fin).0,
                        Bn254::pairing(r.into_affine(), g2_fin).0,
                    )
                }
                D2Path::Miller(state) => {
                    let fs = read_miller_products(ctx, state, n2, 2).await;
                    crate::par::join2(
                        || final_exponentiation(fs[0]),
                        || final_exponentiation(fs[1]),
                    )
                }
            };
            let d2_left = Mo::mask(ArkGT(d2l), ht, &ArkFr(round_d2[0]));
            let d2_right = Mo::mask(ArkGT(d2r), ht, &ArkFr(round_d2[1]));
            let e1_beta = ArkG1(self.read_g1(&beta_points, 0).await);
            let e2_beta = ArkG2(self.read_g2(&e2_beta_buf, 0).await);

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
            let beta_inv = beta.0.inverse().expect("beta nonzero");

            blinds.r_c = blinds.r_c + blinds.r_d2 * beta.0 + blinds.r_d1 * beta_inv;

            // --- Apply first challenge + second message ---
            let mut enc = self.encoder();
            encode_fold_add_scaled_base(
                ctx,
                &mut enc,
                Curve::G1,
                v1,
                self.g1_affine(),
                &fr_canonical_words(&beta.0),
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
                self.g2_affine(),
                &fr_canonical_words(&beta_inv),
                n,
                0,
                0,
                0,
            );

            let v1_aff2 = encode_normalize(ctx, &mut enc, Curve::G1, v1, n);
            let v2_aff2 = encode_normalize(ctx, &mut enc, Curve::G2, &v2, n);
            // C+: (v1L, v2R); C-: (v1R, v2L). P batch = v1_aff2 as-is,
            // Q batch = [v2R | v2L].
            let q_swap =
                ctx.empty_buffer("q-swap", n as u64 * 32 * 4, wgpu::BufferUsages::COPY_DST);
            let half_q = n2 as u64 * 32 * 4;
            enc.copy_buffer_to_buffer(&v2_aff2, half_q, &q_swap, 0, half_q);
            enc.copy_buffer_to_buffer(&v2_aff2, 0, &q_swap, half_q, half_q);
            let c_state = encode_miller_computed_coop(ctx, &mut enc, &v1_aff2, &q_swap, n);
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

            let round_c = [sample::<Mo>(), sample::<Mo>()];
            let round_e1 = [sample::<Mo>(), sample::<Mo>()];
            let round_e2 = [sample::<Mo>(), sample::<Mo>()];

            let c_fs = read_miller_products(ctx, &c_state, n2, 2).await;
            let (cp, cm) = crate::par::join2(
                || final_exponentiation(c_fs[0]),
                || final_exponentiation(c_fs[1]),
            );
            let c_plus = Mo::mask(ArkGT(cp), ht, &ArkFr(round_c[0]));
            let c_minus = Mo::mask(ArkGT(cm), ht, &ArkFr(round_c[1]));
            let e1_plus = Mo::mask(
                ArkG1(self.read_g1(&e_points_g1, 0).await),
                &setup.h1,
                &ArkFr(round_e1[0]),
            );
            let e1_minus = Mo::mask(
                ArkG1(self.read_g1(&e_points_g1, 1).await),
                &setup.h1,
                &ArkFr(round_e1[1]),
            );
            let e2_plus = Mo::mask(
                ArkG2(self.read_g2(&e_points_g2, 0).await),
                &setup.h2,
                &ArkFr(round_e2[0]),
            );
            let e2_minus = Mo::mask(
                ArkG2(self.read_g2(&e_points_g2, 1).await),
                &setup.h2,
                &ArkFr(round_e2[1]),
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
            let alpha_inv = alpha.0.inverse().expect("alpha nonzero");

            blinds.r_c = blinds.r_c + round_c[0] * alpha.0 + round_c[1] * alpha_inv;
            blinds.r_d1 = round_d1[0] * alpha.0 + round_d1[1];
            blinds.r_d2 = round_d2[0] * alpha_inv + round_d2[1];
            blinds.r_e1 = blinds.r_e1 + round_e1[0] * alpha.0 + round_e1[1] * alpha_inv;
            blinds.r_e2 = blinds.r_e2 + round_e2[0] * alpha.0 + round_e2[1] * alpha_inv;

            // --- Apply second challenge (fold to n/2) ---
            let mut enc = self.encoder();
            encode_fold_scale_add(
                ctx,
                &mut enc,
                Curve::G1,
                v1,
                &fr_canonical_words(&alpha.0),
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
                &fr_canonical_words(&alpha_inv),
                n2,
                0,
                n2,
                0,
            );
            encode_fold_scalars(ctx, &mut enc, &s1, &fr_to_words(&alpha.0), n2, 0, n2, 0);
            encode_fold_scalars(ctx, &mut enc, &s2, &fr_to_words(&alpha_inv), n2, 0, n2, 0);
            ctx.queue.submit([enc.finish()]);
        }

        // --- Final scalar product message ---
        let gamma: ArkFr = transcript.challenge_scalar(b"gamma");
        let gamma_inv = gamma.0.inverse().expect("gamma nonzero");

        let v1_0 = self.read_g1(v1, 0).await;
        let v2_0 = self.read_g2(&v2, 0).await;
        let s1_0 = self.read_fr(&s1, 0).await;
        let s2_0 = self.read_fr(&s2, 0).await;

        let scalar_product_proof = if Mo::BLINDING {
            Some(scalar_product_proof::<T>(
                ArkG1(v1_0),
                ArkG2(v2_0),
                &blinds,
                setup,
                transcript,
            ))
        } else {
            None
        };

        let (r_final1, r_final2) = (sample::<Mo>(), sample::<Mo>());
        let e1 = ArkG1(v1_0 + h1 * (gamma.0 * s1_0 + r_final1));
        let e2 = ArkG2(v2_0 + h2 * (gamma_inv * s2_0 + r_final2));

        transcript.append_serde(b"final_e1", &e1);
        transcript.append_serde(b"final_e2", &e2);
        let _d: ArkFr = transcript.challenge_scalar(b"d");

        let proof = DoryProof {
            vmv_message,
            first_messages,
            second_messages,
            final_message: ScalarProductMessage { e1, e2 },
            nu,
            sigma,
            e2: zk_e2,
            y_com: zk_y_com,
            sigma1_proof: zk_sigma1,
            sigma2_proof: zk_sigma2,
            scalar_product_proof,
        };
        (proof, zk_r_y)
    }
}

/// ZK scalar product proof, mirroring `DoryProverState::scalar_product_proof`.
fn scalar_product_proof<T: Transcript<Curve = BN254>>(
    v1: ArkG1,
    v2: ArkG2,
    blinds: &Blinds,
    setup: &dory_pcs::ProverSetup<BN254>,
    transcript: &mut T,
) -> ScalarProductProof<ArkG1, ArkG2, ArkFr, ArkGT> {
    let (g1, g2) = (setup.g1_vec[0], setup.g2_vec[0]);
    let ht = &setup.ht;
    let r = || ArkFr::random();
    let (sd1, sd2) = (r(), r());
    let (d1, d2) = (sd1 * g1, g2.scale(&sd2));
    let (rp1, rp2, rq, rr) = (r(), r(), r(), r());
    let p1 = <BN254 as PairingCurve>::pair(&d1, &g2) + ht.scale(&rp1);
    let p2 = <BN254 as PairingCurve>::pair(&g1, &d2) + ht.scale(&rp2);
    let q = <BN254 as PairingCurve>::pair(&d1, &v2)
        + <BN254 as PairingCurve>::pair(&v1, &d2)
        + ht.scale(&rq);
    let rr_val = <BN254 as PairingCurve>::pair(&d1, &d2) + ht.scale(&rr);
    for (label, val) in [
        (b"sigma_p1" as &[u8], &p1),
        (b"sigma_p2", &p2),
        (b"sigma_q", &q),
        (b"sigma_r", &rr_val),
    ] {
        transcript.append_serde(label, val);
    }
    let c = transcript.challenge_scalar(b"sigma_c");
    ScalarProductProof {
        p1,
        p2,
        q,
        r: rr_val,
        e1: d1 + c * v1,
        e2: d2 + v2.scale(&c),
        r1: rp1 + c * ArkFr(blinds.r_d1),
        r2: rp2 + c * ArkFr(blinds.r_d2),
        r3: rr + c * rq + c * c * ArkFr(blinds.r_c),
    }
}
