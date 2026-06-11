//! GPU Dory prover: commit and evaluation proof, mirroring
//! `dory_pcs::create_evaluation_proof` (Transparent mode) message for
//! message so proofs are byte-identical to the CPU implementation and verify
//! with `dory_pcs::verify`.
//!
//! GPU: row-commitment MSMs, tier-2 and per-round multipairing Miller loops,
//! the vector-matrix product, v2 construction, all vector folds, and every
//! protocol MSM. CPU: transcript, challenges, final exponentiations, the two
//! single pairings of the VMV message, and the length-1 final message.

use ark_bn254::{Fr, G1Projective, G2Projective};
use dory_pcs::backends::arkworks::{ArkFr, ArkG1, ArkG2, ArkGT, BN254};
use dory_pcs::primitives::poly::compute_left_right_vectors;
use dory_pcs::primitives::transcript::Transcript;
use dory_pcs::{DoryProof, ProverSetup, Transparent};

use crate::context::GpuContext;
use crate::coop::encode_miller_prepared_auto;
use crate::fold::{build_fixed_base_table_g2, encode_vmv};
use crate::msm::{encode_msm, encode_normalize, encode_prep_scalars, Curve, MsmCall};
use crate::open::OpeningInputs;
use crate::pairing::{
    encode_product_reduce, final_exponentiation, pack_prepared_g2, read_miller_products,
};
use crate::repr::{
    fr_from_words, fr_to_words, g1_affine_to_words, g1_proj_from_words, g2_affine_to_words,
    g2_proj_from_words, pack_slice, G1_PROJ_WORDS, G2_PROJ_WORDS,
};

type Proof = DoryProof<ArkG1, ArkG2, ArkGT>;

pub struct GpuDory {
    pub ctx: std::sync::Arc<GpuContext>,
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
    pub fn new(ctx: std::sync::Arc<GpuContext>, setup: ProverSetup<BN254>) -> Self {
        let g1_proj: Vec<G1Projective> = setup.g1_vec.iter().map(|g| g.0).collect();
        let g2_proj: Vec<G2Projective> = setup.g2_vec.iter().map(|g| g.0).collect();
        let g1_aff = crate::par::normalize_batch(&g1_proj);
        let g2_aff = crate::par::normalize_batch(&g2_proj);

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

    pub(crate) fn g1_affine(&self) -> &wgpu::Buffer {
        &self.g1_affine
    }

    pub(crate) fn g2_affine(&self) -> &wgpu::Buffer {
        &self.g2_affine
    }

    pub(crate) fn prepared_g2(&self) -> &wgpu::Buffer {
        &self.prepared_g2
    }

    pub(crate) fn fb_table(&self) -> &wgpu::Buffer {
        &self.fb_table
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

    pub(crate) fn encoder(&self) -> wgpu::CommandEncoder {
        self.ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default())
    }

    pub(crate) async fn read_g1(&self, buf: &wgpu::Buffer, idx: u32) -> G1Projective {
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

    pub(crate) async fn read_g2(&self, buf: &wgpu::Buffer, idx: u32) -> G2Projective {
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

    pub(crate) async fn read_fr(&self, buf: &wgpu::Buffer, idx: u32) -> Fr {
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

        let mut enc = self.encoder();
        let rows_affine = encode_normalize(&self.ctx, &mut enc, Curve::G1, &rows_proj, rows);
        self.ctx.queue.submit([enc.finish()]);
        self.ctx.poll_wait();

        let mut enc = self.encoder();
        let state = encode_miller_prepared_auto(
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

        let f = read_miller_products(&self.ctx, &state, rows, 1).await[0];
        let tier2 = ArkGT(final_exponentiation(f));

        GpuCommitment {
            tier2,
            rows_proj,
            nu,
            sigma,
        }
    }

    /// Opening proof on the matrix-resident path: computes `v_vec = L^T M`
    /// with the GPU VMV kernel, then delegates to the virtual-matrix prover
    /// (`prove_virtual`) in Transparent mode. Proofs are byte-identical to
    /// `dory_pcs::create_evaluation_proof`.
    pub async fn prove<T: Transcript<Curve = BN254>>(
        &self,
        matrix: &wgpu::Buffer,
        commitment: &GpuCommitment,
        point: &[Fr],
        transcript: &mut T,
    ) -> Proof {
        let (nu, sigma) = (commitment.nu, commitment.sigma);
        assert_eq!(point.len(), nu + sigma, "point dimension mismatch");
        let n_full = 1u32 << sigma;
        let n_rows = 1u32 << nu;

        let point_ark: Vec<ArkFr> = point.iter().map(|p| ArkFr(*p)).collect();
        let (left_vec, right_vec) = compute_left_right_vectors(&point_ark, nu, sigma);
        let left: Vec<Fr> = left_vec.iter().map(|x| x.0).collect();
        let right: Vec<Fr> = right_vec.iter().map(|x| x.0).collect();

        let left_buf = self.ctx.buffer_from(
            "left-vec",
            bytemuck::cast_slice(&pack_slice(&left, fr_to_words)),
            wgpu::BufferUsages::empty(),
        );
        let v_buf =
            self.ctx
                .empty_buffer("v-vec", n_full as u64 * 32, wgpu::BufferUsages::COPY_SRC);
        let mut enc = self.encoder();
        encode_vmv(
            &self.ctx, &mut enc, matrix, &left_buf, &v_buf, n_rows, n_full, 0, 0,
        );
        self.ctx.queue.submit([enc.finish()]);

        let v_bytes = self.ctx.read_buffer(&v_buf, 0, n_full as u64 * 32).await;
        let v_words: &[u32] = bytemuck::cast_slice(&v_bytes);
        let v_vec: Vec<Fr> = v_words.chunks_exact(8).map(fr_from_words).collect();

        let rows_bytes = self
            .ctx
            .read_buffer(
                &commitment.rows_proj,
                0,
                n_rows as u64 * G1_PROJ_WORDS as u64 * 4,
            )
            .await;
        let rows_words: &[u32] = bytemuck::cast_slice(&rows_bytes);
        let rows: Vec<G1Projective> = rows_words
            .chunks_exact(G1_PROJ_WORDS)
            .map(g1_proj_from_words)
            .collect();

        let inputs = OpeningInputs {
            v_vec: &v_vec,
            rows: &rows,
            left: &left,
            right: &right,
            nu,
            sigma,
        };
        let (proof, _) = self
            .prove_virtual::<Transparent, T>(&inputs, || unreachable!("transparent"), transcript)
            .await;
        proof
    }
}
