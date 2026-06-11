//! Stateful orchestration for Jolt's unfused full-GPU Dory: registers each
//! committed polynomial's raw data on the device at commit time (tier-1 +
//! tier-2 run as batched GPU passes), then reuses the resident buffers at
//! opening time for the joint RLC matrix, the GPU vector-matrix product,
//! the GPU row-commitment RLC, and the GPU opening rounds.

use std::collections::HashMap;

use ark_bn254::{Fr, G1Projective};
use ark_std::Zero;
use dory_pcs::backends::arkworks::{ArkFr, ArkGT, BN254};
use dory_pcs::primitives::transcript::Transcript;
use dory_pcs::{DoryProof, Mode};
use rayon::prelude::*;

use crate::commit::{encode_onehot_rows, encode_rlc_combine, RLC_META_STRIDE};
use crate::fold::{encode_fold_scale_add, encode_vmv};
use crate::msm::{encode_msm, encode_normalize, encode_prep_scalars, Curve, MsmCall};
use crate::open::OpeningBuffers;
use crate::pairing::{
    encode_miller_prepared, encode_product_reduce, final_exponentiation, read_miller_products,
};
use crate::prove::GpuDory;
use crate::repr::{
    fr_canonical_words, fr_from_words, fr_to_words, g1_proj_from_words, g1_proj_to_words,
    pack_slice, G1_AFFINE_WORDS, G1_PROJ_WORDS,
};

type Proof =
    DoryProof<dory_pcs::backends::arkworks::ArkG1, dory_pcs::backends::arkworks::ArkG2, ArkGT>;

/// Raw polynomial data for an unfused commit.
pub enum PolyUpload {
    /// Row-major Fr matrix (Montgomery), `rows * cols` entries.
    Dense { matrix: Vec<Fr>, rows: u32 },
    /// Bucket index per (chunk, column): `indices[c * cols + col]`,
    /// `ONEHOT_NONE` for none. Output rows = `k * rows_per_k`.
    OneHot {
        indices: Vec<u32>,
        k: u32,
        rows_per_k: u32,
    },
}

impl PolyUpload {
    fn num_rows(&self) -> u32 {
        match self {
            PolyUpload::Dense { rows, .. } => *rows,
            PolyUpload::OneHot { k, rows_per_k, .. } => k * rows_per_k,
        }
    }
}

pub struct CommitOut {
    pub id: u64,
    pub rows: Vec<G1Projective>,
    pub tier2: ArkGT,
}

enum PolyKind {
    Dense { rows: u32 },
    OneHot { rows_per_k: u32 },
}

struct PolyGpu {
    kind: PolyKind,
    data: wgpu::Buffer,
    rows_proj: wgpu::Buffer,
    num_rows: u32,
}

struct Joint {
    v1: wgpu::Buffer,
    recipe: Vec<(u64, Fr)>,
    num_rows: u32,
}

pub struct JoltGpuDory {
    pub gpu: GpuDory,
    cols: u32,
    polys: HashMap<u64, PolyGpu>,
    next_id: u64,
    joint: Option<Joint>,
}

impl JoltGpuDory {
    pub fn new(gpu: GpuDory, cols: u32) -> Self {
        assert!(
            cols as usize <= gpu.setup().g1_vec.len(),
            "setup smaller than matrix width"
        );
        Self {
            gpu,
            cols,
            polys: HashMap::new(),
            next_id: 0,
            joint: None,
        }
    }

    /// Unfused commit for a batch of polynomials: uploads raw data, runs all
    /// tier-1 row commitments (dense MSMs / one-hot gather-adds) and one
    /// batched tier-2 multipairing on the GPU. Raw data and row commitments
    /// stay resident for the opening.
    pub async fn commit_batch(&mut self, uploads: Vec<PolyUpload>) -> Vec<CommitOut> {
        assert!(!uploads.is_empty());
        let ctx = std::sync::Arc::clone(&self.gpu.ctx);
        let cols = self.cols;

        // Tier 1: per-poly row commitments, one submission.
        let mut enc = self.gpu.encoder();
        let mut staged: Vec<(PolyKind, wgpu::Buffer, wgpu::Buffer, u32)> = Vec::new();
        for upload in &uploads {
            let num_rows = upload.num_rows();
            let rows_proj = ctx.empty_buffer(
                "tier1-rows",
                num_rows as u64 * G1_PROJ_WORDS as u64 * 4,
                wgpu::BufferUsages::COPY_SRC,
            );
            match upload {
                PolyUpload::Dense { matrix, rows } => {
                    assert_eq!(matrix.len() as u32, rows * cols, "dense matrix shape");
                    let data = ctx.buffer_from(
                        "dense-matrix",
                        bytemuck::cast_slice(&pack_slice(matrix, fr_to_words)),
                        wgpu::BufferUsages::COPY_SRC,
                    );
                    let prepared =
                        encode_prep_scalars(&ctx, &mut enc, Curve::G1, &data, rows * cols);
                    encode_msm(
                        &ctx,
                        &mut enc,
                        &MsmCall {
                            curve: Curve::G1,
                            bases: self.gpu.g1_affine(),
                            scalars: &prepared,
                            rows: *rows,
                            n: cols,
                            base_offset: 0,
                            scalar_offset: 0,
                            scalar_stride: cols,
                            results: &rows_proj,
                            out_offset: 0,
                        },
                    );
                    staged.push((PolyKind::Dense { rows: *rows }, data, rows_proj, num_rows));
                }
                PolyUpload::OneHot {
                    indices,
                    k,
                    rows_per_k,
                } => {
                    assert_eq!(
                        indices.len() as u32,
                        rows_per_k * cols,
                        "one-hot index shape"
                    );
                    let data = ctx.buffer_from(
                        "onehot-indices",
                        bytemuck::cast_slice(indices),
                        wgpu::BufferUsages::COPY_SRC,
                    );
                    encode_onehot_rows(
                        &ctx,
                        &mut enc,
                        &data,
                        self.gpu.g1_affine(),
                        &rows_proj,
                        cols,
                        *rows_per_k,
                        k * rows_per_k,
                        0,
                    );
                    staged.push((
                        PolyKind::OneHot {
                            rows_per_k: *rows_per_k,
                        },
                        data,
                        rows_proj,
                        num_rows,
                    ));
                }
            }
        }
        ctx.queue.submit([enc.finish()]);
        ctx.poll_wait();

        // Tier 2: normalize all rows, concatenate into stride-padded groups
        // (zeroed pad = the (0,0) masked-pair convention), one multipairing.
        let stride = staged
            .iter()
            .map(|(_, _, _, n)| *n)
            .max()
            .unwrap()
            .next_power_of_two();
        assert!(
            stride as usize <= self.gpu.setup().g2_vec.len(),
            "tier-2 rows exceed setup generators"
        );
        let n_groups = staged.len() as u32;
        let total = stride * n_groups;
        let concat = ctx.empty_buffer(
            "tier2-concat",
            total as u64 * G1_AFFINE_WORDS as u64 * 4,
            wgpu::BufferUsages::COPY_DST,
        );
        let mut enc = self.gpu.encoder();
        for (i, (_, _, rows_proj, num_rows)) in staged.iter().enumerate() {
            let affine = encode_normalize(&ctx, &mut enc, Curve::G1, rows_proj, *num_rows);
            enc.copy_buffer_to_buffer(
                &affine,
                0,
                &concat,
                i as u64 * stride as u64 * G1_AFFINE_WORDS as u64 * 4,
                *num_rows as u64 * G1_AFFINE_WORDS as u64 * 4,
            );
        }
        ctx.queue.submit([enc.finish()]);
        ctx.poll_wait();

        let mut enc = self.gpu.encoder();
        let state = encode_miller_prepared(
            &ctx,
            &mut enc,
            &concat,
            self.gpu.prepared_g2(),
            stride,
            total,
        );
        encode_product_reduce(&ctx, &mut enc, &state, stride, n_groups);
        ctx.queue.submit([enc.finish()]);
        ctx.poll_wait();

        let fs = read_miller_products(&ctx, &state, stride, n_groups).await;
        let tier2s: Vec<ArkGT> = fs
            .into_par_iter()
            .map(|f| ArkGT(final_exponentiation(f)))
            .collect();

        // Row readbacks (CPU hint copies) + registry.
        let mut out = Vec::with_capacity(staged.len());
        for ((kind, data, rows_proj, num_rows), tier2) in staged.into_iter().zip(tier2s) {
            let bytes = ctx
                .read_buffer(&rows_proj, 0, num_rows as u64 * G1_PROJ_WORDS as u64 * 4)
                .await;
            let words: &[u32] = bytemuck::cast_slice(&bytes);
            let rows: Vec<G1Projective> = words
                .chunks_exact(G1_PROJ_WORDS)
                .map(g1_proj_from_words)
                .collect();

            let id = self.next_id;
            self.next_id += 1;
            self.polys.insert(
                id,
                PolyGpu {
                    kind,
                    data,
                    rows_proj,
                    num_rows,
                },
            );
            out.push(CommitOut { id, rows, tier2 });
        }
        out
    }

    /// GPU RLC of the registered polynomials' row commitments:
    /// `v1 = sum_p coeff_p * rows_p`, padded to `num_rows`. The folded
    /// buffer is retained as the opening's `v1`; the recipe is retained for
    /// the joint-matrix build. Returns the folded rows (the opening hint).
    pub async fn combine_rows(&mut self, parts: &[(u64, Fr)], num_rows: u32) -> Vec<G1Projective> {
        let ctx = std::sync::Arc::clone(&self.gpu.ctx);
        let point_bytes = G1_PROJ_WORDS as u64 * 4;

        // Scratch layout: [stage | accA | accB], each num_rows points.
        let region = num_rows as u64 * point_bytes;
        let scratch = ctx.empty_buffer(
            "combine-scratch",
            3 * region,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        );
        let identities = ctx.buffer_from(
            "identities",
            bytemuck::cast_slice(&pack_slice(
                &vec![G1Projective::zero(); num_rows as usize],
                g1_proj_to_words,
            )),
            wgpu::BufferUsages::COPY_SRC,
        );

        // acc starts at the identity vector (a zeroed buffer is NOT the
        // identity point).
        let mut enc = self.gpu.encoder();
        enc.copy_buffer_to_buffer(&identities, 0, &scratch, region, region);
        let mut acc_off = num_rows; // accA, in points
        let mut out_off = 2 * num_rows; // accB
        for (id, coeff) in parts {
            let poly = self.polys.get(id).expect("unknown poly id in recipe");
            // stage = rows_p padded with identities.
            enc.copy_buffer_to_buffer(&identities, 0, &scratch, 0, region);
            enc.copy_buffer_to_buffer(
                &poly.rows_proj,
                0,
                &scratch,
                0,
                poly.num_rows.min(num_rows) as u64 * point_bytes,
            );
            encode_fold_scale_add(
                &ctx,
                &mut enc,
                Curve::G1,
                &scratch,
                &fr_canonical_words(coeff),
                num_rows,
                0,
                acc_off,
                out_off,
            );
            std::mem::swap(&mut acc_off, &mut out_off);
        }
        ctx.queue.submit([enc.finish()]);
        ctx.poll_wait();

        // Persist the folded acc as the opening's v1 (padded to 2^sigma by
        // the caller via `open`, which knows sigma).
        let v1 = ctx.empty_buffer(
            "joint-v1",
            region,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        );
        let mut enc = self.gpu.encoder();
        enc.copy_buffer_to_buffer(&scratch, acc_off as u64 * point_bytes, &v1, 0, region);
        ctx.queue.submit([enc.finish()]);

        let bytes = ctx.read_buffer(&v1, 0, region).await;
        let words: &[u32] = bytemuck::cast_slice(&bytes);
        let rows: Vec<G1Projective> = words
            .chunks_exact(G1_PROJ_WORDS)
            .map(g1_proj_from_words)
            .collect();

        self.joint = Some(Joint {
            v1,
            recipe: parts.to_vec(),
            num_rows,
        });
        rows
    }

    /// Full GPU opening from the retained commit state: builds the joint
    /// RLC matrix on the GPU, runs the GPU VMV for `v_vec`, computes the
    /// evaluation as `<v_vec, right>`, and drives the opening rounds. The
    /// retained state is consumed.
    pub async fn open<Mo: Mode, T: Transcript<Curve = BN254>>(
        &mut self,
        left: &[Fr],
        right: &[Fr],
        nu: usize,
        sigma: usize,
        transcript: &mut T,
    ) -> (Proof, Option<ArkFr>) {
        let joint = self
            .joint
            .take()
            .expect("combine_rows must run before open");
        assert_eq!(joint.num_rows, 1 << nu, "joint v1 dimension mismatch");
        let ctx = std::sync::Arc::clone(&self.gpu.ctx);
        let cols = self.cols;
        let num_rows = joint.num_rows;
        let n_full = 1u32 << sigma;

        // Meta + megabuffers for the joint matrix kernel.
        let mut meta: Vec<u32> = Vec::with_capacity(joint.recipe.len() * RLC_META_STRIDE);
        let mut dense_words: u64 = 0;
        let mut onehot_words: u64 = 0;
        let mut rows_per_k = 0u32;
        for (id, coeff) in &joint.recipe {
            let poly = self.polys.get(id).expect("unknown poly id in recipe");
            match poly.kind {
                PolyKind::Dense { rows } => {
                    meta.push(0);
                    meta.push((dense_words / 8) as u32);
                    meta.push(rows);
                    meta.push(0);
                    dense_words += rows as u64 * cols as u64 * 8;
                }
                PolyKind::OneHot { rows_per_k: rpk } => {
                    assert!(
                        rows_per_k == 0 || rows_per_k == rpk,
                        "one-hot polys must share rows_per_k"
                    );
                    rows_per_k = rpk;
                    meta.push(1);
                    meta.push(onehot_words as u32);
                    meta.push(0);
                    meta.push(0);
                    onehot_words += rpk as u64 * cols as u64;
                }
            }
            meta.extend_from_slice(&fr_to_words(coeff));
        }
        // No one-hot polys: any nonzero rows_per_k keeps r/rows_per_k well-defined.
        if rows_per_k == 0 {
            rows_per_k = 1;
        }

        let dense_buf = ctx.empty_buffer(
            "rlc-dense",
            (dense_words * 4).max(4),
            wgpu::BufferUsages::COPY_DST,
        );
        let onehot_buf = ctx.empty_buffer(
            "rlc-onehot",
            (onehot_words * 4).max(4),
            wgpu::BufferUsages::COPY_DST,
        );
        let mut enc = self.gpu.encoder();
        let (mut dense_off, mut onehot_off) = (0u64, 0u64);
        for (id, _) in &joint.recipe {
            let poly = &self.polys[id];
            match poly.kind {
                PolyKind::Dense { rows } => {
                    let bytes = rows as u64 * cols as u64 * 32;
                    enc.copy_buffer_to_buffer(&poly.data, 0, &dense_buf, dense_off, bytes);
                    dense_off += bytes;
                }
                PolyKind::OneHot { rows_per_k: rpk } => {
                    let bytes = rpk as u64 * cols as u64 * 4;
                    enc.copy_buffer_to_buffer(&poly.data, 0, &onehot_buf, onehot_off, bytes);
                    onehot_off += bytes;
                }
            }
        }

        let meta_buf = ctx.buffer_from(
            "rlc-meta",
            bytemuck::cast_slice(&meta),
            wgpu::BufferUsages::empty(),
        );
        let matrix = ctx.empty_buffer(
            "joint-matrix",
            num_rows as u64 * cols as u64 * 32,
            wgpu::BufferUsages::empty(),
        );
        encode_rlc_combine(
            &ctx,
            &mut enc,
            &meta_buf,
            &dense_buf,
            &onehot_buf,
            &matrix,
            num_rows,
            cols,
            rows_per_k,
            joint.recipe.len() as u32,
        );

        // GPU VMV: v_vec = L^T M.
        let left_buf = ctx.buffer_from(
            "left-vec",
            bytemuck::cast_slice(&pack_slice(left, fr_to_words)),
            wgpu::BufferUsages::empty(),
        );
        let v_vec_buf = ctx.empty_buffer("v-vec", n_full as u64 * 32, wgpu::BufferUsages::COPY_SRC);
        encode_vmv(
            &ctx, &mut enc, &matrix, &left_buf, &v_vec_buf, num_rows, cols, 0, 0,
        );

        // v1 padded to 2^sigma with explicit identities.
        let v1 = ctx.buffer_from(
            "v1-padded",
            bytemuck::cast_slice(&pack_slice(
                &vec![G1Projective::zero(); n_full as usize],
                g1_proj_to_words,
            )),
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        );
        enc.copy_buffer_to_buffer(
            &joint.v1,
            0,
            &v1,
            0,
            num_rows as u64 * G1_PROJ_WORDS as u64 * 4,
        );
        ctx.queue.submit([enc.finish()]);
        ctx.poll_wait();

        // The evaluation y = L^T M R = <v_vec, right> (used by ZK mode).
        let v_bytes = ctx.read_buffer(&v_vec_buf, 0, n_full as u64 * 32).await;
        let v_words: &[u32] = bytemuck::cast_slice(&v_bytes);
        let v_vec_cpu: Vec<Fr> = v_words.chunks_exact(8).map(fr_from_words).collect();
        let y: Fr = v_vec_cpu
            .par_iter()
            .zip(right.par_iter())
            .map(|(v, r)| *v * *r)
            .sum();

        let buffers = OpeningBuffers {
            v1: &v1,
            v_vec: &v_vec_buf,
            left,
            right,
            nu,
            sigma,
        };
        let result = self
            .gpu
            .prove_from_buffers::<Mo, T>(&buffers, || y, transcript)
            .await;

        // A proof consumes the registered polynomials.
        self.polys.clear();
        self.next_id = 0;
        result
    }
}
