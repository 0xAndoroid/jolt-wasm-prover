//! Full unfused-commit pipeline tests: synthetic dense + one-hot
//! polynomials in jolt-core's matrix layout, committed and opened entirely
//! on the GPU, checked against dory-pcs CPU references — commitments and
//! Transparent proofs byte-identical, ZK proofs accepted by the stock
//! verifier.

#![cfg(not(target_arch = "wasm32"))]

use ark_bn254::{Fr, G1Projective};
use ark_ff::UniformRand;
use ark_std::Zero;
use dory_gpu::commit::ONEHOT_NONE;
use dory_gpu::jolt::{JoltGpuDory, PolyUpload};
use dory_gpu::prove::GpuDory;
use dory_gpu::GpuContext;
use dory_pcs::backends::arkworks::{
    ArkFr, ArkworksPolynomial, Blake2bTranscript, G1Routines, G2Routines, BN254,
};
use dory_pcs::primitives::poly::Polynomial;
use dory_pcs::{ProverSetup, Transparent, ZK};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

const DOMAIN: &[u8] = b"dory-gpu-jolt-unfused";

struct Spec {
    nu: usize,
    sigma: usize,
    k: u32,
    dense_rows: u32,
}

/// Synthetic polynomials in jolt-core's layout: a few dense rows occupying
/// the top of the matrix, plus one-hot polynomials with row k*rows_per_k+c.
struct Instance {
    uploads: Vec<PolyUpload>,
    /// Full coefficient matrix per polynomial (CPU reference).
    full: Vec<Vec<ArkFr>>,
    coeffs: Vec<Fr>,
    point: Vec<ArkFr>,
    setup: ProverSetup<BN254>,
    spec: Spec,
}

fn gen(spec: Spec) -> Instance {
    let cols = 1usize << spec.sigma;
    let num_rows = 1usize << spec.nu;
    let rows_per_k = num_rows as u32 / spec.k;
    let mut rng = ChaCha20Rng::seed_from_u64(42 + spec.nu as u64);

    let mut uploads = Vec::new();
    let mut full = Vec::new();

    for _ in 0..2 {
        let matrix: Vec<Fr> = (0..spec.dense_rows as usize * cols)
            .map(|_| Fr::rand(&mut rng))
            .collect();
        let mut f = vec![ArkFr(Fr::zero()); num_rows * cols];
        for (i, m) in matrix.iter().enumerate() {
            f[i] = ArkFr(*m);
        }
        full.push(f);
        uploads.push(PolyUpload::Dense {
            matrix,
            rows: spec.dense_rows,
        });
    }

    for poly in 0..3 {
        let indices: Vec<u32> = (0..rows_per_k as usize * cols)
            .map(|i| {
                // Sprinkle in some "None" cells (RamRa can skip cycles).
                if poly == 2 && i % 7 == 0 {
                    ONEHOT_NONE
                } else {
                    rng.gen_range(0..spec.k)
                }
            })
            .collect();
        let mut f = vec![ArkFr(Fr::zero()); num_rows * cols];
        for c in 0..rows_per_k as usize {
            for col in 0..cols {
                let k = indices[c * cols + col];
                if k != ONEHOT_NONE {
                    let r = k as usize * rows_per_k as usize + c;
                    f[r * cols + col] = ArkFr(Fr::from(1u64));
                }
            }
        }
        full.push(f);
        uploads.push(PolyUpload::OneHot {
            indices,
            k: spec.k,
            rows_per_k,
        });
    }

    let coeffs: Vec<Fr> = (0..uploads.len()).map(|_| Fr::rand(&mut rng)).collect();
    let point: Vec<ArkFr> = (0..spec.nu + spec.sigma)
        .map(|_| ArkFr(Fr::rand(&mut rng)))
        .collect();
    let setup = ProverSetup::<BN254>::new(spec.sigma * 2);

    Instance {
        uploads,
        full,
        coeffs,
        point,
        setup,
        spec,
    }
}

fn cpu_rows(
    full: &[ArkFr],
    cols: usize,
    setup: &ProverSetup<BN254>,
    nu: usize,
    sigma: usize,
) -> (
    dory_pcs::backends::arkworks::ArkGT,
    Vec<dory_pcs::backends::arkworks::ArkG1>,
) {
    let trimmed_rows = full.len() / cols;
    let _ = trimmed_rows;
    let poly = ArkworksPolynomial::new(full.to_vec());
    let (tier2, rows, _) = poly
        .commit::<BN254, Transparent, G1Routines>(nu, sigma, setup)
        .expect("cpu commit");
    (tier2, rows)
}

fn run(spec: Spec, zk: bool) {
    pollster::block_on(async {
        let inst = gen(spec);
        let spec = &inst.spec;
        let cols = 1usize << spec.sigma;
        let num_rows = 1usize << spec.nu;

        let ctx = GpuContext::new().await.expect("gpu context");
        let gpu = GpuDory::new(std::sync::Arc::new(ctx), inst.setup.clone());
        let mut jolt = JoltGpuDory::new(gpu, cols as u32);

        // --- Commit: byte-identical tier-2 commitments and rows ---
        let outs = jolt.commit_batch(inst.uploads).await;
        for (i, out) in outs.iter().enumerate() {
            let (cpu_tier2, cpu_rows) =
                cpu_rows(&inst.full[i], cols, &inst.setup, spec.nu, spec.sigma);
            // The dense CPU reference computes rows for the zero-padded tail
            // as well; compare the GPU's actual rows against the prefix.
            for (r, gpu_row) in out.rows.iter().enumerate() {
                assert_eq!(
                    *gpu_row, cpu_rows[r].0,
                    "poly {i}: row commitment {r} mismatch"
                );
            }
            for tail in cpu_rows[out.rows.len()..].iter() {
                assert!(tail.0.is_zero(), "poly {i}: CPU tail row not zero");
            }
            assert_eq!(out.tier2, cpu_tier2, "poly {i}: tier-2 mismatch");
        }

        // --- combine_rows: GPU RLC of row commitments ---
        let parts: Vec<(u64, Fr)> = outs
            .iter()
            .zip(&inst.coeffs)
            .map(|(o, c)| (o.id, *c))
            .collect();
        let combined = jolt.combine_rows(&parts, num_rows as u32).await;
        let mut expected_rows = vec![G1Projective::zero(); num_rows];
        for (out, coeff) in outs.iter().zip(&inst.coeffs) {
            for (r, row) in out.rows.iter().enumerate() {
                expected_rows[r] += *row * *coeff;
            }
        }
        for (r, (g, e)) in combined.iter().zip(&expected_rows).enumerate() {
            assert_eq!(g, e, "combined row {r} mismatch");
        }

        // --- CPU reference: joint polynomial + proof ---
        let mut joint = vec![ArkFr(Fr::zero()); num_rows * cols];
        for (f, coeff) in inst.full.iter().zip(&inst.coeffs) {
            for (j, v) in f.iter().enumerate() {
                joint[j] = ArkFr(joint[j].0 + v.0 * coeff);
            }
        }
        let joint_poly = ArkworksPolynomial::new(joint);
        let expected_rows_ark: Vec<_> = expected_rows
            .iter()
            .map(|r| dory_pcs::backends::arkworks::ArkG1(*r))
            .collect();

        let left: Vec<Fr>;
        let right: Vec<Fr>;
        {
            use dory_pcs::primitives::poly::MultilinearLagrange;
            let (l, r) = joint_poly.compute_evaluation_vectors(&inst.point, spec.nu, spec.sigma);
            left = l.iter().map(|x| x.0).collect();
            right = r.iter().map(|x| x.0).collect();
        }

        if zk {
            let mut t_gpu = Blake2bTranscript::new(DOMAIN);
            let (gpu_proof, y_blind) = jolt
                .open::<ZK, _>(&left, &right, spec.nu, spec.sigma, &mut t_gpu)
                .await;
            assert!(y_blind.is_some());

            let joint_tier2 = {
                let (tier2, _, _) = joint_poly
                    .commit::<BN254, Transparent, G1Routines>(spec.nu, spec.sigma, &inst.setup)
                    .expect("cpu commit");
                tier2
            };
            let evaluation = joint_poly.evaluate(&inst.point);
            let mut t_verify = Blake2bTranscript::new(DOMAIN);
            dory_pcs::verify::<ArkFr, BN254, G1Routines, G2Routines, Blake2bTranscript<BN254>>(
                joint_tier2,
                evaluation,
                &inst.point,
                &gpu_proof,
                inst.setup.to_verifier_setup(),
                &mut t_verify,
            )
            .expect("ZK GPU proof must verify");
        } else {
            let mut t_cpu = Blake2bTranscript::new(DOMAIN);
            let (cpu_proof, _) = dory_pcs::prove::<
                ArkFr,
                BN254,
                G1Routines,
                G2Routines,
                ArkworksPolynomial,
                Blake2bTranscript<BN254>,
                Transparent,
            >(
                &joint_poly,
                &inst.point,
                expected_rows_ark,
                ArkFr(Fr::zero()),
                spec.nu,
                spec.sigma,
                &inst.setup,
                &mut t_cpu,
            )
            .expect("cpu prove");

            let mut t_gpu = Blake2bTranscript::new(DOMAIN);
            let (gpu_proof, _) = jolt
                .open::<Transparent, _>(&left, &right, spec.nu, spec.sigma, &mut t_gpu)
                .await;
            assert_eq!(gpu_proof, cpu_proof, "joint opening proof mismatch");
        }
    });
}

#[test]
fn unfused_square_transparent_byte_identical() {
    run(
        Spec {
            nu: 5,
            sigma: 5,
            k: 4,
            dense_rows: 8,
        },
        false,
    );
}

#[test]
fn unfused_rectangular_transparent_byte_identical() {
    run(
        Spec {
            nu: 4,
            sigma: 5,
            k: 4,
            dense_rows: 8,
        },
        false,
    );
}

#[test]
fn unfused_square_zk_verifies() {
    run(
        Spec {
            nu: 5,
            sigma: 5,
            k: 4,
            dense_rows: 8,
        },
        true,
    );
}
