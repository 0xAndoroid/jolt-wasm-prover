//! Tests for the virtual-matrix opening (`prove_virtual`) and the batched
//! tier-2 commitment (`tier2_batch`): Transparent proofs must be
//! byte-identical to dory-pcs (square and rectangular), ZK proofs must
//! verify with the stock verifier, and batched tier-2 outputs must match
//! `multi_pair_g2_setup`.

#![cfg(not(target_arch = "wasm32"))]

use ark_bn254::{Fr, G1Projective};
use ark_ff::UniformRand;
use ark_std::Zero;
use dory_gpu::open::OpeningInputs;
use dory_gpu::prove::GpuDory;
use dory_gpu::GpuContext;
use dory_pcs::backends::arkworks::{
    ArkFr, ArkworksPolynomial, Blake2bTranscript, G1Routines, G2Routines, BN254,
};
use dory_pcs::primitives::arithmetic::PairingCurve;
use dory_pcs::primitives::poly::{MultilinearLagrange, Polynomial};
use dory_pcs::{ProverSetup, Transparent, ZK};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

const DOMAIN: &[u8] = b"dory-gpu-open-virtual";

struct Instance {
    coeffs: Vec<ArkFr>,
    point: Vec<ArkFr>,
    setup: ProverSetup<BN254>,
    nu: usize,
    sigma: usize,
}

fn gen(nu: usize, sigma: usize) -> Instance {
    let log_n = nu + sigma;
    let mut rng = ChaCha20Rng::seed_from_u64((nu * 100 + sigma) as u64);
    let coeffs: Vec<ArkFr> = (0..1usize << log_n)
        .map(|_| ArkFr(Fr::rand(&mut rng)))
        .collect();
    let point: Vec<ArkFr> = (0..log_n).map(|_| ArkFr(Fr::rand(&mut rng))).collect();
    let setup = ProverSetup::<BN254>::new(sigma * 2);
    Instance {
        coeffs,
        point,
        setup,
        nu,
        sigma,
    }
}

/// CPU-side opening inputs exactly as Jolt produces them: row commitments
/// from the tier-1 commit, evaluation vectors and the VMV product from the
/// polynomial trait.
fn cpu_inputs(inst: &Instance) -> (Vec<G1Projective>, Vec<Fr>, Vec<Fr>, Vec<Fr>) {
    let poly = ArkworksPolynomial::new(inst.coeffs.clone());
    let (_, rows, _) = poly
        .commit::<BN254, Transparent, G1Routines>(inst.nu, inst.sigma, &inst.setup)
        .expect("cpu commit");
    let (left, right) = poly.compute_evaluation_vectors(&inst.point, inst.nu, inst.sigma);
    let v_vec = poly.vector_matrix_product(&left, inst.nu, inst.sigma);
    (
        rows.iter().map(|r| r.0).collect(),
        v_vec.iter().map(|v| v.0).collect(),
        left.iter().map(|v| v.0).collect(),
        right.iter().map(|v| v.0).collect(),
    )
}

fn run_transparent_equality(nu: usize, sigma: usize) {
    pollster::block_on(async {
        let inst = gen(nu, sigma);
        let poly = ArkworksPolynomial::new(inst.coeffs.clone());

        let (rows_proj, v_vec, left, right) = cpu_inputs(&inst);

        // CPU reference proof.
        let rows_ark = rows_proj
            .iter()
            .map(|r| dory_pcs::backends::arkworks::ArkG1(*r))
            .collect();
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
            &poly,
            &inst.point,
            rows_ark,
            ArkFr(Fr::zero()),
            inst.nu,
            inst.sigma,
            &inst.setup,
            &mut t_cpu,
        )
        .expect("cpu prove");

        // GPU virtual-matrix proof.
        let ctx = GpuContext::new().await.expect("gpu context");
        let gpu = GpuDory::new(std::sync::Arc::new(ctx), inst.setup.clone());
        let inputs = OpeningInputs {
            v_vec: &v_vec,
            rows: &rows_proj,
            left: &left,
            right: &right,
            nu: inst.nu,
            sigma: inst.sigma,
        };
        let mut t_gpu = Blake2bTranscript::new(DOMAIN);
        let (gpu_proof, y_blind) = gpu
            .prove_virtual::<Transparent, _>(&inputs, || unreachable!(), &mut t_gpu)
            .await;

        assert!(y_blind.is_none());
        assert_eq!(
            gpu_proof, cpu_proof,
            "proof mismatch at nu={nu} sigma={sigma}"
        );
    });
}

#[test]
fn prove_virtual_square_byte_identical() {
    run_transparent_equality(5, 5);
}

#[test]
fn prove_virtual_rectangular_byte_identical() {
    run_transparent_equality(4, 5);
}

#[test]
fn prove_virtual_zk_verifies_with_stock_verifier() {
    pollster::block_on(async {
        let (nu, sigma) = (5usize, 5usize);
        let inst = gen(nu, sigma);
        let poly = ArkworksPolynomial::new(inst.coeffs.clone());
        let (rows_proj, v_vec, left, right) = cpu_inputs(&inst);

        let ctx = GpuContext::new().await.expect("gpu context");
        let gpu = GpuDory::new(std::sync::Arc::new(ctx), inst.setup.clone());
        let inputs = OpeningInputs {
            v_vec: &v_vec,
            rows: &rows_proj,
            left: &left,
            right: &right,
            nu,
            sigma,
        };
        let mut t_gpu = Blake2bTranscript::new(DOMAIN);
        let evaluation = poly.evaluate(&inst.point);
        let (gpu_proof, y_blind) = gpu
            .prove_virtual::<ZK, _>(&inputs, || evaluation.0, &mut t_gpu)
            .await;
        assert!(y_blind.is_some(), "ZK proof must return the y blinding");
        assert!(gpu_proof.y_com.is_some());
        assert!(gpu_proof.scalar_product_proof.is_some());

        // The tier-2 commitment is unblinded (Jolt commits in Transparent
        // mode and opens in ZK mode).
        let (tier2, _, _) = poly
            .commit::<BN254, Transparent, G1Routines>(nu, sigma, &inst.setup)
            .expect("cpu commit");

        let mut t_verify = Blake2bTranscript::new(DOMAIN);
        dory_pcs::verify::<ArkFr, BN254, G1Routines, G2Routines, Blake2bTranscript<BN254>>(
            tier2,
            evaluation,
            &inst.point,
            &gpu_proof,
            inst.setup.to_verifier_setup(),
            &mut t_verify,
        )
        .expect("ZK GPU proof must verify");
    });
}

#[test]
fn tier2_batch_matches_multi_pair() {
    pollster::block_on(async {
        let setup = ProverSetup::<BN254>::new(12);
        let mut rng = ChaCha20Rng::seed_from_u64(7);

        // Varied group sizes, including identity rows (empty one-hot
        // buckets) and a non-power-of-two count that exercises padding.
        let mut groups: Vec<Vec<G1Projective>> = Vec::new();
        for (len, zero_stride) in [(64usize, 0usize), (13, 3), (32, 1), (1, 0)] {
            let mut g: Vec<G1Projective> = (0..len).map(|_| G1Projective::rand(&mut rng)).collect();
            if zero_stride > 0 {
                for i in (0..len).step_by(zero_stride) {
                    g[i] = G1Projective::zero();
                }
            }
            groups.push(g);
        }

        let ctx = GpuContext::new().await.expect("gpu context");
        let gpu = GpuDory::new(std::sync::Arc::new(ctx), setup.clone());
        let gpu_out = gpu.tier2_batch(&groups).await;

        for (g, gpu_gt) in groups.iter().zip(&gpu_out) {
            let rows: Vec<_> = g
                .iter()
                .map(|p| dory_pcs::backends::arkworks::ArkG1(*p))
                .collect();
            let cpu = <BN254 as PairingCurve>::multi_pair_g2_setup(&rows, &setup.g2_vec[..g.len()]);
            assert_eq!(*gpu_gt, cpu, "tier-2 mismatch for group of {}", g.len());
        }
    });
}
