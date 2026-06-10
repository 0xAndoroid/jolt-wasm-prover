#![cfg(not(target_arch = "wasm32"))]

use ark_bn254::Fr;
use ark_ff::UniformRand;
use dory_gpu::prove::GpuDory;
use dory_gpu::GpuContext;
use dory_pcs::backends::arkworks::{
    ArkFr, ArkworksPolynomial, Blake2bTranscript, G1Routines, G2Routines, BN254,
};
use dory_pcs::primitives::poly::Polynomial;
use dory_pcs::{ProverSetup, Transparent};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use tracing as _;

const DOMAIN: &[u8] = b"dory-gpu-e2e";

fn run_e2e(log_n: usize) {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .try_init();
    pollster::block_on(async {
        let nu = log_n / 2;
        let sigma = log_n - nu;
        assert_eq!(nu, sigma, "test sizes must be square");

        let mut rng = ChaCha20Rng::seed_from_u64(log_n as u64);
        let coeffs_fr: Vec<Fr> = (0..1 << log_n).map(|_| Fr::rand(&mut rng)).collect();
        let coeffs: Vec<ArkFr> = coeffs_fr.iter().map(|c| ArkFr(*c)).collect();
        let point_fr: Vec<Fr> = (0..log_n).map(|_| Fr::rand(&mut rng)).collect();
        let point: Vec<ArkFr> = point_fr.iter().map(|p| ArkFr(*p)).collect();

        let prover_setup = ProverSetup::<BN254>::new(log_n);
        let verifier_setup = prover_setup.to_verifier_setup();

        // CPU reference.
        let poly = ArkworksPolynomial::new(coeffs.clone());
        let (cpu_tier2, cpu_rows, blind) = poly
            .commit::<BN254, Transparent, G1Routines>(nu, sigma, &prover_setup)
            .expect("cpu commit");
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
            &point,
            cpu_rows,
            blind,
            nu,
            sigma,
            &prover_setup,
            &mut t_cpu,
        )
        .expect("cpu prove");

        // GPU.
        let ctx = GpuContext::new().await.expect("gpu context");
        let gpu = GpuDory::new(std::rc::Rc::new(ctx), prover_setup.clone());
        let matrix = gpu.upload_matrix(&coeffs_fr);
        let commitment = gpu.commit(&matrix, nu, sigma).await;
        assert_eq!(commitment.tier2, cpu_tier2, "tier-2 commitment mismatch");

        let mut t_gpu = Blake2bTranscript::new(DOMAIN);
        let gpu_proof = gpu.prove(&matrix, &commitment, &point_fr, &mut t_gpu).await;

        assert_eq!(
            gpu_proof.vmv_message, cpu_proof.vmv_message,
            "vmv message mismatch"
        );
        for (i, (g, c)) in gpu_proof
            .first_messages
            .iter()
            .zip(&cpu_proof.first_messages)
            .enumerate()
        {
            assert_eq!(g, c, "first message mismatch at round {i}");
        }
        for (i, (g, c)) in gpu_proof
            .second_messages
            .iter()
            .zip(&cpu_proof.second_messages)
            .enumerate()
        {
            assert_eq!(g, c, "second message mismatch at round {i}");
        }
        assert_eq!(gpu_proof, cpu_proof, "full proof mismatch");

        // The GPU proof verifies with the stock CPU verifier.
        let evaluation = poly.evaluate(&point);
        let mut t_verify = Blake2bTranscript::new(DOMAIN);
        dory_pcs::verify::<ArkFr, BN254, G1Routines, G2Routines, Blake2bTranscript<BN254>>(
            commitment.tier2,
            evaluation,
            &point,
            &gpu_proof,
            verifier_setup,
            &mut t_verify,
        )
        .expect("gpu proof verification");
    });
}

#[test]
fn dory_e2e_2_10() {
    run_e2e(10);
}

#[test]
fn dory_e2e_2_12() {
    run_e2e(12);
}
