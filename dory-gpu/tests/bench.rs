#![cfg(not(target_arch = "wasm32"))]

// CPU (dory-pcs, rayon + prepared-point cache) vs GPU (dory-gpu) commit and
// opening benchmark. Run with:
//   cargo nextest run -p dory-gpu --test bench --run-ignored all --no-capture
// or: cargo test -p dory-gpu --test bench --release -- --ignored --nocapture

use ark_bn254::Fr;
use ark_ff::UniformRand;
use dory_gpu::prove::GpuDory;
use dory_gpu::GpuContext;
use dory_pcs::backends::arkworks::{
    init_cache, ArkFr, ArkworksPolynomial, Blake2bTranscript, G1Routines, G2Routines, BN254,
};
use dory_pcs::primitives::poly::Polynomial;
use dory_pcs::{ProverSetup, Transparent};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::time::Instant;

const DOMAIN: &[u8] = b"dory-gpu-bench";

#[test]
#[ignore = "benchmark; run explicitly"]
fn bench_cpu_vs_gpu() {
    pollster::block_on(async {
        let ctx = std::sync::Arc::new(GpuContext::new().await.expect("gpu context"));
        let mut results = Vec::new();

        for log_n in [16usize, 18, 20] {
            let nu = log_n / 2;
            let sigma = log_n - nu;
            let mut rng = ChaCha20Rng::seed_from_u64(log_n as u64);
            eprintln!("--- 2^{log_n}: generating setup + data ---");
            let coeffs_fr: Vec<Fr> = (0..1 << log_n).map(|_| Fr::rand(&mut rng)).collect();
            let coeffs: Vec<ArkFr> = coeffs_fr.iter().map(|c| ArkFr(*c)).collect();
            let point_fr: Vec<Fr> = (0..log_n).map(|_| Fr::rand(&mut rng)).collect();
            let point: Vec<ArkFr> = point_fr.iter().map(|p| ArkFr(*p)).collect();
            let setup = ProverSetup::<BN254>::new(log_n);
            init_cache(&setup.g1_vec, &setup.g2_vec);
            let verifier_setup = setup.to_verifier_setup();

            // CPU.
            let poly = ArkworksPolynomial::new(coeffs.clone());
            let t = Instant::now();
            let (cpu_tier2, cpu_rows, blind) = poly
                .commit::<BN254, Transparent, G1Routines>(nu, sigma, &setup)
                .unwrap();
            let cpu_commit = t.elapsed();
            eprintln!("  cpu commit done: {cpu_commit:?}");
            let mut t_cpu = Blake2bTranscript::new(DOMAIN);
            let t = Instant::now();
            let (cpu_proof, _) = dory_pcs::prove::<
                ArkFr,
                BN254,
                G1Routines,
                G2Routines,
                ArkworksPolynomial,
                Blake2bTranscript<BN254>,
                Transparent,
            >(
                &poly, &point, cpu_rows, blind, nu, sigma, &setup, &mut t_cpu,
            )
            .unwrap();
            let cpu_prove = t.elapsed();
            eprintln!("  cpu prove done: {cpu_prove:?}");

            // GPU (setup upload outside the timed sections, like the CPU
            // generator setup; first iteration also warms pipeline compiles).
            let gpu = GpuDory::new(ctx.clone(), setup.clone());
            let matrix = gpu.upload_matrix(&coeffs_fr);
            // Warmup pass to amortize Metal pipeline compilation.
            eprintln!("  gpu setup uploaded");
            if log_n == 16 {
                let c = gpu.commit(&matrix, nu, sigma).await;
                let mut t_w = Blake2bTranscript::new(DOMAIN);
                let _ = gpu.prove(&matrix, &c, &point_fr, &mut t_w).await;
            }
            eprintln!("  gpu warmup done");
            let t = Instant::now();
            let commitment = gpu.commit(&matrix, nu, sigma).await;
            let gpu_commit = t.elapsed();
            eprintln!("  gpu commit done: {gpu_commit:?}");
            assert_eq!(commitment.tier2, cpu_tier2, "commitment mismatch");
            let mut t_gpu = Blake2bTranscript::new(DOMAIN);
            let t = Instant::now();
            let gpu_proof = gpu.prove(&matrix, &commitment, &point_fr, &mut t_gpu).await;
            let gpu_prove = t.elapsed();
            assert_eq!(gpu_proof, cpu_proof, "proof mismatch at 2^{log_n}");

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
            .unwrap();

            eprintln!(
                "2^{log_n}: commit cpu {cpu_commit:?} gpu {gpu_commit:?} | prove cpu {cpu_prove:?} gpu {gpu_prove:?}"
            );
            results.push((log_n, cpu_commit, gpu_commit, cpu_prove, gpu_prove));
        }

        eprintln!("\n| size | CPU commit | GPU commit | CPU open | GPU open |");
        eprintln!("|------|-----------|-----------|----------|----------|");
        for (log_n, cc, gc, cp, gp) in results {
            eprintln!(
                "| 2^{log_n} | {:.3}s | {:.3}s | {:.3}s | {:.3}s |",
                cc.as_secs_f64(),
                gc.as_secs_f64(),
                cp.as_secs_f64(),
                gp.as_secs_f64()
            );
        }
    });
}
