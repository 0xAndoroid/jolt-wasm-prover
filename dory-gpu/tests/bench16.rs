#![cfg(not(target_arch = "wasm32"))]

// Focused GPU-only run at 2^16 with stage tracing, for performance work.

use ark_bn254::Fr;
use ark_ff::UniformRand;
use dory_gpu::prove::GpuDory;
use dory_gpu::GpuContext;
use dory_pcs::backends::arkworks::{Blake2bTranscript, BN254};
use dory_pcs::ProverSetup;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::time::Instant;

#[test]
#[ignore = "benchmark; run explicitly"]
fn gpu_only_2_16() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .try_init();
    pollster::block_on(async {
        let log_n = 16usize;
        let (nu, sigma) = (8usize, 8usize);
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let coeffs_fr: Vec<Fr> = (0..1 << log_n).map(|_| Fr::rand(&mut rng)).collect();
        let point_fr: Vec<Fr> = (0..log_n).map(|_| Fr::rand(&mut rng)).collect();
        let setup = ProverSetup::<BN254>::new(log_n);

        let ctx = std::sync::Arc::new(GpuContext::new().await.unwrap());
        let gpu = GpuDory::new(ctx, setup);
        let matrix = gpu.upload_matrix(&coeffs_fr);

        for iter in 0..2 {
            let t = Instant::now();
            let commitment = gpu.commit(&matrix, nu, sigma).await;
            let commit_time = t.elapsed();
            let mut tr = Blake2bTranscript::new(b"bench16");
            let t = Instant::now();
            let _proof = gpu.prove(&matrix, &commitment, &point_fr, &mut tr).await;
            let prove_time = t.elapsed();
            eprintln!("iter {iter}: gpu commit {commit_time:?}, gpu prove {prove_time:?}");
        }
    });
}
