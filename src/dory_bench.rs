//! Browser benchmark entry points: Dory commit + opening on CPU-WASM
//! (dory-pcs with the rayon worker pool) vs WebGPU (dory-gpu). Both paths
//! produce identical proofs; the GPU result is verified with the CPU
//! verifier inside the benchmark.

use ark_bn254::Fr;
use ark_ff::UniformRand;
use ark_std::rand::SeedableRng;
use dory_gpu::prove::GpuDory;
use dory_gpu::GpuContext;
use dory_pcs::backends::arkworks::{
    ArkFr, ArkworksPolynomial, Blake2bTranscript, G1Routines, G2Routines, BN254,
};
use dory_pcs::primitives::poly::Polynomial;
use dory_pcs::{ProverSetup, Transparent};
use wasm_bindgen::prelude::*;

const DOMAIN: &[u8] = b"dory-browser-bench";

fn now_ms() -> f64 {
    js_sys::Date::now()
}

fn clog(msg: &str) {
    web_sys::console::log_1(&JsValue::from_str(msg));
}

struct BenchData {
    coeffs_fr: Vec<Fr>,
    coeffs: Vec<ArkFr>,
    point_fr: Vec<Fr>,
    point: Vec<ArkFr>,
    setup: ProverSetup<BN254>,
    nu: usize,
    sigma: usize,
}

fn gen_data(log_n: usize) -> BenchData {
    let nu = log_n / 2;
    let sigma = log_n - nu;
    assert_eq!(nu, sigma, "square sizes only");
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(log_n as u64);
    let coeffs_fr: Vec<Fr> = (0..1usize << log_n).map(|_| Fr::rand(&mut rng)).collect();
    let coeffs: Vec<ArkFr> = coeffs_fr.iter().map(|c| ArkFr(*c)).collect();
    let point_fr: Vec<Fr> = (0..log_n).map(|_| Fr::rand(&mut rng)).collect();
    let point: Vec<ArkFr> = point_fr.iter().map(|p| ArkFr(*p)).collect();
    let setup = ProverSetup::<BN254>::new(log_n);
    BenchData {
        coeffs_fr,
        coeffs,
        point_fr,
        point,
        setup,
        nu,
        sigma,
    }
}

fn result_json(commit_ms: f64, open_ms: f64, setup_ms: f64) -> String {
    format!(
        "{{\"commit_ms\":{commit_ms:.1},\"open_ms\":{open_ms:.1},\"setup_ms\":{setup_ms:.1},\"verified\":true}}"
    )
}

/// CPU-WASM baseline. Must run on a worker thread with the rayon pool
/// initialized (`init_thread_pool`).
#[wasm_bindgen]
pub fn dory_bench_cpu(log_n: usize) -> Result<String, JsValue> {
    let t = now_ms();
    let data = gen_data(log_n);
    let setup_ms = now_ms() - t;

    let poly = ArkworksPolynomial::new(data.coeffs.clone());
    let t = now_ms();
    let (tier2, rows, blind) = poly
        .commit::<BN254, Transparent, G1Routines>(data.nu, data.sigma, &data.setup)
        .map_err(|e| JsValue::from_str(&format!("cpu commit: {e}")))?;
    let commit_ms = now_ms() - t;

    let mut transcript = Blake2bTranscript::new(DOMAIN);
    let t = now_ms();
    let (proof, _) = dory_pcs::prove::<
        ArkFr,
        BN254,
        G1Routines,
        G2Routines,
        ArkworksPolynomial,
        Blake2bTranscript<BN254>,
        Transparent,
    >(
        &poly,
        &data.point,
        rows,
        blind,
        data.nu,
        data.sigma,
        &data.setup,
        &mut transcript,
    )
    .map_err(|e| JsValue::from_str(&format!("cpu prove: {e}")))?;
    let open_ms = now_ms() - t;

    let evaluation = poly.evaluate(&data.point);
    let verifier_setup = data.setup.to_verifier_setup();
    let mut t_verify = Blake2bTranscript::new(DOMAIN);
    dory_pcs::verify::<ArkFr, BN254, G1Routines, G2Routines, Blake2bTranscript<BN254>>(
        tier2,
        evaluation,
        &data.point,
        &proof,
        verifier_setup,
        &mut t_verify,
    )
    .map_err(|e| JsValue::from_str(&format!("cpu verify: {e}")))?;

    Ok(result_json(commit_ms, open_ms, setup_ms))
}

thread_local! {
    static GPU_CTX: std::cell::RefCell<Option<std::sync::Arc<GpuContext>>> =
        const { std::cell::RefCell::new(None) };
}

/// WebGPU path. Runs on a worker (WebGPU is available in dedicated workers);
/// the rayon pool is used for the per-round final exponentiations. The
/// GpuContext (and with it every compiled pipeline) is cached across calls.
#[wasm_bindgen]
pub async fn dory_bench_gpu(log_n: usize) -> Result<String, JsValue> {
    let t = now_ms();
    clog("gen_data…");
    let data = gen_data(log_n);
    let cached = GPU_CTX.with(|c| c.borrow().clone());
    let ctx = match cached {
        Some(ctx) => ctx,
        None => {
            clog("gpu ctx…");
            let ctx = std::sync::Arc::new(
                GpuContext::new()
                    .await
                    .map_err(|e| JsValue::from_str(&format!("webgpu init: {e}")))?,
            );
            GPU_CTX.with(|c| *c.borrow_mut() = Some(ctx.clone()));
            ctx
        }
    };
    clog("GpuDory::new (prepared lines + tables)…");
    let gpu = GpuDory::new(ctx, data.setup.clone());
    let matrix = gpu.upload_matrix(&data.coeffs_fr);
    let setup_ms = now_ms() - t;
    clog("commit…");

    let t = now_ms();
    let commitment = gpu.commit(&matrix, data.nu, data.sigma).await;
    let commit_ms = now_ms() - t;
    clog("open…");

    let mut transcript = Blake2bTranscript::new(DOMAIN);
    let t = now_ms();
    let proof = gpu
        .prove(&matrix, &commitment, &data.point_fr, &mut transcript)
        .await;
    let open_ms = now_ms() - t;

    let poly = ArkworksPolynomial::new(data.coeffs.clone());
    let evaluation = poly.evaluate(&data.point);
    let verifier_setup = data.setup.to_verifier_setup();
    let mut t_verify = Blake2bTranscript::new(DOMAIN);
    dory_pcs::verify::<ArkFr, BN254, G1Routines, G2Routines, Blake2bTranscript<BN254>>(
        commitment.tier2,
        evaluation,
        &data.point_fr.iter().map(|p| ArkFr(*p)).collect::<Vec<_>>(),
        &proof,
        verifier_setup,
        &mut t_verify,
    )
    .map_err(|e| JsValue::from_str(&format!("gpu proof failed verification: {e}")))?;

    Ok(result_json(commit_ms, open_ms, setup_ms))
}
