pub mod engine;

#[cfg(target_arch = "wasm32")]
mod wasm_tracing;

#[cfg(target_arch = "wasm32")]
mod wasm {
    use crate::engine;
    use wasm_bindgen::prelude::*;

    pub use wasm_bindgen_rayon::init_thread_pool;

    #[wasm_bindgen(start)]
    pub fn wasm_main() {
        console_error_panic_hook::set_once();
        web_sys::console::log_1(&"jolt-wasm-prover build tag 2".into());
    }

    #[wasm_bindgen]
    pub fn init_tracing() {
        crate::wasm_tracing::init();
    }

    #[wasm_bindgen]
    pub fn get_trace_json() -> String {
        crate::wasm_tracing::get_trace_json()
    }

    #[wasm_bindgen]
    pub fn clear_trace() {
        crate::wasm_tracing::clear();
    }

    /// Inline registration is inventory-based (link-time ctors, run by
    /// `__wasm_call_ctors` at instantiation). Kept as a no-op so worker.js
    /// doesn't change.
    #[wasm_bindgen]
    pub fn init_inlines() -> Result<(), JsValue> {
        use jolt_inlines_keccak256 as _;
        use jolt_inlines_secp256k1 as _;
        use jolt_inlines_sha2 as _;
        Ok(())
    }

    /// Installs the webgpu arm's gates from the JS config. Zero values keep
    /// the defaults. Call before [`webgpu_warmup`].
    #[wasm_bindgen]
    pub fn webgpu_configure(disable: bool, min_terms: u32, handoff_len: u32) {
        let mut options = jolt_kernels::webgpu::WebGpuOptions {
            disable,
            ..Default::default()
        };
        if min_terms > 0 {
            options.min_terms = min_terms as usize;
        }
        if handoff_len > 0 {
            options.handoff_len = handoff_len as usize;
        }
        jolt_kernels::webgpu::configure(options);
    }

    /// Brings the WebGPU engine up (adapter, device, every pipeline),
    /// blocking this worker until the dedicated GPU worker reports. The GPU
    /// worker MUST already be pumping the job queue (`gpu-ready` handshake
    /// in worker.js), or this never returns. Errors mean the prover runs
    /// CPU-only — the caller just logs them.
    #[wasm_bindgen]
    pub fn webgpu_warmup() -> Result<(), JsValue> {
        jolt_kernels::webgpu::warmup().map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen]
    pub struct WasmProver {
        preprocessing: engine::ProverPrep,
        elf_bytes: Vec<u8>,
    }

    #[wasm_bindgen]
    impl WasmProver {
        #[wasm_bindgen(constructor)]
        pub fn new(
            srs_bytes: &[u8],
            verifier_preprocessing_bytes: &[u8],
            elf_bytes: &[u8],
        ) -> Result<WasmProver, JsValue> {
            let preprocessing =
                engine::build_prover_preprocessing(srs_bytes, verifier_preprocessing_bytes)
                    .map_err(|e| JsValue::from_str(&e))?;
            Ok(Self {
                preprocessing,
                elf_bytes: elf_bytes.to_vec(),
            })
        }

        fn prove_with_inputs(&self, inputs: &[u8]) -> Result<ProveResult, JsValue> {
            let out = engine::prove(&self.preprocessing, &self.elf_bytes, inputs)
                .map_err(|e| JsValue::from_str(&e))?;
            Ok(ProveResult { out })
        }

        pub fn prove_sha2(&self, input: &[u8]) -> Result<ProveResult, JsValue> {
            let inputs = postcard::to_allocvec(&input)
                .map_err(|e| JsValue::from_str(&format!("input serialization error: {e}")))?;
            self.prove_with_inputs(&inputs)
        }

        pub fn prove_ecdsa(
            &self,
            z: &[u64],
            r: &[u64],
            s: &[u64],
            q: &[u64],
        ) -> Result<ProveResult, JsValue> {
            let z: [u64; 4] = z
                .try_into()
                .map_err(|_| JsValue::from_str("z must be 4 u64s"))?;
            let r: [u64; 4] = r
                .try_into()
                .map_err(|_| JsValue::from_str("r must be 4 u64s"))?;
            let s: [u64; 4] = s
                .try_into()
                .map_err(|_| JsValue::from_str("s must be 4 u64s"))?;
            let q: [u64; 8] = q
                .try_into()
                .map_err(|_| JsValue::from_str("q must be 8 u64s"))?;

            let mut inputs = Vec::new();
            for part in [
                postcard::to_allocvec(&z),
                postcard::to_allocvec(&r),
                postcard::to_allocvec(&s),
            ] {
                inputs.extend_from_slice(
                    &part.map_err(|e| JsValue::from_str(&format!("serialization error: {e}")))?,
                );
            }
            inputs.extend_from_slice(
                &postcard::to_allocvec(&q)
                    .map_err(|e| JsValue::from_str(&format!("serialization error: {e}")))?,
            );
            self.prove_with_inputs(&inputs)
        }

        pub fn prove_keccak_chain(
            &self,
            input: &[u8],
            num_iters: u32,
        ) -> Result<ProveResult, JsValue> {
            self.prove_hash_chain(input, num_iters)
        }

        pub fn prove_sha2_chain(
            &self,
            input: &[u8],
            num_iters: u32,
        ) -> Result<ProveResult, JsValue> {
            self.prove_hash_chain(input, num_iters)
        }

        fn prove_hash_chain(&self, input: &[u8], num_iters: u32) -> Result<ProveResult, JsValue> {
            let input: [u8; 32] = input
                .try_into()
                .map_err(|_| JsValue::from_str("input must be 32 bytes"))?;

            let mut inputs = Vec::new();
            inputs.extend_from_slice(
                &postcard::to_allocvec(&input)
                    .map_err(|e| JsValue::from_str(&format!("input serialization error: {e}")))?,
            );
            inputs.extend_from_slice(
                &postcard::to_allocvec(&num_iters).map_err(|e| {
                    JsValue::from_str(&format!("num_iters serialization error: {e}"))
                })?,
            );
            self.prove_with_inputs(&inputs)
        }
    }

    #[wasm_bindgen]
    pub struct ProveResult {
        out: engine::ProveOutput,
    }

    #[wasm_bindgen]
    impl ProveResult {
        #[wasm_bindgen(getter)]
        pub fn proof(&self) -> Vec<u8> {
            self.out.proof_bytes.clone()
        }

        #[wasm_bindgen(getter)]
        pub fn proof_size(&self) -> usize {
            self.out.proof_bytes.len()
        }

        #[wasm_bindgen(getter)]
        pub fn compressed_proof_size(&self) -> usize {
            self.out.proof_bytes.len()
        }

        #[wasm_bindgen(getter)]
        pub fn program_io(&self) -> Vec<u8> {
            self.out.io_bytes.clone()
        }

        #[wasm_bindgen(getter)]
        pub fn num_cycles(&self) -> usize {
            self.out.unpadded_cycles
        }

        #[wasm_bindgen(getter)]
        pub fn padded_cycles(&self) -> usize {
            self.out.padded_cycles
        }
    }

    #[wasm_bindgen]
    pub struct WasmVerifier {
        preprocessing: engine::VerifierPrep,
    }

    #[wasm_bindgen]
    impl WasmVerifier {
        #[wasm_bindgen(constructor)]
        pub fn new(preprocessing_bytes: &[u8]) -> Result<WasmVerifier, JsValue> {
            let preprocessing = engine::decode_verifier_preprocessing(preprocessing_bytes)
                .map_err(|e| JsValue::from_str(&e))?;
            Ok(Self { preprocessing })
        }

        pub fn verify(&self, proof_bytes: &[u8], program_io_bytes: &[u8]) -> Result<bool, JsValue> {
            engine::verify(&self.preprocessing, proof_bytes, program_io_bytes)
                .map(|_| true)
                .map_err(|e| JsValue::from_str(&e))
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub use wasm::*;
