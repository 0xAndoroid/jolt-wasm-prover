#![cfg(not(target_arch = "wasm32"))]

// Diagnostic: cost of sequential dependent dispatches of the big pairing
// kernels, by dispatch count and by pair count.

use dory_gpu::shader::{
    fq_header, ShaderBuilder, FIELD_WGSL, FQ12_WGSL, FQ2_WGSL, PAIRING_FQ12_WGSL,
};
use dory_gpu::GpuContext;
use std::time::Instant;

fn fq12_module_source() -> String {
    ShaderBuilder::new()
        .push(&fq_header())
        .push(&FIELD_WGSL)
        .push(FQ2_WGSL)
        .push(FQ12_WGSL)
        .push(PAIRING_FQ12_WGSL)
        .build()
}

#[test]
fn dispatch_overhead_profile() {
    pollster::block_on(async {
        let ctx = GpuContext::new().await.expect("gpu context");
        let sqr_a = ctx.pipeline("pair_fq12", "f_sqr_a", fq12_module_source);
        let sqr_b = ctx.pipeline("pair_fq12", "f_sqr_b", fq12_module_source);

        for n_pairs in [32u32, 2048] {
            let f = ctx.empty_buffer("f", n_pairs as u64 * 96 * 4, wgpu::BufferUsages::COPY_SRC);
            let mask = ctx.empty_buffer("mask", n_pairs as u64 * 4, wgpu::BufferUsages::empty());
            let scratch = ctx.empty_buffer(
                "scratch",
                n_pairs as u64 * 144 * 4,
                wgpu::BufferUsages::empty(),
            );
            let params = ctx.buffer_from(
                "params",
                bytemuck::cast_slice(&[n_pairs, 0u32, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
                wgpu::BufferUsages::UNIFORM,
            );

            for n_dispatch in [16u32, 128, 512] {
                let t = Instant::now();
                let mut enc = ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    for i in 0..n_dispatch {
                        let p = if i % 2 == 0 { &sqr_a } else { &sqr_b };
                        ctx.dispatch_in_pass(
                            &mut pass,
                            p,
                            &[(0, &params), (1, &f), (2, &mask), (6, &scratch)],
                            (n_pairs.div_ceil(64), 1, 1),
                        );
                    }
                }
                let encoded = t.elapsed();
                ctx.queue.submit([enc.finish()]);
                ctx.poll_wait();
                let total = t.elapsed();
                eprintln!(
                    "pairs={n_pairs:5} dispatches={n_dispatch:4}: encode {:6.1?}, total {:8.1?}, per-dispatch {:6.2?}",
                    encoded,
                    total,
                    total / n_dispatch
                );
            }
        }
    });
}
