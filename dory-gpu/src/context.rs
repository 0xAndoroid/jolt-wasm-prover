//! wgpu device management: initialization, pipeline cache, buffer helpers,
//! and async readback that works on both native (Metal) and WASM (WebGPU).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use wgpu::util::DeviceExt;

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub limits: wgpu::Limits,
    // Mutex (not RefCell) so the context can be shared across threads on
    // native, where wgpu types are Send + Sync; on WASM the context stays
    // on its creating thread and the locks are uncontended.
    modules: Mutex<HashMap<&'static str, wgpu::ShaderModule>>,
    pipelines: Mutex<HashMap<(&'static str, &'static str), Arc<wgpu::ComputePipeline>>>,
}

impl GpuContext {
    pub async fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no WebGPU adapter: {e}"))?;

        let adapter_limits = adapter.limits();
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            max_buffer_size: adapter_limits.max_buffer_size,
            max_compute_workgroup_storage_size: adapter_limits.max_compute_workgroup_storage_size,
            max_compute_invocations_per_workgroup: adapter_limits
                .max_compute_invocations_per_workgroup,
            ..wgpu::Limits::default()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("dory-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: limits.clone(),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("WebGPU device request failed: {e}"))?;

        Ok(Self {
            device,
            queue,
            limits,
            modules: Mutex::new(HashMap::new()),
            pipelines: Mutex::new(HashMap::new()),
        })
    }

    /// Returns a cached compute pipeline, compiling the module on first use.
    /// `module_key` identifies the assembled source; `source` is only invoked
    /// on a cache miss.
    pub fn pipeline(
        &self,
        module_key: &'static str,
        entry: &'static str,
        source: impl FnOnce() -> String,
    ) -> Arc<wgpu::ComputePipeline> {
        if let Some(p) = self.pipelines.lock().unwrap().get(&(module_key, entry)) {
            return p.clone();
        }
        let mut modules = self.modules.lock().unwrap();
        let module = modules.entry(module_key).or_insert_with(|| {
            // WARNING: keep default runtime checks (forced loop bounding ON).
            // On Apple Metal, disabling naga's loop bounding makes the
            // optimizer miscompile multi-mul field kernels (wrong results);
            // with it on, oversized kernels hang instead — so kernels must
            // stay small, which the MSM reduction shape accounts for.
            self.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(module_key),
                    source: wgpu::ShaderSource::Wgsl(source().into()),
                })
        });
        let pipeline = Arc::new(self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            },
        ));
        self.pipelines
            .lock()
            .unwrap()
            .insert((module_key, entry), pipeline.clone());
        pipeline
    }

    pub fn buffer_from(
        &self,
        label: &str,
        contents: &[u8],
        extra: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::STORAGE | extra,
            })
    }

    pub fn empty_buffer(&self, label: &str, size: u64, extra: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(4),
            usage: wgpu::BufferUsages::STORAGE | extra,
            mapped_at_creation: false,
        })
    }

    /// Encodes one compute dispatch with sequentially numbered bindings.
    pub fn encode_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        buffers: &[&wgpu::Buffer],
        workgroups: (u32, u32, u32),
    ) {
        let pairs: Vec<(u32, &wgpu::Buffer)> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| (i as u32, *b))
            .collect();
        self.encode_pass_indexed(encoder, pipeline, &pairs, workgroups);
    }

    /// Encodes one compute dispatch with explicit binding indices. Bindings
    /// not statically used by the entry point must be omitted (auto layout
    /// drops them).
    pub fn encode_pass_indexed(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        buffers: &[(u32, &wgpu::Buffer)],
        workgroups: (u32, u32, u32),
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        self.dispatch_in_pass(&mut pass, pipeline, buffers, workgroups);
    }

    /// Records one dispatch into an already-open compute pass. Long dispatch
    /// sequences must share a pass: every pass boundary is a Metal encoder
    /// switch costing tens of milliseconds, while in-pass hazard barriers are
    /// nearly free.
    pub fn dispatch_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        pipeline: &wgpu::ComputePipeline,
        buffers: &[(u32, &wgpu::Buffer)],
        workgroups: (u32, u32, u32),
    ) {
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: *i,
                resource: b.as_entire_binding(),
            })
            .collect();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    }

    /// One-shot dispatch helper for single-kernel jobs.
    pub fn run_pass(
        &self,
        pipeline: &wgpu::ComputePipeline,
        buffers: &[&wgpu::Buffer],
        workgroups: (u32, u32, u32),
    ) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.encode_pass(&mut encoder, pipeline, buffers, workgroups);
        self.queue.submit([encoder.finish()]);
    }

    /// Drives the device on native; no-op on WebGPU where the browser polls.
    pub fn poll_wait(&self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("device poll failed");
        }
    }

    /// Copies a region of `src` into a staging buffer and maps it for reading.
    pub async fn read_buffer(&self, src: &wgpu::Buffer, offset: u64, size: u64) -> Vec<u8> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging-read"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(src, offset, &staging, 0, size);
        self.queue.submit([encoder.finish()]);

        let (tx, rx) = futures_channel::oneshot::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        self.poll_wait();
        rx.await
            .expect("map_async callback dropped")
            .expect("buffer mapping failed");
        let data = staging.slice(..).get_mapped_range().to_vec();
        staging.unmap();
        data
    }
}
