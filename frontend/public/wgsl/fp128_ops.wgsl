// Elementwise fp128 ops: params[0].x = n, params[0].y = op
// (0 mul, 1 add, 2 sub, 3 mul-add). Used by the prover self-test and by
// bench/test_fp128_wgsl.py.

@group(0) @binding(0) var<uniform> params: array<vec4<u32>, 4>;
@group(0) @binding(1) var<storage, read> a: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> b: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> c: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> out: array<vec4<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params[0].x) {
        return;
    }
    switch params[0].y {
        case 0u: { out[i] = fp128_mul(a[i], b[i]); }
        case 1u: { out[i] = fp128_add(a[i], b[i]); }
        case 2u: { out[i] = fp128_sub(a[i], b[i]); }
        default: { out[i] = fp128_muladd(a[i], b[i], c[i]); }
    }
}
