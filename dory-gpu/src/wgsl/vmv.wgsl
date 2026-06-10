// Fr kernels for Dory: the vector-matrix product and scalar-vector folds.
// Module: field.wgsl instantiated for Fr.

struct VmvParams {
    rows: u32,
    cols: u32,
    left_offset: u32,
    out_offset: u32,
}

@group(0) @binding(0) var<uniform> vmv_params: VmvParams;
@group(0) @binding(1) var<storage, read> vmv_matrix: array<u32>;
@group(0) @binding(2) var<storage, read> vmv_left: array<u32>;
@group(0) @binding(3) var<storage, read_write> vmv_out: array<u32>;

fn vmv_load_m(idx: u32) -> Fe {
    var w: Fe8;
    for (var k = 0u; k < 8u; k++) {
        w[k] = vmv_matrix[idx * 8u + k];
    }
    return fe_unpack(w);
}

fn vmv_load_l(idx: u32) -> Fe {
    var w: Fe8;
    for (var k = 0u; k < 8u; k++) {
        w[k] = vmv_left[idx * 8u + k];
    }
    return fe_unpack(w);
}

fn vmv_store(idx: u32, v: Fe) {
    let w = fe_pack(v);
    for (var k = 0u; k < 8u; k++) {
        vmv_out[idx * 8u + k] = w[k];
    }
}

// out[j] = sum_i left[i] * M[i * cols + j]; one thread per column.
@compute @workgroup_size(64)
fn vmv_columns(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    let p = vmv_params;
    if (j >= p.cols) {
        return;
    }
    var acc = fe_zero();
    for (var i = 0u; i < p.rows; i++) {
        let m = vmv_load_m(i * p.cols + j);
        let l = vmv_load_l(p.left_offset + i);
        acc = fe_add(acc, fe_mont_mul(m, l));
    }
    vmv_store(p.out_offset + j, acc);
}

struct FoldScalarsParams {
    n: u32,
    s_offset: u32,
    a_offset: u32,
    out_offset: u32,
}

@group(0) @binding(4) var<uniform> fs_params: FoldScalarsParams;
@group(0) @binding(5) var<storage, read> fs_k: array<u32>;
@group(0) @binding(6) var<storage, read_write> fs_s: array<u32>;

fn fs_load(idx: u32) -> Fe {
    var w: Fe8;
    for (var k = 0u; k < 8u; k++) {
        w[k] = fs_s[idx * 8u + k];
    }
    return fe_unpack(w);
}

fn fs_store(idx: u32, v: Fe) {
    let w = fe_pack(v);
    for (var k = 0u; k < 8u; k++) {
        fs_s[idx * 8u + k] = w[k];
    }
}

// s[out_offset + i] = k * s[s_offset + i] + s[a_offset + i]  (k Montgomery, 8 words)
@compute @workgroup_size(64)
fn fold_scalars(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let p = fs_params;
    if (i >= p.n) {
        return;
    }
    var kw: Fe8;
    for (var k = 0u; k < 8u; k++) {
        kw[k] = fs_k[k];
    }
    let kf = fe_unpack(kw);
    let folded = fe_add(fe_mont_mul(kf, fs_load(p.s_offset + i)), fs_load(p.a_offset + i));
    fs_store(p.out_offset + i, folded);
}
