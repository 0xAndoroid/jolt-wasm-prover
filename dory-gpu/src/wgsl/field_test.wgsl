// Test entry points exercising field.wgsl / fq2.wgsl against CPU references.
// Layout: a, b, out are tightly packed 8-word (Fq) or 16-word (Fq2) elements.

@group(0) @binding(0) var<storage, read> in_a: array<u32>;
@group(0) @binding(1) var<storage, read> in_b: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

fn load_fe(buf_sel: u32, idx: u32) -> Fe {
    var w: Fe8;
    for (var k = 0u; k < 8u; k++) {
        if (buf_sel == 0u) {
            w[k] = in_a[idx * 8u + k];
        } else {
            w[k] = in_b[idx * 8u + k];
        }
    }
    return fe_unpack(w);
}

fn store_fe(idx: u32, v: Fe) {
    let w = fe_pack(v);
    for (var k = 0u; k < 8u; k++) {
        out[idx * 8u + k] = w[k];
    }
}

fn load_fe2(buf_sel: u32, idx: u32) -> Fe2 {
    return Fe2(load_fe(buf_sel, idx * 2u), load_fe(buf_sel, idx * 2u + 1u));
}

fn store_fe2(idx: u32, v: Fe2) {
    store_fe(idx * 2u, v.c0);
    store_fe(idx * 2u + 1u, v.c1);
}

@compute @workgroup_size(64)
fn t_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i * 8u >= arrayLength(&out)) { return; }
    store_fe(i, fe_mont_mul(load_fe(0u, i), load_fe(1u, i)));
}

@compute @workgroup_size(64)
fn t_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i * 8u >= arrayLength(&out)) { return; }
    store_fe(i, fe_add(load_fe(0u, i), load_fe(1u, i)));
}

@compute @workgroup_size(64)
fn t_sub(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i * 8u >= arrayLength(&out)) { return; }
    store_fe(i, fe_sub(load_fe(0u, i), load_fe(1u, i)));
}

@compute @workgroup_size(64)
fn t_neg(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i * 8u >= arrayLength(&out)) { return; }
    store_fe(i, fe_neg(load_fe(0u, i)));
}

@compute @workgroup_size(64)
fn t_from_mont(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i * 8u >= arrayLength(&out)) { return; }
    store_fe(i, fe_from_mont(load_fe(0u, i)));
}

// Repeated squaring stress test: out = a^(2^100) via 100 sequential mont muls.
@compute @workgroup_size(64)
fn t_sqr_chain(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i * 8u >= arrayLength(&out)) { return; }
    var acc = load_fe(0u, i);
    for (var k = 0u; k < 100u; k++) {
        acc = fe_mont_mul(acc, acc);
    }
    store_fe(i, acc);
}

@compute @workgroup_size(64)
fn t_fq2_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i * 16u >= arrayLength(&out)) { return; }
    store_fe2(i, fq2_mul(load_fe2(0u, i), load_fe2(1u, i)));
}

@compute @workgroup_size(64)
fn t_fq2_sqr(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i * 16u >= arrayLength(&out)) { return; }
    store_fe2(i, fq2_sqr(load_fe2(0u, i)));
}

@compute @workgroup_size(64)
fn t_fq2_mul_by_nonresidue(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i * 16u >= arrayLength(&out)) { return; }
    store_fe2(i, fq2_mul_by_nonresidue(load_fe2(0u, i)));
}
