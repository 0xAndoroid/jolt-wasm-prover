// 2D-GLV G1 folds (jolt-core parity: vector_add_scalar_mul_g1_online /
// fixed_scalar_mul_vs_then_add use the same decomposition): the shared
// challenge scalar is CPU-decomposed into k = ±|k1| ± |k2|·λ with
// |ki| < 2^128, and each thread runs a 2-point Shamir ladder over
// (P, φ(P)) with φ(x, y, z) = (β·x, y, z) — half the doublings of the
// 254-bit ladder. The negate mask folds both GLV signs and the fork's
// sign conventions (CPU-side).
//
// Requires GLV_BETA (Fq, Montgomery) from the header.

struct GlvParams {
    n: u32,
    v_offset: u32,
    a_offset: u32,
    out_offset: u32,
    max_bits: u32,
    negate: u32,
    _p0: u32,
    _p1: u32,
    k1: vec4<u32>,
    k2: vec4<u32>,
    k3: vec4<u32>,
    k4: vec4<u32>,
}

@group(0) @binding(0) var<uniform> g1_glv_params: GlvParams;
@group(0) @binding(1) var<storage, read_write> g1_glv_v: array<u32>;
@group(0) @binding(2) var<storage, read> g1_glv_bases: array<u32>;

fn g1_glv_kbit(k: vec4<u32>, b: u32) -> bool {
    return ((k[b >> 5u] >> (b & 31u)) & 1u) == 1u;
}

fn g1_glv_load_fe(buf_base: u32) -> Fe {
    var w: Fe8;
    for (var t = 0u; t < 8u; t++) {
        w[t] = g1_glv_v[buf_base + t];
    }
    return fe_unpack(w);
}

fn g1_glv_load_point(idx: u32) -> g1_Point {
    let base = idx * 24u;
    return g1_Point(
        g1_glv_load_fe(base),
        g1_glv_load_fe(base + 8u),
        g1_glv_load_fe(base + 16u),
    );
}

fn g1_glv_store_point(idx: u32, p: g1_Point) {
    let base = idx * 24u;
    var w = fe_pack(p.x);
    for (var t = 0u; t < 8u; t++) {
        g1_glv_v[base + t] = w[t];
    }
    w = fe_pack(p.y);
    for (var t = 0u; t < 8u; t++) {
        g1_glv_v[base + 8u + t] = w[t];
    }
    w = fe_pack(p.z);
    for (var t = 0u; t < 8u; t++) {
        g1_glv_v[base + 16u + t] = w[t];
    }
}

fn g1_glv_load_base_fe(buf_base: u32) -> Fe {
    var w: Fe8;
    for (var t = 0u; t < 8u; t++) {
        w[t] = g1_glv_bases[buf_base + t];
    }
    return fe_unpack(w);
}

fn g1_glv_shamir(p1: g1_Point, p2: g1_Point) -> g1_Point {
    let gp = g1_glv_params;
    var acc = g1_point_identity();
    for (var t = 0u; t < gp.max_bits; t++) {
        let b = gp.max_bits - 1u - t;
        acc = g1_point_double(acc);
        if (g1_glv_kbit(gp.k1, b)) {
            acc = g1_point_add(acc, p1);
        }
        if (g1_glv_kbit(gp.k2, b)) {
            acc = g1_point_add(acc, p2);
        }
    }
    return acc;
}

// v[out + i] = k * v[v_offset + i] + v[a_offset + i]
@compute @workgroup_size(64)
fn glv_scale_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let gp = g1_glv_params;
    if (i >= gp.n) {
        return;
    }
    let p = g1_glv_load_point(gp.v_offset + i);
    let a = g1_glv_load_point(gp.a_offset + i);

    var p1 = p;
    p1.y = fe_select((gp.negate & 1u) == 1u, fe_neg(p1.y), p1.y);
    var p2 = g1_Point(fe_mont_mul(p.x, GLV_BETA), p.y, p.z);
    p2.y = fe_select((gp.negate & 2u) == 2u, fe_neg(p2.y), p2.y);

    let acc = g1_glv_shamir(p1, p2);
    g1_glv_store_point(gp.out_offset + i, g1_point_add(acc, a));
}

// v[out + i] = v[v_offset + i] + k * bases[a_offset + i]  (affine bases,
// never the identity)
@compute @workgroup_size(64)
fn glv_add_scaled_base(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let gp = g1_glv_params;
    if (i >= gp.n) {
        return;
    }
    let base = (gp.a_offset + i) * 16u;
    let bx = g1_glv_load_base_fe(base);
    let by = g1_glv_load_base_fe(base + 8u);

    var p1 = g1_Point(bx, by, fe_mont_one());
    p1.y = fe_select((gp.negate & 1u) == 1u, fe_neg(p1.y), p1.y);
    var p2 = g1_Point(fe_mont_mul(bx, GLV_BETA), by, fe_mont_one());
    p2.y = fe_select((gp.negate & 2u) == 2u, fe_neg(p2.y), p2.y);

    let acc = g1_glv_shamir(p1, p2);
    let v = g1_glv_load_point(gp.v_offset + i);
    g1_glv_store_point(gp.out_offset + i, g1_point_add(acc, v));
}
