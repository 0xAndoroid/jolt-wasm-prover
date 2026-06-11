// 4D-GLV G2 folds (jolt-core parity: the glv_four / dory_g2 routines): the
// shared challenge is CPU-decomposed into k = Σ ±|ki|·λ^(i-1) with
// |ki| ~ 2^66, and each thread runs a 4-point Shamir ladder over
// ψ^0..3(P) — a quarter of the doublings of the 254-bit ladder. ψ^k is
// recomputed lazily per taken add (conjugation + one Fq2 mul per
// coordinate) to keep registers low. The negate mask folds the GLV signs
// (CPU-side, fork conventions normalized).
//
// Requires PSI{1,2,3}{X,Y}_C{0,1} (Fq2 Frobenius coefficients, Montgomery)
// from the header.

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

@group(0) @binding(0) var<uniform> g2_glv_params: GlvParams;
@group(0) @binding(1) var<storage, read_write> g2_glv_v: array<u32>;
@group(0) @binding(2) var<storage, read> g2_glv_bases: array<u32>;

fn g2_glv_kbit(k: vec4<u32>, b: u32) -> bool {
    return ((k[b >> 5u] >> (b & 31u)) & 1u) == 1u;
}

fn g2_glv_load_fe2(buf_base: u32) -> Fe2 {
    var w0: Fe8;
    var w1: Fe8;
    for (var t = 0u; t < 8u; t++) {
        w0[t] = g2_glv_v[buf_base + t];
        w1[t] = g2_glv_v[buf_base + 8u + t];
    }
    return Fe2(fe_unpack(w0), fe_unpack(w1));
}

fn g2_glv_load_point(idx: u32) -> g2_Point {
    let base = idx * 48u;
    return g2_Point(
        g2_glv_load_fe2(base),
        g2_glv_load_fe2(base + 16u),
        g2_glv_load_fe2(base + 32u),
    );
}

fn g2_glv_store_fe2(buf_base: u32, v: Fe2) {
    let w0 = fe_pack(v.c0);
    let w1 = fe_pack(v.c1);
    for (var t = 0u; t < 8u; t++) {
        g2_glv_v[buf_base + t] = w0[t];
        g2_glv_v[buf_base + 8u + t] = w1[t];
    }
}

fn g2_glv_store_point(idx: u32, p: g2_Point) {
    let base = idx * 48u;
    g2_glv_store_fe2(base, p.x);
    g2_glv_store_fe2(base + 16u, p.y);
    g2_glv_store_fe2(base + 32u, p.z);
}

fn g2_glv_load_base_fe2(buf_base: u32) -> Fe2 {
    var w0: Fe8;
    var w1: Fe8;
    for (var t = 0u; t < 8u; t++) {
        w0[t] = g2_glv_bases[buf_base + t];
        w1[t] = g2_glv_bases[buf_base + 8u + t];
    }
    return Fe2(fe_unpack(w0), fe_unpack(w1));
}

// psi^k of a projective point (frobenius_psi_power_projective): odd k
// conjugates all coordinates; x *= psiK_coef2, y *= psiK_coef3. The negate
// flag flips y.
fn g2_glv_psi(p: g2_Point, k: u32, negate: bool) -> g2_Point {
    var x = p.x;
    var y = p.y;
    var z = p.z;
    if ((k & 1u) == 1u) {
        x = fq2_conjugate(x);
        y = fq2_conjugate(y);
        z = fq2_conjugate(z);
    }
    if (k == 1u) {
        x = fq2_mul(x, Fe2(PSI1X_C0, PSI1X_C1));
        y = fq2_mul(y, Fe2(PSI1Y_C0, PSI1Y_C1));
    } else if (k == 2u) {
        x = fq2_mul(x, Fe2(PSI2X_C0, PSI2X_C1));
        y = fq2_mul(y, Fe2(PSI2Y_C0, PSI2Y_C1));
    } else if (k == 3u) {
        x = fq2_mul(x, Fe2(PSI3X_C0, PSI3X_C1));
        y = fq2_mul(y, Fe2(PSI3Y_C0, PSI3Y_C1));
    }
    y = fq2_select(negate, fq2_neg(y), y);
    return g2_Point(x, y, z);
}

fn g2_glv_shamir(p: g2_Point) -> g2_Point {
    let gp = g2_glv_params;
    var acc = g2_point_identity();
    for (var t = 0u; t < gp.max_bits; t++) {
        let b = gp.max_bits - 1u - t;
        acc = g2_point_double(acc);
        if (g2_glv_kbit(gp.k1, b)) {
            acc = g2_point_add(acc, g2_glv_psi(p, 0u, (gp.negate & 1u) == 1u));
        }
        if (g2_glv_kbit(gp.k2, b)) {
            acc = g2_point_add(acc, g2_glv_psi(p, 1u, (gp.negate & 2u) == 2u));
        }
        if (g2_glv_kbit(gp.k3, b)) {
            acc = g2_point_add(acc, g2_glv_psi(p, 2u, (gp.negate & 4u) == 4u));
        }
        if (g2_glv_kbit(gp.k4, b)) {
            acc = g2_point_add(acc, g2_glv_psi(p, 3u, (gp.negate & 8u) == 8u));
        }
    }
    return acc;
}

// v[out + i] = k * v[v_offset + i] + v[a_offset + i]
@compute @workgroup_size(64)
fn glv_scale_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let gp = g2_glv_params;
    if (i >= gp.n) {
        return;
    }
    let p = g2_glv_load_point(gp.v_offset + i);
    let a = g2_glv_load_point(gp.a_offset + i);
    let acc = g2_glv_shamir(p);
    g2_glv_store_point(gp.out_offset + i, g2_point_add(acc, a));
}

// v[out + i] = v[v_offset + i] + k * bases[a_offset + i]  (affine bases,
// never the identity)
@compute @workgroup_size(64)
fn glv_add_scaled_base(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let gp = g2_glv_params;
    if (i >= gp.n) {
        return;
    }
    let base = (gp.a_offset + i) * 32u;
    let p = g2_Point(
        g2_glv_load_base_fe2(base),
        g2_glv_load_base_fe2(base + 16u),
        fq2_mont_one(),
    );
    let acc = g2_glv_shamir(p);
    let v = g2_glv_load_point(gp.v_offset + i);
    g2_glv_store_point(gp.out_offset + i, g2_point_add(acc, v));
}
