// Fq12 side of the Miller loop: f-state init, squaring, sparse line
// application (mul_by_034) and per-product reduction. Every kernel contains
// at most one full Fq6 multiplication so the Apple Metal compiler copes.
//
// f layout: 96 words = [c0.c0, c0.c1, c0.c2, c1.c0, c1.c1, c1.c2], 16 each.
// scratch layout per pair: 144 words = a (48) | b (48) | scaled line (48).

struct PairParams {
    n_pairs: u32,
    line_source: u32,
    step: u32,
    prep_stride: u32,
    q_source: u32,
    segment: u32,
    stride: u32,
    check_q: u32,
    prep_mod: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<uniform> fp: PairParams;
@group(0) @binding(1) var<storage, read_write> fq_f: array<u32>;
@group(0) @binding(2) var<storage, read_write> fq_mask: array<u32>;
@group(0) @binding(3) var<storage, read> fq_q: array<u32>;
@group(0) @binding(4) var<storage, read> fq_lines: array<u32>;
@group(0) @binding(5) var<storage, read> fq_p: array<u32>;
@group(0) @binding(6) var<storage, read_write> fq_scratch: array<u32>;
@group(0) @binding(7) var<storage, read> fq_prepared: array<u32>;

fn f_fe2_load(base: u32) -> Fe2 {
    var w0: Fe8;
    var w1: Fe8;
    for (var k = 0u; k < 8u; k++) {
        w0[k] = fq_f[base + k];
        w1[k] = fq_f[base + 8u + k];
    }
    return Fe2(fe_unpack(w0), fe_unpack(w1));
}

fn f_fe2_store(base: u32, v: Fe2) {
    let w0 = fe_pack(v.c0);
    let w1 = fe_pack(v.c1);
    for (var k = 0u; k < 8u; k++) {
        fq_f[base + k] = w0[k];
        fq_f[base + 8u + k] = w1[k];
    }
}

fn f_fe6_load(base: u32) -> Fe6 {
    return Fe6(f_fe2_load(base), f_fe2_load(base + 16u), f_fe2_load(base + 32u));
}

fn f_fe6_store(base: u32, v: Fe6) {
    f_fe2_store(base, v.c0);
    f_fe2_store(base + 16u, v.c1);
    f_fe2_store(base + 32u, v.c2);
}

fn sc_fe2_load(base: u32) -> Fe2 {
    var w0: Fe8;
    var w1: Fe8;
    for (var k = 0u; k < 8u; k++) {
        w0[k] = fq_scratch[base + k];
        w1[k] = fq_scratch[base + 8u + k];
    }
    return Fe2(fe_unpack(w0), fe_unpack(w1));
}

fn sc_fe2_store(base: u32, v: Fe2) {
    let w0 = fe_pack(v.c0);
    let w1 = fe_pack(v.c1);
    for (var k = 0u; k < 8u; k++) {
        fq_scratch[base + k] = w0[k];
        fq_scratch[base + 8u + k] = w1[k];
    }
}

fn sc_fe6_load(base: u32) -> Fe6 {
    return Fe6(sc_fe2_load(base), sc_fe2_load(base + 16u), sc_fe2_load(base + 32u));
}

fn sc_fe6_store(base: u32, v: Fe6) {
    sc_fe2_store(base, v.c0);
    sc_fe2_store(base + 16u, v.c1);
    sc_fe2_store(base + 32u, v.c2);
}

fn p_fe_load(base: u32) -> Fe {
    var w: Fe8;
    for (var k = 0u; k < 8u; k++) {
        w[k] = fq_p[base + k];
    }
    return fe_unpack(w);
}

fn line_fe2(idx: u32, comp: u32) -> Fe2 {
    var base: u32;
    if (fp.line_source == 0u) {
        base = idx * 48u + comp * 16u;
    } else {
        base = ((idx % fp.prep_mod) * fp.prep_stride + fp.step) * 48u + comp * 16u;
    }
    var w0: Fe8;
    var w1: Fe8;
    for (var k = 0u; k < 8u; k++) {
        if (fp.line_source == 0u) {
            w0[k] = fq_lines[base + k];
            w1[k] = fq_lines[base + 8u + k];
        } else {
            w0[k] = fq_prepared[base + k];
            w1[k] = fq_prepared[base + 8u + k];
        }
    }
    return Fe2(fe_unpack(w0), fe_unpack(w1));
}

// f = 1; mask = (p == identity) || (check_q && q == identity).
@compute @workgroup_size(64)
fn f_init(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= fp.n_pairs) {
        return;
    }
    f_fe6_store(i * 96u, Fe6(fq2_mont_one(), fq2_zero(), fq2_zero()));
    f_fe6_store(i * 96u + 48u, fq6_zero());

    let px = p_fe_load(i * 16u);
    let py = p_fe_load(i * 16u + 8u);
    var masked = fe_is_zero(px) && fe_is_zero(py);
    if (fp.check_q == 1u) {
        var qz = true;
        for (var k = 0u; k < 32u; k++) {
            qz = qz && (fq_q[i * 32u + k] == 0u);
        }
        masked = masked || qz;
    }
    fq_mask[i] = u32(masked);
}

// Squaring part 1: scratch.a = (c0 - c1) * (c0 - v*c1)   (one fq6 mul)
@compute @workgroup_size(64)
fn f_sqr_a(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= fp.n_pairs || fq_mask[i] == 1u) {
        return;
    }
    let c0 = f_fe6_load(i * 96u);
    let c1 = f_fe6_load(i * 96u + 48u);
    let v0 = fq6_mul(fq6_sub(c0, c1), fq6_sub(c0, fq6_mul_by_nonresidue(c1)));
    sc_fe6_store(i * 144u, v0);
}

// Squaring part 2: v2 = c0*c1; c1' = 2 v2; c0' = v0 + v*v2 + v2.
@compute @workgroup_size(64)
fn f_sqr_b(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= fp.n_pairs || fq_mask[i] == 1u) {
        return;
    }
    let c0 = f_fe6_load(i * 96u);
    let c1 = f_fe6_load(i * 96u + 48u);
    let v0 = sc_fe6_load(i * 144u);
    let v2 = fq6_mul(c0, c1);
    f_fe6_store(i * 96u + 48u, fq6_double(v2));
    f_fe6_store(i * 96u, fq6_add(fq6_add(v0, fq6_mul_by_nonresidue(v2)), v2));
}

// mul_by_034 part 1: scale the line by P, compute a = c0 * f.c0 (component
// wise) and b = f.c1.mul_by_01(c3, c4); stash a, b and the scaled line.
@compute @workgroup_size(64)
fn f_apply_a(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= fp.n_pairs || fq_mask[i] == 1u) {
        return;
    }
    let px = p_fe_load(i * 16u);
    let py = p_fe_load(i * 16u + 8u);
    let c0 = fq2_mul_by_fq(line_fe2(i, 0u), py);
    let c3 = fq2_mul_by_fq(line_fe2(i, 1u), px);
    let c4 = line_fe2(i, 2u);

    let f0 = f_fe6_load(i * 96u);
    let f1 = f_fe6_load(i * 96u + 48u);
    let a = Fe6(fq2_mul(f0.c0, c0), fq2_mul(f0.c1, c0), fq2_mul(f0.c2, c0));
    let b = fq6_mul_by_01(f1, c3, c4);

    sc_fe6_store(i * 144u, a);
    sc_fe6_store(i * 144u + 48u, b);
    sc_fe2_store(i * 144u + 96u, c0);
    sc_fe2_store(i * 144u + 112u, c3);
    sc_fe2_store(i * 144u + 128u, c4);
}

// mul_by_034 part 2: e = (f.c0 + f.c1).mul_by_01(c0 + c3, c4);
// f.c1 = e - (a + b); f.c0 = v*b + a.
@compute @workgroup_size(64)
fn f_apply_b(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= fp.n_pairs || fq_mask[i] == 1u) {
        return;
    }
    let f0 = f_fe6_load(i * 96u);
    let f1 = f_fe6_load(i * 96u + 48u);
    let a = sc_fe6_load(i * 144u);
    let b = sc_fe6_load(i * 144u + 48u);
    let c0 = sc_fe2_load(i * 144u + 96u);
    let c3 = sc_fe2_load(i * 144u + 112u);
    let c4 = sc_fe2_load(i * 144u + 128u);

    let e = fq6_mul_by_01(fq6_add(f0, f1), fq2_add(c0, c3), c4);
    f_fe6_store(i * 96u + 48u, fq6_sub(e, fq6_add(a, b)));
    f_fe6_store(i * 96u, fq6_add(fq6_mul_by_nonresidue(b), a));
}

// Product reduction, three passes per halving step (full fq12 mul):
// dst = seg*segment + i, src = dst + stride (i < stride).
// red_a: scratch.a = dst.c0 * src.c0
// red_b: scratch.b = dst.c1 * src.c1
// red_c: c1' = (dst.c0+dst.c1)(src.c0+src.c1) - a - b; c0' = a + v*b.
fn reduce_dst(gid: vec3<u32>) -> u32 {
    return gid.y * fp.segment + gid.x;
}

@compute @workgroup_size(64)
fn f_red_a(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= fp.stride) {
        return;
    }
    let dst = reduce_dst(gid);
    let src = dst + fp.stride;
    sc_fe6_store(
        dst * 144u,
        fq6_mul(f_fe6_load(dst * 96u), f_fe6_load(src * 96u)),
    );
}

@compute @workgroup_size(64)
fn f_red_b(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= fp.stride) {
        return;
    }
    let dst = reduce_dst(gid);
    let src = dst + fp.stride;
    sc_fe6_store(
        dst * 144u + 48u,
        fq6_mul(f_fe6_load(dst * 96u + 48u), f_fe6_load(src * 96u + 48u)),
    );
}

@compute @workgroup_size(64)
fn f_red_c(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= fp.stride) {
        return;
    }
    let dst = reduce_dst(gid);
    let src = dst + fp.stride;
    let a0 = f_fe6_load(dst * 96u);
    let a1 = f_fe6_load(dst * 96u + 48u);
    let b0 = f_fe6_load(src * 96u);
    let b1 = f_fe6_load(src * 96u + 48u);
    let va = sc_fe6_load(dst * 144u);
    let vb = sc_fe6_load(dst * 144u + 48u);
    let e = fq6_mul(fq6_add(a0, a1), fq6_add(b0, b1));
    f_fe6_store(dst * 96u + 48u, fq6_sub(e, fq6_add(va, vb)));
    f_fe6_store(dst * 96u, fq6_add(va, fq6_mul_by_nonresidue(vb)));
}
