// BN254 optimal-ate Miller loop, split into small per-step kernels driven by
// a host-side dispatch schedule (the ate digit sequence is fixed, so the
// whole loop encodes into one command buffer with no readbacks).
//
// Formulas ported verbatim from arkworks (bn/g2.rs homogeneous projective
// doubling/addition steps, fp12 mul_by_034, quadratic-ext squaring), twist
// type D. P-side inputs are affine, so f matches arkworks' multi_miller_loop
// bit for bit.
//
// Pair layouts (packed u32 words): f = 96 (two Fe6), T = 48 (Fq2 x,y,z),
// q affine = 32, p affine = 16, line = 48 (three Fq2), fq6 scratch = 48.
//
// Required header constants: FQ_TWO_INV (Fe), G2_3B_C0/C1 (3*b' twist coeff),
// TWQX_C0/C1, TWQY_C0/C1 (psi twist constants, Montgomery).

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

@group(0) @binding(0) var<uniform> pp: PairParams;
@group(0) @binding(1) var<storage, read_write> pair_f: array<u32>;
@group(0) @binding(2) var<storage, read_write> pair_t: array<u32>;
@group(0) @binding(3) var<storage, read> pair_q: array<u32>;
@group(0) @binding(4) var<storage, read_write> pair_lines: array<u32>;
@group(0) @binding(5) var<storage, read> pair_p: array<u32>;
@group(0) @binding(6) var<storage, read_write> pair_scratch: array<u32>;
@group(0) @binding(7) var<storage, read_write> pair_q12: array<u32>;

fn pq_fe2(base: u32) -> Fe2 {
    var w0: Fe8;
    var w1: Fe8;
    for (var k = 0u; k < 8u; k++) {
        w0[k] = pair_q[base + k];
        w1[k] = pair_q[base + 8u + k];
    }
    return Fe2(fe_unpack(w0), fe_unpack(w1));
}

fn pq12_fe2(base: u32) -> Fe2 {
    var w0: Fe8;
    var w1: Fe8;
    for (var k = 0u; k < 8u; k++) {
        w0[k] = pair_q12[base + k];
        w1[k] = pair_q12[base + 8u + k];
    }
    return Fe2(fe_unpack(w0), fe_unpack(w1));
}

fn pt_fe2_load(base: u32) -> Fe2 {
    var w0: Fe8;
    var w1: Fe8;
    for (var k = 0u; k < 8u; k++) {
        w0[k] = pair_t[base + k];
        w1[k] = pair_t[base + 8u + k];
    }
    return Fe2(fe_unpack(w0), fe_unpack(w1));
}

fn pt_fe2_store(base: u32, v: Fe2) {
    let w0 = fe_pack(v.c0);
    let w1 = fe_pack(v.c1);
    for (var k = 0u; k < 8u; k++) {
        pair_t[base + k] = w0[k];
        pair_t[base + 8u + k] = w1[k];
    }
}

fn line_store(base: u32, v: Fe2) {
    let w0 = fe_pack(v.c0);
    let w1 = fe_pack(v.c1);
    for (var k = 0u; k < 8u; k++) {
        pair_lines[base + k] = w0[k];
        pair_lines[base + 8u + k] = w1[k];
    }
}

fn q12_store(base: u32, v: Fe2) {
    let w0 = fe_pack(v.c0);
    let w1 = fe_pack(v.c1);
    for (var k = 0u; k < 8u; k++) {
        pair_q12[base + k] = w0[k];
        pair_q12[base + 8u + k] = w1[k];
    }
}

// T <- (q.x, q.y, 1) for each pair.
@compute @workgroup_size(64)
fn pair_init(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= pp.n_pairs) {
        return;
    }
    let qx = pq_fe2(i * 32u);
    let qy = pq_fe2(i * 32u + 16u);
    pt_fe2_store(i * 48u, qx);
    pt_fe2_store(i * 48u + 16u, qy);
    pt_fe2_store(i * 48u + 32u, fq2_mont_one());
}

// arkworks homogeneous-projective doubling step (twist D): updates T, writes the
// line (-h, 3j, i).
@compute @workgroup_size(64)
fn pair_dbl_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= pp.n_pairs) {
        return;
    }
    let tx = pt_fe2_load(idx * 48u);
    let ty = pt_fe2_load(idx * 48u + 16u);
    let tz = pt_fe2_load(idx * 48u + 32u);

    var a = fq2_mul(tx, ty);
    a = fq2_mul_by_fq(a, FQ_TWO_INV);
    let b = fq2_sqr(ty);
    let c = fq2_sqr(tz);
    // e = b' * 3c = (3b') * c
    let e = fq2_mul(Fe2(G2_3B_C0, G2_3B_C1), c);
    let f = fq2_add(fq2_double(e), e);
    var g = fq2_add(b, f);
    g = fq2_mul_by_fq(g, FQ_TWO_INV);
    let h = fq2_sub(fq2_sqr(fq2_add(ty, tz)), fq2_add(b, c));
    let ii = fq2_sub(e, b);
    let j = fq2_sqr(tx);
    let e2 = fq2_sqr(e);

    pt_fe2_store(idx * 48u, fq2_mul(a, fq2_sub(b, f)));
    pt_fe2_store(idx * 48u + 16u, fq2_sub(fq2_sqr(g), fq2_add(fq2_double(e2), e2)));
    pt_fe2_store(idx * 48u + 32u, fq2_mul(b, h));

    line_store(idx * 48u, fq2_neg(h));
    line_store(idx * 48u + 16u, fq2_add(fq2_double(j), j));
    line_store(idx * 48u + 32u, ii);
}

// arkworks homogeneous-projective addition step (twist D): line (lambda, -theta, j).
// q_source selects q / -q / q1 / q2.
@compute @workgroup_size(64)
fn pair_add_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= pp.n_pairs) {
        return;
    }
    var qx: Fe2;
    var qy: Fe2;
    if (pp.q_source < 2u) {
        qx = pq_fe2(idx * 32u);
        qy = pq_fe2(idx * 32u + 16u);
        qy = fq2_select(pp.q_source == 1u, fq2_neg(qy), qy);
    } else if (pp.q_source == 2u) {
        qx = pq12_fe2(idx * 64u);
        qy = pq12_fe2(idx * 64u + 16u);
    } else {
        qx = pq12_fe2(idx * 64u + 32u);
        qy = pq12_fe2(idx * 64u + 48u);
    }

    let tx = pt_fe2_load(idx * 48u);
    let ty = pt_fe2_load(idx * 48u + 16u);
    let tz = pt_fe2_load(idx * 48u + 32u);

    let theta = fq2_sub(ty, fq2_mul(qy, tz));
    let lambda = fq2_sub(tx, fq2_mul(qx, tz));
    let c = fq2_sqr(theta);
    let d = fq2_sqr(lambda);
    let e = fq2_mul(lambda, d);
    let f = fq2_mul(tz, c);
    let g = fq2_mul(tx, d);
    let h = fq2_sub(fq2_add(e, f), fq2_double(g));

    pt_fe2_store(idx * 48u, fq2_mul(lambda, h));
    pt_fe2_store(idx * 48u + 16u, fq2_sub(fq2_mul(theta, fq2_sub(g, h)), fq2_mul(e, ty)));
    pt_fe2_store(idx * 48u + 32u, fq2_mul(tz, e));

    let j = fq2_sub(fq2_mul(theta, qx), fq2_mul(lambda, qy));
    line_store(idx * 48u, lambda);
    line_store(idx * 48u + 16u, fq2_neg(theta));
    line_store(idx * 48u + 32u, j);
}

// q1 = psi(q), q2 = -psi^2(q): Fq2 conjugation + twist constants
// (arkworks mul_by_char); written to pair_q12 as two affine points per pair.
@compute @workgroup_size(64)
fn pair_frob(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= pp.n_pairs) {
        return;
    }
    let qx = pq_fe2(idx * 32u);
    let qy = pq_fe2(idx * 32u + 16u);
    let twqx = Fe2(TWQX_C0, TWQX_C1);
    let twqy = Fe2(TWQY_C0, TWQY_C1);

    let q1x = fq2_mul(fq2_conjugate(qx), twqx);
    let q1y = fq2_mul(fq2_conjugate(qy), twqy);
    let q2x = fq2_mul(fq2_conjugate(q1x), twqx);
    // q2.y = -(conj(q1.y) * twqy)
    let q2y = fq2_neg(fq2_mul(fq2_conjugate(q1y), twqy));

    q12_store(idx * 64u, q1x);
    q12_store(idx * 64u + 16u, q1y);
    q12_store(idx * 64u + 32u, q2x);
    q12_store(idx * 64u + 48u, q2y);
}
