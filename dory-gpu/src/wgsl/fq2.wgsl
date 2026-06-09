// Fq2 = Fq[u]/(u^2 + 1) for BN254. Builds on field.wgsl (Fe = Fq).

struct Fe2 {
    c0: Fe,
    c1: Fe,
}

fn fq2_zero() -> Fe2 {
    return Fe2(fe_zero(), fe_zero());
}

fn fq2_mont_one() -> Fe2 {
    return Fe2(fe_mont_one(), fe_zero());
}

fn fq2_is_zero(a: Fe2) -> bool {
    return fe_is_zero(a.c0) && fe_is_zero(a.c1);
}

fn fq2_eq(a: Fe2, b: Fe2) -> bool {
    return fe_eq(a.c0, b.c0) && fe_eq(a.c1, b.c1);
}

fn fq2_select(c: bool, a: Fe2, b: Fe2) -> Fe2 {
    return Fe2(fe_select(c, a.c0, b.c0), fe_select(c, a.c1, b.c1));
}

fn fq2_add(a: Fe2, b: Fe2) -> Fe2 {
    return Fe2(fe_add(a.c0, b.c0), fe_add(a.c1, b.c1));
}

fn fq2_sub(a: Fe2, b: Fe2) -> Fe2 {
    return Fe2(fe_sub(a.c0, b.c0), fe_sub(a.c1, b.c1));
}

fn fq2_neg(a: Fe2) -> Fe2 {
    return Fe2(fe_neg(a.c0), fe_neg(a.c1));
}

fn fq2_double(a: Fe2) -> Fe2 {
    return Fe2(fe_add(a.c0, a.c0), fe_add(a.c1, a.c1));
}

// Karatsuba: (a0 + a1 u)(b0 + b1 u) = (a0 b0 - a1 b1) + ((a0+a1)(b0+b1) - a0 b0 - a1 b1) u
fn fq2_mul(a: Fe2, b: Fe2) -> Fe2 {
    let v0 = fe_mont_mul(a.c0, b.c0);
    let v1 = fe_mont_mul(a.c1, b.c1);
    let s = fe_mont_mul(fe_add(a.c0, a.c1), fe_add(b.c0, b.c1));
    return Fe2(fe_sub(v0, v1), fe_sub(fe_sub(s, v0), v1));
}

// (a0 + a1 u)^2 = (a0+a1)(a0-a1) + (2 a0 a1) u
fn fq2_sqr(a: Fe2) -> Fe2 {
    let p = fe_mont_mul(fe_add(a.c0, a.c1), fe_sub(a.c0, a.c1));
    let m = fe_mont_mul(a.c0, a.c1);
    return Fe2(p, fe_add(m, m));
}

fn fq2_mul_by_fq(a: Fe2, b: Fe) -> Fe2 {
    return Fe2(fe_mont_mul(a.c0, b), fe_mont_mul(a.c1, b));
}

// Multiplication by the Fq6/Fq12 tower non-residue xi = 9 + u:
// (9 c0 - c1) + (c0 + 9 c1) u
fn fq2_mul_by_nonresidue(a: Fe2) -> Fe2 {
    let t0 = fe_mul9(a.c0);
    let t1 = fe_mul9(a.c1);
    return Fe2(fe_sub(t0, a.c1), fe_add(a.c0, t1));
}

fn fq2_conjugate(a: Fe2) -> Fe2 {
    return Fe2(a.c0, fe_neg(a.c1));
}
