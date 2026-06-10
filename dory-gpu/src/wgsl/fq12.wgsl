// Fq6 = Fq2[v]/(v^3 - xi), xi = 9 + u, and helpers for the Fq12 tower
// (Fq12 = Fq6[w]/(w^2 - v)). Only what the Miller loop needs.

struct Fe6 {
    c0: Fe2,
    c1: Fe2,
    c2: Fe2,
}

fn fq6_zero() -> Fe6 {
    return Fe6(fq2_zero(), fq2_zero(), fq2_zero());
}

fn fq6_add(a: Fe6, b: Fe6) -> Fe6 {
    return Fe6(fq2_add(a.c0, b.c0), fq2_add(a.c1, b.c1), fq2_add(a.c2, b.c2));
}

fn fq6_sub(a: Fe6, b: Fe6) -> Fe6 {
    return Fe6(fq2_sub(a.c0, b.c0), fq2_sub(a.c1, b.c1), fq2_sub(a.c2, b.c2));
}

fn fq6_double(a: Fe6) -> Fe6 {
    return Fe6(fq2_double(a.c0), fq2_double(a.c1), fq2_double(a.c2));
}

// Multiplication by v: (c0, c1, c2) -> (xi*c2, c0, c1).
fn fq6_mul_by_nonresidue(a: Fe6) -> Fe6 {
    return Fe6(fq2_mul_by_nonresidue(a.c2), a.c0, a.c1);
}

// Full Karatsuba product (Devegili et al. section 4); 6 Fq2 muls.
fn fq6_mul(s: Fe6, o: Fe6) -> Fe6 {
    let ad = fq2_mul(s.c0, o.c0);
    let be = fq2_mul(s.c1, o.c1);
    let cf = fq2_mul(s.c2, o.c2);
    let x = fq2_sub(fq2_sub(fq2_mul(fq2_add(s.c1, s.c2), fq2_add(o.c1, o.c2)), be), cf);
    let y = fq2_sub(fq2_sub(fq2_mul(fq2_add(s.c0, s.c1), fq2_add(o.c0, o.c1)), ad), be);
    let z = fq2_sub(fq2_add(fq2_sub(fq2_mul(fq2_add(s.c0, s.c2), fq2_add(o.c0, o.c2)), ad), be), cf);
    return Fe6(
        fq2_add(ad, fq2_mul_by_nonresidue(x)),
        fq2_add(y, fq2_mul_by_nonresidue(cf)),
        z,
    );
}

// Sparse product with (c0, c1, 0); 5 Fq2 muls (arkworks fp6 mul_by_01).
fn fq6_mul_by_01(s: Fe6, c0: Fe2, c1: Fe2) -> Fe6 {
    let a_a = fq2_mul(s.c0, c0);
    let b_b = fq2_mul(s.c1, c1);
    let t1 = fq2_add(
        fq2_mul_by_nonresidue(fq2_sub(fq2_mul(c1, fq2_add(s.c1, s.c2)), b_b)),
        a_a,
    );
    let t3 = fq2_add(fq2_sub(fq2_mul(c0, fq2_add(s.c0, s.c2)), a_a), b_b);
    let t2 = fq2_sub(
        fq2_sub(fq2_mul(fq2_add(c0, c1), fq2_add(s.c0, s.c1)), a_a),
        b_b,
    );
    return Fe6(t1, t2, t3);
}
