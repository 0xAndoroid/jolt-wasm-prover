// Short Weierstrass curve ops (a = 0) over point field PF_Fe, in homogeneous
// projective coordinates, using the complete formulas of Renes-Costello-Batina
// (eprint 2015/1060, Algorithms 7-9). Complete = branch-free and correct for
// identity / doubling inputs, which keeps GPU lanes divergence-free.
//
// Template: PF_* identifiers are substituted per instantiation (G1: Fq, G2: Fq2).
// Requires PF_mul_by_3b to be defined by the instantiation.

struct PF_Point {
    x: PF_Fe,
    y: PF_Fe,
    z: PF_Fe,
}

struct PF_Affine {
    x: PF_Fe,
    y: PF_Fe,
}

fn PF_point_identity() -> PF_Point {
    return PF_Point(PF_zero(), PF_one(), PF_zero());
}

fn PF_point_is_identity(p: PF_Point) -> bool {
    return PF_is_zero(p.z);
}

fn PF_point_select(c: bool, a: PF_Point, b: PF_Point) -> PF_Point {
    return PF_Point(
        PF_select(c, a.x, b.x),
        PF_select(c, a.y, b.y),
        PF_select(c, a.z, b.z),
    );
}

fn PF_point_neg(p: PF_Point) -> PF_Point {
    return PF_Point(p.x, PF_neg(p.y), p.z);
}

// Complete projective addition (RCB Algorithm 7).
fn PF_point_add(p: PF_Point, q: PF_Point) -> PF_Point {
    var t0 = PF_mul(p.x, q.x);
    var t1 = PF_mul(p.y, q.y);
    var t2 = PF_mul(p.z, q.z);
    var t3 = PF_add(p.x, p.y);
    var t4 = PF_add(q.x, q.y);
    t3 = PF_mul(t3, t4);
    t4 = PF_add(t0, t1);
    t3 = PF_sub(t3, t4);
    t4 = PF_add(p.y, p.z);
    var x3 = PF_add(q.y, q.z);
    t4 = PF_mul(t4, x3);
    x3 = PF_add(t1, t2);
    t4 = PF_sub(t4, x3);
    x3 = PF_add(p.x, p.z);
    var y3 = PF_add(q.x, q.z);
    x3 = PF_mul(x3, y3);
    y3 = PF_add(t0, t2);
    y3 = PF_sub(x3, y3);
    x3 = PF_add(t0, t0);
    t0 = PF_add(x3, t0);
    t2 = PF_mul_by_3b(t2);
    var z3 = PF_add(t1, t2);
    t1 = PF_sub(t1, t2);
    y3 = PF_mul_by_3b(y3);
    x3 = PF_mul(t4, y3);
    t2 = PF_mul(t3, t1);
    x3 = PF_sub(t2, x3);
    y3 = PF_mul(y3, t0);
    t1 = PF_mul(t1, z3);
    y3 = PF_add(t1, y3);
    t0 = PF_mul(t0, t3);
    z3 = PF_mul(z3, t4);
    z3 = PF_add(z3, t0);
    return PF_Point(x3, y3, z3);
}

// Complete mixed addition (RCB Algorithm 8). q must not be the identity.
fn PF_point_madd(p: PF_Point, q: PF_Affine) -> PF_Point {
    var t0 = PF_mul(p.x, q.x);
    var t1 = PF_mul(p.y, q.y);
    var t3 = PF_add(q.x, q.y);
    var t4 = PF_add(p.x, p.y);
    t3 = PF_mul(t3, t4);
    t4 = PF_add(t0, t1);
    t3 = PF_sub(t3, t4);
    t4 = PF_mul(q.x, p.z);
    t4 = PF_add(t4, p.x);
    var t5 = PF_mul(q.y, p.z);
    t5 = PF_add(t5, p.y);
    var x3 = PF_add(t0, t0);
    t0 = PF_add(x3, t0);
    var t2 = PF_mul_by_3b(p.z);
    var z3 = PF_add(t1, t2);
    t1 = PF_sub(t1, t2);
    var y3 = PF_mul_by_3b(t4);
    x3 = PF_mul(t5, y3);
    t2 = PF_mul(t3, t1);
    x3 = PF_sub(t2, x3);
    y3 = PF_mul(y3, t0);
    t1 = PF_mul(t1, z3);
    y3 = PF_add(t1, y3);
    t0 = PF_mul(t0, t3);
    z3 = PF_mul(z3, t5);
    z3 = PF_add(z3, t0);
    return PF_Point(x3, y3, z3);
}

// Complete doubling (RCB Algorithm 9).
fn PF_point_double(p: PF_Point) -> PF_Point {
    var t0 = PF_mul(p.y, p.y);
    var z3 = PF_add(t0, t0);
    z3 = PF_add(z3, z3);
    z3 = PF_add(z3, z3);
    var t1 = PF_mul(p.y, p.z);
    var t2 = PF_mul(p.z, p.z);
    t2 = PF_mul_by_3b(t2);
    var x3 = PF_mul(t2, z3);
    var y3 = PF_add(t0, t2);
    z3 = PF_mul(t1, z3);
    t1 = PF_add(t2, t2);
    t2 = PF_add(t1, t2);
    t0 = PF_sub(t0, t2);
    y3 = PF_mul(t0, y3);
    y3 = PF_add(x3, y3);
    t1 = PF_mul(p.x, p.y);
    x3 = PF_mul(t0, t1);
    x3 = PF_add(x3, x3);
    return PF_Point(x3, y3, z3);
}
