// 256-bit prime field arithmetic in Montgomery form (R = 2^256).
//
// Limb scheme: WGSL has no 64-bit integers and no widening multiply, so field
// elements are 16 limbs of 16 bits (one per u32). 16x16-bit products fit a u32
// exactly, and the CIOS accumulation `t + a*b + c` with 16-bit t, c peaks at
// 2^32 - 1, so carries never overflow.
//
// Storage format is 8 packed u32 words — bit-identical to arkworks' Montgomery
// BigInt<4> serialized as little-endian u32 words, so host <-> GPU transfers
// are pure memcpy.
//
// Required header constants: FE_MOD (array<u32,16>), FE_NP (u32, -p^-1 mod 2^16),
// FE_MONT_ONE (array<u32,16>).

alias Fe = array<u32, 16>;
alias Fe8 = array<u32, 8>;

fn fe_unpack(w: Fe8) -> Fe {
    var r: Fe;
    for (var k = 0u; k < 8u; k++) {
        let v = w[k];
        r[2u * k] = v & 0xffffu;
        r[2u * k + 1u] = v >> 16u;
    }
    return r;
}

fn fe_pack(a: Fe) -> Fe8 {
    var r: Fe8;
    for (var k = 0u; k < 8u; k++) {
        r[k] = a[2u * k] | (a[2u * k + 1u] << 16u);
    }
    return r;
}

fn fe_zero() -> Fe {
    var r: Fe;
    return r;
}

fn fe_mont_one() -> Fe {
    return FE_MONT_ONE;
}

fn fe_is_zero(a: Fe) -> bool {
    var acc = 0u;
    for (var i = 0u; i < 16u; i++) {
        acc |= a[i];
    }
    return acc == 0u;
}

fn fe_eq(a: Fe, b: Fe) -> bool {
    var acc = 0u;
    for (var i = 0u; i < 16u; i++) {
        acc |= a[i] ^ b[i];
    }
    return acc == 0u;
}

fn fe_select(c: bool, a: Fe, b: Fe) -> Fe {
    var r: Fe;
    for (var i = 0u; i < 16u; i++) {
        r[i] = select(b[i], a[i], c);
    }
    return r;
}

// Reduces t (< 2p, with optional 2^256 carry bit `hi`) to [0, p).
fn fe_reduce_once(t: Fe, hi: u32) -> Fe {
    var s: Fe;
    var brw = 0u;
    for (var i = 0u; i < 16u; i++) {
        let d = t[i] - FE_MOD[i] - brw;
        s[i] = d & 0xffffu;
        brw = (d >> 16u) & 1u;
    }
    let use_sub = (hi != 0u) | (brw == 0u);
    return fe_select(use_sub, s, t);
}

fn fe_add(a: Fe, b: Fe) -> Fe {
    var t: Fe;
    var c = 0u;
    for (var i = 0u; i < 16u; i++) {
        let s = a[i] + b[i] + c;
        t[i] = s & 0xffffu;
        c = s >> 16u;
    }
    return fe_reduce_once(t, c);
}

fn fe_sub(a: Fe, b: Fe) -> Fe {
    var t: Fe;
    var brw = 0u;
    for (var i = 0u; i < 16u; i++) {
        let d = a[i] - b[i] - brw;
        t[i] = d & 0xffffu;
        brw = (d >> 16u) & 1u;
    }
    // If we borrowed, add p back.
    var r: Fe;
    var c = 0u;
    for (var i = 0u; i < 16u; i++) {
        let s = t[i] + FE_MOD[i] + c;
        r[i] = s & 0xffffu;
        c = s >> 16u;
    }
    return fe_select(brw == 1u, r, t);
}

fn fe_neg(a: Fe) -> Fe {
    var t: Fe;
    var brw = 0u;
    for (var i = 0u; i < 16u; i++) {
        let d = FE_MOD[i] - a[i] - brw;
        t[i] = d & 0xffffu;
        brw = (d >> 16u) & 1u;
    }
    return fe_select(fe_is_zero(a), fe_zero(), t);
}

fn fe_double(a: Fe) -> Fe {
    return fe_add(a, a);
}

// Montgomery multiplication: a * b * R^-1 mod p (CIOS).
fn fe_mont_mul(a: Fe, b: Fe) -> Fe {
    var t: array<u32, 18>;
    for (var i = 0u; i < 16u; i++) {
        let bi = b[i];
        var c = 0u;
        for (var j = 0u; j < 16u; j++) {
            let s = t[j] + a[j] * bi + c;
            t[j] = s & 0xffffu;
            c = s >> 16u;
        }
        let s16 = t[16] + c;
        t[16] = s16 & 0xffffu;
        t[17] = s16 >> 16u;

        let m = (t[0] * FE_NP) & 0xffffu;
        let s0 = t[0] + m * FE_MOD[0];
        c = s0 >> 16u;
        for (var j = 1u; j < 16u; j++) {
            let s = t[j] + m * FE_MOD[j] + c;
            t[j - 1u] = s & 0xffffu;
            c = s >> 16u;
        }
        let s16b = t[16] + c;
        t[15] = s16b & 0xffffu;
        t[16] = t[17] + (s16b >> 16u);
    }
    var lo: Fe;
    for (var i = 0u; i < 16u; i++) {
        lo[i] = t[i];
    }
    return fe_reduce_once(lo, t[16]);
}

fn fe_sqr(a: Fe) -> Fe {
    return fe_mont_mul(a, a);
}

fn fe_mul9(a: Fe) -> Fe {
    let a2 = fe_add(a, a);
    let a4 = fe_add(a2, a2);
    let a8 = fe_add(a4, a4);
    return fe_add(a8, a);
}

// Converts out of Montgomery form (multiply by 1): a * R^-1 mod p.
fn fe_from_mont(a: Fe) -> Fe {
    var t: array<u32, 17>;
    for (var i = 0u; i < 16u; i++) {
        t[i] = a[i];
    }
    for (var i = 0u; i < 16u; i++) {
        let m = (t[0] * FE_NP) & 0xffffu;
        let s0 = t[0] + m * FE_MOD[0];
        var c = s0 >> 16u;
        for (var j = 1u; j < 16u; j++) {
            let s = t[j] + m * FE_MOD[j] + c;
            t[j - 1u] = s & 0xffffu;
            c = s >> 16u;
        }
        let s16 = t[16] + c;
        t[15] = s16 & 0xffffu;
        t[16] = s16 >> 16u;
    }
    var lo: Fe;
    for (var i = 0u; i < 16u; i++) {
        lo[i] = t[i];
    }
    return fe_reduce_once(lo, t[16]);
}
