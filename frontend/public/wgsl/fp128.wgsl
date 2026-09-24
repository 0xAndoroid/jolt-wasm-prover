// Solinas field p = 2^128 - C, C = 0xFFFF_A7F7 (jolt_field::Prime128OffsetA7F7).
// Elements are 4 little-endian u32 limbs (x = lowest), the same bytes Rust
// stores for a canonical u128. Inputs may be non-canonical (any u128);
// outputs are always canonical (< p).
//
// Everything is 32-bit: WGSL has no u64 and no mulhi, so a 32x32 product is
// assembled from 16-bit halves. Reduction uses 2^128 == C (mod p): a carry
// out of the top limb folds back as "+C", and subtracting p from a value in
// [p, 2^128) is the same wrapping "+C".

const FP128_C: u32 = 0xFFFFA7F7u;
// p = [0x5809, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF]
const FP128_P0: u32 = 0x00005809u;

fn mul_wide(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xFFFFu;
    let a1 = a >> 16u;
    let b0 = b & 0xFFFFu;
    let b1 = b >> 16u;
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = p01 + p10;
    let mid_carry = select(0u, 1u, mid < p01);
    let lo = p00 + (mid << 16u);
    let lo_carry = select(0u, 1u, lo < p00);
    let hi = p11 + (mid >> 16u) + (mid_carry << 16u) + lo_carry;
    return vec2<u32>(lo, hi);
}

// acc is a 96-bit column accumulator (x lowest); adds a*b.
fn mac(acc: ptr<function, vec3<u32>>, a: u32, b: u32) {
    let p = mul_wide(a, b);
    let s0 = (*acc).x + p.x;
    let c0 = select(0u, 1u, s0 < p.x);
    let s1 = (*acc).y + p.y;
    let c1 = select(0u, 1u, s1 < p.y);
    let s1c = s1 + c0;
    let c1c = select(0u, 1u, s1c < c0);
    *acc = vec3<u32>(s0, s1c, (*acc).z + c1 + c1c);
}

fn shift_column(acc: ptr<function, vec3<u32>>) -> u32 {
    let out = (*acc).x;
    *acc = vec3<u32>((*acc).y, (*acc).z, 0u);
    return out;
}

// a + b with carry-out (x..w = sum, carry in the 5th component).
fn add128_carry(a: vec4<u32>, b: vec4<u32>) -> array<u32, 5> {
    let s0 = a.x + b.x;
    let c0 = select(0u, 1u, s0 < a.x);
    let s1 = a.y + b.y;
    let c1a = select(0u, 1u, s1 < a.y);
    let s1c = s1 + c0;
    let c1 = c1a + select(0u, 1u, s1c < c0);
    let s2 = a.z + b.z;
    let c2a = select(0u, 1u, s2 < a.z);
    let s2c = s2 + c1;
    let c2 = c2a + select(0u, 1u, s2c < c1);
    let s3 = a.w + b.w;
    let c3a = select(0u, 1u, s3 < a.w);
    let s3c = s3 + c2;
    let c3 = c3a + select(0u, 1u, s3c < c2);
    return array<u32, 5>(s0, s1c, s2c, s3c, c3);
}

fn sub128_borrow(a: vec4<u32>, b: vec4<u32>) -> array<u32, 5> {
    let d0 = a.x - b.x;
    let b0 = select(0u, 1u, a.x < b.x);
    let d1 = a.y - b.y;
    let b1a = select(0u, 1u, a.y < b.y);
    let d1b = d1 - b0;
    let b1 = b1a + select(0u, 1u, d1 < b0);
    let d2 = a.z - b.z;
    let b2a = select(0u, 1u, a.z < b.z);
    let d2b = d2 - b1;
    let b2 = b2a + select(0u, 1u, d2 < b1);
    let d3 = a.w - b.w;
    let b3a = select(0u, 1u, a.w < b.w);
    let d3b = d3 - b2;
    let b3 = b3a + select(0u, 1u, d3 < b2);
    return array<u32, 5>(d0, d1b, d2b, d3b, b3);
}

// x + k*C (wrapping), k in {0,1,2}; returns sum + carry flag.
fn add_c(x: vec4<u32>, k: u32) -> array<u32, 5> {
    return add128_carry(x, vec4<u32>(k * FP128_C, 0u, 0u, 0u));
}

fn geq_p(x: vec4<u32>) -> bool {
    return x.w == 0xFFFFFFFFu && x.z == 0xFFFFFFFFu && x.y == 0xFFFFFFFFu && x.x >= FP128_P0;
}

// Canonicalize a value in [0, 2^128): subtract p once if needed.
fn fp128_canon(x: vec4<u32>) -> vec4<u32> {
    let r = add_c(x, select(0u, 1u, geq_p(x)));
    return vec4<u32>(r[0], r[1], r[2], r[3]);
}

fn fp128_add(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    // a + b < 2^129: fold the carry as +C; that fold can carry once more
    // (only when the wrapped sum was already >= p), and that second fold
    // lands far below p.
    let s = add128_carry(a, b);
    let f1 = add_c(vec4<u32>(s[0], s[1], s[2], s[3]), s[4]);
    let f2 = add_c(vec4<u32>(f1[0], f1[1], f1[2], f1[3]), f1[4]);
    return fp128_canon(vec4<u32>(f2[0], f2[1], f2[2], f2[3]));
}

fn fp128_sub(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    // a - b in (-2^128, 2^128): each borrow means "+p", i.e. -C wrapping.
    let d = sub128_borrow(a, b);
    let d1 = sub128_borrow(vec4<u32>(d[0], d[1], d[2], d[3]), vec4<u32>(d[4] * FP128_C, 0u, 0u, 0u));
    let d2 = sub128_borrow(vec4<u32>(d1[0], d1[1], d1[2], d1[3]), vec4<u32>(d1[4] * FP128_C, 0u, 0u, 0u));
    return fp128_canon(vec4<u32>(d2[0], d2[1], d2[2], d2[3]));
}

fn fp128_mul(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    // Schoolbook 4x4 limbs, column by column, into t0..t7.
    var acc = vec3<u32>(0u, 0u, 0u);
    mac(&acc, a.x, b.x);
    let t0 = shift_column(&acc);
    mac(&acc, a.x, b.y);
    mac(&acc, a.y, b.x);
    let t1 = shift_column(&acc);
    mac(&acc, a.x, b.z);
    mac(&acc, a.y, b.y);
    mac(&acc, a.z, b.x);
    let t2 = shift_column(&acc);
    mac(&acc, a.x, b.w);
    mac(&acc, a.y, b.z);
    mac(&acc, a.z, b.y);
    mac(&acc, a.w, b.x);
    let t3 = shift_column(&acc);
    mac(&acc, a.y, b.w);
    mac(&acc, a.z, b.z);
    mac(&acc, a.w, b.y);
    let t4 = shift_column(&acc);
    mac(&acc, a.z, b.w);
    mac(&acc, a.w, b.z);
    let t5 = shift_column(&acc);
    mac(&acc, a.w, b.w);
    let t6 = shift_column(&acc);
    let t7 = shift_column(&acc);

    // Fold 1: lo + hi*C, hi*C is a 5-limb (160-bit) product.
    acc = vec3<u32>(0u, 0u, 0u);
    mac(&acc, t4, FP128_C);
    let h0 = shift_column(&acc);
    mac(&acc, t5, FP128_C);
    let h1 = shift_column(&acc);
    mac(&acc, t6, FP128_C);
    let h2 = shift_column(&acc);
    mac(&acc, t7, FP128_C);
    let h3 = shift_column(&acc);
    let h4 = shift_column(&acc);
    let r = add128_carry(vec4<u32>(t0, t1, t2, t3), vec4<u32>(h0, h1, h2, h3));
    // Fold 2: h4 and the add carry both sit at 2^128; (h4 + carry) <= C, so
    // (h4 + carry)*C is a 64-bit value.
    let top = mul_wide(h4 + r[4], FP128_C);
    let f1 = add128_carry(vec4<u32>(r[0], r[1], r[2], r[3]), vec4<u32>(top.x, top.y, 0u, 0u));
    // Fold 3: one last carry (value was < 2^128 + 2^64) is a tiny +C.
    let f2 = add_c(vec4<u32>(f1[0], f1[1], f1[2], f1[3]), f1[4]);
    return fp128_canon(vec4<u32>(f2[0], f2[1], f2[2], f2[3]));
}

fn fp128_muladd(a: vec4<u32>, b: vec4<u32>, c: vec4<u32>) -> vec4<u32> {
    return fp128_add(fp128_mul(a, b), c);
}

// ---------- small-integer multiplies (shared by the sumcheck kernels) ----------
const ZERO4: vec4<u32> = vec4<u32>(0u, 0u, 0u, 0u);
fn fp_small(k: u32) -> vec4<u32> { return vec4<u32>(k, 0u, 0u, 0u); }

// x * k for k < 2^32, reduced (5-limb product, fold the top limb as +top*C).
fn fp128_mul_small(x: vec4<u32>, k: u32) -> vec4<u32> {
  let p0 = mul_wide(x.x, k);
  let p1 = mul_wide(x.y, k);
  let p2 = mul_wide(x.z, k);
  let p3 = mul_wide(x.w, k);
  let s1 = p1.x + p0.y; let c1 = select(0u, 1u, s1 < p0.y);
  let s2a = p2.x + p1.y; let c2a = select(0u, 1u, s2a < p1.y);
  let s2 = s2a + c1; let c2 = c2a + select(0u, 1u, s2 < c1);
  let s3a = p3.x + p2.y; let c3a = select(0u, 1u, s3a < p2.y);
  let s3 = s3a + c2; let c3 = c3a + select(0u, 1u, s3 < c2);
  let top = p3.y + c3; // < 2^32 (product < 2^160)
  let tc = mul_wide(top, FP128_C);
  let f = add128_carry(vec4<u32>(p0.x, s1, s2, s3), vec4<u32>(tc.x, tc.y, 0u, 0u));
  let f2 = add_c(vec4<u32>(f[0], f[1], f[2], f[3]), f[4]);
  return fp128_canon(vec4<u32>(f2[0], f2[1], f2[2], f2[3]));
}

// x * k for signed k (two's complement i32 in a u32).
fn fp128_mul_signed(x: vec4<u32>, k: u32) -> vec4<u32> {
  let neg = (k & 0x80000000u) != 0u;
  let mag = select(k, 0u - k, neg);
  let v = fp128_mul_small(x, mag);
  return select(v, fp128_sub(ZERO4, v), neg);
}
