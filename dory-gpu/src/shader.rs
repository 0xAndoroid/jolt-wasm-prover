//! WGSL source assembly: composes kernels from template snippets, injecting
//! field/curve constants derived from arkworks at runtime so the shader
//! constants can never drift from the CPU implementation.

use ark_ff::{BigInt, BigInteger, Fp, FpConfig, MontConfig, PrimeField};

/// Fully unrolled 254-bit Montgomery field ops. Loops over limbs defeat
/// Metal's register promotion (naga's forced loop bounding blocks unrolling),
/// pushing every limb access through device-memory stack — a ~100x slowdown.
/// Generating straight-line code with constant indices keeps limbs in
/// registers.
pub static FIELD_WGSL: std::sync::LazyLock<String> = std::sync::LazyLock::new(field_ops_unrolled);

/// Loop-based variant of the field ops. The fused G1 MSM kernel (shared-mem
/// buckets, many inlined point-op sites) miscompiles on Apple Metal with the
/// unrolled ops — its total code size explodes — so it keeps the rolled form.
pub const FIELD_ROLLED_WGSL: &str = include_str!("wgsl/field.wgsl");

fn field_ops_unrolled() -> String {
    let mut s = String::with_capacity(64 * 1024);

    s.push_str("fn fe_unpack(w: Fe8) -> Fe {\n    var r: Fe;\n");
    for k in 0..8 {
        s.push_str(&format!(
            "    r[{}] = w[{k}] & 0xffffu;\n    r[{}] = w[{k}] >> 16u;\n",
            2 * k,
            2 * k + 1
        ));
    }
    s.push_str("    return r;\n}\n\n");

    s.push_str("fn fe_pack(a: Fe) -> Fe8 {\n    var r: Fe8;\n");
    for k in 0..8 {
        s.push_str(&format!(
            "    r[{k}] = a[{}] | (a[{}] << 16u);\n",
            2 * k,
            2 * k + 1
        ));
    }
    s.push_str("    return r;\n}\n\n");

    s.push_str("fn fe_zero() -> Fe {\n    var r: Fe;\n    return r;\n}\n\n");
    s.push_str("fn fe_mont_one() -> Fe {\n    var r: Fe = FE_MONT_ONE;\n    return r;\n}\n\n");

    s.push_str("fn fe_is_zero(a: Fe) -> bool {\n    var acc = 0u;\n");
    for i in 0..16 {
        s.push_str(&format!("    acc |= a[{i}];\n"));
    }
    s.push_str("    return acc == 0u;\n}\n\n");

    s.push_str("fn fe_eq(a: Fe, b: Fe) -> bool {\n    var acc = 0u;\n");
    for i in 0..16 {
        s.push_str(&format!("    acc |= a[{i}] ^ b[{i}];\n"));
    }
    s.push_str("    return acc == 0u;\n}\n\n");

    s.push_str("fn fe_select(c: bool, a: Fe, b: Fe) -> Fe {\n    var r: Fe;\n");
    for i in 0..16 {
        s.push_str(&format!("    r[{i}] = select(b[{i}], a[{i}], c);\n"));
    }
    s.push_str("    return r;\n}\n\n");

    // Reduces t (< 2p, with optional 2^256 carry bit `hi`) to [0, p).
    s.push_str("fn fe_reduce_once(t: Fe, hi: u32) -> Fe {\n    var s: Fe;\n    var d = 0u;\n    var brw = 0u;\n");
    for i in 0..16 {
        s.push_str(&format!(
            "    d = t[{i}] - FE_MOD[{i}] - brw;\n    s[{i}] = d & 0xffffu;\n    brw = (d >> 16u) & 1u;\n"
        ));
    }
    s.push_str("    return fe_select((hi != 0u) | (brw == 0u), s, t);\n}\n\n");

    s.push_str(
        "fn fe_add(a: Fe, b: Fe) -> Fe {\n    var t: Fe;\n    var x = 0u;\n    var c = 0u;\n",
    );
    for i in 0..16 {
        s.push_str(&format!(
            "    x = a[{i}] + b[{i}] + c;\n    t[{i}] = x & 0xffffu;\n    c = x >> 16u;\n"
        ));
    }
    s.push_str("    return fe_reduce_once(t, c);\n}\n\n");

    s.push_str("fn fe_sub(a: Fe, b: Fe) -> Fe {\n    var t: Fe;\n    var r: Fe;\n    var x = 0u;\n    var brw = 0u;\n    var c = 0u;\n");
    for i in 0..16 {
        s.push_str(&format!(
            "    x = a[{i}] - b[{i}] - brw;\n    t[{i}] = x & 0xffffu;\n    brw = (x >> 16u) & 1u;\n"
        ));
    }
    for i in 0..16 {
        s.push_str(&format!(
            "    x = t[{i}] + FE_MOD[{i}] + c;\n    r[{i}] = x & 0xffffu;\n    c = x >> 16u;\n"
        ));
    }
    s.push_str("    return fe_select(brw == 1u, r, t);\n}\n\n");

    s.push_str("fn fe_neg(a: Fe) -> Fe {\n    var t: Fe;\n    var x = 0u;\n    var brw = 0u;\n");
    for i in 0..16 {
        s.push_str(&format!(
            "    x = FE_MOD[{i}] - a[{i}] - brw;\n    t[{i}] = x & 0xffffu;\n    brw = (x >> 16u) & 1u;\n"
        ));
    }
    s.push_str("    return fe_select(fe_is_zero(a), fe_zero(), t);\n}\n\n");

    s.push_str("fn fe_double(a: Fe) -> Fe {\n    return fe_add(a, a);\n}\n\n");

    // CIOS Montgomery multiplication, fully unrolled.
    s.push_str("fn fe_mont_mul(a: Fe, b: Fe) -> Fe {\n    var t: array<u32, 18>;\n    var x = 0u;\n    var c = 0u;\n    var m = 0u;\n    var bi = 0u;\n");
    for _i in 0..16 {
        s.push_str("    bi = b[");
        s.push_str(&_i.to_string());
        s.push_str("];\n    c = 0u;\n");
        for j in 0..16 {
            s.push_str(&format!(
                "    x = t[{j}] + a[{j}] * bi + c;\n    t[{j}] = x & 0xffffu;\n    c = x >> 16u;\n"
            ));
        }
        s.push_str("    x = t[16] + c;\n    t[16] = x & 0xffffu;\n    t[17] = x >> 16u;\n");
        s.push_str("    m = (t[0] * FE_NP) & 0xffffu;\n");
        s.push_str("    x = t[0] + m * FE_MOD[0];\n    c = x >> 16u;\n");
        for j in 1..16 {
            s.push_str(&format!(
                "    x = t[{j}] + m * FE_MOD[{j}] + c;\n    t[{}] = x & 0xffffu;\n    c = x >> 16u;\n",
                j - 1
            ));
        }
        s.push_str(
            "    x = t[16] + c;\n    t[15] = x & 0xffffu;\n    t[16] = t[17] + (x >> 16u);\n",
        );
    }
    s.push_str("    var lo: Fe;\n");
    for i in 0..16 {
        s.push_str(&format!("    lo[{i}] = t[{i}];\n"));
    }
    s.push_str("    return fe_reduce_once(lo, t[16]);\n}\n\n");

    s.push_str("fn fe_sqr(a: Fe) -> Fe {\n    return fe_mont_mul(a, a);\n}\n\n");

    s.push_str("fn fe_mul9(a: Fe) -> Fe {\n    let a2 = fe_add(a, a);\n    let a4 = fe_add(a2, a2);\n    let a8 = fe_add(a4, a4);\n    return fe_add(a8, a);\n}\n\n");

    // Fermat inversion: bit loop stays rolled (the body dominates); 0 -> 0.
    s.push_str(
        "fn fe_inv(a: Fe) -> Fe {\n    var acc = fe_mont_one();\n    for (var i = 0i; i < 256; i++) {\n        let b = 255u - u32(i);\n        acc = fe_mont_mul(acc, acc);\n        let limb = FE_P_MINUS_2[b >> 4u];\n        let is_set = ((limb >> (b & 15u)) & 1u) == 1u;\n        acc = fe_select(is_set, fe_mont_mul(acc, a), acc);\n    }\n    return acc;\n}\n\n",
    );

    // Montgomery reduction by 1 (out of Montgomery form), unrolled.
    s.push_str("fn fe_from_mont(a: Fe) -> Fe {\n    var t: array<u32, 17>;\n    var x = 0u;\n    var c = 0u;\n    var m = 0u;\n");
    for i in 0..16 {
        s.push_str(&format!("    t[{i}] = a[{i}];\n"));
    }
    for _i in 0..16 {
        s.push_str("    m = (t[0] * FE_NP) & 0xffffu;\n");
        s.push_str("    x = t[0] + m * FE_MOD[0];\n    c = x >> 16u;\n");
        for j in 1..16 {
            s.push_str(&format!(
                "    x = t[{j}] + m * FE_MOD[{j}] + c;\n    t[{}] = x & 0xffffu;\n    c = x >> 16u;\n",
                j - 1
            ));
        }
        s.push_str("    x = t[16] + c;\n    t[15] = x & 0xffffu;\n    t[16] = x >> 16u;\n");
    }
    s.push_str("    var lo: Fe;\n");
    for i in 0..16 {
        s.push_str(&format!("    lo[{i}] = t[{i}];\n"));
    }
    s.push_str("    return fe_reduce_once(lo, t[16]);\n}\n");

    s
}
pub const FQ2_WGSL: &str = include_str!("wgsl/fq2.wgsl");
pub const CURVE_WGSL: &str = include_str!("wgsl/curve.wgsl");
pub const FIELD_TEST_WGSL: &str = include_str!("wgsl/field_test.wgsl");
pub const CURVE_TEST_WGSL: &str = include_str!("wgsl/curve_test.wgsl");
pub const MSM_COMMON_WGSL: &str = include_str!("wgsl/msm_common.wgsl");
pub const MSM_SPLIT_WGSL: &str = include_str!("wgsl/msm_split.wgsl");
pub const MSM_PREP_WGSL: &str = include_str!("wgsl/msm_prep.wgsl");
pub const NORMALIZE_WGSL: &str = include_str!("wgsl/normalize.wgsl");
pub const FOLD_WGSL: &str = include_str!("wgsl/fold.wgsl");
pub const VMV_WGSL: &str = include_str!("wgsl/vmv.wgsl");
pub const FQ12_WGSL: &str = include_str!("wgsl/fq12.wgsl");
pub const PAIRING_WGSL: &str = include_str!("wgsl/pairing.wgsl");
pub const PAIRING_FQ12_WGSL: &str = include_str!("wgsl/pairing_fq12.wgsl");

/// 256-bit big integer as 16 radix-2^16 limbs, little-endian, each stored in a u32.
pub fn limbs16(v: &BigInt<4>) -> [u32; 16] {
    let mut out = [0u32; 16];
    for (i, w) in v.0.iter().enumerate() {
        for k in 0..4 {
            out[i * 4 + k] = ((w >> (16 * k)) & 0xffff) as u32;
        }
    }
    out
}

fn wgsl_array16(v: &BigInt<4>) -> String {
    let limbs = limbs16(v);
    let body: Vec<String> = limbs.iter().map(|l| format!("{l:#x}u")).collect();
    format!("Fe({})", body.join(","))
}

/// Montgomery representation limbs of a field element (the raw BigInt inside `Fp`).
pub fn mont_limbs<P: FpConfig<4>>(x: Fp<P, 4>) -> BigInt<4> {
    x.0
}

/// Emits the constants header expected by `field.wgsl` for a given 254-bit
/// Montgomery field (works for both BN254 Fq and Fr: same limb count, R = 2^256).
pub fn field_header<C: MontConfig<4>>() -> String {
    let modulus = C::MODULUS;
    let np16 = (C::INV & 0xffff) as u32;
    let mont_one = C::R;
    let mut p_minus_2 = modulus;
    p_minus_2.sub_with_borrow(&BigInt::from(2u64));
    // The alias declarations must precede the globals so that naga's MSL
    // backend gives the globals and all function locals one unified type
    // (mixing `alias Fe` values with raw array<u32,16> values miscompiles).
    format!(
        "alias Fe = array<u32, 16>;\n\
         alias Fe8 = array<u32, 8>;\n\
         var<private> FE_MOD: Fe = {};\n\
         const FE_NP: u32 = {:#x}u;\n\
         var<private> FE_MONT_ONE: Fe = {};\n\
         var<private> FE_P_MINUS_2: Fe = {};\n",
        wgsl_array16(&modulus),
        np16,
        wgsl_array16(&mont_one),
        wgsl_array16(&p_minus_2)
    )
}

/// Builder that concatenates snippets with optional identifier substitution.
pub struct ShaderBuilder {
    source: String,
}

impl ShaderBuilder {
    pub fn new() -> Self {
        Self {
            source: String::new(),
        }
    }

    pub fn push(mut self, snippet: &str) -> Self {
        self.source.push_str(snippet);
        self.source.push('\n');
        self
    }

    /// Appends a snippet with `from -> to` whole-token prefix substitutions.
    /// Substitution is plain string replacement; templates use distinctive
    /// prefixes (e.g. `PF_`) so collisions are impossible.
    pub fn push_subst(mut self, snippet: &str, substitutions: &[(&str, &str)]) -> Self {
        let mut text = snippet.to_string();
        for (from, to) in substitutions {
            text = text.replace(from, to);
        }
        self.source.push_str(&text);
        self.source.push('\n');
        self
    }

    pub fn build(self) -> String {
        self.source
    }
}

impl Default for ShaderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Substitution instantiating `curve.wgsl` for G1 (point field = Fq).
pub const G1_SUBST: &[(&str, &str)] = &[("PF_", "g1_")];

/// Substitution instantiating `curve.wgsl` for G2 (point field = Fq2).
pub const G2_SUBST: &[(&str, &str)] = &[("PF_", "g2_")];

/// Glue mapping the `g1_*` names used by the curve template onto Fq ops.
/// b = 3 for G1, so 3b = 9, computed with an add chain instead of a full mul.
pub const G1_GLUE: &str = "
alias g1_Fe = Fe;
const g1_FEW: u32 = 8u;
fn g1_mul(a: Fe, b: Fe) -> Fe { return fe_mont_mul(a, b); }
fn g1_add(a: Fe, b: Fe) -> Fe { return fe_add(a, b); }
fn g1_sub(a: Fe, b: Fe) -> Fe { return fe_sub(a, b); }
fn g1_neg(a: Fe) -> Fe { return fe_neg(a); }
fn g1_zero() -> Fe { return fe_zero(); }
fn g1_one() -> Fe { return fe_mont_one(); }
fn g1_is_zero(a: Fe) -> bool { return fe_is_zero(a); }
fn g1_select(c: bool, a: Fe, b: Fe) -> Fe { return fe_select(c, a, b); }
fn g1_mul_by_3b(a: Fe) -> Fe { return fe_mul9(a); }
fn g1_inv(a: Fe) -> Fe { return fe_inv(a); }
fn g1_unpack_words(p: array<u32,16>) -> Fe {
    var w: Fe8;
    for (var k = 0u; k < 8u; k++) { w[k] = p[k]; }
    return fe_unpack(w);
}
fn g1_pack_words(v: Fe) -> array<u32,16> {
    let w = fe_pack(v);
    var r: array<u32,16>;
    for (var k = 0u; k < 8u; k++) { r[k] = w[k]; }
    return r;
}
";

/// Glue mapping the `g2_*` names used by the curve template onto Fq2 ops.
/// 3b' is a full Fq2 constant (emitted in the header as G2_3B_C0/G2_3B_C1).
pub const G2_GLUE: &str = "
alias g2_Fe = Fe2;
const g2_FEW: u32 = 16u;
fn g2_mul(a: Fe2, b: Fe2) -> Fe2 { return fq2_mul(a, b); }
fn g2_add(a: Fe2, b: Fe2) -> Fe2 { return fq2_add(a, b); }
fn g2_sub(a: Fe2, b: Fe2) -> Fe2 { return fq2_sub(a, b); }
fn g2_neg(a: Fe2) -> Fe2 { return fq2_neg(a); }
fn g2_zero() -> Fe2 { return fq2_zero(); }
fn g2_one() -> Fe2 { return fq2_mont_one(); }
fn g2_is_zero(a: Fe2) -> bool { return fq2_is_zero(a); }
fn g2_select(c: bool, a: Fe2, b: Fe2) -> Fe2 { return fq2_select(c, a, b); }
fn g2_mul_by_3b(a: Fe2) -> Fe2 { return fq2_mul(a, Fe2(G2_3B_C0, G2_3B_C1)); }
fn g2_inv(a: Fe2) -> Fe2 { return fq2_inv(a); }
fn g2_unpack_words(p: array<u32,16>) -> Fe2 {
    var w0: Fe8;
    var w1: Fe8;
    for (var k = 0u; k < 8u; k++) { w0[k] = p[k]; w1[k] = p[k + 8u]; }
    return Fe2(fe_unpack(w0), fe_unpack(w1));
}
fn g2_pack_words(v: Fe2) -> array<u32,16> {
    let a = fe_pack(v.c0);
    let b = fe_pack(v.c1);
    var r: array<u32,16>;
    for (var k = 0u; k < 8u; k++) { r[k] = a[k]; r[k + 8u] = b[k]; }
    return r;
}
";

/// Constants header for the G2 twist coefficient 3*b' (Montgomery form).
pub fn g2_3b_header() -> String {
    use ark_ec::short_weierstrass::SWCurveConfig;
    let b = <ark_bn254::g2::Config as SWCurveConfig>::COEFF_B;
    let three_b = b + b + b;
    format!(
        "var<private> G2_3B_C0: Fe = {};\nvar<private> G2_3B_C1: Fe = {};\n",
        wgsl_array16(&three_b.c0.0),
        wgsl_array16(&three_b.c1.0)
    )
}

/// Constants header for the pairing kernels: 1/2 in Fq and the psi twist
/// constants (all Montgomery form, from arkworks BnConfig).
pub fn pairing_header() -> String {
    use ark_ec::bn::BnConfig;
    use ark_ff::Field;
    let two_inv = ark_bn254::Fq::from(2u64).inverse().expect("2 invertible");
    let twqx = <ark_bn254::Config as BnConfig>::TWIST_MUL_BY_Q_X;
    let twqy = <ark_bn254::Config as BnConfig>::TWIST_MUL_BY_Q_Y;
    format!(
        "var<private> FQ_TWO_INV: Fe = {};\n\
         var<private> TWQX_C0: Fe = {};\n\
         var<private> TWQX_C1: Fe = {};\n\
         var<private> TWQY_C0: Fe = {};\n\
         var<private> TWQY_C1: Fe = {};\n",
        wgsl_array16(&two_inv.0),
        wgsl_array16(&twqx.c0.0),
        wgsl_array16(&twqx.c1.0),
        wgsl_array16(&twqy.c0.0),
        wgsl_array16(&twqy.c1.0),
    )
}

/// MSM tuning constants. Window sizes are bounded by workgroup shared memory:
/// G1 buckets are 2^(8-1) * 96B = 12 KB, G2 buckets 2^(7-1) * 192B = 12 KB.
pub const MSM_CHUNK: u32 = 512;
pub const G1_WINDOW: u32 = 8;
pub const G2_WINDOW: u32 = 7;

pub fn msm_windows(c: u32) -> u32 {
    // Carries cannot escape the top window for BN254 Fr (254 bits): the top
    // raw digit is at most 2^(254 - (nw-1)*c) - 1 + 1 < 2^(c-1).
    254_u32.div_ceil(c)
}

/// Consts header for the MSM modules (`msm_common.wgsl` and friends).
pub fn msm_header(c: u32) -> String {
    format!(
        "const MSM_C: u32 = {c}u;\n\
         const MSM_NB: u32 = {}u;\n\
         const MSM_NW: u32 = {}u;\n\
         const MSM_CHUNK: u32 = {}u;\n",
        1u32 << (c - 1),
        msm_windows(c),
        MSM_CHUNK,
    )
}

/// Header for Fq instantiation of `field.wgsl`.
pub fn fq_header() -> String {
    field_header::<ark_bn254::FqConfig>()
}

/// Header for Fr instantiation of `field.wgsl`.
pub fn fr_header() -> String {
    field_header::<ark_bn254::FrConfig>()
}

/// Fr modulus limbs in radix-2^16, for canonical-form comparisons on GPU.
pub fn fr_modulus_limbs16() -> [u32; 16] {
    limbs16(&<ark_bn254::FrConfig as MontConfig<4>>::MODULUS)
}

pub fn fq_modulus() -> BigInt<4> {
    ark_bn254::Fq::MODULUS
}
