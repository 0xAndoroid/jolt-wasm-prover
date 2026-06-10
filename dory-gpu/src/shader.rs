//! WGSL source assembly: composes kernels from template snippets, injecting
//! field/curve constants derived from arkworks at runtime so the shader
//! constants can never drift from the CPU implementation.

use ark_ff::{BigInt, BigInteger, Fp, FpConfig, MontConfig, PrimeField};

pub const FIELD_WGSL: &str = include_str!("wgsl/field.wgsl");
pub const FQ2_WGSL: &str = include_str!("wgsl/fq2.wgsl");
pub const CURVE_WGSL: &str = include_str!("wgsl/curve.wgsl");
pub const FIELD_TEST_WGSL: &str = include_str!("wgsl/field_test.wgsl");
pub const CURVE_TEST_WGSL: &str = include_str!("wgsl/curve_test.wgsl");
pub const MSM_WGSL: &str = include_str!("wgsl/msm.wgsl");
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
