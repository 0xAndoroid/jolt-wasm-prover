//! Cooperative Miller-loop schedule: the whole prepared-line ate loop as a
//! stream of Fq2 micro-ops over a 44-slot shared-memory file, executed by
//! one 32-lane workgroup per pair in a single dispatch
//! (`wgsl/coop_pairing.wgsl`).
//!
//! The op stream mirrors, formula for formula, the per-thread kernels it
//! replaces: `fq6_mul` / `fq6_mul_by_01` (fq12.wgsl) composed into the
//! Fq12 squaring and the `mul_by_034` line application of
//! pairing_fq12.wgsl, iterated over arkworks' ate schedule — field values
//! are exact, so the resulting f matches the sequential pipeline (and
//! arkworks) bit for bit.

use ark_ec::bn::BnConfig;
use std::sync::LazyLock;

use crate::context::GpuContext;
use crate::pairing::MillerState;
use crate::shader::{fq_header, ShaderBuilder, COOP_PAIRING_WGSL, FIELD_WGSL, FQ2_WGSL};

const ADD: u32 = 0;
const SUB: u32 = 1;
const SUB_XI: u32 = 2; // a - xi*b
const ADD_XI: u32 = 3; // a + xi*b
const DBL: u32 = 4;
const MUL: u32 = 5;
const CONST01: u32 = 6; // one (a=1) / zero (a=0)
const LINE: u32 = 7; // prepared line a, component b
const LOADP: u32 = 8; // P.x (a=0) / P.y (a=1); dst 9 also computes the mask
const CONSTBUF: u32 = 9; // consts[a]: two_inv, 3b', twqx, twqy
const CONJ: u32 = 10; // Fq2 conjugate of slot a
const LOADQ: u32 = 11; // Q.x (a=0) / Q.y (a=1)

// Slot file layout. f = [A0..A2 | B0..B2] (the two Fq6 halves, Fq2 coeffs).
const A0: u32 = 0;
const A1: u32 = 1;
const A2: u32 = 2;
const B0: u32 = 3;
const B1: u32 = 4;
const B2: u32 = 5;
const LC0: u32 = 6;
const LC1: u32 = 7;
const LC2: u32 = 8; // doubles as c4 (unscaled line component)
const PX: u32 = 9;
const PY: u32 = 10;
// Slots 11..=40: stage temps (sqr / apply / dbl / add reuse them).
// Persistent slots for the computed-line path:
const TX: u32 = 41;
const TY: u32 = 42;
const TZ: u32 = 43;
const QX: u32 = 44;
const QY: u32 = 45;
const NQY: u32 = 46;
const Q1X: u32 = 47;
const Q1Y: u32 = 48;
const Q2X: u32 = 49;
const Q2Y: u32 = 50;
const TWOINV: u32 = 51;
const THRB: u32 = 52;
const TWQX: u32 = 53;
const TWQY: u32 = 54;
const ZERO: u32 = 55;

struct Schedule {
    ops: Vec<[u32; 4]>,
    groups: Vec<u32>,
}

impl Schedule {
    fn group(&mut self, ops: &[[u32; 4]]) {
        assert!(ops.len() <= 32, "group exceeds lane count");
        // Within a group every op must write a distinct slot and no op may
        // read a slot another op of the same group writes.
        let dsts: Vec<u32> = ops.iter().map(|o| o[0]).collect();
        for (i, op) in ops.iter().enumerate() {
            for (j, d) in dsts.iter().enumerate() {
                if i != j {
                    assert_ne!(op[0], *d, "duplicate dst in group");
                }
                if i != j && op[3] != CONST01 && op[3] != LINE && op[3] != LOADP {
                    assert_ne!(op[1], *d, "op reads a slot written in-group");
                    if op[3] != DBL {
                        assert_ne!(op[2], *d, "op reads a slot written in-group");
                    }
                }
            }
        }
        let first = self.ops.len() as u32;
        self.groups.push((first << 8) | ops.len() as u32);
        self.ops.extend_from_slice(ops);
    }

    /// f^2, Karatsuba over the Fq6 halves: v0 = (c0-c1)(c0 - v*c1),
    /// v2 = c0*c1; c0' = v0 + v*v2 + v2; c1' = 2*v2 (f_sqr_a/b).
    fn sqr(&mut self) {
        // Temps: X (c0-c1) 11..13, Y (c0 - v*c1) 14..16, A/B pair sums
        // 17..22, X/Y pair sums 23..28, v0 muls 29..34, v2 muls 35..40.
        self.group(&[
            [11, A0, B0, SUB],
            [12, A1, B1, SUB],
            [13, A2, B2, SUB],
            [14, A0, B2, SUB_XI],
            [15, A1, B0, SUB],
            [16, A2, B1, SUB],
            [17, A1, A2, ADD],
            [18, A0, A1, ADD],
            [19, A0, A2, ADD],
            [20, B1, B2, ADD],
            [21, B0, B1, ADD],
            [22, B0, B2, ADD],
        ]);
        self.group(&[
            [23, 12, 13, ADD],
            [24, 11, 12, ADD],
            [25, 11, 13, ADD],
            [26, 15, 16, ADD],
            [27, 14, 15, ADD],
            [28, 14, 16, ADD],
        ]);
        self.group(&[
            [29, 11, 14, MUL],
            [30, 12, 15, MUL],
            [31, 13, 16, MUL],
            [32, 23, 26, MUL],
            [33, 24, 27, MUL],
            [34, 25, 28, MUL],
            [35, A0, B0, MUL],
            [36, A1, B1, MUL],
            [37, A2, B2, MUL],
            [38, 17, 20, MUL],
            [39, 18, 21, MUL],
            [40, 19, 22, MUL],
        ]);
        // fq6_mul combine: x = m3-be-cf; y = m4-ad-be; z = m5-ad+be-cf;
        // out = (ad + xi*x, y + xi*cf, z).
        self.group(&[
            [32, 32, 30, SUB],
            [33, 33, 29, SUB],
            [34, 34, 29, SUB],
            [38, 38, 36, SUB],
            [39, 39, 35, SUB],
            [40, 40, 35, SUB],
        ]);
        self.group(&[
            [32, 32, 31, SUB],
            [33, 33, 30, SUB],
            [34, 34, 30, ADD],
            [38, 38, 37, SUB],
            [39, 39, 36, SUB],
            [40, 40, 36, ADD],
        ]);
        self.group(&[
            [34, 34, 31, SUB],    // P2 = z
            [40, 40, 37, SUB],    // Q2
            [29, 29, 32, ADD_XI], // P0 = ad + xi*x
            [33, 33, 31, ADD_XI], // P1 = y + xi*cf
            [35, 35, 38, ADD_XI], // Q0
            [39, 39, 37, ADD_XI], // Q1
        ]);
        // c1' = 2*v2; c0' = v0 + nonres(v2) + v2 with
        // nonres(Q) = (xi*Q2, Q0, Q1).
        self.group(&[
            [B0, 35, 0, DBL],
            [B1, 39, 0, DBL],
            [B2, 40, 0, DBL],
            [29, 29, 40, ADD_XI], // P0 + xi*Q2
            [33, 33, 35, ADD],    // P1 + Q0
            [34, 34, 39, ADD],    // P2 + Q1
        ]);
        self.group(&[[A0, 29, 35, ADD], [A1, 33, 39, ADD], [A2, 34, 40, ADD]]);
    }

    /// f *= line (mul_by_034): c0 = l0*P.y, c3 = l1*P.x, c4 = l2;
    /// a = f.c0 scaled by c0; b = f.c1.mul_by_01(c3, c4);
    /// e = (f.c0+f.c1).mul_by_01(c0+c3, c4);
    /// f.c1' = e - (a+b); f.c0' = nonres(b) + a.
    /// `line = Some(idx)` loads prepared line `idx`; `None` consumes the
    /// LC slots written by a preceding dbl/add step.
    fn apply(&mut self, line: Option<u32>) {
        // Temps: S (f0+f1) 11..13, B-sums 14..16, c0 17, c3 18, u 19,
        // cs 20, us 21, S-sums 22..24, a 25..27, b-muls 28..32,
        // e-muls 33..37, t 38..40.
        if let Some(line) = line {
            self.group(&[
                [LC0, line, 0, LINE],
                [LC1, line, 1, LINE],
                [LC2, line, 2, LINE],
            ]);
        }
        self.group(&[
            [17, LC0, PY, MUL],
            [18, LC1, PX, MUL],
            [11, A0, B0, ADD],
            [12, A1, B1, ADD],
            [13, A2, B2, ADD],
            [14, B1, B2, ADD],
            [15, B0, B1, ADD],
            [16, B0, B2, ADD],
        ]);
        self.group(&[
            [19, 17, 18, ADD],
            [20, 18, LC2, ADD],
            [22, 12, 13, ADD],
            [23, 11, 12, ADD],
            [24, 11, 13, ADD],
        ]);
        self.group(&[[21, 19, LC2, ADD]]);
        self.group(&[
            [25, A0, 17, MUL],
            [26, A1, 17, MUL],
            [27, A2, 17, MUL],
            [28, B0, 18, MUL],
            [29, B1, LC2, MUL],
            [30, LC2, 14, MUL],
            [31, 18, 16, MUL],
            [32, 20, 15, MUL],
            [33, 11, 19, MUL],
            [34, 12, LC2, MUL],
            [35, LC2, 22, MUL],
            [36, 19, 24, MUL],
            [37, 21, 23, MUL],
        ]);
        // mul_by_01 combine: t1 = xi*(mA - bb) + aa; t2 = mC - aa - bb;
        // t3 = mB - aa + bb.
        self.group(&[
            [30, 30, 29, SUB],
            [35, 35, 34, SUB],
            [31, 31, 28, SUB],
            [36, 36, 33, SUB],
            [32, 32, 28, SUB],
            [37, 37, 33, SUB],
        ]);
        self.group(&[
            [28, 28, 30, ADD_XI], // b0
            [33, 33, 35, ADD_XI], // e0
            [31, 31, 29, ADD],    // b2
            [36, 36, 34, ADD],    // e2
            [32, 32, 29, SUB],    // b1
            [37, 37, 34, SUB],    // e1
        ]);
        self.group(&[
            [38, 25, 28, ADD],    // a0 + b0
            [39, 26, 32, ADD],    // a1 + b1
            [40, 27, 31, ADD],    // a2 + b2
            [A0, 25, 31, ADD_XI], // a0 + xi*b2
            [A1, 28, 26, ADD],    // b0 + a1
            [A2, 32, 27, ADD],    // b1 + a2
        ]);
        self.group(&[[B0, 33, 38, SUB], [B1, 37, 39, SUB], [B2, 36, 40, SUB]]);
    }

    /// arkworks homogeneous-projective doubling step (pair_dbl_step):
    /// updates T and writes the line (-h, 3j, i) into the LC slots.
    fn dbl_step(&mut self) {
        self.group(&[
            [11, TX, TY, MUL],
            [12, TY, TY, MUL], // b
            [13, TZ, TZ, MUL], // c
            [14, TX, TX, MUL], // j
            [15, TY, TZ, ADD],
        ]);
        self.group(&[
            [16, 11, TWOINV, MUL], // a
            [17, 13, THRB, MUL],   // e
            [18, 15, 15, MUL],     // (ty+tz)^2
            [19, 12, 13, ADD],     // b + c
        ]);
        self.group(&[
            [20, 17, 17, ADD],  // 2e (avoid DBL self-read ambiguity)
            [21, 17, 17, MUL],  // e^2
            [LC2, 17, 12, SUB], // i = e - b
            [22, 18, 19, SUB],  // h
            [23, 14, 14, ADD],  // 2j
        ]);
        self.group(&[
            [24, 20, 17, ADD],    // f = 3e
            [LC1, 23, 14, ADD],   // 3j
            [LC0, ZERO, 22, SUB], // -h
            [26, 21, 21, ADD],    // 2e^2
        ]);
        self.group(&[
            [27, 12, 24, ADD], // b + f
            [28, 12, 24, SUB], // b - f
            [29, 26, 21, ADD], // 3e^2
        ]);
        self.group(&[
            [30, 27, TWOINV, MUL], // g
            [TX, 16, 28, MUL],     // a * (b - f)
            [TZ, 12, 22, MUL],     // b * h
        ]);
        self.group(&[[31, 30, 30, MUL]]); // g^2
        self.group(&[[TY, 31, 29, SUB]]);
    }

    /// arkworks homogeneous-projective addition step (pair_add_step) against
    /// the affine Q in slots (qxs, qys): updates T and writes the line
    /// (lambda, -theta, j) into the LC slots.
    fn add_step(&mut self, qxs: u32, qys: u32) {
        self.group(&[[11, qys, TZ, MUL], [12, qxs, TZ, MUL]]);
        self.group(&[
            [13, TY, 11, SUB], // theta
            [14, TX, 12, SUB], // lambda
        ]);
        self.group(&[
            [15, 13, 13, MUL], // c
            [16, 14, 14, MUL], // d
        ]);
        self.group(&[
            [17, 14, 16, MUL],  // e
            [18, TZ, 15, MUL],  // f
            [19, TX, 16, MUL],  // g
            [20, 13, qxs, MUL], // theta * qx
            [21, 14, qys, MUL], // lambda * qy
        ]);
        self.group(&[
            [23, 19, 19, ADD],    // 2g
            [24, 17, 18, ADD],    // e + f
            [LC2, 20, 21, SUB],   // j
            [LC0, 14, ZERO, ADD], // lambda
            [LC1, ZERO, 13, SUB], // -theta
            [25, 17, TY, MUL],    // e * ty
        ]);
        self.group(&[[26, 24, 23, SUB]]); // h
        self.group(&[
            [TX, 14, 26, MUL],
            [27, 19, 26, SUB], // g - h
            [TZ, TZ, 17, MUL],
        ]);
        self.group(&[[28, 13, 27, MUL]]);
        self.group(&[[TY, 28, 25, SUB]]);
    }

    /// q1 = psi(q), q2 = -psi^2(q) (pair_frob).
    fn frob(&mut self) {
        self.group(&[[11, QX, 0, CONJ], [12, QY, 0, CONJ]]);
        self.group(&[[Q1X, 11, TWQX, MUL], [Q1Y, 12, TWQY, MUL]]);
        self.group(&[[13, Q1X, 0, CONJ], [14, Q1Y, 0, CONJ]]);
        self.group(&[[Q2X, 13, TWQX, MUL], [15, 14, TWQY, MUL]]);
        self.group(&[[Q2Y, ZERO, 15, SUB]]);
    }
}

fn prologue(s: &mut Schedule) {
    // P loads (PX's op also evaluates the pair mask) + f = 1.
    s.group(&[
        [PX, 0, 0, LOADP],
        [PY, 1, 0, LOADP],
        [A0, 1, 0, CONST01],
        [A1, 0, 0, CONST01],
        [A2, 0, 0, CONST01],
        [B0, 0, 0, CONST01],
        [B1, 0, 0, CONST01],
        [B2, 0, 0, CONST01],
    ]);
}

/// The full prepared-path Miller schedule (prologue + ate loop), mirroring
/// `encode_miller_prepared`'s iteration exactly.
static SCHEDULE: LazyLock<Schedule> = LazyLock::new(|| {
    let mut s = Schedule {
        ops: Vec::new(),
        groups: Vec::new(),
    };
    prologue(&mut s);

    let ate = <ark_bn254::Config as BnConfig>::ATE_LOOP_COUNT;
    let mut line = 0u32;
    for i in (1..ate.len()).rev() {
        if i != ate.len() - 1 {
            s.sqr();
        }
        s.apply(Some(line));
        line += 1;
        if ate[i - 1] != 0 {
            s.apply(Some(line));
            line += 1;
        }
    }
    s.apply(Some(line));
    s.apply(Some(line + 1));
    assert_eq!(line + 2, crate::pairing::prepared_line_count());
    s
});

/// The computed-path Miller schedule: G2 line computation (dbl/add steps,
/// Frobenius finals) interleaved with the f evaluation, all in one
/// dispatch — mirrors `encode_miller_computed`'s sequence exactly.
static SCHEDULE_COMPUTED: LazyLock<Schedule> = LazyLock::new(|| {
    let mut s = Schedule {
        ops: Vec::new(),
        groups: Vec::new(),
    };
    prologue(&mut s);
    s.group(&[
        [TWOINV, 0, 0, CONSTBUF],
        [THRB, 1, 0, CONSTBUF],
        [TWQX, 2, 0, CONSTBUF],
        [TWQY, 3, 0, CONSTBUF],
        [ZERO, 0, 0, CONST01],
        [QX, 0, 0, LOADQ],
        [QY, 1, 0, LOADQ],
    ]);
    s.group(&[
        [TX, QX, ZERO, ADD],
        [TY, QY, ZERO, ADD],
        [TZ, 1, 0, CONST01],
        [NQY, ZERO, QY, SUB],
    ]);

    let ate = <ark_bn254::Config as BnConfig>::ATE_LOOP_COUNT;
    for i in (1..ate.len()).rev() {
        if i != ate.len() - 1 {
            s.sqr();
        }
        s.dbl_step();
        s.apply(None);
        let bit = ate[i - 1];
        if bit == 1 {
            s.add_step(QX, QY);
            s.apply(None);
        } else if bit == -1 {
            s.add_step(QX, NQY);
            s.apply(None);
        }
    }
    s.frob();
    s.add_step(Q1X, Q1Y);
    s.apply(None);
    s.add_step(Q2X, Q2Y);
    s.apply(None);
    s
});

fn coop_module_source() -> String {
    ShaderBuilder::new()
        .push(&fq_header())
        .push(&FIELD_WGSL)
        .push(FQ2_WGSL)
        .push(COOP_PAIRING_WGSL)
        .build()
}

/// Twist/Frobenius constants consumed by the computed-path schedule, packed
/// as four Fq2 values: two_inv (embedded in c0), 3b', TWIST_MUL_BY_Q_X/Y.
fn consts_words() -> Vec<u32> {
    use ark_ec::short_weierstrass::SWCurveConfig;
    use ark_ff::Field;
    let two_inv = ark_bn254::Fq::from(2u64).inverse().expect("2 invertible");
    let b = <ark_bn254::g2::Config as SWCurveConfig>::COEFF_B;
    let three_b = b + b + b;
    let twqx = <ark_bn254::Config as BnConfig>::TWIST_MUL_BY_Q_X;
    let twqy = <ark_bn254::Config as BnConfig>::TWIST_MUL_BY_Q_Y;
    let mut words = Vec::with_capacity(4 * 16);
    for fe2 in [
        ark_bn254::Fq2::new(two_inv, ark_ff::Zero::zero()),
        three_b,
        twqx,
        twqy,
    ] {
        words.extend_from_slice(&crate::repr::fq2_to_words(&fe2));
    }
    words
}

#[allow(clippy::too_many_arguments)]
fn encode_with_schedule(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    sched: &Schedule,
    p: &wgpu::Buffer,
    q: Option<&wgpu::Buffer>,
    prepared: Option<&wgpu::Buffer>,
    prep_mod: u32,
    n_pairs: u32,
) -> MillerState {
    let ops = ctx.buffer_from(
        "coop-ops",
        bytemuck::cast_slice(&sched.ops),
        wgpu::BufferUsages::empty(),
    );
    let groups = ctx.buffer_from(
        "coop-groups",
        bytemuck::cast_slice(&sched.groups),
        wgpu::BufferUsages::empty(),
    );
    let consts = ctx.buffer_from(
        "coop-consts",
        bytemuck::cast_slice(&consts_words()),
        wgpu::BufferUsages::empty(),
    );
    // The kernel statically references every binding; unused sides get
    // dummies.
    let dummy = ctx.empty_buffer("coop-dummy", 4, wgpu::BufferUsages::empty());

    let state = MillerState::new_for_coop(ctx, n_pairs);

    let wg_x = n_pairs.min(32768);
    let wg_y = n_pairs.div_ceil(wg_x);
    let params = ctx.buffer_from(
        "coop-params",
        bytemuck::cast_slice(&[
            n_pairs,
            crate::pairing::prepared_line_count(),
            prep_mod,
            sched.groups.len() as u32,
            wg_x,
            u32::from(q.is_some()),
            0,
            0,
        ]),
        wgpu::BufferUsages::UNIFORM,
    );
    let pipeline = ctx.pipeline("coop_pairing", "miller_coop", coop_module_source);
    ctx.encode_pass_indexed(
        encoder,
        &pipeline,
        &[
            (0, &params),
            (1, &state.f),
            (3, q.unwrap_or(&dummy)),
            (5, p),
            (6, &consts),
            (7, prepared.unwrap_or(&dummy)),
            (8, &ops),
            (9, &groups),
        ],
        (wg_x, wg_y, 1),
    );
    state
}

/// Cooperative replacement for `encode_miller_prepared`: pairs
/// (p[i], prepared_q[i % prep_mod]), whole Miller loop in one dispatch.
/// Returns the same `MillerState` (f + reduce scratch) as the sequential
/// encoder, so product reduction and readback are unchanged.
pub fn encode_miller_prepared_coop(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    p: &wgpu::Buffer,
    prepared: &wgpu::Buffer,
    prep_mod: u32,
    n_pairs: u32,
) -> MillerState {
    encode_with_schedule(
        ctx,
        encoder,
        &SCHEDULE,
        p,
        None,
        Some(prepared),
        prep_mod,
        n_pairs,
    )
}

/// Cooperative replacement for `encode_miller_computed`: pairs
/// (p[i], q[i]) with on-device line computation, whole Miller loop in one
/// dispatch. Identity P or Q masks the pair to f = 1.
pub fn encode_miller_computed_coop(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    p: &wgpu::Buffer,
    q: &wgpu::Buffer,
    n_pairs: u32,
) -> MillerState {
    encode_with_schedule(
        ctx,
        encoder,
        &SCHEDULE_COMPUTED,
        p,
        Some(q),
        None,
        1,
        n_pairs,
    )
}

/// Width threshold for the cooperative kernels: below it the sequential
/// pipeline is dispatch-bound (~450 sequential dispatches dominate); above
/// it the per-pair-thread pipeline's full occupancy and register-resident
/// Fq6 chains win over 32-lane cooperation with ~1/3 lane utilization.
pub const COOP_MAX_PAIRS: u32 = 512;

/// Prepared-line Miller loop with automatic dispatch policy.
pub fn encode_miller_prepared_auto(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    p: &wgpu::Buffer,
    prepared: &wgpu::Buffer,
    prep_mod: u32,
    n_pairs: u32,
) -> MillerState {
    if n_pairs <= COOP_MAX_PAIRS {
        encode_miller_prepared_coop(ctx, encoder, p, prepared, prep_mod, n_pairs)
    } else {
        crate::pairing::encode_miller_prepared(ctx, encoder, p, prepared, prep_mod, n_pairs)
    }
}

/// Computed-line Miller loop with automatic dispatch policy.
pub fn encode_miller_computed_auto(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    p: &wgpu::Buffer,
    q: &wgpu::Buffer,
    n_pairs: u32,
) -> MillerState {
    if n_pairs <= COOP_MAX_PAIRS {
        encode_miller_computed_coop(ctx, encoder, p, q, n_pairs)
    } else {
        crate::pairing::encode_miller_computed(ctx, encoder, p, q, n_pairs)
    }
}
