// Workgroup-cooperative prepared-line Miller loop: one workgroup of
// COOP_LANES lanes per pair runs the ENTIRE ate loop in a single dispatch,
// replacing the ~450-dispatch-per-multipairing sequential structure.
//
// The loop is a stream of Fq2 micro-ops (add/sub/xi-combine/double/mul/
// line-load) over a shared-memory slot file, grouped into barrier-separated
// batches of independent ops. The stream is BUILT ON THE CPU (coop.rs
// mirrors the exact fq6_mul / mul_by_01 / mul_by_034 / squaring formulas of
// fq12.wgsl + pairing_fq12.wgsl) and read from a storage buffer, so the
// kernel body stays tiny — one fq2_mul call site — which keeps the Apple
// Metal compiler happy, and barriers sit in trivially uniform control flow.
//
// Bindings: 0 params, 1 f out (96 words/pair), 5 p (affine G1, 16 words),
// 7 prepared lines, 8 op stream, 9 group spans.
//
// Op encoding (vec4<u32>): (dst_slot, a, b, kind)
//   kind 0:  slot[dst] = slot[a] + slot[b]
//   kind 1:  slot[dst] = slot[a] - slot[b]
//   kind 2:  slot[dst] = slot[a] - xi*slot[b]
//   kind 3:  slot[dst] = slot[a] + xi*slot[b]
//   kind 4:  slot[dst] = 2*slot[a]
//   kind 5:  slot[dst] = slot[a] * slot[b]
//   kind 6:  slot[dst] = Fq2(one, zero) if a == 1 else Fq2(zero, zero)
//   kind 7:  slot[dst] = prepared line component b of line index a
//   kind 8:  slot[dst] = (P.x, 0) if a == 0 else (P.y, 0); also evaluates
//            the pair mask (P == (0,0), or-ed with Q == 0 when check_q is
//            set) when dst == 9 (the P.x load)
//   kind 9:  slot[dst] = consts[a] (two_inv, 3b', twist-Frobenius consts)
//   kind 10: slot[dst] = conjugate(slot[a])
//   kind 11: slot[dst] = Q component a (0 = x, 1 = y) of this pair
// Group span encoding (u32): (first_op << 8) | op_count.

struct CoopParams {
    n_pairs: u32,
    prep_stride: u32,
    prep_mod: u32,
    n_groups: u32,
    wg_x: u32,
    check_q: u32,
    _p0: u32,
    _p1: u32,
}

@group(0) @binding(0) var<uniform> cp: CoopParams;
@group(0) @binding(1) var<storage, read_write> coop_f: array<u32>;
@group(0) @binding(3) var<storage, read> coop_q: array<u32>;
@group(0) @binding(5) var<storage, read> coop_p: array<u32>;
@group(0) @binding(6) var<storage, read> coop_consts: array<u32>;
@group(0) @binding(7) var<storage, read> coop_prepared: array<u32>;
@group(0) @binding(8) var<storage, read> coop_ops: array<vec4<u32>>;
@group(0) @binding(9) var<storage, read> coop_groups: array<u32>;

const COOP_LANES: u32 = 32u;
const COOP_SLOTS: u32 = 56u;

var<workgroup> coop_slot: array<u32, 896>; // COOP_SLOTS * 16 packed words
var<workgroup> coop_masked: u32;

fn slot_load(idx: u32) -> Fe2 {
    var w0: Fe8;
    var w1: Fe8;
    let base = idx * 16u;
    for (var k = 0u; k < 8u; k++) {
        w0[k] = coop_slot[base + k];
        w1[k] = coop_slot[base + 8u + k];
    }
    return Fe2(fe_unpack(w0), fe_unpack(w1));
}

fn slot_store(idx: u32, v: Fe2) {
    let w0 = fe_pack(v.c0);
    let w1 = fe_pack(v.c1);
    let base = idx * 16u;
    for (var k = 0u; k < 8u; k++) {
        coop_slot[base + k] = w0[k];
        coop_slot[base + 8u + k] = w1[k];
    }
}

fn p_fe(pair: u32, comp: u32) -> Fe {
    var w: Fe8;
    for (var k = 0u; k < 8u; k++) {
        w[k] = coop_p[pair * 16u + comp * 8u + k];
    }
    return fe_unpack(w);
}

fn line_fe2(pair: u32, line: u32, comp: u32) -> Fe2 {
    let base = ((pair % cp.prep_mod) * cp.prep_stride + line) * 48u + comp * 16u;
    var w0: Fe8;
    var w1: Fe8;
    for (var k = 0u; k < 8u; k++) {
        w0[k] = coop_prepared[base + k];
        w1[k] = coop_prepared[base + 8u + k];
    }
    return Fe2(fe_unpack(w0), fe_unpack(w1));
}

fn q_fe2(pair: u32, comp: u32) -> Fe2 {
    let base = pair * 32u + comp * 16u;
    var w0: Fe8;
    var w1: Fe8;
    for (var k = 0u; k < 8u; k++) {
        w0[k] = coop_q[base + k];
        w1[k] = coop_q[base + 8u + k];
    }
    return Fe2(fe_unpack(w0), fe_unpack(w1));
}

fn const_fe2(idx: u32) -> Fe2 {
    var w0: Fe8;
    var w1: Fe8;
    for (var k = 0u; k < 8u; k++) {
        w0[k] = coop_consts[idx * 16u + k];
        w1[k] = coop_consts[idx * 16u + 8u + k];
    }
    return Fe2(fe_unpack(w0), fe_unpack(w1));
}

@compute @workgroup_size(32)
fn miller_coop(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let pair = wid.y * cp.wg_x + wid.x;
    let lane = lid.x;
    if (pair >= cp.n_pairs) {
        return;
    }

    for (var g = 0u; g < cp.n_groups; g++) {
        let span = coop_groups[g];
        let first = span >> 8u;
        let count = span & 0xffu;
        let skip = g > 0u && coop_masked == 1u;
        if (lane < count && !skip) {
            let op = coop_ops[first + lane];
            var r: Fe2;
            switch op.w {
                case 0u: {
                    r = fq2_add(slot_load(op.y), slot_load(op.z));
                }
                case 1u: {
                    r = fq2_sub(slot_load(op.y), slot_load(op.z));
                }
                case 2u: {
                    r = fq2_sub(slot_load(op.y), fq2_mul_by_nonresidue(slot_load(op.z)));
                }
                case 3u: {
                    r = fq2_add(slot_load(op.y), fq2_mul_by_nonresidue(slot_load(op.z)));
                }
                case 4u: {
                    r = fq2_double(slot_load(op.y));
                }
                case 5u: {
                    r = fq2_mul(slot_load(op.y), slot_load(op.z));
                }
                case 6u: {
                    if (op.y == 1u) {
                        r = fq2_mont_one();
                    } else {
                        r = fq2_zero();
                    }
                }
                case 7u: {
                    r = line_fe2(pair, op.y, op.z);
                }
                case 8u: {
                    let v = p_fe(pair, op.y);
                    r = Fe2(v, fe_zero());
                    if (op.x == 9u) {
                        let px = p_fe(pair, 0u);
                        let py = p_fe(pair, 1u);
                        var masked = fe_is_zero(px) && fe_is_zero(py);
                        if (cp.check_q == 1u) {
                            var qz = true;
                            for (var k = 0u; k < 32u; k++) {
                                qz = qz && (coop_q[pair * 32u + k] == 0u);
                            }
                            masked = masked || qz;
                        }
                        coop_masked = u32(masked);
                    }
                }
                case 9u: {
                    r = const_fe2(op.y);
                }
                case 10u: {
                    let v = slot_load(op.y);
                    r = Fe2(v.c0, fe_neg(v.c1));
                }
                case 11u, default: {
                    r = q_fe2(pair, op.y);
                }
            }
            slot_store(op.x, r);
        }
        workgroupBarrier();
    }

    // f = slots 0..5 (masked pairs kept the initial f = 1).
    if (lane < 6u) {
        let v = slot_load(lane);
        let w0 = fe_pack(v.c0);
        let w1 = fe_pack(v.c1);
        let base = pair * 96u + lane * 16u;
        for (var k = 0u; k < 8u; k++) {
            coop_f[base + k] = w0[k];
            coop_f[base + 8u + k] = w1[k];
        }
    }
}
