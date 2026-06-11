// One-hot tier-1 row commitments, striped for occupancy: phase A spreads
// each output row's column scan across ONEHOT_STRIPES threads writing
// projective partials; phase B reduces the stripes per row. One thread per
// row alone (a few thousand threads) starves the GPU — striping multiplies
// parallelism by ONEHOT_STRIPES at the cost of (stripes - 1) extra point
// adds per row. Bases are setup generators (never identity); a single
// mixed-add call site per kernel keeps both within the Apple Metal size
// envelope.
//
// Index layout: indices[c * cols + col] = bucket in [0, K) or ONEHOT_NONE.
// Output row layout matches jolt-core's scatter: row = k * rows_per_k + c.
// Both phases are dispatched inside one compute pass (WebGPU orders
// storage writes between dispatches of a pass), sharing one scratch buffer
// across polynomials.

const ONEHOT_STRIPES: u32 = 32u;

struct OnehotParams {
    cols: u32,
    rows_per_k: u32,
    num_rows: u32,
    out_offset: u32,
}

@group(0) @binding(0) var<uniform> PF_oh_params: OnehotParams;
@group(0) @binding(1) var<storage, read> PF_oh_indices: array<u32>;
@group(0) @binding(2) var<storage, read> PF_oh_bases: array<u32>;
@group(0) @binding(3) var<storage, read_write> PF_oh_out: array<u32>;
@group(0) @binding(4) var<storage, read_write> PF_oh_partials: array<u32>;

fn PF_oh_load_fe(base: u32) -> PF_Fe {
    var w: array<u32, 16>;
    for (var k = 0u; k < PF_FEW; k++) {
        w[k] = PF_oh_bases[base + k];
    }
    return PF_unpack_words(w);
}

fn PF_oh_load_affine(idx: u32) -> PF_Affine {
    let base = idx * 2u * PF_FEW;
    return PF_Affine(PF_oh_load_fe(base), PF_oh_load_fe(base + PF_FEW));
}

fn PF_oh_partial_load(idx: u32) -> PF_Point {
    var w: array<u32, 16>;
    let base = idx * 3u * PF_FEW;
    var p: PF_Point;
    for (var k = 0u; k < PF_FEW; k++) {
        w[k] = PF_oh_partials[base + k];
    }
    p.x = PF_unpack_words(w);
    for (var k = 0u; k < PF_FEW; k++) {
        w[k] = PF_oh_partials[base + PF_FEW + k];
    }
    p.y = PF_unpack_words(w);
    for (var k = 0u; k < PF_FEW; k++) {
        w[k] = PF_oh_partials[base + 2u * PF_FEW + k];
    }
    p.z = PF_unpack_words(w);
    return p;
}

fn PF_oh_partial_store(idx: u32, p: PF_Point) {
    let base = idx * 3u * PF_FEW;
    var w = PF_pack_words(p.x);
    for (var k = 0u; k < PF_FEW; k++) {
        PF_oh_partials[base + k] = w[k];
    }
    w = PF_pack_words(p.y);
    for (var k = 0u; k < PF_FEW; k++) {
        PF_oh_partials[base + PF_FEW + k] = w[k];
    }
    w = PF_pack_words(p.z);
    for (var k = 0u; k < PF_FEW; k++) {
        PF_oh_partials[base + 2u * PF_FEW + k] = w[k];
    }
}

// Phase A: thread (r, s) accumulates the columns of stripe s for output
// row r into partials[r * ONEHOT_STRIPES + s].
@compute @workgroup_size(64)
fn onehot_partials(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    let s = gid.y;
    let p = PF_oh_params;
    if (r >= p.num_rows) {
        return;
    }
    let k = r / p.rows_per_k;
    let c = r % p.rows_per_k;
    let span = (p.cols + ONEHOT_STRIPES - 1u) / ONEHOT_STRIPES;
    let start = s * span;
    let end = min(start + span, p.cols);
    var acc = PF_point_identity();
    for (var col = start; col < end; col++) {
        if (PF_oh_indices[c * p.cols + col] == k) {
            acc = PF_point_madd(acc, PF_oh_load_affine(col));
        }
    }
    PF_oh_partial_store(r * ONEHOT_STRIPES + s, acc);
}

// Phase B: thread r folds its ONEHOT_STRIPES partials into the output row.
@compute @workgroup_size(64)
fn onehot_reduce(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    let p = PF_oh_params;
    if (r >= p.num_rows) {
        return;
    }
    var acc = PF_point_identity();
    for (var s = 0u; s < ONEHOT_STRIPES; s++) {
        acc = PF_point_add(acc, PF_oh_partial_load(r * ONEHOT_STRIPES + s));
    }
    let base = (p.out_offset + r) * 3u * PF_FEW;
    var w = PF_pack_words(acc.x);
    for (var k = 0u; k < PF_FEW; k++) {
        PF_oh_out[base + k] = w[k];
    }
    w = PF_pack_words(acc.y);
    for (var k = 0u; k < PF_FEW; k++) {
        PF_oh_out[base + PF_FEW + k] = w[k];
    }
    w = PF_pack_words(acc.z);
    for (var k = 0u; k < PF_FEW; k++) {
        PF_oh_out[base + 2u * PF_FEW + k] = w[k];
    }
}
