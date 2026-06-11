// Joint RLC matrix: out[r][col] = sum over polys of coeff_p * M_p[r][col],
// where dense polys store Fr entries row-major and one-hot polys store a
// bucket index per (chunk, column) — their matrix entry is 1 at row
// k * rows_per_k + c when indices[c * cols + col] == k, so the contribution
// is just the (Montgomery) coefficient. One thread per output cell,
// dispatched 2D: x over columns, y over rows.
//
// Meta layout, RLC_META_STRIDE u32 per poly:
//   [0] kind (0 dense, 1 one-hot)
//   [1] data offset (Fe8 index for dense, u32 index for one-hot)
//   [2] dense row count (kind 0 only)
//   [3] padding
//   [4..12] coefficient, Montgomery Fe8

struct RlcParams {
    num_rows: u32,
    cols: u32,
    rows_per_k: u32,
    n_polys: u32,
}

@group(0) @binding(0) var<uniform> rlc_params: RlcParams;
@group(0) @binding(1) var<storage, read> rlc_meta: array<u32>;
@group(0) @binding(2) var<storage, read> rlc_dense: array<u32>;
@group(0) @binding(3) var<storage, read> rlc_onehot: array<u32>;
@group(0) @binding(4) var<storage, read_write> rlc_out: array<u32>;

fn rlc_load_dense(idx: u32) -> Fe {
    var w: Fe8;
    for (var t = 0u; t < 8u; t++) {
        w[t] = rlc_dense[idx * 8u + t];
    }
    return fe_unpack(w);
}

fn rlc_load_coeff(m: u32) -> Fe {
    var w: Fe8;
    for (var t = 0u; t < 8u; t++) {
        w[t] = rlc_meta[m + 4u + t];
    }
    return fe_unpack(w);
}

@compute @workgroup_size(64)
fn rlc_combine(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let r = gid.y;
    let p = rlc_params;
    if (col >= p.cols || r >= p.num_rows) {
        return;
    }
    let k = r / p.rows_per_k;
    let c = r % p.rows_per_k;
    var acc = fe_zero();
    for (var i = 0u; i < p.n_polys; i++) {
        let m = i * 12u;
        let kind = rlc_meta[m];
        let off = rlc_meta[m + 1u];
        if (kind == 0u) {
            if (r < rlc_meta[m + 2u]) {
                let entry = rlc_load_dense(off + r * p.cols + col);
                acc = fe_add(acc, fe_mont_mul(rlc_load_coeff(m), entry));
            }
        } else if (rlc_onehot[off + c * p.cols + col] == k) {
            acc = fe_add(acc, rlc_load_coeff(m));
        }
    }
    let w = fe_pack(acc);
    let base = (r * p.cols + col) * 8u;
    for (var t = 0u; t < 8u; t++) {
        rlc_out[base + t] = w[t];
    }
}
