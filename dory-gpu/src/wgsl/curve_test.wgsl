// Test entry points for curve.wgsl instantiations (PF_ substituted to g1_/g2_).
// Points are packed field words: projective = 3 elements, affine = 2 elements.
// PF_FEW is replaced with the packed u32 word count of the point field.

@group(0) @binding(0) var<storage, read> in_a: array<u32>;
@group(0) @binding(1) var<storage, read> in_b: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

fn PF_t_load_fe(buf_sel: u32, slot: u32) -> PF_Fe {
    let words = PF_FEW;
    var packed: array<u32, 16>;
    for (var k = 0u; k < words; k++) {
        if (buf_sel == 0u) {
            packed[k] = in_a[slot * words + k];
        } else {
            packed[k] = in_b[slot * words + k];
        }
    }
    return PF_unpack_words(packed);
}

fn PF_t_store_fe(slot: u32, v: PF_Fe) {
    let packed = PF_pack_words(v);
    let words = PF_FEW;
    for (var k = 0u; k < words; k++) {
        out[slot * words + k] = packed[k];
    }
}

fn PF_t_load_point(buf_sel: u32, idx: u32) -> PF_Point {
    return PF_Point(
        PF_t_load_fe(buf_sel, idx * 3u),
        PF_t_load_fe(buf_sel, idx * 3u + 1u),
        PF_t_load_fe(buf_sel, idx * 3u + 2u),
    );
}

fn PF_t_load_affine(buf_sel: u32, idx: u32) -> PF_Affine {
    return PF_Affine(
        PF_t_load_fe(buf_sel, idx * 2u),
        PF_t_load_fe(buf_sel, idx * 2u + 1u),
    );
}

fn PF_t_store_point(idx: u32, p: PF_Point) {
    PF_t_store_fe(idx * 3u, p.x);
    PF_t_store_fe(idx * 3u + 1u, p.y);
    PF_t_store_fe(idx * 3u + 2u, p.z);
}

@compute @workgroup_size(64)
fn t_point_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i * 3u * PF_FEW >= arrayLength(&out)) { return; }
    PF_t_store_point(i, PF_point_add(PF_t_load_point(0u, i), PF_t_load_point(1u, i)));
}

@compute @workgroup_size(64)
fn t_point_madd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i * 3u * PF_FEW >= arrayLength(&out)) { return; }
    PF_t_store_point(i, PF_point_madd(PF_t_load_point(0u, i), PF_t_load_affine(1u, i)));
}

@compute @workgroup_size(64)
fn t_point_double(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i * 3u * PF_FEW >= arrayLength(&out)) { return; }
    PF_t_store_point(i, PF_point_double(PF_t_load_point(0u, i)));
}

// Scalar mul by plain double-and-add over 256 bits; scalar packed canonical in in_b.
@compute @workgroup_size(64)
fn t_point_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i * 3u * PF_FEW >= arrayLength(&out)) { return; }
    let p = PF_t_load_point(0u, i);
    var acc = PF_point_identity();
    for (var bit = 0i; bit < 256; bit++) {
        let b = 255u - u32(bit);
        acc = PF_point_double(acc);
        let word = in_b[i * 8u + b / 32u];
        let is_set = ((word >> (b % 32u)) & 1u) == 1u;
        let added = PF_point_add(acc, p);
        acc = PF_point_select(is_set, added, acc);
    }
    PF_t_store_point(i, acc);
}
