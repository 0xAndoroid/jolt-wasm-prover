// Split-pipeline MSM used for G2: the Fq2-inlined fused kernel crashes
// Apple's Metal shader compiler, so accumulation runs against buckets in
// GLOBAL memory across three small kernels (clear / accumulate / reduce),
// followed by the same window combine as the fused path. G2 MSMs are always
// rows = 1 and n <= 2^sigma in Dory, so the global bucket buffer stays small
// (windows * chunks * 64 * 192 B).
//
// Buffer/bindings/layout shared with msm.wgsl conventions:
// 0 params, 1 bases (affine), 2 canon scalars, 3 carry masks,
// 4 partials (rw), 5 results (rw), 6 buckets (rw).

@group(0) @binding(6) var<storage, read_write> PF_msm_buckets: array<u32>;

var<workgroup> PF_sh_red: array<u32, 64u * PF_PW>;

fn PF_gbucket_load(idx: u32) -> PF_Point {
    var w: array<u32, 16>;
    let base = idx * PF_PW;
    var p: PF_Point;
    for (var k = 0u; k < PF_FEW; k++) {
        w[k] = PF_msm_buckets[base + k];
    }
    p.x = PF_unpack_words(w);
    for (var k = 0u; k < PF_FEW; k++) {
        w[k] = PF_msm_buckets[base + PF_FEW + k];
    }
    p.y = PF_unpack_words(w);
    for (var k = 0u; k < PF_FEW; k++) {
        w[k] = PF_msm_buckets[base + 2u * PF_FEW + k];
    }
    p.z = PF_unpack_words(w);
    return p;
}

fn PF_gbucket_store(idx: u32, p: PF_Point) {
    let base = idx * PF_PW;
    var w = PF_pack_words(p.x);
    for (var k = 0u; k < PF_FEW; k++) {
        PF_msm_buckets[base + k] = w[k];
    }
    w = PF_pack_words(p.y);
    for (var k = 0u; k < PF_FEW; k++) {
        PF_msm_buckets[base + PF_FEW + k] = w[k];
    }
    w = PF_pack_words(p.z);
    for (var k = 0u; k < PF_FEW; k++) {
        PF_msm_buckets[base + 2u * PF_FEW + k] = w[k];
    }
}

fn PF_shred_load(slot: u32) -> PF_Point {
    var w: array<u32, 16>;
    let base = slot * PF_PW;
    var p: PF_Point;
    for (var k = 0u; k < PF_FEW; k++) {
        w[k] = PF_sh_red[base + k];
    }
    p.x = PF_unpack_words(w);
    for (var k = 0u; k < PF_FEW; k++) {
        w[k] = PF_sh_red[base + PF_FEW + k];
    }
    p.y = PF_unpack_words(w);
    for (var k = 0u; k < PF_FEW; k++) {
        w[k] = PF_sh_red[base + 2u * PF_FEW + k];
    }
    p.z = PF_unpack_words(w);
    return p;
}

fn PF_shred_store(slot: u32, p: PF_Point) {
    let base = slot * PF_PW;
    var w = PF_pack_words(p.x);
    for (var k = 0u; k < PF_FEW; k++) {
        PF_sh_red[base + k] = w[k];
    }
    w = PF_pack_words(p.y);
    for (var k = 0u; k < PF_FEW; k++) {
        PF_sh_red[base + PF_FEW + k] = w[k];
    }
    w = PF_pack_words(p.z);
    for (var k = 0u; k < PF_FEW; k++) {
        PF_sh_red[base + 2u * PF_FEW + k] = w[k];
    }
}

@compute @workgroup_size(64)
fn msm_clear_buckets(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = PF_msm_params;
    let total = p.rows * MSM_NW * p.n_chunks * MSM_NB;
    if (gid.x < total) {
        PF_gbucket_store(gid.x, PF_point_identity());
    }
}

@compute @workgroup_size(64)
fn msm_acc_global(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let chunk = wid.x;
    let w = wid.y;
    let row = wid.z;
    let tid = lid.x;
    let p = PF_msm_params;

    let start = chunk * MSM_CHUNK;
    let count = min(MSM_CHUNK, max(p.n, start) - start);

    for (var e = tid; e < MSM_CHUNK; e += 64u) {
        if (e < count) {
            let sidx = p.scalar_offset + row * p.scalar_stride + start + e;
            PF_sh_digits[e] = PF_msm_digit(sidx, w);
        }
    }
    workgroupBarrier();

    let bucket_base = ((row * MSM_NW + w) * p.n_chunks + chunk) * MSM_NB;
    for (var e = 0u; e < count; e++) {
        let d = PF_sh_digits[e];
        let mag = d >> 1u;
        if (mag != 0u && ((mag - 1u) & 63u) == tid) {
            let b = mag - 1u;
            let q = PF_load_base_affine(p.base_offset + start + e, (d & 1u) == 1u);
            let acc = PF_gbucket_load(bucket_base + b);
            let added = PF_point_madd(acc, q);
            let infinity = PF_is_zero(q.x) && PF_is_zero(q.y);
            PF_gbucket_store(bucket_base + b, PF_point_select(infinity, acc, added));
        }
    }
}

// In-place weighting: bucket[i] <- ((i % MSM_NB) + 1) * bucket[i]. Pure
// per-thread work, no shared memory — kept as its own kernel because the
// Apple Metal compiler crashes on larger Fq2 kernels.
@compute @workgroup_size(64)
fn msm_weight_global(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = PF_msm_params;
    let total = p.rows * MSM_NW * p.n_chunks * MSM_NB;
    if (gid.x < total) {
        let k = (gid.x % MSM_NB) + 1u;
        PF_gbucket_store(gid.x, PF_point_mul_small(PF_gbucket_load(gid.x), k));
    }
}

// Tree-sum of the (already weighted) buckets of one (row, window, chunk)
// into a partial point. Requires MSM_NB <= 64.
@compute @workgroup_size(64)
fn msm_sum_global(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let chunk = wid.x;
    let w = wid.y;
    let row = wid.z;
    let tid = lid.x;
    let p = PF_msm_params;

    let bucket_base = ((row * MSM_NW + w) * p.n_chunks + chunk) * MSM_NB;
    var v = PF_point_identity();
    if (tid < MSM_NB) {
        v = PF_gbucket_load(bucket_base + tid);
    }
    PF_shred_store(tid, v);
    workgroupBarrier();
    for (var off = 32u; off > 0u; off >>= 1u) {
        if (tid < off) {
            PF_shred_store(tid, PF_point_add(PF_shred_load(tid), PF_shred_load(tid + off)));
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        let pidx = (row * MSM_NW + w) * p.n_chunks + chunk;
        PF_partial_store(pidx, PF_shred_load(0u));
    }
}
