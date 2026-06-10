// Fused single-kernel MSM bucket accumulation (G1). Builds on
// msm_common.wgsl. Buckets live in workgroup shared memory; one dispatch per
// (chunk, window, row) accumulates and reduces to a partial point.
//
// G2 uses the split variant (msm_split.wgsl): its Fq2-inlined version of this
// kernel crashes the Apple Metal shader compiler.
//
// Bases are always affine; the identity is encoded as (0, 0) and skipped via
// select (projective inputs go through normalize.wgsl first).

var<workgroup> PF_sh_buckets: array<u32, MSM_NB * PF_PW>;

fn PF_bucket_load(b: u32) -> PF_Point {
    var w: array<u32, 16>;
    let base = b * PF_PW;
    var p: PF_Point;
    for (var k = 0u; k < PF_FEW; k++) {
        w[k] = PF_sh_buckets[base + k];
    }
    p.x = PF_unpack_words(w);
    for (var k = 0u; k < PF_FEW; k++) {
        w[k] = PF_sh_buckets[base + PF_FEW + k];
    }
    p.y = PF_unpack_words(w);
    for (var k = 0u; k < PF_FEW; k++) {
        w[k] = PF_sh_buckets[base + 2u * PF_FEW + k];
    }
    p.z = PF_unpack_words(w);
    return p;
}

fn PF_bucket_store(b: u32, p: PF_Point) {
    let base = b * PF_PW;
    var w = PF_pack_words(p.x);
    for (var k = 0u; k < PF_FEW; k++) {
        PF_sh_buckets[base + k] = w[k];
    }
    w = PF_pack_words(p.y);
    for (var k = 0u; k < PF_FEW; k++) {
        PF_sh_buckets[base + PF_FEW + k] = w[k];
    }
    w = PF_pack_words(p.z);
    for (var k = 0u; k < PF_FEW; k++) {
        PF_sh_buckets[base + 2u * PF_FEW + k] = w[k];
    }
}

@compute @workgroup_size(64)
fn msm_bucket_acc(
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

    for (var b = tid; b < MSM_NB; b += 64u) {
        PF_bucket_store(b, PF_point_identity());
    }
    for (var e = tid; e < MSM_CHUNK; e += 64u) {
        if (e < count) {
            let sidx = p.scalar_offset + row * p.scalar_stride + start + e;
            PF_sh_digits[e] = PF_msm_digit(sidx, w);
        }
    }
    workgroupBarrier();

    // Each thread owns buckets b with (b & 63) == tid and scans the chunk.
    for (var e = 0u; e < count; e++) {
        let d = PF_sh_digits[e];
        let mag = d >> 1u;
        if (mag != 0u && ((mag - 1u) & 63u) == tid) {
            let b = mag - 1u;
            let q = PF_load_base_affine(p.base_offset + start + e, (d & 1u) == 1u);
            let acc = PF_bucket_load(b);
            let added = PF_point_madd(acc, q);
            let infinity = PF_is_zero(q.x) && PF_is_zero(q.y);
            PF_bucket_store(b, PF_point_select(infinity, acc, added));
        }
    }
    workgroupBarrier();

    // Weighted bucket reduction. Thread t owns buckets {t, t+64}, so
    // sum_b (b+1)*B_b = sum_t [ (t+1)*S_t + 64*B_{t+64} ] with S_t the plain
    // sum of the owned buckets. Each thread reduces its owned buckets in
    // registers, writes its weighted partial to slot t, and one 64-wide tree
    // finishes the window. (Kept deliberately compact: oversized kernels trip
    // an Apple Metal compiler hang.)
    var weighted = PF_point_identity();
    var plain = PF_point_identity();
    for (var b = tid; b < MSM_NB; b += 64u) {
        let v = PF_bucket_load(b);
        plain = PF_point_add(plain, v);
        if (b >= 64u) {
            var scaled = v;
            for (var k = 0u; k < 6u; k++) {
                scaled = PF_point_double(scaled);
            }
            weighted = PF_point_add(weighted, scaled);
        }
    }
    weighted = PF_point_add(weighted, PF_point_mul_small(plain, tid + 1u));
    workgroupBarrier();
    PF_bucket_store(tid, weighted);
    workgroupBarrier();
    for (var off = 32u; off > 0u; off >>= 1u) {
        if (tid < off) {
            PF_bucket_store(tid, PF_point_add(PF_bucket_load(tid), PF_bucket_load(tid + off)));
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        let pidx = (row * MSM_NW + w) * p.n_chunks + chunk;
        PF_partial_store(pidx, PF_bucket_load(0u));
    }
}
