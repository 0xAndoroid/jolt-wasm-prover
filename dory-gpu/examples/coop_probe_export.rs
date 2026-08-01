//! W6-V1p coop-Miller step-0 probe export (.webgpu-w6-specs.md §V1).
//!
//! Serializes the CPU-built coop schedules, the composed WGSL module, and
//! {inputs, arkworks-reference expected f outputs} for batches matching the
//! production burst shapes (512/2048/8192 per-absorb, 32768 = coalesced
//! 4x8192 against the same 8192 prepared tables), as raw LE u32 sidecars +
//! a JSON manifest for the standalone node/Dawn harness.
//!
//! Usage: cargo run --release -p dory-gpu --example coop_probe_export -- [outdir]

use ark_bn254::{Bn254, Fq12, G1Affine, G2Affine};
use ark_ec::pairing::Pairing;
use ark_ff::UniformRand;
use dory_gpu::coop::{export_consts_words, export_module_source, export_schedule};
use dory_gpu::pairing::{pack_prepared_g2, prepared_line_count};
use dory_gpu::repr::{fq12_to_words, g1_affine_to_words, g2_affine_to_words};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const SEED: u64 = 4242;
const MAX_PAIRS: usize = 32768;
const MAX_PREP: usize = 8192;
const PREP_SIZES: [(usize, usize); 4] = [(512, 512), (2048, 2048), (8192, 8192), (32768, 8192)];
const COMP_SIZES: [usize; 3] = [512, 2048, 8192];

fn write_words(dir: &Path, name: &str, words: &[u32]) {
    fs::write(dir.join(name), bytemuck::cast_slice::<u32, u8>(words)).expect("write");
    eprintln!(
        "  {name}: {} words ({} bytes)",
        words.len(),
        words.len() * 4
    );
}

fn expected_words(fs12: &[Fq12]) -> Vec<u32> {
    let mut out = Vec::with_capacity(fs12.len() * 96);
    for f in fs12 {
        out.extend_from_slice(&fq12_to_words(f));
    }
    out
}

fn main() {
    let out = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "/tmp/coop-probe-data".into()),
    );
    fs::create_dir_all(&out).expect("outdir");
    let stride = prepared_line_count();

    let (ops_p, groups_p) = export_schedule(false);
    let (ops_c, groups_c) = export_schedule(true);
    eprintln!(
        "schedules: prepared {} ops / {} groups, computed {} ops / {} groups, stride {stride}",
        ops_p.len(),
        groups_p.len(),
        ops_c.len(),
        groups_c.len()
    );
    fs::write(out.join("coop_module.wgsl"), export_module_source()).expect("wgsl");
    write_words(&out, "ops_prep.bin", bytemuck::cast_slice(&ops_p));
    write_words(&out, "groups_prep.bin", &groups_p);
    write_words(&out, "ops_comp.bin", bytemuck::cast_slice(&ops_c));
    write_words(&out, "groups_comp.bin", &groups_c);
    write_words(&out, "consts.bin", &export_consts_words());

    let mut rng = ChaCha20Rng::seed_from_u64(SEED);
    let ps: Vec<G1Affine> = (0..MAX_PAIRS).map(|_| G1Affine::rand(&mut rng)).collect();
    let qs: Vec<G2Affine> = (0..MAX_PREP).map(|_| G2Affine::rand(&mut rng)).collect();

    let t = Instant::now();
    let p_words: Vec<u32> = ps.iter().flat_map(g1_affine_to_words).collect();
    write_words(&out, "p_main.bin", &p_words);
    let q_words: Vec<u32> = qs.iter().flat_map(g2_affine_to_words).collect();
    write_words(&out, "q_main.bin", &q_words);
    let prepared_words = pack_prepared_g2(&qs);
    write_words(&out, "prepared_main.bin", &prepared_words);
    eprintln!("inputs packed in {:.1}s", t.elapsed().as_secs_f32());

    let t = Instant::now();
    let preps: Vec<ark_ec::bn::G2Prepared<ark_bn254::Config>> =
        qs.par_iter().map(|q| (*q).into()).collect();
    for (n, prep_mod) in PREP_SIZES {
        let fs12: Vec<Fq12> = (0..n)
            .into_par_iter()
            .map(|i| Bn254::multi_miller_loop([ps[i]], [preps[i % prep_mod].clone()]).0)
            .collect();
        write_words(
            &out,
            &format!("expected_prep_{n}.bin"),
            &expected_words(&fs12),
        );
    }
    for n in COMP_SIZES {
        let fs12: Vec<Fq12> = (0..n)
            .into_par_iter()
            .map(|i| Bn254::multi_miller_loop([ps[i]], [qs[i]]).0)
            .collect();
        write_words(
            &out,
            &format!("expected_comp_{n}.bin"),
            &expected_words(&fs12),
        );
    }
    eprintln!("expected outputs in {:.1}s", t.elapsed().as_secs_f32());

    // Mask-path KATs (mirrors tests/pairing.rs): identity rows keep f = 1.
    let kat_n = 8usize;
    let one = Fq12::from(1u64);
    {
        let mut rows: Vec<u32> = Vec::with_capacity(kat_n * 16);
        let mut exp: Vec<Fq12> = Vec::with_capacity(kat_n);
        for i in 0..kat_n {
            if i == 3 {
                rows.extend_from_slice(&[0u32; 16]);
                exp.push(one);
            } else {
                rows.extend_from_slice(&g1_affine_to_words(&ps[i]));
                exp.push(Bn254::multi_miller_loop([ps[i]], [preps[i].clone()]).0);
            }
        }
        write_words(&out, "p_kat_prep.bin", &rows);
        write_words(&out, "expected_kat_prep.bin", &expected_words(&exp));
    }
    {
        let mut p_rows: Vec<u32> = Vec::with_capacity(kat_n * 16);
        let mut q_rows: Vec<u32> = Vec::with_capacity(kat_n * 32);
        let mut exp: Vec<Fq12> = Vec::with_capacity(kat_n);
        for i in 0..kat_n {
            let p_id = i == 1;
            let q_id = i == 3;
            if p_id {
                p_rows.extend_from_slice(&[0u32; 16]);
            } else {
                p_rows.extend_from_slice(&g1_affine_to_words(&ps[i]));
            }
            if q_id {
                q_rows.extend_from_slice(&[0u32; 32]);
            } else {
                q_rows.extend_from_slice(&g2_affine_to_words(&qs[i]));
            }
            if p_id || q_id {
                exp.push(one);
            } else {
                exp.push(Bn254::multi_miller_loop([ps[i]], [qs[i]]).0);
            }
        }
        write_words(&out, "p_kat_comp.bin", &p_rows);
        write_words(&out, "q_kat_comp.bin", &q_rows);
        write_words(&out, "expected_kat_comp.bin", &expected_words(&exp));
    }

    let prep_cfgs: Vec<String> = PREP_SIZES
        .iter()
        .map(|(n, m)| format!(r#"{{"kind":"prep","n":{n},"prep_mod":{m}}}"#))
        .collect();
    let comp_cfgs: Vec<String> = COMP_SIZES
        .iter()
        .map(|n| format!(r#"{{"kind":"comp","n":{n},"prep_mod":1}}"#))
        .collect();
    let manifest = format!(
        r#"{{
  "seed": {SEED},
  "stride": {stride},
  "prep": {{"ops": {}, "groups": {}}},
  "comp": {{"ops": {}, "groups": {}}},
  "kat_n": {kat_n},
  "configs": [{},
    {}]
}}
"#,
        ops_p.len(),
        groups_p.len(),
        ops_c.len(),
        groups_c.len(),
        prep_cfgs.join(",\n    "),
        comp_cfgs.join(",\n    ")
    );
    fs::write(out.join("manifest.json"), manifest).expect("manifest");
    eprintln!("export complete -> {}", out.display());
}
