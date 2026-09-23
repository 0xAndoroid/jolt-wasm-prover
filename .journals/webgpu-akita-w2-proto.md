---
created: 2026-09-23
updated: 2026-09-23
tags: [webgpu, akita, prototype]
---
# W2 prototype — digit-range sumcheck rounds on WebGPU (standalone)

Goal: measure whether akita's stage-8 `digit_range_prove` (four b=8 direct-leaf instances,
rounds 24/21/20/19, ≈0.50 s CPU-WASM at 2^18) fits the W2 kill rule (save ≥0.42 s) when run
as dependent WGSL rounds under WebKit. No Rust, no wasm, nothing under `src/` or `frontend/`.

Steps: see `.audit/webgpu-akita-w2-proto.tsv`.

## Design in one paragraph
Digits stay compact for rounds 0–2 (round 0: private 16-class histogram + signed LUT; round 1:
workgroup atomic histogram over 256 octet-classes + GPU-built LUT; round 2: pairs from octet
class bytes via a 256-entry folded LUT). Round 3 fuses materialization (LUT + fold by r2) with
its evaluation; rounds ≥4 are one fused fold+eval dispatch over a ping-pong pair of fp128 tables.
Every round ends in a 1-workgroup reduce, an 80 B copy and a `mapAsync` readback; the host
derives the next challenge (murmur-fmix stand-in or a fixed pre-drawn list) and writes only `r`
into the next round's uniform. Gruen split-eq uses a fixed split so the eq tables are uploaded
once per instance.

## Handoff notes
- `bench/proto-w2/README.md` holds the numbers and the GO/PARK verdict.
- Parity items integration must close are listed under "Deviations" in the README; the biggest
  is the challenge derivation (Blake2b transcript vs the stand-in hash) and the 5-coefficient
  message format.
- `--shape small` is the exact-reference gate; run it after any kernel edit.
