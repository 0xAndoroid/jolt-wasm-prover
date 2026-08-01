# W6-V45 evidence — V4 (guard-trip cheapening) + V5 (zk×webgpu battery)

jolt `webgpu/v45-riders` @5ae2ced17 (3 commits on wave2 63b389d31), demo `w6/v45-verify`.
Clear wasm identity `d98af47f470c60cb…`; zk wasm identity `e5acbaa251b6ac62…`.

## V4 — forced-trip cost @2^22 (interleaved same-window rounds, cold sessions, lock + ioreg gate, identity-verified :8181/:8182/:8183, PW_CHANNEL=chrome, /bench.html)

Throwaway trip-once builds (never committed): BEFORE = wave2 @63b389d31 + hack
(id c89765deda2c…), AFTER = V4 @fcd9bcaa7 + hack (id baf4397a8fe1…), CLEAN =
@fcd9bcaa7 (id 55b2a3226e76…). One trip per session console-confirmed
(clean=0, trip arms=1); all 9 proofs valid.

| arm | r1 | r2 | r3 | med | trip cost |
|---|---|---|---|---|---|
| clean | 32.43 | 30.98 | 31.08 | 31.08 | — |
| BEFORE (mixed adds) | 38.47 | 38.57 | 36.38 | 38.47 | **+7.39s** (paired +6.03/+7.59/+5.30) |
| AFTER (batch-affine) | 35.43 | 34.69 | 34.99 | 34.99 | **+3.91s** (paired +3.00/+3.72/+3.91) |

Trip cost −3.5s ≈ −47% (target was +4–5s; landed +3.9). Reproduces U4's +7.1s prior on the BEFORE arm.

## V5 — zk×webgpu battery (all green)

1. **Lockstep byte gate (jolt-kernels, zk build):** full suite `--features webgpu,zk`
   **208/208** (7 skipped = standing quarantines). The 4-arm commit battery
   (xyzz/jacobian/barrier-walk/forced-trip) under zk asserts: tier-1 hint rows
   EXACT vs optimized twin (the whole device surface — blind enters only at
   masked tier-2), blinds nonzero (catches silently-transparent zk) and
   arm-distinct, commitments arm-distinct, device engagement counter ≥1.
   Trip arm ⇒ the V4 batch-affine patch-up is byte-correct under zk too.
2. **Native zk e2e:** `-p jolt-prover --features prover-fixtures,zk,webgpu`
   zk_e2e **7/7** incl. new `zk_muldiv_webgpu_backend_proof_is_accepted` +
   `zk_advice_consumer_webgpu_backend_proof_is_accepted` — valid + Zk claims +
   verifier accepts + engagement (commit_consume ≥1, 0 patch-ups, 0 warns via
   process-global counting subscriber).
3. **Browser zk arms @2^16** (zk build e5acbaa2, :8184, exit 0):
   4/4 valid (off×2, on×2), hiding-live (4 distinct proof SHAs), GPU engaged
   (miller 101/run), proof sizes exactly equal 77,169B, no BlindFold stack
   trap. Walls: off 8.93/18.17, on 9.63/9.22 (zk tail ≈ +6s serial CPU at
   this scale — validity datum only).
4. **Clear mode unchanged:** native off-vs-forced bytes IDENTICAL ×2 guests =
   EXACT U0 pins (sha2@2^13 be302f855b01e22b…, sha2_chain@2^19
   f4bbda7c44c193ce…), arm-active ×2; browser 7-arm gate @2^16 exit 0, SHA
   3cc194fe018e64c1 = exact wave-2 canonical on all arms.

## Determinism (battery item 4)

OsRng is hardwired in upstream jolt-prover (recorder.rs ModeRecorder alias +
ctor sites, stages/drivers.rs:955, blindfold.rs:141/162) and vendored dory's
Mode::sample. No injection point exists outside upstream — per-seed byte
parity needs the upstream rng seam (thread the recorder's existing R generic
through prove()): **+0.5d, parked**; the eventual upstream PR wants it.

## The flag (default-OFF, shipped)

zk is a compile-time mode: demo `zk` cargo feature (default `[]` — OFF),
jolt-prover `webgpu` feature (OFF), zk×webgpu = both. Default builds are
clear-mode and byte-pinned. zk artifacts regenerate via
`cargo run --release --features native,zk --bin generate-preprocessing`
(verifier bin +1090B = BlindFold vc_setup; SRS bytes mode-independent).
