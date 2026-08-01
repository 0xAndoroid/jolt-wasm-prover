# Draft bug report — Chromium issue tracker (Blink>WebGPU)

> Status: DRAFT for the user to file (crbug, plus an Apple Feedback Assistant variant with the
> same body — the failure signature points at the Metal compiler/driver layer under Dawn, but
> Chrome is the only harness we can hand anyone).

---

**Title:** Silent wrong results from a WebGPU compute dispatch reading a large prior dispatch's
output on Apple M5 Max — SIMD-lane-patterned corruption at ≥262K producer threads, persists
across pass/submit/drain boundaries; same page clean on M4; zero validation errors

**Component:** Blink>WebGPU (suspected WGSL→MSL compilation or Metal execution; Apple-silicon-
die-specific)

## Environment

| | affected | negative control |
|---|---|---|
| Hardware | MacBook Pro, **Apple M5 Max**, 40-core GPU, 128 GB | Mac mini, **Apple M4**, 10-core GPU, 16 GB |
| macOS | **26.5.2 (25F84)** | **26.5.2 (25F84)** — same build |
| Browser | Chromium 145.0.7632.6 (Chrome for Testing) | Chrome 150.0.7871.187 AND Chromium 145.0.7632.6 — both pass |
| Adapter | `vendor=apple architecture=metal-3` | `vendor=apple architecture=metal-3` |
| Chrome 150 on affected hardware | **untested** (not installed on that machine) | — |

Same OS build, same browser binary, same page — the die is the only variable in the matrix.

## Summary

A pure per-thread WGSL compute kernel (elliptic-curve bucket accumulation over BN254 G1,
16-bit-limb Montgomery arithmetic — **no atomics, no barriers, no workgroup memory, no
uniformity-sensitive control flow**; each thread reads its input slice and writes its own
disjoint output) returns wrong limb values on Apple M5 Max at large dispatch sizes:

- **655,360 threads (10,240 workgroups @64): corrupt on ~every dispatch** of the Jacobian
  variant (22/22 repeats across four fresh browser sessions; 48–176 wrong output points out of
  40,960 per dispatch). An XYZZ-coordinate variant of the same shape corrupts intermittently
  (1/18 repeats standalone; the production application sees Jacobian ~every proof, XYZZ ~half).
  2,621,440 threads corrupts harder (176–336 wrong points per dispatch).
- **Onset between 196,608 (PASS) and 262,144 threads (CORRUPT)** — bisection in
  `results/m5-s3-1-bisect.json`. 163,840 threads and below: bit-exact PASS across 26 runs
  including two 10-repeat batteries, same die, same browser, same page, same checks code path.
- **A single dispatch in a fresh browser process corrupts** (96 wrong points on the first and
  only dispatch) — no warm-up or repetition required.
- **The accumulate kernel alone is clean at 655,360 threads**: dispatching only `bucket_madd_csr`
  and reading its partials directly (copy + mapAsync) shows zero invalid values across 4
  repeats. Corruption appears only when the second kernel (`bucket_reduce`, 40,960 threads)
  runs after it, reading the buffer madd wrote.
- **Boundaries between the two dispatches do NOT suppress it** (the localization ladder,
  each rung ×6 repeats at 655K and 2.62M producer threads):
  1. one compute pass, both dispatches (production shape) — CORRUPT;
  2. two passes, one encoder+submit — CORRUPT;
  3. two separate encoder+submits — CORRUPT;
  4. two submits with **`await queue.onSubmittedWorkDone()` between them** (producer fully
     retired before the consumer is even submitted, queue empty) — **CORRUPT**;
  5. producer alone, output read back via `copyBufferToBuffer` + `mapAsync` — **CLEAN**.
  So the consumer compute dispatch produces wrong results over data that is fully settled and
  that the copy engine reads correctly. Comparable corruption magnitudes on every corrupt rung
  (`results/m5-s4-*.json`, `m5-s5-1-drain.json`). Not a missing implicit barrier; consistent
  with a compute-path read/execution fault dependent on device state left by the preceding
  large dispatch.
- **Apple M4, same macOS build: PASS at every size** (through 2,621,440 threads), under both
  Chromium 145 and Chrome 150, with byte-identical output digests across the two browsers.
- **Zero validation errors, zero `popErrorScope` hits, zero device loss** — the failure is
  silent wrong arithmetic.

Every output point is a sum of known curve points, so correctness is checked three ways on the
CPU (BigInt): the wrong outputs are **off the curve** `y² = x³ + 3z⁶` (impossible for any true
sum — garbage limbs, not a wrong-algorithm bug), and corrupt dispatches are additionally
**nondeterministic**: repeating the identical dispatch on identical input bytes produces
different wrong words (`nondetWords` 112–256 across repeats), while on M4 all repeats are
byte-identical.

### Corruption geometry (the diagnostic meat)

Corrupt output indices cluster inside one or two 64-thread workgroups per dispatch, in
SIMD-lane-structured patterns:

- runs of **16 consecutive outputs starting at a workgroup boundary** (= half of a 32-lane
  simdgroup), e.g. outputs 4288–4303 (workgroup 67), 11648–11663 (wg 182), 18304–18319 (wg 286);
- or a **4-on/4-off stride-8 pattern** inside one workgroup (= alternating quads), e.g.
  12772–12775, 12780–12783, 12788–12791, 12796–12799.

Corrupted values frequently contain **zeroed 16-bit limbs at packed-word boundaries**
(`…be900000…`, `…da9f0000…` in otherwise plausible 254-bit values) — lanes appear to compute or
store zeros mid-chain rather than suffer random bit flips. The signature looks like quad/
half-simdgroup lanes of specific threadgroups executing or retiring wrongly at high occupancy.

## Steps to reproduce

1. Download `index.html` (fully self-contained: inline WGSL, JS BigInt self-checking; no
   network, no build step, ~250 KB).
2. Open in Chrome/Chromium on an M5-class machine and click **Run full sweep**, or drive it
   headless with a raw binary (no automation deps):
   `node run-standalone.mjs "<path-to-chrome-binary>" 'brows=2560&repeats=6'`
3. The matrix table prints PASS/CORRUPT per cell; the JSON pane carries per-point dumps
   (`detail`: output index, workgroup, raw limbs hex).

The failing dispatch in one line: `bucket_madd_csr` (Jacobian accumulate, workgroup_size 64,
655,360 threads = 10,240 workgroups) then `bucket_reduce` (40,960 threads) in one compute pass,
one submit, then one readback — repeated 6× on identical input buffers.

## Expected

All cells PASS: every output on-curve, equal to the CPU BigInt reference on sampled buckets,
bit-identical across repeats — the behavior on M4 (same OS build) and on M5 at ≤163,840
threads.

## Actual

On M5 Max × Chromium 145: `verdict: CORRUPT` at the 655,360-thread cells; per-repeat off-curve
counts 48–176 (of 40,960 outputs), nondeterministic across repeats; `notes: []` (no errors
anywhere). Raw captures attached (`results/m5-*.json`).

## Additional context

- The kernels are byte-verbatim from the Jolt zkVM's WebGPU prover backend, where this
  surfaced as invalid cryptographic proofs on M5 Max hardware (first-parsed G1 point failed
  curve-membership on deserialize). The application now ships an on-curve integrity guard with
  CPU fallback for exactly this; the page is the extracted standalone harness.
- The same kernels pass a 26-vector known-answer suite **on the affected die under the affected
  browser** — at KAT dispatch sizes (~10² threads). Only occupancy separates pass from fail,
  which is why conformance-style testing misses it.
- Hand-written Metal-native (MSL) kernels performing the same bucket-accumulation math run
  clean at scale on the same M5 die in a sibling backend — the die computes this math correctly
  via a different compilation path. Not conclusive for silicon-vs-compiler (different register
  allocation/occupancy), but it narrows suspicion toward the WGSL→MSL path (Tint codegen or the
  Metal compiler's consumption of it) under high occupancy on this die.
- Ecosystem note, different WGSL frontend: naga (Deno/wgpu) on Metal also mishandles kernels of
  this family (loop-bounding hangs on large kernels; one Fq6-arithmetic miscompile we
  quarantine). These ~2,000-line unrolled carry-chain kernels appear to be effective stress
  tests for Metal shader compilation generally; the present report is strictly the Tint/Dawn
  path.

## Attachments

- `index.html` (the repro), `README.md` (how to read failures), `run-standalone.mjs`
- `results/m4-*.json` (negative controls: M4 sweeps incl. 655K/2.6M-thread cells, both browsers)
- `results/m5-*.json` (affected captures: session-2 655K-thread corruption ×2 sessions,
  session-3 threshold bisect / single-kernel / single-shot / geometry runs)

## Open cells / caveats

- M5 Max × Chrome 150: untested (no Chrome installed on the affected machine) — unknown whether
  the current stable channel is affected. Chromium 145 is the Chrome-for-Testing build matching
  Playwright's pin, full (non-headless-shell) binary.
- We cannot distinguish from web content whether the consumer dispatch *reads* wrong data or
  *computes* wrongly — the observable is the ladder above. A Metal-level reproduction attempt
  (same two pipelines via MTLComputeCommandEncoder) is the natural next step for a platform
  engineer.
