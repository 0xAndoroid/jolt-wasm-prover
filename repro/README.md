# WebGPU G1 bucket-kernel corruption repro

A single-file, dependency-free reproduction of silent compute corruption observed on
**Apple M5 Max × Chrome/Chromium 145** (Metal backend): two production WGSL kernels from the
[Jolt zkVM](https://github.com/a16z/jolt) WebGPU prover return **wrong elliptic-curve limbs at
high dispatch occupancy** (~40K+ threads) while passing byte-exactly at small dispatch sizes on
the same die, and passing at every size on Apple M4. No validation errors, no device loss —
silent wrong arithmetic.

## Files

- `index.html` — the whole repro: inline WGSL (byte-verbatim production kernels), JS harness,
  BigInt reference checking. Open it, click **Run full sweep**. No build step, no network.
- `run-standalone.mjs` — zero-dependency headless driver: serves the page, spawns a browser
  **binary** directly (`--headless=new`), receives results via the page's `?post=1` beacon.
  `node run-standalone.mjs "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"`.
  Exit 3 flags a vacuous run (no hardware Metal adapter).
- `run-repro.mjs` — same, via Playwright if you have it: `PW_CHANNEL=chromium node run-repro.mjs`.
  Playwright's default headless *shell* has **no WebGPU**; always pick a real browser via
  `PW_CHANNEL` or `PW_EXECUTABLE`.
- `results/` — captured matrix results (JSON) from the machines listed below.
- `BUG-REPORT.md` — draft report (crbug shape).

## What the page does

The corrupting production dispatch is a CSR bucket accumulation over BN254 G1
(16-bit-limb Montgomery arithmetic, ~2200 lines WGSL per kernel, generated):

1. `bucket_madd_csr` — each thread walks a contiguous slice of one bucket's column indices and
   accumulates `acc += bases[col]` (Jacobian mixed addition). Pure per-thread: **no atomics, no
   barriers, no workgroup memory**.
2. `bucket_reduce` — each thread folds 16 segment partials into one Jacobian point.

Both run back-to-back in one compute pass, one submit, then a readback — exactly the production
encoding. An XYZZ-coordinate variant pair of the same shape is also included (in production it
corrupts on ~half the runs vs ~every run for Jacobian).

The page builds a deterministic synthetic workload (seeded PRNG; base table = `(i+1)·G`), runs
each cell at a sweep of thread counts, and checks every output on the CPU with BigInt reference
arithmetic:

| check | catches |
|---|---|
| on-curve (`y² = x³ + 3z⁶`, Montgomery-stripped) | garbage limbs — what production hit |
| exact value vs CPU reference (16 sampled buckets/cell) | wrong-but-on-curve results |
| canonical range (every limb value < p) | non-reduced residues |
| cross-repeat byte determinism (identical inputs) | intermittent corruption |
| infinity-set match | dropped dispatches / never-written outputs |

A corrupt `madd+reduce` cell triggers one extra instrumented dispatch that reads the madd
partials directly, localizing the failing stage.

The sweep (default): 256 / 4,096 / 40,960 / 163,840 / 262,144 / 655,360 madd threads ×
{Jacobian, XYZZ} (an XXL checkbox adds 2,621,440). 40,960 = the production dispatch shape at
2^18 trace cycles for the narrow polynomial groups; 655,360 and up = the wide (80-column)
groups. Repeats: 3 at small cells, 6 at ≥40,960.

## How to read failures

Every cell row ends in **PASS** or **CORRUPT**. A corrupt cell's JSON (`Results JSON` pane, or
the runner's stdout) carries per-repeat counts plus up to 16 `detail` entries per check:
`{kind, out, brow, bucket, x, y, z}` with raw Montgomery limbs in hex — enough to see *which*
threads produced garbage. `notes` lists any validation errors / device loss (expected: none —
that's the point).

Sanity anchor: the same page must PASS all cells on a known-good config first (it does, on
Apple M4 under both Chrome 150 and Chromium 145, with byte-identical digests across the two
browsers — see `results/`). The BigInt reference is independent of the GPU: a CORRUPT verdict
is a real arithmetic divergence, not a harness artifact.

## Query parameters (for bisection)

`?autorun=1` run on load · `brows=160,640` cell list (madd threads = brows×256) ·
`repeats=10` · `kernels=jacobian,xyzz` · `stage=madd` (skip reduce, check partials directly) ·
`split=none|pass|submit|drain` boundary between the two dispatches (drain awaits
`onSubmittedWorkDone` between submits) · `dist=skew` power-law bucket sizes · `chain=N`
back-to-back write+submit iterations per repeat, each to its own out-slice · `churn=1`
per-repeat buffer create/destroy · `cpuload=N` memory-streaming workers ·
`details=N` per-repeat corruption-dump cap · `rowWidth=8192&k=16&segs=16` geometry ·
`nohot=64` idle-position rate · `seed=1` · `xl=1` add a 655,360-thread cell.
Both runners forward their query argv: `node run-standalone.mjs "<binary>" 'brows=2560&repeats=6'`.

## Provenance

WGSL is byte-verbatim from the Jolt WebGPU backend's kernel export
(`jolt-kernels` `webgpu::wgsl`, tree `webgpu/wave1` @ `7912a9194`), embedded in `index.html`
as inert `<script type="x-wgsl">` blocks:

| module | sha256 |
|---|---|
| bucket_madd_csr.wgsl | `d63bc5f3b0fe8cf0483559ad26ec1e4c83236fa3a650848fc7ba8a9445415f74` |
| bucket_reduce.wgsl | `a1866927827ea39a42f6017fc0aaf8d35f27fc864698205f939dbc1fed15329b` |
| bucket_madd_csr_xyzz.wgsl | `18edbf69bdb8f9bc9f3bbd219d0018f9c038c2b624e331224058314f70094d73` |
| bucket_reduce_xyzz.wgsl | `c17604744f0fa04ebd412f9b0322e583a5efc46fed43557f4ac04e9049b568a4` |

## Result matrix

See `results/*.json` for raw captures. Machines: M5 Max = MacBook Pro (40-core GPU, 128 GB,
macOS 26.5.2/25F84); M4 = Mac mini (10-core GPU, 16 GB). Browser on the M5: Chromium
145.0.7632.6 (Chrome for Testing, Playwright cache build) — the adapter is
`vendor=apple architecture=metal-3` on both machines.

| machine | browser | ≤196,608 threads | 262,144+ threads | 655,360 threads | 2,621,440 threads |
|---|---|---|---|---|---|
| M4 | Chrome 150.0.7871.187 | PASS | PASS | PASS | PASS |
| M4 | Chromium 145.0.7632.6 | PASS | PASS | PASS | PASS |
| M5 Max | Chromium 145.0.7632.6 | PASS (incl. 2× 10-repeat batteries @163,840) | **CORRUPT** (onset; 560 off-curve pts over 4 repeats) | **CORRUPT** — Jacobian 22/22 repeats across 4 sessions (48–176 off-curve each), XYZZ 1/18 | **CORRUPT** — Jacobian (176–336 off-curve per repeat) |
| M5 Max | Chrome 150 | **untested — not installed on that machine** | | | |

**Onset: between 196,608 (PASS ×4) and 262,144 threads (CORRUPT).** A single dispatch in a
fresh browser process corrupts (`results/m5-s3-3-singleshot.json`) — no warm-up, no repetition
needed. **The accumulate kernel alone is clean at 655,360 threads** (`stage=madd` run, partials
checked directly: zero invalid) — the minimal failing form is the two-dispatch sequence: big
madd, then reduce reading its output. The boundary ladder (`results/m5-s4-*.json`,
`m5-s5-1-drain.json`): corruption persists whether the two dispatches share a pass, sit in
separate passes, separate submits, or separate submits with a full
`queue.onSubmittedWorkDone()` drain between them — while the same producer output read back
via the copy engine is clean. The consumer compute dispatch misreads/miscomputes over fully
settled memory.

M4 output digests are byte-identical across the two browsers (deterministic reference bytes).
On the M5, corrupt repeats also differ from each other (`nondetWords` > 0) — the corruption is
nondeterministic across identical dispatches. `notes` is empty in every corrupt capture: zero
validation errors, zero device loss.

**Corruption geometry** (from `detail` dumps): corrupt outputs cluster inside one or two
64-thread workgroups per dispatch, in SIMD-lane-structured patterns — either 16 consecutive
outputs starting at a workgroup boundary (half a 32-lane simdgroup) or a 4-on/4-off stride-8
pattern (alternating quads). Corrupted values frequently contain zeroed 16-bit limbs (e.g.
`…be900000…`, `…da9f0000…` at packed-limb boundaries) — lanes appear to compute or store zeros,
not random bit flips.
