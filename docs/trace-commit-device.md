# Trace-commit device seam

Patch `0006` adds a device hook to jolt-akita's stage-0 packed one-hot trace
commit (`TracePackedOneHotCommitOperation::commit_group` → `commit_packed`).
An installed `TraceCommitDevice` is offered every commit first; the CPU kernels
run otherwise (see Qualification). The WebGPU kernel plugs in here;
`CpuReferenceDevice` (`src/trace_commit_reference.rs`, feature
`trace-commit-device`) is its bit-exact oracle.

## Shipped shapes (sha2 / sha2-chain, `frontend/public` artifacts, native probe)

| padded log2 | guest       | K  | D   | n_a | P (positions/block) | digits_inner | segment_rings | blocks/column | num_blocks | C / capacity | routes to device |
|-------------|-------------|----|-----|-----|---------------------|--------------|---------------|---------------|------------|--------------|------------------|
| 12          | sha2        | 16 | 256 | 1   | 512                 | 1            | 256           | —             | 32 (flat)  | 57 / 64      | no (segment_rings < P) |
| 16          | sha2-chain  | 16 | 512 | 1   | 1024                | 1            | 2048          | 2             | 128        | 57 / 64      | yes |
| 18          | sha2-chain  | 16 | 512 | 1   | 2048                | 1            | 8192          | 4             | 256        | 57 / 64      | yes |
| 20          | sha2-chain  | 16 | 512 | 1   | 4096                | 1            | 32768         | 8             | 512        | 57 / 64      | yes |
| 21          | sha2-chain  | 16 | 512 | 1   | 8192                | 1            | 65536         | 8             | 512        | 57 / 64      | yes |

`segment_rings = T·K/D`, `blocks_per_column = segment_rings / P`,
`num_blocks = column_capacity · blocks_per_column`. Iteration counts: 17 → 2^16,
69 → 2^18, 278 → 2^20, 556 → 2^21 (`SHA2_CHAIN_ITERS`).

## Cost (M5 Max, `nice -n 10`, release, `RUST_LOG=jolt_akita=info`)

| padded log2 | extract (host → device buffers) | convert (limbs → rings) | CPU kernels commit | cpu-ref device commit |
|-------------|-----------------------------------|--------------------------|--------------------|-----------------------|
| 16          | 0.97 ms                           | 0.56 ms                  | —                  | 21.9 ms |
| 18          | 2.94 ms (3.0 % of CPU commit)     | 0.23 ms                  | 97.2 ms            | 79.4 ms |
| 20          | 7.60 ms                           | 0.46 ms                  | —                  | 301 ms |
| 21          | 14.6 ms                           | 0.34 ms                  | —                  | 601 ms |

## ABI (`jolt_akita::{TraceCommitShape, TraceCommitJob, TraceCommitDevice}`)

Field: p = 2^128 − C, C = 0xFFFF_A7F7. Every coefficient is a canonical
`u128 < p` as 4 little-endian `u32` limbs (`limb0` = bits 0..32).

Inputs (`TraceCommitJob`):

- `shape`: all counts above plus `num_rows = T`, `num_columns = C`,
  `column_capacity`, `ring_dimension = D`.
- `hot: &[u8]`, length `T·C`, index `t·C + c`; the selected one-hot row of
  trace row `t`, column `c`, `< K`. `0` means "nothing to add" unless bit `c`
  of `masks[t]` is set (a committed zero).
- `masks: &[u64]`, length `T`; committed-digit-zero mask per trace row.
- `a_plane: &[u32]`, digit-zero plane of the setup matrix, index
  `((a·P + q)·D + i)·4 + limb`; ring `A[a][q]` is setup row `a`, column
  `q·num_digits_inner`.

Output (`Some(Ok(limbs))`): length `num_blocks·n_a·D·4`, index
`((block·n_a + a)·D + i)·4 + limb`, `block = c·blocks_per_column + trace_block`;
columns `c ≥ num_columns` (padding) are all zero. `None` declines the shape
(CPU kernels run, logged at `info`); `Some(Err)` aborts the proof.

Math per committed `(t, c)`: `ring = t / (D/K)`, `r = t mod (D/K)`,
`trace_block = ring / P`, `q = ring mod P`, `s = r·K + hot[t·C + c]`;
`out[block][a] += X^s · A[a][q]` in the negacyclic ring
`F[X]/(X^D + 1)`: coefficient `i` gets `+A[i − s]` for `i ≥ s` and
`−A[i − s + D]` for `i < s`.

Reference accumulation (`CpuReferenceDevice`, GPU-shaped): per output ring,
separate positive and negative sums as `lo: u128` + 32-bit wrap counter, fold
with `2^128 ≡ C (mod p)`, then `pos − neg mod p`. Max additions per
coefficient = `P·D/K` (2^17 at 2^21), so a `u32` wrap counter is exact.

## Qualification and fallback

- Seam (`device.rs`): routes only `K ≤ D` and `segment_rings ≥ P` (block-aligned
  segments); otherwise the CPU kernels run and no device call happens.
- `CpuReferenceDevice` accepts `K = 16, D = 512, n_a = 1, num_digits_inner = 1`
  (every shipped sha2-chain size); everything else returns `None`.
- Result validation: length mismatch → `AkitaError::InvalidSize`; a limb value
  `≥ p` → `AkitaError::InvalidInput`.
- `set_trace_commit_device(Arc<dyn TraceCommitDevice>)` is a process-global
  `OnceLock`; the second install fails with `InvalidSetup`.

## Usage

```bash
./setup-wasm-deps.sh                                   # applies patches/0006
JOLT_TRACE_COMMIT_DEVICE=cpu-ref cargo run --release --features native,trace-commit-device --bin test-roundtrip
RUST_LOG=jolt_akita=info SHA2_CHAIN_ITERS=69 cargo run --release --features native,trace-commit-device --bin test-roundtrip
RUSTC_BOOTSTRAP=1 CARGO_UNSTABLE_BUILD_STD="panic_abort,std" wasm-pack build --release --target web -- --features trace-commit-device
```

Native (`engine::install_trace_commit_device_from_env`) reads
`JOLT_TRACE_COMMIT_DEVICE=cpu-ref`; wasm installs through
`engine::install_trace_commit_device` (nothing installed by default). Proof bytes
are identical with and without the device (sha256 printed by `test-roundtrip`).
