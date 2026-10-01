// Matches the wasm-bindgen exports in src/lib.rs; the generated pkg is not needed for static checks.
export interface ProveResult {
  proof: Uint8Array
  proof_size: number
  program_io: Uint8Array
  verifier_preprocessing: Uint8Array
  num_cycles: number
  padded_cycles: number
  trace_ms: number
  setup_ms: number
  prove_ms: number
  gpu_status: string
  gpu_selftest_ms: number
  gpu_selftest_mismatches: number
  gpu_roundtrip_us: number
  gpu_commit: string
  gpu_digit_range: string
  gpu_stage2: string
}
export class WasmProver {
  constructor(schedules: Uint8Array, preprocessing: Uint8Array, elf: Uint8Array)
  prove_sha2(input: Uint8Array): ProveResult
  prove_ecdsa(z: BigUint64Array, r: BigUint64Array, s: BigUint64Array, q: BigUint64Array): ProveResult
  prove_keccak_chain(input: Uint8Array, iterations: number): ProveResult
  prove_sha2_chain(input: Uint8Array, iterations: number): ProveResult
}
export class WasmVerifier {
  constructor(preprocessing: Uint8Array)
  verify(proof: Uint8Array, io: Uint8Array): boolean
}
export default function init(options: {module_or_path: string}): Promise<{memory: WebAssembly.Memory}>
export function initThreadPool(threads: number): Promise<void>
export function init_tracing(): void
export function get_trace_json(): string
export function clear_trace(): void
export function gpu_mailbox_ptr(): number
export function gpu_selftest(): string
export function gpu_is_dead(): boolean
export function set_gpu_enabled(): void
export function set_gpu_disabled(): void
export function set_gpu_commit_enabled(enabled: boolean): void
export function set_gpu_digit_range_enabled(enabled: boolean): void
export function set_gpu_op_timeout_ms(ms: number): void
export function set_gpu_unavailable(): void
export function set_digit_range_parity_rounds(rounds: number): void
export function set_gpu_stage2_enabled(enabled: boolean): void
export function set_relation_range_parity_rounds(rounds: number): void
export function gpu_trip_probe(n: number): number
