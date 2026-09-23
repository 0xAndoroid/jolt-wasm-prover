export type ProgramName = 'sha2' | 'ecdsa' | 'keccak'

export type LoadState = 'idle' | 'loading' | 'ready'

export type AppStatus = 'loading' | 'ready' | 'proving' | 'error'

export type ProveMode = 'gpu' | 'cpu'

// worker.js `initGpu()` report: 'ok' | 'disabled' | 'unavailable' | 'error: …'
export interface GpuInfo {
  status: string
  reason?: string
  adapter?: { vendor?: string; architecture?: string; device?: string; description?: string }
  selftestMs?: number
  roundtripUs?: number
}

export interface ProofSummary {
  mode: ProveMode
  proveMs: number
  totalMs: number
  commitMs?: number
  digitRangeMs?: number
}

export interface ProgramState {
  loadState: LoadState
  proofBytes: Uint8Array | null
  programIoBytes: Uint8Array | null
  // Akita verifier setups are exact in the proof shape: the prover emits the
  // verifier preprocessing for each proof and verification consumes it.
  verifierPreprocessingBytes: Uint8Array | null
  verifyResult: { valid: boolean; elapsed: number } | null
  lastProof: ProofSummary | null
}

export interface ProgramFiles {
  program: string
  elf: string
}

// Messages sent to the worker
export type WorkerRequest =
  | { type: 'init'; data: { numThreads: number; cacheBust?: string; gpu?: boolean } }
  | { type: 'set-gpu'; data: { enabled: boolean } }
  | {
      type: 'load-program'
      data: {
        program: ProgramName
        programPreprocessing: ArrayBuffer
        elfBytes: ArrayBuffer
      }
    }
  | { type: 'prove'; data: { program: 'sha2'; input: number[] } }
  // secp256k1 scalars / coordinates as little-endian u64 limbs ('0x…' strings;
  // worker.js maps them through BigInt).
  | {
      type: 'prove'
      data: { program: 'ecdsa'; z: string[]; r: string[]; s: string[]; q: string[] }
    }
  | {
      type: 'prove'
      data: { program: 'keccak'; input: number[]; numIters: number }
    }
  | {
      type: 'verify'
      data: {
        program: ProgramName
        proof: Uint8Array
        programIo: Uint8Array
        verifierPreprocessing: Uint8Array
      }
    }
  | { type: 'get-trace' }
  | { type: 'clear-trace' }

// Messages received from the worker
export type WorkerResponse =
  | { type: 'init-done'; gpu: GpuInfo }
  | { type: 'gpu-status'; gpu: GpuInfo }
  | { type: 'program-loaded'; program: ProgramName }
  | {
      type: 'prove-done'
      program: ProgramName
      proof: Uint8Array
      proofSize: number
      programIo: Uint8Array
      verifierPreprocessing: Uint8Array
      numCycles: number | null
      paddedCycles: number | null
      traceMs: number
      setupMs: number
      proveMs: number
      gpuStatus: string
      gpuCommit: string
      gpuDigitRange: string
      peakMemory: number | null
      elapsed: number
    }
  | {
      type: 'verify-done'
      program: ProgramName
      valid: boolean
      elapsed: number
    }
  | { type: 'trace'; trace: string }
  | { type: 'trace-cleared' }
  | { type: 'error'; error: string; gpu?: GpuInfo }
