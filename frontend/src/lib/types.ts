export type ProgramName = 'sha2' | 'keccak'

export type LoadState = 'idle' | 'loading' | 'ready'

export type AppStatus = 'loading' | 'ready' | 'proving' | 'error'

// 'off' = disabled via ?webgpu=0; 'unavailable' = requested but the device
// handshake failed (prover fell back to CPU); 'pending' = init in flight.
export type GpuStatus = 'pending' | 'on' | 'unavailable' | 'off'

// Gate overrides for the WebGPU arm; {} takes the production defaults.
export interface WebGpuConfig {
  minTerms?: number
  handoffLen?: number
  millerCpuFraction?: number
  commitPipeline?: boolean
  bucketXyzz?: boolean
  minTermsCommit?: number
}

export interface ProgramState {
  loadState: LoadState
  proofBytes: Uint8Array | null
  programIoBytes: Uint8Array | null
  verifyResult: { valid: boolean; elapsed: number } | null
}

export interface ProgramFiles {
  prover: string
  verifier: string
  elf: string
}

// Messages sent to the worker
export type WorkerRequest =
  | { type: 'init'; data: { numThreads: number; webgpu: WebGpuConfig | null } }
  | {
      type: 'load-program'
      data: {
        program: ProgramName
        proverPreprocessing: ArrayBuffer
        verifierPreprocessing: ArrayBuffer
        elfBytes: ArrayBuffer
      }
    }
  | { type: 'prove'; data: { program: 'sha2'; input: number[] } }
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
      }
    }
  | { type: 'get-trace' }
  | { type: 'clear-trace' }

// Messages received from the worker
export type WorkerResponse =
  | { type: 'init-done'; webgpu: { readyMs: number; warmupMs: number } | null }
  | { type: 'program-loaded'; program: ProgramName }
  | {
      type: 'prove-done'
      program: ProgramName
      proof: Uint8Array
      proofSize: number
      programIo: Uint8Array
      numCycles: number | null
      paddedCycles: number | null
      peakMemory: number | null
      elapsed: number
      millerServed: number
    }
  | {
      type: 'verify-done'
      program: ProgramName
      valid: boolean
      elapsed: number
    }
  | { type: 'trace'; trace: string }
  | { type: 'trace-cleared' }
  | { type: 'error'; error: string }
