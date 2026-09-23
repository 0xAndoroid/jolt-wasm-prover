export type ProgramName = 'sha2' | 'keccak'

export type LoadState = 'idle' | 'loading' | 'ready'

export type AppStatus = 'loading' | 'ready' | 'proving' | 'error'

export interface ProgramState {
  loadState: LoadState
  proofBytes: Uint8Array | null
  programIoBytes: Uint8Array | null
  // Akita verifier setups are exact in the proof shape: the prover emits the
  // verifier preprocessing for each proof and verification consumes it.
  verifierPreprocessingBytes: Uint8Array | null
  verifyResult: { valid: boolean; elapsed: number } | null
}

export interface ProgramFiles {
  program: string
  elf: string
}

// Messages sent to the worker
export type WorkerRequest =
  | { type: 'init'; data: { numThreads: number; cacheBust?: string } }
  | {
      type: 'load-program'
      data: {
        program: ProgramName
        programPreprocessing: ArrayBuffer
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
        verifierPreprocessing: Uint8Array
      }
    }
  | { type: 'get-trace' }
  | { type: 'clear-trace' }

// Messages received from the worker
export type WorkerResponse =
  | { type: 'init-done' }
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
  | { type: 'error'; error: string }
