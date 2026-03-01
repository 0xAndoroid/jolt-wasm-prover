import type { ProgramName, ProgramFiles } from './types'

export const PROGRAMS: ProgramName[] = ['sha2', 'keccak']

export const PROGRAM_FILES: Record<ProgramName, ProgramFiles> = {
  sha2: {
    prover: 'sha2_prover.bin',
    verifier: 'sha2_verifier.bin',
    elf: 'sha2.elf',
  },
  keccak: {
    prover: 'keccak_prover.bin',
    verifier: 'keccak_verifier.bin',
    elf: 'keccak.elf',
  },
}

export const CACHE_BUST = 'v=2'

export const SHA2_MAX_BYTES = 2048
