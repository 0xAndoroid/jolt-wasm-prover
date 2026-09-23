import type { ProgramName, ProgramFiles } from './types'

export const PROGRAMS: ProgramName[] = ['sha2', 'keccak']

export const PROGRAM_FILES: Record<ProgramName, ProgramFiles> = {
  sha2: {
    program: 'sha2_program.bin',
    elf: 'sha2.elf',
  },
  keccak: {
    program: 'keccak_program.bin',
    elf: 'keccak.elf',
  },
}

// Bump when artifacts in public/ change so cached copies are refetched.
export const CACHE_BUST = 'v=3'

export const SHA2_MAX_BYTES = 2048
