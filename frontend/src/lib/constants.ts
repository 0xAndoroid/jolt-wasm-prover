import type { ProgramName, ProgramFiles } from './types'

export const PROGRAMS: ProgramName[] = ['sha2', 'ecdsa', 'keccak']

export const PROGRAM_FILES: Record<ProgramName, ProgramFiles> = {
  sha2: {
    program: 'sha2_program.bin',
    elf: 'sha2.elf',
  },
  ecdsa: {
    program: 'ecdsa_program.bin',
    elf: 'ecdsa.elf',
  },
  keccak: {
    program: 'keccak_program.bin',
    elf: 'keccak.elf',
  },
}

// Bump when artifacts in public/ change so cached copies are refetched.
export const CACHE_BUST = 'v=3'

export const SHA2_MAX_BYTES = 2048

// One valid secp256k1 signature over z = SHA-256(message). Limbs are
// little-endian u64 (limb 0 least significant); q = (x limbs 0..4, y limbs 4..8).
export const ECDSA_TEST_VECTOR = {
  message: 'hello world',
  z: ['0x9088f7ace2efcde9', '0xc484efe37a5380ee', '0xa52e52d7da7dabfa', '0xb94d27b9934d3e08'],
  r: ['0xb8fc413b4b967ed8', '0x248d4b0b2829ab00', '0x587f69296af3cd88', '0x3a5d6a386e6cf7c0'],
  s: ['0x66a82f274e3dcafc', '0x299a02486be40321', '0x6212d714118f617e', '0x9d452f63cf91018d'],
  q: [
    '0x0012563f32ed0216',
    '0xee00716af6a73670',
    '0x91fc70e34e00e6c8',
    '0xeeb6be8b9e68868b',
    '0x4780de3d5fda972d',
    '0xcb1b42d72491e47f',
    '0xdc7f31262e4ba2b7',
    '0xdc7b004d3bb2800d',
  ],
}

// 'gpu' | 'cpu'; written only when the user clicks the mode selector.
export const MODE_STORAGE_KEY = 'jolt-prover-mode'
