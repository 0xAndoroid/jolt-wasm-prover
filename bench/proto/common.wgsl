// Shared declarations for the one-hot commit accumulate prototype.
// Field p = 2^128 - C, C = 0xFFFFA7F7. Canonical fp128 = 4 x u32 little-endian limbs.
// Accumulators are 8 x u32 holding 16-bit digits (L[k] = sum of digit k of every term),
// exact for <= 65536 terms (65536 * 0xFFFF < 2^32).

struct Params {
  positions: u32,    // positions_per_block (2048 at 2^18)
  num_chunks: u32,   // positions / CHUNK (CHUNK is the main kernel's override constant)
  blocks: u32,       // blocks_per_column
  colcap: u32,       // column_capacity (hot row stride in bytes)
  part_mode: u32,    // 0: 16-bit digit partials; 1: (wrapping limb sum, carry count) partials (v7 only)
}

const C_LO: u32 = 0xFFFFA7F7u;          // 2^128 mod p
const P0: u32 = 0x00005809u;            // p limb 0; limbs 1..3 are 0xFFFFFFFF
const FULL: u32 = 0xFFFFFFFFu;
