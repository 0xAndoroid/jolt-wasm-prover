# ECDSA (1 sig, inlines) prover tab — orchestrator journal

Feature: add back the ECDSA tab in jolt.rs (secp256k1 ECDSA verify of one fixed
test vector via `jolt-inlines-secp256k1`), on today's akita + WebGPU main.
Branch `feat/ecdsa-inline-guest`, card 568, orchestrator task d03fb23b.

## Findings (Sep 23, 16:50)
- Guest `guests/secp256k1` (`secp256k1_ecdsa_verify`, heap 100000, inline
  `ecdsa_verify`), `preprocessing/generate.rs` GUESTS row `ecdsa` (max trace
  2^18), `frontend/public/ecdsa.elf` + `ecdsa_program.bin` (regenerated Sep 22),
  `WasmProver::prove_ecdsa`, worker.js `ecdsa` case and the secp256k1 inline
  linkage in lib.rs/test_roundtrip.rs all already exist on main.
- Missing: React UI entry (types/constants/use-prover/app + inputs component),
  test-roundtrip coverage of `ecdsa`, WebKit UI test coverage, docs lists.
- Test vector (from `git show 71ae90f^:www/index.js`): message "hello world",
  z/r/s 4 LE u64 limbs, q 8 LE u64 limbs (see ECDSA_TEST_VECTOR).

## Steps (playbook verbatim)
1. plan 1/1 — skip: all Rust/worker plumbing exists; remaining work is one UI
   entry + roundtrip coverage + docs, plan is the findings list above.
2. implement 1/1 — 
3. review i/3 — 
4. merge 1/1 — 
5. deploy 1/1 — 

Decision ledger: `.audit/ecdsa-guest.tsv`.
