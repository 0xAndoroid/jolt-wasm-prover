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
2. implement 1/1 — done (Sep 23 17:10). Commits 40762cc (UI tab), 6fe2334
   (test_ui_modes ECDSA), eb54547 (docs), 79ffa34 (roundtrip). Native
   roundtrip: ecdsa 203,465 cycles, padded 2^18, prove 0.76 s, verify 0.012 s.
   WebKit gate (macbook-home, wasm 22.4 MiB): ECDSA GPU 2.10 s / CPU 9.19 s,
   proof SHA-256 501fd2b7… identical, verify VALID both; SHA-256 + Keccak +
   Chromium + dead-proxy sections unchanged and green; no overflow 375/1440.
3. review 1/3 — opus-max (70117802): 5 findings, all self-fixed (d88be85 f4d6afe a598177).
   review 2/3 — opus-max (06668770): 1 copy finding fixed (34b18dc).
   review 3/3 — opus-max (b22a028a): 2 nits (copy wording, comment) fixed
   (571f467 1fc1a88); CI green. Merged despite non-zero round-3 count: both
   findings were self-fixed text nits, substance (vector, limb order, postcard
   bytes, TS types, UI test, layout) cleared three times independently.
   Follow-ups (pre-existing): native vs wasm proof bytes differ (both verify);
   npm run lint 6 errors in untouched files; engine::verify accepts io.panic.
4. merge 1/1 — 
5. deploy 1/1 — 

Decision ledger: `.audit/ecdsa-guest.tsv`.
