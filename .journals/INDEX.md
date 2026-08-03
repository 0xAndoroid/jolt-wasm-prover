# .journals — archived WebGPU campaign files from the main checkout root

Archived 2026-08-02 (Mac mini repo-hygiene pass). Working journals, specs, and evidence
artifacts left as untracked dotfiles in the repo root by the WebGPU prover campaign
(Jul 30 – Aug 1, 2026). Preserved verbatim, except `w5-u4/on-trace.log` (126 MB raw
trace) which was gzipped to fit GitHub's per-file limit.

## Campaign journal & research
`.webgpu-campaign-journal.md` — main running journal of the WebGPU prover campaign
(waves W1–W6, ~330 KB). `.webgpu-research-r1.md` — round-1 research memo.
`.webgpu-phase2-architecture.md` — phase-2 (BlindFold ZK) architecture doc.

## Wave specs
`.webgpu-w3-specs.md` / `.webgpu-w5-specs.md` / `.webgpu-w6-specs.md` — per-wave kernel
and lane specs handed to implementation agents.

## Evidence (`.webgpu-lane-evidence/`)
Per-experiment measurement artifacts (JSON probes, stage logs, soak runs, screenshots):
`u3/`, `u3b/` — U3 lane probes and determinism/soak pairs at 2^16–2^23; `v2/` — V2
close-out soak/timed logs and native byte-parity check; `w3-t2a/t2b/t4-verify/` — wave-3
ticket verification runs; `w4-ui-shots/` — wave-4 frontend screenshots; `w5-u4/` — wave-5
U4 gate runs, trip JSONLs, and the gzipped 126 MB `on-trace.log`; `w6m/` — wave-6
measurement set.
