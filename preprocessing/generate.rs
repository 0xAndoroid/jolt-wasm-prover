//! Compiles the guests and writes the browser-served artifacts into
//! `frontend/public/`:
//!
//! - `akita_schedules.bin` — the three base Akita schedule catalogs
//!   (`jolt-akita/schedules/*.aks`), trimmed to the rows a 32-bit target can
//!   deserialize, as one bincode `AkitaScheduleArtifacts` shared by every
//!   program;
//! - `{name}_program.bin` — `JoltProgramPreprocessing` (bincode/serde), the
//!   protocol-independent program shape the browser turns into a shape-exact
//!   Akita setup at prove time;
//! - `{name}.elf` — the guest RISC-V ELF.
//!
//! There is no static prover artifact: the Akita setup is exact in the trace
//! length and not serializable, so it is derived per proof (see
//! `src/engine.rs`).
//!
//! Guests are built through jolt-host (the `jolt` CLI must be installed from
//! the pinned jolt rev); jolt-sdk's host machinery is Dory-only and does not
//! compile against a jolt-prover built with `akita`.

use std::path::{Path, PathBuf};

use common::jolt_device::{MemoryConfig, MemoryLayout};
use jolt_akita::{AkitaScheduleArtifacts, AKITA_ONE_HOT_K16, AKITA_ONE_HOT_K256};
use jolt_host::Program;
use jolt_program::preprocess::JoltProgramPreprocessing;
use serde_json::value::RawValue;
use serde_json::Value;

struct Guest {
    /// Artifact prefix (`{name}_program.bin`, `{name}.elf`).
    name: &'static str,
    /// Cargo package under `guests/`.
    package: &'static str,
    /// The `#[jolt::provable]` function.
    func: &'static str,
    /// Must equal the guest's `#[jolt::provable(heap_size = ..)]`: the guest
    /// derives its IO region addresses from the same memory layout.
    heap_size: u64,
    /// Upper bound on the padded trace length the preprocessing admits.
    max_trace_length: usize,
}

const GUESTS: &[Guest] = &[
    Guest {
        name: "sha2",
        package: "sha2-guest",
        func: "sha2",
        heap_size: 32768,
        max_trace_length: 1 << 17,
    },
    Guest {
        name: "ecdsa",
        package: "secp256k1-ecdsa-verify-guest",
        func: "secp256k1_ecdsa_verify",
        heap_size: 100_000,
        max_trace_length: 1 << 18,
    },
    Guest {
        name: "keccak",
        package: "sha3-chain-guest",
        func: "sha3_chain",
        heap_size: 32768,
        max_trace_length: 1 << 22,
    },
    Guest {
        name: "sha2_chain",
        package: "sha2-chain-guest",
        func: "sha2_chain",
        heap_size: 32768,
        max_trace_length: 1 << 23,
    },
];

fn emit(guest: &Guest, public_dir: &Path, target_root: &Path) {
    let name = guest.name;
    println!(
        "[{name}] Compiling guest {}::{} ...",
        guest.package, guest.func
    );
    let memory_config = MemoryConfig {
        heap_size: guest.heap_size,
        ..MemoryConfig::default()
    };
    let mut program = Program::new(guest.package);
    program.set_func(guest.func);
    program.set_std(false);
    program.set_memory_config(memory_config);
    let target_dir = target_root.join(name);
    program.build(target_dir.to_str().expect("utf-8 target dir"));

    let elf_contents = program.get_elf_contents().expect("ELF contents");
    write_file(
        public_dir,
        &format!("{name}.elf"),
        name,
        "ELF",
        &elf_contents,
    );

    println!("[{name}] Decoding program ...");
    let (bytecode, memory_init, program_size, entry_address) = program.decode();
    let memory_config = MemoryConfig {
        program_size: Some(program_size),
        ..memory_config
    };
    let preprocessing = JoltProgramPreprocessing::new(
        bytecode,
        memory_init,
        MemoryLayout::new(&memory_config),
        entry_address,
        guest.max_trace_length,
        program.instruction_profile(),
    )
    .expect("program preprocessing");

    let bytes = bincode::serde::encode_to_vec(&preprocessing, bincode::config::standard())
        .expect("program preprocessing encode");
    write_file(
        public_dir,
        &format!("{name}_program.bin"),
        name,
        "Program preprocessing",
        &bytes,
    );
}

/// Catalog files under `jolt_akita::AkitaScheduleArtifacts::packaged_directory()`,
/// named after the `jolt_akita::configs` schedule families. Family binding is
/// re-checked by `validate_schedule_artifacts`.
const SCHEDULE_FILES: [&str; 3] = [
    "jolt-fp128-dense-bounded.aks",
    "jolt-fp128-onehot-k16.aks",
    "jolt-fp128-onehot-k256.aks",
];

/// Schedule fields typed `usize` in `akita-types`. Rows for committed groups
/// of 2^32 or more coefficients carry values above `u32::MAX` here, which the
/// wasm32 deserializer rejects (JSON integer out of range for `usize`).
const USIZE_SCHEDULE_FIELDS: [&str; 2] = ["input_witness_len", "live_ring_elements_per_claim"];

/// Separator the canonical artifact formatter writes between rows: the rows
/// array is broken one row per line at indent depth two.
const ROW_SEPARATOR: &str = ",\n    ";

fn emit_schedule_artifacts(public_dir: &Path) {
    let directory = AkitaScheduleArtifacts::packaged_directory();
    println!("[akita] Loading schedule catalogs from {directory:?} ...");
    let [dense, one_hot_k16, one_hot_k256] = SCHEDULE_FILES.map(|file| {
        let bytes = std::fs::read(directory.join(file)).expect("read schedule catalog");
        trim_catalog_for_wasm32(file, &bytes)
    });
    let artifacts = AkitaScheduleArtifacts::new(dense, one_hot_k16, one_hot_k256);
    validate_schedule_artifacts(&artifacts);
    let bytes = bincode::serde::encode_to_vec(&artifacts, bincode::config::standard())
        .expect("schedule artifacts encode");
    write_file(
        public_dir,
        "akita_schedules.bin",
        "akita",
        "Schedule catalogs",
        &bytes,
    );
}

/// Drops the catalog rows a 32-bit target cannot deserialize. Those rows
/// describe committed groups of 2^32+ coefficients, which no browser prover
/// can hold, so nothing selectable is lost; row lookup is exact-key, so the
/// remaining rows resolve unchanged. Rows are independent tokens of the
/// canonical artifact form, so splicing them out keeps the artifact canonical.
/// The catalog digest bound into the transcript does change: proofs made with
/// these artifacts verify against the verifier preprocessing the prover emits,
/// not against jolt's full packaged catalogs.
fn trim_catalog_for_wasm32(file: &str, bytes: &[u8]) -> Vec<u8> {
    #[derive(serde::Deserialize)]
    struct Envelope<'a> {
        #[serde(borrow)]
        rows: Vec<&'a RawValue>,
    }
    let text = std::str::from_utf8(bytes).expect("schedule catalog is UTF-8");
    let envelope: Envelope = serde_json::from_str(text).expect("schedule catalog envelope");
    let (first, last) = match (envelope.rows.first(), envelope.rows.last()) {
        (Some(first), Some(last)) => (first.get(), last.get()),
        _ => panic!("{file}: schedule catalog has no rows"),
    };
    let start = text.find(first).expect("first row is in the catalog text");
    let end = text.rfind(last).expect("last row is in the catalog text") + last.len();
    let kept: Vec<&str> = envelope
        .rows
        .iter()
        .map(|row| row.get())
        .filter(|row| {
            let value: Value = serde_json::from_str(row).expect("schedule catalog row");
            !exceeds_u32_usize(&value, None)
        })
        .collect();
    println!(
        "[akita] {file}: keeping {} of {} rows (dropped rows need a 64-bit usize)",
        kept.len(),
        envelope.rows.len()
    );
    let mut out = String::with_capacity(bytes.len());
    out.push_str(&text[..start]);
    out.push_str(&kept.join(ROW_SEPARATOR));
    out.push_str(&text[end..]);
    out.into_bytes()
}

fn exceeds_u32_usize(value: &Value, key: Option<&str>) -> bool {
    match value {
        Value::Object(map) => map.iter().any(|(k, v)| exceeds_u32_usize(v, Some(k))),
        Value::Array(items) => items.iter().any(|v| exceeds_u32_usize(v, key)),
        Value::Number(n) => {
            key.is_some_and(|k| USIZE_SCHEDULE_FIELDS.contains(&k))
                && n.as_u64().is_some_and(|n| n > u64::from(u32::MAX))
        }
        _ => false,
    }
}

/// Runs the same trusted-catalog decoding the prover and verifier run at
/// setup (family binding, policy digest, canonical re-encoding, row audit).
fn validate_schedule_artifacts(artifacts: &AkitaScheduleArtifacts) {
    artifacts.dense_catalog().expect("dense catalog validates");
    for k in [AKITA_ONE_HOT_K16, AKITA_ONE_HOT_K256] {
        artifacts
            .one_hot_catalog(k)
            .unwrap_or_else(|error| panic!("one-hot K={k} catalog validates: {error}"));
    }
}

fn write_file(public_dir: &Path, filename: &str, program: &str, kind: &str, bytes: &[u8]) {
    let path = public_dir.join(filename);
    std::fs::write(&path, bytes).expect("write file");
    println!("[{program}] {kind}: {} bytes -> {path:?}", bytes.len());
}

// Inline registration is inventory-based (link-time); keeping the crates
// linked is all the tracer needs.
use jolt_inlines_keccak256 as _;
use jolt_inlines_secp256k1 as _;
use jolt_inlines_sha2 as _;

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let public_dir = root.join("frontend/public");
    std::fs::create_dir_all(&public_dir).expect("create frontend/public");
    let target_root = std::env::var_os("JOLT_GUEST_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("target/guests"));

    // `jolt build -p` resolves the guest against the workspace of the current
    // directory, and the guests keep their own (guests/Cargo.toml).
    std::env::set_current_dir(root.join("guests")).expect("enter guests/ workspace");
    emit_schedule_artifacts(&public_dir);
    for guest in GUESTS {
        emit(guest, &public_dir, &target_root);
    }
    println!("Done!");
}
