set positional-arguments

# Same file list as scripts/lint/shell.sh.
shell_files := "setup-wasm-deps.sh scripts/build-pages.sh scripts/pre-commit.sh scripts/lint/*.sh"

# List recipes
help:
    @{{ just_executable() }} --list --unsorted

# Format Rust, Python (bench/) and shell scripts
fmt:
    cargo fmt -q --message-format=short
    uvx ruff@0.17.3 format bench
    shfmt -w {{ shell_files }}

# Check formatting without modifying files
fmt-check:
    cargo fmt -q --message-format=short -- --check
    uvx ruff@0.17.3 format --check bench
    shfmt -d {{ shell_files }}

# Apply Clippy's suggested fixes to the native build (scripts/lint/rust.sh fix); extra args go to cargo
fix *args:
    scripts/lint/rust.sh fix "$@"

# Rust lint as in CI (rustfmt check + Clippy): scope native, device or wasm; default all
clippy *args:
    scripts/lint/rust.sh "$@"

# CI lint steps: scripts/lint/{rust,web,python,shell}.sh, then typos
lint:
    scripts/lint/rust.sh
    scripts/lint/web.sh
    scripts/lint/python.sh
    scripts/lint/shell.sh
    typos

# Native tests (release build); extra args go to cargo nextest run
test *args:
    cargo nextest run --cargo-quiet --release --features native "$@"

# Clone and patch the wasm32 dependencies into .wasm-deps/; --revert restores Cargo.toml
setup-wasm-deps *args:
    ./setup-wasm-deps.sh "$@"

# Build pkg/ with the full browser feature set (needs setup-wasm-deps)
wasm:
    RUSTC_BOOTSTRAP=1 CARGO_UNSTABLE_BUILD_STD="panic_abort,std" wasm-pack build --release --target web -- --features webgpu,trace-commit-device,digit-range-device,relation-range-device

# Release build with the native feature (library and both binaries); extra args go to cargo
build-native *args:
    cargo build -q --message-format=short --release --features native "$@"

# Build the guests and write their artifacts to frontend/public/ (needs the jolt CLI)
generate:
    cargo run -q --message-format=short --release --features native --bin generate-preprocessing

# Prove and verify sha2, ecdsa and sha2_chain natively; extra args go to cargo run
roundtrip *args:
    cargo run -q --message-format=short --release --features native --bin test-roundtrip "$@"

# Vite dev server on http://localhost:8080; extra args go to vite
dev *args:
    npm --prefix frontend run dev -- "$@"

# Type-check and build the frontend into frontend/dist/; extra args go to vite build
build *args:
    npm --prefix frontend run build -- "$@"

# Serve frontend/dist/ and pkg/ on http://127.0.0.1:8080
serve:
    node server.mjs

# Assemble frontend/dist/ for Cloudflare Pages; --wasm runs setup-wasm-deps and wasm first
build-pages *args:
    scripts/build-pages.sh "$@"
