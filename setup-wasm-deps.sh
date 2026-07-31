#!/usr/bin/env bash
# Browser (wasm32) proving needs two one-line fixes that are not upstream yet
# (see patches/ and the "WASM runtime patches" section of README.md). This
# script clones a16z/jolt and a16z/arkworks-algebra at the exact revs the
# manifest pins, applies the patches, and rewrites the override block in
# Cargo.toml so the whole dependency graph builds against the patched
# checkouts in .wasm-deps/.
#
#   ./setup-wasm-deps.sh            set up patched local deps
#   ./setup-wasm-deps.sh --revert   restore the pinned upstream-rev block
#
# Native builds (generate-preprocessing, test-roundtrip) do not need this.
set -euo pipefail

JOLT_REV=70a294ad58629af59ab89f646d6d0079b57174cb
ARK_REV=76bb3a4518928f1ff7f15875f940d614bb9845e6
JOLT_URL=https://github.com/a16z/jolt
ARK_URL=https://github.com/a16z/arkworks-algebra

ROOT="$(cd "$(dirname "$0")" && pwd)"
DEPS_REL=".wasm-deps"
DEPS="$ROOT/$DEPS_REL"
MANIFEST="$ROOT/Cargo.toml"
BEGIN_MARK='# >>> wasm-deps overrides'
END_MARK='# <<< wasm-deps overrides'

splice_block() { # splice_block <blockfile> — replaces marker block content in Cargo.toml
    local blockfile=$1
    grep -q "$BEGIN_MARK" "$MANIFEST" && grep -q "$END_MARK" "$MANIFEST" || {
        echo "error: override markers not found in Cargo.toml" >&2
        exit 1
    }
    awk -v blockfile="$blockfile" -v begin="$BEGIN_MARK" -v end="$END_MARK" '
        index($0, begin) == 1 {
            print
            while ((getline line < blockfile) > 0) print line
            skip = 1
            next
        }
        index($0, end) == 1 { skip = 0 }
        !skip { print }
    ' "$MANIFEST" > "$MANIFEST.tmp"
    mv "$MANIFEST.tmp" "$MANIFEST"
}

write_pinned_block() {
    cat > "$1" <<EOF
[patch.crates-io]
ark-bn254 = { git = "$ARK_URL", branch = "dev/twist-shout" }
ark-ff = { git = "$ARK_URL", branch = "dev/twist-shout" }
ark-ec = { git = "$ARK_URL", branch = "dev/twist-shout" }
ark-serialize = { git = "$ARK_URL", branch = "dev/twist-shout" }
EOF
}

write_local_block() {
    cat > "$1" <<EOF
# LOCAL patched checkouts (written by setup-wasm-deps.sh — do not commit)
[patch.crates-io]
ark-bn254 = { path = "$DEPS_REL/arkworks-algebra/curves/bn254" }
ark-ff = { path = "$DEPS_REL/arkworks-algebra/ff" }
ark-ec = { path = "$DEPS_REL/arkworks-algebra/ec" }
ark-serialize = { path = "$DEPS_REL/arkworks-algebra/serialize" }

[patch.'$ARK_URL']
ark-bn254 = { path = "$DEPS_REL/arkworks-algebra/curves/bn254" }
ark-ff = { path = "$DEPS_REL/arkworks-algebra/ff" }
ark-ec = { path = "$DEPS_REL/arkworks-algebra/ec" }
ark-serialize = { path = "$DEPS_REL/arkworks-algebra/serialize" }
ark-poly = { path = "$DEPS_REL/arkworks-algebra/poly" }
ark-secp256k1 = { path = "$DEPS_REL/arkworks-algebra/curves/secp256k1" }
jolt-optimizations = { path = "$DEPS_REL/arkworks-algebra/jolt-optimizations" }

[patch.'$JOLT_URL']
jolt-prover = { path = "$DEPS_REL/jolt/crates/jolt-prover" }
jolt-verifier = { path = "$DEPS_REL/jolt/crates/jolt-verifier" }
jolt-witness = { path = "$DEPS_REL/jolt/crates/jolt-witness" }
jolt-dory = { path = "$DEPS_REL/jolt/crates/jolt-dory" }
jolt-field = { path = "$DEPS_REL/jolt/crates/jolt-field" }
jolt-crypto = { path = "$DEPS_REL/jolt/crates/jolt-crypto" }
jolt-transcript = { path = "$DEPS_REL/jolt/crates/jolt-transcript" }
jolt-program = { path = "$DEPS_REL/jolt/crates/jolt-program" }
tracer = { path = "$DEPS_REL/jolt/tracer" }
common = { path = "$DEPS_REL/jolt/common" }
jolt-sdk = { path = "$DEPS_REL/jolt/jolt-sdk" }
jolt-inlines-sha2 = { path = "$DEPS_REL/jolt/jolt-inlines/sha2" }
jolt-inlines-secp256k1 = { path = "$DEPS_REL/jolt/jolt-inlines/secp256k1" }
jolt-inlines-keccak256 = { path = "$DEPS_REL/jolt/jolt-inlines/keccak256" }
EOF
}

if [[ "${1:-}" == "--revert" ]]; then
    block="$(mktemp)"
    write_pinned_block "$block"
    splice_block "$block"
    rm -f "$block"
    echo "Cargo.toml override block restored to pinned upstream revs."
    echo "Remove $DEPS_REL/ manually if no longer needed."
    exit 0
fi

fetch_at_rev() { # fetch_at_rev <dir> <url> <rev>
    local dir=$1 url=$2 rev=$3
    rm -rf "$dir"
    mkdir -p "$dir"
    git -C "$dir" init -q
    git -C "$dir" remote add origin "$url"
    git -C "$dir" fetch -q --depth 1 origin "$rev"
    git -C "$dir" checkout -q --detach FETCH_HEAD
}

echo "Fetching a16z/jolt @ $JOLT_REV ..."
fetch_at_rev "$DEPS/jolt" "$JOLT_URL" "$JOLT_REV"
echo "Fetching a16z/arkworks-algebra @ $ARK_REV ..."
fetch_at_rev "$DEPS/arkworks-algebra" "$ARK_URL" "$ARK_REV"

echo "Applying wasm32 patches ..."
git -C "$DEPS/jolt" apply --verbose "$ROOT/patches/0001-jolt-coefflut-u64.patch"
git -C "$DEPS/arkworks-algebra" apply --verbose "$ROOT/patches/0002-arkworks-wasm-nested-pool.patch"

block="$(mktemp)"
write_local_block "$block"
splice_block "$block"
rm -f "$block"

cat <<'DONE'

Done. Cargo.toml now builds against the patched checkouts in .wasm-deps/.
Next:
  CARGO_UNSTABLE_BUILD_STD="panic_abort,std" wasm-pack build --release --target web
(restart any running server.mjs afterwards — it caches compressed responses)

Do not commit Cargo.toml/Cargo.lock in this state; restore the pinned block
with ./setup-wasm-deps.sh --revert before committing.
DONE
