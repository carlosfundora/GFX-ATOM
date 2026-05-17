#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/local/ai/projects/gfxATOM-Rust"
TARGET="$ROOT/vendor-crates"

mkdir -p "$TARGET"
rm -rf "$TARGET"/*

# Canonical crate collections to mirror into the fusion workspace.
rsync -a --delete /home/local/ai/projects/rust/crates/ "$TARGET/canonical-rust-crates/"
rsync -a --delete /home/local/ai/projects/harness/rust/crates/ "$TARGET/harness-rust-crates/"
rsync -a --delete /home/local/ai/projects/DATALORE/rust/crates/ "$TARGET/datalore-rust-crates/"
rsync -a --delete /home/local/ai/projects/DEMERZEL/rust/crates/ "$TARGET/demerzel-rust-crates/"
if [ -d /home/local/ai/projects/mirrors/rust/crates/harness ]; then
  rsync -a --delete /home/local/ai/projects/mirrors/rust/crates/harness/ "$TARGET/mirrors-rust-crates-harness/"
fi

echo "Rust crate mirror refreshed at $TARGET"
