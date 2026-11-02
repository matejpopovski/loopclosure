#!/usr/bin/env bash
# Build the browser demo into web/: the wasm package, plus the log it replays.
set -euo pipefail
cd "$(dirname "$0")/.."
wasm-pack build crates/wasm --release --target web --out-dir ../../web/pkg --no-typescript
rm -f web/pkg/.gitignore web/pkg/package.json web/pkg/README.md
mkdir -p web/data
cp data/intel.script.gz web/data/
