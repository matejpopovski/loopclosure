#!/usr/bin/env bash
# Download the Intel Research Lab log from its source and check it against the
# checksum published on the DSpace@MIT record. The repository already carries a
# copy, so this is only needed to audit that copy.
set -euo pipefail
cd "$(dirname "$0")/.."
url="https://dspace.mit.edu/server/api/core/bitstreams/6edc81dc-3d77-4f15-8e23-f3b336f85ebf/content"
expected="a12279da144d9a974520dffbcaf86693"
target="${1:-data/intel.script.gz}"
curl --fail --location --silent --show-error --output "$target" "$url"
if command -v md5sum >/dev/null; then
  actual="$(md5sum "$target" | cut -d' ' -f1)"
else
  actual="$(md5 -q "$target")"
fi
if [ "$actual" != "$expected" ]; then
  echo "checksum mismatch: got $actual, expected $expected" >&2
  exit 1
fi
echo "verified $target"
