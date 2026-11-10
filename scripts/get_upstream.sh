#!/usr/bin/env bash
# Clone the authors' code at the reproduced commit and apply the DSM patch (loss only; U_t is unchanged).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${NTI_UPSTREAM:-$ROOT/upstream/neuralTI-solvation}"
COMMIT=f756b308c4391a4dfcde49d7e72c5922ebb623e5
if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/balintmate/neuralTI-solvation "$DEST"
fi
git -C "$DEST" checkout -q "$COMMIT"
if git -C "$DEST" apply --check "$ROOT/patches/dsm.patch" 2>/dev/null; then
  git -C "$DEST" apply "$ROOT/patches/dsm.patch"
  echo "patched $DEST"
else
  echo "patch already applied or does not apply cleanly: $DEST"
fi
