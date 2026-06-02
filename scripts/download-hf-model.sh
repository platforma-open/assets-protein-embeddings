#!/usr/bin/env bash
#
# Download a HuggingFace model into the variant's indexed_model/ directory.
#
# Usage:
#   download-hf-model.sh <modelInfo.json>
#
# modelInfo.json schema:
#   {
#     "huggingface_repo": "facebook/esm2_t33_650M_UR50D",
#     "revision": "main",                  // commit hash recommended for reproducibility
#     "target_dir": "esm2-650M"
#   }
#
# The asset's package.json must declare `block-software.entrypoints.main.asset.root`
# pointing at `./indexed_model/<target_dir>` — pl-pkg picks up the downloaded files from there.

set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <modelInfo.json>" >&2
    exit 1
fi

MODEL_INFO="$1"

if [ ! -f "$MODEL_INFO" ]; then
    echo "Error: $MODEL_INFO not found" >&2
    exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
    echo "Error: jq is required (install via 'brew install jq' or 'apt-get install jq')" >&2
    exit 1
fi

HF_REPO=$(jq -r '.huggingface_repo' "$MODEL_INFO")
REVISION=$(jq -r '.revision // "main"' "$MODEL_INFO")
TARGET_DIR=$(jq -r '.target_dir' "$MODEL_INFO")

if [ -z "$HF_REPO" ] || [ "$HF_REPO" = "null" ]; then
    echo "Error: huggingface_repo missing from $MODEL_INFO" >&2
    exit 1
fi
if [ -z "$TARGET_DIR" ] || [ "$TARGET_DIR" = "null" ]; then
    echo "Error: target_dir missing from $MODEL_INFO" >&2
    exit 1
fi

OUTPUT_DIR="indexed_model/$TARGET_DIR"
mkdir -p "$OUTPUT_DIR"

# Install huggingface_hub on demand. The asset build can run in CI with a clean
# Python; we don't want a global pip dependency on every developer's machine.
if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "huggingface-cli not found; installing huggingface_hub via pip..."
    pip install --quiet --user 'huggingface_hub>=0.20,<1.0' || {
        echo "Error: failed to install huggingface_hub. Try 'pip install huggingface_hub'." >&2
        exit 1
    }
    # User-installed binaries live under ~/.local/bin (or pip's default user scripts dir).
    PATH="$HOME/.local/bin:$PATH"
    export PATH
fi

echo "Downloading $HF_REPO @ $REVISION to $OUTPUT_DIR ..."
huggingface-cli download "$HF_REPO" \
    --revision "$REVISION" \
    --local-dir "$OUTPUT_DIR" \
    --local-dir-use-symlinks False

echo "Done. Files in $OUTPUT_DIR:"
ls -la "$OUTPUT_DIR" | head -20
