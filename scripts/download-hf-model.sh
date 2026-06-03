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
#     "target_dir": "esm2-650M",
#     "include": ["config.json", "model.safetensors", ...]  // optional; see default below
#   }
#
# By default only the safetensors weights plus tokenizer/config files are fetched.
# The PyTorch (pytorch_model.bin) and TensorFlow (tf_model.h5) checkpoints are
# byte-for-byte duplicates of the same weights — fetching them triples the asset
# size for no benefit, so they are excluded. Override `include` per model if a
# different file set is required.
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

# Files to fetch. Override per-model via an "include" array in modelInfo.json.
# Default: safetensors weights + tokenizer/config + model card, skipping the
# redundant PyTorch (.bin) and TensorFlow (.h5) duplicates of the same weights.
DEFAULT_INCLUDE='["config.json","model.safetensors","tokenizer_config.json","special_tokens_map.json","vocab.txt","README.md"]'
INCLUDE_JSON=$(jq -c --argjson def "$DEFAULT_INCLUDE" '.include // $def' "$MODEL_INFO")

INCLUDE_ARGS=(--include)
while IFS= read -r pattern; do
    INCLUDE_ARGS+=("$pattern")
done < <(echo "$INCLUDE_JSON" | jq -r '.[]')

echo "Downloading $HF_REPO @ $REVISION to $OUTPUT_DIR (files: $(echo "$INCLUDE_JSON" | jq -rc 'join(", ")')) ..."
huggingface-cli download "$HF_REPO" \
    --revision "$REVISION" \
    "${INCLUDE_ARGS[@]}" \
    --local-dir "$OUTPUT_DIR" \
    --local-dir-use-symlinks False

# Bundle the model license alongside the weights so it ships inside the asset.
# MIT requires the license text and copyright notice to accompany any
# redistribution; HF does not ship a LICENSE file, so we provide our own
# (tracked next to modelInfo.json) and copy it into the asset root.
LICENSE_SRC="$(dirname "$MODEL_INFO")/LICENSE"
if [ -f "$LICENSE_SRC" ]; then
    cp "$LICENSE_SRC" "$OUTPUT_DIR/LICENSE"
    echo "Bundled license: $LICENSE_SRC -> $OUTPUT_DIR/LICENSE"
else
    echo "WARNING: no LICENSE at $LICENSE_SRC; asset will ship without an explicit license." >&2
fi

echo "Done. Files in $OUTPUT_DIR:"
ls -la "$OUTPUT_DIR" | head -20
