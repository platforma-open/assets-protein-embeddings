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
#     "convert_to_safetensors": true                        // optional; see below
#   }
#
# By default only the safetensors weights plus tokenizer/config files are fetched.
# The PyTorch (pytorch_model.bin) and TensorFlow (tf_model.h5) checkpoints are
# byte-for-byte duplicates of the same weights — fetching them triples the asset
# size for no benefit, so they are excluded. Override `include` per model if a
# different file set is required.
#
# Some repos ship ONLY pytorch_model.bin and no safetensors (e.g. wukevin/tcr-bert).
# For those, set "include" to fetch pytorch_model.bin and set
# "convert_to_safetensors": true — after download the weights are re-saved as
# model.safetensors (via scripts/convert-to-safetensors.py) and the .bin dropped,
# so the published asset always carries safetensors.
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
CONVERT_TO_SAFETENSORS=$(jq -r '.convert_to_safetensors // false' "$MODEL_INFO")

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

# Convert a .bin-only checkpoint to safetensors when requested. Needs torch +
# transformers; install on demand if the active python can't import them. The
# helper drops pytorch_model.bin afterward so only safetensors ships.
if [ "$CONVERT_TO_SAFETENSORS" = "true" ]; then
    PYBIN="${PYTHON:-python3}"
    if ! "$PYBIN" -c 'import torch, transformers, safetensors' >/dev/null 2>&1; then
        echo "Installing torch/transformers/safetensors for .bin->safetensors conversion (one-time, large)..."
        "$PYBIN" -m pip install --quiet --user torch transformers safetensors || {
            echo "Error: failed to install conversion deps. Install torch+transformers, then re-run." >&2
            exit 1
        }
    fi
    echo "Converting pytorch_model.bin -> model.safetensors in $OUTPUT_DIR ..."
    "$PYBIN" "$(dirname "$0")/convert-to-safetensors.py" "$OUTPUT_DIR"
fi

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

# Drop the huggingface-cli download cache (.cache/huggingface/*.metadata resume
# stubs) so the published asset carries only the model files + LICENSE.
rm -rf "$OUTPUT_DIR/.cache"

echo "Done. Files in $OUTPUT_DIR:"
ls -la "$OUTPUT_DIR" | head -20
