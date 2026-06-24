#!/usr/bin/env bash
#
# Download a model weights archive from a direct URL (e.g. Zenodo) into the
# variant's indexed_model/ directory, then bundle the license.
#
# This is the non-HuggingFace counterpart to download-hf-model.sh: some models
# (e.g. AbLang2) distribute their weights as a tarball on an archive host rather
# than as a HuggingFace repo. The pip package provides the model *code*; this
# asset ships only the *weights*.
#
# Usage:
#   download-archive-model.sh <modelInfo.json>
#
# modelInfo.json schema:
#   {
#     "archive_url": "https://zenodo.org/records/10185169/files/ablang2-weights.tar.gz",
#     "target_dir": "ablang2",
#     "keep": ["model.pt", "hparams.json"],   // optional; default keeps everything extracted
#     "license": "BSD-3-Clause-Clear"          // informational
#   }
#
# Zenodo record URLs are immutable per version, so the URL itself pins the
# weights — there is no separate revision to record.
#
# The asset's package.json must declare `block-software.entrypoints.main.asset.root`
# pointing at `./indexed_model/<target_dir>` — pl-pkg picks up the files from there.

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

ARCHIVE_URL=$(jq -r '.archive_url' "$MODEL_INFO")
TARGET_DIR=$(jq -r '.target_dir' "$MODEL_INFO")

if [ -z "$ARCHIVE_URL" ] || [ "$ARCHIVE_URL" = "null" ]; then
    echo "Error: archive_url missing from $MODEL_INFO" >&2
    exit 1
fi
if [ -z "$TARGET_DIR" ] || [ "$TARGET_DIR" = "null" ]; then
    echo "Error: target_dir missing from $MODEL_INFO" >&2
    exit 1
fi

OUTPUT_DIR="indexed_model/$TARGET_DIR"
mkdir -p "$OUTPUT_DIR"

TMP_ARCHIVE="$OUTPUT_DIR/tmp-archive.tar.gz"
echo "Downloading $ARCHIVE_URL to $OUTPUT_DIR ..."
curl -fSL "$ARCHIVE_URL" -o "$TMP_ARCHIVE"

echo "Extracting ..."
tar -xzf "$TMP_ARCHIVE" -C "$OUTPUT_DIR"
rm -f "$TMP_ARCHIVE"

# Prune to the explicit keep-list when provided, so the published asset carries
# only the files the model actually needs (no extras the archive may bundle).
KEEP_JSON=$(jq -c '.keep // empty' "$MODEL_INFO")
if [ -n "$KEEP_JSON" ]; then
    # Build a newline list of basenames to keep (plus LICENSE, added below).
    KEEP_LIST=$(echo "$KEEP_JSON" | jq -r '.[]')
    echo "Pruning to keep-list: $(echo "$KEEP_JSON" | jq -rc 'join(", ")')"
    while IFS= read -r -d '' f; do
        base=$(basename "$f")
        if [ "$base" = "LICENSE" ]; then continue; fi
        if ! grep -qxF "$base" <<<"$KEEP_LIST"; then
            echo "  removing $base"
            rm -f "$f"
        fi
    done < <(find "$OUTPUT_DIR" -type f -print0)
fi

# Bundle the model license alongside the weights so it ships inside the asset.
LICENSE_SRC="$(dirname "$MODEL_INFO")/LICENSE"
if [ -f "$LICENSE_SRC" ]; then
    cp "$LICENSE_SRC" "$OUTPUT_DIR/LICENSE"
    echo "Bundled license: $LICENSE_SRC -> $OUTPUT_DIR/LICENSE"
else
    echo "WARNING: no LICENSE at $LICENSE_SRC; asset will ship without an explicit license." >&2
fi

echo "Done. Files in $OUTPUT_DIR:"
ls -la "$OUTPUT_DIR" | head -20
