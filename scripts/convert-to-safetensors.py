#!/usr/bin/env python3
"""Convert a downloaded HF model directory from pytorch_model.bin to model.safetensors.

Some HuggingFace repos ship only the legacy PyTorch checkpoint (pytorch_model.bin)
and no safetensors copy — e.g. wukevin/tcr-bert. The embedding inference path
loads weights from safetensors, and the asset download step skips .bin by default,
so for those repos we load the model and re-save it with safe_serialization=True,
then drop the now-redundant .bin.

Usage:
    convert-to-safetensors.py <model_dir>

<model_dir> is the directory the weights were downloaded into (it must contain
config.json + pytorch_model.bin + tokenizer files).
"""

import os
import sys

from transformers import AutoModel


def main(model_dir: str) -> None:
    bin_path = os.path.join(model_dir, "pytorch_model.bin")
    if not os.path.exists(bin_path):
        # Nothing to convert — either already safetensors or a different layout.
        print(f"No pytorch_model.bin in {model_dir}; nothing to convert.")
        return

    # Load the base encoder (we only need the transformer body for embeddings;
    # any task head in the checkpoint is intentionally dropped). save_pretrained
    # with safe_serialization=True writes model.safetensors.
    model = AutoModel.from_pretrained(model_dir)
    model.save_pretrained(model_dir, safe_serialization=True)

    os.remove(bin_path)
    print(f"Removed {bin_path} (superseded by model.safetensors)")
    print(f"Wrote {os.path.join(model_dir, 'model.safetensors')}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: convert-to-safetensors.py <model_dir>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
