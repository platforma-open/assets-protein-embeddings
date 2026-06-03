# assets-protein-embeddings

Protein language model weights, distributed as Platforma assets. Consumed by `blocks/sequence-embeddings`.

## Variants

| Variant     | Source                                            | Size    | Purpose                                                                                                    |
| ----------- | ------------------------------------------------- | ------- | ---------------------------------------------------------------------------------------------------------- |
| `esm2-650M` | `facebook/esm2_t33_650M_UR50D` on HuggingFace Hub | ~2.6 GB | Universal protein LM, GPU mode (fp16)                                                                      |
| `esm2-150M` | `facebook/esm2_t30_150M_UR50D` on HuggingFace Hub | ~600 MB | Universal protein LM, CPU mode (smaller checkpoint; ONNX int8 export happens block-side at packaging time) |

## Building locally

```bash
pnpm install
pnpm build       # downloads weights via huggingface_hub, then builds each asset tarball
```

The build step shells out to `scripts/download-hf-model.sh`, which uses `huggingface-cli` (installed on-demand if missing) to fetch the pinned model revision into each variant's `indexed_model/` directory. Only the `safetensors` weights plus tokenizer/config and the model card are downloaded — the redundant PyTorch (`pytorch_model.bin`) and TensorFlow (`tf_model.h5`) copies of the same weights are skipped, keeping each asset to roughly a third of the full-repo size.

## License

Both ESM-2 checkpoints are distributed by Meta under MIT (confirmed against the `facebook/esm2_*` HuggingFace model cards). Bundling and redistribution are permitted with attribution.

The HuggingFace snapshot ships no `LICENSE` file, so each variant directory carries the MIT text in a tracked `LICENSE` file (`esm2-150M/LICENSE`, `esm2-650M/LICENSE`, `Copyright (c) Meta Platforms, Inc. and affiliates.`). The build step copies it into `indexed_model/{variant}/LICENSE` so the license and copyright notice ship **inside the published asset, next to the weights** — satisfying MIT's redistribution requirement. The model card `README.md` is bundled too, for attribution.
