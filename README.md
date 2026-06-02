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

The build step shells out to `scripts/download-hf-model.sh`, which uses `huggingface-cli` (installed on-demand if missing) to fetch the pinned model revision into each variant's `indexed_model/` directory.

## License

Both ESM-2 checkpoints are distributed by Meta under MIT (confirmed against the `facebook/esm2_*` HuggingFace model cards). Bundling and redistribution are permitted with attribution. See the per-variant LICENSE file inside each `indexed_model/{variant}/` directory after a successful build (HF includes the LICENSE in the snapshot download).
