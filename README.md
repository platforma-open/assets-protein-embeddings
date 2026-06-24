# assets-protein-embeddings

Protein language model weights, distributed as Platforma assets. Consumed by `blocks/sequence-embeddings`.

## Variants

| Variant         | Source                                            | Dim  | Purpose                                                                                                    |
| --------------- | ------------------------------------------------- | ---- | ---------------------------------------------------------------------------------------------------------- |
| `esm2-650M`     | `facebook/esm2_t33_650M_UR50D` on HuggingFace Hub | 1280 | Universal protein LM, GPU mode (fp16). Canonical peptide model + universal fallback                        |
| `esm2-150M`     | `facebook/esm2_t30_150M_UR50D` on HuggingFace Hub | 640  | Universal protein LM, CPU mode (smaller checkpoint; ONNX int8 export happens block-side at packaging time) |
| `currab`        | `brineylab/CurrAb` on HuggingFace Hub             | 1280 | Antibody specialist (ESM-2 650M arch; heavy / light / paired)                                              |
| `vhhbert`       | `COGNANO/VHHBERT` on HuggingFace Hub              | 768  | VHH / nanobody specialist (RoBERTa-base)                                                                    |
| `h3berta`       | `Chrode/H3BERTa` on HuggingFace Hub               | 768  | Antibody CDR-H3 specialist (RoBERTa-base; heavy CDR3, transfers to VHH CDR3)                                |
| `tcr-bert`      | `wukevin/tcr-bert` on HuggingFace Hub             | 768  | TCR specialist (BERT; TCR canonical + TCR CDR3)                                                             |
| `peptideclm-2`  | `aaronfeller/peptideclm-2-hybrid-large` on HF Hub | 1024 | Peptide specialist for non-canonical / cyclic peptides (SMILES input; `trust_remote_code`)                 |
| `ablang2`       | Zenodo record `10185169` (`ablang2` pip package)  | 480  | Paired antibody specialist. Weights only; model code + tokenizer come from the `ablang2` runenv package    |

## Building locally

```bash
pnpm install
pnpm build       # downloads weights via huggingface_hub, then builds each asset tarball
```

The build step shells out to `scripts/download-hf-model.sh`, which uses `huggingface-cli` (installed on-demand if missing) to fetch the pinned model revision into each variant's `indexed_model/` directory. Only the `safetensors` weights plus tokenizer/config and the model card are downloaded — the redundant PyTorch (`pytorch_model.bin`) and TensorFlow (`tf_model.h5`) copies of the same weights are skipped, keeping each asset to roughly a third of the full-repo size.

Each variant's file set is driven by its `modelInfo.json` (`huggingface_repo`, pinned `revision`, `target_dir`, `license`). The download script's default `include` list matches the ESM-2 layout (`vocab.txt` tokenizer); variants whose tokenizer differs override `include`:

- **`h3berta`** — RoBERTa tokenizer, so `include` also fetches `tokenizer.json` + `added_tokens.json`.
- **`peptideclm-2`** — SMILES model with no `vocab.txt` (uses `tokenizer.json`), and it ships **custom modeling code** (`ChemPepMTR.py`, `config.py`, `__init__.py`) that `include` bundles. The block must load it with `trust_remote_code=True`. PeptideCLM-2 consumes **SMILES**, not amino-acid strings — AA→SMILES conversion (when needed) is block-side inference logic, never part of this static asset.
- **`tcr-bert`** — upstream ships only `pytorch_model.bin` (no safetensors). `convert_to_safetensors: true` makes the script load the weights and re-save them as `model.safetensors` (via `scripts/convert-to-safetensors.py`, which needs torch + transformers — installed on demand if absent), then drop the `.bin`. The published asset always carries safetensors.

`ablang2` is **not** a HuggingFace model: AbLang2 distributes its weights as a tarball on Zenodo, and the `ablang2` pip package (in the torch-cuda runenv) carries the model code + tokenizer but **not** the weights. So this variant uses a separate downloader, `scripts/download-archive-model.sh`, driven by an `archive_url` + `keep` list in its `modelInfo.json` (Zenodo record URLs are immutable, so the URL pins the version — there is no `revision`). The asset ships only `model.pt` + `hparams.json`. At runtime the block copies these into the pip package's `model-weights-ablang2-paired/` directory so `ablang2.pretrained()` loads them offline instead of fetching from Zenodo — that runtime placement is block-side logic, not part of this static asset.

## License

All variants are commercially redistributable. Each variant directory carries a tracked `LICENSE` file (the HuggingFace snapshots ship none of their own), which the build step copies into `indexed_model/{variant}/LICENSE` so the license and copyright notice ship **inside the published asset, next to the weights** — satisfying the redistribution requirement. The model card `README.md` is bundled too, for attribution.

| Variant        | License    | Copyright                                                        | Verified against                                   |
| -------------- | ---------- | --------------------------------------------------------------- | -------------------------------------------------- |
| `esm2-*`       | MIT        | Meta Platforms, Inc. and affiliates                             | `facebook/esm2_*` HF model cards                   |
| `currab`       | MIT        | 2025 brineylab                                                  | `github.com/brineylab/curriculum-paper` LICENSE    |
| `vhhbert`      | MIT        | 2024 COGNANO, Inc.                                              | `github.com/cognano/AVIDa-SARS-CoV-2` LICENSE      |
| `h3berta`      | MIT        | 2025 H3BERTa authors (IBMM, University of Bern)                 | `Chrode/H3BERTa` HF card (no upstream LICENSE file) |
| `tcr-bert`     | Apache-2.0 | (see LICENSE)                                                   | `github.com/wukevin/tcr-bert` LICENSE (verbatim)   |
| `peptideclm-2` | MIT        | 2025 Aaron Feller                                              | `github.com/AaronFeller/{PeptideCLM-2,PeptideMTR}` LICENSE |
| `ablang2`      | BSD-3-Clause-Clear | 2021 Tobias Hegelund Olsen                             | Zenodo record `10185169` ("AbLang2 weights") license field |

`tcr-bert` ships the full Apache-2.0 text verbatim; the upstream repo carries no `NOTICE` file, so the `LICENSE` alone satisfies redistribution. `ablang2` ships the **Clear** BSD-3-Clause variant — that is the license declared on the Zenodo *weights* record (the `ablang2` code wheel is plain BSD-3-Clause). Licenses were re-verified at source on 2026-06-24.
