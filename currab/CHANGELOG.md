# @platforma-open/milaboratories.protein-embeddings-assets.currab

## 1.1.0

### Minor Changes

- 51efa05: Add modality-specialist embedding models for the sequence-embeddings block (v2):

  - **CurrAb** (`brineylab/CurrAb`, MIT, 1280-dim) — antibody specialist, ESM-2 650M architecture
  - **VHHBERT** (`COGNANO/VHHBERT`, MIT, 768-dim) — VHH / nanobody specialist
  - **H3BERTa** (`Chrode/H3BERTa`, MIT, 768-dim) — antibody CDR-H3 specialist
  - **TCR-BERT** (`wukevin/tcr-bert`, Apache-2.0, 768-dim) — TCR specialist; converted from the upstream `.bin`-only checkpoint to safetensors at build time
  - **PeptideCLM-2** (`aaronfeller/peptideclm-2-hybrid-large`, MIT, 1024-dim) — non-canonical / cyclic peptide specialist (SMILES input, `trust_remote_code`)
  - **AbLang2** (Zenodo `10185169`, BSD-3-Clause-Clear, 480-dim) — paired antibody specialist. Weights-only asset (`model.pt` + `hparams.json`); the model code + tokenizer come from the `ablang2` pip package in the runenv. Fetched via the new `scripts/download-archive-model.sh` (Zenodo, not HuggingFace).

  HuggingFace revisions are pinned to a commit hash; AbLang2 is pinned by its immutable Zenodo record URL. Every asset ships the model license bundled inside it. ESM-2 remains the universal fallback and canonical peptide model.
