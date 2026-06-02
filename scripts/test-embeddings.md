# Weight smoke-test — `test-embeddings.py`

Loads an ESM-2 model from a local directory and either runs a synthetic sweep
of (sequence count × sequence length) batches, or embeds real sequences from a
file. Validates output against the `sequence-embeddings` block spec: mean
pooling over residue tokens (R11), penultimate-layer extraction (R12), and
embedding dims (R14/R17: 640 for esm2-150M, 1280 for esm2-650M — derived from
the model's `config.json`, so any ESM-2 checkpoint dir works). No HuggingFace
Hub call — it loads purely from the local path, exactly as the block runtime
will (R22a).

## Pointing at a specific model directory

`--models esm2-150M` is a shortcut for the named asset in *this repo*; it
resolves to `esm2-150M/indexed_model/esm2-150M/`. That directory is a complete,
self-contained HF checkpoint (`config.json`, `model.safetensors`, tokenizer
files). To run on another machine, copy that directory plus this script, then
point at it explicitly:

```bash
test-embeddings.py --model-path /data/esm2-650M --device cuda --dtype fp16
```

`--model-path` takes any directory containing `config.json` + weights; the
expected embedding dim is read from `config.json` (no hardcoded assumption).

## Prerequisites

1. **Build the assets first** so the weights exist locally:
   ```bash
   cd support/assets-protein-embeddings
   env -u PL_PKG_DEV pnpm build        # PL_PKG_DEV must NOT be 'local' for assets
   ```
   This populates `esm2-150M/indexed_model/esm2-150M/` and
   `esm2-650M/indexed_model/esm2-650M/`.

2. **A Python env with `torch` + `transformers`.** Any 3.10+ works.

## Setup (CPU / Apple Silicon — e.g. this Mac)

```bash
cd support/assets-protein-embeddings
python3 -m venv .venv-test            # .venv-test/ is gitignored
.venv-test/bin/pip install torch transformers numpy
```

## Setup (GPU box — CUDA)

Install the CUDA build of torch (pick the index URL for your CUDA version):

```bash
python3 -m venv .venv-test
.venv-test/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121
.venv-test/bin/pip install transformers numpy
```

Verify CUDA is visible:
```bash
.venv-test/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Running

### Synthetic sweep (default)

```bash
# Both models, auto device (CUDA > MPS > CPU), default sweep
.venv-test/bin/python scripts/test-embeddings.py

# CPU only, just the small model
.venv-test/bin/python scripts/test-embeddings.py --models esm2-150M --device cpu

# GPU, fp16 (spec default for GPU), bigger batches + a long sequence
.venv-test/bin/python scripts/test-embeddings.py \
    --device cuda --dtype fp16 \
    --counts 8 32 128 256 --lengths 32 128 512 1022
```

### Real sequences from a file

Provide one amino-acid sequence per line (blank lines and `>`/`#` header lines
are skipped). The sweep flags `--counts`/`--lengths` are ignored; batching is
controlled by `--batch-size`. Add `--out` to save the per-sequence vectors as a
TSV (`sequence`, `dim_0` … `dim_{D-1}`):

```bash
.venv-test/bin/python scripts/test-embeddings.py \
    --model-path /data/esm2-650M --device cuda --dtype fp16 \
    --seqs-file my_peptides.txt --batch-size 64 --out my_peptides.emb.tsv
```

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--models` | both (if no `--model-path`) | named asset(s) in this repo: `esm2-150M`, `esm2-650M` |
| `--model-path` | — | explicit model director(ies); dim read from `config.json` |
| `--seqs-file` | — | file with one sequence per line; embeds these instead of synthetic |
| `--batch-size` | `32` | batch size for `--seqs-file` embedding |
| `--out` | — | write per-sequence embeddings TSV (only with `--seqs-file`) |
| `--counts` | `1 8 32` | synthetic-sweep sequence counts (ignored with `--seqs-file`) |
| `--lengths` | `16 128 512` | synthetic-sweep sequence lengths aa (ignored with `--seqs-file`) |
| `--max-length` | `1024` | token cap incl. specials; sequences longer are truncated. ESM-2 hard limit 1026 |
| `--device` | `auto` | `auto` \| `cpu` \| `cuda` \| `mps` |
| `--dtype` | `auto` | `auto` (fp16 on CUDA, fp32 else) \| `fp16` \| `fp32` |
| `--layer` | `penultimate` | `penultimate` (R12) \| `last` \| `<int>` hidden_states index |

## Reading the output

Each row reports forward-pass time (`fwd_s`), throughput (`seq/s`), the output
embedding dimension (`dim`), and `PASS`/`FAIL`. `PASS` means the output shape is
`(count, expected_dim)` and all values are finite.

In file mode it prints sequence count, length stats (min/mean/max), throughput,
the embedding dim, and `PASS`/`FAIL`; with `--out` it also writes the TSV.

`count=1` rows are often slow — that is torch warmup/thread spin-up, not real
per-sequence cost. Read throughput from the larger-batch rows.

The transformers load report (`lm_head ... UNEXPECTED`) is silenced — it is
expected and harmless: we load `EsmModel` (encoder only), not the MLM head.
