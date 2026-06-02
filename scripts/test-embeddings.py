#!/usr/bin/env python3
"""Weight smoke-test for the ESM-2 protein-embedding asset packages.

Loads each model from a local model directory and either:
  (a) runs a synthetic sweep of (sequence count x sequence length) batches, or
  (b) embeds real sequences read from a file (one amino-acid sequence per line).

Validates the embedding output against the `sequence-embeddings` block spec:
  - mean pooling across non-special residue positions (spec R11)
  - output embedding dimension == model hidden_size (spec R14/R17:
    640 for esm2-150M, 1280 for esm2-650M)
  - penultimate-layer extraction (spec R12: layer 29 for 150M, 32 for 650M)

It loads weights purely from a local path via from_pretrained(local_path) with
no HuggingFace Hub call (spec R22a), exactly as the block runtime will. The
model directory is self-contained, so you can copy just that directory (plus
this script) to a GPU machine and run there.

Examples:
  # named asset(s) in this repo, synthetic sweep
  test-embeddings.py --models esm2-150M --device cpu

  # an explicit model directory copied to a GPU box
  test-embeddings.py --model-path /data/esm2-650M --device cuda --dtype fp16

  # embed real sequences from a file, save the vectors
  test-embeddings.py --model-path /data/esm2-650M \\
      --seqs-file peptides.txt --batch-size 64 --out peptides.emb.tsv
"""
import argparse
import json
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer, EsmModel

REPO = Path(__file__).resolve().parent.parent

# Named assets in this repo (the `root` of each entrypoint's indexed_model dir).
MODEL_DIRS = {
    "esm2-150M": REPO / "esm2-150M" / "indexed_model" / "esm2-150M",
    "esm2-650M": REPO / "esm2-650M" / "indexed_model" / "esm2-650M",
}

# A fixed 20-aa canonical alphabet to synthesize reproducible test sequences.
AA = "ACDEFGHIKLMNPQRSTVWY"


def make_sequences(count: int, length: int) -> list[str]:
    """Deterministic pseudo-random canonical-AA sequences of a fixed length."""
    seqs = []
    for i in range(count):
        # Simple deterministic generator; no randomness so runs are comparable.
        seq = "".join(AA[(i * 31 + j * 17) % len(AA)] for j in range(length))
        seqs.append(seq)
    return seqs


def read_seqs_file(path: Path) -> list[str]:
    """One sequence per line. Skips blank lines and FASTA-style headers
    (lines starting with '>' or '#'). Uppercases and strips whitespace."""
    seqs = []
    for raw in path.read_text().splitlines():
        s = raw.strip().upper()
        if not s or s.startswith(">") or s.startswith("#"):
            continue
        seqs.append(s)
    return seqs


def pick_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "mps":
        return torch.device("mps")
    # auto: prefer CUDA (GPU box), then Apple MPS, else CPU.
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(spec: str, device: torch.device) -> torch.dtype:
    """fp16 on CUDA by default (spec: GPU runs ESM-2 650M in fp16); fp32 else."""
    if spec == "fp16":
        return torch.float16
    if spec == "fp32":
        return torch.float32
    # auto
    return torch.float16 if device.type == "cuda" else torch.float32


def resolve_layer(spec: str, num_layers: int) -> int:
    """Map a layer spec to a hidden_states index.

    hidden_states has length num_layers + 1 (index 0 = embeddings).
    'last' -> num_layers ; 'penultimate' -> num_layers - 1 (spec R12).
    """
    if spec == "last":
        return num_layers
    if spec == "penultimate":
        return num_layers - 1
    return int(spec)


def resolve_targets(args) -> list[tuple[str, Path]]:
    """Build the (label, model_dir) list from --models and/or --model-path."""
    targets: list[tuple[str, Path]] = []
    for name in args.models or []:
        targets.append((name, MODEL_DIRS[name]))
    for p in args.model_path or []:
        d = Path(p).expanduser().resolve()
        targets.append((d.name, d))
    if not targets:  # default: both named assets
        targets = [(n, d) for n, d in MODEL_DIRS.items()]
    return targets


@torch.no_grad()
def embed(model, tokenizer, seqs, device, layer_idx, max_length):
    """Tokenize, forward, mean-pool over real residue tokens (spec R11).

    Excludes special tokens (<cls>/<eos>/<pad>) by zeroing every special-token
    id in the attention-masked positions, so only true residue positions
    contribute to the mean.
    """
    enc = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True,
                    max_length=max_length, add_special_tokens=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True)
    hidden = out.hidden_states[layer_idx]  # (B, T, D)

    # <eos> position varies per row with length, so derive the residue mask
    # per-token from the special-token ids rather than a fixed layout.
    attn = enc["attention_mask"].unsqueeze(-1).float()  # (B,T,1)
    ids = enc["input_ids"]
    residue_mask = torch.ones_like(ids, dtype=torch.float)
    for sid in set(tokenizer.all_special_ids):
        residue_mask[ids == sid] = 0.0
    residue_mask = (residue_mask.unsqueeze(-1) * attn).to(hidden.dtype)  # (B,T,1)

    summed = (hidden * residue_mask).sum(dim=1)         # (B,D)
    counts = residue_mask.sum(dim=1).clamp(min=1.0)     # (B,1)
    return (summed / counts).float().cpu()


def load_model(label, mdir, device, dtype, layer_spec):
    cfg = json.load(open(mdir / "config.json"))
    num_layers = cfg["num_hidden_layers"]
    expected_dim = cfg["hidden_size"]
    layer_idx = resolve_layer(layer_spec, num_layers)
    print(f"\n=== {label}  (dir={mdir})\n    hidden={expected_dim}, "
          f"layers={num_layers}, extract hidden_states[{layer_idx}], "
          f"device={device}, dtype={dtype} ===")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(mdir))
    # add_pooling_layer=False: we mean-pool ourselves (R11), the pooler is unused.
    model = EsmModel.from_pretrained(
        str(mdir), add_pooling_layer=False, torch_dtype=dtype).to(device).eval()
    print(f"  load: {time.perf_counter() - t0:6.2f}s")
    return tokenizer, model, layer_idx, expected_dim


def run_synthetic(tok, model, layer_idx, expected_dim, counts, lengths,
                  device, max_length):
    print(f"  {'count':>6} {'len':>5} {'fwd_s':>8} {'seq/s':>8} "
          f"{'dim':>5} {'ok':>4}")
    for length in lengths:
        for count in counts:
            seqs = make_sequences(count, length)
            t1 = time.perf_counter()
            emb = embed(model, tok, seqs, device, layer_idx, max_length)
            dt = time.perf_counter() - t1
            ok = (emb.shape == (count, expected_dim)
                  and torch.isfinite(emb).all().item())
            print(f"  {count:6d} {length:5d} {dt:8.3f} {count / dt:8.1f} "
                  f"{emb.shape[1]:5d} {'PASS' if ok else 'FAIL':>4}")


def run_file(tok, model, layer_idx, expected_dim, seqs, batch_size,
             device, max_length, out_path):
    lens = [len(s) for s in seqs]
    n_trunc = sum(1 for ln in lens if ln > max_length - 2)  # -2 for <cls>/<eos>
    print(f"  sequences: {len(seqs)}  len[min/mean/max]="
          f"{min(lens)}/{sum(lens) // len(lens)}/{max(lens)}  "
          f"batch_size={batch_size}"
          + (f"  (truncated to {max_length - 2} aa: {n_trunc})" if n_trunc else ""))
    all_emb = []
    t0 = time.perf_counter()
    for i in range(0, len(seqs), batch_size):
        all_emb.append(embed(model, tok, seqs[i:i + batch_size], device,
                             layer_idx, max_length))
    emb = torch.cat(all_emb, dim=0)
    dt = time.perf_counter() - t0
    ok = (emb.shape == (len(seqs), expected_dim)
          and torch.isfinite(emb).all().item())
    print(f"  embedded {emb.shape[0]} seqs in {dt:.3f}s "
          f"({emb.shape[0] / dt:.1f} seq/s), dim={emb.shape[1]}, "
          f"{'PASS' if ok else 'FAIL'}")
    if out_path:
        out = Path(out_path)
        if out.suffix == "":
            out = out.with_suffix(".tsv")
        with open(out, "w") as fh:
            fh.write("sequence\t" + "\t".join(
                f"dim_{j}" for j in range(emb.shape[1])) + "\n")
            for s, vec in zip(seqs, emb.tolist()):
                fh.write(s + "\t" + "\t".join(f"{v:.6f}" for v in vec) + "\n")
        print(f"  wrote embeddings -> {out}  ({emb.shape[0]} x {emb.shape[1]})")


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--models", nargs="+", choices=list(MODEL_DIRS),
                    help="named asset(s) in this repo")
    ap.add_argument("--model-path", nargs="+",
                    help="explicit model director(ies) (a HF ESM-2 checkpoint "
                         "dir with config.json + weights)")
    ap.add_argument("--seqs-file",
                    help="file with one amino-acid sequence per line; when set, "
                         "embeds these instead of synthetic sequences")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="batch size for --seqs-file embedding")
    ap.add_argument("--out", help="write per-sequence embeddings TSV "
                                  "(only with --seqs-file)")
    ap.add_argument("--counts", nargs="+", type=int, default=[1, 8, 32],
                    help="synthetic-sweep sequence counts (no --seqs-file)")
    ap.add_argument("--lengths", nargs="+", type=int, default=[16, 128, 512],
                    help="synthetic-sweep sequence lengths (no --seqs-file)")
    ap.add_argument("--max-length", type=int, default=1024,
                    help="token cap incl. specials; ESM-2 hard limit is 1026")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "fp32"],
                    help="auto = fp16 on CUDA, fp32 elsewhere (spec: GPU fp16)")
    ap.add_argument("--layer", default="penultimate",
                    help="penultimate | last | <int hidden_states index>")
    args = ap.parse_args()

    transformers.logging.set_verbosity_error()  # silence lm_head load report
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    torch.manual_seed(0)
    print(f"torch {torch.__version__}  device={device}  dtype={dtype}")

    seqs = read_seqs_file(Path(args.seqs_file)) if args.seqs_file else None
    if args.seqs_file and not seqs:
        raise SystemExit(f"no sequences found in {args.seqs_file}")

    for label, mdir in resolve_targets(args):
        if not mdir.exists():
            print(f"\n[SKIP] {label}: model dir not found at {mdir}")
            continue
        tok, model, layer_idx, dim = load_model(
            label, mdir, device, dtype, args.layer)
        if seqs is not None:
            run_file(tok, model, layer_idx, dim, seqs, args.batch_size,
                     device, args.max_length, args.out)
        else:
            run_synthetic(tok, model, layer_idx, dim, args.counts,
                          args.lengths, device, args.max_length)


if __name__ == "__main__":
    main()
