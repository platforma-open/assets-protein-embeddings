#!/usr/bin/env python3
"""Minimal ESM-2 embedding smoke-test: load local weights, embed, mean-pool.

Usage:
  test-embeddings-minimal.py [MODEL_DIR] [SEQ ...]
  # defaults: the 150M asset dir + three sample sequences
"""
import sys
import torch
from transformers import AutoTokenizer, EsmModel

MODEL = sys.argv[1] if len(sys.argv) > 1 else "esm2-150M/indexed_model/esm2-150M"
SEQS = sys.argv[2:] or ["SIINFEKL", "GILGFVFTL", "CASSLAPGATNEKLFF"]

tok = AutoTokenizer.from_pretrained(MODEL)
model = EsmModel.from_pretrained(MODEL, add_pooling_layer=False).eval()

enc = tok(SEQS, return_tensors="pt", padding=True)
with torch.no_grad():
    hidden = model(**enc).last_hidden_state           # (B, T, D)  [final layer]

# Mean-pool over residue tokens: attention mask minus <cls> (first) and <eos> (last).
mask = enc["attention_mask"].clone()
mask[:, 0] = 0
mask[torch.arange(len(SEQS)), enc["attention_mask"].sum(1) - 1] = 0
mask = mask.unsqueeze(-1).float()
emb = (hidden * mask).sum(1) / mask.sum(1)            # (B, D)

print(f"{len(SEQS)} sequences -> {tuple(emb.shape)}, "
      f"finite={torch.isfinite(emb).all().item()}")
