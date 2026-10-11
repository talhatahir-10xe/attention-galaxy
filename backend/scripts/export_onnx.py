"""Export a GPT-2-family model to ONNX for fully in-browser inference.

The browser app needs three things the standard ONNX export does NOT provide:
  * next-token logits        (candidates / probabilities / entropy)
  * per-layer attention       (mean over heads is done later, in JS)
  * token embeddings          (k-means colouring of the stars, done in JS)

So we wrap the model and expose all three as graph outputs, export to fp32 ONNX
(eager attention, so `output_attentions` is real), then dynamically quantise to
int8 — small (~85MB) and reliable on both WASM-CPU and WebGPU, which is what we
want for phones.

Run from the `backend` dir (after `pip install -r requirements.txt onnx onnxruntime`):

    python scripts/export_onnx.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.abspath(
    os.path.join(BACKEND_DIR, "..", "frontend", "public", "model")
)
MODEL_NAME = os.environ.get("EXPORT_MODEL", "HuggingFaceTB/SmolLM2-135M")
SAFE_NAME = MODEL_NAME.split("/")[-1].lower()  # filesystem-safe basename


class ExportWrapper(nn.Module):
    """Returns (logits, token_embeddings, stacked_attentions)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor):
        out = self.model(
            input_ids=input_ids,
            output_attentions=True,
            use_cache=False,
        )
        # architecture-agnostic token embeddings (GPT-2 wte, Llama embed_tokens…)
        emb = self.model.get_input_embeddings()(input_ids)  # [1, seq, hidden]
        # tuple[num_layers] of [1, heads, seq, seq] -> [num_layers, heads, seq, seq]
        attn = torch.stack(out.attentions, dim=0)[:, 0]
        return out.logits, emb, attn


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Loading '{MODEL_NAME}' (eager attention)…")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, attn_implementation="eager"
    )
    model.eval()
    wrapper = ExportWrapper(model)

    num_layers = model.config.num_hidden_layers
    print(f"  layers={num_layers}  heads={model.config.num_attention_heads}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Example input (seq=5) — dynamic axes make the real length flexible.
    example = torch.tensor([tokenizer.encode("The cat sat on the")[:5]], dtype=torch.long)

    fp32_path = os.path.join(OUT_DIR, f"{SAFE_NAME}.onnx")
    print(f"Exporting fp32 ONNX → {fp32_path}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (example,),
            fp32_path,
            input_names=["input_ids"],
            output_names=["logits", "emb", "attn"],
            dynamic_axes={
                "input_ids": {1: "seq"},
                "logits": {1: "seq"},
                "emb": {1: "seq"},
                "attn": {2: "seq", 3: "seq"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,  # stable TorchScript exporter (predictable dynamic_axes)
        )

    # Dynamic int8 quantisation (weights → int8; activations computed in float).
    from onnxruntime.quantization import quantize_dynamic, QuantType

    int8_path = os.path.join(OUT_DIR, f"{SAFE_NAME}-int8.onnx")
    print(f"Quantising → {int8_path}")
    quantize_dynamic(
        fp32_path,
        int8_path,
        weight_type=QuantType.QInt8,
    )

    def mb(path):
        return os.path.getsize(path) / 1e6

    print(f"  fp32: {mb(fp32_path):.0f} MB   int8: {mb(int8_path):.0f} MB")

    # Bundle the tokenizer for the browser (Transformers.js loads these files).
    tok_dir = os.path.join(OUT_DIR, "tokenizer")
    os.makedirs(tok_dir, exist_ok=True)
    tokenizer.save_pretrained(tok_dir)
    # tokenizer.json is self-contained; drop the redundant slow-tokenizer files.
    for redundant in ("vocab.json", "merges.txt"):
        p = os.path.join(tok_dir, redundant)
        if os.path.exists(p):
            os.remove(p)
    print(f"Wrote tokenizer → {tok_dir}")

    # Write a tiny meta file the frontend reads.
    import json

    meta = {
        "model": MODEL_NAME,
        "numLayers": num_layers,
        "defaultLayer": num_layers // 2,
        "file": f"{SAFE_NAME}-int8.onnx",
        "hidden": model.config.hidden_size,
        "heads": model.config.num_attention_heads,
        "eos": tokenizer.eos_token_id,
        "tokenizer": "tokenizer",
    }
    with open(os.path.join(OUT_DIR, "model.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    # Remove the large fp32 file so only the int8 model ships.
    os.remove(fp32_path)
    print(f"Done. Wrote {int8_path}, tokenizer/, and model.json")


if __name__ == "__main__":
    main()
