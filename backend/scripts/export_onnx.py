"""Export a causal language model to ONNX for fully in-browser inference.

A standard ONNX export gives you next-token logits, but the visualizer also needs
two things it does not provide:

  * per-layer attention weights  (shown as the links between words)
  * token embeddings             (used to colour the stars by k-means cluster)

This script wraps the model to expose all three as graph outputs, exports to fp32
ONNX with eager attention (so ``output_attentions`` returns real weights), then
dynamically quantises the weights to int8 to keep the file small and fast on the
WASM-CPU runtime used in the browser. The wrapper is architecture-agnostic, so
GPT-2, Llama/SmolLM2, Qwen and similar models all export the same way.

Run from the ``backend`` directory, after installing the requirements:

    python scripts/export_onnx.py                    # default: SmolLM2-135M
    EXPORT_MODEL=gpt2 python scripts/export_onnx.py  # or any HF causal LM
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
