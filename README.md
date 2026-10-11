<div align="center">

# ✦ Attention Galaxy

**See a language model think — one word at a time, entirely in your browser.**

A small transformer continues your prompt word by word. As it writes, the left
panel shows *how* it chooses each word (the candidate probabilities) and the
right-hand galaxy shows *why* — every word becomes a star, wired to the earlier
words it paid attention to, each link labelled with a readable percentage.

[**▶ Live demo**](https://talhatahir-10xe.github.io/attention-galaxy/)

No backend · no install · the model runs 100% client-side

</div>

---

## Highlights

- **Runs in the browser.** Inference happens on your device via
  [`onnxruntime-web`](https://onnxruntime.ai/docs/tutorials/web/) (WASM-CPU) —
  nothing you type is sent anywhere.
- **Real attention, not a mock-up.** A custom ONNX export surfaces the model's
  per-layer attention weights, which are rendered faithfully as links between words.
- **Watch it reason step by step.** Each word plays out the transformer's actual
  mechanic: *ask → look back → predict → lock → fly.*
- **Explore interactively.** Click any word to see exactly where its attention
  went; scrub the layer slider to compare early (grammar), middle (meaning) and
  late (summary) layers.
- **Free to host.** A ~3 MB static site on GitHub Pages; the model is served from
  the Hugging Face Hub.

---

## How it works

```
Browser only
  Transformers.js tokenizer ──▶ onnxruntime-web  (SmolLM2-135M · int8 · WASM-CPU)
                                   │ logits → candidate words + probabilities
                                   │ attn   → per-layer attention (averaged heads)
                                   │ emb    → k-means star colours
                                   ▼
                              Canvas 2D galaxy + prediction panel
```

| Model | Runtime | Tokenizer | Rendering |
|-------|---------|-----------|-----------|
| [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) (2024, 135M params, 30 layers), int8-quantized ONNX (~158 MB) | `onnxruntime-web` on WASM-CPU — works on laptops and phones, no special headers | the model's own tokenizer via [Transformers.js](https://huggingface.co/docs/transformers.js) | a single HTML5 Canvas with one `requestAnimationFrame` loop |

Decoding is greedy with a repetition penalty and no-repeat-ngram constraint, so
the output stays coherent rather than looping. The model is downloaded once (on
an explicit click) and cached by the browser thereafter.

### Why a custom ONNX export?

Standard ONNX exports don't expose attention weights — which is the one thing
this app is built around. `backend/scripts/export_onnx.py` wraps the model so it
outputs **logits + token embeddings + per-layer attention** (using eager
attention so the weights are real), exports to ONNX, quantizes to int8, and
bundles the tokenizer. The wrapper is architecture-agnostic, so GPT-2, Llama /
SmolLM2, Qwen and others all export the same way. You only need this to rebuild
or swap the model — the deployed app already includes everything it needs.

---

## Run locally

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

The galaxy starts empty. Click **Download model**, then type a prompt and press
**Generate**. In development the weights load from `frontend/public/model/`, so
no internet is needed after `npm install`.

### Build a static site

```bash
cd frontend
npm run build        # outputs ./dist
npm run preview
```

### Rebuild or swap the model (optional)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/export_onnx.py                    # default: SmolLM2-135M
EXPORT_MODEL=gpt2 python scripts/export_onnx.py  # or any HF causal LM
```

This writes `*-int8.onnx` + `tokenizer/` + `model.json` into
`frontend/public/model/`.

---

## Deployment

The site deploys to **GitHub Pages** for free; because the model exceeds GitHub's
100 MB file limit, it's hosted separately on the **Hugging Face Hub** (CDN-backed,
CORS-enabled). A GitHub Actions workflow builds and publishes on every push to
`main`. Full walkthrough in **[DEPLOY.md](DEPLOY.md)**.

---

## Project structure

```
attention-galaxy/
├── frontend/                   # the app (static, no runtime server)
│   ├── index.html
│   ├── src/
│   │   ├── main.js             # UI wiring, model download, controls
│   │   ├── infer.js            # in-browser generation (onnxruntime-web)
│   │   ├── galaxy.js           # canvas render loop, physics, links, panels
│   │   └── style.css
│   └── public/model/           # model.json + tokenizer (weights hosted on HF Hub)
├── backend/                    # build-time only: produces the ONNX model
│   ├── requirements.txt
│   └── scripts/export_onnx.py
└── .github/workflows/deploy.yml
```

---

## License

[MIT](LICENSE)
