# ✦ Attention Galaxy

> Watch a small language model **think** — entirely in your browser, no server.
> It writes a sentence one word at a time; the left panel shows *how* it picks
> each next word (probabilities), and the right-hand **galaxy** shows *why*: each
> word becomes a star, wired to earlier words by attention, every relationship
> shown as a readable **percentage**.

<!--
HERO GIF PLACEHOLDER
Record a ~10s screen capture of one generation (ask → gather → predict → lock →
fly) followed by a click-to-spotlight, save it as docs/hero.gif, and uncomment:

![Attention Galaxy demo](docs/hero.gif)
-->

---

## What it does

For every word the model generates, Attention Galaxy animates the real
mechanic behind a transformer:

1. **Ask** — the most recent word pulses; it's about to predict the next one.
2. **Look back → gather** — beams light up from that word to every earlier word,
   each labelled with its **attention %** (they add up to 100%). Light packets
   flow inward to show information being gathered. The scene is held until the
   next word is chosen.
3. **Predict** — the left panel's candidate bars grow to their probabilities.
4. **Lock** — the winner highlights; the losers fade.
5. **Fly** — the winning word flies into the galaxy, lands at its own slot, and
   wires in its attention links.

After the sentence finishes, **click any star** (or any word in the sentence) to
see that word's attention as a ranked percentage list — its links light up, the
matching word in the sentence is highlighted, and everything else dims.

This is faithful **Level 1** interpretability: attention truthfully shows *which
earlier words the model used*.

---

## Runs fully in the browser — no backend

The model (**SmolLM2-135M**, a modern 2024 small model — 135M params / 30 layers,
int8-quantized to ~165 MB) runs in your browser via
[`onnxruntime-web`](https://onnxruntime.ai/docs/tutorials/web/) on the WASM-CPU
backend — which works on laptops and phones alike. Tokenization uses the model's
own tokenizer via [Transformers.js](https://huggingface.co/docs/transformers.js).
The model downloads once and is cached by the browser after that. Decoding is
greedy with a repetition penalty + no-repeat-ngram, so output stays coherent
instead of looping.

```bash
cd frontend
npm install
npm run dev        # open http://localhost:5173
```

That's it — the galaxy starts empty; type a prompt and press **Generate**. The
model downloads on first load (progress shows in the status line), then runs
locally for anything you type. Nothing runs until you ask it to.

### Static build (deployable to any static host)

```bash
cd frontend
npm run build && npm run preview
```

`dist/` is a self-contained static site — serve it from anywhere.

### Free global hosting: GitHub Pages + Hugging Face Hub

The site is tiny (~3 MB) and goes on **GitHub Pages**; only the 158 MB model is
too big for GitHub, so it lives on the **Hugging Face Hub** (free, CDN-backed,
CORS-enabled — purpose-built for weights). No backend, no bills. See
[DEPLOY.md](DEPLOY.md) for the full walkthrough; in short:

1. Upload `smollm2-135m-int8.onnx` to a public HF model repo.
2. Push this repo to GitHub, enable Pages (Source: **GitHub Actions**).
3. Add a repo **variable** `VITE_MODEL_URL` = the HF `resolve/main/…onnx` URL.

The included workflow (`.github/workflows/deploy.yml`) builds and deploys on every
push. At build time it bakes `VITE_MODEL_URL` in; the browser fetches the weights
from the Hub and caches them after the first visit. Locally, `npm run dev` still
uses the on-disk file in `public/model/` — no env var needed.

---

## How it works

```
Browser only:
  Transformers.js tokenizer  ─▶  onnxruntime-web (SmolLM2-135M int8, WASM-CPU)
                                   │  logits  →  candidates / probabilities
                                   │  attn    →  per-layer attention (mean heads)
                                   │  emb     →  k-means star colours
                                   ▼
                              Section-8 JSON  ─▶  Canvas 2D galaxy + panels
```

- **`src/infer.js`** runs a greedy generation loop (capturing per-step candidate
  probabilities and entropy), then one final forward pass to read attention and
  token embeddings. It produces the same JSON contract (`SPEC.md` §8) the old
  backend did, so the renderer is unchanged.
- **`src/galaxy.js`** draws both panels on one canvas with a single rAF loop.
  Stars sit on a jittered grid (blue-noise) so every word — including duplicates
  — stays well separated and readable; a light physics model adds drift. The
  **layout is fixed to the middle layer**, so the layer slider re-wires the
  displayed attention without ever making stars jump.

### The custom ONNX model

Standard ONNX exports don't expose attention weights. So
`backend/scripts/export_onnx.py` wraps the model to output **logits + token
embeddings + per-layer attention** (using eager attention so `output_attentions`
is real), exports to ONNX, dynamically quantizes to int8, and bundles the
tokenizer. The result lands in `frontend/public/model/`. You only need this if
you want to rebuild or swap the model — the committed model works out of the box.
The wrapper is architecture-agnostic, so GPT-2, Llama/SmolLM2, Qwen, etc. all
export the same way.

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/export_onnx.py          # → frontend/public/model/*-int8.onnx + tokenizer/ + model.json
```

Default export is `HuggingFaceTB/SmolLM2-135M`; swap with `EXPORT_MODEL=gpt2` (or
any HF causal LM) before running the script. The script bundles that model's
tokenizer too, so the frontend stays in sync.

### Which layer's attention?

Transformer attention differs per layer (early = local grammar, middle = meaning,
late = summary). The UI defaults to a **middle layer**, averaged over heads. The
**layer slider** in the controls row re-wires the displayed attention live
(early / middle / late); the star layout stays fixed so nothing jumps.

---

## Project layout

```
attention-galaxy/
├── frontend/                 # the entire app (no server needed at runtime)
│   ├── index.html
│   ├── public/
│   │   └── model/            # smollm2-135m-int8.onnx + tokenizer/ + model.json
│   │   # (onnxruntime-web WASM loads from a pinned CDN; cached after 1st load)
│   └── src/
│       ├── main.js           # wiring, controls, model download/progress
│       ├── infer.js          # in-browser generation (onnxruntime-web)
│       ├── galaxy.js         # render loop, physics, links, particles, panels
│       └── style.css
└── backend/                  # BUILD-TIME ONLY: produces the ONNX model
    ├── requirements.txt
    └── scripts/export_onnx.py
```

---

## License

MIT — see [LICENSE](LICENSE).
