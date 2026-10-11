# Attention Galaxy — Progress & Context

> Working log for context preservation. For end-user docs see `README.md`; for the
> original brief see `SPEC.md`. This file captures **what was built, the key
> decisions and *why*, gotchas learned, and what's next** — so the conversation
> can be compacted without losing context.

_Last updated: 2026-10-11_

---

## 1. What this is

A browser app that visualizes a small language model "thinking": it continues a
user's prompt one word at a time. Left panel = **how** it picks each next word
(candidate probabilities). Right "galaxy" = **why** (each word is a star wired to
earlier words by attention, every relationship shown as a %). Core animation per
word: **ask → gather → predict → lock → fly** (SPEC.md §3).

It started as backend (Python/FastAPI/GPT-2) + frontend (vanilla JS canvas). It
is **now a frontend-only app** — the model runs entirely in the browser.

---

## 2. Current architecture (frontend-only, no server at runtime)

```
Browser only:
  Transformers.js tokenizer  ─▶  onnxruntime-web (SmolLM2-135M int8, WASM-CPU)
                                   │ logits → candidates / probs / entropy
                                   │ attn   → per-layer attention (mean heads)
                                   │ emb    → k-means star colours
                                   ▼
                              Section-8 JSON  ─▶  Canvas 2D galaxy + panels
```

- **Model:** `HuggingFaceTB/SmolLM2-135M` (2024, **Llama architecture**, 30
  layers, 9 attn heads w/ GQA=3 kv heads, hidden 576, vocab 49152). int8
  dynamic-quantized ONNX, **~165 MB**, EOS token id = 0.
- **Runtime:** `onnxruntime-web/wasm` (CPU, single-threaded → no COOP/COEP
  headers needed → works on GitHub Pages / mobile). WebGPU intentionally not
  used (unreliable on mobile).
- **Tokenizer:** `@huggingface/transformers` (Transformers.js) `AutoTokenizer`,
  loading bundled files from `public/model/tokenizer/`. Pure-JS, no WASM.
- **No backend at runtime.** `backend/` is a **build-time tool only** (produces
  the ONNX model + tokenizer).

### Data contract (SPEC.md §8) produced by `infer.js`
`{ prompt, model, numLayers, defaultLayer, tokens[], clusters{k,assign}, attention{byLayer[L][i][j]}, steps[] }`
- `embedding2d` from the original spec is **not used** (layout is a grid, see §5).
  Only `clusters.assign` is used (for star colours).
- `attention.byLayer[l][i][j]` = mean over heads, causal (0 for j>i),
  row-renormalised to sum to 1. 4-dp rounded.

---

## 3. Files & their roles

```
frontend/
  index.html            top bar (prompt+Generate), canvas, caption, controls (layer slider)
  vite.config.js        base:"./", optimizeDeps.exclude:["onnxruntime-web"], chunkSizeWarningLimit
  src/
    main.js             wiring: model download+progress, Generate/Replay, caption token spans
    infer.js            in-browser inference: tokenizer load, generation loop, extraction → contract
    galaxy.js           the whole renderer: canvas, physics, links, particles, panels, interaction
    style.css           dark theme; [hidden]{display:none!important} (important, see gotchas)
  public/model/
    smollm2-135m-int8.onnx   the model (~165MB)
    model.json               {model,numLayers,defaultLayer,file,hidden,heads,eos,tokenizer}
    tokenizer/               tokenizer.json + tokenizer_config.json + special_tokens_map.json
backend/                 BUILD-TIME ONLY
  requirements.txt       torch, transformers==4.44.2, onnx, onnxruntime, onnxscript
  scripts/export_onnx.py arch-agnostic exporter (logits+emb+attn), int8 quant, bundles tokenizer
```

---

## 4. How to run / rebuild

**Run the app (no backend):**
```bash
cd frontend && npm install && npm run dev      # http://localhost:5173
# or static: npm run build && npm run preview
```
Empty galaxy on load → model downloads (progress in status line) → type a prompt → Generate.

**Rebuild / swap the model:**
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/export_onnx.py                  # default: HuggingFaceTB/SmolLM2-135M
EXPORT_MODEL=gpt2 python scripts/export_onnx.py  # swap to any HF causal LM
```
Writes `frontend/public/model/<name>-int8.onnx` + `tokenizer/` + `model.json`.

**Environment notes (this machine):**
- Node.js installed (no sudo) to `~/.local/node`, symlinked into `~/.local/bin`
  (already on PATH). Python 3.12, venv at `backend/.venv`.
- The user's HF cache already had gpt2/distilgpt2/SmolLM2/Qwen models.

---

## 5. Key decisions & why (IMPORTANT — don't re-litigate)

- **Frontend-only / in-browser model.** User requirement. Chose a **custom ONNX
  export** because standard exports (Transformers.js, optimum) **don't expose
  attention weights**, which the whole app needs. Wrapper outputs
  `logits + token-embeddings + per-layer attention` using **eager attention**
  (so `output_attentions` is real).
- **SmolLM2-135M** over gpt2/distilgpt2. distilgpt2 and gpt2 (2019) produced
  incoherent/looping output even with good decoding; SmolLM2 (2024) gives
  genuinely coherent sentences at the same ~size. Validated in Python before
  committing.
- **int8 (not fp16).** Reliable on WASM-CPU (fp16 needs WebGPU). Quantization
  didn't hurt output quality (verified). Attention stays accurate (softmax in
  float).
- **Anti-repetition decoding is essential.** Greedy alone loops badly on small
  models. Using **repetition_penalty=1.3 + no_repeat_ngram_size=3** + stop at
  first sentence end (after ≥6 tokens) + EOS + loop guards. Applied to logits
  *before* softmax so the shown candidate probs and the winner are consistent.
  Deterministic (still greedy).
- **ORT WASM loaded from pinned CDN** (`onnxruntime-web@1.20.1`): self-hosting in
  `/public/ort` broke because Vite's dev server refuses to serve a `public/` file
  as a dynamically `import()`ed module (ORT imports its `.mjs` glue). Absolute
  CDN URLs bypass the bundler; browser caches them (offline after 1st load).
- **Star layout = blue-noise jittered grid + deterministic shuffle** (NOT
  embedding positions). Duplicate words have identical embeddings → they stacked
  in a line. Grid gives every word a well-separated slot; shuffle scatters
  duplicates. Physics (`galaxy.js _recomputeHomes` + `_physics`) is spring-to-slot
  + gentle repulsion + faint attention tug. Layout fixed to **middle layer** so
  the layer slider never makes stars jump.
- **transformers==4.44.2 for export** (not 5.x): transformers 5.x GPT-2 masking
  uses `aten::diff`, unsupported by the TorchScript ONNX exporter. 4.44 exports
  GPT-2 **and** Llama/SmolLM2 cleanly. Export uses `dynamo=False`.
- **Removed on later request:** nerd-mode checkbox (layer slider now always
  visible), bundled examples + auto-run-on-load (now only generates on user
  input), the entire Python runtime backend (FastAPI `main.py`, `hf_backend.py`,
  `interface.py`, `reduce.py`, `make_examples.py`, `api.js`, `gpt-tokenizer`).

---

## 6. Notable UI/UX behaviors implemented

- Attention shown numerically only where focused: **gather beams** (active word)
  and **clicked star** (spotlight). The focused attention scene (bright beams, %
  labels, packets, dimmed rest) **persists through the whole prediction step**
  (gather→predict→lock→fly), not just a 1s flash.
- **Click a star → the matching word in the sentence is highlighted** (gold pill)
  so duplicate words (which "the"?) are disambiguated. Caption words are also
  clickable to select their star. (`main.js renderCaptionTokens`,
  `galaxy.js selectIndex` + `onSelect`.)
- Link contrast exaggerated (gamma `w^0.7`), bigger stars/labels with dark pills,
  bigger % pills, more/bigger particles on strong links.
- Layer slider spans all 30 layers, default middle (15).

### Tuning knobs (constants)
- `galaxy.js`: `PHASE` (ms per sub-step), `PHYS` (k_home/k_rep/k_att/damping/cap),
  `AMBIENT_TOPK`, `AMBIENT_MIN`, `W_GAMMA`, `PALETTE`, `PANEL_FRAC`.
- `infer.js`: `CFG` (MAX_NEW_TOKENS=24, cap 40, TOP_K=5, MAX_TOKENS_TOTAL=48,
  REPETITION_PENALTY=1.3, NO_REPEAT_NGRAM=3).

---

## 7. Gotchas learned (avoid repeating)

- **`[hidden]{display:none!important}`** is required in `style.css` — id-level
  `display:flex` rules out-specific the browser default and the attribute stops
  working (once caused a permanent "Thinking…" overlay).
- **Caption double-word bug:** `_updateCaption` must not re-append the just-landed
  winner (it's already counted in `landedCount`). Fixed.
- **Vite dev + `public/` modules:** don't `import()` anything from `public/`
  (ORT wasm glue). Use CDN or `?url`.
- **Transformers.js bundles its own onnxruntime-web** — fine, because tokenizer
  is pure-JS and never loads WASM; no conflict with our ORT session. Bundle is
  ~896 KB (gzip ~245 KB).
- The user cannot be assumed present for auto-notifications; verify in a browser.
  I cannot drive a browser from this environment — the user tests each change.

---

## 8. Status

- All original milestones M0–M9 were completed as the backend+frontend app, then
  the app was **converted to frontend-only in-browser inference** and iterated on
  model quality + UX per user feedback.
- **Confirmed working in the user's browser** (Chromium, laptop): SmolLM2-135M
  generates coherent sentences; galaxy renders; layer slider, spotlight, caption
  highlight all functioning.

---

## 9. Next steps (not yet done)

1. **Mobile build / responsiveness** (the immediate next task): make the two-panel
   canvas layout + controls usable from ~380px width; test on a phone browser.
2. **Deployment (GitHub Pages + HF Hub): WIRED UP** (see `DEPLOY.md`). The 158 MB
   model exceeds GitHub's 100 MB limit, so it's git-ignored and hosted on the
   Hugging Face Hub (CORS + CDN); the site goes on GitHub Pages. `infer.js` reads
   `import.meta.env.VITE_MODEL_URL` (build-time var) for the weights URL, falling
   back to the local `public/model/` file when unset (so `npm run dev` is offline).
   `.github/workflows/deploy.yml` builds + deploys on push to `main`, baking in the
   `VITE_MODEL_URL` repo variable. Verified: a build without the `.onnx` present
   succeeds, excludes the model from `dist/`, and bakes the URL in. **User must
   still: create the HF repo + upload the model, push to GitHub, set the
   `VITE_MODEL_URL` repo variable, and enable Pages (Source: GitHub Actions).**
3. **Size/perf for mobile (optional):** evaluate smaller/4-bit quant, or lazy
   layer loading; measure first-gen latency on a phone.
4. **Hero GIF** for the README (placeholder comment is in README.md).

---

## 10. Quick validation recipes (offline, no browser)

- ONNX + decoding quality (Python):
  `backend/.venv/bin/python -I <script>` loading
  `frontend/public/model/smollm2-135m-int8.onnx` via `onnxruntime`, replicating
  the JS decoding (rep penalty + no-repeat-ngram + sentence stop). Used to confirm
  quality before shipping.
- Tokenizer parity (Node, from `frontend/`): `AutoTokenizer.from_pretrained("tokenizer")`
  with `env.localModelPath` → encode/decode must match Python exactly.
- `npm run build` must succeed; `node --check src/*.js` for syntax.
```
