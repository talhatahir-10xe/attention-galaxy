# Attention Galaxy — Build Specification

> A browser-based web app that shows a small language model **thinking**. It
> generates a sentence one word at a time; a left panel shows *how* it picks
> each next word (probabilities), and a right-hand "galaxy" shows *why* — each
> word becomes a star placed by meaning, wired to earlier words by attention,
> with every relationship shown as a readable **percentage**.

**Repo/folder name:** `attention-galaxy` (alt names: *Lexigalaxy*, *Token Nebula*).

---

## 0. How this document is meant to be used

This is a complete brief for an AI coding agent (Claude Code). The human will
place this file in an empty folder, open Claude Code there, and ask it to build
the app described here. There is **no pre-existing git repo**; create the folder
structure from scratch. Build in the **milestone order** in Section 11. Each
milestone has an explicit **acceptance criterion** — do not advance until it is
met. Favour clarity and correctness first; visual polish comes in a later phase.

---

## 1. What the finished app does (plain description)

- It is a **web application**. The user runs it locally: start a Python backend
  and a frontend dev server, open `http://localhost:5173` in a browser.
- The user types a short prompt (e.g. "The cat"). A small model (**GPT-2**)
  continues it a few words at a time.
- The screen is split into **two connected panels**:
  - **Left — Prediction panel:** shows the model choosing the next word as a
    list of candidate words with probability bars (percentages).
  - **Right — Galaxy:** each chosen word becomes a glowing labelled star,
    positioned by meaning (embeddings), wired to earlier words by attention.
- Before each word is chosen, the most recent word **looks back** at all earlier
  words and **gathers** information from them — shown as beams labelled with
  **attention percentages**. Then the winning word **flies** from the left panel
  into the galaxy and wires in its attention links.
- After the sentence finishes, the user can **click any star** to see that
  word's attention as a ranked percentage list (and the matching links light up;
  everything else dims).

### Two ways to run it (both required)

1. **Full local app (clone + run):** backend + frontend. The user types their
   own prompts; the backend runs GPT-2 and returns real data.
2. **Live demo (zero-install):** the frontend alone, hosted as a static site
   (e.g. GitHub Pages later), running on a few **pre-computed example sentences**
   bundled as JSON. In this mode the prompt box is **disabled**, showing the
   message: *"Running the live demo on sample sentences — clone and run locally
   to try your own prompt."* The example picker lets visitors cycle the bundled
   sentences.

The frontend consumes the **same JSON** in both modes, so rendering code is
identical.

---

## 2. UI layout

```
┌───────────────────────────────────────────────────────────────┐
│  [ prompt input ........................... ]  [ Generate ]     │   top bar
├────────────────────────┬──────────────────────────────────────┤
│  PREDICTION PANEL       │              GALAXY                   │
│  (~34% width)           │              (~66% width)            │
│                         │                                       │
│  predicting next word   │        ★ cat                          │
│  sat   ████████ 42%     │          \                            │
│  slept ███ 18%          │           ★ sat ──── ★ on             │
│  jumped██ 12%           │                       \               │
│  ran   █ 9%             │                        ★ the ── ★ mat │
│  meowed█ 7%             │     (soft nebula "meaning" zones)     │
│                         │                                       │
│  (after click, this     │   brighter/thicker link = more        │
│   panel shows the        │   attention; numbers shown for the   │
│   clicked word's         │   active beams / clicked word only)  │
│   attention list)        │                                       │
├────────────────────────┴──────────────────────────────────────┤
│   The cat sat on the mat▋            (sentence builds here)      │   caption
├───────────────────────────────────────────────────────────────┤
│   [Replay]   [ ] nerd mode   —  (nerd mode reveals: layer slider)│   controls
└───────────────────────────────────────────────────────────────┘
```

Must remain legible and usable from ~380px (mobile) up to desktop widths.

---

## 3. The core mechanic: look back → gather → predict

This is the heart of the app. For each generated word, animate these sub-steps
in order (suggested durations in ms; tune for feel):

1. **Ask** (~500): the most recent word (the "asker") pulses — it is the one
   about to predict the next word.
2. **Look back / gather** (~1300): beams light up from the asker to **every
   earlier word**, brightness/thickness = attention weight. Each beam is
   **labelled with its attention %** (the percentages across a word's beams add
   up to 100%). Small light packets flow **inward** (earlier word → asker) to
   show information being gathered.
3. **Predict** (~1000): the left panel's candidate bars grow to their
   probabilities (with % labels).
4. **Lock** (~600): the winning candidate highlights; the other four fade away.
5. **Fly** (~800): the winning word travels from its panel row across into the
   galaxy, lands at its meaning-position, and its attention links settle in. It
   becomes the next "asker" for the following word.

This is faithful **Level 1** interpretability: attention truthfully shows *which
earlier words the model used*. (We are **not** claiming per-candidate
attribution — that is a future "Level 2", Section 13.)

---

## 4. Attention as numbers (important for the dev)

Attention is literally a set of weights that, for one word at one layer, **sum
to 1 (100%)** across the earlier words. So it displays exactly like the
prediction bars. Rules:

- Show attention **numerically wherever the user's eye is focused**, never on all
  links at once (that is clutter):
  - On the **active beams** during the gather step.
  - On the **clicked word's** links, plus a ranked list in the left panel, in
    explore mode.
- Everywhere else, attention is shown only as link **brightness + thickness**.

### Which layer's attention? (resolve the "12 layers" question)
GPT-2 has 12 layers; attention differs per layer (early = local grammar, middle
= meaning, late = summary). The app:
- **Defaults to a middle layer** (the most human-readable), averaged over heads,
  and never mentions layers in the default UI.
- Exposes a **layer slider only inside "nerd mode"** (off by default). Sliding it
  changes the displayed attention numbers/links live; the star **layout does not
  jump** (layout is fixed to the middle-layer attention — see 9.2).

---

## 5. Feature list, grouped by build phase

### Phase 1 — Clear & correct (plain look, perfectly readable)
- Two-panel layout + top prompt bar + bottom caption.
- Token-by-token generation driving the left panel + right galaxy.
- Prediction panel: candidate words, probability bars, **% labels**, winner
  locks, losers vanish.
- Look-back → gather → predict mechanic with **% labels on beams**.
- Winner flies into galaxy and becomes a labelled star.
- Stars positioned by embedding (meaning); attention links as lines whose
  thickness = weight.
- Click a star → ranked **attention % list** in the left panel + % labels on its
  links; other stars/links dim. Click empty space → reset.
- Replay button.

### Phase 2 — Beautiful (same app, now mesmerising)
- **Gravity motion**: stars drift to position and get tugged by attention
  (physics force layout) instead of snapping.
- Glowing stars and links; eased animations.
- **Particle flow** along strong links (direction: attended word → attending
  word).
- Soft coloured **nebula zones** behind the meaning-clusters.

### Phase 3 — Opt-in depth ("nerd mode", hidden by default)
- **Layer slider** (1…12, default middle) that re-wires displayed attention live.
- (Stretch) attention-**head** selector.

---

## 6. Architecture

Two parts separated by one **JSON data contract** (Section 8). The frontend only
ever consumes the contract — from the live backend, or from bundled example
files. Identical rendering either way.

```
Backend (Python, GPT-2)  ──JSON (Section 8)──▶  Frontend (browser, canvas)
   POST /generate                                 renders panels + galaxy
   + make_examples.py  ──writes──▶  frontend/public/examples/*.json
```

---

## 7. Tech stack & model

| Layer | Choice | Notes |
|---|---|---|
| Backend | Python 3.11+, FastAPI, Uvicorn | one `/generate` endpoint |
| Model | **GPT-2 small (124M)** via HuggingFace `transformers` + PyTorch | **runs on CPU**; no GPU required |
| Attention/embeddings | `output_attentions=True`, `output_hidden_states=True` | exactly the data we need |
| Dim reduction | scikit-learn PCA(2) + k-means | 2D positions + meaning clusters |
| Frontend | Vanilla JS + HTML5 Canvas 2D | matches the demo; no framework |
| Dev/build | Vite | dev server + static build for the live demo |
| Live hosting (later) | GitHub Pages (static) | serves frontend + bundled examples |

- Default model `gpt2`. Expose `MODEL_NAME` env var; document `distilgpt2`
  (faster) and `Qwen/Qwen2.5-0.5B-Instruct` (modern, also CPU-friendly) as
  drop-in alternatives. Do not block v1 on them.
- Decoding: **greedy** by default (deterministic, reproducible demos). Optional
  `temperature`.
- Config (env/CLI): `MODEL_NAME=gpt2`, `MAX_NEW_TOKENS=8` (hard cap 20),
  `TOP_K=5` (candidates shown, incl. winner), `MAX_TOKENS_TOTAL=40`
  (truncate prompt from the left if exceeded).

---

## 8. Data contract (build both sides to this)

One JSON object per generation. Floats rounded to 4 dp. Attention is **causal**
(0 where `j > i`) and each displayed row is **re-normalised to sum to 1**.
Positions normalised to `[0,1]`.

```jsonc
{
  "prompt": "The cat",
  "model": "gpt2",
  "numLayers": 12,
  "defaultLayer": 6,                 // middle layer, 0-indexed; frontend default

  "tokens": [
    { "index": 0, "text": "The",  "isPrompt": true  },
    { "index": 1, "text": " cat", "isPrompt": true  },
    { "index": 2, "text": " sat", "isPrompt": false }
    // ...
  ],

  "embedding2d": [ [0.20,0.74], [0.34,0.30], [0.56,0.47] ],   // per token
  "clusters": { "k": 3, "assign": [2,0,1], "centroids": [[0.3,0.8],[0.5,0.45],[0.4,0.3]] },

  // Attention, averaged over heads, per layer: byLayer[layer][i][j]
  // = how much token i attends to token j (j <= i), causal, row-normalised.
  // Size ~ numLayers * seq * seq (small; seq <= MAX_TOKENS_TOTAL).
  "attention": {
    "byLayer": [ /* [numLayers][seq][seq] */ ]
  },

  // One entry per GENERATED token, in order.
  "steps": [
    {
      "generatedIndex": 2,
      "token": " sat",
      "chosenProb": 0.42,
      "entropy": 1.83,                         // optional; "surprise"
      "candidates": [                           // top-k incl. winner, sorted desc
        { "text": " sat",   "prob": 0.42 },
        { "text": " slept", "prob": 0.18 },
        { "text": " jumped","prob": 0.12 },
        { "text": " ran",   "prob": 0.09 },
        { "text": " meowed","prob": 0.07 }
      ]
    }
    // ...
  ]
}
```

**Rules**
- `attention.byLayer` dimensions must match `numLayers` and `seq =
  tokens.length`.
- Bundled examples in `frontend/public/examples/*.json` use this exact schema.
- `tokens[].text` keeps the tokenizer's leading spaces; frontend trims for
  display only.

> Note: per-head attention is intentionally **not** in v1 (byLayer = mean over
> heads keeps payloads small and matches the default UI). A per-head field can be
> added later for a head explorer.

---

## 9. Backend specification

### 9.1 Endpoints
- `POST /generate` — body `{ "prompt": string, "maxNewTokens"?: int,
  "temperature"?: float }` → the Section 8 object. Validate: non-empty prompt,
  `maxNewTokens <= 20`; on bad input 400 `{ "error": "..." }`.
- `GET /health` → `{ "status": "ok", "model": MODEL_NAME }`.
- CORS: allow the Vite dev origin (and `*` for the static demo build).

### 9.2 Extraction logic
1. Tokenize prompt. Run a manual autoregressive loop for `maxNewTokens` steps;
   at each step forward with `output_attentions`/`output_hidden_states`, softmax
   the next-token logits, record `chosenProb`, `entropy`, top-`TOP_K`
   `candidates`; greedily append the winner.
2. One **final full forward pass** over the complete sequence → build
   `attention.byLayer` (mean over heads per layer, causal, row-renormalised).
3. **embedding2d:** take the model's input token embeddings (`wte`) for the
   sequence tokens, PCA→2D, normalise to `[0,1]` (deterministic).
4. **clusters:** k-means (`k=3`) over `embedding2d`.
5. `defaultLayer = numLayers // 2`.
6. Assemble and return.

### 9.3 Keep the model behind one interface
```
class InferenceBackend:
    def generate(prompt, max_new_tokens, temperature) -> dict  # Section 8
```
HuggingFace implementation in `backend/hf_backend.py`; `main.py` depends only on
the interface. (Lets a different engine be swapped in later without touching the
API or frontend.)

### 9.4 Example generator
`backend/scripts/make_examples.py` runs `generate` on 3–4 curated prompts and
writes the resulting JSON to `frontend/public/examples/`. Commit these so the
static demo works with no backend.

---

## 10. Frontend specification

Single full-width `<canvas>` for both panels + one `requestAnimationFrame` loop;
plain state object, no framework. DOM is used only for the top bar, caption, and
the controls row.

### 10.1 Mode selection (`src/api.js`)
- If a backend base URL is configured/reachable → **live mode**: `POST
  /generate` with the typed prompt.
- Otherwise → **static mode**: load `public/examples/<name>.json`; **disable**
  the prompt input and show the clone-to-use-your-own message; show an example
  picker.

### 10.2 Galaxy — nodes & physics (Phase 2)
- `nodes[i] = { x, y, vx, vy, born, landed, homeX, homeY, color, cluster }`;
  `home` from `embedding2d` mapped into the galaxy region (right of the panel).
- Each frame, per landed node, sum forces: spring to home (`k_home`), repulsion
  from other nodes (`k_rep/d²`), attention attraction to attended nodes
  (`k_att * w`). Integrate with damping + speed cap. Suggested start:
  `k_home=0.08, k_rep=430, k_att=0.0006, damping=0.80, cap=9`.
- **Layout uses the middle-layer attention only** (so the nerd-mode slider
  changes displayed links/numbers but never makes stars jump).
- Node colour from `clusters.assign` via a fixed palette.

### 10.3 Attention links & numbers
- For the **currently displayed layer** (default middle; nerd-mode slider
  overrides display only), draw a link for each `(i,j), j<i` with weight `w`:
  `alpha = 0.05 + w*0.55`, `lineWidth = 0.4 + w*3.4`.
- **Numbers (`round(w*100)%`)** are drawn **only** for: active beams during the
  gather step, and the clicked word's links in explore mode. Never for the whole
  web at once.

### 10.4 Prediction panel
- During a step, render `steps[g].candidates` as labelled rows with bars filling
  to `prob` (eased) and `%` labels; winner highlights on Lock; losers fade.
- In explore mode with a selected word, the same panel renders that word's
  attention (current layer) as a ranked bars+% list.

### 10.5 Particles & nebula (Phase 2)
- Particles travel attended→attending on strong links (and only the focused
  word's links in spotlight). Small recycled pool.
- Nebula: soft radial glows at `clusters.centroids`, low alpha, cluster-coloured.

### 10.6 Interaction
- Click/tap a star → toggle spotlight on that word (dim the rest; glow the
  attended words; show numbers + ranked list). Click empty space → clear.
- **Replay** re-runs the generation animation on current data.
- **Nerd mode** checkbox reveals the **layer slider** (`1…numLayers`, default
  middle, with an "early / middle (clearest) / late" descriptor).

---

## 11. Milestones (build in this order)

### M0 — Scaffold
- [ ] Create the folder structure (Section 12), MIT `LICENSE`.
- [ ] Backend: FastAPI app with `/health`. Frontend: Vite app, blank full-width
      canvas + top bar + controls row (non-functional).
- **Accept:** `/health` returns ok; `npm run dev` shows the shell.

### M1 — Backend `/generate`
- [ ] `hf_backend.py`: load `gpt2`, manual loop capturing candidates/probs/
      entropy; final pass → `attention.byLayer`; PCA→`embedding2d`; k-means
      clusters; assemble Section 8 object.
- **Accept:** `curl -X POST localhost:8000/generate -d '{"prompt":"The cat"}'`
      returns valid JSON; dimensions consistent; attention rows ≈ sum to 1.

### M2 — Example bundle + static mode
- [ ] `make_examples.py` writes 3–4 examples to `frontend/public/examples/`.
- [ ] `api.js` static-mode: load example JSON, disable prompt box with the
      clone-to-use message, add an example picker.
- **Accept:** frontend runs fully offline from bundled JSON.

### M3 — Phase 1 core (plain, correct)
- [ ] Two-panel render; prompt tokens appear as plain labelled dots placed by
      `embedding2d`; links as plain lines (thickness=weight).
- [ ] Prediction panel with bars + % ; winner lock; losers vanish.
- [ ] Generation sequencing drives panel + galaxy; caption builds the sentence.
- **Accept:** on an example, the sentence generates step by step and reads
      clearly **without any glow/particles**.

### M4 — Look-back mechanic + numbers
- [ ] Asker pulse; gather beams with inward packets and **% labels**; then the
      winner **flies** from panel into the galaxy and wires links.
- **Accept:** a viewer can see which earlier words were used and by how much (%),
      for each generated word.

### M5 — Click-to-spotlight + ranked numbers
- [ ] Click a star → dim others, light its links with **% labels**, show ranked
      attention list in the panel. Click empty → reset.
- **Accept:** clicking any word shows a correct attention breakdown summing to
      ~100%.

### M6 — Nerd mode (layer slider)
- [ ] Checkbox reveals the layer slider; default middle; changing it updates
      displayed links/numbers/ranked-list live; layout stays put.
- **Accept:** sliding from early→late visibly changes attention; default is a
      middle layer; normal UI never shows the slider.

### M7 — Phase 2 beauty
- [ ] Gravity drift, glow, eased motion, particle flow, nebula zones.
- **Accept:** same behaviour as M3–M6 but visually polished; still 60 fps for
      sentences up to ~20 tokens.

### M8 — Live demo build + run docs
- [ ] Static build of the frontend that serves bundled examples (ready for
      GitHub Pages later). Document the exact local run commands.
- **Accept:** the built static site runs the examples with no backend; local
      full app runs user prompts.

### M9 — Polish & README
- [ ] README with a hero GIF placeholder, one-line pitch, the two run modes, and
      "how it works". Visual/tuning pass.
- **Accept:** a newcomer understands and runs it from the README alone.

> Performance (optional, later): the force layout + link draw is O(N²). If long
> sentences drop below 60 fps, move the force simulation to WebGL/WebGPU with a
> CPU fallback, and note the measured speedup in the README.

---

## 12. Folder structure

```
attention-galaxy/
├── README.md
├── SPEC.md                      # this file
├── LICENSE
├── backend/
│   ├── requirements.txt         # fastapi, uvicorn, torch, transformers, scikit-learn
│   ├── main.py                  # FastAPI routes + CORS
│   ├── interface.py             # InferenceBackend + contract types
│   ├── hf_backend.py            # GPT-2 extraction
│   ├── reduce.py                # PCA 2D + k-means
│   └── scripts/make_examples.py
└── frontend/
    ├── package.json
    ├── index.html
    ├── vite.config.js
    ├── public/examples/*.json
    └── src/
        ├── main.js              # wiring, controls, mode switch
        ├── api.js               # live fetch OR load example JSON
        ├── galaxy.js            # render loop, physics, links, particles, panels
        └── style.css
```

---

## 13. How to run (put this in the README too)

**Full local app**
```
# 1) Backend
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000               # first run downloads GPT-2 (~500MB)

# 2) Frontend (new terminal)
cd frontend
npm install
npm run dev                                         # open http://localhost:5173
```
Type a prompt → Generate. (CPU is fine; a short generation takes ~1–2s.)

**Live demo (no backend)**
```
cd frontend
npm run build && npm run preview                    # serves bundled examples only
```
Prompt box is disabled here with a note to clone and run locally for custom
prompts.

---

## 14. Future work (not in v1)
- **Level 2 attribution:** show which earlier words pushed toward which specific
  candidate (beyond attention — a heavier interpretability method). Add as an
  "advanced" overlay.
- **Head explorer:** per-head attention + a head selector.
- **3D galaxy** (three.js); **layer auto-sweep** animation; **share links**
  (prompt encoded in URL); **model picker** (`gpt2` / `distilgpt2` /
  `Qwen2.5-0.5B`).

## 15. Non-goals (v1)
Chat UX, large models, training, accounts, server-side saving, mobile-native app.

## 16. Definition of done (v1)
M0–M9 accepted; full local app generates user prompts on CPU; static live demo
runs bundled examples with the disabled-prompt note; README explains both modes;
smooth 60 fps up to ~20 tokens.
