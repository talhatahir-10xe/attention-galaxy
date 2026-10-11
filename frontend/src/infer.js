/**
 * In-browser inference — replaces the Python backend entirely.
 *
 * Loads a custom int8 ONNX export of SmolLM2-135M (which exposes logits, token
 * embeddings AND per-layer attention) and runs it with onnxruntime-web on the
 * WASM-CPU backend (reliable on laptops and phones alike). Tokenization uses the
 * model's own tokenizer via Transformers.js. The extraction loop produces the
 * same Section-8 JSON contract, so galaxy.js renders identically.
 */
import * as ort from "onnxruntime-web/wasm";
import { AutoTokenizer, env as tjsEnv } from "@huggingface/transformers";

// Single-threaded WASM CPU: no SharedArrayBuffer / COOP-COEP headers needed, so
// it works on GitHub Pages and mobile browsers out of the box.
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
// Load the ORT WASM runtime from a pinned CDN. onnxruntime-web dynamically
// `import()`s its wasm glue, which Vite's dev server refuses to serve from
// /public; an absolute CDN URL bypasses the bundler and the browser caches both
// files after first load (so it still works offline afterwards).
ort.env.wasm.wasmPaths =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

const MODEL_BASE = import.meta.env.BASE_URL + "model/";

// The 158 MB weights file is too big for GitHub's 100 MB limit, so in PRODUCTION
// it's fetched from the Hugging Face Hub (serves with CORS + a CDN). In DEV
// (`npm run dev`) we load it locally from public/model/ so development stays
// offline. A VITE_MODEL_URL build-time var overrides the default if you ever
// move the file. The small files — model.json and the tokenizer — always stay
// local alongside the site.
const MODEL_WEIGHTS_URL =
  import.meta.env.VITE_MODEL_URL ||
  (import.meta.env.PROD
    ? "https://huggingface.co/talhatahir01022001/attention-galaxy/resolve/main/smollm2-135m-int8.onnx"
    : null);

// Transformers.js: load the tokenizer from our bundled files only (no network).
tjsEnv.allowRemoteModels = false;
tjsEnv.allowLocalModels = true;
tjsEnv.localModelPath = MODEL_BASE;

// Generation config.
const CFG = {
  MAX_NEW_TOKENS: 24,
  MAX_NEW_TOKENS_CAP: 40,
  TOP_K: 5,
  MAX_TOKENS_TOTAL: 48,
  // Anti-repetition decoding — without this, greedy GPT-2 loops badly
  // ("a mouse with a mouse with a mouse"). These give coherent sentences while
  // staying deterministic (still greedy, just repetition-discouraged).
  REPETITION_PENALTY: 1.3,
  NO_REPEAT_NGRAM: 3,
};
const R = 1e4;
const round4 = (x) => Math.round(x * R) / R;

let _meta = null;
let _sessionPromise = null;
let _tokenizer = null;

const enc = (text) => _tokenizer.encode(text, { add_special_tokens: false });
const dec = (ids) => _tokenizer.decode(ids, { skip_special_tokens: false });

async function loadMeta() {
  if (!_meta) {
    const res = await fetch(MODEL_BASE + "model.json");
    if (!res.ok) throw new Error("model.json not found — run export_onnx.py");
    _meta = await res.json();
  }
  return _meta;
}

/** Fetch an ArrayBuffer while reporting download progress (0..1). */
async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch model (${res.status})`);
  const total = +res.headers.get("content-length") || 0;
  if (!res.body || !total) return await res.arrayBuffer();

  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress?.(received / total);
  }
  const out = new Uint8Array(received);
  let pos = 0;
  for (const c of chunks) {
    out.set(c, pos);
    pos += c.length;
  }
  return out.buffer;
}

/** Download + instantiate the ONNX session and tokenizer once (cached). */
export async function initModel(onProgress) {
  if (!_sessionPromise) {
    _sessionPromise = (async () => {
      const m = await loadMeta();
      const weightsUrl = MODEL_WEIGHTS_URL || MODEL_BASE + m.file;
      const [tokenizer, buf] = await Promise.all([
        AutoTokenizer.from_pretrained(m.tokenizer),
        fetchWithProgress(weightsUrl, onProgress),
      ]);
      _tokenizer = tokenizer;
      return ort.InferenceSession.create(buf, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
    })();
  }
  return _sessionPromise;
}

export async function getMeta() {
  return loadMeta();
}

// -------------------------------------------------------------- math helpers
function softmaxInPlace(arr) {
  let max = -Infinity;
  for (const v of arr) if (v > max) max = v;
  let sum = 0;
  for (let i = 0; i < arr.length; i++) {
    arr[i] = Math.exp(arr[i] - max);
    sum += arr[i];
  }
  const inv = 1 / sum;
  for (let i = 0; i < arr.length; i++) arr[i] *= inv;
  return arr;
}

/**
 * In-place repetition penalty (HuggingFace convention): tokens already present
 * in the sequence get their logit divided by the penalty (if positive) or
 * multiplied (if negative), discouraging the model from repeating them.
 */
function applyRepetitionPenalty(logits, ids, penalty) {
  if (penalty === 1) return;
  const seen = new Set(ids);
  for (const id of seen) {
    const v = logits[id];
    logits[id] = v > 0 ? v / penalty : v * penalty;
  }
}

/**
 * Block tokens that would complete an n-gram already seen in the sequence
 * (no_repeat_ngram_size) by forcing their logit to -Infinity.
 */
function banRepeatNgrams(logits, ids, n) {
  if (n <= 0 || ids.length < n) return;
  const prefix = ids.slice(ids.length - (n - 1)).join(",");
  for (let i = 0; i + n <= ids.length; i++) {
    const key = ids.slice(i, i + n - 1).join(",");
    if (key === prefix) logits[ids[i + n - 1]] = -Infinity;
  }
}

function topK(probs, k) {
  // indices of the k largest probabilities, descending
  const idx = Array.from({ length: probs.length }, (_, i) => i);
  // partial selection is overkill; a full sort of 50k is fine and simple
  idx.sort((a, b) => probs[b] - probs[a]);
  return idx.slice(0, k);
}

/** Tiny deterministic k-means for star colours (k clusters over emb vectors). */
function kmeans(vectors, k, iters = 12) {
  const n = vectors.length;
  const dim = vectors[0].length;
  k = Math.max(1, Math.min(k, n));
  // deterministic spread init: evenly spaced points
  const centroids = [];
  for (let c = 0; c < k; c++) {
    centroids.push(Float32Array.from(vectors[Math.floor((c * n) / k)]));
  }
  const assign = new Array(n).fill(0);
  for (let it = 0; it < iters; it++) {
    let moved = false;
    for (let i = 0; i < n; i++) {
      let best = 0;
      let bestD = Infinity;
      for (let c = 0; c < k; c++) {
        let d = 0;
        const ct = centroids[c];
        const v = vectors[i];
        for (let j = 0; j < dim; j++) {
          const diff = v[j] - ct[j];
          d += diff * diff;
        }
        if (d < bestD) {
          bestD = d;
          best = c;
        }
      }
      if (assign[i] !== best) {
        assign[i] = best;
        moved = true;
      }
    }
    // recompute centroids
    const sums = Array.from({ length: k }, () => new Float64Array(dim));
    const counts = new Array(k).fill(0);
    for (let i = 0; i < n; i++) {
      counts[assign[i]]++;
      const s = sums[assign[i]];
      const v = vectors[i];
      for (let j = 0; j < dim; j++) s[j] += v[j];
    }
    for (let c = 0; c < k; c++) {
      if (!counts[c]) continue;
      for (let j = 0; j < dim; j++) centroids[c][j] = sums[c][j] / counts[c];
    }
    if (!moved && it > 0) break;
  }
  return { k, assign };
}

// -------------------------------------------------------------- inference
async function runForward(session, ids) {
  const input = new ort.Tensor(
    "int64",
    BigInt64Array.from(ids.map((x) => BigInt(x))),
    [1, ids.length]
  );
  return session.run({ input_ids: input });
}

/**
 * Generate from a prompt entirely in the browser.
 * @returns {Promise<object>} Section-8 contract object
 */
export async function generate(prompt, opts = {}) {
  const meta = await loadMeta();
  const session = await initModel(opts.onProgress);
  const numLayers = meta.numLayers;
  const EOS_ID = meta.eos ?? 0;

  let maxNew = opts.maxNewTokens ?? CFG.MAX_NEW_TOKENS;
  maxNew = Math.max(1, Math.min(maxNew, CFG.MAX_NEW_TOKENS_CAP));

  let promptIds = Array.from(enc(prompt));
  if (!promptIds.length) promptIds = [EOS_ID];
  const room = Math.max(1, CFG.MAX_TOKENS_TOTAL - maxNew);
  if (promptIds.length > room) promptIds = promptIds.slice(-room);
  const promptLen = promptIds.length;

  const ids = [...promptIds];
  const steps = [];
  const genIds = [];

  for (let step = 0; step < maxNew; step++) {
    const out = await runForward(session, ids);
    const logits = out.logits.data; // Float32Array [1, seq, vocab]
    const vocab = out.logits.dims[out.logits.dims.length - 1];
    const seq = ids.length;
    const row = Float32Array.prototype.slice.call(
      logits,
      (seq - 1) * vocab,
      seq * vocab
    );
    // Discourage repetition before picking (keeps output coherent, not looping).
    applyRepetitionPenalty(row, ids, CFG.REPETITION_PENALTY);
    banRepeatNgrams(row, ids, CFG.NO_REPEAT_NGRAM);
    const probs = softmaxInPlace(row);

    // entropy over the full distribution
    let entropy = 0;
    for (let i = 0; i < probs.length; i++) {
      const p = probs[i];
      if (p > 0) entropy -= p * Math.log(p);
    }

    const top = topK(probs, Math.min(CFG.TOP_K, vocab));
    const winnerId = top[0];
    const candidates = top.map((id) => ({
      text: dec([id]),
      prob: round4(probs[id]),
    }));

    // stop at end-of-text once we have output
    if (winnerId === EOS_ID && genIds.length) break;

    steps.push({
      generatedIndex: ids.length,
      token: dec([winnerId]),
      chosenProb: round4(probs[winnerId]),
      entropy: round4(entropy),
      candidates,
    });
    ids.push(winnerId);
    genIds.push(winnerId);

    // stop at first sentence end (after a few words)
    const tokTxt = steps[steps.length - 1].token.replace(/\s+$/, "");
    const last = tokTxt.slice(-1);
    if (genIds.length >= 6 && (last === "." || last === "!" || last === "?")) break;

    // degenerate-loop guards
    const g = genIds;
    if (g.length >= 3 && g[g.length - 1] === g[g.length - 2] && g[g.length - 2] === g[g.length - 3]) break;
    if (
      g.length >= 4 &&
      g[g.length - 1] === g[g.length - 3] &&
      g[g.length - 2] === g[g.length - 4]
    )
      break;
  }

  const seq = ids.length;

  // Final full forward for attention + embeddings.
  const finalOut = await runForward(session, ids);
  const attn = finalOut.attn.data; // Float32Array [L, heads, seq, seq]
  const heads = finalOut.attn.dims[1];
  const emb = finalOut.emb.data; // Float32Array [1, seq, hidden]
  const hidden = finalOut.emb.dims[2];

  // byLayer[l][i][j] = mean over heads, causal, row-renormalised
  const byLayer = [];
  for (let l = 0; l < numLayers; l++) {
    const mat = [];
    for (let i = 0; i < seq; i++) {
      const rowArr = new Array(seq).fill(0);
      let sum = 0;
      for (let j = 0; j <= i; j++) {
        let acc = 0;
        for (let h = 0; h < heads; h++) {
          acc += attn[((l * heads + h) * seq + i) * seq + j];
        }
        const v = acc / heads;
        rowArr[j] = v;
        sum += v;
      }
      if (sum > 0) for (let j = 0; j <= i; j++) rowArr[j] /= sum;
      for (let j = 0; j < seq; j++) rowArr[j] = round4(rowArr[j]);
      mat.push(rowArr);
    }
    byLayer.push(mat);
  }

  // clusters (colours) via k-means over token embeddings
  const vectors = [];
  for (let i = 0; i < seq; i++) {
    vectors.push(emb.subarray(i * hidden, (i + 1) * hidden));
  }
  const clusters = kmeans(vectors, 3);

  const tokens = ids.map((id, i) => ({
    index: i,
    text: dec([id]),
    isPrompt: i < promptLen,
  }));

  return {
    prompt,
    model: meta.model,
    numLayers,
    defaultLayer: meta.defaultLayer,
    tokens,
    clusters,
    attention: { byLayer },
    steps,
  };
}
