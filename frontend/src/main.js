/**
 * Wiring: DOM controls, the Galaxy renderer, and in-browser inference.
 *
 * There is no backend — the model runs in the browser (infer.js). The model is
 * downloaded only when the user asks for it (the first-run gate), and the galaxy
 * stays empty until the user presses Generate. While the model is predicting,
 * the Generate button becomes a Stop button that settles the view immediately.
 */
import "./style.css";
import { Galaxy } from "./galaxy.js";
import { generate, initModel, getMeta } from "./infer.js";

const $ = (id) => document.getElementById(id);

const els = {
  prompt: $("prompt"),
  generate: $("generate"),
  caption: $("caption-text"),
  cursor: $("cursor"),
  replay: $("replay"),
  layer: $("layer"),
  layerValue: $("layer-value"),
  layerDescriptor: $("layer-descriptor"),
  hideSink: $("hide-sink"),
  avgLayers: $("avg-layers"),
  status: $("status"),
  loading: $("loading"),
  loadingText: $("loading-text"),
  // first-run download gate
  modelGate: $("model-gate"),
  gateTitle: $("gate-title"),
  gateDesc: $("gate-desc"),
  downloadBtn: $("download-model"),
  gateProgress: $("gate-progress"),
  gateBarFill: $("gate-bar-fill"),
  gatePct: $("gate-pct"),
  gateError: $("gate-error"),
};

const state = {
  modelReady: false,
  generating: false, // true from the moment Generate is pressed until settled
  skipRequested: false, // user asked to skip the slow animation → final view
  data: null,
  modelName: "the model",
};

const galaxy = new Galaxy($("canvas"), {
  onCaption: (text) => {
    // Plain text while the sentence is still building.
    els.caption.textContent = text;
  },
  onStatus: (text) => {
    els.status.textContent = text;
  },
  onRunStateChange: (running) => {
    // Called by the renderer when the word-by-word animation starts (true) or
    // ends — whether it finished on its own or was stopped (false).
    if (!running) {
      setGenerating(false);
      if (state.data) renderCaptionTokens(null);
    }
  },
  // A star was selected (or cleared) — re-render the caption as clickable
  // tokens and highlight the exact word that was selected.
  onSelect: (index) => renderCaptionTokens(index),
});

/**
 * Render the finished sentence in the caption as per-token spans, highlighting
 * the selected token so you can see exactly which word (e.g. which "the") maps
 * to the selected star. Tokens are clickable to select their star in return.
 */
function renderCaptionTokens(highlightIndex = null) {
  const toks = state.data?.tokens || [];
  const frag = document.createDocumentFragment();
  let activeSpan = null;
  toks.forEach((t, i) => {
    const raw = t.text.replace(/\n/g, " ");
    const m = raw.match(/^(\s*)([\s\S]*)$/);
    const lead = i === 0 ? "" : m[1]; // drop any leading space on the first word
    const word = m[2];
    if (lead) frag.appendChild(document.createTextNode(lead));
    const span = document.createElement("span");
    span.className = "tok";
    span.textContent = word;
    span.dataset.index = String(t.index);
    if (t.index === highlightIndex) {
      span.classList.add("tok-active");
      activeSpan = span;
    }
    frag.appendChild(span);
  });
  els.caption.replaceChildren(frag);
  if (activeSpan) activeSpan.scrollIntoView({ block: "nearest", inline: "center" });
}

// Click a word in the sentence → select its star.
els.caption.addEventListener("click", (e) => {
  const span = e.target.closest(".tok");
  if (!span) return;
  galaxy.selectIndex(parseInt(span.dataset.index, 10));
});

// -------------------------------------------------------------- layer slider
function layerDescriptor(layer, numLayers) {
  const third = numLayers / 3;
  if (layer < third) return "early · local grammar";
  if (layer < 2 * third) return "middle · meaning (clearest)";
  return "late · summary";
}

function syncLayerUI() {
  if (!state.data) return;
  const n = state.data.numLayers;
  const avg = els.avgLayers.checked;
  els.layer.min = "0";
  els.layer.max = String(n - 1);
  els.layer.value = String(galaxy.currentLayer);
  els.layer.disabled = avg; // slider is meaningless while averaging all layers
  els.layerValue.textContent = avg ? "avg" : `${galaxy.currentLayer + 1} / ${n}`;
  els.layerDescriptor.textContent = avg
    ? "averaging all layers"
    : layerDescriptor(galaxy.currentLayer, n);
}

els.layer.addEventListener("input", () => {
  galaxy.setLayer(parseInt(els.layer.value, 10));
  syncLayerUI();
});

// Attention display toggles.
els.hideSink.addEventListener("change", () => {
  galaxy.setHideSink(els.hideSink.checked);
});
els.avgLayers.addEventListener("change", () => {
  galaxy.setAvgLayers(els.avgLayers.checked);
  syncLayerUI();
});

// -------------------------------------------------------------- generation
function setLoading(on, text = "Thinking…") {
  els.loading.hidden = !on;
  els.loadingText.textContent = text;
}

/** Toggle the UI between idle and "generating" (inference + animation). */
function setGenerating(on) {
  state.generating = on;
  els.generate.textContent = on ? "⏭ Skip" : "Generate";
  els.generate.classList.toggle("skip", on);
  els.generate.disabled = !state.modelReady; // stays clickable (to Skip)
  els.prompt.disabled = on || !state.modelReady;
  els.replay.disabled = on || !state.data;
  els.cursor.style.display = on ? "inline" : "none";
}

function loadData(data) {
  state.data = data;
  galaxy.setData(data);
  syncLayerUI();
  galaxy.play(); // plays the word-by-word animation; settles via onRunStateChange
}

/** Show the finished galaxy immediately (no animation) — used after a Stop. */
function showFinalView(data) {
  state.data = data;
  galaxy.setData(data);
  syncLayerUI();
  galaxy.showFinal();
}

/**
 * Skip the slow word-by-word animation and jump to the final, explorable galaxy.
 * The model still generates the WHOLE sentence first: if inference is already
 * done (animation playing) we settle instantly; if it's still running we wait
 * for it to finish, showing the loading spinner meanwhile.
 */
function requestSkip() {
  if (!state.generating) return;
  state.skipRequested = true;
  if (galaxy.running) {
    galaxy.finish(); // animation already playing → jump to the final state
  } else {
    // Inference still running: let it finish, then settle directly.
    els.generate.textContent = "Finishing…";
    els.generate.disabled = true;
    setLoading(true, "Finishing the sentence…");
  }
}

async function runGenerate() {
  if (!state.modelReady) {
    els.status.textContent = "Model is still loading…";
    return;
  }
  const prompt = els.prompt.value.trim();
  if (!prompt) {
    els.status.textContent = "Type a prompt first.";
    return;
  }
  state.skipRequested = false;
  setGenerating(true);
  setLoading(true, "Running the model in your browser…");
  els.status.textContent = "";
  try {
    // Always generate the full sentence; skipping only affects the animation.
    const data = await generate(prompt);
    setLoading(false);
    if (state.skipRequested) {
      showFinalView(data); // skip the animation → final view directly
    } else {
      loadData(data); // play the word-by-word animation
    }
  } catch (err) {
    setLoading(false);
    els.status.textContent = `Error: ${err.message}`;
    console.error(err);
    setGenerating(false);
  }
}

// -------------------------------------------------------------- controls
els.generate.addEventListener("click", () => {
  if (state.generating) requestSkip();
  else runGenerate();
});
els.prompt.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !state.generating && state.modelReady) runGenerate();
});
els.replay.addEventListener("click", () => {
  if (state.generating || !state.data) return;
  state.skipRequested = false;
  setGenerating(true);
  galaxy.replay();
});

// -------------------------------------------------------------- model download
async function downloadModel() {
  els.gateError.hidden = true;
  els.downloadBtn.hidden = true;
  els.gateProgress.hidden = false;
  els.gateBarFill.style.width = "0%";
  els.gatePct.textContent = "Starting…";
  try {
    await initModel((frac) => {
      const pct = Math.round(frac * 100);
      els.gateBarFill.style.width = pct + "%";
      els.gatePct.textContent = `Downloading… ${pct}%`;
    });
    state.modelReady = true;
    els.modelGate.hidden = true;
    setGenerating(false); // configures buttons for the ready state
    els.status.textContent = `Ready — ${state.modelName} runs in your browser. Type a prompt and press Generate.`;
  } catch (err) {
    console.error(err);
    els.gateProgress.hidden = true;
    els.downloadBtn.hidden = false;
    els.gateError.hidden = false;
    els.gateError.textContent = `Download failed: ${err.message}. Please try again.`;
  }
}

els.downloadBtn.addEventListener("click", downloadModel);

// -------------------------------------------------------------- bootstrap
async function init() {
  els.cursor.style.display = "none";
  els.prompt.disabled = true;
  els.generate.disabled = true;
  els.replay.disabled = true;

  // Mirror the toggles' initial checked state into the renderer.
  galaxy.setAvgLayers(els.avgLayers.checked);
  galaxy.setHideSink(els.hideSink.checked);

  // Fetch the tiny local metadata (NOT the big weights) so the gate can name the
  // real model and layer count before the user commits to the download.
  try {
    const meta = await getMeta();
    state.modelName = String(meta.model).split("/").pop();
    els.gateDesc.innerHTML =
      `Watch <strong>${state.modelName}</strong> continue your sentence one word ` +
      `at a time — and see <strong>which earlier words it pays attention to</strong> ` +
      `for each prediction, across its ${meta.numLayers} layers, drawn as a galaxy ` +
      `of connections with clear percentages. Click any word to inspect exactly ` +
      `where its attention went.` +
      `<span class="gate-note">The model (~158&nbsp;MB) runs right in your ` +
      `browser — downloads once, then it's cached.</span>`;
    els.downloadBtn.textContent = `Download ${state.modelName} · ~158 MB`;
  } catch (err) {
    console.error("Could not read model metadata:", err);
    // Keep the generic gate copy; the download button still works.
  }

  // Show the gate — nothing downloads until the user clicks the button.
  els.modelGate.hidden = false;
}

init();
