/**
 * Wiring: DOM controls, the Galaxy renderer, and in-browser inference.
 *
 * There is no backend — the model runs in the browser (infer.js). Bundled
 * examples render instantly as demos while the model downloads in the
 * background; once it's ready the prompt box generates for any input.
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
  status: $("status"),
  loading: $("loading"),
  loadingText: $("loading-text"),
};

const state = {
  modelReady: false,
  data: null,
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
    els.generate.disabled = running || !state.modelReady;
    els.replay.disabled = running || !state.data;
    els.cursor.style.display = running ? "inline" : "none";
    // Once settled, make the sentence words clickable right away.
    if (!running && state.data) renderCaptionTokens(null);
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
  els.layer.min = "0";
  els.layer.max = String(n - 1);
  els.layer.value = String(galaxy.currentLayer);
  els.layerValue.textContent = `${galaxy.currentLayer + 1} / ${n}`;
  els.layerDescriptor.textContent = layerDescriptor(galaxy.currentLayer, n);
}

els.layer.addEventListener("input", () => {
  galaxy.setLayer(parseInt(els.layer.value, 10));
  syncLayerUI();
});

// -------------------------------------------------------------- generation
function setLoading(on, text = "Thinking…") {
  els.loading.hidden = !on;
  els.loadingText.textContent = text;
}

function loadData(data) {
  state.data = data;
  galaxy.setData(data);
  syncLayerUI();
  els.replay.disabled = false;
  galaxy.play();
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
  setLoading(true, "Running the model in your browser…");
  els.generate.disabled = true;
  els.status.textContent = "";
  try {
    const data = await generate(prompt);
    loadData(data);
  } catch (err) {
    els.status.textContent = `Error: ${err.message}`;
    console.error(err);
  } finally {
    setLoading(false);
    els.generate.disabled = false;
  }
}

// -------------------------------------------------------------- controls
els.generate.addEventListener("click", runGenerate);
els.prompt.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !els.generate.disabled) runGenerate();
});
els.replay.addEventListener("click", () => galaxy.replay());

// -------------------------------------------------------------- bootstrap
async function init() {
  els.cursor.style.display = "none";
  els.prompt.disabled = true;
  els.generate.disabled = true;

  // The galaxy starts empty — nothing runs until the user presses Generate.
  // Download + warm up the in-browser model; progress shows in the status line.
  try {
    els.status.textContent = "Downloading model… 0%";
    await initModel((frac) => {
      els.status.textContent = `Downloading model… ${Math.round(frac * 100)}%`;
    });
    const meta = await getMeta();
    const modelName = String(meta.model).split("/").pop();
    state.modelReady = true;
    els.prompt.disabled = false;
    els.generate.disabled = false;
    els.prompt.placeholder = "Type a prompt, e.g. The mouse ran up the";
    els.status.textContent = `Ready — ${modelName} runs in your browser. Type a prompt and press Generate.`;
  } catch (err) {
    console.error(err);
    els.status.textContent = "Model failed to load — " + err.message;
  }
}

init();
