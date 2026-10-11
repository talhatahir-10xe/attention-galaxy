# Deploying Attention Galaxy — free, global, no backend

The whole app is static, so it can be hosted for **$0** and used by anyone in the
world. The only wrinkle is the model file (**158 MB**), which is over GitHub's
**100 MB per-file limit**. The fix: host the *site* on **GitHub Pages** and the
*model* on the **Hugging Face Hub** (free, CDN-backed, serves with CORS — it's
literally built for model weights).

```
Visitor ──▶ GitHub Pages  (your site: ~3 MB HTML/JS/CSS + tokenizer)
        └─▶ Hugging Face Hub CDN  (smollm2-135m-int8.onnx, 158 MB, cached after 1st load)
```

Nothing in this repo needs a paid service, and the model download is cached by
the browser so repeat visits are instant.

---

## One-time setup

### 1. Host the model on Hugging Face (free)

1. Make an account at <https://huggingface.co> (free).
2. Create a **new model repository** (not a Space): e.g. `your-name/attention-galaxy`.
   Keep it **public**.
3. Upload `frontend/public/model/smollm2-135m-int8.onnx` to it — drag-and-drop in
   the web UI works, or use the CLI:
   ```bash
   pip install -U "huggingface_hub[cli]"
   huggingface-cli login
   huggingface-cli upload your-name/attention-galaxy \
     frontend/public/model/smollm2-135m-int8.onnx smollm2-135m-int8.onnx
   ```
4. Your model's direct URL is then:
   ```
   https://huggingface.co/your-name/attention-galaxy/resolve/main/smollm2-135m-int8.onnx
   ```
   (The `resolve/main/` form returns the raw file with CORS + a CDN. Copy this URL.)

### 2. Put this repo on GitHub

The 158 MB `.onnx` is git-ignored (see `.gitignore`), so the push stays well under
GitHub's limits — only the small files go up.

```bash
cd /path/to/attention_galaxy
git init
git add .
git commit -m "Attention Galaxy — frontend-only in-browser LLM visualizer"
git branch -M main
git remote add origin https://github.com/your-name/attention-galaxy.git
git push -u origin main
```

### 3. (Optional) Override the model URL

The Hugging Face URL is already hardcoded as the production default in
`src/infer.js`, so **you can skip this step**. Only if you later move the model to
a different host: in the GitHub repo go to **Settings → Secrets and variables →
Actions → Variables tab → New repository variable**, name it `VITE_MODEL_URL`, and
set the new `resolve/main/…onnx` URL. The build uses it in preference to the
hardcoded default. (It's a plain *variable*, not a secret — the URL is public.)

### 4. Turn on GitHub Pages

**Settings → Pages → Build and deployment → Source: GitHub Actions.**

That's it. The included workflow (`.github/workflows/deploy.yml`) runs on every
push to `main`: it builds the frontend with `VITE_MODEL_URL` baked in and deploys
to Pages. Your site goes live at:

```
https://your-name.github.io/attention-galaxy/
```

(First deploy takes a minute or two — watch the **Actions** tab.)

---

## Updating later

- **Code change:** just `git push` — the workflow redeploys automatically.
- **New / different model:** upload the new `.onnx` to the HF repo, update the
  `VITE_MODEL_URL` variable if the filename changed, and re-run the workflow
  (push anything, or **Actions → Deploy → Run workflow**).

---

## Notes & gotchas

- **Why not Google Drive / Dropbox for the model?** They don't send CORS headers,
  so the browser refuses the cross-origin fetch; large files also hit virus-scan
  interstitial pages and per-file rate limits. HF Hub avoids all of this.
- **GitHub Pages bandwidth:** Pages has a ~100 GB/month soft limit. Because the
  heavy model is served by HF (not Pages), your Pages bandwidth is just the ~3 MB
  site — you'd need tens of thousands of visitors/month to approach the cap.
- **ORT WASM runtime** loads from the jsDelivr CDN (pinned `onnxruntime-web@1.20.1`)
  and is cached after first load; no action needed.
- **Mobile:** it works, but 158 MB is a heavy first download on a phone and needs
  ~0.5–1 GB RAM to run. A smaller / 4-bit model variant for mobile is a planned
  follow-up (see `PROGRESS.md` §9).
