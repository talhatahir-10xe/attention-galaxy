import { defineConfig } from "vite";

// `base: "./"` keeps asset paths relative so the static build works from any
// sub-path (e.g. GitHub Pages project sites) with no extra config.
export default defineConfig({
  base: "./",
  server: {
    port: 5173,
    host: true,
  },
  // onnxruntime-web ships its own wasm; let it load at runtime from public/ort
  // rather than having esbuild/rollup try to pre-bundle the wasm glue.
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    chunkSizeWarningLimit: 4000,
  },
});
