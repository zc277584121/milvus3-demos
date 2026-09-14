import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Deployable under a sub-path (e.g. demos.milvus.io/function-chain-rerank) by
// setting VITE_BASE_PATH at build time. Defaults to "/" for local development.
const rawBase = process.env.VITE_BASE_PATH || "/";
const base = rawBase.endsWith("/") ? rawBase : `${rawBase}/`;

export default defineConfig({
  base,
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:48020",
        changeOrigin: false,
      },
      "/healthz": "http://127.0.0.1:48020",
    },
  },
});
