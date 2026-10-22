import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Deployable under a sub-path (e.g. demos.milvus.io/minhash-near-duplicate) by
// setting VITE_BASE_PATH at build time. Defaults to "/" for local development.
const rawBase = process.env.VITE_BASE_PATH || "/";
const base = rawBase.endsWith("/") ? rawBase : `${rawBase}/`;

export default defineConfig({
  base,
  plugins: [react()],
  server: {
    port: 5176,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:48050",
        changeOrigin: false,
      },
      "/healthz": "http://127.0.0.1:48050",
    },
  },
});
