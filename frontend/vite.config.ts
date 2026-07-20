import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      // Proxy /api and /heatmaps to the FastAPI backend during development
      "/api":      { target: "http://localhost:8000", changeOrigin: true },
      "/heatmaps": { target: "http://localhost:8000", changeOrigin: true },
      "/reports":  { target: "http://localhost:8000", changeOrigin: true },
      "/uploads":  { target: "http://localhost:8000", changeOrigin: true },
    },
  },
});
