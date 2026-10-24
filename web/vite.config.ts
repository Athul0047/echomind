import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

const proxy = {
  "/api": { target: "http://127.0.0.1:8765", changeOrigin: false, ws: true },
};

export default defineConfig({
  plugins: [react()],
  server: { host: "127.0.0.1", proxy },
  preview: { host: "127.0.0.1", proxy },
  test: {
    environment: "jsdom",
    setupFiles: "./src/testSetup.ts",
    css: true,
    restoreMocks: true,
  },
});
