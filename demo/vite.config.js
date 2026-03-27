import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

export default defineConfig({
  plugins: [react()],
  root: path.resolve(process.cwd(), "demo"),
  resolve: {
    alias: [
      {
        find: "alphaloop/react",
        replacement: path.resolve(process.cwd(), "src/react/index.ts"),
      },
      {
        find: "alphaloop",
        replacement: path.resolve(process.cwd(), "src/index.ts"),
      },
    ],
  },
  server: {
    host: "127.0.0.1",
    port: 4173,
  },
  preview: {
    host: "127.0.0.1",
    port: 4173,
  },
});
