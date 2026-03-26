import { defineConfig } from "tsup";

export default defineConfig([
  {
    entry: {
      index: "src/index.ts",
      handler: "src/handler.ts",
    },
    format: ["esm"],
    dts: true,
    outDir: "dist",
    external: [
      "ai",
      "zod",
      "react",
      "@assistant-ui/react",
      "@assistant-ui/react-ai-sdk",
    ],
    target: "es2022",
    platform: "browser",
    clean: true,
  },
  {
    entry: {
      "react/index": "src/react/index.ts",
    },
    format: ["esm"],
    dts: true,
    outDir: "dist",
    external: [
      "ai",
      "zod",
      "react",
      "alphaloop",
      "@assistant-ui/react",
      "@assistant-ui/react-ai-sdk",
    ],
    target: "es2022",
    platform: "browser",
  },
]);
