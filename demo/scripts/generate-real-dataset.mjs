import fs from "node:fs/promises";
import path from "node:path";
import { buildRealSourceDocuments } from "../shared/realSourceDocs.mjs";

const ROOT = process.cwd();
const ENV_PATH = path.resolve(ROOT, "../alphabook/.dev.vars");
const OUT_PATH = path.resolve(ROOT, "demo/public/realDataset.json");

const docs = buildRealSourceDocuments();
const env = await loadEnvFile(ENV_PATH);
const apiKey = env.OPENAI_API_KEY;
const model = env.OPENAI_EMBEDDING_MODEL || "text-embedding-3-large";

if (!apiKey) {
  throw new Error(`OPENAI_API_KEY missing from ${ENV_PATH}`);
}

const embeddings = [];
const batchSize = 16;

for (let start = 0; start < docs.length; start += batchSize) {
  const batch = docs.slice(start, start + batchSize);
  const response = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      input: batch.map((doc) => doc.text),
    }),
  });

  if (!response.ok) {
    throw new Error(
      `Embedding batch failed (${response.status}): ${await response.text()}`,
    );
  }

  const json = await response.json();
  for (const item of json.data) {
    embeddings.push(item.embedding);
  }
  console.log(`Embedded ${Math.min(start + batch.length, docs.length)} / ${docs.length}`);
}

const dataset = docs.map((doc, index) => ({
  ...doc,
  embedding: embeddings[index],
}));

await fs.mkdir(path.dirname(OUT_PATH), { recursive: true });
await fs.writeFile(OUT_PATH, JSON.stringify(dataset));
console.log(`Wrote ${OUT_PATH}`);

async function loadEnvFile(filePath) {
  const text = await fs.readFile(filePath, "utf8");
  const values = {};

  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const match = line.match(/^([A-Z0-9_]+)=(.*)$/);
    if (!match) continue;
    let value = match[2];
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    values[match[1]] = value;
  }

  return values;
}
