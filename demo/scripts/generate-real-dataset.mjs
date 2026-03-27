import fs from "node:fs/promises";
import path from "node:path";
import { buildRealSourceChunks } from "../shared/realSourceDocs.mjs";

const ROOT = process.cwd();
const ENV_PATH = path.resolve(ROOT, "../alphabook/.dev.vars");
const METADATA_OUT_PATH = path.resolve(ROOT, "demo/public/realDataset.json");
const EMBEDDINGS_OUT_PATH = path.resolve(ROOT, "demo/public/realEmbeddings.bin");

const chunks = buildRealSourceChunks();
const env = await loadEnvFile(ENV_PATH);
const apiKey = env.OPENAI_API_KEY;
const model = env.OPENAI_EMBEDDING_MODEL || "text-embedding-3-large";
const dimensions = Number(env.OPENAI_EMBEDDING_DIMENSIONS || 1536);

if (!apiKey) {
  throw new Error(`OPENAI_API_KEY missing from ${ENV_PATH}`);
}

const embeddings = [];
const batchSize = 16;

for (let start = 0; start < chunks.length; start += batchSize) {
  const batch = chunks.slice(start, start + batchSize);
  const response = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      input: batch.map((chunk) => chunk.text),
      dimensions,
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
  console.log(`Embedded ${Math.min(start + batch.length, chunks.length)} / ${chunks.length}`);
}

const actualDimensions = embeddings[0]?.length ?? 0;
const quantized = new Int8Array(chunks.length * actualDimensions);

for (let chunkIndex = 0; chunkIndex < chunks.length; chunkIndex++) {
  const embedding = embeddings[chunkIndex];
  let magnitude = 0;
  for (const value of embedding) {
    magnitude += value * value;
  }
  const scale = magnitude > 0 ? 1 / Math.sqrt(magnitude) : 1;

  for (let dimensionIndex = 0; dimensionIndex < actualDimensions; dimensionIndex++) {
    const normalized = embedding[dimensionIndex] * scale;
    quantized[chunkIndex * actualDimensions + dimensionIndex] = Math.max(
      -127,
      Math.min(127, Math.round(normalized * 127)),
    );
  }
}

const metadata = {
  dimensions: actualDimensions,
  chunks,
};

await fs.mkdir(path.dirname(METADATA_OUT_PATH), { recursive: true });
await fs.writeFile(METADATA_OUT_PATH, JSON.stringify(metadata));
await fs.writeFile(EMBEDDINGS_OUT_PATH, Buffer.from(quantized.buffer));
console.log(`Wrote ${METADATA_OUT_PATH}`);
console.log(`Wrote ${EMBEDDINGS_OUT_PATH}`);

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
