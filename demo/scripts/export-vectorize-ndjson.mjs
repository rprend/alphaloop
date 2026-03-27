import fs from "node:fs/promises";
import path from "node:path";
import { loadCompactRealDataset } from "../shared/realSearch.mjs";

const ROOT = process.cwd();
const DATASET_PATH = path.resolve(ROOT, "demo/public/realDataset.json");
const EMBEDDINGS_PATH = path.resolve(ROOT, "demo/public/realEmbeddings.bin");
const OUTPUT_PATH = path.resolve(ROOT, "demo/data/vectorize.ndjson");

const dataset = await loadCompactRealDataset({
  metadataPath: DATASET_PATH,
  embeddingsPath: EMBEDDINGS_PATH,
  fsModule: fs,
});

await fs.mkdir(path.dirname(OUTPUT_PATH), { recursive: true });

const lines = dataset.chunks.map((chunk, chunkIndex) => {
  const offset = chunkIndex * dataset.dimensions;
  const values = Array.from(
    dataset.embeddings.slice(offset, offset + dataset.dimensions),
    (value) => value / 127,
  );

  return JSON.stringify({
    id: chunk.id,
    values,
    metadata: {
      title: chunk.title,
      text: chunk.text,
      pillar: chunk.metadata?.pillar,
      documentId: chunk.metadata?.documentId,
      chunkIndex: chunk.metadata?.chunkIndex,
    },
  });
});

await fs.writeFile(OUTPUT_PATH, `${lines.join("\n")}\n`);
console.log(`Wrote ${OUTPUT_PATH}`);
