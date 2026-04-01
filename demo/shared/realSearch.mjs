import { embed } from "ai";

function estimateTokens(text) {
  return Math.max(1, Math.ceil(text.length / 4));
}

function normalizeVector(values) {
  let magnitude = 0;
  for (const value of values) {
    magnitude += value * value;
  }
  const scale = magnitude > 0 ? 1 / Math.sqrt(magnitude) : 1;
  return values.map((value) => value * scale);
}

function quantizedSimilarity(queryEmbedding, embeddings, offset, dimensions) {
  let dot = 0;
  for (let index = 0; index < dimensions; index++) {
    dot += queryEmbedding[index] * (embeddings[offset + index] / 127);
  }
  return dot;
}

export function hydrateRealDataset(metadata, embeddingsBuffer) {
  return {
    chunks: metadata.chunks,
    dimensions: metadata.dimensions,
    embeddings: new Int8Array(embeddingsBuffer),
  };
}

export async function loadCompactRealDataset({
  metadataPath,
  embeddingsPath,
  fsModule,
}) {
  const [metadataText, embeddingsBuffer] = await Promise.all([
    fsModule.readFile(metadataPath, "utf8"),
    fsModule.readFile(embeddingsPath),
  ]);

  return hydrateRealDataset(JSON.parse(metadataText), embeddingsBuffer);
}

function buildStressChunk(baseChunk, scenario, replicaIndex) {
  return {
    id:
      scenario.replicaCount === 1 && replicaIndex === 0
        ? baseChunk.id
        : `${baseChunk.id}::${scenario.id}::replica-${replicaIndex + 1}`,
    text: baseChunk.text,
    score: baseChunk.score,
    metadata: {
      ...baseChunk.metadata,
      scenarioId: scenario.id,
      replicaIndex,
    },
  };
}

export function getScenarioStats(dataset, scenario) {
  const baseTokens = dataset.chunks.reduce(
    (total, chunk) => total + estimateTokens(chunk.text),
    0,
  );

  return {
    embeddedChunks: dataset.chunks.length,
    virtualChunks: dataset.chunks.length * scenario.replicaCount,
    estimatedTokens: baseTokens * scenario.replicaCount,
    replicaCount: scenario.replicaCount,
  };
}

export function createRealSearch({
  dataset,
  scenario,
  embeddingModel,
  embeddingDimensions,
  pageSize = 500,
  onSearchStats,
}) {
  const rankingCache = new Map();

  async function getRankedBaseChunks(query) {
    const cacheKey = query.trim().toLowerCase();
    const cached = rankingCache.get(cacheKey);
    if (cached) {
      return cached;
    }

    const { embedding } = await embed({
      model: embeddingModel,
      value: query,
      providerOptions: embeddingDimensions
        ? {
            openai: {
              dimensions: embeddingDimensions,
            },
          }
        : undefined,
    });
    const normalizedQuery = normalizeVector(embedding);

    const ranked = dataset.chunks
      .map((chunk, index) => ({
        id: chunk.id,
        text: chunk.text,
        score: quantizedSimilarity(
          normalizedQuery,
          dataset.embeddings,
          index * dataset.dimensions,
          dataset.dimensions,
        ),
        metadata: {
          title: chunk.title,
          ...chunk.metadata,
        },
      }))
      .sort((a, b) => b.score - a.score);

    rankingCache.set(cacheKey, ranked);
    return ranked;
  }

  return async function search(query, { minScore, topK, cursor }) {
    const ranked = await getRankedBaseChunks(query);
    const threshold = minScore ?? 0;
    const baseMatches = ranked.filter((chunk) => chunk.score >= threshold);
    const totalStrongMatches = baseMatches.length * scenario.replicaCount;
    const cappedStrongMatches =
      topK == null ? totalStrongMatches : Math.min(topK, totalStrongMatches);

    const start = cursor ? Number(cursor) : 0;
    const end = Math.min(start + pageSize, cappedStrongMatches);
    const chunks = [];
    const baseCount = baseMatches.length;

    for (let index = start; index < end; index++) {
      const replicaIndex = Math.floor(index / Math.max(1, baseCount));
      const baseIndex = index % Math.max(1, baseCount);
      const baseChunk = baseMatches[baseIndex];
      if (!baseChunk) break;
      chunks.push(buildStressChunk(baseChunk, scenario, replicaIndex));
    }

    const baseTokenEstimate = baseMatches.reduce(
      (sum, chunk) => sum + estimateTokens(chunk.text),
      0,
    );

    const page = {
      chunks,
      nextCursor: end < cappedStrongMatches ? String(end) : undefined,
      totalStrongMatches: cappedStrongMatches,
      totalBaseMatches: baseMatches.length,
      estimatedTokens: baseTokenEstimate * scenario.replicaCount,
    };

    onSearchStats?.({
      query,
      ...page,
    });

    return page;
  };
}

export function createVectorizeSearch({
  index,
  totalChunks,
  scenario,
  embeddingModel,
  embeddingDimensions,
  apiKey,
  embeddingModelId,
  pageSize = 500,
  onSearchStats,
}) {
  const rankingCache = new Map();

  async function createQueryEmbedding(query) {
    if (apiKey && embeddingModelId) {
      const response = await fetch("https://api.openai.com/v1/embeddings", {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model: embeddingModelId,
          input: query,
          dimensions: embeddingDimensions,
        }),
      });

      if (!response.ok) {
        throw new Error(
          `Embedding request failed (${response.status}): ${await response.text()}`,
        );
      }

      const json = await response.json();
      return json.data?.[0]?.embedding || [];
    }

    const { embedding } = await embed({
      model: embeddingModel,
      value: query,
      providerOptions: embeddingDimensions
        ? {
            openai: {
              dimensions: embeddingDimensions,
            },
          }
        : undefined,
    });

    return embedding;
  }

  async function getRankedBaseChunks(query) {
    const cacheKey = query.trim().toLowerCase();
    const cached = rankingCache.get(cacheKey);
    if (cached) {
      return cached;
    }

    const embedding = await createQueryEmbedding(query);
    const normalizedQuery = normalizeVector(embedding);
    const topK = Math.max(1000, totalChunks || 0);

    const response = await index.query(normalizedQuery, {
      topK,
      returnMetadata: "all",
    });

    const ranked = (response.matches || [])
      .map((match) => ({
        id: match.id,
        text: match.metadata?.text || "",
        score: match.score ?? 0,
        metadata: {
          title: match.metadata?.title || match.id,
          pillar: match.metadata?.pillar,
          documentId: match.metadata?.documentId,
          chunkIndex: match.metadata?.chunkIndex,
        },
      }))
      .sort((a, b) => b.score - a.score);

    rankingCache.set(cacheKey, ranked);
    return ranked;
  }

  return async function search(query, { minScore, topK, cursor }) {
    const ranked = await getRankedBaseChunks(query);
    const threshold = minScore ?? 0;
    const baseMatches = ranked.filter((chunk) => chunk.score >= threshold);
    const totalStrongMatches = baseMatches.length * scenario.replicaCount;
    const cappedStrongMatches =
      topK == null ? totalStrongMatches : Math.min(topK, totalStrongMatches);

    const start = cursor ? Number(cursor) : 0;
    const end = Math.min(start + pageSize, cappedStrongMatches);
    const chunks = [];
    const baseCount = baseMatches.length;

    for (let index = start; index < end; index++) {
      const replicaIndex = Math.floor(index / Math.max(1, baseCount));
      const baseIndex = index % Math.max(1, baseCount);
      const baseChunk = baseMatches[baseIndex];
      if (!baseChunk) break;
      chunks.push(buildStressChunk(baseChunk, scenario, replicaIndex));
    }

    const baseTokenEstimate = baseMatches.reduce(
      (sum, chunk) => sum + estimateTokens(chunk.text),
      0,
    );

    const page = {
      chunks,
      nextCursor: end < cappedStrongMatches ? String(end) : undefined,
      totalStrongMatches: cappedStrongMatches,
      totalBaseMatches: baseMatches.length,
      estimatedTokens: baseTokenEstimate * scenario.replicaCount,
    };

    onSearchStats?.({
      query,
      ...page,
    });

    return page;
  };
}
