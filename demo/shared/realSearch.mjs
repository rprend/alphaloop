import { embed } from "ai";

function cosineSimilarity(a, b) {
  let dot = 0;
  let magA = 0;
  let magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  if (magA === 0 || magB === 0) return 0;
  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

function estimateTokens(text) {
  return Math.max(1, Math.ceil(text.length / 4));
}

function buildStressText(baseChunk, scenario, replicaIndex) {
  const sections = [];

  for (let copy = 0; copy < scenario.textMultiplier; copy++) {
    sections.push(
      `Stress replica ${replicaIndex + 1} / ${scenario.replicaCount}, repetition ${copy + 1} / ${scenario.textMultiplier}.`,
    );
    sections.push(baseChunk.text);
  }

  return sections.join("\n\n");
}

function buildStressChunk(baseChunk, scenario, replicaIndex) {
  return {
    id:
      scenario.replicaCount === 1 &&
      scenario.textMultiplier === 1 &&
      replicaIndex === 0
        ? baseChunk.id
        : `${baseChunk.id}::${scenario.id}::replica-${replicaIndex + 1}`,
    text: buildStressText(baseChunk, scenario, replicaIndex),
    score: baseChunk.score,
    metadata: {
      ...baseChunk.metadata,
      scenarioId: scenario.id,
      replicaIndex,
      textMultiplier: scenario.textMultiplier,
    },
  };
}

export function getScenarioStats(dataset, scenario) {
  const baseTokens = dataset.reduce(
    (total, chunk) => total + estimateTokens(chunk.text),
    0,
  );

  return {
    embeddedDocuments: dataset.length,
    virtualDocuments: dataset.length * scenario.replicaCount,
    estimatedTokens:
      baseTokens * scenario.replicaCount * scenario.textMultiplier,
    replicaCount: scenario.replicaCount,
    textMultiplier: scenario.textMultiplier,
  };
}

export function createRealSearch({
  dataset,
  scenario,
  embeddingModel,
  pageSize = 20,
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
    });

    const ranked = dataset
      .map((chunk) => ({
        id: chunk.id,
        text: chunk.text,
        score: cosineSimilarity(embedding, chunk.embedding),
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
      estimatedTokens:
        baseTokenEstimate * scenario.replicaCount * scenario.textMultiplier,
    };

    onSearchStats?.({
      query,
      ...page,
    });

    return page;
  };
}
