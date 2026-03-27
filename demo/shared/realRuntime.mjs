import { createOpenAI } from "@ai-sdk/openai";
import { createAlphaloop } from "../../dist/index.js";
import { createRealSearch, getScenarioStats } from "./realSearch.mjs";
import { REAL_STRESS_SCENARIOS } from "./realStressScenarios.mjs";

export async function runRealStressScenario({
  dataset,
  search,
  embeddedChunkCount,
  apiKey,
  modelId,
  rerankModelId,
  embeddingModelId,
  embeddingDimensions,
  query,
  scenarioId,
  minScore,
  topK,
  maxContextTokens = 100_000,
  maxExpandedQueries,
  maxIterations,
  onEvent,
}) {
  const scenario = REAL_STRESS_SCENARIOS[scenarioId];
  if (!scenario) {
    throw new Error(`Unknown scenario: ${scenarioId}`);
  }

  const resolvedModelId = modelId || "gpt-5.2";
  const resolvedRerankModelId = rerankModelId || resolvedModelId;
  const resolvedEmbeddingModelId = embeddingModelId || "text-embedding-3-large";
  const openai = createOpenAI({ apiKey });
  const searchStats = {
    maxEstimatedTokens: 0,
    maxStrongMatches: 0,
    maxBaseMatches: 0,
  };
  const recordSearchStats = (page) => {
    searchStats.maxEstimatedTokens = Math.max(
      searchStats.maxEstimatedTokens,
      page.estimatedTokens,
    );
    searchStats.maxStrongMatches = Math.max(
      searchStats.maxStrongMatches,
      page.totalStrongMatches,
    );
    searchStats.maxBaseMatches = Math.max(
      searchStats.maxBaseMatches,
      page.totalBaseMatches,
    );
  };

  let resolvedSearch =
    search ||
    createRealSearch({
      dataset,
      scenario,
      embeddingModel: openai.embedding(resolvedEmbeddingModelId),
      embeddingDimensions,
      onSearchStats: recordSearchStats,
    });

  if (search) {
    const baseSearch = search;
    resolvedSearch = async (query, options) => {
      const page = await baseSearch(query, options);
      searchStats.maxEstimatedTokens = Math.max(
        searchStats.maxEstimatedTokens,
        page.estimatedTokens,
      );
      searchStats.maxStrongMatches = Math.max(
        searchStats.maxStrongMatches,
        page.totalStrongMatches,
      );
      searchStats.maxBaseMatches = Math.max(
        searchStats.maxBaseMatches,
        page.totalBaseMatches,
      );
      return page;
    };
  }

  const loop = createAlphaloop({
    model: openai(resolvedModelId),
    rerankModel: openai(resolvedRerankModelId),
    search: resolvedSearch,
    minScore: minScore ?? scenario.minScore,
    maxContextTokens,
    maxExpandedQueries: maxExpandedQueries ?? scenario.maxExpandedQueries,
    maxIterations: maxIterations ?? scenario.maxIterations,
  });

  const stream = loop.stream(query || scenario.query, {
    minScore: minScore ?? scenario.minScore,
    topK,
    maxContextTokens,
  });

  let result;
  while (true) {
    const next = await stream.next();
    if (next.done) {
      result = next.value;
      break;
    }
    onEvent?.(next.value);
  }

  return {
    query: query || scenario.query,
    scenario,
    result,
    stats: dataset
      ? getScenarioStats(dataset, scenario)
      : {
          embeddedChunks: embeddedChunkCount ?? 0,
          virtualChunks: (embeddedChunkCount ?? 0) * scenario.replicaCount,
          estimatedTokens: searchStats.maxEstimatedTokens,
          replicaCount: scenario.replicaCount,
        },
    searchStats,
    runtime: {
      modelId: resolvedModelId,
      rerankModelId: resolvedRerankModelId,
      embeddingModelId: resolvedEmbeddingModelId,
      embeddingDimensions,
      minScoreUsed: minScore ?? scenario.minScore,
      topKUsed: topK,
      maxContextTokens,
      maxExpandedQueries: maxExpandedQueries ?? scenario.maxExpandedQueries,
      maxIterations: maxIterations ?? scenario.maxIterations,
    },
  };
}
