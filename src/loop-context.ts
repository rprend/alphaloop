import { DEFAULTS } from "./defaults.js";
import type {
  AlphaloopConfig,
  AlphaloopRunOptions,
  AlphaloopStreamEvent,
  LoopContext,
} from "./types.js";

export function createLoopContext(
  config: AlphaloopConfig,
  options: AlphaloopRunOptions = {},
  emit: (event: AlphaloopStreamEvent) => void = () => {},
): LoopContext {
  const maxContextTokens =
    options.maxContextTokens ??
    config.maxContextTokens ??
    DEFAULTS.maxContextTokens;
  const softTokenLimit = Math.floor(
    maxContextTokens *
      (config.softContextLimitRatio ?? DEFAULTS.softContextLimitRatio),
  );
  const hardTokenLimit = Math.floor(
    maxContextTokens *
      (config.hardContextLimitRatio ?? DEFAULTS.hardContextLimitRatio),
  );

  return {
    config: {
      ...config,
      minScore: options.minScore ?? config.minScore ?? DEFAULTS.minScore,
      topK: options.topK ?? config.topK,
      maxExpandedQueries:
        config.maxExpandedQueries ?? DEFAULTS.maxExpandedQueries,
      maxIterations: config.maxIterations ?? DEFAULTS.maxIterations,
      relevanceThreshold:
        config.relevanceThreshold ?? DEFAULTS.relevanceThreshold,
      enableClassifier: config.enableClassifier ?? DEFAULTS.enableClassifier,
      maxContextTokens,
      softContextLimitRatio:
        config.softContextLimitRatio ?? DEFAULTS.softContextLimitRatio,
      hardContextLimitRatio:
        config.hardContextLimitRatio ?? DEFAULTS.hardContextLimitRatio,
      outputTokenReserve:
        config.outputTokenReserve ?? DEFAULTS.outputTokenReserve,
    },
    seenChunks: new Map(),
    rankedChunks: new Map(),
    triedQueries: new Set(),
    visibleChunkIds: new Set(),
    visibleDocuments: new Map(),
    observations: [],
    tokenUsage: 0,
    softTokenLimit,
    hardTokenLimit,
    iterations: [],
    totalChunksMatched: 0,
    retrievalRequests: 0,
    shardCount: 0,
    recursionDepth: 0,
    emit,
  };
}
