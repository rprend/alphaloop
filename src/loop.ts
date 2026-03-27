import { DEFAULTS } from "./defaults.js";
import { embeddingSearch } from "./steps/embedding-search.js";
import { queryExpansion } from "./steps/query-expansion.js";
import { rerank } from "./steps/rerank.js";
import { iterativeSearch } from "./steps/iterative-search.js";
import { classify } from "./steps/classifier.js";
import type {
  AlphaloopConfig,
  AlphaloopRunOptions,
  AlphaloopResult,
  AlphaloopStreamEvent,
  LoopContext,
} from "./types.js";
import { alphaloopTools } from "./tools.js";

function createLoopContext(
  config: AlphaloopConfig,
  options: AlphaloopRunOptions,
  emit: (event: AlphaloopStreamEvent) => void,
): LoopContext {
  return {
    config: {
      ...config,
      minScore: options.minScore ?? config.minScore ?? DEFAULTS.minScore,
      maxExpandedQueries:
        config.maxExpandedQueries ?? DEFAULTS.maxExpandedQueries,
      maxIterations: config.maxIterations ?? DEFAULTS.maxIterations,
      relevanceThreshold:
        config.relevanceThreshold ?? DEFAULTS.relevanceThreshold,
      enableClassifier: config.enableClassifier ?? DEFAULTS.enableClassifier,
      maxContextTokens:
        options.maxContextTokens ??
        config.maxContextTokens ??
        DEFAULTS.maxContextTokens,
    },
    seenChunks: new Map(),
    rankedChunks: new Map(),
    triedQueries: new Set(),
    iterations: [],
    totalChunksMatched: 0,
    retrievalRequests: 0,
    shardCount: 0,
    recursionDepth: 0,
    emit,
  };
}

/**
 * Run the full 5-step agentic retrieval loop.
 */
async function run(
  query: string,
  config: AlphaloopConfig,
  options: AlphaloopRunOptions = {},
  emit: (event: AlphaloopStreamEvent) => void = () => {},
): Promise<AlphaloopResult> {
  const ctx = createLoopContext(config, options, emit);

  // Step 1: Initial embedding search
  const initialChunks = await embeddingSearch(query, ctx);

  // Step 2: Query expansion
  const expandedChunks = await queryExpansion(query, initialChunks, ctx);

  // Step 3: Re-rank all collected chunks
  const allChunks = Array.from(ctx.seenChunks.values());
  await rerank(query, allChunks, ctx, { sourceQuery: query, iteration: 0 });

  // Step 4: Iterative search
  await iterativeSearch(query, ctx);

  // Step 5: Optional classifier
  if (ctx.config.enableClassifier) {
    await classify(query, ctx);
  }

  // Collect final results
  const finalChunks = Array.from(ctx.rankedChunks.values()).sort(
    (a, b) => b.relevance - a.relevance,
  );

  emit({
    type: "complete",
    totalChunks: finalChunks.length,
    iterations: ctx.iterations.length,
    totalChunksMatched: ctx.totalChunksMatched,
    minScore: ctx.config.minScore,
    shardCount: ctx.shardCount,
    recursionDepth: ctx.recursionDepth,
  });

  return {
    chunks: finalChunks,
    iterations: ctx.iterations,
    totalChunksConsidered: ctx.seenChunks.size,
    totalChunksMatched: ctx.totalChunksMatched,
    minScoreUsed: ctx.config.minScore,
    recursionDepth: ctx.recursionDepth,
    shardCount: ctx.shardCount,
  };
}

/**
 * Creates an alphaloop instance with the given configuration.
 */
export function createAlphaloop(config: AlphaloopConfig) {
  return {
    /** Run the full loop and return results. */
    async run(
      query: string,
      options?: AlphaloopRunOptions,
    ): Promise<AlphaloopResult> {
      return run(query, config, options);
    },

    /** Run the loop as an async iterable of stream events. */
    async *stream(
      query: string,
      options?: AlphaloopRunOptions,
    ): AsyncGenerator<AlphaloopStreamEvent, AlphaloopResult> {
      const events: AlphaloopStreamEvent[] = [];
      let resolve: (() => void) | null = null;

      const emit = (event: AlphaloopStreamEvent) => {
        events.push(event);
        resolve?.();
      };

      const resultPromise = run(query, config, options, emit);

      // Yield events as they arrive
      let done = false;
      resultPromise.then(() => {
        done = true;
        resolve?.();
      });

      while (!done) {
        if (events.length > 0) {
          yield events.shift()!;
        } else {
          await new Promise<void>((r) => {
            resolve = r;
          });
        }
      }

      // Drain remaining events
      while (events.length > 0) {
        yield events.shift()!;
      }

      return await resultPromise;
    },

    /** Get the loop as AI SDK tools for use with streamText. */
    tools() {
      return alphaloopTools(config);
    },
  };
}
