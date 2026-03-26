import { DEFAULTS } from "./defaults.js";
import { embeddingSearch } from "./steps/embedding-search.js";
import { queryExpansion } from "./steps/query-expansion.js";
import { rerank } from "./steps/rerank.js";
import { iterativeSearch } from "./steps/iterative-search.js";
import { classify } from "./steps/classifier.js";
import type {
  AlphaloopConfig,
  AlphaloopResult,
  AlphaloopStreamEvent,
  EmbeddingChunk,
  LoopContext,
  RankedChunk,
} from "./types.js";
import { alphaloopTools } from "./tools.js";

function createLoopContext(
  config: AlphaloopConfig,
  emit: (event: AlphaloopStreamEvent) => void,
): LoopContext {
  return {
    config: {
      ...config,
      initialTopK: config.initialTopK ?? DEFAULTS.initialTopK,
      maxExpandedQueries:
        config.maxExpandedQueries ?? DEFAULTS.maxExpandedQueries,
      maxIterations: config.maxIterations ?? DEFAULTS.maxIterations,
      relevanceThreshold:
        config.relevanceThreshold ?? DEFAULTS.relevanceThreshold,
      enableClassifier: config.enableClassifier ?? DEFAULTS.enableClassifier,
    },
    seenChunks: new Map(),
    rankedChunks: new Map(),
    triedQueries: new Set(),
    iterations: [],
    emit,
  };
}

/**
 * Run the full 5-step agentic retrieval loop.
 */
async function run(
  query: string,
  config: AlphaloopConfig,
  emit: (event: AlphaloopStreamEvent) => void = () => {},
): Promise<AlphaloopResult> {
  const ctx = createLoopContext(config, emit);

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
  });

  return {
    chunks: finalChunks,
    iterations: ctx.iterations,
    totalChunksConsidered: ctx.seenChunks.size,
  };
}

/**
 * Creates an alphaloop instance with the given configuration.
 */
export function createAlphaloop(config: AlphaloopConfig) {
  return {
    /** Run the full loop and return results. */
    async run(query: string): Promise<AlphaloopResult> {
      return run(query, config);
    },

    /** Run the loop as an async iterable of stream events. */
    async *stream(
      query: string,
    ): AsyncGenerator<AlphaloopStreamEvent, AlphaloopResult> {
      const events: AlphaloopStreamEvent[] = [];
      let resolve: (() => void) | null = null;

      const emit = (event: AlphaloopStreamEvent) => {
        events.push(event);
        resolve?.();
      };

      const resultPromise = run(query, config, emit);

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
