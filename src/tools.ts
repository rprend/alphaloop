import { tool } from "ai";
import { z } from "zod";
import { DEFAULTS } from "./defaults.js";
import { embeddingSearch } from "./steps/embedding-search.js";
import { queryExpansion } from "./steps/query-expansion.js";
import { rerank } from "./steps/rerank.js";
import { iterativeSearch } from "./steps/iterative-search.js";
import { classify } from "./steps/classifier.js";
import type {
  AlphaloopConfig,
  AlphaloopStreamEvent,
  LoopContext,
} from "./types.js";

function createLoopContext(
  config: AlphaloopConfig,
): { ctx: LoopContext; progress: string[]; logLines: Array<{ key: string; value: string }> } {
  const progress: string[] = [];
  const logLines: Array<{ key: string; value: string }> = [];

  const emit = (event: AlphaloopStreamEvent) => {
    switch (event.type) {
      case "embedding_search":
        progress.push(`Searching embeddings for "${event.query}"`);
        logLines.push({
          key: "Initial search",
          value: `Found ${event.chunksFound} chunks`,
        });
        break;
      case "query_expansion":
        progress.push(
          `Expanded query into ${event.queries.length} variants`,
        );
        for (const q of event.queries.slice(0, 4)) {
          logLines.push({ key: "Query variant", value: q });
        }
        if (event.queries.length > 4) {
          logLines.push({
            key: "Query variants",
            value: `+${event.queries.length - 4} more variants`,
          });
        }
        logLines.push({
          key: "New chunks found",
          value: `${event.newChunksFound} new (${event.totalUnique} total)`,
        });
        break;
      case "rerank":
        progress.push(
          `Re-ranked ${event.totalChunks} chunks, kept ${event.keptChunks}`,
        );
        logLines.push({
          key: "Re-ranking",
          value: `Kept ${event.keptChunks} of ${event.totalChunks} (dropped ${event.droppedChunks})`,
        });
        break;
      case "iterative_search":
        progress.push(
          `Iteration ${event.iteration}: found ${event.newChunksFound} new chunks`,
        );
        for (const q of event.newQueries.slice(0, 3)) {
          logLines.push({
            key: `Iteration ${event.iteration}`,
            value: q,
          });
        }
        break;
      case "classifier":
        progress.push(
          `Classified ${event.classified} chunks, kept ${event.kept}`,
        );
        logLines.push({
          key: "Classifier",
          value: `Kept ${event.kept} of ${event.classified} unranked chunks`,
        });
        break;
      case "complete":
        progress.push(
          `Complete: ${event.totalChunks} relevant chunks found`,
        );
        break;
    }
  };

  const ctx: LoopContext = {
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

  return { ctx, progress, logLines };
}

/**
 * Create AI SDK tools that expose the agentic retrieval loop.
 */
export function alphaloopTools(config: AlphaloopConfig) {
  return {
    deep_search: tool({
      description:
        "Search the knowledge base using an agentic retrieval loop with query expansion, LLM re-ranking, and iterative refinement. Returns the most relevant passages.",
      inputSchema: z.object({
        query: z.string().describe("The search query"),
        maxResults: z
          .number()
          .optional()
          .describe("Maximum results to return (default: 20)"),
      }),
      execute: async ({ query, maxResults }) => {
        const { ctx, progress, logLines } = createLoopContext(config);

        // Step 1: Initial embedding search
        const initialChunks = await embeddingSearch(query, ctx);

        // Step 2: Query expansion
        await queryExpansion(query, initialChunks, ctx);

        // Step 3: Re-rank all collected chunks
        const allChunks = Array.from(ctx.seenChunks.values());
        await rerank(query, allChunks, ctx, {
          sourceQuery: query,
          iteration: 0,
        });

        // Step 4: Iterative search
        await iterativeSearch(query, ctx);

        // Step 5: Optional classifier
        if (ctx.config.enableClassifier) {
          await classify(query, ctx);
        }

        // Collect results
        const finalChunks = Array.from(ctx.rankedChunks.values())
          .sort((a, b) => b.relevance - a.relevance)
          .slice(0, maxResults ?? 20);

        return {
          chunks: finalChunks.map((c) => ({
            id: c.id,
            text: c.text,
            relevance: c.relevance,
            rationale: c.rationale,
            metadata: c.metadata,
          })),
          totalConsidered: ctx.seenChunks.size,
          iterationsRun: ctx.iterations.length,
          __progress: progress,
          __logLines: logLines,
        };
      },
    }),
  };
}
