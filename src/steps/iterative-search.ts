import { generateObject } from "ai";
import { z } from "zod";
import { reduceChunksToSummary } from "../context-budget.js";
import { ITERATIVE_SEARCH_PROMPT } from "../defaults.js";
import { collectStrongMatches } from "../search-adapter.js";
import type { LoopContext } from "../types.js";
import { rerank } from "./rerank.js";

/**
 * Step 4: Iterative search (concept expansion).
 * Feeds top-ranked passages back to LLM to discover new search angles,
 * then searches and re-ranks the new results.
 */
export async function iterativeSearch(
  originalQuery: string,
  ctx: LoopContext,
): Promise<void> {
  for (
    let iteration = 1;
    iteration <= ctx.config.maxIterations;
    iteration++
  ) {
    // Get current top-ranked chunks to feed back
    const topChunks = Array.from(ctx.rankedChunks.values())
      .sort((a, b) => b.relevance - a.relevance)
      .slice(0, Math.max(10, ctx.rankedChunks.size));

    if (topChunks.length === 0) break;

    const passageSummaries = await reduceChunksToSummary(
      `ranked evidence iteration ${iteration}`,
      originalQuery,
      topChunks,
      ctx,
      "Compress these ranked passages into an evidence-complete concept summary for generating follow-up searches.",
    );

    const count = Math.max(3, Math.ceil(ctx.config.maxExpandedQueries / 2));
    const systemPrompt = ITERATIVE_SEARCH_PROMPT.replace(
      "{count}",
      String(count),
    );

    const { object } = await generateObject({
      model: ctx.config.model,
      schema: z.object({
        queries: z.array(z.string()),
      }),
      system: systemPrompt,
      prompt: `Original query: "${originalQuery}"

Top relevant passages found so far:
${passageSummaries}

Generate ${count} NEW search queries that explore concepts discovered in these passages but not covered by the original query.`,
      abortSignal: ctx.config.signal,
    });

    // Filter already-tried queries
    const newQueries = object.queries.filter((q) => {
      const normalized = q.toLowerCase().trim();
      return !ctx.triedQueries.has(normalized);
    });

    if (newQueries.length === 0) {
      ctx.iterations.push({
        iteration,
        newQueries: [],
        chunksFound: 0,
        totalUniqueChunks: ctx.seenChunks.size,
      });
      break; // No new angles to explore
    }

    // Search with new queries
    const allResults = await Promise.all(
      newQueries.map(async (query) => {
        const normalized = query.toLowerCase().trim();
        ctx.triedQueries.add(normalized);
        return collectStrongMatches(query, ctx);
      }),
    );

    // Collect only truly new chunks
    const newChunks = [];
    let matched = 0;
    let requests = 0;
    for (const results of allResults) {
      matched += results.matched;
      requests += results.requests;
      for (const chunk of results.chunks) {
        if (!ctx.seenChunks.has(chunk.id)) {
          ctx.seenChunks.set(chunk.id, chunk);
          newChunks.push(chunk);
        }
      }
    }
    ctx.totalChunksMatched += matched;
    ctx.retrievalRequests += requests;

    // Re-rank the new chunks
    if (newChunks.length > 0) {
      await rerank(originalQuery, newChunks, ctx, {
        sourceQuery: newQueries.join(", "),
        iteration,
      });
    }

    ctx.iterations.push({
      iteration,
      newQueries,
      chunksFound: newChunks.length,
      totalUniqueChunks: ctx.seenChunks.size,
    });

    ctx.emit({
      type: "iterative_search",
      iteration,
      newQueries,
      newChunksFound: newChunks.length,
      totalUnique: ctx.seenChunks.size,
      shardCount: ctx.shardCount,
      recursionDepth: ctx.recursionDepth,
    });

    // Stop early if no new chunks found
    if (newChunks.length === 0) break;
  }
}
