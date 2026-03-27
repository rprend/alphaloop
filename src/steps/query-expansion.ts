import { generateObject } from "ai";
import { z } from "zod";
import { reduceChunksToSummary } from "../context-budget.js";
import { QUERY_EXPANSION_PROMPT } from "../defaults.js";
import { collectStrongMatches } from "../search-adapter.js";
import type { EmbeddingChunk, LoopContext } from "../types.js";

/**
 * Step 2: Query expansion.
 * Uses the LLM to generate diverse query variants, then runs them in parallel.
 */
export async function queryExpansion(
  originalQuery: string,
  initialChunks: EmbeddingChunk[],
  ctx: LoopContext,
): Promise<EmbeddingChunk[]> {
  const count = ctx.config.maxExpandedQueries;
  const systemPrompt = (
    ctx.config.queryExpansionPrompt ?? QUERY_EXPANSION_PROMPT
  ).replace("{count}", String(count));

  const contextSummary =
    initialChunks.length > 0
      ? await reduceChunksToSummary(
          "initial strong matches",
          originalQuery,
          initialChunks,
          ctx,
          "Preserve every retrieval-relevant concept, phrasing variation, and named entity from these matches.",
        )
      : "";

  const { object: queries } = await generateObject({
    model: ctx.config.model,
    schema: z.object({
      queries: z.array(z.string()).describe("Array of expanded query strings"),
    }),
    system: systemPrompt,
    prompt: `Original query: "${originalQuery}"

${contextSummary ? `Here are the distilled strong matches for context:\n${contextSummary}` : "No initial results available."}

Generate ${count} diverse query variants.`,
    abortSignal: ctx.config.signal,
  });

  // Filter out queries we've already tried
  const newQueries = queries.queries.filter((q) => {
    const normalized = q.toLowerCase().trim();
    return !ctx.triedQueries.has(normalized);
  });

  // Run all new queries in parallel
  const allResults = await Promise.all(
    newQueries.map(async (query) => {
      const normalized = query.toLowerCase().trim();
      ctx.triedQueries.add(normalized);
      return collectStrongMatches(query, ctx);
    }),
  );

  // Collect new chunks
  let newCount = 0;
  let matched = 0;
  let requests = 0;
  const allNewChunks: EmbeddingChunk[] = [];
  for (const results of allResults) {
    matched += results.matched;
    requests += results.requests;
    for (const chunk of results.chunks) {
      if (!ctx.seenChunks.has(chunk.id)) {
        ctx.seenChunks.set(chunk.id, chunk);
        allNewChunks.push(chunk);
        newCount++;
      }
    }
  }

  ctx.emit({
    type: "query_expansion",
    queries: newQueries,
    newChunksFound: newCount,
    totalUnique: ctx.seenChunks.size,
    shardCount: ctx.shardCount,
    recursionDepth: ctx.recursionDepth,
  });

  ctx.totalChunksMatched += matched;
  ctx.retrievalRequests += requests;

  return allNewChunks;
}
