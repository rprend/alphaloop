import { generateObject } from "ai";
import { z } from "zod";
import { QUERY_EXPANSION_PROMPT } from "../defaults.js";
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

  const contextSnippets = initialChunks
    .slice(0, 5)
    .map((c) => c.text.slice(0, 200))
    .join("\n---\n");

  const { object: queries } = await generateObject({
    model: ctx.config.model,
    schema: z.object({
      queries: z.array(z.string()).describe("Array of expanded query strings"),
    }),
    system: systemPrompt,
    prompt: `Original query: "${originalQuery}"

${contextSnippets ? `Here are some initial results for context:\n${contextSnippets}` : "No initial results available."}

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
      ctx.triedQueries.add(query.toLowerCase().trim());
      return ctx.config.search(query, {
        topK: Math.ceil(ctx.config.initialTopK / 2),
      });
    }),
  );

  // Collect new chunks
  let newCount = 0;
  const allNewChunks: EmbeddingChunk[] = [];
  for (const results of allResults) {
    for (const chunk of results) {
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
  });

  return allNewChunks;
}
