import type { EmbeddingChunk, LoopContext } from "../types.js";

/**
 * Step 1: Initial embedding search (recall baseline).
 * Searches the user's embeddings with the original query and collects chunks.
 */
export async function embeddingSearch(
  query: string,
  ctx: LoopContext,
): Promise<EmbeddingChunk[]> {
  const results = await ctx.config.search(query, {
    topK: ctx.config.initialTopK,
  });

  let newCount = 0;
  for (const chunk of results) {
    if (!ctx.seenChunks.has(chunk.id)) {
      ctx.seenChunks.set(chunk.id, chunk);
      newCount++;
    }
  }

  ctx.triedQueries.add(query);

  ctx.emit({
    type: "embedding_search",
    query,
    chunksFound: newCount,
  });

  return results;
}
