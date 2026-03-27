import { collectStrongMatches } from "../search-adapter.js";
import type { EmbeddingChunk, LoopContext } from "../types.js";

/**
 * Step 1: Initial embedding search (recall baseline).
 * Searches the user's embeddings with the original query and collects chunks.
 */
export async function embeddingSearch(
  query: string,
  ctx: LoopContext,
): Promise<EmbeddingChunk[]> {
  const { chunks: results, matched, requests } = await collectStrongMatches(
    query,
    ctx,
  );

  let newCount = 0;
  for (const chunk of results) {
    if (!ctx.seenChunks.has(chunk.id)) {
      ctx.seenChunks.set(chunk.id, chunk);
      newCount++;
    }
  }

  ctx.triedQueries.add(query.toLowerCase().trim());
  ctx.totalChunksMatched += matched;
  ctx.retrievalRequests += requests;

  ctx.emit({
    type: "embedding_search",
    query,
    chunksFound: newCount,
    chunksMatched: matched,
    pagesFetched: requests,
    minScore: ctx.config.minScore,
    topK: ctx.config.topK,
  });

  return results;
}
