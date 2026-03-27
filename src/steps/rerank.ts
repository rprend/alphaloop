import { generateObject } from "ai";
import { z } from "zod";
import {
  availablePromptTokens,
  estimateTokens,
  normalizeChunkForModel,
  shardByTokens,
} from "../context-budget.js";
import { RERANK_PROMPT } from "../defaults.js";
import type { EmbeddingChunk, LoopContext, RankedChunk } from "../types.js";

const RerankResponseSchema = z.object({
  results: z.array(
    z.object({
      id: z.string(),
      relevance: z.number().min(0).max(1),
      rationale: z.string().nullable(),
    }),
  ),
});

/**
 * Step 3: LLM re-ranking.
 * Batches chunks to the LLM for relevance scoring, filters below threshold.
 */
export async function rerank(
  query: string,
  chunks: EmbeddingChunk[],
  ctx: LoopContext,
  options?: { sourceQuery?: string; iteration?: number },
  depth = 1,
): Promise<RankedChunk[]> {
  const model = ctx.config.rerankModel ?? ctx.config.model;
  const systemPrompt = ctx.config.rerankPrompt ?? RERANK_PROMPT;
  ctx.recursionDepth = Math.max(ctx.recursionDepth, depth);
  ctx.emit({
    type: "phase",
    label: `Preparing ${chunks.length} chunks for re-ranking`,
  });

  const allRanked: RankedChunk[] = [];
  const prepared = await Promise.all(
    chunks.map(async (chunk) => ({
      chunk,
      promptText: await normalizeChunkForModel(chunk, query, ctx, depth + 1),
    })),
  );

  const combinedTokens = prepared.reduce(
    (total, item) => total + estimateTokens(item.promptText, ctx),
    0,
  );
  const shards =
    combinedTokens <= availablePromptTokens(ctx)
      ? [prepared]
      : shardByTokens(prepared, (item) => item.promptText, ctx);
  ctx.shardCount += shards.length;

  const results = await Promise.all(
    shards.map(async (shard) => {
      const passageList = shard
        .map((item, idx) => `[${idx}] (id: ${item.chunk.id})\n${item.promptText}`)
        .join("\n\n");

      const { object } = await generateObject({
        model,
        schema: RerankResponseSchema,
        system: systemPrompt,
        prompt: `Search query: "${query}"

Score each passage for relevance to the query:

${passageList}`,
        abortSignal: ctx.config.signal,
      });

      return object.results.map((r) => {
        const original =
          shard.find((item) => item.chunk.id === r.id)?.chunk ?? shard[0].chunk;
        return {
          ...original,
          relevance: r.relevance,
          rationale: r.rationale ?? undefined,
          sourceQuery: options?.sourceQuery,
          iteration: options?.iteration,
        } satisfies RankedChunk;
      });
    }),
  );

  for (const batch of results) {
    allRanked.push(...batch);
  }

  // Filter by threshold and sort
  const kept = allRanked.filter(
    (c) => c.relevance >= ctx.config.relevanceThreshold,
  );
  kept.sort((a, b) => b.relevance - a.relevance);

  // Update ranked chunks map (keep best score per chunk)
  for (const chunk of kept) {
    const existing = ctx.rankedChunks.get(chunk.id);
    if (!existing || chunk.relevance > existing.relevance) {
      ctx.rankedChunks.set(chunk.id, chunk);
    }
  }

  ctx.emit({
    type: "rerank",
    totalChunks: chunks.length,
    keptChunks: kept.length,
    droppedChunks: chunks.length - kept.length,
    topChunkPreview: kept[0]?.text.slice(0, 100),
    shardCount: shards.length,
    recursionDepth: ctx.recursionDepth,
  });

  return kept;
}
