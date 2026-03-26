import { generateObject } from "ai";
import { z } from "zod";
import { RERANK_PROMPT } from "../defaults.js";
import type { EmbeddingChunk, LoopContext, RankedChunk } from "../types.js";

const BATCH_SIZE = 20;

const RerankResponseSchema = z.object({
  results: z.array(
    z.object({
      id: z.string(),
      relevance: z.number().min(0).max(1),
      rationale: z.string().optional(),
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
): Promise<RankedChunk[]> {
  const model = ctx.config.rerankModel ?? ctx.config.model;
  const systemPrompt = ctx.config.rerankPrompt ?? RERANK_PROMPT;

  // Batch chunks for re-ranking
  const batches: EmbeddingChunk[][] = [];
  for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
    batches.push(chunks.slice(i, i + BATCH_SIZE));
  }

  const allRanked: RankedChunk[] = [];

  // Process batches in parallel (up to 5 concurrent)
  const concurrency = 5;
  for (let i = 0; i < batches.length; i += concurrency) {
    const batchSlice = batches.slice(i, i + concurrency);

    const results = await Promise.all(
      batchSlice.map(async (batch) => {
        const passageList = batch
          .map(
            (c, idx) =>
              `[${idx}] (id: ${c.id})\n${c.text.slice(0, 500)}`,
          )
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
          const original = batch.find((c) => c.id === r.id) ?? batch[0];
          return {
            ...original,
            relevance: r.relevance,
            rationale: r.rationale,
            sourceQuery: options?.sourceQuery,
            iteration: options?.iteration,
          } satisfies RankedChunk;
        });
      }),
    );

    for (const batch of results) {
      allRanked.push(...batch);
    }
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
  });

  return kept;
}
