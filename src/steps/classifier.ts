import { generateObject } from "ai";
import { z } from "zod";
import {
  availablePromptTokens,
  estimateTokens,
  normalizeChunkForModel,
  shardByTokens,
} from "../context-budget.js";
import { CLASSIFIER_PROMPT } from "../defaults.js";
import type { EmbeddingChunk, LoopContext, RankedChunk } from "../types.js";

const ClassifyResponseSchema = z.object({
  relevant: z.boolean(),
  confidence: z.number().min(0).max(1),
  rationale: z.string().nullable(),
});

/**
 * Step 5 (optional): Abstract concept classifier.
 * Uses the LLM to classify chunks that may be implicitly related to abstract concepts.
 * Catches things too subtle for embedding similarity.
 */
export async function classify(
  query: string,
  ctx: LoopContext,
): Promise<void> {
  // Get chunks that haven't been ranked yet (borderline/missed by re-ranker)
  const unranked: EmbeddingChunk[] = [];
  for (const [id, chunk] of ctx.seenChunks) {
    if (!ctx.rankedChunks.has(id)) {
      unranked.push(chunk);
    }
  }

  if (unranked.length === 0) {
    ctx.emit({ type: "classifier", classified: 0, kept: 0, dropped: 0 });
    return;
  }

  const prepared = await Promise.all(
    unranked.map(async (chunk) => ({
      chunk,
      promptText: await normalizeChunkForModel(chunk, query, ctx),
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

  let kept = 0;
  let dropped = 0;

  const results = await Promise.all(
    shards.flatMap((shard) =>
      shard.map(async ({ chunk, promptText }) => {
        const { object } = await generateObject({
          model: ctx.config.rerankModel ?? ctx.config.model,
          schema: ClassifyResponseSchema,
          system: CLASSIFIER_PROMPT,
          prompt: `Concept: "${query}"

Passage:
${promptText}

Is this passage relevant to the concept above?`,
          abortSignal: ctx.config.signal,
        });

        return { chunk, ...object };
      }),
    ),
  );

  for (const result of results) {
    if (result.relevant && result.confidence >= 0.5) {
      const ranked: RankedChunk = {
        ...result.chunk,
        relevance: result.confidence * 0.8,
        rationale: result.rationale ?? undefined,
        sourceQuery: "classifier",
      };
      ctx.rankedChunks.set(ranked.id, ranked);
      kept++;
    } else {
      dropped++;
    }
  }

  ctx.emit({
    type: "classifier",
    classified: unranked.length,
    kept,
    dropped,
    shardCount: shards.length,
    recursionDepth: ctx.recursionDepth,
  });
}
