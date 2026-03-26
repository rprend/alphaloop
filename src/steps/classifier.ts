import { generateObject } from "ai";
import { z } from "zod";
import { CLASSIFIER_PROMPT } from "../defaults.js";
import type { EmbeddingChunk, LoopContext, RankedChunk } from "../types.js";

const ClassifyResponseSchema = z.object({
  relevant: z.boolean(),
  confidence: z.number().min(0).max(1),
  rationale: z.string().optional(),
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

  // Classify in parallel batches
  const concurrency = 10;
  let kept = 0;
  let dropped = 0;

  for (let i = 0; i < unranked.length; i += concurrency) {
    const batch = unranked.slice(i, i + concurrency);

    const results = await Promise.all(
      batch.map(async (chunk) => {
        const { object } = await generateObject({
          model: ctx.config.rerankModel ?? ctx.config.model,
          schema: ClassifyResponseSchema,
          system: CLASSIFIER_PROMPT,
          prompt: `Concept: "${query}"

Passage:
${chunk.text.slice(0, 800)}

Is this passage relevant to the concept above?`,
          abortSignal: ctx.config.signal,
        });

        return { chunk, ...object };
      }),
    );

    for (const result of results) {
      if (result.relevant && result.confidence >= 0.5) {
        const ranked: RankedChunk = {
          ...result.chunk,
          relevance: result.confidence * 0.8, // Slight discount vs. re-ranked
          rationale: result.rationale,
          sourceQuery: "classifier",
        };
        ctx.rankedChunks.set(ranked.id, ranked);
        kept++;
      } else {
        dropped++;
      }
    }
  }

  ctx.emit({
    type: "classifier",
    classified: unranked.length,
    kept,
    dropped,
  });
}
