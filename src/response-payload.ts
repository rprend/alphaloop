import { availablePromptTokens, estimateTokens, reduceChunksToSummary } from "./context-budget.js";
import type { LoopContext, RankedChunk } from "./types.js";

export async function buildToolPayload(
  query: string,
  chunks: RankedChunk[],
  ctx: LoopContext,
): Promise<{
  chunks: Array<{
    id: string;
    text?: string;
    relevance: number;
    rationale?: string;
    metadata?: Record<string, unknown>;
  }>;
  evidenceSummary?: string;
}> {
  const rawPayloadTokens = chunks.reduce((total, chunk) => {
    return total + estimateTokens(chunk.text, ctx);
  }, 0);

  if (rawPayloadTokens <= availablePromptTokens(ctx, 3_000)) {
    return {
      chunks: chunks.map((c) => ({
        id: c.id,
        text: c.text,
        relevance: c.relevance,
        rationale: c.rationale,
        metadata: c.metadata,
      })),
    };
  }

  const evidenceSummary = await reduceChunksToSummary(
    "final evidence",
    query,
    chunks,
    ctx,
    "Build an answer-ready evidence summary that preserves all key facts, claims, and chunk IDs.",
  );

  return {
    chunks: chunks.map((c) => ({
      id: c.id,
      relevance: c.relevance,
      rationale: c.rationale,
      metadata: c.metadata,
    })),
    evidenceSummary,
  };
}
