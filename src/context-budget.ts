import { generateObject } from "ai";
import { z } from "zod";
import type { EmbeddingChunk, LoopContext, RankedChunk } from "./types.js";

const DEFAULT_OUTPUT_RESERVE = 4_000;
const DEFAULT_PROMPT_OVERHEAD = 1_500;

const SummarySchema = z.object({
  summary: z.string(),
});

export function estimateTokens(text: string, ctx: LoopContext): number {
  const estimator = ctx.config.tokenEstimator;
  if (estimator) {
    return Math.max(1, estimator(text));
  }

  return Math.max(1, Math.ceil(text.length / 4));
}

export function availablePromptTokens(
  ctx: LoopContext,
  overhead = DEFAULT_PROMPT_OVERHEAD,
): number {
  return Math.max(1_000, ctx.config.maxContextTokens - DEFAULT_OUTPUT_RESERVE - overhead);
}

export function shardByTokens<T>(
  items: T[],
  getText: (item: T) => string,
  ctx: LoopContext,
  overhead = DEFAULT_PROMPT_OVERHEAD,
): T[][] {
  const maxTokens = availablePromptTokens(ctx, overhead);
  const shards: T[][] = [];
  let current: T[] = [];
  let currentTokens = 0;

  for (const item of items) {
    const itemTokens = estimateTokens(getText(item), ctx);

    if (current.length > 0 && currentTokens + itemTokens > maxTokens) {
      shards.push(current);
      current = [];
      currentTokens = 0;
    }

    current.push(item);
    currentTokens += itemTokens;
  }

  if (current.length > 0) {
    shards.push(current);
  }

  return shards.length > 0 ? shards : [items];
}

export async function summarizeTextRecursively(
  label: string,
  text: string,
  query: string,
  ctx: LoopContext,
  depth = 1,
): Promise<string> {
  ctx.recursionDepth = Math.max(ctx.recursionDepth, depth);

  if (estimateTokens(text, ctx) <= availablePromptTokens(ctx)) {
    return text;
  }

  const maxSegmentTokens = Math.max(4_000, Math.floor(availablePromptTokens(ctx) / 2));
  const segments = splitTextByTokens(text, maxSegmentTokens, ctx);
  ctx.shardCount += segments.length;

  const summaries = await Promise.all(
    segments.map(async (segment, index) => {
      const normalized = await summarizeTextRecursively(
        `${label} segment ${index + 1}`,
        segment,
        query,
        ctx,
        depth + 1,
      );

      const { object } = await generateObject({
        model: ctx.config.rerankModel ?? ctx.config.model,
        schema: SummarySchema,
        system:
          "You condense text without dropping any salient facts, concepts, or named identifiers that could affect retrieval or relevance judgment.",
        prompt: `Query: "${query}"

Label: ${label}

Produce a compact but information-complete summary of this text:

${normalized}`,
        abortSignal: ctx.config.signal,
      });

      return object.summary;
    }),
  );

  return summarizeTextRecursively(
    `${label} merged`,
    summaries.join("\n\n"),
    query,
    ctx,
    depth + 1,
  );
}

export async function normalizeChunkForModel(
  chunk: EmbeddingChunk | RankedChunk,
  query: string,
  ctx: LoopContext,
  depth = 1,
): Promise<string> {
  const prefix = `Chunk ID: ${chunk.id}\nScore: ${chunk.score}`;
  const textBudget = availablePromptTokens(ctx, DEFAULT_PROMPT_OVERHEAD + estimateTokens(prefix, ctx));

  if (estimateTokens(chunk.text, ctx) <= textBudget) {
    return `${prefix}\nText:\n${chunk.text}`;
  }

  const distilled = await summarizeTextRecursively(
    `chunk ${chunk.id}`,
    chunk.text,
    query,
    ctx,
    depth + 1,
  );
  return `${prefix}\nText:\n${distilled}`;
}

export async function reduceChunksToSummary(
  label: string,
  query: string,
  chunks: Array<EmbeddingChunk | RankedChunk>,
  ctx: LoopContext,
  promptTail: string,
  depth = 1,
): Promise<string> {
  ctx.recursionDepth = Math.max(ctx.recursionDepth, depth);

  const normalized = await Promise.all(
    chunks.map((chunk) => normalizeChunkForModel(chunk, query, ctx, depth + 1)),
  );
  const combined = normalized.join("\n\n---\n\n");

  if (estimateTokens(combined, ctx) <= availablePromptTokens(ctx)) {
    return combined;
  }

  const shards = shardByTokens(normalized, (item) => item, ctx);
  ctx.shardCount += shards.length;

  const reduced = await Promise.all(
    shards.map(async (shard, index) => {
      const shardText = shard.join("\n\n---\n\n");
      const { object } = await generateObject({
        model: ctx.config.model,
        schema: SummarySchema,
        system:
          "You compress evidence while preserving every salient concept, citation handle, and distinction needed for later retrieval or synthesis.",
        prompt: `Query: "${query}"

Context label: ${label} shard ${index + 1}

${promptTail}

${shardText}`,
        abortSignal: ctx.config.signal,
      });

      return object.summary;
    }),
  );

  return summarizeTextRecursively(label, reduced.join("\n\n"), query, ctx, depth + 1);
}

function splitTextByTokens(
  text: string,
  maxTokens: number,
  ctx: LoopContext,
): string[] {
  if (estimateTokens(text, ctx) <= maxTokens) {
    return [text];
  }

  const approxCharsPerToken = Math.max(1, Math.ceil(text.length / estimateTokens(text, ctx)));
  const maxChars = Math.max(1_000, maxTokens * approxCharsPerToken);
  const segments: string[] = [];

  for (let start = 0; start < text.length; start += maxChars) {
    segments.push(text.slice(start, start + maxChars));
  }

  return segments;
}
