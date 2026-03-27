import type { EmbeddingChunk, LoopContext } from "./types.js";

export async function collectStrongMatches(
  query: string,
  ctx: LoopContext,
): Promise<{ chunks: EmbeddingChunk[]; matched: number; requests: number }> {
  if (typeof ctx.config.search === "function") {
    return collectPagedMatches(query, ctx, ctx.config.search);
  }

  if (typeof ctx.config.searchStream !== "function") {
    throw new Error("Alphaloop requires either `search` or `searchStream`.");
  }

  return collectStreamMatches(query, ctx, ctx.config.searchStream);
}

async function collectPagedMatches(
  query: string,
  ctx: LoopContext,
  search: NonNullable<LoopContext["config"]["search"]>,
): Promise<{ chunks: EmbeddingChunk[]; matched: number; requests: number }> {
  const chunks: EmbeddingChunk[] = [];
  let matched = 0;
  let requests = 0;
  let cursor: string | undefined;

  while (true) {
    const page = await search(query, {
      minScore: ctx.config.minScore,
      cursor,
      signal: ctx.config.signal,
    });
    requests++;

    const filtered = page.chunks.filter((chunk) => chunk.score >= ctx.config.minScore);
    matched += filtered.length;
    chunks.push(...filtered);

    if (!page.nextCursor || filtered.length !== page.chunks.length) {
      break;
    }

    cursor = page.nextCursor;
  }

  return { chunks, matched, requests };
}

async function collectStreamMatches(
  query: string,
  ctx: LoopContext,
  searchStream: NonNullable<LoopContext["config"]["searchStream"]>,
): Promise<{ chunks: EmbeddingChunk[]; matched: number; requests: number }> {
  const chunks: EmbeddingChunk[] = [];
  let matched = 0;
  let requests = 0;

  for await (const chunk of searchStream(query, {
    minScore: ctx.config.minScore,
    signal: ctx.config.signal,
  })) {
    requests++;

    if (chunk.score < ctx.config.minScore) {
      continue;
    }

    matched++;
    chunks.push(chunk);
  }

  return { chunks, matched, requests };
}
