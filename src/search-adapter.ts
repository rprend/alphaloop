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
  const topK = ctx.config.topK;

  while (true) {
    const page = await search(query, {
      minScore: ctx.config.minScore,
      topK,
      cursor,
      signal: ctx.config.signal,
    });
    requests++;

    const filtered = page.chunks.filter((chunk) =>
      ctx.config.minScore == null ? true : chunk.score >= ctx.config.minScore,
    );
    const remaining =
      topK == null ? filtered.length : Math.max(topK - chunks.length, 0);
    const accepted = topK == null ? filtered : filtered.slice(0, remaining);
    matched += accepted.length;
    chunks.push(...accepted);

    if (
      (topK != null && chunks.length >= topK) ||
      !page.nextCursor ||
      filtered.length !== page.chunks.length
    ) {
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
  const topK = ctx.config.topK;

  for await (const chunk of searchStream(query, {
    minScore: ctx.config.minScore,
    topK,
    signal: ctx.config.signal,
  })) {
    requests++;

    if (ctx.config.minScore != null && chunk.score < ctx.config.minScore) {
      continue;
    }

    matched++;
    chunks.push(chunk);

    if (topK != null && chunks.length >= topK) {
      break;
    }
  }

  return { chunks, matched, requests };
}
