import { estimateTokens } from "./context-budget.js";
import type {
  CorpusDocument,
  EmbeddingChunk,
  GrepCorpusMatch,
  LoopContext,
  SearchObservation,
  SessionSnapshot,
} from "./types.js";

const DEFAULT_SNIPPET_TOKENS = 500;

function truncateToTokenBudget(
  text: string,
  tokenBudget: number,
  ctx: LoopContext,
): string {
  if (tokenBudget <= 0) {
    return "";
  }

  if (estimateTokens(text, ctx) <= tokenBudget) {
    return text;
  }

  const approxCharsPerToken = Math.max(
    1,
    Math.ceil(text.length / estimateTokens(text, ctx)),
  );
  const maxChars = Math.max(80, tokenBudget * approxCharsPerToken);
  return `${text.slice(0, maxChars).trimEnd()}...`;
}

export function refreshTokenUsage(ctx: LoopContext): number {
  const visibleChunks = Array.from(ctx.visibleChunkIds)
    .map((id) => ctx.seenChunks.get(id))
    .filter((chunk): chunk is EmbeddingChunk => Boolean(chunk));
  const visibleChunkTokens = visibleChunks.reduce(
    (total, chunk) => total + estimateTokens(chunk.text, ctx),
    0,
  );
  const visibleDocumentTokens = Array.from(ctx.visibleDocuments.values()).reduce(
    (total, doc) => total + estimateTokens(doc.text, ctx),
    0,
  );
  const observationTokens = ctx.observations.reduce(
    (total, observation) => total + estimateTokens(observation.summary, ctx),
    0,
  );

  ctx.tokenUsage = visibleChunkTokens + visibleDocumentTokens + observationTokens;
  return ctx.tokenUsage;
}

export function getSessionSnapshot(ctx: LoopContext): SessionSnapshot {
  refreshTokenUsage(ctx);
  const remainingTokens = Math.max(0, ctx.config.maxContextTokens - ctx.tokenUsage);
  const status =
    ctx.tokenUsage >= ctx.hardTokenLimit
      ? "hard_limit"
      : ctx.tokenUsage >= ctx.softTokenLimit
        ? "soft_limit"
        : "ok";

  return {
    tokenUsage: ctx.tokenUsage,
    maxTokens: ctx.config.maxContextTokens,
    softLimit: ctx.softTokenLimit,
    hardLimit: ctx.hardTokenLimit,
    remainingTokens,
    status,
    visibleChunkCount: ctx.visibleChunkIds.size,
    visibleDocumentCount: ctx.visibleDocuments.size,
    encounteredChunkCount: ctx.seenChunks.size,
  };
}

export function makeTokenPressureNotice(ctx: LoopContext): string | undefined {
  const snapshot = getSessionSnapshot(ctx);
  if (snapshot.status === "hard_limit") {
    return `Visible context is at the hard cutoff (${snapshot.tokenUsage}/${snapshot.maxTokens}). Only prune_chunks should be used until space is freed.`;
  }

  if (snapshot.status === "soft_limit") {
    return `Visible context is above the soft limit (${snapshot.tokenUsage}/${snapshot.maxTokens}). Prefer prune_chunks or conclude soon.`;
  }

  return undefined;
}

export function canUseContextTool(ctx: LoopContext, toolName: string): {
  allowed: boolean;
  reason?: string;
} {
  if (toolName === "prune_chunks") {
    return { allowed: true };
  }

  const snapshot = getSessionSnapshot(ctx);
  if (snapshot.status !== "hard_limit") {
    return { allowed: true };
  }

  return {
    allowed: false,
    reason: `Tool ${toolName} is blocked because visible context is at the hard cutoff (${snapshot.tokenUsage}/${snapshot.maxTokens}). Prune chunks or finish answering.`,
  };
}

export function appendObservation(
  ctx: LoopContext,
  observation: Omit<SearchObservation, "id">,
): SearchObservation {
  const entry: SearchObservation = {
    id: `${observation.type}-${ctx.observations.length + 1}`,
    ...observation,
  };
  ctx.observations.push(entry);
  refreshTokenUsage(ctx);
  return entry;
}

export function addVisibleChunks(
  ctx: LoopContext,
  chunks: EmbeddingChunk[],
  tokenBudget?: number,
): EmbeddingChunk[] {
  const accepted: EmbeddingChunk[] = [];
  let remaining = tokenBudget ?? Number.POSITIVE_INFINITY;

  for (const chunk of chunks) {
    if (ctx.visibleChunkIds.has(chunk.id)) {
      continue;
    }

    const chunkTokens = estimateTokens(chunk.text, ctx);
    if (
      accepted.length > 0 &&
      remaining !== Number.POSITIVE_INFINITY &&
      chunkTokens > remaining
    ) {
      break;
    }

    ctx.visibleChunkIds.add(chunk.id);
    accepted.push(chunk);
    if (remaining !== Number.POSITIVE_INFINITY) {
      remaining -= chunkTokens;
    }
  }

  refreshTokenUsage(ctx);
  return accepted;
}

export function pruneVisibleChunks(
  ctx: LoopContext,
  chunkIds: string[],
): { removed: string[]; missing: string[] } {
  const removed: string[] = [];
  const missing: string[] = [];

  for (const chunkId of chunkIds) {
    if (ctx.visibleChunkIds.delete(chunkId)) {
      removed.push(chunkId);
    } else {
      missing.push(chunkId);
    }
  }

  const visibleDocIds = new Set(
    Array.from(ctx.visibleChunkIds)
      .map((chunkId) => ctx.seenChunks.get(chunkId))
      .flatMap((chunk) => {
        const docId = getChunkDocumentId(chunk);
        return docId ? [docId] : [];
      }),
  );

  for (const docId of Array.from(ctx.visibleDocuments.keys())) {
    if (!visibleDocIds.has(docId)) {
      ctx.visibleDocuments.delete(docId);
    }
  }

  refreshTokenUsage(ctx);
  return { removed, missing };
}

export function getChunkDocumentId(
  chunk: EmbeddingChunk | undefined,
): string | undefined {
  const metadata = chunk?.metadata;
  if (!metadata) {
    return undefined;
  }

  const documentId = metadata.documentId ?? metadata.docId ?? metadata.sourceId;
  return typeof documentId === "string" ? documentId : undefined;
}

export function collectEncounteredDocumentChunks(
  ctx: LoopContext,
  docId: string,
): EmbeddingChunk[] {
  return Array.from(ctx.seenChunks.values()).filter(
    (chunk) => getChunkDocumentId(chunk) === docId,
  );
}

export function addVisibleDocument(
  ctx: LoopContext,
  document: CorpusDocument,
  tokenBudget?: number,
): CorpusDocument {
  const doc = {
    ...document,
    text:
      tokenBudget == null
        ? document.text
        : truncateToTokenBudget(document.text, tokenBudget, ctx),
  };
  ctx.visibleDocuments.set(doc.id, doc);
  if (doc.chunks) {
    addVisibleChunks(ctx, doc.chunks, tokenBudget);
  } else {
    refreshTokenUsage(ctx);
  }
  return doc;
}

export function buildChunkResultPayload(
  ctx: LoopContext,
  chunks: EmbeddingChunk[],
): Array<{
  id: string;
  text: string;
  score: number;
  metadata?: Record<string, unknown>;
}> {
  return chunks.map((chunk) => ({
    id: chunk.id,
    text: truncateToTokenBudget(chunk.text, DEFAULT_SNIPPET_TOKENS, ctx),
    score: chunk.score,
    metadata: chunk.metadata,
  }));
}

export function buildGrepPayload(
  ctx: LoopContext,
  matches: GrepCorpusMatch[],
): Array<{
  chunkId: string;
  docId?: string;
  snippet: string;
  metadata?: Record<string, unknown>;
}> {
  return matches.map((match) => ({
    chunkId: match.chunkId,
    docId: match.docId,
    snippet: truncateToTokenBudget(match.snippet, DEFAULT_SNIPPET_TOKENS, ctx),
    metadata: match.metadata,
  }));
}
