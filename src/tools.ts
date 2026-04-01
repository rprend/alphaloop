import { tool } from "ai";
import { z } from "zod";
import { buildToolPayload } from "./response-payload.js";
import { createLoopContext } from "./loop-context.js";
import {
  addVisibleChunks,
  addVisibleDocument,
  appendObservation,
  buildChunkResultPayload,
  buildGrepPayload,
  canUseContextTool,
  collectEncounteredDocumentChunks,
  getChunkDocumentId,
  getSessionSnapshot,
  makeTokenPressureNotice,
  pruneVisibleChunks,
  refreshTokenUsage,
} from "./session-state.js";
import { collectStrongMatches } from "./search-adapter.js";
import { embeddingSearch } from "./steps/embedding-search.js";
import { queryExpansion } from "./steps/query-expansion.js";
import { rerank } from "./steps/rerank.js";
import { iterativeSearch } from "./steps/iterative-search.js";
import { classify } from "./steps/classifier.js";
import type {
  AlphaloopConfig,
  AlphaloopRunOptions,
  AlphaloopStreamEvent,
  CorpusDocument,
  GrepCorpusMatch,
  LoopContext,
} from "./types.js";

function applyRunOptions(
  ctx: LoopContext,
  config: AlphaloopConfig,
  options: AlphaloopRunOptions = {},
) {
  ctx.config.minScore = options.minScore ?? config.minScore ?? ctx.config.minScore;
  ctx.config.topK = options.topK ?? config.topK;
  ctx.config.maxContextTokens =
    options.maxContextTokens ?? config.maxContextTokens ?? ctx.config.maxContextTokens;
  ctx.softTokenLimit = Math.floor(
    ctx.config.maxContextTokens * ctx.config.softContextLimitRatio,
  );
  ctx.hardTokenLimit = Math.floor(
    ctx.config.maxContextTokens * ctx.config.hardContextLimitRatio,
  );
  refreshTokenUsage(ctx);
}

function createToolSession(
  config: AlphaloopConfig,
  onEvent?: (event: AlphaloopStreamEvent) => void,
): {
  ctx: LoopContext;
  progress: string[];
  logLines: Array<{ key: string; value: string }>;
} {
  const progress: string[] = [];
  const logLines: Array<{ key: string; value: string }> = [];

  const emit = (event: AlphaloopStreamEvent) => {
    onEvent?.(event);
    switch (event.type) {
      case "embedding_search":
        progress.push(`Searching embeddings for "${event.query}"`);
        logLines.push({
          key: "Initial search",
          value: `Found ${event.chunksFound} new of ${event.chunksMatched} matched${event.topK ? ` (topK ${event.topK})` : ""}`,
        });
        break;
      case "query_expansion":
        progress.push(`Expanded query into ${event.queries.length} variants`);
        for (const q of event.queries.slice(0, 4)) {
          logLines.push({ key: "Query variant", value: q });
        }
        if (event.queries.length > 4) {
          logLines.push({
            key: "Query variants",
            value: `+${event.queries.length - 4} more variants`,
          });
        }
        logLines.push({
          key: "New chunks found",
          value: `${event.newChunksFound} new (${event.totalUnique} total)`,
        });
        break;
      case "rerank":
        progress.push(`Re-ranked ${event.totalChunks} chunks, kept ${event.keptChunks}`);
        logLines.push({
          key: "Re-ranking",
          value: `Kept ${event.keptChunks} of ${event.totalChunks} (dropped ${event.droppedChunks})`,
        });
        break;
      case "iterative_search":
        progress.push(`Iteration ${event.iteration}: found ${event.newChunksFound} new chunks`);
        for (const q of event.newQueries.slice(0, 3)) {
          logLines.push({
            key: `Iteration ${event.iteration}`,
            value: q,
          });
        }
        break;
      case "classifier":
        progress.push(`Classified ${event.classified} chunks, kept ${event.kept}`);
        logLines.push({
          key: "Classifier",
          value: `Kept ${event.kept} of ${event.classified} unranked chunks`,
        });
        break;
      case "complete":
        progress.push(`Complete: ${event.totalChunks} relevant chunks found`);
        break;
      default:
        break;
    }
  };

  return {
    ctx: createLoopContext(config, {}, emit),
    progress,
    logLines,
  };
}

function getVisibleToolBudget(
  ctx: LoopContext,
  requestedTokenBudget?: number,
): number {
  const snapshot = getSessionSnapshot(ctx);
  const defaultBudget = Math.max(
    0,
    snapshot.remainingTokens - ctx.config.outputTokenReserve,
  );
  return Math.max(0, Math.min(requestedTokenBudget ?? defaultBudget, defaultBudget));
}

function serializeSnapshot(ctx: LoopContext) {
  const snapshot = getSessionSnapshot(ctx);
  return {
    ...snapshot,
    note: makeTokenPressureNotice(ctx),
    visibleChunkIds: Array.from(ctx.visibleChunkIds),
    visibleDocumentIds: Array.from(ctx.visibleDocuments.keys()),
    recentObservations: ctx.observations.slice(-6),
  };
}

function resolveDocument(
  ctx: LoopContext,
  docId: string,
  document: CorpusDocument | null,
): CorpusDocument | null {
  if (document) {
    return document;
  }

  const chunks = collectEncounteredDocumentChunks(ctx, docId);
  if (chunks.length === 0) {
    return null;
  }

  return {
    id: docId,
    text: chunks.map((chunk) => chunk.text).join("\n\n"),
    metadata: chunks[0]?.metadata,
    chunks,
  };
}

function fallbackGrep(
  ctx: LoopContext,
  pattern: string,
  limit: number,
): GrepCorpusMatch[] {
  const regex = new RegExp(pattern, "i");
  const matches: GrepCorpusMatch[] = [];

  for (const chunk of ctx.seenChunks.values()) {
    const found = chunk.text.match(regex);
    if (!found) {
      continue;
    }

    const start = Math.max(0, found.index ?? 0 - 80);
    const end = Math.min(chunk.text.length, (found.index ?? 0) + found[0].length + 120);
    matches.push({
      chunkId: chunk.id,
      docId: getChunkDocumentId(chunk),
      snippet: chunk.text.slice(start, end),
      metadata: chunk.metadata,
    });

    if (matches.length >= limit) {
      break;
    }
  }

  return matches;
}

export function alphaloopTools(config: AlphaloopConfig) {
  return alphaloopToolsWithSession(config);
}

export function alphaloopToolsWithSession(
  config: AlphaloopConfig,
  onEvent?: (event: AlphaloopStreamEvent) => void,
) {
  const { ctx, progress, logLines } = createToolSession(config, onEvent);

  return {
    search_corpus: tool({
      description:
        "Search the corpus for new chunks. This is the low-level retrieval action for iterative search sessions.",
      inputSchema: z.object({
        query: z.string().describe("The search query"),
        minScore: z.number().optional(),
        topK: z.number().int().positive().optional(),
        tokenBudget: z.number().int().positive().optional(),
      }),
      execute: async ({ query, minScore, topK, tokenBudget }) => {
        const gate = canUseContextTool(ctx, "search_corpus");
        if (!gate.allowed) {
          return { error: gate.reason, session: serializeSnapshot(ctx) };
        }

        applyRunOptions(ctx, config, { minScore, topK });
        const { chunks, matched, requests } = await collectStrongMatches(query, ctx);
        const newChunks = chunks.filter((chunk) => !ctx.seenChunks.has(chunk.id));
        for (const chunk of newChunks) {
          ctx.seenChunks.set(chunk.id, chunk);
        }
        ctx.triedQueries.add(query.toLowerCase().trim());
        ctx.totalChunksMatched += matched;
        ctx.retrievalRequests += requests;

        const added = addVisibleChunks(ctx, newChunks, getVisibleToolBudget(ctx, tokenBudget));
        appendObservation(ctx, {
          type: "search_corpus",
          summary: `search_corpus("${query}") surfaced ${newChunks.length} new chunks and added ${added.length} to visible context.`,
        });

        return {
          query,
          chunks: buildChunkResultPayload(ctx, added),
          totalNewChunks: newChunks.length,
          totalMatched: matched,
          requests,
          session: serializeSnapshot(ctx),
        };
      },
    }),

    grep_corpus: tool({
      description:
        "Regex search over the corpus or the currently encountered search space. Useful for exact identifiers or strings.",
      inputSchema: z.object({
        pattern: z.string().describe("Regex pattern or exact string"),
        limit: z.number().int().positive().max(20).optional(),
      }),
      execute: async ({ pattern, limit }) => {
        const gate = canUseContextTool(ctx, "grep_corpus");
        if (!gate.allowed) {
          return { error: gate.reason, session: serializeSnapshot(ctx) };
        }

        const resultLimit = limit ?? 5;
        const matches = ctx.config.grepCorpus
          ? await ctx.config.grepCorpus(pattern, {
              signal: ctx.config.signal,
              limit: resultLimit,
            })
          : fallbackGrep(ctx, pattern, resultLimit);

        const matchingChunks = matches
          .map((match) => ctx.seenChunks.get(match.chunkId))
          .filter((chunk): chunk is NonNullable<typeof chunk> => Boolean(chunk));
        addVisibleChunks(ctx, matchingChunks, getVisibleToolBudget(ctx));
        appendObservation(ctx, {
          type: "grep_corpus",
          summary: `grep_corpus("${pattern}") returned ${matches.length} matches.`,
        });

        return {
          pattern,
          matches: buildGrepPayload(ctx, matches),
          session: serializeSnapshot(ctx),
        };
      },
    }),

    read_document: tool({
      description:
        "Read a full source document by ID and bring it into the visible working set with token-aware truncation.",
      inputSchema: z.object({
        docId: z.string().describe("Document identifier"),
        tokenBudget: z.number().int().positive().optional(),
      }),
      execute: async ({ docId, tokenBudget }) => {
        const gate = canUseContextTool(ctx, "read_document");
        if (!gate.allowed) {
          return { error: gate.reason, session: serializeSnapshot(ctx) };
        }

        const document = resolveDocument(
          ctx,
          docId,
          ctx.config.readDocument
            ? await ctx.config.readDocument(docId, {
                signal: ctx.config.signal,
                excludeChunkIds: Array.from(ctx.seenChunks.keys()),
              })
            : null,
        );

        if (!document) {
          appendObservation(ctx, {
            type: "read_document",
            summary: `read_document("${docId}") found no document.`,
          });
          return {
            docId,
            found: false,
            session: serializeSnapshot(ctx),
          };
        }

        for (const chunk of document.chunks ?? []) {
          if (!ctx.seenChunks.has(chunk.id)) {
            ctx.seenChunks.set(chunk.id, chunk);
          }
        }

        const visibleDocument = addVisibleDocument(
          ctx,
          document,
          getVisibleToolBudget(ctx, tokenBudget),
        );
        appendObservation(ctx, {
          type: "read_document",
          summary: `read_document("${docId}") added document context${document.chunks?.length ? ` with ${document.chunks.length} chunks` : ""}.`,
        });

        return {
          docId,
          found: true,
          document: {
            id: visibleDocument.id,
            text: visibleDocument.text,
            metadata: visibleDocument.metadata,
            chunkIds: visibleDocument.chunks?.map((chunk) => chunk.id) ?? [],
          },
          session: serializeSnapshot(ctx),
        };
      },
    }),

    prune_chunks: tool({
      description:
        "Remove chunk IDs from the visible working set while preserving latent encounter history.",
      inputSchema: z.object({
        chunkIds: z.array(z.string()).min(1).describe("Chunk IDs to prune"),
      }),
      execute: async ({ chunkIds }) => {
        const result = pruneVisibleChunks(ctx, chunkIds);
        appendObservation(ctx, {
          type: "prune_chunks",
          summary: `prune_chunks removed ${result.removed.length} visible chunks.`,
        });

        return {
          removed: result.removed,
          missing: result.missing,
          session: serializeSnapshot(ctx),
        };
      },
    }),

    deep_search: tool({
      description:
        "Run the compiled high-recall retrieval loop. Use this when a one-shot agentic retrieval pass is more efficient than manual tool chaining.",
      inputSchema: z.object({
        query: z.string().describe("The search query"),
        maxResults: z.number().optional().describe("Maximum results to return (default: 20)"),
        minScore: z.number().optional(),
        topK: z.number().int().positive().optional(),
        maxContextTokens: z.number().optional(),
      }),
      execute: async ({ query, maxResults, minScore, topK, maxContextTokens }) => {
        const gate = canUseContextTool(ctx, "deep_search");
        if (!gate.allowed) {
          return { error: gate.reason, session: serializeSnapshot(ctx) };
        }

        applyRunOptions(ctx, config, { minScore, topK, maxContextTokens });
        ctx.rankedChunks.clear();
        ctx.iterations = [];

        const initialChunks = await embeddingSearch(query, ctx);
        await queryExpansion(query, initialChunks, ctx);
        const allChunks = Array.from(ctx.seenChunks.values());
        await rerank(query, allChunks, ctx, {
          sourceQuery: query,
          iteration: 0,
        });
        await iterativeSearch(query, ctx);

        if (ctx.config.enableClassifier) {
          await classify(query, ctx);
        }

        const finalChunks = Array.from(ctx.rankedChunks.values())
          .sort((a, b) => b.relevance - a.relevance)
          .slice(0, maxResults ?? 20);
        addVisibleChunks(ctx, finalChunks, getVisibleToolBudget(ctx));
        appendObservation(ctx, {
          type: "deep_search",
          summary: `deep_search("${query}") kept ${finalChunks.length} ranked chunks visible.`,
        });

        const payload = await buildToolPayload(query, finalChunks, ctx);

        return {
          ...payload,
          totalConsidered: ctx.seenChunks.size,
          totalMatched: ctx.totalChunksMatched,
          iterationsRun: ctx.iterations.length,
          minScoreUsed: ctx.config.minScore,
          topKUsed: ctx.config.topK,
          shardCount: ctx.shardCount,
          recursionDepth: ctx.recursionDepth,
          session: serializeSnapshot(ctx),
          __progress: progress,
          __logLines: logLines,
        };
      },
    }),
  };
}
