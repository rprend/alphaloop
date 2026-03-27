import {
  streamText,
  convertToModelMessages,
  createUIMessageStream,
  createUIMessageStreamResponse,
  stepCountIs,
  tool,
  type UIMessage,
} from "ai";
import { z } from "zod";
import { DEFAULTS } from "./defaults.js";
import { buildToolPayload } from "./response-payload.js";
import { embeddingSearch } from "./steps/embedding-search.js";
import { queryExpansion } from "./steps/query-expansion.js";
import { rerank } from "./steps/rerank.js";
import { iterativeSearch } from "./steps/iterative-search.js";
import { classify } from "./steps/classifier.js";
import type {
  AlphaloopConfig,
  AlphaloopRunOptions,
  AlphaloopStreamEvent,
  LoopContext,
} from "./types.js";

export type AlphaloopHandlerConfig = AlphaloopConfig & {
  systemPrompt?: string;
  additionalTools?: Parameters<typeof streamText>[0]["tools"];
  maxToolSteps?: number;
};

const DEFAULT_SYSTEM_PROMPT = `You are a helpful research assistant. When the user asks a question, use the deep_search tool to find relevant information from the knowledge base before answering. Synthesize the retrieved passages into a clear, well-structured response. Cite specific passages when appropriate.`;

export function createAlphaloopHandler(config: AlphaloopHandlerConfig) {
  return async function handler(request: Request): Promise<Response> {
    if (request.method !== "POST") {
      return new Response("Method not allowed", { status: 405 });
    }

    const body = (await request.json()) as { messages: UIMessage[] };
    const { messages } = body;

    if (!messages || !Array.isArray(messages)) {
      return new Response("Missing messages array", { status: 400 });
    }

    const stream = createUIMessageStream({
      execute: async ({ writer }) => {
        const result = streamText({
          model: config.model,
          system: config.systemPrompt ?? DEFAULT_SYSTEM_PROMPT,
          messages: await convertToModelMessages(messages),
          tools: {
            deep_search: tool({
              description:
                "Search the knowledge base using an agentic retrieval loop with query expansion, re-ranking, and iterative refinement.",
              inputSchema: z.object({
                query: z.string().describe("The search query"),
                maxResults: z
                  .number()
                  .optional()
                  .describe("Maximum results to return (default: 20)"),
                minScore: z
                  .number()
                  .optional()
                  .describe("Minimum vector score a chunk must meet to be considered a strong match"),
                maxContextTokens: z
                  .number()
                  .optional()
                  .describe("Maximum context tokens for any single LLM call"),
              }),
              execute: async ({ query, maxResults, minScore, maxContextTokens }) => {
                const ctx = createStreamingLoopContext(
                  config,
                  { minScore, maxContextTokens },
                  (event) => {
                    writer.write({
                      type: "data-search-progress" as any,
                      data: event,
                    } as any);
                  },
                );

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
                const payload = await buildToolPayload(query, finalChunks, ctx);

                return {
                  ...payload,
                  totalConsidered: ctx.seenChunks.size,
                  totalMatched: ctx.totalChunksMatched,
                  iterationsRun: ctx.iterations.length,
                  minScoreUsed: ctx.config.minScore,
                  shardCount: ctx.shardCount,
                  recursionDepth: ctx.recursionDepth,
                };
              },
            }),
            ...config.additionalTools,
          },
          stopWhen: stepCountIs(config.maxToolSteps ?? 5),
          abortSignal: request.signal,
        });

        writer.merge(result.toUIMessageStream());
      },
    });

    return createUIMessageStreamResponse({ stream });
  };
}

function createStreamingLoopContext(
  config: AlphaloopConfig,
  options: AlphaloopRunOptions = {},
  emit: (event: AlphaloopStreamEvent) => void,
): LoopContext {
  return {
    config: {
      ...config,
      minScore: options.minScore ?? config.minScore ?? DEFAULTS.minScore,
      maxExpandedQueries:
        config.maxExpandedQueries ?? DEFAULTS.maxExpandedQueries,
      maxIterations: config.maxIterations ?? DEFAULTS.maxIterations,
      relevanceThreshold:
        config.relevanceThreshold ?? DEFAULTS.relevanceThreshold,
      enableClassifier: config.enableClassifier ?? DEFAULTS.enableClassifier,
      maxContextTokens:
        options.maxContextTokens ??
        config.maxContextTokens ??
        DEFAULTS.maxContextTokens,
    },
    seenChunks: new Map(),
    rankedChunks: new Map(),
    triedQueries: new Set(),
    iterations: [],
    totalChunksMatched: 0,
    retrievalRequests: 0,
    shardCount: 0,
    recursionDepth: 0,
    emit,
  };
}
