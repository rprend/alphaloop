import {
  streamText,
  convertToModelMessages,
  createUIMessageStream,
  createUIMessageStreamResponse,
  stepCountIs,
  tool,
  type UIMessage,
} from "ai";
import { alphaloopToolsWithSession } from "./tools.js";
import type { AlphaloopConfig } from "./types.js";

export type AlphaloopHandlerConfig = AlphaloopConfig & {
  systemPrompt?: string;
  additionalTools?: Parameters<typeof streamText>[0]["tools"];
  maxToolSteps?: number;
};

const DEFAULT_SYSTEM_PROMPT = `You are a helpful research assistant operating a bounded search session.

Prefer the low-level tools when the question is ambiguous, requires exact strings, or you need to manage context carefully:
- search_corpus for retrieval
- grep_corpus for exact matches
- read_document for full source expansion
- prune_chunks when visible context grows too large

Use deep_search when a one-shot compiled retrieval pass is the fastest option.

Always watch the session token usage returned by tools. If the session crosses the soft limit, prune aggressively. If it hits the hard limit, do not call more context-gathering tools until you prune. Synthesize the retrieved passages into a clear response and cite specific passages when appropriate.`;

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
            ...alphaloopToolsWithSession(
              {
                ...config,
                signal: request.signal,
              },
              (event) => {
                writer.write({
                  type: "data-search-progress" as any,
                  data: event,
                } as any);
              },
            ),
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
