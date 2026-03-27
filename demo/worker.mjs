import { runRealStressScenario } from "./shared/realRuntime.mjs";

let datasetPromise;

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === "/api/run") {
      if (request.method !== "POST") {
        return new Response("Method not allowed", { status: 405 });
      }

      if (!env.OPENAI_API_KEY) {
        return new Response("OPENAI_API_KEY is not configured", { status: 500 });
      }

      const body = await request.json();
      const dataset = await loadDataset(request, env);
      const encoder = new TextEncoder();

      const stream = new ReadableStream({
        async start(controller) {
          const write = (payload) => {
            controller.enqueue(
              encoder.encode(`${JSON.stringify(payload)}\n`),
            );
          };

          try {
            const run = await runRealStressScenario({
              dataset,
              apiKey: env.OPENAI_API_KEY,
              modelId: env.OPENAI_MODEL,
              rerankModelId: env.OPENAI_RERANK_MODEL,
              embeddingModelId: env.OPENAI_EMBEDDING_MODEL,
              scenarioId: body.scenarioId,
              query: body.query,
              minScore: body.minScore,
              topK: body.topK,
              maxContextTokens: body.maxContextTokens,
              maxExpandedQueries: body.maxExpandedQueries,
              maxIterations: body.maxIterations,
              onEvent(event) {
                write({ type: "event", event });
              },
            });

            write({
              type: "result",
              result: {
                ...run.result,
                chunks: run.result.chunks.slice(0, 12).map((chunk) => ({
                  ...chunk,
                  text:
                    chunk.text.length > 3000
                      ? `${chunk.text.slice(0, 3000)}...`
                      : chunk.text,
                })),
              },
              stats: run.stats,
              searchStats: run.searchStats,
              runtime: run.runtime,
              scenario: run.scenario,
              query: run.query,
            });
          } catch (error) {
            write({
              type: "error",
              message:
                error instanceof Error ? error.message : "Unknown worker error",
            });
          } finally {
            controller.close();
          }
        },
      });

      return new Response(stream, {
        headers: {
          "content-type": "application/x-ndjson; charset=utf-8",
          "cache-control": "no-store",
        },
      });
    }

    return env.ASSETS.fetch(request);
  },
};

async function loadDataset(request, env) {
  if (!datasetPromise) {
    datasetPromise = (async () => {
      const datasetRequest = new Request(
        new URL("/realDataset.json", request.url).toString(),
      );
      const response = await env.ASSETS.fetch(datasetRequest);
      if (!response.ok) {
        throw new Error(`Failed to load dataset asset (${response.status})`);
      }

      return response.json();
    })();
  }

  return datasetPromise;
}
