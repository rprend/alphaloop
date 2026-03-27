import { createOpenAI } from "@ai-sdk/openai";
import { runRealStressScenario } from "./shared/realRuntime.mjs";
import { createVectorizeSearch } from "./shared/realSearch.mjs";
import { REAL_STRESS_SCENARIOS } from "./shared/realStressScenarios.mjs";

let indexInfoPromise;

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

      if (!env.CORPUS_INDEX) {
        return new Response("CORPUS_INDEX is not configured", { status: 500 });
      }

      const body = await request.json();
      const scenario = REAL_STRESS_SCENARIOS[body.scenarioId];
      if (!scenario) {
        return new Response("Unknown scenario", { status: 400 });
      }

      const openai = createOpenAI({ apiKey: env.OPENAI_API_KEY });
      const indexInfo = await loadIndexInfo(env);
      const embeddingDimensions = Number(env.OPENAI_EMBEDDING_DIMENSIONS || 1536);
      const search = createVectorizeSearch({
        index: env.CORPUS_INDEX,
        totalChunks: indexInfo.vectorCount,
        scenario,
        embeddingModel: openai.embedding(
          env.OPENAI_EMBEDDING_MODEL || "text-embedding-3-large",
        ),
        embeddingDimensions,
      });
      const events = [
        {
          type: "phase",
          label: "Starting server-side search",
        },
      ];

      try {
        const run = await runRealStressScenario({
          search,
          embeddedChunkCount: indexInfo.vectorCount,
          apiKey: env.OPENAI_API_KEY,
          modelId: env.OPENAI_MODEL,
          rerankModelId: env.OPENAI_RERANK_MODEL,
          embeddingModelId: env.OPENAI_EMBEDDING_MODEL,
          embeddingDimensions,
          scenarioId: body.scenarioId,
          query: body.query,
          minScore: body.minScore,
          topK: body.topK,
          maxContextTokens: body.maxContextTokens,
          maxExpandedQueries: body.maxExpandedQueries,
          maxIterations: body.maxIterations,
          onEvent(event) {
            events.push(event);
          },
        });

        return Response.json({
          events,
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
        }, {
          headers: {
            "cache-control": "no-store",
          },
        });
      } catch (error) {
        console.error("alphaloop worker error", error);
        return Response.json({
          error: error instanceof Error ? error.message : "Unknown worker error",
          events,
        }, {
          status: 500,
          headers: {
            "cache-control": "no-store",
          },
        });
      }
    }

    return env.ASSETS.fetch(request);
  },
};

async function loadIndexInfo(env) {
  if (!indexInfoPromise) {
    indexInfoPromise = env.CORPUS_INDEX.describe().then((info) => {
      if ((info.vectorCount || 0) === 0) {
        indexInfoPromise = undefined;
      }
      return info;
    });
  }
  return indexInfoPromise;
}
