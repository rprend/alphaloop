import { DurableObject } from "cloudflare:workers";
import { createOpenAI } from "@ai-sdk/openai";
import { runRealStressScenario } from "./shared/realRuntime.mjs";
import { createVectorizeSearch } from "./shared/realSearch.mjs";
import { REAL_STRESS_SCENARIOS } from "./shared/realStressScenarios.mjs";

const INITIAL_EVENT = {
  type: "phase",
  label: "Starting server-side search",
};

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === "/api/run" && request.method === "POST") {
      const body = await request.json();
      const scenario = REAL_STRESS_SCENARIOS[body.scenarioId];
      if (!scenario) {
        return new Response("Unknown scenario", { status: 400 });
      }

      const jobId = crypto.randomUUID();
      const stub = env.RUN_JOBS.getByName(jobId);
      const initResponse = await stub.fetch("https://run-job/init", {
        method: "POST",
        headers: {
          "content-type": "application/json",
        },
        body: JSON.stringify(body),
      });

      if (!initResponse.ok) {
        return new Response(await initResponse.text(), {
          status: initResponse.status,
        });
      }

      return Response.json(
        {
          jobId,
          status: "queued",
        },
        {
          headers: {
            "cache-control": "no-store",
          },
        },
      );
    }

    if (url.pathname.startsWith("/api/run/") && request.method === "GET") {
      const jobId = url.pathname.slice("/api/run/".length);
      if (!jobId) {
        return new Response("Missing job id", { status: 400 });
      }

      const stub = env.RUN_JOBS.getByName(jobId);
      return stub.fetch("https://run-job/status");
    }

    return env.ASSETS.fetch(request);
  },
};

export class RunJob extends DurableObject {
  constructor(ctx, env) {
    super(ctx, env);
    this.ctx = ctx;
    this.env = env;
    this.indexInfoPromise = undefined;
  }

  async fetch(request) {
    const url = new URL(request.url);

    try {
      if (url.pathname === "/init" && request.method === "POST") {
        const existing = await this.ctx.storage.get("job");
        if (existing?.status === "running" || existing?.status === "completed") {
          return Response.json(existing, {
            headers: {
              "cache-control": "no-store",
            },
          });
        }

        const body = await request.json();
        const scenario = REAL_STRESS_SCENARIOS[body.scenarioId];
        if (!scenario) {
          return new Response("Unknown scenario", { status: 400 });
        }

      const job = {
          status: "queued",
          events: [INITIAL_EVENT],
          result: null,
          stats: null,
          searchStats: null,
          runtime: null,
          scenario,
          query: body.query || scenario.query,
          error: null,
          request: body,
      };
      console.log("run-job init", {
        id: this.ctx.id.toString(),
        scenarioId: body.scenarioId,
      });
      await this.ctx.storage.put("job", job);
      await this.ctx.storage.setAlarm(Date.now() + 100);

        return Response.json(
          {
            jobId: this.ctx.id.toString(),
            status: "running",
          },
          {
            headers: {
              "cache-control": "no-store",
            },
          },
        );
      }

      if (url.pathname === "/status" && request.method === "GET") {
        const job = (await this.ctx.storage.get("job")) || {
          status: "queued",
          events: [],
        };
        return Response.json(job, {
          headers: {
            "cache-control": "no-store",
          },
        });
      }
    } catch (error) {
      console.error("run job fetch error", error);
      return Response.json(
        {
          status: "error",
          error: error instanceof Error ? error.message : "Unknown run job error",
        },
        { status: 500 },
      );
    }

    return new Response("Not found", { status: 404 });
  }

  async alarm() {
    console.log("run-job alarm", { id: this.ctx.id.toString() });
    const job = await this.ctx.storage.get("job");
    if (!job?.request || job.status === "completed") {
      console.log("run-job alarm noop", {
        id: this.ctx.id.toString(),
        status: job?.status,
        hasRequest: Boolean(job?.request),
      });
      return;
    }
    await this.execute(job.request);
  }

  async execute(body) {
    try {
      console.log("run-job execute start", {
        id: this.ctx.id.toString(),
        scenarioId: body.scenarioId,
      });
      let jobState = (await this.ctx.storage.get("job")) || {
        status: "queued",
        events: [INITIAL_EVENT],
      };
      const persist = (patch) => {
        jobState = {
          ...jobState,
          ...patch,
        };
        return this.ctx.storage.put("job", jobState);
      };
      await persist({
        ...jobState,
        status: "running",
      });

      if (!this.env.OPENAI_API_KEY) {
        throw new Error("OPENAI_API_KEY is not configured");
      }

      if (!this.env.CORPUS_INDEX) {
        throw new Error("CORPUS_INDEX is not configured");
      }

      const scenario = REAL_STRESS_SCENARIOS[body.scenarioId];
      const openai = createOpenAI({ apiKey: this.env.OPENAI_API_KEY });
      const indexInfo = await this.loadIndexInfo();
      const embeddingDimensions = Number(
        this.env.OPENAI_EMBEDDING_DIMENSIONS || 1536,
      );
      const events = [...(jobState.events || [INITIAL_EVENT])];
      const search = createVectorizeSearch({
        index: this.env.CORPUS_INDEX,
        totalChunks: indexInfo.vectorCount,
        scenario,
        embeddingModel: openai.embedding(
          this.env.OPENAI_EMBEDDING_MODEL || "text-embedding-3-large",
        ),
        embeddingDimensions,
        apiKey: this.env.OPENAI_API_KEY,
        embeddingModelId:
          this.env.OPENAI_EMBEDDING_MODEL || "text-embedding-3-large",
      });

      const run = await runRealStressScenario({
        search,
        embeddedChunkCount: indexInfo.vectorCount,
        apiKey: this.env.OPENAI_API_KEY,
        modelId: this.env.OPENAI_MODEL,
        rerankModelId: this.env.OPENAI_RERANK_MODEL,
        embeddingModelId: this.env.OPENAI_EMBEDDING_MODEL,
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
          this.ctx.waitUntil(
            persist({
              status: "running",
              events: [...events],
            }),
          );
        },
      });

      await persist({
        status: "completed",
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
        error: null,
        request: body,
      });
      console.log("run-job execute complete", {
        id: this.ctx.id.toString(),
        totalMatched: run.result.totalChunksMatched,
      });
    } catch (error) {
      console.error("alphaloop worker error", error);
      const existing = (await this.ctx.storage.get("job")) || {
        status: "running",
        events: [INITIAL_EVENT],
      };
      await this.ctx.storage.put("job", {
        ...existing,
        status: "error",
        error: error instanceof Error ? error.message : "Unknown worker error",
      });
    }
  }

  async loadIndexInfo() {
    if (!this.indexInfoPromise) {
      this.indexInfoPromise = this.env.CORPUS_INDEX.describe().then((info) => {
        if ((info.vectorCount || 0) === 0) {
          this.indexInfoPromise = undefined;
        }
        return info;
      });
    }
    return this.indexInfoPromise;
  }
}
