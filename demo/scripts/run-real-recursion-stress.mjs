import fs from "node:fs/promises";
import path from "node:path";
import { runRealStressScenario } from "../shared/realRuntime.mjs";
import { REAL_SCENARIO_LIST } from "../shared/realStressScenarios.mjs";

const ROOT = process.cwd();
const ENV_PATH = path.resolve(ROOT, "../alphabook/.dev.vars");
const DATASET_PATH = path.resolve(ROOT, "demo/public/realDataset.json");

const env = await loadEnvFile(ENV_PATH);
const dataset = JSON.parse(await fs.readFile(DATASET_PATH, "utf8"));
const requestedScenarioIds = new Set(process.argv.slice(2));
const scenarios =
  requestedScenarioIds.size === 0
    ? REAL_SCENARIO_LIST
    : REAL_SCENARIO_LIST.filter((scenario) =>
        requestedScenarioIds.has(scenario.id),
      );

for (const scenario of scenarios) {
  console.log(`\n=== ${scenario.label} ===`);
  const events = [];
  const startedAt = Date.now();

  const run = await runRealStressScenario({
    dataset,
    apiKey: env.OPENAI_API_KEY,
    modelId: env.OPENAI_MODEL || "gpt-5.2",
    embeddingModelId: env.OPENAI_EMBEDDING_MODEL || "text-embedding-3-large",
    scenarioId: scenario.id,
    query: scenario.query,
    onEvent(event) {
      events.push(event);
      console.log(`[${event.type}]`, JSON.stringify(event));
    },
  });

  console.log(
    JSON.stringify(
      {
        scenario: scenario.label,
        durationMs: Date.now() - startedAt,
        totalMatched: run.result.totalChunksMatched,
        totalConsidered: run.result.totalChunksConsidered,
        shardCount: run.result.shardCount,
        recursionDepth: run.result.recursionDepth,
        events: events.length,
        estimatedTokens: run.searchStats.maxEstimatedTokens,
        strongMatches: run.searchStats.maxStrongMatches,
        baseMatches: run.searchStats.maxBaseMatches,
      },
      null,
      2,
    ),
  );
}

async function loadEnvFile(filePath) {
  const text = await fs.readFile(filePath, "utf8");
  const values = {};

  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const match = line.match(/^([A-Z0-9_]+)=(.*)$/);
    if (!match) continue;
    let value = match[2];
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    values[match[1]] = value;
  }

  return values;
}
