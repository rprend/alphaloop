import { build } from "vite";
import viteConfig from "../vite.config.js";
import { createAlphaloop } from "../../dist/index.js";
import { createSyntheticLanguageModel } from "../shared/fakeModel.mjs";
import { createScenarioDataset, searchScenario } from "../shared/stressData.mjs";

const SCENARIOS = ["branch2", "branch8", "branch36"];

function formatNumber(value) {
  return value.toLocaleString();
}

async function runScenario(scenarioId) {
  const dataset = createScenarioDataset(scenarioId);
  const model = createSyntheticLanguageModel();
  const searchStats = {
    totalStrongMatches: 0,
    estimatedTokens: 0,
  };

  const loop = createAlphaloop({
    model,
    rerankModel: model,
    minScore: dataset.scenario.minScore,
    maxExpandedQueries: 4,
    maxIterations: 1,
    search: async (query, options) => {
      const page = await searchScenario(dataset, query, options);
      searchStats.totalStrongMatches = Math.max(
        searchStats.totalStrongMatches,
        page.totalStrongMatches,
      );
      searchStats.estimatedTokens = Math.max(
        searchStats.estimatedTokens,
        page.estimatedTokens,
      );
      return {
        chunks: page.chunks,
        nextCursor: page.nextCursor,
      };
    },
  });

  const stream = loop.stream(dataset.scenario.query, {
    minScore: dataset.scenario.minScore,
    maxContextTokens: 100_000,
  });
  const eventCounts = {};
  let final;

  while (true) {
    const next = await stream.next();
    if (next.done) {
      final = next.value;
      break;
    }

    eventCounts[next.value.type] = (eventCounts[next.value.type] ?? 0) + 1;
  }

  console.log(`\nScenario: ${dataset.scenario.label}`);
  console.log(`Documents: ${formatNumber(dataset.chunks.length)}`);
  console.log(`Strong matches: ${formatNumber(searchStats.totalStrongMatches)}`);
  console.log(`Estimated tokens returned by vector search: ${formatNumber(searchStats.estimatedTokens)}`);
  console.log(`Recursive shard count: ${formatNumber(final.shardCount)}`);
  console.log(`Recursive depth: ${formatNumber(final.recursionDepth)}`);
  console.log(`Total unique chunks considered: ${formatNumber(final.totalChunksConsidered)}`);
  console.log(`Event counts: ${JSON.stringify(eventCounts)}`);

  if (searchStats.estimatedTokens <= 100_000) {
    throw new Error(`Scenario ${scenarioId} did not exceed the 100,000 token budget.`);
  }

  if (final.shardCount <= 1) {
    throw new Error(`Scenario ${scenarioId} did not trigger recursive sharding.`);
  }
}

await build(viteConfig);
console.log("Built demo app.");

for (const scenarioId of SCENARIOS) {
  await runScenario(scenarioId);
}

console.log("\nRecursive stress test completed successfully.");
