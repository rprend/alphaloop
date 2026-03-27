import { createAlphaloop } from "../../dist/index.js";
import { createSyntheticLanguageModel } from "./fakeModel.mjs";
import { createScenarioDataset, searchScenario } from "./stressData.mjs";

export async function runStressScenario(scenarioId, overrides = {}) {
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

  const events = [];
  const stream = loop.stream(dataset.scenario.query, overrides);
  let final;
  while (true) {
    const next = await stream.next();
    if (next.done) {
      final = next.value;
      break;
    }
    events.push(next.value);
  }

  return {
    dataset,
    events,
    result: final,
    searchStats,
  };
}
