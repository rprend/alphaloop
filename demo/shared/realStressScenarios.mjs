export const REAL_STRESS_SCENARIOS = {
  realBase: {
    id: "realBase",
    label: "Real embedded corpus",
    query: "How can a team leader stay calm under pressure and make clear decisions during conflict?",
    minScore: 0.5,
    replicaCount: 1,
    textMultiplier: 1,
    maxExpandedQueries: 6,
    maxIterations: 2,
  },
  recursive: {
    id: "recursive",
    label: "Recursive branch",
    query: "How can a team leader stay calm under pressure and make clear decisions during conflict?",
    minScore: 0.5,
    replicaCount: 2,
    textMultiplier: 4,
    maxExpandedQueries: 6,
    maxIterations: 2,
  },
  millions: {
    id: "millions",
    label: "Millions of tokens",
    query: "How can a team leader stay calm under pressure and make clear decisions during conflict?",
    minScore: 0.5,
    replicaCount: 3,
    textMultiplier: 12,
    maxExpandedQueries: 8,
    maxIterations: 3,
  },
  extreme: {
    id: "extreme",
    label: "10x larger",
    query: "How can a team leader stay calm under pressure and make clear decisions during conflict?",
    minScore: 0.5,
    replicaCount: 5,
    textMultiplier: 20,
    maxExpandedQueries: 8,
    maxIterations: 3,
  },
};

export const REAL_SCENARIO_LIST = Object.values(REAL_STRESS_SCENARIOS);
