export const REAL_STRESS_SCENARIOS = {
  realBase: {
    id: "realBase",
    label: "Alpha Book-sized chunks",
    query: "How can a team leader stay calm under pressure and make clear decisions during conflict?",
    minScore: 0.64,
    replicaCount: 1,
    maxExpandedQueries: 6,
    maxIterations: 2,
  },
  recursive: {
    id: "recursive",
    label: "Recursive branch",
    query: "How can a team leader stay calm under pressure and make clear decisions during conflict?",
    minScore: 0.64,
    replicaCount: 64,
    maxExpandedQueries: 6,
    maxIterations: 2,
  },
  millions: {
    id: "millions",
    label: "Millions of chunks",
    query: "How can a team leader stay calm under pressure and make clear decisions during conflict?",
    minScore: 0.64,
    replicaCount: 4096,
    maxExpandedQueries: 8,
    maxIterations: 3,
  },
  extreme: {
    id: "extreme",
    label: "10x larger",
    query: "How can a team leader stay calm under pressure and make clear decisions during conflict?",
    minScore: 0.64,
    replicaCount: 40000,
    maxExpandedQueries: 8,
    maxIterations: 3,
  },
};

export const REAL_SCENARIO_LIST = Object.values(REAL_STRESS_SCENARIOS);
