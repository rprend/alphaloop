const THEMES = [
  "atlas recursion ledger",
  "quartz orchard signal",
  "helios archive cadence",
  "redwood circuit memory",
];

function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), t | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function buildParagraph(theme, docIndex, wordsPerParagraph, rng) {
  const [a, b, c] = theme.split(" ");
  const filler = [
    "signal",
    "recursion",
    "ledger",
    "archive",
    "sequence",
    "vector",
    "memory",
    "branch",
    "cascade",
    "evidence",
    "compression",
    "shard",
    "summary",
  ];
  const words = [];

  for (let i = 0; i < wordsPerParagraph; i++) {
    if (i % 37 === 0) words.push(a);
    else if (i % 37 === 7) words.push(b);
    else if (i % 37 === 15) words.push(c);
    else if (i % 37 === 22) words.push(`doc${docIndex}`);
    else words.push(filler[Math.floor(rng() * filler.length)]);
  }

  return words.join(" ");
}

function cosineSimilarity(a, b) {
  let dot = 0;
  let magA = 0;
  let magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  if (magA === 0 || magB === 0) return 0;
  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

function embed(text) {
  const lower = text.toLowerCase();
  const basis = [
    "atlas",
    "recursion",
    "ledger",
    "quartz",
    "orchard",
    "signal",
    "helios",
    "archive",
    "cadence",
    "redwood",
    "circuit",
    "memory",
    "branch",
    "compression",
    "shard",
  ];

  return basis.map((term) => {
    const matches = lower.match(new RegExp(term, "g"));
    return matches ? matches.length : 0;
  });
}

export const STRESS_SCENARIOS = {
  branch2: {
    id: "branch2",
    label: "Branch x2",
    documents: 80,
    paragraphsPerDoc: 8,
    wordsPerParagraph: 220,
    query: "atlas recursion ledger",
    minScore: 0.55,
  },
  branch8: {
    id: "branch8",
    label: "Branch x8",
    documents: 420,
    paragraphsPerDoc: 8,
    wordsPerParagraph: 230,
    query: "atlas recursion ledger",
    minScore: 0.55,
  },
  branch36: {
    id: "branch36",
    label: "Millions of tokens",
    documents: 1800,
    paragraphsPerDoc: 9,
    wordsPerParagraph: 240,
    query: "atlas recursion ledger",
    minScore: 0.55,
  },
};

export function createScenarioDataset(scenarioId) {
  const scenario = STRESS_SCENARIOS[scenarioId];
  if (!scenario) {
    throw new Error(`Unknown scenario: ${scenarioId}`);
  }

  const rng = mulberry32(scenario.documents * scenario.wordsPerParagraph);
  const chunks = [];
  let estimatedTokens = 0;

  for (let docIndex = 0; docIndex < scenario.documents; docIndex++) {
    const theme = THEMES[docIndex % THEMES.length];
    const paragraphs = [];
    for (let p = 0; p < scenario.paragraphsPerDoc; p++) {
      paragraphs.push(
        buildParagraph(theme, docIndex, scenario.wordsPerParagraph, rng),
      );
    }

    const text =
      `Document ${docIndex} | Theme ${theme}\n` +
      paragraphs.join("\n\n");
    estimatedTokens += Math.ceil(text.length / 4);
    chunks.push({
      id: `doc-${scenario.id}-${docIndex}`,
      text,
      score: 0,
      metadata: {
        docIndex,
        theme,
        scenario: scenario.id,
      },
      embedding: embed(text),
    });
  }

  return {
    scenario,
    chunks,
    estimatedTokens,
  };
}

export async function searchScenario(
  dataset,
  query,
  { minScore, topK, cursor },
) {
  const queryEmbedding = embed(query);
  const ranked = dataset.chunks
    .map((chunk) => ({
      ...chunk,
      score: cosineSimilarity(queryEmbedding, chunk.embedding),
    }))
    .filter((chunk) => (minScore == null ? true : chunk.score >= minScore))
    .sort((a, b) => b.score - a.score);

  const capped = topK == null ? ranked : ranked.slice(0, topK);
  const start = cursor ? Number(cursor) : 0;
  const pageSize = 120;
  const page = capped.slice(start, start + pageSize).map(({ embedding, ...chunk }) => chunk);
  const nextCursor =
    start + pageSize < capped.length ? String(start + pageSize) : undefined;

  return {
    chunks: page,
    nextCursor,
    totalStrongMatches: capped.length,
    estimatedTokens: capped.reduce(
      (total, chunk) => total + Math.ceil(chunk.text.length / 4),
      0,
    ),
  };
}
