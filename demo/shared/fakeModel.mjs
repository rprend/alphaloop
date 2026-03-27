function usage(inputLength, outputLength) {
  return {
    inputTokens: {
      total: Math.ceil(inputLength / 4),
      noCache: Math.ceil(inputLength / 4),
      cacheRead: 0,
      cacheWrite: 0,
    },
    outputTokens: {
      total: Math.ceil(outputLength / 4),
      text: Math.ceil(outputLength / 4),
      reasoning: 0,
    },
  };
}

function collectPromptText(prompt) {
  return prompt
    .flatMap((message) =>
      typeof message.content === "string" ? [message.content] : message.content,
    )
    .map((part) =>
      typeof part === "string" ? part : "text" in part ? part.text : "",
    )
    .join("\n");
}

function parseQuery(text) {
  const match = text.match(/query:\s*"([^"]+)"/i) || text.match(/concept:\s*"([^"]+)"/i);
  return match ? match[1].toLowerCase() : "";
}

function unique(items) {
  return Array.from(new Set(items));
}

function summarize(text, maxLength = 1400) {
  const ids = unique(Array.from(text.matchAll(/Chunk ID:\s*([^\n]+)/g)).map((m) => m[1])).slice(0, 24);
  const themes = unique(
    Array.from(
      text.matchAll(/(atlas recursion ledger|quartz orchard signal|helios archive cadence|redwood circuit memory)/gi),
    ).map((m) => m[1].toLowerCase()),
  );
  const compact = text.replace(/\s+/g, " ").slice(0, maxLength);
  return [
    ids.length ? `chunk_ids=${ids.join(",")}` : null,
    themes.length ? `themes=${themes.join(",")}` : null,
    `evidence=${compact}`,
  ]
    .filter(Boolean)
    .join(" | ");
}

function buildRerankResponse(text) {
  const query = parseQuery(text);
  const queryTerms = query.split(/\s+/).filter(Boolean);
  const blocks = Array.from(text.matchAll(/\(id: ([^)]+)\)\n([\s\S]*?)(?=\n\n\[\d+\] \(id: |\s*$)/g));

  return {
    results: blocks.map(([, id, block]) => {
      const lower = block.toLowerCase();
      const hits = queryTerms.reduce(
        (sum, term) => sum + (lower.match(new RegExp(term, "g"))?.length ?? 0),
        0,
      );
      const relevance = Math.min(0.99, 0.55 + hits / Math.max(queryTerms.length * 10, 1));
      return {
        id,
        relevance,
        rationale: `Synthetic judge found ${hits} query-term hits in ${id}.`,
      };
    }),
  };
}

function buildClassifierResponse(text) {
  const query = parseQuery(text);
  const lower = text.toLowerCase();
  const hits = query
    .split(/\s+/)
    .filter(Boolean)
    .reduce((sum, term) => sum + (lower.match(new RegExp(term, "g"))?.length ?? 0), 0);

  return {
    relevant: hits > 2,
    confidence: Math.min(0.96, 0.45 + hits / 20),
    rationale: `Synthetic classifier counted ${hits} thematic hits.`,
  };
}

function buildQueryExpansionResponse(text) {
  const query = parseQuery(text) || "atlas recursion ledger";
  return {
    queries: [
      `${query} archive`,
      `${query} evidence`,
      `${query} branch`,
      `${query} summary`,
      `${query} cascade`,
      `${query} memory`,
      `${query} compression`,
      `${query} orchard`,
    ],
  };
}

function buildSummaryResponse(text) {
  return {
    summary: summarize(text),
  };
}

function detectMode(text) {
  if (text.includes("Score each passage for relevance")) return "rerank";
  if (text.includes("Is this passage relevant to the concept above?")) return "classifier";
  if (text.includes("Generate") && text.includes("query variants")) return "expand";
  if (text.includes("Generate") && text.includes("NEW search queries")) return "expand";
  return "summary";
}

export function createSyntheticLanguageModel() {
  return {
    specificationVersion: "v3",
    provider: "synthetic",
    modelId: "synthetic-recursive-lab",
    supportedUrls: {},
    async doGenerate(options) {
      const promptText = collectPromptText(options.prompt);
      const mode = detectMode(promptText);
      const object =
        mode === "rerank"
          ? buildRerankResponse(promptText)
          : mode === "classifier"
            ? buildClassifierResponse(promptText)
            : mode === "expand"
              ? buildQueryExpansionResponse(promptText)
              : buildSummaryResponse(promptText);
      const text = JSON.stringify(object);

      return {
        content: [{ type: "text", text }],
        finishReason: { unified: "stop", raw: "stop" },
        usage: usage(promptText.length, text.length),
        warnings: [],
        response: {
          id: `synthetic-${Date.now()}`,
          modelId: "synthetic-recursive-lab",
          timestamp: new Date(),
        },
      };
    },
    async doStream() {
      throw new Error("Streaming is not implemented for the synthetic demo model.");
    },
  };
}
