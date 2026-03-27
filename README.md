# alphaloop

Turn any embeddings dataset into an agentic retrieval loop. Drop-in NPM package that adds query expansion, LLM re-ranking, and iterative refinement to your existing vector search.

Works with any AI SDK model provider (OpenAI, Anthropic, Google, Cloudflare Workers AI) and any vector database.

## Install

```bash
npm install alphaloop ai zod
```

## Quick Start

### 1. Create the handler (backend)

```ts
import { createAlphaloopHandler } from "alphaloop/handler";
import { openai } from "@ai-sdk/openai";

const handler = createAlphaloopHandler({
  model: openai("gpt-4o"),
  minScore: 0.75,
  search: async (query, { minScore, topK, cursor }) => {
    // Default mode is comprehensive over minScore.
    // If topK is passed at runtime, stop after the first K strong matches.
    const page = await myVectorDB.search(query, {
      minScore,
      topK,
      cursor,
    });
    return {
      chunks: page.results.map((r) => ({
        id: r.id,
        text: r.text,
        score: r.score,
        metadata: r.metadata,
      })),
      nextCursor: page.nextCursor,
    };
  },
});

// Cloudflare Worker
export default { fetch: handler };

// Or Next.js API route
export const POST = handler;

// Or Express
app.post("/api/chat", (req, res) => handler(req).then((r) => r));
```

### 2. Connect the frontend

```tsx
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { SearchProgress, Citations } from "alphaloop/react";

const transport = new DefaultChatTransport({ api: "/api/chat" });

function Search() {
  const { messages, sendMessage, status } = useChat({ transport });
  // Render messages, progress, and citations
}
```

That's it. The handler runs the full agentic loop and streams results.

## How It Works

Alphaloop runs a 4-step retrieval loop over your embeddings:

### 1. Embedding Search (recall baseline)

Calls your `search` function with the original query. By default it fetches all chunks whose vector score is `>= minScore`. At runtime you can override that and request `topK` strong matches instead.

### 2. Query Expansion

The LLM generates diverse query variants — synonyms, rephrasings, related concepts, more specific/abstract formulations. All variants are searched in parallel and results are deduplicated. This alone massively improves recall.

For example, a query about "grief" might expand to:
- "coping with loss and bereavement"
- "emotional response to death of a loved one"
- "processing absence and longing"
- "psychological stages of mourning"

### 3. LLM Re-ranking

All collected chunks are sent to the LLM for relevance scoring (0-1). The LLM reads each passage and evaluates how relevant it is to the original query. This catches implicit and subtle matches that embedding similarity misses — a passage about Buddhist impermanence might score highly for a grief query even though the word "grief" never appears.

### 4. Iterative Refinement

Top-ranked passages are fed back to the LLM to generate NEW search queries based on discovered concepts. This is concept expansion — "I found passages about attachment and impermanence, what else should I search for?" The loop repeats up to `maxIterations` times or until no new relevant chunks are found.

### 5. Classification (optional)

An optional step that classifies remaining unranked chunks against abstract concepts, catching things even the re-ranker might miss.

## Configuration

```ts
createAlphaloopHandler({
  // Required
  model: openai("gpt-4o"),       // Any AI SDK LanguageModel
  search: mySearchFunction,      // Your paged search, or use searchStream

  // Optional
  minScore: 0.75,                // Strong-match threshold (default: 0)
  topK: 100,                     // Optional default topK override
  rerankModel: openai("gpt-4o-mini"), // Cheaper model for re-ranking
  maxExpandedQueries: 8,         // Query variants per round (default: 8)
  maxIterations: 3,              // Refinement rounds (default: 3)
  relevanceThreshold: 0.3,       // Min relevance score 0-1 (default: 0.3)
  enableClassifier: false,       // Enable classifier step (default: false)
  maxContextTokens: 100_000,     // Max tokens per single LLM call
  systemPrompt: "...",           // Custom system prompt
  additionalTools: { ... },      // Extra AI SDK tools
  maxToolSteps: 5,               // Max tool call steps (default: 5)
});
```

## Search Function

Your search function takes a string query and returns strong matches. Alphaloop defaults to comprehensive retrieval over `minScore`, but callers can override that at runtime with either a different `minScore` or a `topK`.

```ts
type EmbeddingSearchFn = (
  query: string,
  options: {
    minScore?: number;
    topK?: number;
    cursor?: string;
    signal?: AbortSignal;
  },
) => Promise<{ chunks: EmbeddingChunk[]; nextCursor?: string }>;

type EmbeddingSearchStreamFn = (
  query: string,
  options: { minScore?: number; topK?: number; signal?: AbortSignal },
) => AsyncIterable<EmbeddingChunk>;

interface EmbeddingChunk {
  id: string;
  text: string;
  score: number;
  metadata?: Record<string, unknown>;
}
```

Works with any vector database: Cloudflare Vectorize, Pinecone, Weaviate, pgvector, in-memory, etc.

## React Components

Optional UI components for displaying search progress and citations. Import from `alphaloop/react`.

### SearchProgress

Shows streaming progress as the loop runs:

```tsx
import { SearchProgress } from "alphaloop/react";

<SearchProgress events={progressEvents} isRunning={true} />
```

### Citations

Expandable source citations with quote-style previews:

```tsx
import { Citations } from "alphaloop/react";

<Citations
  chunks={citations}
  getSourceUrl={(chunk) => `/docs/${chunk.id}`} // Optional: make source IDs clickable links
/>
```

## Programmatic Usage

Use the core API directly without the handler:

```ts
import { createAlphaloop } from "alphaloop";
import { openai } from "@ai-sdk/openai";

const loop = createAlphaloop({
  model: openai("gpt-4o"),
  search: mySearchFn,
  minScore: 0.75,
});

// Run in comprehensive threshold mode
const result = await loop.run("What is consciousness?", {
  minScore: 0.8,
  maxContextTokens: 100_000,
});

// Or switch this run to topK mode
const focused = await loop.run("What is consciousness?", {
  topK: 50,
});
console.log(result.chunks);     // Ranked results
console.log(result.iterations); // Loop telemetry
console.log(result.totalChunksMatched);

// Or use as AI SDK tools
const tools = loop.tools();
// tools.deep_search — use with streamText()
```

## Streaming Progress

The handler streams progress events during tool execution using the AI SDK's `createUIMessageStream`. Register the `dataPartSchemas` in your `useChat` call to receive them:

```tsx
import { useChat } from "@ai-sdk/react";
import { jsonSchema } from "ai";

const { messages } = useChat({
  transport,
  dataPartSchemas: {
    "search-progress": {
      schema: jsonSchema({ type: "object", properties: { type: { type: "string" } } }),
    },
  },
});

// Progress events appear as message parts with type "data-search-progress"
```

## Cloudflare Workers

Alphaloop uses Web Standards only (ReadableStream, fetch, TextEncoder) — no Node.js APIs. It works on Cloudflare Workers out of the box.

```ts
// wrangler.toml — no special flags needed
// The loop is mostly network calls (LLM API, vector DB)
// which don't count toward the CPU time limit.
```

## License

MIT
