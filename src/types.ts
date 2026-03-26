import type { LanguageModel } from "ai";

/** A single chunk returned from the user's embedding store. */
export interface EmbeddingChunk {
  /** Unique identifier for deduplication across iterations. */
  id: string;
  /** The text content of this chunk. */
  text: string;
  /** Similarity score from the vector search (higher = more similar). */
  score: number;
  /** Arbitrary metadata attached to the chunk. */
  metadata?: Record<string, unknown>;
}

/** The function the user provides to search their embeddings. */
export type EmbeddingSearchFn = (
  query: string,
  options: { topK: number },
) => Promise<EmbeddingChunk[]>;

/** Configuration for the agentic retrieval loop. */
export interface AlphaloopConfig {
  /** The user's embedding search function. Takes a string query, returns chunks. */
  search: EmbeddingSearchFn;

  /** AI SDK LanguageModel for query expansion, re-ranking, and synthesis. */
  model: LanguageModel;

  /** Optional: separate (cheaper/faster) model for re-ranking. Defaults to `model`. */
  rerankModel?: LanguageModel;

  /** Maximum chunks to retrieve per search call (default: 200). */
  initialTopK?: number;

  /** Maximum expanded queries to generate per round (default: 8). */
  maxExpandedQueries?: number;

  /** Maximum iterative refinement rounds (default: 3). */
  maxIterations?: number;

  /** Minimum relevance score for re-ranked chunks, 0–1 (default: 0.3). */
  relevanceThreshold?: number;

  /** Enable the optional classifier step (default: false). */
  enableClassifier?: boolean;

  /** Custom system prompt for query expansion. */
  queryExpansionPrompt?: string;

  /** Custom system prompt for re-ranking. */
  rerankPrompt?: string;

  /** Abort signal for cancellation. */
  signal?: AbortSignal;
}

/** A chunk that has been scored by the LLM re-ranker. */
export interface RankedChunk extends EmbeddingChunk {
  /** LLM-assigned relevance score (0–1). */
  relevance: number;
  /** Why the LLM considered this chunk relevant. */
  rationale?: string;
  /** Which query variant originally found this chunk. */
  sourceQuery?: string;
  /** Which iteration found this chunk (0 = initial search). */
  iteration?: number;
}

/** Telemetry for a single loop iteration. */
export interface LoopIterationResult {
  iteration: number;
  newQueries: string[];
  chunksFound: number;
  totalUniqueChunks: number;
}

/** Final result of the agentic loop. */
export interface AlphaloopResult {
  /** All retrieved and ranked chunks, best first. */
  chunks: RankedChunk[];
  /** Telemetry about each iteration. */
  iterations: LoopIterationResult[];
  /** Total unique chunks considered across all iterations. */
  totalChunksConsidered: number;
}

/** Stream events emitted during loop execution. */
export type AlphaloopStreamEvent =
  | {
      type: "embedding_search";
      query: string;
      chunksFound: number;
    }
  | {
      type: "query_expansion";
      queries: string[];
      newChunksFound: number;
      totalUnique: number;
    }
  | {
      type: "rerank";
      totalChunks: number;
      keptChunks: number;
      droppedChunks: number;
      topChunkPreview?: string;
    }
  | {
      type: "iterative_search";
      iteration: number;
      newQueries: string[];
      newChunksFound: number;
      totalUnique: number;
    }
  | {
      type: "classifier";
      classified: number;
      kept: number;
      dropped: number;
    }
  | {
      type: "complete";
      totalChunks: number;
      iterations: number;
    }
  | {
      type: "error";
      message: string;
    };

/** Internal context passed between loop steps. */
export interface LoopContext {
  config: Required<
    Pick<
      AlphaloopConfig,
      | "initialTopK"
      | "maxExpandedQueries"
      | "maxIterations"
      | "relevanceThreshold"
      | "enableClassifier"
    >
  > &
    AlphaloopConfig;
  /** All unique chunks seen so far, keyed by id. */
  seenChunks: Map<string, EmbeddingChunk>;
  /** Chunks that passed re-ranking, keyed by id. */
  rankedChunks: Map<string, RankedChunk>;
  /** All queries tried so far (for dedup). */
  triedQueries: Set<string>;
  /** Iteration telemetry. */
  iterations: LoopIterationResult[];
  /** Callback to emit stream events. */
  emit: (event: AlphaloopStreamEvent) => void;
}
