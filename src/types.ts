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

export interface SearchPage {
  chunks: EmbeddingChunk[];
  nextCursor?: string;
}

export interface SearchRequestOptions {
  minScore?: number;
  topK?: number;
  cursor?: string;
  signal?: AbortSignal;
  excludeChunkIds?: string[];
}

/** Paged search contract for vector stores. */
export type EmbeddingSearchFn = (
  query: string,
  options: SearchRequestOptions,
) => Promise<SearchPage>;

/** Streaming search contract for vector stores. */
export type EmbeddingSearchStreamFn = (
  query: string,
  options: Omit<SearchRequestOptions, "cursor">,
) => AsyncIterable<EmbeddingChunk>;

export interface CorpusDocument {
  id: string;
  text: string;
  metadata?: Record<string, unknown>;
  chunks?: EmbeddingChunk[];
}

export interface GrepCorpusMatch {
  chunkId: string;
  docId?: string;
  snippet: string;
  metadata?: Record<string, unknown>;
}

export type ReadDocumentFn = (
  docId: string,
  options: {
    signal?: AbortSignal;
    excludeChunkIds?: string[];
  },
) => Promise<CorpusDocument | null>;

export type GrepCorpusFn = (
  pattern: string,
  options: {
    signal?: AbortSignal;
    limit?: number;
  },
) => Promise<GrepCorpusMatch[]>;

/** Configuration for the agentic retrieval loop. */
export type AlphaloopConfig =
  | (AlphaloopSharedConfig & {
      /** The user's paged embedding search function. */
      search: EmbeddingSearchFn;
      searchStream?: never;
    })
  | (AlphaloopSharedConfig & {
      /** The user's streaming embedding search function. */
      searchStream: EmbeddingSearchStreamFn;
      search?: never;
    });

export interface AlphaloopSharedConfig {
  /** Minimum vector similarity score for a chunk to be treated as a strong match. */
  minScore?: number;
  /** Optional default topK override. When set, retrieval stops after K strong matches. */
  topK?: number;

  /** AI SDK LanguageModel for query expansion, re-ranking, and synthesis. */
  model: LanguageModel;

  /** Optional: separate (cheaper/faster) model for re-ranking. Defaults to `model`. */
  rerankModel?: LanguageModel;

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

  /** Maximum context budget for any single LLM call (default: 100,000). */
  maxContextTokens?: number;

  /** Optional token estimator override. */
  tokenEstimator?: (text: string) => number;

  /** Optional regex search over the corpus. */
  grepCorpus?: GrepCorpusFn;

  /** Optional document reader for expanding chunk hits into full source documents. */
  readDocument?: ReadDocumentFn;

  /** Soft context pressure ratio for the visible working set (default: 0.5). */
  softContextLimitRatio?: number;

  /** Hard context cutoff ratio for the visible working set (default: 0.8). */
  hardContextLimitRatio?: number;

  /** Reserve tokens for the model's next response in the visible working set (default: 4,000). */
  outputTokenReserve?: number;

  /** Abort signal for cancellation. */
  signal?: AbortSignal;
}

export interface AlphaloopRunOptions {
  minScore?: number;
  topK?: number;
  maxContextTokens?: number;
}

export interface SearchObservation {
  id: string;
  type:
    | "search_corpus"
    | "grep_corpus"
    | "read_document"
    | "prune_chunks"
    | "deep_search";
  summary: string;
}

export interface SessionSnapshot {
  tokenUsage: number;
  maxTokens: number;
  softLimit: number;
  hardLimit: number;
  remainingTokens: number;
  status: "ok" | "soft_limit" | "hard_limit";
  visibleChunkCount: number;
  visibleDocumentCount: number;
  encounteredChunkCount: number;
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
  /** Total matched chunks before deduplication across all search calls. */
  totalChunksMatched: number;
  /** Runtime minScore used for this run. */
  minScoreUsed: number;
  /** Runtime topK used for this run, if any. */
  topKUsed?: number;
  /** Deepest recursive shard depth reached. */
  recursionDepth: number;
  /** Total shards processed across recursive LLM steps. */
  shardCount: number;
}

/** Stream events emitted during loop execution. */
export type AlphaloopStreamEvent =
  | {
      type: "phase";
      label: string;
    }
  | {
      type: "embedding_search";
      query: string;
      chunksFound: number;
      chunksMatched: number;
      pagesFetched: number;
      minScore: number;
      topK?: number;
    }
  | {
      type: "query_expansion";
      queries: string[];
      newChunksFound: number;
      totalUnique: number;
      shardCount?: number;
      recursionDepth?: number;
    }
  | {
      type: "rerank";
      totalChunks: number;
      keptChunks: number;
      droppedChunks: number;
      topChunkPreview?: string;
      shardCount?: number;
      recursionDepth?: number;
    }
  | {
      type: "iterative_search";
      iteration: number;
      newQueries: string[];
      newChunksFound: number;
      totalUnique: number;
      shardCount?: number;
      recursionDepth?: number;
    }
  | {
      type: "classifier";
      classified: number;
      kept: number;
      dropped: number;
      shardCount?: number;
      recursionDepth?: number;
    }
  | {
      type: "complete";
      totalChunks: number;
      iterations: number;
      totalChunksMatched: number;
      minScore: number;
      topK?: number;
      shardCount: number;
      recursionDepth: number;
    }
  | {
      type: "error";
      message: string;
    };

export interface ResolvedLoopConfig extends AlphaloopSharedConfig {
  minScore: number;
  topK?: number;
  maxExpandedQueries: number;
  maxIterations: number;
  relevanceThreshold: number;
  enableClassifier: boolean;
  maxContextTokens: number;
  softContextLimitRatio: number;
  hardContextLimitRatio: number;
  outputTokenReserve: number;
}

/** Internal context passed between loop steps. */
export interface LoopContext {
  config: ResolvedLoopConfig &
    Omit<AlphaloopConfig, keyof AlphaloopSharedConfig> &
    AlphaloopSharedConfig;
  /** All unique chunks seen so far, keyed by id. */
  seenChunks: Map<string, EmbeddingChunk>;
  /** Chunks that passed re-ranking, keyed by id. */
  rankedChunks: Map<string, RankedChunk>;
  /** All queries tried so far (for dedup). */
  triedQueries: Set<string>;
  /** Chunks currently visible in the active search session. */
  visibleChunkIds: Set<string>;
  /** Documents currently visible in the active search session. */
  visibleDocuments: Map<string, CorpusDocument>;
  /** Tool observations kept in the visible search session. */
  observations: SearchObservation[];
  /** Current token usage of the visible working set. */
  tokenUsage: number;
  /** Soft context pressure limit. */
  softTokenLimit: number;
  /** Hard context cutoff. */
  hardTokenLimit: number;
  /** Iteration telemetry. */
  iterations: LoopIterationResult[];
  /** Total matched chunks before dedupe. */
  totalChunksMatched: number;
  /** Number of retrieval pages or stream pulls observed. */
  retrievalRequests: number;
  /** Total recursive shards executed. */
  shardCount: number;
  /** Maximum recursion depth observed. */
  recursionDepth: number;
  /** Callback to emit stream events. */
  emit: (event: AlphaloopStreamEvent) => void;
}
