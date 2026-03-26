"use client";

import { useState, type FC } from "react";

export interface SearchProgressEvent {
  type: string;
  query?: string;
  chunksFound?: number;
  queries?: string[];
  newChunksFound?: number;
  totalUnique?: number;
  totalChunks?: number;
  keptChunks?: number;
  droppedChunks?: number;
  iteration?: number;
  newQueries?: string[];
}

export interface SearchChunk {
  id: string;
  text: string;
  relevance: number;
  rationale?: string;
  metadata?: Record<string, unknown>;
}

function progressLabel(event: SearchProgressEvent): string {
  switch (event.type) {
    case "embedding_search":
      return `Searching for "${event.query}"`;
    case "query_expansion": {
      const variants = event.queries?.length ?? 0;
      const newResults = event.newChunksFound ?? 0;
      return `Expanding query — ${variants} variants, ${newResults} new results`;
    }
    case "rerank":
      return `Re-ranking — kept ${event.keptChunks} of ${event.totalChunks}`;
    case "iterative_search":
      return `Refining (round ${event.iteration}) — ${event.newChunksFound} new results`;
    case "classifier":
      return "Classifying results";
    case "complete":
      return `Found ${event.totalChunks} relevant passages`;
    default:
      return "Searching...";
  }
}

export const SearchProgress: FC<{
  events: SearchProgressEvent[];
  isRunning: boolean;
}> = ({ events, isRunning }) => {
  if (events.length === 0 && !isRunning) return null;

  return (
    <div style={{ padding: "8px 0", fontSize: "13px" }}>
      {events.map((event, i) => (
        <div
          key={i}
          style={{
            color: "#6b7280",
            lineHeight: "1.8",
            display: "flex",
            alignItems: "center",
            gap: "6px",
          }}
        >
          <span
            style={{
              width: "4px",
              height: "4px",
              borderRadius: "50%",
              background: "#d1d5db",
              flexShrink: 0,
            }}
          />
          {progressLabel(event)}
        </div>
      ))}
      {isRunning && (
        <div
          style={{
            color: "#9ca3af",
            lineHeight: "1.8",
            display: "flex",
            alignItems: "center",
            gap: "6px",
          }}
        >
          <span
            style={{
              width: "4px",
              height: "4px",
              borderRadius: "50%",
              background: "#3b82f6",
              flexShrink: 0,
              animation: "alphaloop-pulse 1.5s ease-in-out infinite",
            }}
          />
          Working...
          <style>{`@keyframes alphaloop-pulse { 0%, 100% { opacity: 0.3; } 50% { opacity: 1; } }`}</style>
        </div>
      )}
    </div>
  );
};

export const Citations: FC<{
  chunks: SearchChunk[];
  /** Optional: build a URL for a chunk so the source link is clickable. */
  getSourceUrl?: (chunk: SearchChunk) => string | undefined;
}> = ({ chunks, getSourceUrl }) => {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  if (chunks.length === 0) return null;

  return (
    <div style={{ marginTop: "16px" }}>
      <div
        style={{
          fontSize: "12px",
          fontWeight: 500,
          color: "#9ca3af",
          marginBottom: "8px",
          textTransform: "uppercase",
          letterSpacing: "0.05em",
        }}
      >
        Sources ({chunks.length})
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: "2px" }}>
        {chunks.map((chunk) => {
          const isExpanded = expandedId === chunk.id;
          const preview =
            chunk.text.length > 180
              ? chunk.text.slice(0, 180) + "..."
              : chunk.text;
          const sourceUrl = getSourceUrl?.(chunk);

          return (
            <div
              key={chunk.id}
              style={{
                display: "flex",
                alignItems: "flex-start",
                gap: "0",
                padding: "8px 0",
              }}
            >
              {/* Caret — clickable toggle */}
              <button
                onClick={() =>
                  setExpandedId(isExpanded ? null : chunk.id)
                }
                style={{
                  background: "none",
                  border: "none",
                  cursor: "pointer",
                  padding: "2px 8px 2px 0",
                  fontSize: "10px",
                  color: "#9ca3af",
                  flexShrink: 0,
                  lineHeight: "1.6",
                }}
                aria-label={isExpanded ? "Collapse" : "Expand"}
              >
                <span
                  style={{
                    display: "inline-block",
                    transition: "transform 150ms",
                    transform: isExpanded
                      ? "rotate(90deg)"
                      : "rotate(0deg)",
                  }}
                >
                  &#9654;
                </span>
              </button>

              {/* Content — normal selectable text */}
              <div style={{ flex: 1, minWidth: 0 }}>
                {/* Quote text */}
                <div
                  style={{
                    borderLeft: "2px solid #e5e7eb",
                    paddingLeft: "10px",
                    color: "#374151",
                    fontSize: "13px",
                    lineHeight: "1.6",
                    fontStyle: "italic",
                    ...(isExpanded
                      ? {}
                      : {
                          overflow: "hidden",
                          display: "-webkit-box",
                          WebkitLineClamp: 3,
                          WebkitBoxOrient: "vertical" as any,
                        }),
                  }}
                >
                  {isExpanded ? chunk.text : preview}
                </div>

                {/* Source link */}
                <div
                  style={{
                    marginTop: "4px",
                    paddingLeft: "12px",
                    fontSize: "12px",
                    lineHeight: "1.5",
                    color: "#9ca3af",
                  }}
                >
                  {sourceUrl ? (
                    <a
                      href={sourceUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      style={{
                        color: "#6b7280",
                        textDecoration: "underline",
                        textUnderlineOffset: "2px",
                      }}
                    >
                      {chunk.id}
                    </a>
                  ) : (
                    <span style={{ color: "#6b7280" }}>{chunk.id}</span>
                  )}
                  <span style={{ margin: "0 4px" }}>&middot;</span>
                  <span>{Math.round(chunk.relevance * 100)}% match</span>
                </div>

                {/* Expanded: rationale */}
                {isExpanded && chunk.rationale && (
                  <div
                    style={{
                      marginTop: "6px",
                      paddingLeft: "12px",
                      fontSize: "12px",
                      color: "#6b7280",
                      lineHeight: "1.6",
                    }}
                  >
                    {chunk.rationale}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export { SearchProgress as DeepSearchToolUI };
