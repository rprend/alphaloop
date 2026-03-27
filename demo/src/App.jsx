import { useState, startTransition } from "react";
import { SearchProgress, Citations } from "alphaloop/react";
import { REAL_SCENARIO_LIST, REAL_STRESS_SCENARIOS } from "../shared/realStressScenarios.mjs";

export default function App() {
  const [scenarioId, setScenarioId] = useState("recursive");
  const [query, setQuery] = useState(REAL_STRESS_SCENARIOS.recursive.query);
  const [mode, setMode] = useState("minScore");
  const [minScore, setMinScore] = useState(REAL_STRESS_SCENARIOS.recursive.minScore);
  const [topK, setTopK] = useState(24);
  const [maxIterations, setMaxIterations] = useState(
    REAL_STRESS_SCENARIOS.recursive.maxIterations,
  );
  const [maxExpandedQueries, setMaxExpandedQueries] = useState(
    REAL_STRESS_SCENARIOS.recursive.maxExpandedQueries,
  );
  const [maxContextTokens, setMaxContextTokens] = useState(100000);
  const [events, setEvents] = useState([]);
  const [citations, setCitations] = useState([]);
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState(null);
  const [isRunning, setIsRunning] = useState(false);

  const activeScenario = REAL_STRESS_SCENARIOS[scenarioId];

  async function runScenario() {
    setIsRunning(true);
    setEvents([]);
    setCitations([]);
    setSummary(null);
    setError(null);

    try {
      const response = await fetch("/api/run", {
        method: "POST",
        headers: {
          "content-type": "application/json",
        },
        body: JSON.stringify({
          scenarioId,
          query,
          minScore: mode === "minScore" ? Number(minScore) : undefined,
          topK: mode === "topK" ? Number(topK) : undefined,
          maxIterations: Number(maxIterations),
          maxExpandedQueries: Number(maxExpandedQueries),
          maxContextTokens: Number(maxContextTokens),
        }),
      });

      if (!response.ok || !response.body) {
        throw new Error(`Request failed (${response.status})`);
      }

      const payload = await response.json();
      if (payload.error) {
        throw new Error(payload.error);
      }

      startTransition(() => {
        setEvents(payload.events || []);
        setCitations(payload.result.chunks);
        setSummary({
          totalConsidered: payload.result.totalChunksConsidered,
          totalMatched: payload.result.totalChunksMatched,
          recursionDepth: payload.result.recursionDepth,
          shardCount: payload.result.shardCount,
          iterationsRun: payload.result.iterations.length,
          estimatedTokens: payload.searchStats.maxEstimatedTokens,
          embeddedChunks: payload.stats.embeddedChunks,
          virtualChunks: payload.stats.virtualChunks,
          strongMatches: payload.searchStats.maxStrongMatches,
          baseMatches: payload.searchStats.maxBaseMatches,
          modelId: payload.runtime.modelId,
          rerankModelId: payload.runtime.rerankModelId,
          embeddingModelId: payload.runtime.embeddingModelId,
          mode: payload.runtime.topKUsed == null ? "minScore" : "topK",
          minScoreUsed: payload.runtime.minScoreUsed,
          topKUsed: payload.runtime.topKUsed,
        });
      });
    } catch (runError) {
      startTransition(() => {
        setError(runError instanceof Error ? runError.message : "Unknown error");
      });
    } finally {
      startTransition(() => {
        setIsRunning(false);
      });
    }
  }

  return (
    <main className="shell">
      <section className="hero">
        <p className="eyebrow">alphaloop recursive stress lab</p>
        <h1>Run a real server-side alphaloop search against a real embedded corpus.</h1>
        <p className="lede">
          This demo uses the stock <code>SearchProgress</code> and{" "}
          <code>Citations</code> components only. The UI calls a Cloudflare
          Worker that runs alphaloop with live OpenAI embeddings and live OpenAI
          model calls.
        </p>
      </section>

      <section className="controls">
        <div className="scenarioGrid">
          {REAL_SCENARIO_LIST.map((scenario) => (
            <button
              key={scenario.id}
              className={scenario.id === scenarioId ? "scenario active" : "scenario"}
              onClick={() => {
                setScenarioId(scenario.id);
                setQuery(scenario.query);
                setMinScore(scenario.minScore);
                setMaxIterations(scenario.maxIterations);
                setMaxExpandedQueries(scenario.maxExpandedQueries);
              }}
            >
              <span>{scenario.label}</span>
              <strong>
                x{scenario.replicaCount.toLocaleString()} virtual replicas
              </strong>
            </button>
          ))}
        </div>

        <label className="queryBlock">
          <span>Stress query</span>
          <textarea
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            rows={3}
          />
        </label>

        <div className="runtimeGrid">
          <label className="field">
            <span>Retrieval mode</span>
            <select value={mode} onChange={(event) => setMode(event.target.value)}>
              <option value="minScore">Comprehensive minScore</option>
              <option value="topK">Focused topK</option>
            </select>
          </label>

          <label className="field">
            <span>minScore</span>
            <input
              type="number"
              step="0.01"
              value={minScore}
              disabled={mode !== "minScore"}
              onChange={(event) => setMinScore(event.target.value)}
            />
          </label>

          <label className="field">
            <span>topK</span>
            <input
              type="number"
              step="1"
              value={topK}
              disabled={mode !== "topK"}
              onChange={(event) => setTopK(event.target.value)}
            />
          </label>

          <label className="field">
            <span>Max iterations</span>
            <input
              type="number"
              step="1"
              value={maxIterations}
              onChange={(event) => setMaxIterations(event.target.value)}
            />
          </label>

          <label className="field">
            <span>Expanded queries</span>
            <input
              type="number"
              step="1"
              value={maxExpandedQueries}
              onChange={(event) => setMaxExpandedQueries(event.target.value)}
            />
          </label>

          <label className="field">
            <span>Max context tokens</span>
            <input
              type="number"
              step="1000"
              value={maxContextTokens}
              onChange={(event) => setMaxContextTokens(event.target.value)}
            />
          </label>
        </div>

        <button className="launch" onClick={runScenario} disabled={isRunning}>
          {isRunning ? "Running recursive search..." : "Run recursive search"}
        </button>
      </section>

      {error ? (
        <section className="panel">
          <div className="panelHeader">
            <h2>Run error</h2>
            <span>Server response</span>
          </div>
          <p className="errorText">{error}</p>
        </section>
      ) : null}

      {summary ? (
        <section className="telemetry">
          <Metric label="Matched chunks" value={summary.totalMatched.toLocaleString()} />
          <Metric label="Unique chunks" value={summary.totalConsidered.toLocaleString()} />
          <Metric label="Largest corpus" value={summary.strongMatches.toLocaleString()} />
          <Metric label="Largest base set" value={summary.baseMatches.toLocaleString()} />
          <Metric label="Estimated tokens" value={summary.estimatedTokens.toLocaleString()} />
          <Metric label="Shards" value={summary.shardCount.toLocaleString()} />
          <Metric label="Recursion depth" value={summary.recursionDepth.toLocaleString()} />
          <Metric label="Loop rounds" value={summary.iterationsRun.toLocaleString()} />
          <Metric label="Embedded chunks" value={summary.embeddedChunks.toLocaleString()} />
          <Metric label="Virtual chunks" value={summary.virtualChunks.toLocaleString()} />
          <Metric
            label="Retrieval"
            value={
              summary.mode === "topK"
                ? `topK ${summary.topKUsed}`
                : `minScore ${summary.minScoreUsed}`
            }
          />
          <Metric label="Model" value={summary.modelId} />
        </section>
      ) : null}

      <section className="panel">
        <div className="panelHeader">
          <h2>Search progress</h2>
          <span>{activeScenario.label}</span>
        </div>
        <SearchProgress events={events} isRunning={isRunning} />
      </section>

      <section className="panel">
        <div className="panelHeader">
          <h2>Citations</h2>
          <span>Top 12 ranked chunks</span>
        </div>
        <Citations chunks={citations} />
      </section>
    </main>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
