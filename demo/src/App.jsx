import { useState, startTransition } from "react";
import { createAlphaloop } from "alphaloop";
import { SearchProgress, Citations } from "alphaloop/react";
import { createSyntheticLanguageModel } from "../shared/fakeModel.mjs";
import {
  createScenarioDataset,
  searchScenario,
  STRESS_SCENARIOS,
} from "../shared/stressData.mjs";

const model = createSyntheticLanguageModel();
const scenarioEntries = Object.values(STRESS_SCENARIOS);

function buildLoop(dataset, searchStatsRef) {
  return createAlphaloop({
    model,
    rerankModel: model,
    minScore: dataset.scenario.minScore,
    maxExpandedQueries: 4,
    maxIterations: 1,
    search: async (query, options) => {
      const page = await searchScenario(dataset, query, options);
      searchStatsRef.totalStrongMatches = Math.max(
        searchStatsRef.totalStrongMatches,
        page.totalStrongMatches,
      );
      searchStatsRef.estimatedTokens = Math.max(
        searchStatsRef.estimatedTokens,
        page.estimatedTokens,
      );
      return {
        chunks: page.chunks,
        nextCursor: page.nextCursor,
      };
    },
  });
}

export default function App() {
  const [scenarioId, setScenarioId] = useState("branch8");
  const [query, setQuery] = useState(STRESS_SCENARIOS.branch8.query);
  const [events, setEvents] = useState([]);
  const [citations, setCitations] = useState([]);
  const [summary, setSummary] = useState(null);
  const [isRunning, setIsRunning] = useState(false);

  const activeScenario = STRESS_SCENARIOS[scenarioId];

  async function runScenario() {
    setIsRunning(true);
    setEvents([]);
    setCitations([]);
    setSummary(null);

    const dataset = createScenarioDataset(scenarioId);
    const searchStats = {
      totalStrongMatches: 0,
      estimatedTokens: 0,
    };
    const loop = buildLoop(dataset, searchStats);
    const stream = loop.stream(query, { minScore: activeScenario.minScore });
    let final;

    while (true) {
      const next = await stream.next();
      if (next.done) {
        final = next.value;
        break;
      }

      startTransition(() => {
        setEvents((current) => [...current, next.value]);
      });
    }

    startTransition(() => {
      setCitations(final.chunks.slice(0, 12));
      setSummary({
        totalConsidered: final.totalChunksConsidered,
        totalMatched: final.totalChunksMatched,
        recursionDepth: final.recursionDepth,
        shardCount: final.shardCount,
        estimatedTokens: searchStats.estimatedTokens,
        documents: dataset.chunks.length,
      });
      setIsRunning(false);
    });
  }

  return (
    <main className="shell">
      <section className="hero">
        <p className="eyebrow">alphaloop recursive stress lab</p>
        <h1>Force recursive branching on a synthetic embedded corpus.</h1>
        <p className="lede">
          This demo uses only the stock <code>SearchProgress</code> and{" "}
          <code>Citations</code> components, backed by an in-memory vector
          search and a deterministic local model.
        </p>
      </section>

      <section className="controls">
        <div className="scenarioGrid">
          {scenarioEntries.map((scenario) => (
            <button
              key={scenario.id}
              className={scenario.id === scenarioId ? "scenario active" : "scenario"}
              onClick={() => {
                setScenarioId(scenario.id);
                setQuery(scenario.query);
              }}
            >
              <span>{scenario.label}</span>
              <strong>{scenario.documents.toLocaleString()} docs</strong>
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

        <button className="launch" onClick={runScenario} disabled={isRunning}>
          {isRunning ? "Running recursive search..." : "Run recursive search"}
        </button>
      </section>

      {summary ? (
        <section className="telemetry">
          <Metric label="Strong matches" value={summary.totalMatched.toLocaleString()} />
          <Metric label="Unique chunks" value={summary.totalConsidered.toLocaleString()} />
          <Metric label="Estimated tokens" value={summary.estimatedTokens.toLocaleString()} />
          <Metric label="Shards" value={summary.shardCount.toLocaleString()} />
          <Metric label="Recursion depth" value={summary.recursionDepth.toLocaleString()} />
          <Metric label="Corpus docs" value={summary.documents.toLocaleString()} />
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
