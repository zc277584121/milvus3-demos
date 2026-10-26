import { useCallback, useEffect, useMemo, useState } from "react";

import "./styles.css";

export const DEMO_ID = "minhash-near-duplicate";

export interface Preset {
  id: string;
  label: string;
  title: string;
  text: string;
  description: string;
}

export interface DatasetSummary {
  dataset_id: string;
  dataset_version: number;
  document_count: number;
  documents_sha256: string;
  presets_sha256: string;
}

export interface MinHashStatus {
  status: "ready" | "not_prepared";
  implementation_status: "implemented";
  execution_path: string;
  capability: string;
  collection_name: string;
  index_name: string;
  text_field: string;
  signature_field: string;
  metric_type: string;
  dataset: DatasetSummary;
  query_presets: Preset[];
  milvus: {
    milvus_uri: string;
    server_version: string;
    expected_server_version: string;
    target_collection: string;
    target_exists: boolean;
  };
  last_prepare: Record<string, unknown> | null;
}

export type DiffKind = "equal" | "removed" | "added";

export interface DiffSegment {
  kind: DiffKind;
  text: string;
}

export interface Diff {
  document_id: string;
  removed_count: number;
  added_count: number;
  segments: DiffSegment[];
}

export interface SearchResult {
  doc_id: string;
  title: string;
  content: string;
  estimated_jaccard: number;
  raw_distance: number;
  above_threshold: boolean;
}

export interface SearchResponse {
  status: "passed";
  execution_path: string;
  query: string;
  collection_name: string;
  index_name: string;
  text_field: string;
  signature_field: string;
  metric_type: string;
  threshold: number;
  limit: number;
  above_threshold_count: number;
  results: SearchResult[];
  diffs: Diff[];
  semantics: string;
  latency_ms: number;
}

const API_BASE = `${import.meta.env.BASE_URL}api/v1`;

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") {
        detail = body.detail;
      } else if (Array.isArray(body.detail)) {
        const messages = body.detail
          .map((item) =>
            typeof item === "object" && item !== null && "msg" in item
              ? String(item.msg)
              : "",
          )
          .filter(Boolean);
        if (messages.length > 0) {
          detail = messages.join("; ");
        }
      }
    } catch {
      // Preserve the HTTP status when a proxy returns a non-JSON response.
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export async function fetchMinHashStatus(): Promise<MinHashStatus> {
  return readJson<MinHashStatus>(
    await fetch(`${API_BASE}/status`, {
      headers: { Accept: "application/json" },
    }),
  );
}

export async function runMinHashSearch(request: {
  query: string;
  threshold: number;
  limit: number;
}): Promise<SearchResponse> {
  return readJson<SearchResponse>(
    await fetch(`${API_BASE}/search`, {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    }),
  );
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function scoreLabel(similarity: number): string {
  return similarity.toFixed(2);
}

// ---------------------------------------------------------------------------
// How MinHash estimates Jaccard similarity
//
// MinHash turns "do these two documents overlap?" into a coin-flip: apply the
// SAME random ordering to both documents' words and take the first word that
// appears in each. The two "first words" are equal exactly when the first word
// of the ordering belongs to BOTH sets, so the collision probability equals
// the true Jaccard similarity. Run many orderings and the collision rate
// converges to Jaccard.
//
// This explainer uses the REAL words of the incoming text and the selected
// candidate, a real random ordering (Fisher–Yates with a seeded PRNG), and a
// real count of collisions. The only "not Milvus" part is the hash/ordering:
// Milvus uses xxhash over the same idea, so its estimated Jaccard lands on the
// same value. That Milvus value is shown verbatim as the anchor.
// ---------------------------------------------------------------------------

const TRIAL_TARGET = 256;
const TRIAL_INTERVAL_MS = 24;

function shingles(text: string): string[] {
  // Word-level, punctuation-stripped, lowercased — mirrors the standard
  // analyzer's tokenization closely enough to make the overlap count honest.
  const normalized = text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
  if (!normalized) {
    return [];
  }
  const seen = new Set<string>();
  const result: string[] = [];
  for (const word of normalized.split(/\s+/)) {
    if (!seen.has(word)) {
      seen.add(word);
      result.push(word);
    }
  }
  return result;
}

// Mulberry32: a tiny seeded PRNG so the ordering is reproducible within a
// session but still a real random permutation of the vocabulary.
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle<T>(items: readonly T[], rand: () => number): T[] {
  const copy = items.slice();
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rand() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

interface MinHashTrial {
  // The shuffled vocabulary shown in the current trial (a short prefix).
  ordering: string[];
  // First word of the ordering that belongs to the query / candidate set.
  queryPick: string | null;
  candidatePick: string | null;
  collision: boolean;
}

function runTrial(
  querySet: Set<string>,
  candidateSet: Set<string>,
  vocabulary: readonly string[],
  rand: () => number,
): MinHashTrial {
  const ordering = shuffle(vocabulary, rand);
  let queryPick: string | null = null;
  let candidatePick: string | null = null;
  for (const word of ordering) {
    if (queryPick === null && querySet.has(word)) {
      queryPick = word;
    }
    if (candidatePick === null && candidateSet.has(word)) {
      candidatePick = word;
    }
    if (queryPick !== null && candidatePick !== null) {
      break;
    }
  }
  return {
    ordering,
    queryPick,
    candidatePick,
    collision: queryPick !== null && queryPick === candidatePick,
  };
}

function HowMinHashWorks({
  query,
  selectedResult,
}: {
  query: string;
  selectedResult: SearchResult | null;
}) {
  const queryShingles = useMemo(() => shingles(query), [query]);
  const candidateShingles = useMemo(
    () => (selectedResult ? shingles(selectedResult.content) : []),
    [selectedResult],
  );

  const querySet = useMemo(() => new Set(queryShingles), [queryShingles]);
  const candidateSet = useMemo(
    () => new Set(candidateShingles),
    [candidateShingles],
  );

  const intersection = useMemo(
    () => queryShingles.filter((word) => candidateSet.has(word)),
    [queryShingles, candidateSet],
  );
  const union = useMemo(
    () => new Set([...queryShingles, ...candidateShingles]),
    [queryShingles, candidateShingles],
  );
  const trueJaccard = useMemo(() => {
    if (union.size === 0) {
      return 0;
    }
    return intersection.length / union.size;
  }, [intersection.length, union.size]);

  const vocabulary = useMemo(() => [...union], [union]);

  const [trials, setTrials] = useState(0);
  const [collisions, setCollisions] = useState(0);
  const [current, setCurrent] = useState<MinHashTrial | null>(null);
  const [running, setRunning] = useState(false);

  // Reset the simulation whenever the compared pair changes.
  useEffect(() => {
    setTrials(0);
    setCollisions(0);
    setCurrent(null);
    setRunning(false);
  }, [query, selectedResult?.doc_id]);

  // Continuous run: one trial per tick until the target count is reached.
  useEffect(() => {
    if (!running) {
      return;
    }
    const id = window.setInterval(() => {
      setTrials((prev) => {
        const next = prev + 1;
        const rand = mulberry32(next * 0x9e3779b9 + 1);
        const trial = runTrial(querySet, candidateSet, vocabulary, rand);
        setCurrent(trial);
        if (trial.collision) {
          setCollisions((c) => c + 1);
        }
        if (next >= TRIAL_TARGET) {
          setRunning(false);
        }
        return next;
      });
    }, TRIAL_INTERVAL_MS);
    return () => window.clearInterval(id);
  }, [running, querySet, candidateSet, vocabulary]);

  const stepOnce = useCallback(() => {
    setTrials((prev) => {
      const next = prev + 1;
      const rand = mulberry32(next * 0x9e3779b9 + 1);
      const trial = runTrial(querySet, candidateSet, vocabulary, rand);
      setCurrent(trial);
      if (trial.collision) {
        setCollisions((c) => c + 1);
      }
      return next;
    });
  }, [querySet, candidateSet, vocabulary]);

  const reset = useCallback(() => {
    setTrials(0);
    setCollisions(0);
    setCurrent(null);
    setRunning(false);
  }, []);

  const collisionRate = trials > 0 ? collisions / trials : 0;
  const milvusValue = selectedResult?.estimated_jaccard ?? null;

  return (
    <section className="under-hood" data-testid="under-the-hood">
      <header className="under-hood-head">
        <h2>How MinHash estimates Jaccard similarity</h2>
        <span className="under-hood-subtitle">
          {selectedResult
            ? "One random ordering, one collision check — repeat to converge"
            : "Run detection to compare two real word sets"}
        </span>
      </header>

      {selectedResult ? (
        <div className="minhash-explainer">
          <div className="explainer-sets">
            <div className="word-set">
              <span className="word-set-label">
                Incoming text · {queryShingles.length} words
              </span>
              <div className="word-chips">
                {queryShingles.map((word) => (
                  <span
                    key={word}
                    className={`word-chip${
                      intersection.includes(word) ? " shared" : ""
                    }`}
                  >
                    {word}
                  </span>
                ))}
              </div>
            </div>
            <div className="word-set">
              <span className="word-set-label">
                {selectedResult.doc_id} · {candidateShingles.length} words
              </span>
              <div className="word-chips">
                {candidateShingles.map((word) => (
                  <span
                    key={word}
                    className={`word-chip${
                      intersection.includes(word) ? " shared" : ""
                    }`}
                  >
                    {word}
                  </span>
                ))}
              </div>
            </div>
          </div>

          <p className="explainer-jaccard">
            {intersection.length} shared words / {union.size} distinct words ={" "}
            <strong className="jaccard-value">
              true Jaccard {trueJaccard.toFixed(2)}
            </strong>
          </p>

          <div className="explainer-trial">
            <div className="trial-head">
              <span className="trial-title">
                {current ? `Trial ${trials}` : "Random ordering"}
              </span>
              <span
                className={`trial-outcome${
                  current ? (current.collision ? " hit" : " miss") : ""
                }`}
              >
                {current
                  ? current.collision
                    ? "✦ collision"
                    : "no collision"
                  : "—"}
              </span>
            </div>
            <div className="trial-ordering" aria-hidden="true">
              {(current?.ordering ?? vocabulary).slice(0, 10).map((word) => (
                <span
                  key={word}
                  className={`ordering-chip${
                    current?.queryPick === word
                      ? " query-pick"
                      : current?.candidatePick === word
                        ? " candidate-pick"
                        : ""
                  }`}
                >
                  {word}
                </span>
              ))}
              {(current?.ordering ?? vocabulary).length > 10 && (
                <span className="ordering-more">…</span>
              )}
            </div>
            <p className="trial-explanation">
              {current ? (
                <>
                  first query word{" "}
                  <code className="pick-label query">
                    {current.queryPick ?? "—"}
                  </code>
                  {" · "}first candidate word{" "}
                  <code className="pick-label candidate">
                    {current.candidatePick ?? "—"}
                  </code>
                  {" · "}
                  {current.collision
                    ? "same word → this trial collides"
                    : "different words → this trial misses"}
                </>
              ) : (
                "Apply one random ordering to both word sets and take the first word each contains."
              )}
            </p>
          </div>

          <div className="explainer-controls">
            <button
              className="secondary-button"
              onClick={stepOnce}
              disabled={running}
              type="button"
            >
              One trial
            </button>
            <button
              className="primary-button"
              onClick={() => {
                if (trials < TRIAL_TARGET) {
                  setRunning(true);
                }
              }}
              disabled={running || trials >= TRIAL_TARGET}
              type="button"
            >
              {running ? "Running…" : `Run ${TRIAL_TARGET} trials`}
            </button>
            <button className="secondary-button" onClick={reset} type="button">
              Reset
            </button>
            <span className="trial-counter">
              {collisions} / {trials} collisions
            </span>
          </div>

          <div className="explainer-anchor">
            <div className="anchor-card">
              <span className="anchor-label">Simulated collision rate</span>
              <span className="anchor-value">
                {trials > 0 ? collisionRate.toFixed(2) : "—"}
              </span>
            </div>
            <div className="anchor-card">
              <span className="anchor-label">Milvus estimated Jaccard</span>
              <span className="anchor-value milvus">
                {milvusValue !== null ? scoreLabel(milvusValue) : "—"}
              </span>
            </div>
            <p className="anchor-note">
              Both converge to the true Jaccard; the simulation uses a seeded
              PRNG while Milvus uses xxhash.
            </p>
          </div>
        </div>
      ) : (
        <p className="under-hood-idle">
          Run detection and select a candidate to see its words compared against
          the incoming text.
        </p>
      )}
    </section>
  );
}

function IncomingPanel({
  title,
  content,
  onTitleChange,
  onContentChange,
}: {
  title: string;
  content: string;
  onTitleChange: (value: string) => void;
  onContentChange: (value: string) => void;
}) {
  return (
    <section className="panel incoming-panel">
      <header className="panel-header">
        <h2>Incoming document</h2>
        <span className="panel-subtitle">Paste or edit your text</span>
      </header>
      <label className="title-field">
        <span className="field-label">Title</span>
        <input
          value={title}
          onChange={(event) => onTitleChange(event.target.value)}
          placeholder="A weekend by the river"
        />
      </label>
      <label className="content-field">
        <span className="field-label">Text</span>
        <textarea
          value={content}
          onChange={(event) => onContentChange(event.target.value)}
          placeholder="Paste a document to check for near-duplicates…"
        />
      </label>
    </section>
  );
}

function CandidatePanel({
  results,
  selectedId,
  threshold,
  onSelect,
}: {
  results: SearchResult[];
  selectedId: string | null;
  threshold: number;
  onSelect: (docId: string) => void;
}) {
  return (
    <section className="panel candidates-panel">
      <header className="panel-header">
        <h2>Retrieved candidates</h2>
        <span className="panel-subtitle">
          {results.length} candidate{results.length === 1 ? "" : "s"}
        </span>
      </header>
      {results.length === 0 ? (
        <p className="empty-note">No candidates retrieved yet.</p>
      ) : (
        <div className="candidate-list">
          {results.map((result) => {
            const selected = result.doc_id === selectedId;
            return (
              <button
                key={result.doc_id}
                className={`candidate-card${selected ? " selected" : ""}`}
                onClick={() => onSelect(result.doc_id)}
                type="button"
              >
                <div className="candidate-head">
                  <span className="candidate-name">{result.doc_id}</span>
                  <span
                    className={`score-badge${result.above_threshold ? " above" : " below"}`}
                  >
                    {scoreLabel(result.estimated_jaccard)}
                  </span>
                </div>
                <p className="candidate-excerpt">
                  {result.content.slice(0, 120)}…
                </p>
                <div className="candidate-foot">
                  <span
                    className={`threshold-badge${result.above_threshold ? " above" : " below"}`}
                  >
                    {result.above_threshold
                      ? "Above threshold"
                      : "Below threshold"}
                  </span>
                  <span className="candidate-title">{result.title}</span>
                </div>
              </button>
            );
          })}
        </div>
      )}
      {results.length > 0 && (
        <p className="candidates-note">
          Select a candidate to inspect the text.
        </p>
      )}
    </section>
  );
}

function DiffView({ diff, result }: { diff: Diff; result: SearchResult }) {
  return (
    <section className="panel compare-panel">
      <header className="panel-header">
        <h2>Compare with selected</h2>
        <span className="panel-subtitle">{diff.document_id}</span>
      </header>
      <div className="compare-meta">
        <span className="compare-filename">{diff.document_id}</span>
        <span className="score-pill">
          Estimated Jaccard {scoreLabel(result.estimated_jaccard)}
        </span>
      </div>
      <div className="diff-legend">
        <span className="legend-removed">Removed</span>
        <span className="legend-added">Added</span>
      </div>
      <div className="diff-body">
        {diff.segments.map((segment, index) => (
          <span key={index} className={`diff-segment ${segment.kind}`}>
            {segment.text}{" "}
          </span>
        ))}
      </div>
      <p className="compare-note">
        Text differences are highlighted by the app.
      </p>
    </section>
  );
}

export function MinHashDemoPage() {
  const [status, setStatus] = useState<MinHashStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [presetId, setPresetId] = useState<string>("");
  const [title, setTitle] = useState("A Weekend by the River");
  const [content, setContent] = useState(
    "On Sunday, we followed the river path to the old bridge. The morning market was already busy, so we bought bread and coffee before continuing. Beyond the bridge, a narrow trail led to a quiet garden. We spent the afternoon there and returned to town before sunset.",
  );
  const [threshold, setThreshold] = useState(0.85);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const presets = useMemo(() => status?.query_presets ?? [], [status]);

  // Load status, then immediately run the first preset so the demo opens with
  // real results instead of an empty workspace the visitor must click through.
  useEffect(() => {
    let cancelled = false;
    fetchMinHashStatus()
      .then(async (next) => {
        if (cancelled) {
          return;
        }
        setStatus(next);
        const first = next.query_presets[0];
        if (!first) {
          return;
        }
        setPresetId(first.id);
        setTitle(first.title);
        setContent(first.text);

        setLoading(true);
        try {
          const initial = await runMinHashSearch({
            query: first.text,
            threshold: clamp(0.85, 0, 1),
            limit: 10,
          });
          if (!cancelled) {
            setResponse(initial);
            const firstHit = initial.results?.[0];
            setSelectedId(firstHit ? firstHit.doc_id : null);
          }
        } catch (err: unknown) {
          if (!cancelled) {
            setError(err instanceof Error ? err.message : String(err));
          }
        } finally {
          if (!cancelled) {
            setLoading(false);
          }
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const applyPreset = useCallback(
    (id: string) => {
      setPresetId(id);
      const preset = presets.find((item) => item.id === id);
      if (preset) {
        setTitle(preset.title);
        setContent(preset.text);
        setSelectedId(null);
        setResponse(null);
      }
    },
    [presets],
  );

  const reset = useCallback(() => {
    const preset = presets.find((item) => item.id === presetId);
    if (preset) {
      setTitle(preset.title);
      setContent(preset.text);
    }
    setThreshold(0.85);
    setResponse(null);
    setSelectedId(null);
  }, [presets, presetId]);

  const detect = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const next = await runMinHashSearch({
        query: content,
        threshold: clamp(threshold, 0, 1),
        limit: 10,
      });
      setResponse(next);
      setSelectedId(next.results.length > 0 ? next.results[0].doc_id : null);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
      setResponse(null);
      setSelectedId(null);
    } finally {
      setLoading(false);
    }
  }, [content, threshold]);

  const selectedResult = useMemo(
    () =>
      response?.results.find((result) => result.doc_id === selectedId) ?? null,
    [response, selectedId],
  );
  const selectedDiff = useMemo(
    () =>
      response?.diffs.find((diff) => diff.document_id === selectedId) ?? null,
    [response, selectedId],
  );

  const aboveCount = response?.above_threshold_count ?? 0;

  return (
    <div className="demo-page">
      <header className="topbar">
        <div className="brand">
          <span className="wordmark">milvus</span>
          <span className="version-badge">3.0</span>
        </div>
        <h1 className="topbar-title">Near-Duplicate Detection with MinHash</h1>
        <a
          className="source-link"
          href="https://github.com/zc277584121/milvus3-demos"
          target="_blank"
          rel="noreferrer"
        >
          <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
            <path
              fill="currentColor"
              d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"
            />
          </svg>
          <span>Source</span>
        </a>
      </header>

      <div className="intro-strip">
        <p>Find copied and lightly edited documents.</p>
        <span className="intro-badge">MinHash + LSH · No embedding model</span>
      </div>

      <div className="toolbar">
        <label className="preset-control">
          <span className="toolbar-label">Example</span>
          <select
            value={presetId}
            onChange={(event) => applyPreset(event.target.value)}
          >
            {presets.map((preset) => (
              <option key={preset.id} value={preset.id}>
                {preset.label}
              </option>
            ))}
          </select>
        </label>
        <button className="secondary-button" onClick={reset} type="button">
          Reset
        </button>
        <button
          className="primary-button"
          onClick={detect}
          disabled={loading}
          type="button"
        >
          {loading ? "Detecting…" : "Detect duplicates"}
        </button>
        <span className="result-count">
          {aboveCount} candidate{aboveCount === 1 ? "" : "s"} above threshold
        </span>
      </div>

      {error && <p className="error-banner">{error}</p>}

      <div className="threshold-control">
        <label htmlFor="threshold-slider">
          <span className="threshold-label">Detection threshold</span>
          <input
            id="threshold-slider"
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={threshold}
            onChange={(event) => setThreshold(Number(event.target.value))}
          />
          <span className="threshold-value">{threshold.toFixed(2)}</span>
        </label>
        <p className="threshold-help">Estimated Jaccard similarity cut-off.</p>
      </div>

      <div className="scroll-main">
        <div className="workspace">
          <IncomingPanel
            title={title}
            content={content}
            onTitleChange={setTitle}
            onContentChange={setContent}
          />
          <CandidatePanel
            results={response?.results ?? []}
            selectedId={selectedId}
            threshold={threshold}
            onSelect={setSelectedId}
          />
          {selectedResult && selectedDiff ? (
            <DiffView diff={selectedDiff} result={selectedResult} />
          ) : (
            <section className="panel compare-panel">
              <header className="panel-header">
                <h2>Compare with selected</h2>
                <span className="panel-subtitle">No document selected</span>
              </header>
              <p className="empty-note">
                Run detection and select a candidate to inspect the text
                differences.
              </p>
            </section>
          )}
        </div>

        <HowMinHashWorks query={content} selectedResult={selectedResult} />

        <footer className="footer">
          <p>Review candidates before merging or removing content.</p>
          <p className="footer-meta">Illustrative scores · Concept data</p>
        </footer>
      </div>
    </div>
  );
}
