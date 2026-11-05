import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

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

// These mirror the fixed MinHash configuration the backend builds the
// collection with (num_hashes=256, mh_lsh_band=64). The explainer keeps them
// in sync so the bands it draws correspond to the real index shape:
// 256 slots / 64 bands = 4 hashes per band.
// Mirrors the backend's DEFAULT_THRESHOLD so the slider starts where the API does.
const DEFAULT_THRESHOLD = 0.5;

// Real Milvus calls behind each step, shown on hover.
type MilvusStep = "create" | "ingest" | "search";

const MILVUS_CODE: Record<MilvusStep, string> = {
  create: `schema.add_field("content", VARCHAR, enable_analyzer=True)
schema.add_field("signature", BINARY_VECTOR, dim=8192)
schema.add_function(FunctionType.MINHASH,
                    num_hashes=256, shingle_size=1)
indexes.add_index("signature", "MINHASH_LSH", "MHJACCARD")`,
  ingest: `client.insert(COLLECTION, data=[{
  "doc_id": "night-market-004.txt",
  "title":  "The Night Market",
  "content": "The night market opened under strings of ..."
}])
client.flush()
client.load_collection()`,
  search: `client.search(COLLECTION, data=[query_text],
              anns_field="signature", metric_type="MHJACCARD",
              params={"refine_k": 10}, limit=10)`,
};

// Minimal Python syntax highlighting for the hover overlay. Kept small on
// purpose: enough color to read as code, not a full parser.
const PY_TOKEN =
  /(#[^\n]*)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(\b(?:from|import|def|return|True|False|None)\b)|(\b(?:FunctionType|DataType|Function|MINHASH|MINHASH_LSH|MHJACCARD|BINARY_VECTOR|VARCHAR)\b)|(\b\d+(?:\.\d+)?\b)|(\b[a-z_][a-z0-9_]*(?=\())|(\b[a-z_][a-z0-9_]*(?=\s*=))/g;

function highlightPython(code: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  let last = 0;
  for (const match of code.matchAll(PY_TOKEN)) {
    const index = match.index ?? 0;
    if (index > last) {
      nodes.push(code.slice(last, index));
    }
    const [, comment, str, keyword, enumName, num, fn, param] = match;
    let className = "";
    if (comment) {
      className = "py-comment";
    } else if (str) {
      className = "py-string";
    } else if (keyword) {
      className = "py-keyword";
    } else if (enumName) {
      className = "py-enum";
    } else if (num) {
      className = "py-number";
    } else if (fn) {
      className = "py-func";
    } else if (param) {
      className = "py-param";
    }
    const text = match[0];
    nodes.push(
      className ? (
        <span key={index} className={className}>
          {text}
        </span>
      ) : (
        text
      ),
    );
    last = index + text.length;
  }
  if (last < code.length) {
    nodes.push(code.slice(last));
  }
  return nodes;
}

// One offline setup step: a number, a title, and the real call it runs.
function StepCard({
  index,
  title,
  code,
}: {
  index: number;
  title: string;
  code: string;
}) {
  return (
    <article className="step-card">
      <header className="step-card-head">
        <span className="step-num">{index}</span>
        <span className="step-name">{title}</span>
      </header>
      <pre className="step-code">
        <code>{highlightPython(code)}</code>
      </pre>
    </article>
  );
}

// A short hover hint that reveals the Milvus call behind a page element. The
// panel keeps the working UI; the code is there for readers who want it.
function CodeHint({
  code,
  label,
  children,
}: {
  code: string;
  label: string;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  return (
    <span
      className="code-hint"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
      tabIndex={0}
      role="button"
      aria-label={label}
    >
      {children}
      {open && (
        <span className="code-hint-pop" role="tooltip">
          <span className="code-hint-label">{label}</span>
          <pre>
            <code>{highlightPython(code)}</code>
          </pre>
        </span>
      )}
    </span>
  );
}

function IncomingPanel({
  content,
  onContentChange,
  threshold,
  onThresholdChange,
}: {
  content: string;
  onContentChange: (value: string) => void;
  threshold: number;
  onThresholdChange: (value: number) => void;
}) {
  return (
    <section className="panel incoming-panel">
      <header className="panel-header">
        <h2>Incoming document</h2>
        <span className="panel-subtitle">Paste or edit, then detect</span>
      </header>
      <label className="content-field">
        <textarea
          value={content}
          onChange={(event) => onContentChange(event.target.value)}
          placeholder="Paste a document to check for near-duplicates…"
        />
      </label>
      <label className="threshold-row" htmlFor="threshold-slider">
        <span className="threshold-row-label">Threshold</span>
        <input
          id="threshold-slider"
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={threshold}
          onChange={(event) => onThresholdChange(Number(event.target.value))}
        />
        <span className="threshold-row-value">{threshold.toFixed(2)}</span>
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
            // Judge against the live slider, not the threshold that happened to
            // be sent with the last request, so dragging updates the list.
            const aboveThreshold = result.estimated_jaccard >= threshold;
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
                    className={`score-badge${aboveThreshold ? " above" : " below"}`}
                  >
                    {scoreLabel(result.estimated_jaccard)}
                  </span>
                </div>
                <p className="candidate-excerpt">
                  {result.content.slice(0, 120)}…
                </p>
                <div className="candidate-foot">
                  <span
                    className={`threshold-badge${aboveThreshold ? " above" : " below"}`}
                  >
                    {aboveThreshold ? "Above threshold" : "Below threshold"}
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
  const [content, setContent] = useState(
    "On Sunday, we followed the river path to the old bridge. The morning market was already busy, so we bought bread and coffee before continuing. Beyond the bridge, a narrow trail led to a quiet garden. We spent the afternoon there and returned to town before sunset.",
  );
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD);
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
        setContent(first.text);

        setLoading(true);
        try {
          const initial = await runMinHashSearch({
            query: first.text,
            threshold: clamp(DEFAULT_THRESHOLD, 0, 1),
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

  // Shared search runner. Presets pass their own text so the request does not
  // race the `setContent` state update; the button passes the edited text.
  const runSearch = useCallback(
    async (queryText: string) => {
      setLoading(true);
      setError(null);
      try {
        const next = await runMinHashSearch({
          query: queryText,
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
    },
    [threshold],
  );

  // Picking an example runs immediately; hand-edited text stays manual.
  const applyPreset = useCallback(
    (id: string) => {
      setPresetId(id);
      const preset = presets.find((item) => item.id === id);
      if (preset) {
        setContent(preset.text);
        setSelectedId(null);
        setResponse(null);
        void runSearch(preset.text);
      }
    },
    [presets, runSearch],
  );

  const reset = useCallback(() => {
    const preset = presets.find((item) => item.id === presetId);
    if (preset) {
      setContent(preset.text);
    }
    setThreshold(DEFAULT_THRESHOLD);
    setResponse(null);
    setSelectedId(null);
  }, [presets, presetId]);

  const detect = useCallback(() => {
    void runSearch(content);
  }, [runSearch, content]);

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

  // Count against the live slider so the toolbar figure matches the badges.
  const aboveCount = useMemo(
    () =>
      (response?.results ?? []).filter(
        (result) => result.estimated_jaccard >= threshold,
      ).length,
    [response, threshold],
  );

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

      {error && <p className="error-banner">{error}</p>}

      <div className="scroll-main">
        <section className="phase">
          <div className="phase-head">
            <span className="phase-tag offline">OFFLINE</span>
          </div>

          <div className="phase-grid">
            <StepCard
              index={1}
              title="Create index"
              code={MILVUS_CODE.create}
            />
            <StepCard index={2} title="Ingest" code={MILVUS_CODE.ingest} />
          </div>
        </section>

        <section className="phase">
          <div className="phase-head">
            <span className="phase-tag online">ONLINE</span>
          </div>

          <div className="online-card">
            <div className="online-head">
              <span className="step-num">3</span>
              <span className="step-name">Search</span>
            </div>

            <pre className="online-code">
              <code>{highlightPython(MILVUS_CODE.search)}</code>
            </pre>

            <div className="query-bar">
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
              <button
                className="secondary-button"
                onClick={reset}
                type="button"
              >
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
                {aboveCount} candidate{aboveCount === 1 ? "" : "s"} above
                threshold
              </span>
            </div>

            <div className="workspace">
              <IncomingPanel
                content={content}
                onContentChange={setContent}
                threshold={threshold}
                onThresholdChange={setThreshold}
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
          </div>
        </section>

        <footer className="footer">
          <p>Review candidates before merging or removing content.</p>
          <p className="footer-meta">Illustrative scores · Concept data</p>
        </footer>
      </div>
    </div>
  );
}
