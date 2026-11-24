import {
  Fragment,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react";

import { Walkthrough } from "./Walkthrough";
import milvusWordmark from "./assets/milvus-wordmark.svg";
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

// Mirrors the backend's DEFAULT_THRESHOLD so the slider starts where the API does.
const DEFAULT_THRESHOLD = 0.5;

// How many candidates Milvus returns. The list shows exactly this set, so the
// count in the header always equals what is on screen.
const SEARCH_LIMIT = 6;

// The three calls this demo makes, shown verbatim below the fold.
const MILVUS_CODE = {
  create: `schema = client.create_schema(auto_id=False)
schema.add_field("content", VARCHAR, max_length=2000, enable_analyzer=True)
schema.add_field("signature", BINARY_VECTOR, dim=8192)

schema.add_function(
    Function(
        function_type=FunctionType.MINHASH,
        input_field_names=["content"],
        output_field_names=["signature"],
        params={"num_hashes": 256, "shingle_size": 1},
    )
)

indexes.add_index("signature", index_type="MINHASH_LSH", metric_type="MHJACCARD")`,
  ingest: `client.insert(
    COLLECTION,
    data=[
        {
            "doc_id": "night-market-004.txt",
            "title": "The Night Market",
            "content": "The night market opened under strings of ...",
        }
    ],
)
client.flush()
client.load_collection()`,
  search: `hits = client.search(
    COLLECTION,
    data=[query_text],
    anns_field="signature",
    search_params={
        "metric_type": "MHJACCARD",
        "params": {"refine_k": 10},
    },
    limit=10,
)`,
} as const;

const PY_TOKEN =
  /(#[^\n]*)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(\b(?:True|False|None)\b)|(\b(?:FunctionType|DataType|MINHASH|MINHASH_LSH|MHJACCARD|BINARY_VECTOR|VARCHAR)\b)|(\b\d+(?:\.\d+)?\b)|(\b[a-z_][a-z0-9_]*(?=\())/g;

function highlightPython(code: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  let last = 0;
  for (const match of code.matchAll(PY_TOKEN)) {
    const index = match.index ?? 0;
    if (index > last) {
      nodes.push(code.slice(last, index));
    }
    const [, comment, str, keyword, enumName, num, fn] = match;
    const className = comment
      ? "py-comment"
      : str
        ? "py-string"
        : keyword
          ? "py-keyword"
          : enumName
            ? "py-enum"
            : num
              ? "py-number"
              : fn
                ? "py-func"
                : "";
    nodes.push(
      className ? (
        <span key={index} className={className}>
          {match[0]}
        </span>
      ) : (
        match[0]
      ),
    );
    last = index + match[0].length;
  }
  if (last < code.length) {
    nodes.push(code.slice(last));
  }
  return nodes;
}

const STEPS = [
  { n: 1, title: "Create index", code: MILVUS_CODE.create },
  { n: 2, title: "Ingest", code: MILVUS_CODE.ingest },
  { n: 3, title: "Search", code: MILVUS_CODE.search },
] as const;

function CandidatePanel({
  results,
  selectedId,
  threshold,
  onSelect,
  pending,
  loading,
}: {
  results: SearchResult[];
  selectedId: string | null;
  threshold: number;
  onSelect: (docId: string) => void;
  pending: boolean;
  loading: boolean;
}) {
  const aboveThreshold = results.filter(
    (result) => result.estimated_jaccard >= threshold,
  );

  function renderCandidates(candidates: SearchResult[]) {
    return (
      <ul className="candidate-list">
        {candidates.map((result) => {
          const above = result.estimated_jaccard >= threshold;
          return (
            <li key={result.doc_id}>
              <button
                className={`candidate-button${above ? " is-above" : " is-below"}`}
                aria-pressed={result.doc_id === selectedId}
                onClick={() => onSelect(result.doc_id)}
                title={`${result.title} · ${above ? "above" : "below"} threshold`}
                type="button"
              >
                <span className="file-icon" aria-hidden="true" />
                <span className="candidate-name">{result.doc_id}</span>
                <span className="candidate-score">
                  {scoreLabel(result.estimated_jaccard)}
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    );
  }

  return (
    <section className="candidates-panel" aria-labelledby="matches-heading">
      <header className="candidates-header">
        <h2 id="matches-heading">Candidate documents</h2>
        <p className="result-count" aria-live="polite">
          {loading
            ? "Searching…"
            : pending
              ? "Edits pending"
              : `${aboveThreshold.length} of ${results.length} above threshold`}
        </p>
      </header>
      {results.length === 0 ? (
        <p className="empty-note">
          {loading
            ? "Finding candidates…"
            : pending
              ? "Apply changes to update candidates."
              : "No candidates retrieved yet."}
        </p>
      ) : (
        renderCandidates(results)
      )}
    </section>
  );
}

interface DocumentEditor {
  active: boolean;
  loading: boolean;
  value: string;
  onStart: () => void;
  onChange: (value: string) => void;
  onCancel: () => void;
  onApply: () => Promise<boolean>;
}

function ComparedDocument({
  title,
  text,
  segments,
  kind,
  count,
  editor,
}: {
  title: string;
  text: string;
  segments: DiffSegment[];
  kind: "added" | "removed";
  count: number;
  editor?: DocumentEditor;
}) {
  const editButtonRef = useRef<HTMLButtonElement>(null);
  const Mark = kind === "added" ? "ins" : "del";
  const words = Array.from(text.matchAll(/\S+/g));
  let wordOffset = 0;
  let textOffset = 0;

  // Diff segments normalize whitespace. Map their words back to the source
  // so line breaks, spaces, and punctuation remain part of the document.
  const parts = segments
    .filter((segment) => segment.kind === "equal" || segment.kind === kind)
    .map((segment, index) => {
      const wordCount = segment.text.match(/\S+/g)?.length ?? 0;
      if (wordCount === 0) return null;
      const first = words[wordOffset];
      wordOffset += wordCount;
      const last = words[wordOffset - 1];
      if (!first || !last) return null;

      const spacing = text.slice(textOffset, first.index);
      const end = last.index + last[0].length;
      const content = text.slice(first.index, end);
      textOffset = end;

      return (
        <Fragment key={index}>
          {spacing}
          {segment.kind === kind ? <Mark>{content}</Mark> : content}
        </Fragment>
      );
    });

  return (
    <article className={`document-pane ${kind}`}>
      <header className="document-header">
        <h3>{title}</h3>
        <div className="document-header-actions">
          {count > 0 && !editor?.active && (
            <span className="change-count">
              {count} {count === 1 ? "word" : "words"} {kind}
            </span>
          )}
          {editor && !editor.active && (
            <button
              ref={editButtonRef}
              className="document-edit-button"
              type="button"
              aria-label="Edit incoming document"
              disabled={editor.loading}
              onClick={editor.onStart}
            >
              Edit
            </button>
          )}
        </div>
      </header>
      {editor?.active ? (
        <div className="document-editor">
          <textarea
            className="document-edit-field"
            aria-label="Incoming document"
            value={editor.value}
            onChange={(event) => editor.onChange(event.target.value)}
            disabled={editor.loading}
            placeholder="Enter a document to compare…"
            maxLength={2000}
            rows={8}
            autoFocus
          />
          <div className="document-editor-actions">
            <button
              className="primary-button"
              type="button"
              disabled={editor.loading || !editor.value.trim()}
              onClick={async () => {
                if (await editor.onApply()) {
                  window.setTimeout(() => editButtonRef.current?.focus(), 0);
                }
              }}
            >
              {editor.loading ? "Detecting…" : "Apply changes"}
            </button>
            <button
              className="secondary-button"
              type="button"
              disabled={editor.loading}
              onClick={() => {
                editor.onCancel();
                window.setTimeout(() => editButtonRef.current?.focus(), 0);
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <p className="document-text">
          {parts}
          {text.slice(textOffset)}
        </p>
      )}
    </article>
  );
}

function DiffView({
  diff,
  result,
  query,
  editor,
  pending,
}: {
  diff: Diff | null;
  result: SearchResult | null;
  query: string;
  editor: DocumentEditor;
  pending: boolean;
}) {
  return (
    <section className="compare-panel" aria-labelledby="comparison-heading">
      <header className="panel-header">
        <h2 id="comparison-heading">What changed</h2>
        {result ? (
          <p className="comparison-score">
            Estimated Jaccard{" "}
            <strong>{scoreLabel(result.estimated_jaccard)}</strong>
          </p>
        ) : (
          <p className="comparison-status" role="status">
            {editor.loading ? "Comparing…" : pending ? "Edits pending" : ""}
          </p>
        )}
      </header>
      <div className="split-diff">
        <ComparedDocument
          title="Incoming document"
          kind="added"
          count={diff?.added_count ?? 0}
          text={query}
          segments={diff?.segments ?? []}
          editor={editor}
        />
        {result && diff ? (
          <ComparedDocument
            title={diff.document_id}
            kind="removed"
            count={diff.removed_count}
            text={result.content}
            segments={diff.segments}
          />
        ) : (
          <article className="document-pane">
            <header className="document-header">
              <h3>Matching document</h3>
            </header>
            <p className="empty-note">
              {editor.loading
                ? "Finding matching documents…"
                : pending
                  ? "Apply changes to update the comparison."
                  : "Choose an example or run detection to find matching documents."}
            </p>
          </article>
        )}
      </div>
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
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState(false);
  const [originalContent, setOriginalContent] = useState("");

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
          setLoading(false);
          return;
        }
        setPresetId(first.id);
        setContent(first.text);

        setLoading(true);
        try {
          const initial = await runMinHashSearch({
            query: first.text,
            threshold: clamp(DEFAULT_THRESHOLD, 0, 1),
            limit: SEARCH_LIMIT,
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
          setLoading(false);
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
      if (!queryText.trim()) {
        setError("Enter a document before running detection.");
        return false;
      }
      setLoading(true);
      setError(null);
      try {
        const next = await runMinHashSearch({
          query: queryText,
          threshold: clamp(threshold, 0, 1),
          limit: SEARCH_LIMIT,
        });
        setResponse(next);
        setSelectedId(next.results.length > 0 ? next.results[0].doc_id : null);
        setEditing(false);
        return true;
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : String(err));
        return false;
      } finally {
        setLoading(false);
      }
    },
    [threshold],
  );

  // Picking an example runs immediately; hand-edited text stays manual.
  const applyPreset = useCallback(
    (id: string) => {
      setEditing(false);
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

  const detect = useCallback(() => {
    return runSearch(content);
  }, [runSearch, content]);

  const pendingEdits = editing && content !== originalContent;
  const currentResponse =
    !pendingEdits && response?.query.trim() === content.trim()
      ? response
      : null;
  const selectedResult = useMemo(
    () =>
      currentResponse?.results.find((result) => result.doc_id === selectedId) ??
      null,
    [currentResponse, selectedId],
  );
  const selectedDiff = useMemo(
    () =>
      currentResponse?.diffs.find((diff) => diff.document_id === selectedId) ??
      null,
    [currentResponse, selectedId],
  );

  return (
    <div className="demo-page">
      <div className="screen">
        <header className="topbar">
          <div className="brand">
            <img className="wordmark" src={milvusWordmark} alt="Milvus" />
            <span className="version-badge">3.0</span>
          </div>
          <h1>Near-duplicate detection with MinHash</h1>
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

        <main>
          <section className="video-hero" aria-label="Video walkthrough">
            <div className="hero-copy">
              <h2>How MinHash finds duplicates</h2>
              <p className="hero-note">
                The video on the right explains the principle behind these
                results: create index → ingest → search.
              </p>
            </div>
            <Walkthrough />
          </section>

          <div className="workbench">
            <div className="workbench-side">
              <form
                className="query-panel"
                aria-label="Document controls"
                onSubmit={(event) => {
                  event.preventDefault();
                  void detect();
                }}
              >
                <label className="preset-control">
                  <span>Example case</span>
                  <select
                    value={presetId}
                    disabled={loading || presets.length === 0}
                    onChange={(event) => applyPreset(event.target.value)}
                  >
                    {presets.map((preset) => (
                      <option key={preset.id} value={preset.id}>
                        {preset.label}
                      </option>
                    ))}
                  </select>
                </label>
                <div className="query-controls">
                  <div className="threshold-control">
                    <label htmlFor="threshold-slider">
                      Similarity threshold
                    </label>
                    <span aria-hidden="true">{scoreLabel(threshold)}</span>
                    <input
                      id="threshold-slider"
                      aria-valuetext={scoreLabel(threshold)}
                      type="range"
                      min={0}
                      max={1}
                      step={0.01}
                      value={threshold}
                      style={
                        {
                          "--threshold-position": `${threshold * 100}%`,
                        } as CSSProperties
                      }
                      onChange={(event) =>
                        setThreshold(Number(event.target.value))
                      }
                    />
                  </div>
                </div>
                {error && (
                  <p className="error-banner" role="alert">
                    {error}
                  </p>
                )}
              </form>

              <CandidatePanel
                results={currentResponse?.results ?? []}
                selectedId={selectedId}
                threshold={threshold}
                onSelect={setSelectedId}
                pending={pendingEdits}
                loading={loading}
              />
            </div>

            <DiffView
              diff={selectedDiff}
              result={selectedResult}
              query={content}
              pending={pendingEdits}
              editor={{
                active: editing,
                loading,
                value: content,
                onStart: () => {
                  setOriginalContent(content);
                  setEditing(true);
                  setError(null);
                },
                onChange: setContent,
                onCancel: () => {
                  setContent(originalContent);
                  setEditing(false);
                  setError(null);
                },
                onApply: detect,
              }}
            />
          </div>
        </main>
      </div>

      <section className="steps" aria-label="Milvus calls behind this demo">
        <h2>What runs behind the page</h2>
        <ol className="step-list">
          {STEPS.map((step) => (
            <li className="step" key={step.title}>
              <header className="step-head">
                <span className="step-number">{step.n}</span>
                <span className="step-title">{step.title}</span>
              </header>
              <pre className="step-code">
                <code>{highlightPython(step.code)}</code>
              </pre>
            </li>
          ))}
        </ol>
      </section>
    </div>
  );
}
