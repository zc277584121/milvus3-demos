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
// Under-the-hood pipeline
//
// A single SVG strip that walks the MinHash + LSH data path from the incoming
// text to the ranked candidates. The "real" anchors are: the word-level
// shingle size (shingle_size=1), the schema/index parameters (dim 8192,
// MHJACCARD, MINHASH_LSH), and — most importantly — the estimated Jaccard
// value Milvus actually returned (MHJACCARD's distance field). The signature
// slots and band hits are illustrative: Milvus computes them server-side and
// the demo never fabricates a signature that purports to match it.
// ---------------------------------------------------------------------------

type PipelineStep = "shingle" | "minhash" | "band" | "score";

interface PipelineNode {
  id: PipelineStep;
  index: number;
  title: string;
  caption: string;
  real: boolean;
}

const PIPELINE_NODES: PipelineNode[] = [
  {
    id: "shingle",
    index: 0,
    title: "Shingle",
    caption: "word-level · size 1",
    real: true,
  },
  {
    id: "minhash",
    index: 1,
    title: "MinHash",
    caption: "256 hashes · 32-bit",
    real: true,
  },
  {
    id: "band",
    index: 2,
    title: "LSH bands",
    caption: "64 bands × 4 hashes",
    real: true,
  },
  {
    id: "score",
    index: 3,
    title: "Jaccard",
    caption: "MHJACCARD · refined",
    real: true,
  },
];

// A deterministic, tiny hash (FNV-1a over UTF-16 code units) used ONLY to
// drive the illustrative slot pattern. It shares the spirit of Milvus's
// hashing — one hash function seeded differently per slot — but is not
// Milvus's xxhash and never claims to reproduce the real signature.
function fnv1a(seed: number, text: string): number {
  let hash = 0x811c9dc5 ^ seed;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

// Which illustrative signature slots are "set" for a given text. Returns the
// set of slot indices (0..SLOT_COUNT-1) whose seeded hash lands in the lowest
// band, so two near-identical texts share most — but not all — slots.
function illustrativeSlots(
  text: string,
  slotCount: number,
  seedBase = 0,
): Set<number> {
  const slots = new Set<number>();
  const words = text.toLowerCase().split(/\s+/).filter(Boolean);
  for (let slot = 0; slot < slotCount; slot += 1) {
    // MinHash keeps the smallest hash over all shingles; we approximate the
    // "smallest" by comparing a seeded hash per word and keeping the min.
    let min = 0xffffffff;
    for (const word of words) {
      const value = fnv1a(seedBase + slot * 31 + 7, word);
      if (value < min) {
        min = value;
      }
    }
    if (min % 3 !== 0) {
      slots.add(slot);
    }
  }
  return slots;
}

const SLOT_COUNT = 32;
const BAND_COUNT = 8;

function wordsOf(text: string): string[] {
  return text.split(/\s+/).filter(Boolean);
}

const PIPELINE_CODE: Record<PipelineStep, string> = {
  shingle: `# Word-level shingles: each token is its own shingle.
# Milvus tokenizes with the default 'standard' analyzer, so a
# light two-word edit keeps most shingles intact.
from pymilvus import Function, FunctionType

function = Function(
    name="minhash_signature",
    function_type=FunctionType.MINHASH,
    input_field_names=["content"],
    output_field_names=["minhash_signature"],
    params={
        "num_hashes": 256,       # signature dim = 32 * 256
        "shingle_size": 1,       # one word per shingle
        "token_level": "word",
        "hash_function": "xxhash",
    },
)`,
  minhash: `# Each shingle is hashed by 256 seeded functions; the smallest
# value per function becomes one 32-bit slot of the signature.
# 256 slots × 32 bits = 8192 bits = the BINARY_VECTOR dimension.
signature_field: minhash_signature
    dtype: BINARY_VECTOR
    dim: 8192                   # 32 * num_hashes
`,
  band: `# The signature is split into 64 bands of 4 hashes each.
# Two documents are a candidate pair when any band matches
# exactly; identical text matches every band.
index = client.prepare_index_params()
index.add_index(
    field_name="minhash_signature",
    index_type="MINHASH_LSH",
    metric_type="MHJACCARD",
    params={
        "mh_element_bit_width": 32,
        "with_raw_data": True,   # keep raw signatures for refinement
    },
)`,
  score: `# Milvus ranks candidates by estimated Jaccard similarity.
# MHJACCARD returns the estimate DIRECTLY in the distance field
# (larger = more similar), so the app reports it verbatim.
hits = client.search(
    collection_name="milvus3_demos_minhash_near_duplicate",
    data=[query_text],
    anns_field="minhash_signature",
    search_params={
        "metric_type": "MHJACCARD",
        "params": {
            "mh_search_with_jaccard": True,
            "refine_k": 10,       # Jaccard refinement over raw signatures
        },
    },
    limit=10,
)
# hits[0][i]["distance"] -> estimated Jaccard similarity`,
};

function PipelineStage({
  node,
  active,
  onHover,
  hasResult,
}: {
  node: PipelineNode;
  active: boolean;
  onHover: (step: PipelineStep | null) => void;
  hasResult: boolean;
}) {
  return (
    <div
      className={`pipeline-stage${active ? " is-active" : ""}`}
      onMouseEnter={() => onHover(node.id)}
      onMouseLeave={() => onHover(null)}
      data-step={node.id}
    >
      <span className="pipeline-index">{node.index + 1}</span>
      <div className="pipeline-stage-body">
        <span className="pipeline-title">{node.title}</span>
        <span className="pipeline-caption">{node.caption}</span>
      </div>
      {node.real && <span className="pipeline-real">real</span>}
      {node.id === "score" && hasResult && (
        <span className="pipeline-live-dot" aria-hidden="true" />
      )}
    </div>
  );
}

function SignatureStrip({
  slots,
  active,
  label,
}: {
  slots: Set<number>;
  active: boolean;
  label: string;
}) {
  return (
    <div className={`signature-strip${active ? " is-active" : ""}`}>
      <span className="signature-label">{label}</span>
      <div className="signature-slots" aria-hidden="true">
        {Array.from({ length: SLOT_COUNT }, (_, index) => (
          <span
            key={index}
            className={`signature-slot${slots.has(index) ? " set" : ""}`}
          />
        ))}
      </div>
    </div>
  );
}

function BandStrip({
  querySlots,
  candidateSlots,
  active,
}: {
  querySlots: Set<number>;
  candidateSlots: Set<number>;
  active: boolean;
}) {
  return (
    <div className={`band-strip${active ? " is-active" : ""}`}>
      {Array.from({ length: BAND_COUNT }, (_, band) => {
        const start = band * 4;
        let match = true;
        for (let offset = 0; offset < 4; offset += 1) {
          const slot = start + offset;
          if (querySlots.has(slot) !== candidateSlots.has(slot)) {
            match = false;
            break;
          }
        }
        return (
          <div
            key={band}
            className={`band${match ? " hit" : ""}`}
            aria-hidden="true"
          >
            <span className="band-label">b{band + 1}</span>
            <div className="band-rows">
              {Array.from({ length: 4 }, (_, offset) => {
                const slot = start + offset;
                const q = querySlots.has(slot);
                const c = candidateSlots.has(slot);
                return (
                  <span
                    key={offset}
                    className={`band-cell${q && c ? " both" : q || c ? " one" : ""}`}
                  />
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function HowMinHashWorks({
  query,
  selectedResult,
}: {
  query: string;
  selectedResult: SearchResult | null;
}) {
  const [hovered, setHovered] = useState<PipelineStep | null>(null);

  const querySlots = useMemo(
    () => illustrativeSlots(query, SLOT_COUNT, 1),
    [query],
  );
  const candidateSlots = useMemo(
    () =>
      selectedResult
        ? illustrativeSlots(selectedResult.content, SLOT_COUNT, 1)
        : null,
    [selectedResult],
  );

  const hoveredNode =
    PIPELINE_NODES.find((node) => node.id === hovered) ?? null;

  return (
    <section
      className="under-hood"
      data-testid="under-the-hood"
      onMouseLeave={() => setHovered(null)}
    >
      <header className="under-hood-head">
        <h2>How MinHash finds near-duplicates</h2>
        <span className="under-hood-subtitle">
          Hover a step to see what it computes
        </span>
      </header>

      <div className="pipeline">
        {PIPELINE_NODES.map((node) => (
          <PipelineStage
            key={node.id}
            node={node}
            active={hovered === node.id}
            onHover={setHovered}
            hasResult={selectedResult !== null}
          />
        ))}
      </div>

      <div className="under-hood-detail" aria-live="polite">
        {hoveredNode ? (
          <>
            <div className="under-hood-copy">
              <SignatureStrip
                slots={querySlots}
                active={hovered === "minhash"}
                label="query signature"
              />
              {candidateSlots && (
                <SignatureStrip
                  slots={candidateSlots}
                  active={false}
                  label="candidate signature"
                />
              )}
              {hovered === "band" && candidateSlots && (
                <BandStrip
                  querySlots={querySlots}
                  candidateSlots={candidateSlots}
                  active
                />
              )}
              {hovered === "score" && selectedResult && (
                <div className="score-anchor">
                  <span className="score-anchor-label">
                    Milvus estimated Jaccard
                  </span>
                  <span className="score-anchor-value">
                    {scoreLabel(selectedResult.estimated_jaccard)}
                  </span>
                  <span className="score-anchor-note">
                    MHJACCARD distance, reported verbatim
                  </span>
                </div>
              )}
              {(hovered === "minhash" || hovered === "band") && (
                <p className="illustrative-note">
                  Signature slots and band hits are illustrative — Milvus
                  computes them server-side with xxhash.
                </p>
              )}
            </div>
            <pre className="under-hood-code">
              <code>{PIPELINE_CODE[hoveredNode.id]}</code>
            </pre>
          </>
        ) : (
          <div className="under-hood-idle" role="status">
            <p>
              {selectedResult
                ? "Hover any step above to inspect the path from text to similarity."
                : "Run detection first, then hover a step to inspect the path."}
            </p>
          </div>
        )}
      </div>
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

  useEffect(() => {
    let cancelled = false;
    fetchMinHashStatus()
      .then((next) => {
        if (!cancelled) {
          setStatus(next);
          if (next.query_presets.length > 0) {
            const first = next.query_presets[0];
            setPresetId(first.id);
            setTitle(first.title);
            setContent(first.text);
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

      <div className="threshold-control">
        <label htmlFor="threshold-slider">
          Detection threshold{" "}
          <span className="threshold-value">{threshold.toFixed(2)}</span>
        </label>
        <input
          id="threshold-slider"
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={threshold}
          onChange={(event) => setThreshold(Number(event.target.value))}
        />
        <p className="threshold-help">
          Threshold applies to estimated Jaccard similarity.
        </p>
      </div>

      <footer className="footer">
        <p>Review candidates before merging or removing content.</p>
        <p className="footer-meta">Illustrative scores · Concept data</p>
      </footer>
    </div>
  );
}
