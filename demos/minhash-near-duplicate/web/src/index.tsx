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

const NUM_HASHES = 256;
const LSH_BAND_COUNT = 64;
const HASHES_PER_BAND = NUM_HASHES / LSH_BAND_COUNT;
const TRIAL_INTERVAL_MS = 24;

// Real Milvus calls behind each step, shown on hover.
type MilvusStep = "function" | "index" | "search";

const MILVUS_CODE: Record<MilvusStep, string> = {
  function: `schema.add_field("content", DataType.VARCHAR, max_length=2000,
                 enable_analyzer=True)
schema.add_field("minhash_signature", DataType.BINARY_VECTOR, dim=8192)

schema.add_function(Function(
    function_type=FunctionType.MINHASH,
    input_field_names=["content"],
    output_field_names=["minhash_signature"],
    params={
        "num_hashes": 256,        # 32 * 256 = 8192
        "shingle_size": 1,
        "hash_function": "xxhash",
        "token_level": "word",
    },
))`,
  index: `indexes.add_index(
    field_name="minhash_signature",
    index_type="MINHASH_LSH",
    metric_type="MHJACCARD",
    params={
        "mh_element_bit_width": 32,
        "mh_lsh_band": 64,        # 256 slots / 64 = 4 per band
        "with_raw_data": True,    # keep signatures for refine
    },
)`,
  search: `hits = client.search(
    collection_name="milvus3_demos_minhash_near_duplicate",
    data=[query_text],            # raw text, not a vector
    anns_field="minhash_signature",
    search_params={
        "metric_type": "MHJACCARD",
        "params": {
            "mh_search_with_jaccard": True,
            "refine_k": 10,
        },
    },
    limit=10,
)
# hits[0][i]["distance"] IS the estimated Jaccard`,
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

// FNV-1a over UTF-16 code units, seeded per signature slot. This is the
// explainer's stand-in for Milvus's xxhash: one hash function per slot, so
// every slot induces a different "random ordering" of the same words.
function fnv1a(seed: number, word: string): number {
  let hash = (0x811c9dc5 ^ seed) >>> 0;
  for (let i = 0; i < word.length; i += 1) {
    hash ^= word.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

interface ScoredWord {
  word: string;
  hash: number;
}

interface MinHashTrial {
  // 0-based slot index this trial fills in the signature.
  slotIndex: number;
  // The vocabulary ordered by this slot's hash — the "random ordering".
  ordering: ScoredWord[];
  // First word of that ordering belonging to each set = the set's minimum hash.
  queryPick: ScoredWord | null;
  candidatePick: ScoredWord | null;
  collision: boolean;
}

// One MinHash computation: hash every word, sort ascending, and take the first
// word each document contains. That word's hash is the slot value, which is
// literally the "minimum hash over the set" definition of MinHash.
export function runTrial(
  querySet: Set<string>,
  candidateSet: Set<string>,
  vocabulary: readonly string[],
  slotIndex: number,
): MinHashTrial {
  const seed = (slotIndex + 1) * 0x9e3779b9;
  const ordering = vocabulary
    .map((word) => ({ word, hash: fnv1a(seed, word) }))
    .sort((a, b) => a.hash - b.hash);

  let queryPick: ScoredWord | null = null;
  let candidatePick: ScoredWord | null = null;
  for (const entry of ordering) {
    if (queryPick === null && querySet.has(entry.word)) {
      queryPick = entry;
    }
    if (candidatePick === null && candidateSet.has(entry.word)) {
      candidatePick = entry;
    }
    if (queryPick !== null && candidatePick !== null) {
      break;
    }
  }

  return {
    slotIndex,
    ordering,
    queryPick,
    candidatePick,
    collision:
      queryPick !== null &&
      candidatePick !== null &&
      queryPick.hash === candidatePick.hash,
  };
}

function formatSlotValue(value: number): string {
  return `0x${value.toString(16).padStart(8, "0")}`;
}

interface BandState {
  index: number;
  // How many of the band's slots have been computed so far.
  filled: number;
  // Whether every slot in the band collided (only meaningful when complete).
  hit: boolean;
  complete: boolean;
}

// Group the flat slot results into LSH bands, mirroring Milvus's
// `mh_lsh_band` / signature-dimension split. A band "hits" only when every one
// of its slots is a collision; one hit is enough to become a candidate.
export function buildBands(
  slotCollisions: readonly boolean[],
  bandCount: number,
  hashesPerBand: number,
): BandState[] {
  return Array.from({ length: bandCount }, (_, index) => {
    const start = index * hashesPerBand;
    const slice = slotCollisions.slice(start, start + hashesPerBand);
    const filled = slice.length;
    const complete = filled === hashesPerBand;
    return {
      index,
      filled,
      complete,
      hit: complete && slice.every(Boolean),
    };
  });
}

function HowMinHashWorks({
  query,
  selectedResult,
  latencyMs,
}: {
  query: string;
  selectedResult: SearchResult | null;
  latencyMs: number | null;
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

  // `slotCollisions[i]` records whether slot i collided. It is both the
  // signature (one bit per hash) and the raw material for the band view.
  const [slotCollisions, setSlotCollisions] = useState<boolean[]>([]);
  const [current, setCurrent] = useState<MinHashTrial | null>(null);
  const [running, setRunning] = useState(false);
  const [hoveredStep, setHoveredStep] = useState<MilvusStep | null>(null);

  const trials = slotCollisions.length;
  const collisions = useMemo(
    () => slotCollisions.filter(Boolean).length,
    [slotCollisions],
  );

  // Reset the simulation whenever the compared pair changes.
  useEffect(() => {
    setSlotCollisions([]);
    setCurrent(null);
    setRunning(false);
  }, [query, selectedResult?.doc_id]);

  const advance = useCallback(() => {
    setSlotCollisions((prev) => {
      if (prev.length >= NUM_HASHES) {
        return prev;
      }
      const next = runTrial(querySet, candidateSet, vocabulary, prev.length);
      setCurrent(next);
      return [...prev, next.collision];
    });
  }, [querySet, candidateSet, vocabulary]);

  // Continuous run: one slot per tick until the signature is full.
  useEffect(() => {
    if (!running) {
      return;
    }
    const id = window.setInterval(() => {
      setSlotCollisions((prev) => {
        if (prev.length >= NUM_HASHES) {
          setRunning(false);
          return prev;
        }
        const next = runTrial(querySet, candidateSet, vocabulary, prev.length);
        setCurrent(next);
        if (prev.length + 1 >= NUM_HASHES) {
          setRunning(false);
        }
        return [...prev, next.collision];
      });
    }, TRIAL_INTERVAL_MS);
    return () => window.clearInterval(id);
  }, [running, querySet, candidateSet, vocabulary]);

  const reset = useCallback(() => {
    setSlotCollisions([]);
    setCurrent(null);
    setRunning(false);
  }, []);

  const bands = useMemo(
    () => buildBands(slotCollisions, LSH_BAND_COUNT, HASHES_PER_BAND),
    [slotCollisions],
  );
  const decidedBands = bands.filter((band) => band.complete).length;
  const hitBands = bands.filter((band) => band.hit).length;
  const isCandidate = hitBands > 0;

  const filled = slotCollisions.length;

  return (
    <section className="under-hood" data-testid="under-the-hood">
      <header className="under-hood-head">
        <h2>Inside Milvus: near-duplicate search</h2>
        <span className="under-hood-subtitle">
          {selectedResult
            ? "Hover any step for details"
            : "Run detection to compare two word sets"}
        </span>
      </header>

      {selectedResult ? (
        <div className="explainer-split">
          <div className="split-col app-col">
            <span className="split-badge app">Your app</span>

            <div className="word-set">
              <span className="word-set-label">
                <span className="step-num">1</span>
                Incoming · {queryShingles.length} words
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

            <div className="payload">
              <span className="step-num">2</span>
              <code>POST /api/v1/search</code>
              <span className="payload-note">text only</span>
              <span className="payload-arrow" aria-hidden="true">
                →
              </span>
            </div>

            <div className="app-tail">
              <div className="response-block">
                <span className="step-num">6</span>
                <span className="response-label">← response</span>
                <span className="response-score">
                  {scoreLabel(selectedResult.estimated_jaccard)}
                </span>
                {latencyMs !== null && (
                  <span className="response-latency">
                    {Math.round(latencyMs)} ms
                  </span>
                )}
              </div>

              <div className="score-pair">
                <span className="score-pair-item">
                  true Jaccard <b>{trueJaccard.toFixed(2)}</b>
                </span>
              </div>
            </div>
          </div>

          <div className="split-col milvus-col">
            <span className="split-badge milvus">Milvus 3.0</span>

            <div
              className="mstep mstep--linked"
              onMouseEnter={() => setHoveredStep("function")}
              onMouseLeave={() => setHoveredStep(null)}
            >
              <div className="mstep-head">
                <span className="step-num">3</span>
                <span className="mstep-name">FunctionType.MINHASH</span>
                <span className="mstep-meta">
                  {filled} / {NUM_HASHES} slots
                </span>
              </div>

              <div className="word-set compact">
                <span className="word-set-label">
                  Stored · {selectedResult.doc_id}
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

              <div className="trial-line">
                {current ? (
                  <>
                    <span className="trial-slot">slot {current.slotIndex}</span>
                    <code className="trial-hash q">
                      {current.queryPick
                        ? formatSlotValue(current.queryPick.hash)
                        : "—"}
                    </code>
                    <span
                      className={`trial-eq${current.collision ? " hit" : " miss"}`}
                    >
                      {current.collision ? "=" : "≠"}
                    </span>
                    <code className="trial-hash c">
                      {current.candidatePick
                        ? formatSlotValue(current.candidatePick.hash)
                        : "—"}
                    </code>
                    <span className="trial-word">
                      {current.queryPick?.word ?? "—"}
                    </span>
                  </>
                ) : (
                  <span className="trial-idle">
                    one hash per slot · smallest wins
                  </span>
                )}
              </div>

              <div className="mstep-controls">
                <button
                  className="mini-button primary"
                  onClick={() => {
                    if (filled < NUM_HASHES) {
                      setRunning(true);
                    }
                  }}
                  disabled={running || filled >= NUM_HASHES}
                  type="button"
                >
                  {running ? "Running…" : `▶ Run ${NUM_HASHES} slots`}
                </button>
                <button
                  className="mini-button"
                  onClick={advance}
                  disabled={running || filled >= NUM_HASHES}
                  type="button"
                >
                  One slot
                </button>
                <button className="mini-button" onClick={reset} type="button">
                  Reset
                </button>
                <span className="mstep-counter">{collisions} matched</span>
              </div>
              <div className="fill-bar" aria-hidden="true">
                <span
                  className="fill-bar-inner"
                  style={{ width: `${(filled / NUM_HASHES) * 100}%` }}
                />
              </div>
            </div>

            <div
              className="mstep mstep--linked"
              onMouseEnter={() => setHoveredStep("index")}
              onMouseLeave={() => setHoveredStep(null)}
            >
              <div className="mstep-head">
                <span className="step-num">4</span>
                <span className="mstep-name">MINHASH_LSH index</span>
                <span className="mstep-meta">
                  {hitBands} / {LSH_BAND_COUNT} bands
                </span>
              </div>

              <div className="band-grid" data-testid="lsh-bands">
                {bands.map((band) => (
                  <span
                    key={band.index}
                    className={`band-col${band.complete ? " complete" : ""}${
                      band.hit ? " hit" : ""
                    }`}
                  >
                    {Array.from({ length: HASHES_PER_BAND }, (_, offset) => {
                      const value =
                        slotCollisions[band.index * HASHES_PER_BAND + offset];
                      const state =
                        value === undefined ? "" : value ? " match" : " miss";
                      return <i key={offset} className={`sig-cell${state}`} />;
                    })}
                  </span>
                ))}
              </div>

              <span
                className={`mstep-verdict${
                  decidedBands === 0 ? "" : isCandidate ? " hit" : " miss"
                }`}
              >
                {decidedBands === 0
                  ? "no band complete"
                  : isCandidate
                    ? "→ candidate"
                    : "→ skipped"}
              </span>
            </div>

            <div
              className="mstep mstep--linked"
              onMouseEnter={() => setHoveredStep("search")}
              onMouseLeave={() => setHoveredStep(null)}
            >
              <div className="mstep-head">
                <span className="step-num">5</span>
                <span className="mstep-name">client.search() · refine_k</span>
                <span className="mstep-meta">
                  {isCandidate ? "ranked" : "not reached"}
                </span>
              </div>

              <div className="metric-line">
                <span className="metric-key">distance</span>
                <span className="metric-eq">=</span>
                <b className="metric-value">
                  {scoreLabel(selectedResult.estimated_jaccard)}
                </b>
                <span className="metric-hint">MHJACCARD · larger = closer</span>
              </div>

              <div className="refine-line">
                <span className="refine-key">refine_k 10</span>
                <span className="refine-note">
                  re-rank from the stored raw signatures
                </span>
              </div>
            </div>
          </div>

          {hoveredStep && (
            <div className="code-overlay">
              <pre>
                <code>{highlightPython(MILVUS_CODE[hoveredStep])}</code>
              </pre>
            </div>
          )}
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

      <div className="scroll-main">
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

        <HowMinHashWorks
          query={content}
          selectedResult={selectedResult}
          latencyMs={response?.latency_ms ?? null}
        />

        <footer className="footer">
          <p>Review candidates before merging or removing content.</p>
          <p className="footer-meta">Illustrative scores · Concept data</p>
        </footer>
      </div>
    </div>
  );
}
