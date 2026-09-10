import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
} from "react";

import { type DemoDefinition } from "@milvus3-demos/demo-ui";
import simpleheat from "simpleheat";

import "./styles.css";

export const QUERY_OPTIONS = [
  {
    id: "Q07",
    label: "Lunar mission trajectory",
    text: "Which diagram shows a spacecraft traveling from Earth orbit down to a lunar landing site along a drawn flight trajectory?",
  },
  {
    id: "Q08",
    label: "Cost-effectiveness curve",
    text: "Which chart plots a curve that rises as cost increases against effectiveness, with labeled axes?",
  },
  {
    id: "Q01",
    label: "Stakeholder expectations",
    text: "How should systems engineers identify stakeholders, elicit their needs, validate expectations against operational scenarios, and baseline the result before defining technical requirements?",
  },
  {
    id: "Q02",
    label: "Requirements flowdown",
    text: "How are system-level requirements flowed down into allocated and derived subsystem requirements while maintaining traceability to their parent requirements?",
  },
  {
    id: "Q03",
    label: "Verification and validation",
    text: "What evidence should engineers collect to show both that a built product conforms to its specified requirements and that typical users can succeed with it in a realistic operational setting?",
  },
  {
    id: "Q04",
    label: "Technical risk scenarios",
    text: "How should a team structure a technical risk so that initiating events, uncertain likelihood, and uncertain consequences can be analyzed together?",
  },
  {
    id: "Q05",
    label: "Change control",
    text: "After someone proposes a baseline change, what sequence of review, board decision, release, implementation, and status accounting should control it?",
  },
  {
    id: "Q06",
    label: "Risk-aware decision analysis",
    text: "When competing engineering alternatives rank closely, how can qualitative and quantitative analysis be combined to decide whether further uncertainty reduction is worthwhile?",
  },
] as const;

export const DEFAULT_QUERY = QUERY_OPTIONS[0].text;

export const embeddingListDemo: DemoDefinition = {
  id: "embedding-list-max-sim",
  title: "Visual PDF Page Retrieval",
  shortTitle: "EmbeddingList MAX_SIM",
  capability: "EmbeddingList + MAX_SIM",
  description:
    "Ask a natural engineering question, rank 40 fixed NASA handbook pages with Milvus MAX_SIM, and inspect a real local ColSmol token-patch explanation.",
  route: "/demos/embedding-list-max-sim",
  status: "implemented",
  accent: "#0891b2",
  highlights: [
    "Real-time CPU FP32 query-token embeddings",
    "Milvus EmbeddingList with MAX_SIM_COSINE",
    "Local ColSmol token-patch explanation on unmodified NASA pages",
  ],
};

interface QueryPreset {
  query_id: string;
  text: string;
}

interface AuthorRecord {
  name: string;
  affiliation: string;
}

interface DatasetSummary {
  dataset_id: string;
  title: string;
  revision: string;
  document_identifier: string;
  ntrs_id: number;
  ntrs_record_url: string;
  official_pdf_url: string;
  distribution: "PUBLIC";
  rights_determination: "PUBLIC_USE_PERMITTED";
  contains_third_party_material: false;
  attribution: string;
  authors: AuthorRecord[];
  render_statement: string;
  endorsement_statement: string;
  renderer: string;
  renderer_version: string;
  render_dpi: number;
  pdf_page_index_base: 1;
  source_pdf_page_count: number;
  source_pdf_bytes: number;
  page_count: number;
  manifest_sha256: string;
  pdf_sha256: string;
}

interface TimingBreakdown {
  total_ms: number;
  [key: string]: number;
}

export interface EmbeddingListStatus {
  status: string;
  implementation_status: "implemented";
  runtime_ready: boolean;
  collection_name: string;
  metric_type: string;
  dataset: DatasetSummary;
  query_presets: QueryPreset[];
  prepare_timings: {
    cold: TimingBreakdown | null;
    warm: TimingBreakdown | null;
  };
  model: {
    model_id: string;
    inference_dtype: "float32";
    device_type: "cpu";
    cpu_only: true;
  };
  milvus: {
    server_version: string;
    target_exists: boolean;
  };
}

interface SearchResult {
  page_id: string;
  pdf_page_index: number;
  printed_page: number;
  title: string;
  section: string;
  document_identifier: string;
  ntrs_id: number;
  ntrs_record_url: string;
  distribution: "PUBLIC";
  rights_determination: "PUBLIC_USE_PERMITTED";
  contains_third_party_material: false;
  score: number;
  local_score: number;
  score_delta: number;
  milvus_rank: number;
  local_rank: number;
  image_width: number;
  image_height: number;
  evidence_label: string;
}

interface ExplanationConcept {
  label: string;
  query_token_indices: number[];
  peak_similarity: number;
  peak_patch_index: number;
}

interface QueryToken {
  index: number;
  text: string;
  char_start: number;
  char_end: number;
  is_stopword: boolean;
  max_similarity: number;
  peak_patch_index: number;
  intensities: number[];
}

interface HeatmapPatch {
  patch_index: number;
  model_sequence_index: number;
  grid_column: number;
  grid_row: number;
  x: number;
  y: number;
  width: number;
  height: number;
  raw_similarity: number;
  intensity: number;
}

interface LocalExplanation {
  page_id: string;
  source: "local_colsmol_query_page_multi_vector";
  rank_source: "milvus_page_level_max_sim_cosine";
  query_vector_count: number;
  page_vector_count: number;
  explained_query_token_count: number;
  ignored_special_token_count: number;
  grid: {
    columns: number;
    rows: number;
    patch_count: number;
    coordinate_space: "normalized_unmodified_page";
  };
  aggregation: {
    similarity: string;
    concept: string;
    patch: string;
    normalization: string;
    spatial_mapping: string;
  };
  token_score_bounds: {
    min: number;
    max: number;
  };
  concepts: ExplanationConcept[];
  tokens: QueryToken[];
  patches: HeatmapPatch[];
}

interface SearchResponse {
  status: "passed";
  execution_path: string;
  query: string;
  metric_type: "MAX_SIM_COSINE";
  dataset: DatasetSummary;
  query_embedding: {
    vector_counts: number[];
    vector_dimension: number;
    inference_device: string;
    inference_dtype: string;
  };
  page_vector_counts: number[];
  ranking_equal: boolean;
  max_score_delta: number;
  score_absolute_tolerance: number;
  results: SearchResult[];
  local_explanations: LocalExplanation[];
  timings: TimingBreakdown & {
    query_inference_ms: number;
    milvus_search_ms: number;
    local_score_ms: number;
    local_explanation_ms: number;
  };
  latency_ms: number;
}

const API_BASE = "/api/embedding-list/v1";

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

export async function fetchEmbeddingListStatus(): Promise<EmbeddingListStatus> {
  return readJson<EmbeddingListStatus>(
    await fetch(`${API_BASE}/status`, {
      headers: { Accept: "application/json" },
    }),
  );
}

export async function prepareEmbeddingList(): Promise<void> {
  await readJson(
    await fetch(`${API_BASE}/prepare`, {
      method: "POST",
      headers: { Accept: "application/json" },
    }),
  );
}

export async function runEmbeddingListSearch(
  query: string,
): Promise<SearchResponse> {
  return readJson<SearchResponse>(
    await fetch(`${API_BASE}/search`, {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query }),
    }),
  );
}

export function pageEvidenceUrl(pageId: string): string {
  return `${API_BASE}/pages/${encodeURIComponent(pageId)}`;
}

function RankCard({
  result,
  selected,
  onSelect,
}: {
  result: SearchResult;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      className={`embedding-rank-card${selected ? " is-selected" : ""}`}
      data-testid="embedding-result"
      data-page-id={result.page_id}
      type="button"
      onClick={onSelect}
      aria-pressed={selected}
    >
      <span className="embedding-rank-number">#{result.milvus_rank}</span>
      <img
        src={pageEvidenceUrl(result.page_id)}
        alt={`Printed page ${result.printed_page}: ${result.title}`}
      />
      <span className="embedding-rank-copy">
        <strong>{result.title}</strong>
        <small>
          printed {result.printed_page} · PDF index {result.pdf_page_index}
        </small>
      </span>
      <span className="embedding-rank-score">
        <small>MAX_SIM</small>
        <strong>{result.score.toFixed(3)}</strong>
      </span>
    </button>
  );
}

function PageExplanation({
  page,
  explanation,
  hoveredToken,
}: {
  page: SearchResult;
  explanation: LocalExplanation;
  hoveredToken: QueryToken | null;
}) {
  const heatmapRef = useRef<HTMLCanvasElement | null>(null);
  const visiblePatches = useMemo(
    () => explanation.patches.filter((patch) => patch.intensity >= 0.04),
    [explanation],
  );
  // The heatmap data source: when a token is hovered we render that single
  // token's full-page intensity map; otherwise the aggregate concept map.
  const heatPoints = useMemo(() => {
    if (hoveredToken) {
      return hoveredToken.intensities
        .map((intensity, index) => {
          const patch = explanation.patches[index];
          return patch && intensity >= 0.04
            ? {
                x: (patch.x + patch.width / 2) * page.image_width,
                y: (patch.y + patch.height / 2) * page.image_height,
                intensity,
              }
            : null;
        })
        .filter((point): point is NonNullable<typeof point> => point !== null);
    }
    return visiblePatches.map((patch) => ({
      x: (patch.x + patch.width / 2) * page.image_width,
      y: (patch.y + patch.height / 2) * page.image_height,
      intensity: patch.intensity,
    }));
  }, [
    hoveredToken,
    explanation.patches,
    visiblePatches,
    page.image_width,
    page.image_height,
  ]);
  const pageAspectRatio = page.image_width / page.image_height;
  const canvasStyle = {
    "--page-aspect-ratio": pageAspectRatio,
  } as CSSProperties;
  const imageAlt = `NASA Systems Engineering Handbook PDF index ${page.pdf_page_index}, printed page ${page.printed_page}: ${page.title}`;

  useEffect(() => {
    const canvas = heatmapRef.current;
    if (!canvas || typeof canvas.getContext !== "function") return undefined;

    // jsdom (unit tests) does not implement the 2D canvas backend; draw only
    // when a real browser context is available.
    let heat: ReturnType<typeof simpleheat> | null = null;
    try {
      const context = canvas.getContext("2d");
      if (!context) return undefined;
      heat = simpleheat(canvas);
    } catch {
      return undefined;
    }

    const columns = Math.max(1, explanation.grid.columns);
    const rows = Math.max(1, explanation.grid.rows);
    const cellWidth = page.image_width / columns;
    const cellHeight = page.image_height / rows;
    const radius = Math.max(cellWidth, cellHeight) * 0.6;

    heat
      .clear()
      .max(1)
      .radius(radius, radius * 0.9)
      .gradient({
        0.1: "blue",
        0.3: "cyan",
        0.5: "lime",
        0.7: "yellow",
        0.9: "red",
      })
      .data(
        heatPoints.map((point) => [point.x, point.y, point.intensity ** 0.7]),
      )
      .draw(0.04);
  }, [
    heatPoints,
    explanation.grid.columns,
    explanation.grid.rows,
    page.image_width,
    page.image_height,
  ]);

  return (
    <div className="embedding-page-explanation">
      <div className="embedding-page-stage">
        <div className="embedding-page-visual">
          <div
            className="embedding-page-canvas"
            data-image-width={page.image_width}
            data-image-height={page.image_height}
            style={canvasStyle}
          >
            <img
              src={pageEvidenceUrl(page.page_id)}
              width={page.image_width}
              height={page.image_height}
              alt={imageAlt}
            />
            <canvas
              ref={heatmapRef}
              className="embedding-heatmap"
              data-testid="local-colsmol-heatmap"
              data-page-id={page.page_id}
              data-patch-count={explanation.grid.patch_count}
              width={page.image_width}
              height={page.image_height}
              aria-label={`Local ColSmol explanation heatmap for printed page ${page.printed_page}`}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function QueryComboBox({
  query,
  onChange,
  onSelectPreset,
}: {
  query: string;
  onChange: (value: string) => void;
  onSelectPreset: (value: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return undefined;
    function onPointerDown(event: MouseEvent) {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  return (
    <div className="embedding-combobox" ref={rootRef}>
      <div className="embedding-combobox-control">
        <input
          id="embedding-query"
          type="text"
          value={query}
          onChange={(event) => onChange(event.target.value)}
          aria-label="Query"
          placeholder="Ask a question about the handbook…"
        />
        <button
          type="button"
          className="embedding-combobox-toggle"
          aria-label="Choose a question"
          aria-haspopup="listbox"
          aria-expanded={open}
          onClick={() => setOpen((value) => !value)}
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 16 16"
            aria-hidden="true"
            focusable="false"
          >
            <path
              d="M4 6l4 4 4-4"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      </div>
      {open && (
        <ul
          className="embedding-combobox-menu"
          role="listbox"
          aria-label="Question presets"
        >
          {QUERY_OPTIONS.map((option) => (
            <li key={option.id} role="presentation">
              <button
                type="button"
                role="option"
                aria-selected={option.text === query}
                onClick={() => {
                  onSelectPreset(option.text);
                  setOpen(false);
                }}
              >
                <span className="embedding-combobox-option-label">
                  {option.text}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function EmbeddingListDemoPage() {
  const [query, setQuery] = useState<string>(DEFAULT_QUERY);
  const [backendStatus, setBackendStatus] =
    useState<EmbeddingListStatus | null>(null);
  const [backendUnavailable, setBackendUnavailable] = useState(false);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [selectedPageId, setSelectedPageId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [preparing, setPreparing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hoveredTokenIndex, setHoveredTokenIndex] = useState<number | null>(
    null,
  );
  const requestGeneration = useRef(0);
  const autoRanRef = useRef(false);

  useEffect(() => {
    let active = true;
    fetchEmbeddingListStatus()
      .then((value) => {
        if (active) {
          setBackendStatus(value);
          setBackendUnavailable(false);
        }
      })
      .catch(() => {
        if (active) {
          setBackendUnavailable(true);
        }
      });
    return () => {
      active = false;
    };
  }, []);

  function changeQuery(value: string) {
    requestGeneration.current += 1;
    setQuery(value);
    setResult(null);
    setSelectedPageId(null);
    setError(null);
    setLoading(false);
  }

  function selectPreset(value: string) {
    setQuery(value);
    setResult(null);
    setSelectedPageId(null);
    setError(null);
    void runSearch(value);
  }

  async function prepareDemo() {
    setPreparing(true);
    setError(null);
    try {
      await prepareEmbeddingList();
      setBackendStatus(await fetchEmbeddingListStatus());
      setBackendUnavailable(false);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Preparation failed.");
    } finally {
      setPreparing(false);
    }
  }

  async function runSearch(override?: string) {
    const submittedQuery = (override ?? query).trim();
    if (!submittedQuery) {
      setError("Enter a query before searching.");
      return;
    }
    const generation = requestGeneration.current + 1;
    requestGeneration.current = generation;
    setResult(null);
    setSelectedPageId(null);
    setLoading(true);
    setError(null);
    try {
      const response = await runEmbeddingListSearch(submittedQuery);
      if (requestGeneration.current === generation) {
        setResult(response);
        setSelectedPageId(response.results[0]?.page_id ?? null);
      }
    } catch (cause) {
      if (requestGeneration.current === generation) {
        setError(cause instanceof Error ? cause.message : "Search failed.");
      }
    } finally {
      if (requestGeneration.current === generation) {
        setLoading(false);
      }
    }
  }

  const ready = backendStatus?.runtime_ready === true;
  const topResults = result?.results.slice(0, 3) ?? [];
  const selectedPage =
    topResults.find((item) => item.page_id === selectedPageId) ?? topResults[0];
  const selectedExplanation = useMemo(
    () =>
      result?.local_explanations.find(
        (item) => item.page_id === selectedPage?.page_id,
      ),
    [result, selectedPage?.page_id],
  );
  // Tokens arrive in model order; keep that order (it matches the original
  // query's reading order) and expose the currently hovered token directly.
  const orderedTokens = useMemo(
    () => (selectedExplanation ? selectedExplanation.tokens : []),
    [selectedExplanation],
  );
  const hoveredToken = useMemo(
    () =>
      orderedTokens.find((token) => token.index === hoveredTokenIndex) ?? null,
    [orderedTokens, hoveredTokenIndex],
  );
  // Normalize token scores to 0..1 for color depth using the backend bounds.
  const tokenScoreMin = selectedExplanation?.token_score_bounds.min ?? 0;
  const tokenScoreMax = selectedExplanation?.token_score_bounds.max ?? 0;
  const tokenScoreRange = tokenScoreMax - tokenScoreMin;

  useEffect(() => {
    if (!ready || autoRanRef.current) {
      return;
    }
    autoRanRef.current = true;
    void runSearch(DEFAULT_QUERY);
  }, [ready]);

  return (
    <main className="embedding-demo-page">
      <header className="embedding-topbar">
        <a className="back-link" href="/">
          ← Portal
        </a>
        <h1>{embeddingListDemo.title}</h1>
      </header>

      <div className="embedding-workspace">
        <section
          className="embedding-query-panel"
          aria-label="PDF visual search"
        >
          <QueryComboBox
            query={query}
            onChange={changeQuery}
            onSelectPreset={selectPreset}
          />
          <div className="embedding-actions">
            <button
              className="embedding-search-button"
              type="button"
              onClick={() => runSearch()}
              disabled={!ready || loading || preparing}
            >
              {loading ? "Embedding on CPU…" : "Search 40 pages"}
            </button>
            {!backendUnavailable && backendStatus && !ready && (
              <button type="button" onClick={prepareDemo} disabled={preparing}>
                {preparing ? "Preparing…" : "Prepare index"}
              </button>
            )}
          </div>
          {error && (
            <p className="embedding-error" role="alert">
              {error}
            </p>
          )}
          {result && orderedTokens.length > 0 && (
            <div
              className="embedding-token-row"
              aria-label="Query token contribution"
            >
              <div className="embedding-token-strip">
                {orderedTokens.map((token) => {
                  const active = token.index === hoveredTokenIndex;
                  const depth =
                    token.is_stopword || tokenScoreRange <= 0
                      ? 0
                      : (token.max_similarity - tokenScoreMin) /
                        tokenScoreRange;
                  // Leading whitespace marks a word boundary: render it as an
                  // un-colored gap between cells so the space is visible again.
                  const leading = token.text.match(/^\s+/)?.[0] ?? "";
                  const word = token.text.slice(leading.length);
                  return (
                    <span
                      key={`${token.index}-${token.char_start}`}
                      className="embedding-token-piece"
                    >
                      <span className="embedding-token-line">
                        {leading.length > 0 && (
                          <span
                            className="embedding-token-space"
                            aria-hidden="true"
                          >
                            {leading}
                          </span>
                        )}
                        <button
                          type="button"
                          className={`embedding-token-cell${
                            active ? " is-active" : ""
                          }${token.is_stopword ? " is-stopword" : ""}`}
                          style={
                            {
                              "--token-depth": depth,
                            } as CSSProperties
                          }
                          onMouseEnter={() => setHoveredTokenIndex(token.index)}
                          onMouseLeave={() =>
                            setHoveredTokenIndex((current) =>
                              current === token.index ? null : current,
                            )
                          }
                          onFocus={() => setHoveredTokenIndex(token.index)}
                          onBlur={() =>
                            setHoveredTokenIndex((current) =>
                              current === token.index ? null : current,
                            )
                          }
                          aria-label={`${token.text} ${token.max_similarity.toFixed(3)}`}
                        >
                          <span className="embedding-token-text">
                            {word || "␣"}
                          </span>
                        </button>
                      </span>
                      <span
                        className={`embedding-token-score${
                          token.is_stopword ? " is-stopword" : ""
                        }`}
                        aria-hidden="true"
                      >
                        {token.max_similarity.toFixed(3)}
                      </span>
                    </span>
                  );
                })}
              </div>
            </div>
          )}
        </section>

        <section className="embedding-results" aria-live="polite">
          {!result && !loading && (
            <div className="embedding-empty-state">
              <span>EmbeddingList + MAX_SIM_COSINE</span>
              <h2>Pick a question to see the matching pages.</h2>
            </div>
          )}
          {loading && (
            <div className="embedding-empty-state" role="status">
              <h2>Searching…</h2>
            </div>
          )}
          {result && selectedPage && selectedExplanation && (
            <>
              <div className="embedding-result-grid">
                <aside
                  className="embedding-top-three"
                  aria-label="Milvus Top 3 pages"
                >
                  {topResults.map((item) => (
                    <RankCard
                      result={item}
                      selected={item.page_id === selectedPage.page_id}
                      onSelect={() => setSelectedPageId(item.page_id)}
                      key={item.page_id}
                    />
                  ))}
                </aside>
                <PageExplanation
                  page={selectedPage}
                  explanation={selectedExplanation}
                  hoveredToken={hoveredToken}
                />
              </div>
              <footer className="embedding-results-footer">
                <p className="eyebrow">Milvus Top 3 of 40</p>
              </footer>
            </>
          )}
        </section>
      </div>
    </main>
  );
}
