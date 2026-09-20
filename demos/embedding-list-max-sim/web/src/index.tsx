import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
} from "react";

import simpleheat from "simpleheat";

import "./styles.css";

export type DemoStatus =
  "available" | "implemented" | "foundation-ready" | "planned";

export interface DemoDefinition {
  id: string;
  title: string;
  shortTitle: string;
  capability: string;
  description: string;
  route: string;
  status: DemoStatus;
  accent: string;
  highlights: string[];
}

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
  title: "Visual PDF Retrieval with EmbeddingList",
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

/* ------------------------------------------------------------------ */
/* MAX_SIM explainer                                                  */
/* ------------------------------------------------------------------ */

// A self-playing SVG that walks a viewer through how MAX_SIM scores one page.
// Everything is driven by ONE 8-second CSS timeline: every keyframe is a slice
// of that single period, so no two loops can drift apart. The timeline only
// runs while the popover is hovered or focused, which restarts it from phase 0
// each time the reader asks to look.
function MaxSimExplainer() {
  // Four query tokens matched against four page patches. Each row's maximum is
  // the number that survives into the sum. The SVG is driven by ONE 12-second
  // CSS timeline in styles.css: every keyframe is a percentage slice of that
  // single period, played once (`forwards`) and held at the finished state.
  //
  // Timeline:  0–31%  decomposition (query→tokens, page→patches as a diagram)
  //           30–74%  row-by-row compare + lock maxima
  //           74–82%  maxima flow out
  //           82–92%  sum equation, then rank badge
  //           92–100% finished state holds
  const ROWS = [
    { token: "space", vals: [0.31, 0.82, 0.4, 0.22], maxCol: 1 },
    { token: "craft", vals: [0.76, 0.28, 0.19, 0.41], maxCol: 0 },
    { token: "lunar", vals: [0.35, 0.18, 0.71, 0.26], maxCol: 2 },
    { token: "landing", vals: [0.24, 0.69, 0.38, 0.2], maxCol: 1 },
  ];
  const MAXIMA = [0.82, 0.76, 0.71, 0.69];
  const TOTAL = MAXIMA.reduce((a, b) => a + b, 0).toFixed(2);
  const QUERY_TEXT = ROWS.map((r) => r.token).join(" ");
  const PATCHES = ["patch1", "patch2", "patch3", "patch4"];

  // Layout (viewBox 0 0 640 500). Decomposition band on top (query row + a
  // page diagram that is cut into patches), matrix below.
  const ROW_CY = [236, 292, 348, 404];
  const COL_CX = [210, 282, 354, 426];
  const MATRIX_X = 156;
  const MATRIX_W = 320;
  const MAX_CHIP_X = 494;

  // Page diagram geometry.
  const PAGE_X = 88;
  const PAGE_Y = 54;
  const PAGE_W = 140;
  const PAGE_H = 84;
  const PAGE_MID_X = PAGE_X + PAGE_W / 2; // 115
  const PAGE_MID_Y = PAGE_Y + PAGE_H / 2; // 96

  return (
    <div className="maxsim-trigger">
      <button
        type="button"
        className="maxsim-trigger-button"
        aria-haspopup="dialog"
        aria-label="Show how MAX_SIM works"
      >
        <svg
          className="maxsim-trigger-icon"
          viewBox="0 0 18 18"
          fill="none"
          aria-hidden="true"
          focusable="false"
        >
          <circle cx="4" cy="4" r="2.1" />
          <circle cx="14" cy="4" r="2.1" />
          <circle cx="4" cy="14" r="2.1" />
          <circle cx="14" cy="14" r="2.1" />
          <path d="M6 4h6M6 14h6M4 6v6M14 6v6" strokeWidth="1.3" />
        </svg>
        <span>How MAX_SIM works</span>
      </button>

      <div
        className="maxsim-popover"
        role="dialog"
        aria-label="MAX_SIM scoring animation"
      >
        <div className="maxsim-popover-head">
          <span className="maxsim-popover-kicker">
            How MAX_SIM scores one page
          </span>
          <span className="maxsim-popover-sub">
            For each query token, keep its best page-patch match — then add the
            maxima.
          </span>
        </div>

        <svg
          className="maxsim-anim"
          viewBox="0 0 640 500"
          fill="none"
          aria-hidden="true"
          focusable="false"
        >
          <defs>
            <linearGradient id="m-panel" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0" stopColor="#fcfefe" />
              <stop offset="1" stopColor="#f1f8f9" />
            </linearGradient>
            <linearGradient id="m-tok-grad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0" stopColor="#14b8d6" />
              <stop offset="1" stopColor="#0789a4" />
            </linearGradient>
            <linearGradient id="m-winner-grad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0" stopColor="#ffc65c" />
              <stop offset="1" stopColor="#f59e0b" />
            </linearGradient>
            <marker
              id="m-arrow"
              viewBox="0 0 10 10"
              refX="8"
              refY="5"
              markerWidth="7"
              markerHeight="7"
              orient="auto-start-reverse"
            >
              <path d="M0 0 L10 5 L0 10 z" fill="#f59e0b" />
            </marker>
          </defs>

          {/* ── Decomposition band: query → tokens, page → patches ── */}
          <g className="m-decomp">
            {/* query source: label + query string pill */}
            <text x="38" y="33" className="m-decomp-lhs-label">
              query
            </text>
            <rect
              x="88"
              y="16"
              width="140"
              height="26"
              rx="13"
              className="m-decomp-src-query"
            />
            <text
              x="158"
              y="33"
              textAnchor="middle"
              className="m-decomp-src-text"
            >
              {QUERY_TEXT}
            </text>
            <path
              d="M234 29 H 254 M 247 24 L 255 29 L 247 34"
              className="m-decomp-arrow query-arrow"
            />

            {/* 4 token chips */}
            {ROWS.map((row, i) => (
              <g
                key={`dtok-${row.token}`}
                className={`m-decomp-chip m-decomp-tok-${i + 1}`}
              >
                <rect
                  x={264 + i * 64}
                  y="16"
                  width="54"
                  height="26"
                  rx="7"
                  className="m-decomp-tok-box"
                />
                <text
                  x={291 + i * 64}
                  y="33"
                  textAnchor="middle"
                  className="m-decomp-tok-text"
                >
                  {row.token}
                </text>
              </g>
            ))}

            {/* page source: label + page diagram */}
            <text
              x="38"
              y={PAGE_MID_Y + 4}
              className="m-decomp-lhs-label m-decomp-lhs-page"
            >
              page
            </text>

            {/* page diagram: a PDF page cut into 4 patches */}
            <g className="m-page">
              {/* page sheet with content lines */}
              <rect
                x={PAGE_X}
                y={PAGE_Y}
                width={PAGE_W}
                height={PAGE_H}
                rx="3"
                className="m-page-sheet"
              />
              {/* content inside the page, one distinct motif per quadrant */}
              <g className="m-page-lines">
                {/* Q1 (top-left): image block */}
                <rect
                  x={PAGE_X + 12}
                  y={PAGE_Y + 12}
                  width={PAGE_MID_X - PAGE_X - 20}
                  height={PAGE_MID_Y - PAGE_Y - 20}
                  rx="2"
                  className="m-page-img"
                />
                {/* Q2 (top-right): two short text lines */}
                <line
                  x1={PAGE_MID_X + 8}
                  y1={PAGE_Y + 18}
                  x2={PAGE_X + PAGE_W - 12}
                  y2={PAGE_Y + 18}
                />
                <line
                  x1={PAGE_MID_X + 8}
                  y1={PAGE_Y + 30}
                  x2={PAGE_X + PAGE_W - 24}
                  y2={PAGE_Y + 30}
                />
                {/* Q3 (bottom-left): two text lines */}
                <line
                  x1={PAGE_X + 12}
                  y1={PAGE_MID_Y + 12}
                  x2={PAGE_MID_X - 8}
                  y2={PAGE_MID_Y + 12}
                />
                <line
                  x1={PAGE_X + 12}
                  y1={PAGE_MID_Y + 24}
                  x2={PAGE_MID_X - 20}
                  y2={PAGE_MID_Y + 24}
                />
                {/* Q4 (bottom-right): list dots + short line */}
                <circle
                  cx={PAGE_MID_X + 12}
                  cy={PAGE_MID_Y + 14}
                  r="2"
                  className="m-page-dot"
                />
                <circle
                  cx={PAGE_MID_X + 12}
                  cy={PAGE_MID_Y + 26}
                  r="2"
                  className="m-page-dot"
                />
                <line
                  x1={PAGE_MID_X + 20}
                  y1={PAGE_MID_Y + 14}
                  x2={PAGE_X + PAGE_W - 12}
                  y2={PAGE_MID_Y + 14}
                />
                <line
                  x1={PAGE_MID_X + 20}
                  y1={PAGE_MID_Y + 26}
                  x2={PAGE_X + PAGE_W - 18}
                  y2={PAGE_MID_Y + 26}
                />
              </g>
              {/* 2×2 cut lines */}
              <g className="m-page-cut">
                <line
                  x1={PAGE_MID_X}
                  y1={PAGE_Y}
                  x2={PAGE_MID_X}
                  y2={PAGE_Y + PAGE_H}
                />
                <line
                  x1={PAGE_X}
                  y1={PAGE_MID_Y}
                  x2={PAGE_X + PAGE_W}
                  y2={PAGE_MID_Y}
                />
              </g>
            </g>

            {/* arrow from page to first patch */}
            <path
              d={`M ${PAGE_X + PAGE_W + 6} ${PAGE_MID_Y} H 254 M 247 ${PAGE_MID_Y - 5} L 255 ${PAGE_MID_Y} L 247 ${PAGE_MID_Y + 5}`}
              className="m-decomp-arrow image-arrow"
            />

            {/* 4 patch fragments cut out of the page (each mirrors a quadrant) */}
            {PATCHES.map((patch, i) => {
              const stagger = [0, -8, 8, 0][i];
              return (
                <g
                  key={`dpatch-${patch}`}
                  className={`m-decomp-chip m-decomp-patch-${i + 1}`}
                  transform={`translate(0 ${stagger})`}
                >
                  <rect
                    x={264 + i * 64}
                    y="68"
                    width="54"
                    height="56"
                    rx="5"
                    className="m-decomp-patch-box"
                  />
                  {/* fragment content mirroring quadrant i */}
                  {i === 0 && (
                    <rect
                      x={270 + i * 64}
                      y="78"
                      width="42"
                      height="26"
                      rx="2"
                      className="m-decomp-patch-img"
                    />
                  )}
                  {i === 1 && (
                    <>
                      <line
                        x1={272 + i * 64}
                        y1="84"
                        x2={312 + i * 64}
                        y2="84"
                        className="m-decomp-patch-line"
                      />
                      <line
                        x1={272 + i * 64}
                        y1="94"
                        x2={304 + i * 64}
                        y2="94"
                        className="m-decomp-patch-line"
                      />
                      <line
                        x1={272 + i * 64}
                        y1="104"
                        x2={310 + i * 64}
                        y2="104"
                        className="m-decomp-patch-line"
                      />
                    </>
                  )}
                  {i === 2 && (
                    <>
                      <line
                        x1={272 + i * 64}
                        y1="88"
                        x2={312 + i * 64}
                        y2="88"
                        className="m-decomp-patch-line"
                      />
                      <line
                        x1={272 + i * 64}
                        y1="100"
                        x2={300 + i * 64}
                        y2="100"
                        className="m-decomp-patch-line"
                      />
                      <line
                        x1={272 + i * 64}
                        y1="112"
                        x2={310 + i * 64}
                        y2="112"
                        className="m-decomp-patch-line"
                      />
                    </>
                  )}
                  {i === 3 && (
                    <>
                      <circle
                        cx={276 + i * 64}
                        cy="88"
                        r="2"
                        className="m-decomp-patch-dot"
                      />
                      <line
                        x1={284 + i * 64}
                        y1="88"
                        x2={312 + i * 64}
                        y2="88"
                        className="m-decomp-patch-line"
                      />
                      <circle
                        cx={276 + i * 64}
                        cy="102"
                        r="2"
                        className="m-decomp-patch-dot"
                      />
                      <line
                        x1={284 + i * 64}
                        y1="102"
                        x2={304 + i * 64}
                        y2="102"
                        className="m-decomp-patch-line"
                      />
                    </>
                  )}
                  <text
                    x={270 + i * 64}
                    y="76"
                    className="m-decomp-patch-label"
                  >
                    {patch}
                  </text>
                </g>
              );
            })}
          </g>

          {/* matrix panel (hero) */}
          <rect
            x={MATRIX_X}
            y="148"
            width={MATRIX_W}
            height="276"
            rx="12"
            className="m-panel"
            fill="url(#m-panel)"
          />

          <text x={MATRIX_X + 18} y="174" className="m-matrix-title">
            Cosine similarity
          </text>
          <text x={MATRIX_X + 18} y="190" className="m-matrix-axes">
            query tokens × page patches
          </text>

          {/* patch column heads */}
          {PATCHES.map((patch, col) => (
            <text
              key={patch}
              x={COL_CX[col]}
              y="212"
              textAnchor="middle"
              className="m-patch-head"
            >
              {patch}
            </text>
          ))}

          {/* max column head */}
          <text
            x={MAX_CHIP_X + 30}
            y="212"
            textAnchor="middle"
            className="m-max-head"
          >
            row max
          </text>

          {/* rows */}
          {ROWS.map((row, r) => (
            <g key={`row-${row.token}`}>
              {/* token label (row head, right-aligned to matrix) */}
              <g className={`m-token m-token-${r + 1}`}>
                <circle cx="76" cy={ROW_CY[r]} r="8" fill="url(#m-tok-grad)" />
                <text
                  x={MATRIX_X - 14}
                  y={ROW_CY[r] + 4}
                  textAnchor="end"
                  className="m-token-label"
                >
                  {row.token}
                </text>
              </g>

              {/* row band (active comparison highlight) */}
              <rect
                x={MATRIX_X + 6}
                y={ROW_CY[r] - 22}
                width={MATRIX_W - 64}
                height="44"
                rx="7"
                className={`m-rowband m-rowband-${r + 1}`}
              />

              {/* 4 cells */}
              {row.vals.map((value, c) => (
                <g key={`cell-${r}-${c}`}>
                  <rect
                    x={COL_CX[c] - 30}
                    y={ROW_CY[r] - 17}
                    width="60"
                    height="34"
                    rx="6"
                    className={`m-cell m-cell-r${r + 1}${
                      c === row.maxCol ? ` m-winner m-winner-${r + 1}` : ""
                    }`}
                  />
                  <text
                    x={COL_CX[c]}
                    y={ROW_CY[r] + 4}
                    textAnchor="middle"
                    className={`m-cell-val m-cell-val-r${r + 1}${
                      c === row.maxCol ? " m-winner-val" : ""
                    }`}
                  >
                    {value.toFixed(2)}
                  </text>
                </g>
              ))}

              {/* arrow + surviving max chip */}
              <path
                d={`M ${COL_CX[3] + 30} ${ROW_CY[r]} H ${MAX_CHIP_X}`}
                className={`m-max-line m-max-line-${r + 1}`}
                markerEnd="url(#m-arrow)"
              />
              <rect
                x={MAX_CHIP_X}
                y={ROW_CY[r] - 16}
                width="60"
                height="32"
                rx="7"
                className={`m-max-chip m-max-chip-${r + 1}`}
              />
              <text
                x={MAX_CHIP_X + 30}
                y={ROW_CY[r] + 4}
                textAnchor="middle"
                className={`m-max-val m-max-val-${r + 1}`}
              >
                {MAXIMA[r].toFixed(2)}
              </text>
            </g>
          ))}

          {/* ── Result: score-of-page equation first, then rank badge ── */}
          <g className="m-sum">
            <text x="320" y="440" textAnchor="middle" className="m-sum-eq">
              <tspan className="m-sum-lhs">score of page 7</tspan>
              <tspan className="m-sum-equals"> = </tspan>
              {MAXIMA.map((m) => m.toFixed(2)).join("  +  ")}
              <tspan className="m-sum-equals"> = </tspan>
              <tspan className="m-sum-total">{TOTAL}</tspan>
            </text>
          </g>

          <g className="m-rank">
            <rect
              x="256"
              y="458"
              width="128"
              height="26"
              rx="13"
              className="m-rank-badge"
            />
            <text
              x="320"
              y="475"
              textAnchor="middle"
              className="m-rank-badge-text"
            >
              Rank #1
            </text>
          </g>
        </svg>
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
        <div className="commerce-identity">
          <span className="milvus-brand" aria-label="Milvus">
            <svg
              className="milvus-brand-logo"
              viewBox="0 0 105 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
              aria-hidden="true"
            >
              <path
                d="M16.9129 18.03C20.1664 18.03 22.8039 15.3353 22.8039 12.0111C22.8039 8.68688 20.1664 5.99209 16.9129 5.99209C13.6594 5.99209 11.0219 8.68688 11.0219 12.0111C11.0219 15.3353 13.6594 18.03 16.9129 18.03Z"
                fill="currentColor"
              />
              <path
                fillRule="evenodd"
                clipRule="evenodd"
                d="M7.97503 3.51684C12.6472 -1.17228 20.2235 -1.17228 24.8956 3.51684C29.5753 8.20595 29.5753 15.8085 24.8956 20.4899C20.2235 25.1713 12.6472 25.1713 7.97503 20.4822L0.338563 12.8182C-0.112854 12.3647 -0.112854 11.6344 0.338563 11.1808L7.97503 3.51684ZM10.661 18.2376C14.0917 21.6814 19.6592 21.6814 23.09 18.2376C26.5207 14.7938 26.5207 9.21296 23.0824 5.76915C19.6517 2.32534 14.0842 2.32534 10.6534 5.76915L5.04082 11.3961C4.70978 11.7343 4.70978 12.2724 5.04082 12.603L10.661 18.2376Z"
                fill="currentColor"
              />
              <path
                d="M31.5385 7.75236L34.9015 11.1885C35.3454 11.642 35.3454 12.3723 34.9015 12.8335L31.5385 16.2696C31.3428 16.4695 31.0043 16.285 31.0645 16.0006C31.6438 13.3793 31.6438 10.6427 31.0645 8.02141C30.9968 7.73699 31.3353 7.54481 31.5385 7.75236Z"
                fill="currentColor"
              />
              <path
                d="M59.9645 4.82776C59.9645 4.34213 60.1157 3.92272 60.4398 3.5916C60.7638 3.26049 61.1527 3.08389 61.6496 3.08389C62.125 3.08389 62.5354 3.26049 62.8595 3.5916C63.1836 3.92272 63.3564 4.34213 63.3564 4.82776C63.3564 5.3134 63.1836 5.73282 62.8595 6.06393C62.5354 6.39505 62.125 6.54957 61.6496 6.54957C61.1743 6.54957 60.7638 6.39505 60.4398 6.06393C60.1373 5.71074 59.9645 5.3134 59.9645 4.82776Z"
                fill="currentColor"
              />
              <path
                d="M60.1373 8.8453C60.1373 8.33759 60.2885 7.9844 60.5694 7.78573C60.8503 7.58706 61.2392 7.47669 61.7145 7.47669C62.0601 7.47669 62.3626 7.52084 62.6435 7.60914C62.9243 7.69743 63.0972 7.74158 63.1836 7.76366V17.3439H60.1373V8.8453Z"
                fill="currentColor"
              />
              <path
                d="M41.9459 11.0084C41.9459 10.3683 42.0539 9.8164 42.27 9.35284C42.486 8.88928 42.7885 8.49194 43.1774 8.16083C43.5663 7.82971 44.02 7.58689 44.5385 7.43237C45.057 7.27785 45.6403 7.18955 46.2453 7.18955C47.0015 7.18955 47.6712 7.322 48.2762 7.58689C48.8595 7.85178 49.3564 8.20498 49.7237 8.66854C50.091 8.20498 50.5663 7.85178 51.1496 7.58689C51.733 7.322 52.4027 7.18955 53.1589 7.18955C53.7638 7.18955 54.3255 7.27785 54.8441 7.43237C55.3626 7.58689 55.8163 7.82971 56.2052 8.16083C56.5941 8.49194 56.8965 8.88928 57.1126 9.35284C57.3286 9.8164 57.4367 10.3683 57.4367 11.0084V17.3217H54.3904V11.8914C54.3904 11.2292 54.2607 10.7656 54.0015 10.4786C53.7422 10.1917 53.3749 10.0592 52.878 10.0592C52.3811 10.0592 51.9922 10.2358 51.6681 10.5669C51.3657 10.898 51.2144 11.4278 51.2144 12.1784V17.3438H48.1681V12.1784C48.1681 11.4499 48.0169 10.9201 47.7144 10.5669C47.412 10.2358 47.0231 10.0592 46.5262 10.0592C46.0508 10.0592 45.662 10.1917 45.4027 10.4786C45.1218 10.7656 44.9922 11.2292 44.9922 11.8914V17.3217H41.9459V11.0084Z"
                fill="currentColor"
              />
              <path
                d="M66.4245 3.39294C66.1436 3.5916 65.9924 3.94479 65.9924 4.4525V17.3218H69.0387V3.37086C68.9523 3.34878 68.7794 3.30463 68.4986 3.21634C68.2177 3.12804 67.9152 3.08389 67.5695 3.08389C67.0942 3.08389 66.7054 3.19427 66.4245 3.39294Z"
                fill="currentColor"
              />
              <path
                d="M74.6122 17.3218C74.2234 16.6154 73.8345 15.887 73.4672 15.1365C73.1 14.3862 72.7541 13.6794 72.4518 13.0394C72.1493 12.3992 71.89 11.8253 71.6956 11.3838C71.5011 10.9203 71.3715 10.6333 71.3067 10.5008C71.1987 10.236 71.0906 9.94899 70.9826 9.63995C70.853 9.30883 70.8098 9.02187 70.8098 8.77905C70.8098 8.42586 70.9394 8.11681 71.1987 7.85192C71.4579 7.6091 71.8252 7.47666 72.3437 7.47666C72.7542 7.47666 73.0783 7.52081 73.3376 7.60911C73.5963 7.69724 73.7265 7.74158 73.7697 7.76363C73.9857 8.40379 74.2018 9.04394 74.461 9.6841C74.6519 10.1982 74.8427 10.6838 75.0224 11.1409L75.0231 11.1428C75.0669 11.2543 75.1101 11.364 75.1524 11.4721C75.3684 12.024 75.5629 12.5096 75.7573 12.929C75.9301 13.3484 76.0598 13.6354 76.1462 13.812C76.1942 13.7139 76.2489 13.5817 76.3177 13.4154C76.3726 13.2826 76.4368 13.1273 76.5135 12.9511C76.6286 12.6863 76.7438 12.3922 76.8654 12.0817C76.9262 11.9264 76.9888 11.7663 77.0536 11.6046C77.1508 11.3618 77.248 11.1135 77.3452 10.8652C77.4425 10.6169 77.5397 10.3684 77.6369 10.1256C77.8314 9.63995 78.0042 9.19846 78.1339 8.86735C78.153 8.82331 78.1711 8.78035 78.1889 8.73824C78.2511 8.59038 78.309 8.45294 78.3931 8.31549C78.4795 8.13889 78.6092 8.00645 78.7388 7.874C78.8684 7.74155 79.0413 7.65325 79.2141 7.58703C79.4086 7.52081 79.6462 7.49873 79.9055 7.49873C80.1647 7.49873 80.3808 7.52081 80.5968 7.58703C80.7913 7.63118 80.9857 7.69741 81.1369 7.76363C81.2882 7.82985 81.4178 7.89607 81.5042 7.94022C81.5539 7.97825 81.5964 8.00172 81.6276 8.01899C81.6508 8.03178 81.6679 8.0412 81.6771 8.05059C81.6339 8.29341 81.5042 8.62453 81.3098 9.11017C81.2393 9.27824 81.1659 9.45793 81.0887 9.64712L81.0877 9.64969C80.9522 9.98158 80.8049 10.3427 80.64 10.7216C80.3808 11.3176 80.0999 11.9578 79.7974 12.62C79.6546 12.9328 79.5117 13.2407 79.3711 13.5437L79.3697 13.5465C79.2133 13.8837 79.0593 14.2155 78.9116 14.5405C78.6308 15.1585 78.3499 15.7104 78.0906 16.2181C77.8314 16.7037 77.637 17.101 77.5074 17.3659L74.6122 17.366V17.3218Z"
                fill="currentColor"
              />
              <path
                d="M88.0724 14.7391C86.8841 14.7391 86.3008 14.0769 86.3008 12.7304V7.76363C86.2144 7.74155 86.0415 7.69741 85.7607 7.60911C85.4798 7.52081 85.1773 7.47666 84.8317 7.47666C84.3564 7.47666 83.9675 7.58703 83.6866 7.7857C83.4057 7.98437 83.2545 8.33756 83.2545 8.84527V13.3043C83.2545 14.0107 83.3841 14.6508 83.6218 15.1806C83.881 15.7104 84.2051 16.1519 84.6372 16.5051C85.0693 16.8583 85.5878 17.1232 86.1712 17.2998C86.7545 17.4763 87.3811 17.5646 88.0724 17.5646C88.7422 17.5646 89.3903 17.4763 89.9736 17.2998C90.5786 17.1232 91.0755 16.8583 91.5076 16.5051C91.9397 16.1519 92.2854 15.7104 92.523 15.1806C92.7607 14.6508 92.8903 14.0327 92.8903 13.3043V7.76363C92.8039 7.74155 92.6311 7.69741 92.3502 7.60911C92.0693 7.52081 91.7669 7.47666 91.4212 7.47666C90.9459 7.47666 90.557 7.58703 90.2761 7.7857C89.9952 7.98437 89.844 8.33756 89.844 8.84527V12.7304C89.844 14.0769 89.2607 14.7391 88.0724 14.7391Z"
                fill="currentColor"
              />
              <path
                d="M95.98 14.3196C96.088 14.4079 96.2392 14.4962 96.4769 14.6065C96.7146 14.7169 96.9954 14.8273 97.3195 14.9156C97.6436 15.0039 98.0108 15.0922 98.4213 15.1584C98.8318 15.2246 99.2639 15.2688 99.7176 15.2688C100.236 15.2688 100.625 15.2246 100.884 15.1142C101.144 15.026 101.273 14.8494 101.273 14.6065C101.273 14.3416 101.165 14.1651 100.949 14.0768C100.733 13.9885 100.387 13.9002 99.9121 13.8339L98.9183 13.7236C98.4213 13.6573 97.946 13.569 97.4923 13.4366C97.0386 13.3042 96.6281 13.1276 96.2825 12.8847C95.9368 12.6419 95.6559 12.3329 95.4615 11.9576C95.267 11.5823 95.159 11.1188 95.159 10.5449C95.159 10.0592 95.2454 9.61774 95.4183 9.2204C95.5911 8.82306 95.8504 8.46987 96.2176 8.16083C96.5849 7.85179 97.0602 7.60897 97.622 7.45445C98.1837 7.27785 98.875 7.18955 99.6528 7.18955C100.452 7.18955 101.122 7.2337 101.705 7.34407C102.289 7.45444 102.764 7.60897 103.196 7.82971C103.434 7.96216 103.628 8.11667 103.779 8.29327C103.931 8.46986 103.995 8.71269 103.995 8.97758C103.995 9.17625 103.952 9.37492 103.866 9.52944C103.779 9.68396 103.693 9.83847 103.585 9.94885C103.477 10.0592 103.369 10.1475 103.282 10.2137C103.196 10.28 103.131 10.3241 103.11 10.3241C103.083 10.2968 103.031 10.2611 102.96 10.2117C102.916 10.1812 102.865 10.1455 102.807 10.1034C102.634 9.99299 102.418 9.9047 102.116 9.79433C101.835 9.70603 101.489 9.61773 101.1 9.52944C100.711 9.44114 100.258 9.41907 99.7393 9.41907C99.1775 9.41907 98.767 9.48529 98.551 9.61774C98.3349 9.75018 98.2269 9.92677 98.2269 10.1475C98.2269 10.3683 98.3349 10.5228 98.5294 10.6111C98.7238 10.6994 99.0263 10.7877 99.4368 10.8539L101.165 11.0967C101.597 11.1629 102.008 11.2512 102.397 11.3837C102.786 11.5161 103.153 11.6927 103.455 11.9135C103.758 12.1563 104.017 12.4433 104.19 12.7964C104.363 13.1496 104.471 13.5911 104.471 14.1209C104.471 15.2025 104.06 16.0414 103.261 16.6595C102.461 17.2555 101.338 17.5645 99.9121 17.5645C99.1775 17.5645 98.5078 17.5204 97.9244 17.41C97.3411 17.2996 96.8658 17.1892 96.4337 17.0568C96.0232 16.9243 95.6775 16.7698 95.4183 16.6374C95.2318 16.5263 95.0901 16.438 94.985 16.3725C94.9439 16.347 94.9085 16.3249 94.8781 16.3063L95.98 14.3196Z"
                fill="currentColor"
              />
            </svg>
            <span className="milvus-version-tag">3.0</span>
          </span>
        </div>
        <div className="commerce-title">
          <h1>Visual PDF Retrieval with EmbeddingList</h1>
          <a
            className="source-link"
            href="https://github.com/zc277584121/milvus3-demos/tree/main/demos/embedding-list-max-sim"
            target="_blank"
            rel="noreferrer"
            aria-label="View source on GitHub"
          >
            <svg
              className="source-link-icon"
              viewBox="0 0 32 32"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
              aria-hidden="true"
            >
              <path
                fillRule="evenodd"
                clipRule="evenodd"
                d="M16 0C7.16 0 0 7.16 0 16C0 23.08 4.58 29.06 10.94 31.18C11.74 31.32 12.04 30.84 12.04 30.42C12.04 30.04 12.02 28.78 12.02 27.44C8 28.18 6.96 26.46 6.64 25.56C6.46 25.1 5.68 23.68 5 23.3C4.44 23 3.64 22.26 4.98 22.24C6.24 22.22 7.14 23.4 7.44 23.88C8.88 26.3 11.18 25.62 12.1 25.2C12.24 24.16 12.66 23.46 13.12 23.06C9.56 22.66 5.84 21.28 5.84 15.16C5.84 13.42 6.46 11.98 7.48 10.86C7.32 10.46 6.76 8.82 7.64 6.62C7.64 6.62 8.98 6.2 12.04 8.26C13.32 7.9 14.68 7.72 16.04 7.72C17.4 7.72 18.76 7.9 20.04 8.26C23.1 6.18 24.44 6.62 24.44 6.62C25.32 8.82 24.76 10.46 24.6 10.86C25.62 11.98 26.24 13.4 26.24 15.16C26.24 21.3 22.5 22.66 18.94 23.06C19.52 23.56 20.02 24.52 20.02 26.02C20.02 28.16 20 29.88 20 30.42C20 30.84 20.3 31.34 21.1 31.18C27.42 29.06 32 23.06 32 16C32 7.16 24.84 0 16 0V0Z"
                fill="currentColor"
              />
            </svg>
            <span>Source</span>
          </a>
          <MaxSimExplainer />
        </div>
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
              {loading ? "Searching…" : "Search 40 pages"}
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
