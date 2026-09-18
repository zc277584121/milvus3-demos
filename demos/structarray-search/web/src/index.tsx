import {
  Fragment,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";

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

export const structArrayDemo: DemoDefinition = {
  id: "structarray-search",
  title: "Data Curation in Autonomous Driving with StructArray",
  shortTitle: "StructArray Hybrid",
  capability: "StructArray parent + child hybrid search",
  description:
    "One query, one model, three answers: search the video summary, search every object observation, then collapse and weighted-rerank both together.",
  route: "/demos/structarray-search",
  status: "available",
  accent: "#0b7a55",
  highlights: [
    "BGE-M3 ONNX embeds summaries, observations, and the query at 1024 dims",
    "Child path groups elements back to one parent video per result",
    "Fusion collapses top-3 child scores and reranks against the parent",
  ],
};

const API_BASE = `${import.meta.env.BASE_URL}api/v1`;

export interface QueryPreset {
  id: string;
  text: string;
  scene_terms?: string[];
  object_terms?: string[];
  color_terms?: string[];
}

export interface ModelStatus {
  model_id: string;
  revision: string;
  vector_dimension: number;
  dense_output_name: string;
  device: "cpu";
  loaded: boolean;
  cache_available: boolean;
}

export interface DatasetStatus {
  dataset_id: string;
  dataset_version: number;
  video_count: number;
  observation_count: number;
  evidence_frame_count: number;
  manifest_sha256: string;
}

export interface MilvusStatus {
  milvus_uri: string;
  server_version: string;
  collection_name: string;
  collection_exists: boolean;
  raw_collection_names: string[];
  row_count: number | null;
  index_names: string[];
}

export interface StructArrayStatus {
  status: "ready" | "not_prepared";
  sample_size: number;
  model: ModelStatus;
  dataset: DatasetStatus;
  milvus: MilvusStatus;
  query_presets: QueryPreset[];
}

export interface ObservationHit {
  description: string;
  object_type: string;
  frame_id: number;
  image_id: string;
  bbox: number[];
  vehicle_type: string;
  color: string;
  orientation: string;
  lights_on: string;
  v_ego: number;
  a_ego: number;
  clip_id: string;
  raw_frame: string | null;
  annotated_frame: string | null;
}

export interface VideoPreview {
  frame_id: number;
  annotated_frame: string | null;
  raw_frame: string | null;
}

export interface BaseResult {
  rank: number;
  score: number;
  video_id: string;
  video_summary: string;
  source_ordinal: number;
  scene_match?: boolean;
  object_match?: boolean;
  color_match?: boolean;
  actual_scene?: string;
  actual_object?: string;
  actual_color?: string;
}

export interface ParentResult extends BaseResult {
  preview: VideoPreview | null;
}

export interface ChildResult extends BaseResult {
  offset: number;
  observation: ObservationHit;
}

export interface FusionResult extends BaseResult {
  preview: VideoPreview | null;
  child_observation?: ObservationHit | null;
}

export interface SearchResponse {
  status: "passed";
  query: string;
  query_vector_dimension: number;
  limit: number;
  parent_weight: number;
  child_weight: number;
  collapse_strategy: string;
  collapse_topk: number;
  latency_ms: number;
  score_semantics: string;
  paths: {
    parent: {
      anns_field: string;
      label: string;
      results: ParentResult[];
    };
    child: {
      anns_field: string;
      group_by_field: string;
      label: string;
      results: ChildResult[];
    };
    fusion: {
      label: string;
      ranker: string;
      parent_anns_field: string;
      child_anns_field: string;
      results: FusionResult[];
    };
  };
  ground_truth?: {
    scene_terms: string[];
    object_terms: string[];
    color_terms?: string[];
    matched_video_ids: string[];
    matched_count: number;
  };
  path_recall?: {
    parent: { matched: number; gt_count: number; recall: number; ndcg: number };
    child: { matched: number; gt_count: number; recall: number; ndcg: number };
    fusion: { matched: number; gt_count: number; recall: number; ndcg: number };
  };
}

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new Error(body || `Request failed with status ${response.status}`);
  }
  return (await response.json()) as T;
}

export async function fetchStructArrayStatus(): Promise<StructArrayStatus> {
  return readJson<StructArrayStatus>(
    await fetch(`${API_BASE}/status`, {
      headers: { Accept: "application/json" },
    }),
  );
}

export async function runHybridSearch(query: string): Promise<SearchResponse> {
  return readJson<SearchResponse>(
    await fetch(`${API_BASE}/search`, {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query,
        limit: 8,
        parent_weight: 0.5,
        collapse_strategy: "topk_sum",
      }),
    }),
  );
}

function evidenceUrl(
  kind: "raw" | "annotated",
  fileName: string | null,
): string | null {
  if (!fileName) return null;
  return `${API_BASE}/evidence/${kind}/${encodeURIComponent(fileName)}`;
}

/* ------------------------------------------------------------------ */
/* Small presentational pieces                                         */
/* ------------------------------------------------------------------ */

function CardMedia({ src, alt }: { src: string | null; alt: string }) {
  if (src === null) {
    return <div className="card-media card-media--missing">no frame</div>;
  }
  return (
    <div className="card-media">
      <img
        src={src}
        alt={alt}
        loading="lazy"
        onError={(event) => {
          event.currentTarget.style.display = "none";
        }}
      />
    </div>
  );
}

interface TermAnnotation {
  hits: string[];
  misses: string[];
}

function annotateTerms(text: string, annotation: TermAnnotation): ReactNode[] {
  // Mismatched words win so a contradicting actual value is never re-marked as
  // a hit; longest-first so "local_residential" beats "local" etc.
  const classByTerm = new Map<string, string>();
  for (const term of annotation.misses) {
    if (term) classByTerm.set(term.toLowerCase(), "gt-miss");
  }
  for (const term of annotation.hits) {
    const key = term.toLowerCase();
    if (key && !classByTerm.has(key)) classByTerm.set(key, "gt-term");
  }
  if (classByTerm.size === 0) return [text];
  const sorted = [...classByTerm.keys()].sort((a, b) => b.length - a.length);
  const pattern = new RegExp(`(${sorted.map(escapeRegExp).join("|")})`, "gi");
  const nodes: ReactNode[] = [];
  let lastIndex = 0;
  for (const match of text.matchAll(pattern)) {
    const index = match.index ?? 0;
    if (index > lastIndex) nodes.push(text.slice(lastIndex, index));
    const className = classByTerm.get(match[0].toLowerCase()) ?? "gt-term";
    nodes.push(
      <mark key={index} className={className}>
        {match[0]}
      </mark>,
    );
    lastIndex = index + match[0].length;
  }
  if (lastIndex < text.length) nodes.push(text.slice(lastIndex));
  return nodes;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

const EMPTY_ANNOTATION: TermAnnotation = { hits: [], misses: [] };

function uniqueWords(terms: readonly (string | undefined)[]): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const term of terms) {
    if (term) {
      for (const word of term.toLowerCase().split(/\s+/)) {
        if (word && !seen.has(word)) {
          seen.add(word);
          result.push(word);
        }
      }
    }
  }
  return result;
}

function buildAnnotation(
  groundTruth: SearchResponse["ground_truth"] | null,
  result: BaseResult,
): TermAnnotation {
  if (groundTruth === undefined || groundTruth === null)
    return EMPTY_ANNOTATION;
  const hits = uniqueWords([
    ...(groundTruth.scene_terms ?? []),
    ...(groundTruth.object_terms ?? []),
    ...(groundTruth.color_terms ?? []),
  ]);
  // A detected word is "wrong" only when the corresponding intent dimension
  // failed — so a passing card never strikes its own correctly matched value.
  const misses: string[] = [];
  if (result.scene_match === false) {
    misses.push(
      ...(result.actual_scene ? result.actual_scene.split(/\s+/) : []),
    );
  }
  if (result.object_match === false) {
    misses.push(
      ...(result.actual_object ? result.actual_object.split(/\s+/) : []),
    );
  }
  if (result.color_match === false) {
    misses.push(
      ...(result.actual_color ? result.actual_color.split(/\s+/) : []),
    );
  }
  return { hits, misses: uniqueWords(misses) };
}

function KeyText({
  text,
  annotation,
}: {
  text: string;
  annotation: TermAnnotation;
}) {
  const trimmed = text.trim();
  return (
    <div className="card-text-block">
      <p className="card-text">{annotateTerms(trimmed, annotation)}</p>
    </div>
  );
}

function CardHead({ rank, gt }: { rank: number; gt: boolean }) {
  return (
    <div className="card-head">
      <span className="card-rank-wrap">
        <span className="card-rank">#{rank}</span>
        <GroundTruthChip gt={gt} />
      </span>
    </div>
  );
}

function DetectedValuesRow({ result }: { result: BaseResult }) {
  // Only preset queries carry intent match flags; freeform queries render no row.
  if (result.scene_match === undefined) return null;
  const items = [
    {
      label: "scene",
      value: result.actual_scene ?? "",
      match: result.scene_match,
    },
    {
      label: "object",
      value: result.actual_object ?? "",
      match: result.object_match,
    },
    {
      label: "color",
      value: result.actual_color ?? "",
      match: result.color_match,
    },
  ];
  return (
    <div className="card-detected">
      {items.map((item) => {
        const valueClass =
          item.match === true
            ? "gt-term"
            : item.match === false
              ? "gt-miss"
              : "";
        return (
          <span key={item.label} className="card-detected-item">
            <span className="card-detected-label">{item.label}</span>
            <span className={`card-detected-value ${valueClass}`.trim()}>
              {item.value || "—"}
            </span>
          </span>
        );
      })}
    </div>
  );
}

function CardBody({
  src,
  alt,
  texts,
  annotation = EMPTY_ANNOTATION,
}: {
  src: string | null;
  alt: string;
  texts: string[];
  annotation?: TermAnnotation;
}) {
  return (
    <div className="card-body">
      <CardMedia src={src} alt={alt} />
      <div className="card-texts">
        {texts.map((text, index) => (
          <KeyText key={index} text={text} annotation={annotation} />
        ))}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Result cards                                                        */
/* ------------------------------------------------------------------ */

function CardShell({
  tone,
  gt = false,
  children,
}: {
  tone: "parent" | "child" | "fusion";
  gt?: boolean;
  children: ReactNode;
}) {
  return (
    <li
      className={`result-card result-card--${tone}${gt ? " result-card--gt" : ""}`}
    >
      {children}
    </li>
  );
}

function GroundTruthChip({ gt }: { gt: boolean }) {
  if (!gt) return null;
  return (
    <span
      className="gt-chip"
      title="Ground truth: matches the scene AND the object AND the color"
    >
      GT
    </span>
  );
}

function ParentCard({
  result,
  annotation,
}: {
  result: ParentResult;
  annotation?: TermAnnotation;
}) {
  const gt =
    result.scene_match === true &&
    result.object_match === true &&
    result.color_match === true;
  return (
    <CardShell tone="parent" gt={gt}>
      <CardHead rank={result.rank} gt={gt} />
      <CardBody
        src={evidenceUrl("annotated", result.preview?.annotated_frame ?? null)}
        alt={`preview frame ${result.preview?.frame_id ?? ""}`}
        texts={[result.video_summary]}
        annotation={annotation}
      />
      <DetectedValuesRow result={result} />
    </CardShell>
  );
}

function ChildCard({
  result,
  annotation,
}: {
  result: ChildResult;
  annotation?: TermAnnotation;
}) {
  const obs = result.observation;
  const gt =
    result.scene_match === true &&
    result.object_match === true &&
    result.color_match === true;
  return (
    <CardShell tone="child" gt={gt}>
      <CardHead rank={result.rank} gt={gt} />
      <CardBody
        src={evidenceUrl("annotated", obs.annotated_frame)}
        alt={`frame ${obs.frame_id}`}
        texts={[result.video_summary]}
        annotation={annotation}
      />
      <DetectedValuesRow result={result} />
    </CardShell>
  );
}

function FusionCard({
  result,
  annotation,
}: {
  result: FusionResult;
  annotation?: TermAnnotation;
}) {
  const gt =
    result.scene_match === true &&
    result.object_match === true &&
    result.color_match === true;
  const childObs = result.child_observation ?? null;
  return (
    <CardShell tone="fusion" gt={gt}>
      <CardHead rank={result.rank} gt={gt} />
      <CardBody
        src={evidenceUrl(
          "annotated",
          childObs?.annotated_frame ?? result.preview?.annotated_frame ?? null,
        )}
        alt={`frame ${childObs?.frame_id ?? result.preview?.frame_id ?? ""}`}
        texts={
          childObs
            ? [result.video_summary, childObs.description]
            : [result.video_summary]
        }
        annotation={annotation}
      />
      <DetectedValuesRow result={result} />
    </CardShell>
  );
}

/* ------------------------------------------------------------------ */
/* Search pipeline (data model)                                        */
/* ------------------------------------------------------------------ */

interface SchemaColumn {
  name: string;
  type: string;
}

const PARENT_COLUMNS: SchemaColumn[] = [
  { name: "video_id", type: "VARCHAR 64 · PK" },
  { name: "video_summary", type: "VARCHAR 4096" },
  { name: "source_ordinal", type: "INT64" },
  { name: "weather", type: "VARCHAR 16" },
  { name: "road_type", type: "VARCHAR 32" },
  { name: "summary_vector", type: "FLOAT_VECTOR 1024" },
  { name: "observations", type: "ARRAY<STRUCT> · max 128" },
];

const OBSERVATION_COLUMNS: SchemaColumn[] = [
  { name: "description", type: "VARCHAR 512" },
  { name: "description_vector", type: "FLOAT_VECTOR 1024" },
  { name: "object_type", type: "VARCHAR 32" },
  { name: "frame_id", type: "INT64" },
  { name: "image_id", type: "VARCHAR 64" },
  { name: "bbox", type: "INT64 ×4" },
];

interface ExampleObservation {
  description: string;
  descriptionVector: string;
  objectType: string;
  frameId: number;
  imageId: string;
  bbox: string;
}

interface ExampleVideo {
  id: string;
  summary: string;
  summaryVector: string;
  ordinal: number;
  weather: string;
  roadType: string;
  observationCount: number;
  observations: ExampleObservation[];
}

const EXAMPLE_ROWS: ExampleVideo[] = [
  {
    id: "video_0001",
    summary:
      "Sunny highway. The ego vehicle follows a white truck for several segments, then a sedan passes on the left before an exit ramp. | Detected objects: 2 car, 1 truck",
    summaryVector: "[0.0023, -1.2345, 0.8912, -0.4421, 0.1278, …]",
    ordinal: 0,
    weather: "sunny",
    roadType: "motorway",
    observationCount: 12,
    observations: [
      {
        description: "A white box truck driving ahead in the right lane.",
        descriptionVector: "[0.9121, 0.3378, -1.2043, 0.5564, -0.0812, …]",
        objectType: "truck",
        frameId: 120,
        imageId: "video_0001_frame_000120_truck_000",
        bbox: "640, 300, 960, 540",
      },
      {
        description: "A blue sedan passing on the left lane.",
        descriptionVector: "[0.4476, -0.9920, 0.1105, 1.3374, -0.6631, …]",
        objectType: "car",
        frameId: 180,
        imageId: "video_0001_frame_000180_car_000",
        bbox: "320, 280, 560, 520",
      },
    ],
  },
  {
    id: "video_0002",
    summary:
      "Cloudy ramp. The ego vehicle merges onto a highway ramp behind a dark SUV, then cruises toward a toll plaza. | Detected objects: 1 suv, 1 van",
    summaryVector: "[-0.7710, 0.2294, -1.0833, 0.6645, 0.3901, …]",
    ordinal: 1,
    weather: "cloudy",
    roadType: "ramp",
    observationCount: 6,
    observations: [
      {
        description: "A dark SUV ahead, brake lights on while merging.",
        descriptionVector: "[-0.3022, 1.1187, 0.7720, -0.4189, -1.0523, …]",
        objectType: "suv",
        frameId: 45,
        imageId: "video_0002_frame_000045_suv_000",
        bbox: "700, 320, 1040, 560",
      },
      {
        description: "A white van waiting at the toll plaza booth ahead.",
        descriptionVector: "[0.6184, -0.2271, 0.8832, 0.1029, -0.5408, …]",
        objectType: "van",
        frameId: 300,
        imageId: "video_0002_frame_000300_van_000",
        bbox: "560, 300, 820, 540",
      },
    ],
  },
  {
    id: "video_0003",
    summary:
      "Residential street at dusk. A cyclist crosses ahead and a parked car sits on the right, then the ego vehicle stops at a crosswalk. | Detected objects: 1 bicycle, 2 car, 1 person",
    summaryVector: "[0.5581, -0.1136, 0.9027, 0.2718, -0.7340, …]",
    ordinal: 2,
    weather: "cloudy",
    roadType: "local_residential",
    observationCount: 9,
    observations: [
      {
        description: "A cyclist crossing the road from right to left.",
        descriptionVector: "[0.2019, 0.6408, -0.5563, 0.8894, 0.0721, …]",
        objectType: "bicycle",
        frameId: 210,
        imageId: "video_0003_frame_000210_bicycle_000",
        bbox: "480, 360, 720, 600",
      },
      {
        description: "A parked sedan on the right shoulder at dusk.",
        descriptionVector: "[-0.4110, 0.3027, 0.6681, -0.2145, 0.4932, …]",
        objectType: "car",
        frameId: 240,
        imageId: "video_0003_frame_000240_car_000",
        bbox: "820, 380, 1080, 620",
      },
    ],
  },
];

function SchemaTableRow({ row }: { row: ExampleVideo }) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const closeTimer = useRef<number | null>(null);
  const remaining = row.observationCount - row.observations.length;

  const scheduleOpen = () => {
    if (closeTimer.current !== null) {
      window.clearTimeout(closeTimer.current);
      closeTimer.current = null;
    }
    setOpen(true);
  };

  const scheduleClose = () => {
    if (closeTimer.current !== null) {
      window.clearTimeout(closeTimer.current);
    }
    closeTimer.current = window.setTimeout(() => setOpen(false), 140);
  };

  useLayoutEffect(() => {
    if (!open) {
      setPos(null);
      return undefined;
    }
    const button = triggerRef.current;
    if (!button) return undefined;
    const rect = button.getBoundingClientRect();
    const panelWidth = 780;
    const margin = 12;
    // Anchor the panel's bottom-right corner to the trigger's bottom-right so
    // it opens to the lower-left and never runs off the right edge.
    let left = rect.right - panelWidth;
    left = Math.max(
      margin,
      Math.min(left, window.innerWidth - margin - panelWidth),
    );
    setPos({ top: rect.bottom + 10, left });
    return undefined;
  }, [open]);

  const nestedTable = (
    <table className="schema-table schema-table--nested">
      <thead>
        <tr>
          {OBSERVATION_COLUMNS.map((column) => (
            <th key={column.name}>
              <code>{column.name}</code>
              <span>{column.type}</span>
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {row.observations.map((observation) => (
          <tr key={observation.imageId}>
            <td className="td-desc">{observation.description}</td>
            <td className="td-vec">
              <code>{observation.descriptionVector}</code>
            </td>
            <td className="td-num">{observation.objectType}</td>
            <td className="td-num">{observation.frameId}</td>
            <td className="td-id">
              <code>{observation.imageId}</code>
            </td>
            <td className="td-num">
              <code>{observation.bbox}</code>
            </td>
          </tr>
        ))}
        {remaining > 0 ? (
          <tr className="schema-ellipsis-row">
            <td colSpan={OBSERVATION_COLUMNS.length}>
              … +{remaining} more observations
            </td>
          </tr>
        ) : null}
      </tbody>
    </table>
  );

  return (
    <tr className="schema-data-row">
      <td className="td-id">
        <code>{row.id}</code>
      </td>
      <td className="td-summary" title={row.summary}>
        <span className="td-summary-clamp">{row.summary}</span>
      </td>
      <td className="td-num">{row.ordinal}</td>
      <td className="td-scalar">{row.weather}</td>
      <td className="td-scalar">{row.roadType}</td>
      <td className="td-vec">
        <code>{row.summaryVector}</code>
      </td>
      <td className="td-nested">
        <button
          ref={triggerRef}
          type="button"
          className={`nested-toggle${open ? " is-open" : ""}`}
          aria-expanded={open}
          aria-haspopup="dialog"
          onMouseEnter={scheduleOpen}
          onMouseLeave={scheduleClose}
          onClick={() => setOpen((value) => !value)}
        >
          <span className="nested-caret" aria-hidden="true">
            ▸
          </span>
          <span>{row.observationCount} objects</span>
        </button>
      </td>
      {open && pos
        ? createPortal(
            <div
              className="nested-popover"
              style={{ top: pos.top, left: pos.left }}
              role="dialog"
              aria-label={`${row.id} nested observations`}
              onMouseEnter={scheduleOpen}
              onMouseLeave={scheduleClose}
            >
              <div className="nested-popover-head">
                <span className="nested-popover-kicker">StructArray</span>
                <span className="nested-popover-title">
                  {row.id} · {row.observationCount} nested observations
                </span>
              </div>
              <div className="nested-popover-scroll">{nestedTable}</div>
            </div>,
            document.body,
          )
        : null}
    </tr>
  );
}

function SchemaTable({ rowCount }: { rowCount: number }) {
  const remainingRows = Math.max(0, rowCount - EXAMPLE_ROWS.length);
  return (
    <div className="schema-table-scroll">
      <table className="schema-table">
        <thead>
          <tr>
            {PARENT_COLUMNS.map((column) => (
              <th
                key={column.name}
                className={
                  column.name === "observations" ? "th-nested" : undefined
                }
              >
                <code>{column.name}</code>
                <span>{column.type}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {EXAMPLE_ROWS.map((row) => (
            <SchemaTableRow key={row.id} row={row} />
          ))}
          {remainingRows > 0 ? (
            <tr className="schema-ellipsis-row">
              <td colSpan={PARENT_COLUMNS.length}>
                … +{remainingRows} more videos
              </td>
            </tr>
          ) : null}
        </tbody>
      </table>
    </div>
  );
}

function SearchPipeline({ status }: { status: StructArrayStatus | null }) {
  const rowCount =
    status?.milvus.row_count ?? status?.dataset.video_count ?? 30;

  return (
    <section className="search-pipeline" aria-label="Data model">
      <header className="pipeline-head">
        <div className="pipeline-head-titles">
          <span className="pipeline-kicker">Data model</span>
        </div>
      </header>

      <div className="pipeline-body">
        <section className="pipeline-schema">
          <span className="pipeline-section-label">Collection schema</span>
          <SchemaTable rowCount={rowCount} />
        </section>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* Result column                                                       */
/* ------------------------------------------------------------------ */

// Lightweight Python tokenizer: one color per syntactic kind — strings green,
// comments grey, numbers orange, callables blue, keyword arguments purple.
const PY_TOKEN = /("[^"]*"|'[^']*'|#[^\n]*|\b\d+\b|[A-Za-z_][A-Za-z0-9_]*)/g;

const PY_FN = new Set([
  "AnnSearchRequest",
  "WeightedRanker",
  "search",
  "hybrid_search",
  "embed",
]);

const PY_KW = new Set([
  "data",
  "anns_field",
  "search_params",
  "param",
  "params",
  "limit",
  "metric_type",
  "ef",
  "group_by_field",
  "ranker",
  "element_scope",
  "collapse",
  "strategy",
  "topk",
]);

function pythonHighlight(code: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  let lastIndex = 0;
  for (const match of code.matchAll(PY_TOKEN)) {
    const index = match.index ?? 0;
    if (index > lastIndex) {
      nodes.push(
        <span key={`plain-${index}`} className="py-plain">
          {code.slice(lastIndex, index)}
        </span>,
      );
    }
    const token = match[0];
    const first = token[0];
    let className: string;
    if (first === '"' || first === "'") {
      className = "py-string";
    } else if (first === "#") {
      className = "py-comment";
    } else if (/\d/.test(first)) {
      className = "py-number";
    } else if (PY_FN.has(token)) {
      className = "py-fn";
    } else if (PY_KW.has(token)) {
      className = "py-param";
    } else {
      className = "py-plain";
    }
    nodes.push(
      <span key={`${index}-${token}`} className={className}>
        {token}
      </span>,
    );
    lastIndex = index + token.length;
  }
  if (lastIndex < code.length) {
    nodes.push(
      <span key={`plain-${lastIndex}`} className="py-plain">
        {code.slice(lastIndex)}
      </span>,
    );
  }
  return nodes;
}

interface PathMetrics {
  matched: number;
  gt_count: number;
  recall: number;
  ndcg: number;
}

const PLAN_CAPTIONS: Record<"parent" | "child" | "fusion", string> = {
  parent: "Plan 1 · Parent — summary_vector",
  child: "Plan 2 · Child — observations[description_vector]",
  fusion: "Plan 3 · Hybrid — topk_sum(3) + WeightedRanker",
};

function buildPlanCode(
  collection: string,
  dimension: number,
): Record<"parent" | "child" | "fusion", string> {
  return {
    parent: `# parent route: one vector per video row
hits = client.search(
    "${collection}",
    data=[embed(query)],              # ${dimension}-dim BGE-M3 vector
    anns_field="summary_vector",      # FLOAT_VECTOR ${dimension}
    search_params={"metric_type": "COSINE", "params": {"ef": 128}},
    limit=8,                          # one hit = one video row
)`,
    child: `# child route: search inside the observation array
hits = client.search(
    "${collection}",
    data=[embed(query)],
    anns_field="observations[description_vector]",  # FLOAT_VECTOR ${dimension}
    search_params={"metric_type": "COSINE", "params": {"ef": 128}},
    limit=8,
    group_by_field="video_id",        # collapse to one hit per video
)`,
    fusion: `# hybrid: collapse child top-3, then weighted-rerank
parent = AnnSearchRequest(
    data=[embed(query)],
    anns_field="summary_vector",
    param={"metric_type": "COSINE", "params": {"ef": 128}},
    limit=8,
)
child = AnnSearchRequest(
    data=[embed(query)],
    anns_field="observations[description_vector]",
    param={
        "metric_type": "COSINE",
        "params": {
            "ef": 128,
            "element_scope": {"collapse":
                {"strategy": "topk_sum", "topk": 3}},
        },
    },
    limit=32,
)
hits = client.hybrid_search(
    "${collection}", [parent, child],
    ranker=WeightedRanker(parent_weight, child_weight),
    limit=8,
)`,
  };
}

function ResultColumn<T extends { rank: number; score: number }>({
  label,
  tone,
  plan,
  code,
  results,
  metrics,
  render,
  empty,
}: {
  label: string;
  tone: "parent" | "child" | "fusion";
  plan?: string;
  code?: string;
  results: T[];
  metrics?: PathMetrics;
  render: (result: T) => ReactNode;
  empty: string;
}) {
  const [showCode, setShowCode] = useState(false);

  return (
    <section
      className={`result-column result-column--${tone}`}
      aria-label={label}
    >
      {plan ? (
        <div
          className={`column-plan-row${showCode ? " is-active" : ""}`}
          onMouseEnter={() => setShowCode(true)}
          onMouseLeave={() => setShowCode(false)}
        >
          <p className="column-plan">{plan}</p>
          {code ? (
            <div className="column-code-overlay" aria-hidden={!showCode}>
              <pre className="pipeline-code">
                <code>{pythonHighlight(code)}</code>
              </pre>
            </div>
          ) : null}
        </div>
      ) : null}
      {metrics ? (
        <div className="column-metrics">
          <span
            className={`column-dot column-dot--${tone}`}
            aria-hidden="true"
          />
          <span
            className="column-ndcg"
            title="Normalized Discounted Cumulative Gain"
          >
            NDCG {metrics.ndcg.toFixed(4)}
          </span>
          <span
            className={`column-recall${metrics.matched === metrics.gt_count ? " is-full" : ""}`}
            title="Ground-truth videos found in the top results"
          >
            Recall {metrics.matched}/{metrics.gt_count}
          </span>
        </div>
      ) : null}
      {results.length === 0 ? (
        <p className="column-empty">{empty}</p>
      ) : (
        <ol className="column-list">
          {results.map((result) => (
            <Fragment key={result.rank}>{render(result)}</Fragment>
          ))}
        </ol>
      )}
    </section>
  );
}

function QueryComboBox({
  query,
  presets,
  onChange,
  onSelectPreset,
  disabled,
}: {
  query: string;
  presets: QueryPreset[];
  onChange: (value: string) => void;
  onSelectPreset: (value: string) => void;
  disabled: boolean;
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
    <div className="sa-combobox" ref={rootRef}>
      <div className="sa-combobox-control">
        <input
          id="hybrid-query"
          type="text"
          value={query}
          onChange={(event) => onChange(event.target.value)}
          aria-label="Query"
          placeholder="Describe the scene you are looking for…"
        />
        <button
          type="button"
          className="sa-combobox-toggle"
          aria-label="Choose a preset query"
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
        <button
          type="button"
          className="sa-combobox-clear"
          aria-label="Clear query"
          title="Clear query"
          onClick={() => onChange("")}
          disabled={disabled || query === ""}
        >
          ×
        </button>
      </div>
      {open && (
        <ul
          className="sa-combobox-menu"
          role="listbox"
          aria-label="Query presets"
        >
          {presets.map((preset) => (
            <li key={preset.id} role="presentation">
              <button
                type="button"
                role="option"
                aria-selected={preset.text === query}
                onClick={() => {
                  onSelectPreset(preset.text);
                  setOpen(false);
                }}
              >
                <span className="sa-combobox-option-label">{preset.text}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Page                                                                */
/* ------------------------------------------------------------------ */

export function StructArrayDemoPage() {
  const [status, setStatus] = useState<StructArrayStatus | null>(null);
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState<"search" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const requestEpoch = useRef(0);
  const autoRan = useRef(false);

  const refreshStatus = async () => {
    const value = await fetchStructArrayStatus();
    setStatus(value);
    if (query === "" && value.query_presets.length > 0) {
      setQuery(value.query_presets[0].text);
    }
    return value;
  };

  const ready = status?.status === "ready";
  const collectionName =
    status?.milvus.collection_name ??
    "milvus3_demos_structarray_hybrid_synthetic";
  const dimension = status?.model.vector_dimension ?? 1024;
  const groundTruth = result?.ground_truth ?? null;

  async function runSearch(overrideQuery?: string) {
    const submittedQuery = (overrideQuery ?? query).trim();
    if (!submittedQuery) {
      setError("Enter a query before searching.");
      return;
    }
    const epoch = ++requestEpoch.current;
    setResult(null);
    setLoading("search");
    setError(null);
    try {
      const response = await runHybridSearch(submittedQuery);
      if (requestEpoch.current === epoch && response?.paths)
        setResult(response);
    } catch (cause) {
      if (requestEpoch.current === epoch) {
        setError(cause instanceof Error ? cause.message : "Search failed.");
      }
    } finally {
      if (requestEpoch.current === epoch) setLoading(null);
    }
  }

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const value = await refreshStatus();
        if (cancelled) return;
        if (
          value.status === "ready" &&
          value.query_presets.length > 0 &&
          !autoRan.current
        ) {
          autoRan.current = true;
          setQuery(value.query_presets[0].text);
          await runSearch(value.query_presets[0].text);
        }
      } catch {
        if (!cancelled) setError("Backend unavailable.");
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <main className="structarray-hybrid-page">
      <header className="hybrid-topbar">
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
          <h1>Data Curation in Autonomous Driving with StructArray</h1>
          <a
            className="source-link"
            href="https://github.com/zc277584121/milvus3-demos/tree/main/demos/structarray-search"
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
        </div>
      </header>

      <SearchPipeline status={status} />

      <section className="hybrid-query-card" aria-label="Query controls">
        <div className="query-input-row">
          <QueryComboBox
            query={query}
            presets={status?.query_presets ?? []}
            onChange={(value) => {
              requestEpoch.current += 1;
              setQuery(value);
              setResult(null);
              setError(null);
            }}
            onSelectPreset={(value) => {
              setQuery(value);
              setResult(null);
              setError(null);
              runSearch(value);
            }}
            disabled={!ready || loading !== null}
          />
          <button
            className="hybrid-search-button"
            type="button"
            onClick={() => runSearch()}
            disabled={!ready || loading !== null}
          >
            {loading === "search" ? "Searching…" : "Search"}
          </button>
        </div>

        {error ? (
          <p className="hybrid-error" role="alert">
            {error}
          </p>
        ) : null}
      </section>

      <section className="hybrid-results" aria-live="polite">
        {!result && !loading ? (
          <div className="hybrid-empty-state">
            <h2>Ask a question, see the evidence.</h2>
          </div>
        ) : null}
        {loading ? (
          <div className="hybrid-empty-state" role="status">
            <h2>Searching…</h2>
          </div>
        ) : null}
        {result ? (
          <div className="hybrid-results-grid">
            <ResultColumn
              label="Parent-only"
              tone="parent"
              plan={PLAN_CAPTIONS.parent}
              code={buildPlanCode(collectionName, dimension).parent}
              results={result.paths.parent.results}
              metrics={result.path_recall?.parent}
              empty="No parent matched."
              render={(item) => (
                <ParentCard
                  result={item}
                  annotation={buildAnnotation(groundTruth, item)}
                />
              )}
            />
            <ResultColumn
              label="Child-only"
              tone="child"
              plan={PLAN_CAPTIONS.child}
              code={buildPlanCode(collectionName, dimension).child}
              results={result.paths.child.results}
              metrics={result.path_recall?.child}
              empty="No child matched."
              render={(item) => (
                <ChildCard
                  result={item}
                  annotation={buildAnnotation(groundTruth, item)}
                />
              )}
            />
            <ResultColumn
              label="Fused"
              tone="fusion"
              plan={PLAN_CAPTIONS.fusion}
              code={buildPlanCode(collectionName, dimension).fusion}
              results={result.paths.fusion.results}
              metrics={result.path_recall?.fusion}
              empty="No fused candidate matched."
              render={(item) => (
                <FusionCard
                  result={item}
                  annotation={buildAnnotation(groundTruth, item)}
                />
              )}
            />
          </div>
        ) : null}
      </section>
    </main>
  );
}
