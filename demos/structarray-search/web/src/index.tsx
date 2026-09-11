import { Fragment, useEffect, useRef, useState, type ReactNode } from "react";

import "./styles.css";

export type DemoStatus = "available" | "implemented" | "foundation-ready" | "planned";

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
  title: "Parent + Child Semantic Hybrid",
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

const API_BASE = "/api/v1";

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
  video_count: number;
  observation_count: number;
  evidence_frame_count: number;
  sample_sha256: string;
  prefix_sha256: string;
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

function shortVideoId(videoId: string): string {
  return videoId.length > 8 ? `${videoId.slice(0, 8)}…` : videoId;
}

/* ------------------------------------------------------------------ */
/* Small presentational pieces                                         */
/* ------------------------------------------------------------------ */

function CardVideoId({ videoId }: { videoId: string }) {
  return (
    <span className="card-vid" title={videoId}>
      {shortVideoId(videoId)}
    </span>
  );
}

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

function CardFoot({ videoId }: { videoId: string }) {
  return (
    <footer className="card-foot">
      <CardVideoId videoId={videoId} />
    </footer>
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
      <CardFoot videoId={result.video_id} />
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
      <CardFoot videoId={result.video_id} />
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
      <CardFoot videoId={result.video_id} />
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
    ],
  },
];

function SchemaTableRow({ row }: { row: ExampleVideo }) {
  const [open, setOpen] = useState(false);
  const remaining = row.observationCount - row.observations.length;
  return (
    <Fragment>
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
            type="button"
            className="nested-toggle"
            aria-expanded={open}
            onClick={() => setOpen((value) => !value)}
          >
            <span
              className={`nested-caret${open ? " is-open" : ""}`}
              aria-hidden="true"
            >
              ▸
            </span>
            <span>{row.observationCount} objects</span>
          </button>
        </td>
      </tr>
      {open ? (
        <tr className="nested-detail-row">
          <td colSpan={PARENT_COLUMNS.length}>
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
          </td>
        </tr>
      ) : null}
    </Fragment>
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
  const collection =
    status?.milvus.collection_name ?? "milvus3_demos_structarray_hybrid_covla";
  const rowCount =
    status?.milvus.row_count ?? status?.dataset.video_count ?? 30;
  const observationCount = status?.dataset.observation_count ?? 519;

  return (
    <section className="search-pipeline" aria-label="Data model">
      <header className="pipeline-head">
        <div className="pipeline-head-titles">
          <span className="pipeline-kicker">Data model</span>
          <span className="pipeline-headline">
            one video row · nested observations
          </span>
        </div>
        <span className="pipeline-meta">
          {collection} · {rowCount} rows · {observationCount} observations
        </span>
      </header>

      <div className="pipeline-body">
        <section className="pipeline-schema">
          <span className="pipeline-section-label">
            Collection schema · {collection}
          </span>
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
    status?.milvus.collection_name ?? "milvus3_demos_structarray_hybrid_covla";
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
        <a className="back-link" href="/">
          ← Home
        </a>
        <div className="hybrid-title">
          <h1>{structArrayDemo.title}</h1>
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
