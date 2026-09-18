import {
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
} from "react";

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

export const functionChainDemo: DemoDefinition = {
  id: "function-chain-rerank",
  title: "Function Chain Reranking",
  shortTitle: "Function Chain Rerank",
  capability: "XGBoost + Function Chain",
  description:
    "Semantic recall reranked by a server-side Function Chain, side by side.",
  route: "/demos/function-chain-rerank",
  status: "available",
  accent: "#2563eb",
  highlights: [
    "Semantic recall vs. reranked order",
    "Five on-chain feature steps",
    "One row of intermediate values",
  ],
};

export interface DatasetIdentity {
  dataset_id: "synthetic-commerce-catalog";
  revision: "synthetic-commerce-catalog-r1";
  manifest_sha256: string;
  dataset_name: "Synthetic Commerce Catalog";
  publisher_and_data_credit: "milvus3-demos project (no external publisher)";
  license: "CC0-1.0";
  license_url: string;
  license_conflict_record: "none";
  publication_review_required: false;
  source_type: "synthetic_catalog_with_simulated_operations";
  synthetic: true;
  real_product_metadata: false;
  real_product_photos: false;
  real_transaction_data: false;
  contains_simulated_operational_signals: true;
  simulated_signal_label: "deterministic_simulated_operational_signal";
  as_of_date: string;
  product_count: number;
  product_type_count: number;
  image_count: number;
  embedding: Record<string, unknown>;
  semantic_score_preprocessing: Record<string, unknown>;
  relevance_ground_truth: Record<string, unknown>;
  field_provenance: Record<string, string>;
}

export interface QueryOption {
  id: string;
  query_text: string;
  story: string;
  split: "train" | "validation";
  dataset: DatasetIdentity;
}

export interface RankedProduct {
  id: number;
  rank: number;
  score: number;
  item_id: string;
  title: string | null;
  product_type: string;
  brand: string | null;
  color: string | null;
  material: string | null;
  style: string | null;
  node_name: string | null;
  description: string | null;
  bullet_points: string[];
  main_image_id: string;
  selected_image_id: string;
  image_role: string;
  source_object_path: string;
  source_url: string;
  display_price_usd: number;
  rating_value: number;
  clicks_30d: number;
  sales_30d: number;
  inventory_units: number;
  inventory_capacity: number;
  release_date: string;
  release_epoch: number;
  rating: number;
  inventory: number;
  return_rate: number;
  freshness: number;
  image_path: string;
  image_mime: "image/jpeg";
  image_width: number;
  image_height: number;
  image_sha256: string;
  operational_signal_provenance: "deterministic_simulated_operational_signal";
  field_provenance: Record<string, string>;
}

export interface ChainStep {
  name: string;
  operation: string;
  output: string;
  description: string;
  code: string;
}

export interface DataModelField {
  name: string;
  type: string;
  role: string;
  note: string;
}

export interface ChainTraceStep {
  name: string;
  value: number | string;
  inputs: Record<string, number | string>;
  note: string;
}

export interface SearchComparison {
  query_id: string | null;
  query_text: string;
  execution_path: "milvus_l0_xgboost_function_chain";
  model_resource_name: string;
  function_chain: {
    stage: "L0_RERANK";
    operation: "xgboost";
    feature_order: string[];
    parallel_inputs: false;
    intermediate_feature_orders: false;
    vector_order_score: "semantic_score";
    business_order_score: "business_score";
    chain_steps: ChainStep[];
  };
  dataset: DatasetIdentity;
  data_model: {
    fields: DataModelField[];
    chain_trace: {
      item_id: string;
      title: string;
      steps: ChainTraceStep[];
    };
  };
  vector_order: RankedProduct[];
  business_order: RankedProduct[];
}

type RankingStrategy = "vector_order" | "business_order";

// A fully resolved pair of rows for one selected product: its Semantic
// (vector_order) row and its Reranked (business_order) row. Both share the
// same raw fields; they differ only in rank and score.
interface CompareRow {
  itemId: string;
  vector: RankedProduct;
  business: RankedProduct;
}

const API_BASE = `${import.meta.env.BASE_URL}api/v1`;
const DERIVED_TITLE_PROVENANCE =
  "deterministic_english_summary_from_synthetic_authored_metadata";

// The DAG's source-product card is a frozen worked example: the semantic top-1
// of the default "compact dark wood desk" query (SYN-DESK-001) with its real
// catalog fields and vector-search score. It stays identical on every open no
// matter which query is searched, so the diagram always shows the same example.
const HARDCODED_SOURCE_PRODUCT: RankedProduct = {
  id: 1,
  rank: 1,
  score: 0.7042073607444763,
  item_id: "SYN-DESK-001",
  title: "Timbermark Compact Dark Walnut Writing Desk, 100 x 50 cm",
  product_type: "DESK",
  brand: "Timbermark",
  color: "Dark Walnut",
  material: "Engineered wood",
  style: "Modern",
  node_name: "Catalog > Desks",
  description:
    "A space-conscious writing desk with a dark walnut wood-look finish for a small home office.",
  bullet_points: [
    "Compact 100 x 50 cm footprint fits a small office",
    "Dark walnut wood-look finish",
    "One cable-routing cutout",
  ],
  main_image_id: "SYN-IMG-001",
  selected_image_id: "SYN-IMG-001",
  image_role: "main",
  source_object_path: "images/001-SYN-DESK-001.jpg",
  source_url: "synthetic://catalog/SYN-DESK-001/generated-product.jpg",
  display_price_usd: 155.9,
  rating_value: 5.0,
  clicks_30d: 1127,
  sales_30d: 28,
  inventory_units: 190,
  inventory_capacity: 200,
  release_date: "2026-07-15",
  release_epoch: 1784073600,
  rating: 1.0,
  inventory: 0.95,
  return_rate: 0.01,
  freshness: 0.950685,
  image_path: "images/001-SYN-DESK-001.jpg",
  image_mime: "image/jpeg",
  image_width: 256,
  image_height: 256,
  image_sha256:
    "2ab8e3c3eee3edf1fe1f8bd98a2ce3e2fd450cb778160f325585da1590a0f398",
  operational_signal_provenance: "deterministic_simulated_operational_signal",
  field_provenance: {
    brand: "synthetic_authored_metadata",
    bullet_points: "synthetic_authored_metadata",
    clicks_30d: "deterministic_simulated_operational_signal",
    color: "synthetic_authored_metadata",
    description: "synthetic_authored_metadata",
    display_price_usd: "deterministic_simulated_operational_signal",
    freshness: "derived_from_deterministic_simulated_operational_signal",
    id: "project_dataset_sequence",
    image_height: "synthetic_generated_product_image_object",
    image_mime: "synthetic_generated_product_image_object",
    image_path: "project_copy_of_hash_pinned_synthetic_generated_jpeg",
    image_role: "synthetic_generated_product_image_object",
    image_sha256: "verified_source_object_digest",
    image_width: "synthetic_generated_product_image_object",
    inventory: "derived_from_deterministic_simulated_operational_signal",
    inventory_capacity: "deterministic_simulated_operational_signal",
    inventory_units: "deterministic_simulated_operational_signal",
    item_id: "synthetic_catalog_identity",
    main_image_id: "synthetic_catalog_identity",
    material: "synthetic_authored_metadata",
    node_name: "synthetic_authored_metadata",
    product_type: "synthetic_authored_metadata",
    rating: "derived_from_deterministic_simulated_operational_signal",
    rating_value: "deterministic_simulated_operational_signal",
    release_date: "deterministic_simulated_operational_signal",
    release_epoch: "derived_from_deterministic_simulated_operational_signal",
    return_rate: "deterministic_simulated_operational_signal",
    sales_30d: "deterministic_simulated_operational_signal",
    selected_image_id: "synthetic_generated_product_image_object",
    source_object_path: "synthetic_generated_product_image_object",
    source_url: "synthetic_generated_product_image_object",
    style: "synthetic_authored_metadata",
    title: "synthetic_authored_metadata",
  },
};

// The DAG's final output chip mirrors the "business" score column below it: the
// XGBoost business score for the same frozen SYN-DESK-001 worked example (the
// rank-1 rerank result of the default "compact dark wood desk" query).
const HARDCODED_SOURCE_BUSINESS_SCORE = 0.6707878708839417;

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }
  return (await response.json()) as T;
}

export async function fetchQueryOptions(
  signal?: AbortSignal,
): Promise<QueryOption[]> {
  return readJson<QueryOption[]>(
    await fetch(`${API_BASE}/queries`, {
      headers: { Accept: "application/json" },
      signal,
    }),
  );
}

export async function fetchSearchComparison(
  queryText: string,
  queryId: string | null,
  signal?: AbortSignal,
): Promise<SearchComparison> {
  return readJson<SearchComparison>(
    await fetch(`${API_BASE}/search`, {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query_id: queryId,
        query_text: queryText,
      }),
      signal,
    }),
  );
}

export function productImageUrl(
  imagePath: string,
  imageSha256?: string,
): string {
  const encodedPath = imagePath
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
  // Content-address the asset URL with its sha256 so a replaced image at the
  // same path gets a new URL and never hides behind a stale immutable-cached
  // entry from a previous build.
  const version = imageSha256 ? `?v=${imageSha256}` : "";
  return `${API_BASE}/assets/${encodedPath}${version}`;
}

function formatCount(value: number): string {
  return value.toLocaleString("en-US");
}

function formatPercentValue(value: number, digits = 0): string {
  return `${(value * 100).toFixed(digits)}%`;
}

function formatPrice(value: number): string {
  return `$${value.toFixed(2)}`;
}

function formatSigned(value: number): string {
  if (value === 0) {
    return "0";
  }
  return value > 0 ? `+${value}` : String(value);
}

function formatText(value: string | null | undefined): string {
  return value ?? "—";
}

// Mirror the on-chain feature formulas so the expandable feature column shows
// the same inputs the XGBoost model actually receives. These are the exact
// num_combine/decay parameters from the backend's chain_features module.
function popularity(clicks_30d: number, sales_30d: number): number {
  return (0.3 / 12000) * clicks_30d + (0.7 / 850) * sales_30d;
}

function priceAffinity(display_price_usd: number): number {
  return Math.max(0.5, 1 - (0.5 / 500) * display_price_usd);
}

// Mirror the on-chain exponential recency decay. This is NOT the same value as
// the `freshness` catalog field (which is a linear 1 - age/730 shelf value);
// XGBoost is trained on this exponential decay of release_epoch, so the card
// must show the chain value to stay honest.
const FRESHNESS_ORIGIN = 1_787_184_000; // dataset as-of date (2026-08-20)
const FRESHNESS_SCALE = 180 * 86_400; // 180 days in seconds
const FRESHNESS_LAMBDA = Math.log(0.5) / FRESHNESS_SCALE;

function chainFreshness(release_epoch: number): number {
  const adjusted = Math.max(0, Math.abs(release_epoch - FRESHNESS_ORIGIN) - 0);
  return Math.exp(FRESHNESS_LAMBDA * adjusted);
}

// The chain-derived feature values for one product, mirroring the backend's
// chain_features module bit-for-bit. These are what XGBoost actually receives.
export interface ChainFeatureValues {
  popularity: number;
  price_affinity: number;
  freshness: number;
  semantic_score: number;
  rating: number;
}

export function chainFeatureValues(
  product: RankedProduct,
  semanticScore: number,
): ChainFeatureValues {
  return {
    popularity: popularity(product.clicks_30d, product.sales_30d),
    price_affinity: priceAffinity(product.display_price_usd),
    freshness: chainFreshness(product.release_epoch),
    // The semantic similarity is the vector-search cosine score, not the
    // XGBoost business score. A business_order product's `score` is the
    // business score, so the caller passes the matching vector_order score in
    // explicitly rather than re-typing the wrong field.
    semantic_score: semanticScore,
    rating: product.rating,
  };
}

function ProductPhoto({ product }: { product: RankedProduct }) {
  const [state, setState] = useState<"loading" | "loaded" | "error">("loading");
  return (
    <div className={`product-photo-shell product-photo-shell--${state}`}>
      {state === "loading" ? (
        <span className="photo-state" aria-live="polite">
          Loading image…
        </span>
      ) : null}
      {state === "error" ? (
        <span className="photo-state photo-state--error" role="status">
          image unavailable
        </span>
      ) : null}
      <img
        className="product-photo"
        src={productImageUrl(product.image_path, product.image_sha256)}
        width={product.image_width}
        height={product.image_height}
        alt={`Product image for ${product.title ?? product.product_type}`}
        loading="lazy"
        decoding="async"
        onLoad={() => setState("loaded")}
        onError={() => setState("error")}
      />
    </div>
  );
}

// A compact star rating from the product's original 1–5 star value, filled in
// proportion so the frontend reads like a normal store listing rather than a
// machine-feature dump.
function StarRating({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(5, value)) * 20;
  return (
    <span
      className="star-rating"
      role="img"
      aria-label={`${value.toFixed(1)} out of 5 stars`}
    >
      <span className="star-rating-bg" aria-hidden="true">
        ★★★★★
      </span>
      <span
        className="star-rating-fill"
        aria-hidden="true"
        style={{ width: `${pct}%` }}
      >
        ★★★★★
      </span>
    </span>
  );
}

function ProductRow({
  product,
  vectorProduct,
  businessProduct,
  strategy,
  compareItems,
  comparing,
  onToggleCompare,
}: {
  product: RankedProduct;
  vectorProduct: RankedProduct;
  businessProduct: RankedProduct;
  strategy: RankingStrategy;
  compareItems: string[];
  comparing: boolean;
  onToggleCompare: (itemId: string) => void;
}) {
  const rankChange = vectorProduct.rank - businessProduct.rank;
  const accessibleName = product.title ?? product.product_type;
  const titleIsDerived =
    product.field_provenance.title === DERIVED_TITLE_PROVENANCE;
  const isVector = strategy === "vector_order";
  // The score shown depends on which column this card lives in: the semantic
  // similarity on the left, the XGBoost business score on the right.
  const scoreLabel = isVector ? "semantic" : "business";
  const scoreValue = product.score;
  const compareSelected = compareItems.includes(product.item_id);
  const compareDisabled = compareItems.length >= 2 && !compareSelected;
  const compareOrder = compareItems.indexOf(product.item_id) + 1;
  return (
    <article
      className={`product-card product-card--${
        rankChange > 0 ? "up" : rankChange < 0 ? "down" : "same"
      } product-card--${strategy === "vector_order" ? "left" : "right"}${
        compareSelected ? " product-card--compare-selected" : ""
      }${comparing ? " product-card--comparing" : ""}`}
      data-testid="product-card"
      data-item-id={product.item_id}
      data-title={product.title ?? ""}
      data-title-provenance={product.field_provenance.title}
      data-image-path={product.image_path}
      data-display-price={product.display_price_usd}
      data-current-rank={product.rank}
      data-ranking-strategy={strategy}
      data-vector-rank={vectorProduct.rank}
      data-business-rank={businessProduct.rank}
      data-rank-delta={rankChange}
      data-business-score={businessProduct.score}
      data-score-label={scoreLabel}
      data-score={scoreValue}
      data-compare-selected={compareSelected}
    >
      {comparing ? (
        <button
          type="button"
          className={`compare-check${
            compareSelected ? " compare-check--selected" : ""
          }`}
          role="checkbox"
          aria-checked={compareSelected}
          aria-label={`Compare ${accessibleName}`}
          data-item-id={product.item_id}
          data-testid="compare-toggle"
          disabled={compareDisabled}
          onClick={() => onToggleCompare(product.item_id)}
        >
          <span className="compare-check-box" aria-hidden="true">
            {compareSelected ? "✓" : ""}
          </span>
        </button>
      ) : null}
      <span className="card-rank" aria-hidden="true">
        <strong className="card-rank-num">#{product.rank}</strong>
        <span
          className={`card-rank-delta card-rank-delta--${
            rankChange > 0 ? "up" : rankChange < 0 ? "down" : "same"
          }`}
        >
          {rankChange > 0
            ? `▲${rankChange}`
            : rankChange < 0
              ? `▼${-rankChange}`
              : "—"}
        </span>
      </span>
      <span className="card-photo">
        <ProductPhoto product={product} />
      </span>
      <span className="card-copy">
        <span className="card-title-row">
          <span className="card-title">{accessibleName}</span>
          {titleIsDerived ? (
            <span className="derived-title-note">summary</span>
          ) : null}
        </span>
        <span className="card-features">
          <span className="feature">
            <small>price</small>
            <strong>{formatPrice(product.display_price_usd)}</strong>
          </span>
          <span className="feature">
            <small>rating</small>
            <StarRating value={product.rating_value} />
          </span>
          <span className="feature">
            <small>listed</small>
            <strong>{product.release_date}</strong>
          </span>
          <span className="feature">
            <small>type</small>
            <strong>{product.product_type}</strong>
          </span>
        </span>
      </span>
      <span className="card-metrics">
        <span className={`card-score card-score--${scoreLabel}`}>
          <small>{scoreLabel}</small>
          <strong>{scoreValue.toFixed(4)}</strong>
        </span>
        {comparing ? (
          <span className="compare-slot" aria-hidden="true">
            {compareOrder > 0 ? compareOrder : ""}
          </span>
        ) : null}
      </span>
    </article>
  );
}

const CHAIN_STEP_PLACEHOLDERS = [
  { name: "popularity", label: "num_combine clicks + sales" },
  { name: "price_affinity", label: "decay price → affinity" },
  { name: "freshness", label: "decay release epoch" },
  { name: "xgboost_rerank", label: "XGBoost → business $score" },
] as const;

// The deterministic dataflow of the Function Chain: which raw schema fields
// feed each step, and what intermediate feature each step emits. This is the
// bridge between the schema table and the executable code.
const STEP_FLOW: Record<
  string,
  { inputs: string[]; output: string; upstreamFeature?: string }
> = {
  popularity: { inputs: ["clicks_30d", "sales_30d"], output: "popularity" },
  price_affinity: { inputs: ["display_price_usd"], output: "price_affinity" },
  freshness: { inputs: ["release_epoch"], output: "freshness" },
  xgboost_rerank: {
    // The vector-search similarity ($score) is rounded into
    // normalized_semantic_score before XGBoost consumes it, so the model reads
    // `semantic_score` and writes a new `$score`. Listing the input as $score
    // would make the node look like a self-loop.
    inputs: [
      "semantic_score",
      "rating",
      "popularity",
      "price_affinity",
      "freshness",
    ],
    output: "$score",
  },
};

// Map each raw schema field (and the two direct model inputs) to the step that
// consumes it — used for hovering a field row and highlighting the consuming
// step, and vice versa. `embedding` is the vector-search ANN field that
// produces the similarity $score; `semantic_score` is that similarity after
// rounding (normalized_semantic_score), the value XGBoost actually reads.
const FIELD_TO_STEP: Record<string, string> = {
  clicks_30d: "popularity",
  sales_30d: "popularity",
  display_price_usd: "price_affinity",
  release_epoch: "freshness",
  embedding: "xgboost_rerank",
  semantic_score: "xgboost_rerank",
  rating: "xgboost_rerank",
};

type HoverTarget = string | null;

// One hue per Function Chain step, expressed in three shades so the color stays
// legible on any background: `main` for borders/dots/badges, `text` for text on
// light backgrounds (WCAG AA), and `code` for tokens on the dark code block.
const STEP_COLORS: Record<
  string,
  { main: string; text: string; code: string }
> = {
  popularity: { main: "#2563eb", text: "#1d4ed8", code: "#93c5fd" },
  price_affinity: { main: "#7c3aed", text: "#6d28d9", code: "#c4b5fd" },
  freshness: { main: "#0891b2", text: "#0e7490", code: "#67e8f9" },
  xgboost_rerank: { main: "#d97706", text: "#b45309", code: "#fcd34d" },
};

// The step a given raw field feeds, or the step that emits a given feature.
function stepForEntity(name: string): string | null {
  const fieldStep = FIELD_TO_STEP[name];
  if (fieldStep) {
    return fieldStep;
  }
  for (const [step, flow] of Object.entries(STEP_FLOW)) {
    if (flow.output === name) {
      return step;
    }
  }
  return null;
}

// The code-token color for a known field/feature (bright enough for the dark
// code background), or null if the string is not a known entity.
function entityCodeColor(name: string): string | null {
  const step = stepForEntity(name);
  return step ? (STEP_COLORS[step]?.code ?? null) : null;
}

// Lightweight Python tokenizer: colorizes strings, comments, numbers, and the
// Function Chain builder verbs (fn.*, col(...), named parameters). A string
// that names a known field/feature is tinted with that entity's accent color,
// so the code visually carries the same data-flow hue as the diagram.
const PYTHON_TOKEN =
  /("[^"]*"|'[^']*'|#[^\n]*|\b(?:fn|col)\b|\b\d+(?:\.\d+)?\b|\b(?:mode|weights|function|origin|scale|offset|decay|decimal|model_resource|output)\b)/g;

function pythonHighlight(code: string) {
  const nodes: React.ReactNode[] = [];
  let lastIndex = 0;
  for (const match of code.matchAll(PYTHON_TOKEN)) {
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
    let className: string | undefined = "py-plain";
    if (first === '"' || first === "'") {
      const inner = token.slice(1, -1);
      const color = entityCodeColor(inner);
      className = color ? "py-entity" : "py-string";
      nodes.push(
        <span
          key={`${index}-${token}`}
          className={className}
          style={
            color
              ? ({ "--entity-code": color } as React.CSSProperties)
              : undefined
          }
        >
          {token}
        </span>,
      );
      lastIndex = index + token.length;
      continue;
    }
    if (first === "#") {
      className = "py-comment";
    } else if (/\d/.test(first)) {
      className = "py-number";
    } else if (token === "fn" || token === "col") {
      className = "py-fn";
    } else {
      className = "py-param";
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

// An intermediate feature (e.g. popularity), colored by the step that emits it.
// Outputs are data, not operations, so entering one clears any open code panel.
function FeatureChip({
  name,
  step,
  active,
  onEnter,
}: {
  name: string;
  step: string;
  active: boolean;
  onEnter: () => void;
}) {
  const palette = STEP_COLORS[step] ?? {
    main: "#334762",
    text: "#334762",
    code: "#cfe3ff",
  };
  return (
    <span
      className={`flow-feature${active ? " is-active" : ""}`}
      data-flow-feature={name}
      data-flow-step={step}
      style={
        {
          "--chip-color": palette.main,
          "--chip-text": palette.text,
        } as React.CSSProperties
      }
      onMouseEnter={onEnter}
    >
      <code>{name}</code>
    </span>
  );
}

// A right-hand visual companion for the hovered step's code: each operation
// gets a small labeled figure (weighted sum, decay curve, or an ensemble of
// trees) so the reader can see at a glance what the code computes.
const GRAPHIC_TEXT = "#334762";
const GRAPHIC_MUTED = "#5c6d82";

function GraphicLabel({
  x,
  y,
  children,
  size = 12,
  fill = GRAPHIC_TEXT,
  anchor = "middle",
  weight = 700,
}: {
  x: string | number;
  y: string | number;
  children: React.ReactNode;
  size?: number;
  fill?: string;
  anchor?: "start" | "middle" | "end";
  weight?: number;
}) {
  return (
    <text
      x={x}
      y={y}
      textAnchor={anchor}
      dominantBaseline="middle"
      fontSize={size}
      fill={fill}
      fontWeight={weight}
    >
      {children}
    </text>
  );
}

function WeightedSumGraphic({
  inputs,
  output,
}: {
  inputs: string[];
  output: string;
}) {
  const [a, b] = inputs;
  return (
    <svg
      className="flow-graphic-svg"
      viewBox="0 0 400 168"
      aria-hidden="true"
      focusable="false"
    >
      <rect
        x="8"
        y="22"
        width="128"
        height="30"
        rx="8"
        fill="#ffffff"
        stroke="#2563eb"
      />
      <GraphicLabel x="72" y="37" size={12} fill="#1d4ed8">
        {a}
      </GraphicLabel>
      <rect
        x="8"
        y="116"
        width="128"
        height="30"
        rx="8"
        fill="#ffffff"
        stroke="#2563eb"
      />
      <GraphicLabel x="72" y="131" size={12} fill="#1d4ed8">
        {b}
      </GraphicLabel>

      <line
        x1="136"
        y1="37"
        x2="200"
        y2="70"
        stroke="#93b6f4"
        strokeWidth="1.5"
      />
      <line
        x1="136"
        y1="131"
        x2="200"
        y2="98"
        stroke="#93b6f4"
        strokeWidth="1.5"
      />
      <GraphicLabel
        x="160"
        y="22"
        size={10}
        fill={GRAPHIC_MUTED}
        anchor="middle"
      >
        w₁
      </GraphicLabel>
      <GraphicLabel x="160" y="150" size={10} fill={GRAPHIC_MUTED}>
        w₂
      </GraphicLabel>

      <circle cx="240" cy="84" r="28" fill="#eaf1ff" stroke="#2563eb" />
      <GraphicLabel x="240" y="84" size={17}>
        Σ
      </GraphicLabel>
      <GraphicLabel x="240" y="103" size={9} fill={GRAPHIC_MUTED}>
        weighted
      </GraphicLabel>

      <line
        x1="268"
        y1="84"
        x2="306"
        y2="84"
        stroke="#2563eb"
        strokeWidth="1.75"
      />
      <rect x="308" y="66" width="84" height="36" rx="9" fill="#2563eb" />
      <GraphicLabel x="350" y="84" size={11} fill="#ffffff">
        {output}
      </GraphicLabel>
      <GraphicLabel x="350" y="114" size={9} fill={GRAPHIC_MUTED}>
        output
      </GraphicLabel>
    </svg>
  );
}

function DecayGraphic({
  input,
  output,
  exponential,
}: {
  input: string;
  output: string;
  exponential: boolean;
}) {
  const curve = exponential
    ? "M 64 34 C 150 40, 240 96, 378 138"
    : "M 64 34 L 378 138";
  return (
    <svg
      className="flow-graphic-svg"
      viewBox="0 0 400 168"
      aria-hidden="true"
      focusable="false"
    >
      <GraphicLabel x="24" y="30" size={11} fill={GRAPHIC_MUTED} anchor="start">
        {output}
      </GraphicLabel>
      <GraphicLabel x="376" y="154" size={11} fill={GRAPHIC_MUTED} anchor="end">
        {input}
      </GraphicLabel>

      <line
        x1="64"
        y1="22"
        x2="64"
        y2="140"
        stroke="#b9c5d4"
        strokeWidth="1.5"
      />
      <line
        x1="64"
        y1="140"
        x2="380"
        y2="140"
        stroke="#b9c5d4"
        strokeWidth="1.5"
      />
      <path d={curve} fill="none" stroke="#0891b2" strokeWidth="2.25" />
      <circle cx="64" cy="34" r="3.5" fill="#0891b2" />
      <circle cx="378" cy="138" r="3.5" fill="#0891b2" />
      <GraphicLabel x="378" y="106" size={10} fill={GRAPHIC_MUTED} anchor="end">
        {exponential ? "exponential decay" : "linear decay"}
      </GraphicLabel>
    </svg>
  );
}

// The reranker is a forest of 128 depth-2 boosted trees. To show "many trees"
// without drawing all of them, three schematic trees are kept upright and
// staggered up-and-to-the-right like a rising hand of cards: tree 2 sits above
// and behind tree 1, tree 3 above and behind tree 2, each partially occluding
// the one before it. Nodes use short feature tokens (the full names already sit
// in the panel header's "in:" line), leaves carry only a +/- sign, and the Σ
// node sums all 128 trees' signed leaf scores into the final $score.
const XGB_TREES = [
  { root: "sim", left: "sim", right: "sim", leaves: ["-", "-", "+", "+"] },
  { root: "pop", left: "sim", right: "price", leaves: ["-", "-", "+", "+"] },
  { root: "rating", left: "pop", right: "pop", leaves: ["-", "+", "+", "+"] },
] as const;

// Staggered placement (upright, no rotation). Each tree's local root sits at
// (x, y) and grows downward; trees 2 and 3 are shifted up-right by (32, -28)
// and (64, -56) so they overlap the previous tree from behind and above while
// keeping their node labels legible.
const XGB_TREE_PLACE = [
  { x: 66, y: 104 },
  { x: 98, y: 76 },
  { x: 130, y: 48 },
] as const;
const XGB_TREE_OPACITY = [1, 0.62, 0.34] as const;

function XgbTree({
  t,
  opacity,
}: {
  t: (typeof XGB_TREES)[number];
  opacity: number;
}) {
  const splitNode = {
    fill: "#ffffff",
    stroke: "#d97706",
    strokeWidth: 1.2,
    rx: 4,
  } as const;
  const leafPos = {
    fill: "#fff7ed",
    stroke: "#d97706",
    strokeWidth: 1.1,
    rx: 6,
  } as const;
  const leafNeg = {
    fill: "#fef2f2",
    stroke: "#d97706",
    strokeWidth: 1.1,
    rx: 6,
  } as const;
  const branch = { stroke: "#d97706", strokeWidth: 1.1 } as const;
  const leafCenters = [-22, -6, 6, 22] as const;
  return (
    <g opacity={opacity}>
      {/* Root. */}
      <rect x="-22" y="-13" width="44" height="15" {...splitNode} />
      <GraphicLabel x={0} y={-5.5} size={6.5} fill={GRAPHIC_TEXT} weight={700}>
        {t.root}
      </GraphicLabel>
      <line x1={0} y1={2} x2={-14} y2={18} {...branch} />
      <line x1={0} y1={2} x2={14} y2={18} {...branch} />

      {/* Internal split nodes. */}
      <rect x="-29" y="18" width="30" height="13" {...splitNode} />
      <GraphicLabel x={-14} y={24.5} size={6} fill={GRAPHIC_TEXT} weight={700}>
        {t.left}
      </GraphicLabel>
      <rect x="-1" y="18" width="30" height="13" {...splitNode} />
      <GraphicLabel x={14} y={24.5} size={6} fill={GRAPHIC_TEXT} weight={700}>
        {t.right}
      </GraphicLabel>

      {/* Internals → leaves. */}
      <line x1={-14} y1={31} x2={-22} y2={46} {...branch} />
      <line x1={-14} y1={31} x2={-6} y2={46} {...branch} />
      <line x1={14} y1={31} x2={6} y2={46} {...branch} />
      <line x1={14} y1={31} x2={22} y2={46} {...branch} />

      {/* Leaf nodes carrying only a +/- sign. */}
      {leafCenters.map((cx, li) => {
        const positive = t.leaves[li] === "+";
        return (
          <g key={cx}>
            <rect
              x={cx - 9}
              y="46"
              width="18"
              height="13"
              {...(positive ? leafPos : leafNeg)}
            />
            <GraphicLabel
              x={cx}
              y={52.5}
              size={7}
              fill={positive ? "#b45309" : "#b91c1c"}
              weight={800}
            >
              {t.leaves[li]}
            </GraphicLabel>
          </g>
        );
      })}
    </g>
  );
}

function XgboostGraphic({
  inputs,
  output,
}: {
  inputs: string[];
  output: string;
}) {
  return (
    <svg
      className="flow-graphic-svg"
      viewBox="0 0 210 170"
      aria-hidden="true"
      focusable="false"
    >
      <GraphicLabel
        x="4"
        y="9"
        size={7.5}
        fill={GRAPHIC_MUTED}
        anchor="start"
        weight={700}
      >
        128 boosted trees · depth 2
      </GraphicLabel>

      {/* Draw back-to-front so the front tree occludes the ones behind it. */}
      {[2, 1, 0].map((i) => {
        const p = XGB_TREE_PLACE[i];
        return (
          <g key={i} transform={`translate(${p.x} ${p.y})`}>
            <XgbTree t={XGB_TREES[i]} opacity={XGB_TREE_OPACITY[i]} />
          </g>
        );
      })}

      {/* Ellipsis: the trees that are not drawn. */}
      <circle cx="156" cy="82" r="2.6" fill={GRAPHIC_MUTED} />
      <circle cx="156" cy="94" r="2.6" fill={GRAPHIC_MUTED} />
      <circle cx="156" cy="106" r="2.6" fill={GRAPHIC_MUTED} />

      {/* Σ sums all 128 trees' leaf scores. */}
      <line
        x1="156"
        y1="114"
        x2="172"
        y2="106"
        stroke="#f0c982"
        strokeWidth={1.2}
      />
      <circle
        cx="184"
        cy="99"
        r="15"
        fill="#fef3c7"
        stroke="#d97706"
        strokeWidth={1.4}
      />
      <GraphicLabel x={184} y={99} size={14} fill="#b45309">
        Σ
      </GraphicLabel>
      <GraphicLabel x={184} y={124} size={6.5} fill={GRAPHIC_MUTED}>
        sum · 128
      </GraphicLabel>
    </svg>
  );
}

function StepGraphic({
  stepName,
  operation,
}: {
  stepName: string;
  operation: string;
}) {
  const flow = STEP_FLOW[stepName];
  if (!flow) return null;
  const figure =
    operation === "num_combine" ? (
      <WeightedSumGraphic inputs={flow.inputs} output={flow.output} />
    ) : operation === "decay" ? (
      <DecayGraphic
        input={flow.inputs[0]}
        output={flow.output}
        exponential={stepName === "freshness"}
      />
    ) : operation === "xgboost" ? (
      <XgboostGraphic inputs={flow.inputs} output={flow.output} />
    ) : null;
  return (
    <div className="flow-step-graphic">
      <header className="flow-graphic-head">
        <strong>{operation}</strong>
        <span>
          in: {flow.inputs.join(" · ")} → out: {flow.output}
        </span>
      </header>
      {figure}
    </div>
  );
}

// Fixed geometry for the single-shot DAG: six columns (source product card →
// operation steps → intermediate feature chips → xgboost → final $score). The
// card and the operation columns sit at deterministic x-coordinates; the card's
// six consumed fields are boxed in place over the real store-card markup, so
// their *vertical* centers are measured at runtime (like RankShiftLinks) rather
// than fixed here.
const DAG_VIEW_W = 1480;
const DAG_VIEW_H = 380;
const DAG_CARD_X = 14;
const DAG_CARD_W = 470;
const DAG_CARD_TOP = 10;
const DAG_CARD_H = 190;
const DAG_STEP_X = 560;
const DAG_STEP_W = 250;
const DAG_STEP_R = DAG_STEP_X + DAG_STEP_W;
// The intermediate-feature "data" column: a fixed-width pill per feature
// (popularity, price_affinity, freshness) that sits BETWEEN its producing
// operation and xgboost, so the reader sees the operation's output as a named
// datum flowing into the model — the same visual language as the final $score.
const DAG_DATA_X = 850;
const DAG_DATA_W = 150;
const DAG_DATA_R = DAG_DATA_X + DAG_DATA_W;
const DAG_XGB_X = 1040;
const DAG_XGB_W = 250;
const DAG_XGB_R = DAG_XGB_X + DAG_XGB_W;
const DAG_OUT_X = 1330;
const DAG_OUT_W = 150;
const DAG_CHIP_H = 40;
const DAG_STEP_H = 72;
const DAG_XGB_H = 200;

// Vertical centers for the three feature-producing steps, ordered TOP-TO-BOTTOM
// to match the order their source fields appear on the card (price near the
// top, release in the middle, clicks/sales at the bottom). This makes the six
// pull edges fan out monotonically and never cross. Centers are 84px apart with
// 72px-tall nodes, leaving a 12px clear gap — no vertical overlap. Each step's
// intermediate-feature chip shares the SAME center, so the operation→chip and
// chip→xgboost legs stay perfectly horizontal.
const DAG_STEP_Y: Record<string, number> = {
  price_affinity: 150,
  freshness: 234,
  popularity: 318,
};
const DAG_XGB_Y = 230;
const DAG_OUT_Y = 230;

// Inbound ports on the xgboost node. The three intermediate-feature flows enter
// the LEFT edge at the SAME heights as their feature chips (horizontal leads,
// never crossing). The two direct pulls (semantic_score, rating) arc OVER the
// feature column and land on the node's TOP edge at two well-separated x
// positions.
const DAG_XGB_IN_Y: Record<string, number> = {
  price_affinity: DAG_STEP_Y.price_affinity,
  freshness: DAG_STEP_Y.freshness,
  popularity: DAG_STEP_Y.popularity,
};
const DAG_XGB_TOP_Y = DAG_XGB_Y - DAG_XGB_H / 2;

// The card is painted BEFORE the edge layer, so a pull edge that starts at a
// field box stays visible across the card's white surface (it is NOT hidden).
// Each tie therefore begins ON its own box — the semantic score chip's right
// edge, the rating box's right edge, or a stacked left-column box's bottom
// edge — and is routed through the card's empty gutters out to the right,
// never crossing text or a neighbouring box.

// The two direct-to-xgboost pulls (semantic_score, rating) skip the
// feature-step column. Each leaves its SEND port, rises into its own
// horizontal "skyline" corridor above the step nodes, then descends onto the
// xgboost node's top edge. The two corridors are vertically separated and
// their drop points are ordered (rating drops before semantic_score) so the
// lines never cross each other, never touch a step node, and stay inside the
// viewBox.
const DAG_DIRECT_PULL: Record<string, { corridor: number; dropX: number }> = {
  // semantic_score's chip sits in the card's RIGHT metric column. Its line
  // exits the chip's right edge, rises a few px into a corridor that runs JUST
  // ABOVE the price_affinity step node's top edge (clear of the node), then
  // descends onto the xgboost top edge. rating's box sits higher in the same
  // right-hand column, so it exits its own right edge and runs a HIGHER
  // corridor. The two corridors stay vertically separated (80 vs 91) and
  // their drop points are ordered so neither vertical drop crosses the other's
  // corridor: semantic_score drops at x=1055 (left), rating at x=1070 (right).
  semantic_score: { corridor: 91, dropX: 1055 },
  rating: { corridor: 80, dropX: 1070 },
};

// The LEFT-edge port each source field plugs into on its consuming step node.
// popularity is a two-input blend, so clicks_30d and sales_30d enter at two
// separate ports. clicks_30d (which appears BELOW sales_30d on the card once
// it exits through the shared bottom funnel) enters the LOWER port and
// sales_30d the UPPER one, so the two leads never cross.
const DAG_PULL_PORT: Record<string, number> = {
  display_price_usd: DAG_STEP_Y.price_affinity,
  release_epoch: DAG_STEP_Y.freshness,
  clicks_30d: DAG_STEP_Y.popularity + 18,
  sales_30d: DAG_STEP_Y.popularity - 18,
};

function dagEdgePath(x1: number, y1: number, x2: number, y2: number): string {
  const dx = Math.max(40, (x2 - x1) * 0.5);
  return `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
}

// A short straight hop connecting an operation node's right edge to its
// intermediate-feature chip, and that chip to xgboost's left edge, at a shared
// height. Unlike dagEdgePath's cubic, this is a simple horizontal line: the
// feature chip is so close to xgboost that a long control-point sag would dip
// the line off its port.
function dagHop(x1: number, y1: number, x2: number, y2: number): string {
  return `M ${x1} ${y1} L ${x2} ${y2}`;
}

// A direct pull edge skips the feature-step column along a rounded "skyline"
// route: it rises straight up from its field box, rounds the top corner onto
// its horizontal corridor, runs rightward, rounds the drop corner, then
// descends straight down onto the xgboost node's top edge. It is drawn as a
// true elbow (straight legs joined by quadratic corner arcs) — NOT a single
// cubic — because a single cubic sags toward its midpoint: for a shallow lift
// the curve's apex can dip back into the opaque card or an adjacent channel.
// The corner radius is capped by the available rise/drop/run so a tiny lift
// never overshoots and dips back down.
//
// A source that sits on the card's right edge (semantic_score, rating) needs a
// short horizontal lead BEFORE it turns upward, so the line leaves the chip's
// right edge perpendicularly and the first visible dash is separated from the
// box's outline (a line that turns straight up while still on the box would
// ride over the ring). `lead` is the length of that horizontal run.
function dagPullOver(
  x1: number,
  y1: number,
  corridor: number,
  dropX: number,
  dropY: number,
  lead = 0,
): string {
  const leadEnd = x1 + lead;
  const radius = Math.min(
    14,
    Math.abs(corridor - y1) - 0.5,
    Math.abs(dropY - corridor) - 0.5,
    (dropX - leadEnd) / 2 - 0.5,
  );
  const r = Math.max(radius, 0.5);
  const riseUp = y1 > corridor;
  const dropDown = dropY > corridor;
  const riseEnd = corridor + (riseUp ? r : -r);
  const dropStart = corridor + (dropDown ? r : -r);
  return [
    `M ${x1} ${y1}`,
    `L ${leadEnd} ${y1}`,
    `L ${leadEnd} ${riseEnd}`,
    `Q ${leadEnd} ${corridor}, ${leadEnd + r} ${corridor}`,
    `L ${dropX - r} ${corridor}`,
    `Q ${dropX} ${corridor}, ${dropX} ${dropStart}`,
    `L ${dropX} ${dropY}`,
  ].join(" ");
}

function stepColorHex(name: string): string {
  return STEP_COLORS[name]?.main ?? "#334762";
}

// The two raw values that bypass the function column and feed XGBoost directly
// (the rounded semantic score, rating) share one neutral "source feature" hue,
// so orange stays reserved for the XGBoost node and its final $score output.
// This keeps "orange data passes straight through the model" from being
// misread.
const SOURCE_FEATURE_HEX = "#64748b";

function fieldColorHex(name: string): string {
  if (name === "semantic_score" || name === "rating") {
    return SOURCE_FEATURE_HEX;
  }
  return stepColorHex(FIELD_TO_STEP[name]);
}

// The leading "source product card" is a pixel-identical replica of a real
// store card (same grid: rank disc + photo + copy + metrics). The six fields
// the Function Chain consumes are boxed IN PLACE on the card itself — there is
// no separate raw-field column — and each box's center is measured at runtime
// to anchor its pull edge. The two operational signals that never appear on a
// real store card (clicks/sales) are appended as faint dashed "simulated" tags.
const CARD_FIELDS: Array<{ name: string; simulated?: boolean }> = [
  { name: "semantic_score" },
  { name: "display_price_usd" },
  { name: "rating" },
  { name: "release_epoch" },
  { name: "clicks_30d", simulated: true },
  { name: "sales_30d", simulated: true },
];

function fieldBoxStyle(name: string): React.CSSProperties {
  return {
    "--chip-color": fieldColorHex(name),
  } as React.CSSProperties;
}

function FlowProductCard({
  product,
  onEnterField,
}: {
  product: RankedProduct | undefined;
  onEnterField: () => void;
}) {
  if (!product) {
    return null;
  }
  const accessibleName = product.title ?? product.product_type;
  return (
    <div
      className="product-card flow-source-card"
      data-flow-card
      data-item-id={product.item_id}
    >
      <span className="card-rank" aria-hidden="true">
        <strong className="card-rank-num">#{product.rank}</strong>
      </span>
      <span className="card-photo">
        <ProductPhoto product={product} />
      </span>
      <span className="card-copy">
        <span className="card-title-row">
          <span className="card-title">{accessibleName}</span>
        </span>
        <span className="card-features">
          <span
            className="flow-field-box"
            data-card-field="display_price_usd"
            style={fieldBoxStyle("display_price_usd")}
            onMouseEnter={onEnterField}
          >
            <span className="feature">
              <small>price</small>
              <strong>{formatPrice(product.display_price_usd)}</strong>
            </span>
          </span>
          <span
            className="flow-field-box"
            data-card-field="rating"
            style={fieldBoxStyle("rating")}
            onMouseEnter={onEnterField}
          >
            <span className="feature">
              <small>rating</small>
              <StarRating value={product.rating_value} />
            </span>
          </span>
          <span
            className="flow-field-box"
            data-card-field="release_epoch"
            style={fieldBoxStyle("release_epoch")}
            onMouseEnter={onEnterField}
          >
            <span className="feature">
              <small>listed</small>
              <strong>{product.release_date}</strong>
            </span>
          </span>
          <span className="feature">
            <small>type</small>
            <strong>{product.product_type}</strong>
          </span>
          <span
            className="flow-field-box flow-field-box--simulated"
            data-card-field="clicks_30d"
            style={fieldBoxStyle("clicks_30d")}
            onMouseEnter={onEnterField}
          >
            <span className="feature">
              <small>clicks_30d</small>
              <strong>{formatCount(product.clicks_30d)}</strong>
            </span>
          </span>
          <span
            className="flow-field-box flow-field-box--simulated"
            data-card-field="sales_30d"
            style={fieldBoxStyle("sales_30d")}
            onMouseEnter={onEnterField}
          >
            <span className="feature">
              <small>sales_30d</small>
              <strong>{formatCount(product.sales_30d)}</strong>
            </span>
          </span>
        </span>
      </span>
      <span className="card-metrics">
        <span
          className="card-score card-score--semantic flow-field-chip"
          data-card-field="semantic_score"
          onMouseEnter={onEnterField}
        >
          <small>semantic</small>
          <strong>{product.score.toFixed(4)}</strong>
        </span>
      </span>
    </div>
  );
}

// A measured rectangle in viewBox units: the outer edges plus the center. Each
// field box (and the semantic score chip) is measured at runtime and stored as
// one of these so the pull edges can attach to a box's own edge and steer
// through the card's empty gutters without crossing text or a sibling box.
interface BoxRect {
  l: number;
  r: number;
  t: number;
  b: number;
  cx: number;
  cy: number;
}

interface FlowAnchors {
  boxes: Map<string, BoxRect>;
  semantic: BoxRect | null;
}

// An orthogonal "staircase" route built from explicit waypoints, with rounded
// corners. Unlike dagPullOver (a fixed rise/corridor/drop), this walks any list
// of turns so a field box can exit its BOTTOM edge, run along a horizontal gap
// channel, descend a vertical channel, then run into its step port. Each
// interior corner is rounded with a quadratic arc whose radius is capped by
// the two adjacent leg lengths, so a short lead-in (a box bottom only a few px
// above its gap channel) never overshoots past the waypoint.
function dagStair(points: Array<{ x: number; y: number }>, radius = 8): string {
  if (points.length < 2) {
    return "";
  }
  let d = `M ${points[0].x} ${points[0].y}`;
  for (let i = 1; i < points.length; i++) {
    const prev = points[i - 1];
    const cur = points[i];
    const next = points[i + 1];
    if (!next) {
      d += ` L ${cur.x} ${cur.y}`;
      continue;
    }
    const inDx = cur.x - prev.x;
    const inDy = cur.y - prev.y;
    const outDx = next.x - cur.x;
    const outDy = next.y - cur.y;
    const inLen = Math.hypot(inDx, inDy);
    const outLen = Math.hypot(outDx, outDy);
    if (inLen < 1e-6 || outLen < 1e-6) {
      d += ` L ${cur.x} ${cur.y}`;
      continue;
    }
    const cap = Math.min(radius, inLen / 2 - 0.5, outLen / 2 - 0.5);
    const cc = Math.max(cap, 0.5);
    const inUx = inDx / inLen;
    const inUy = inDy / inLen;
    const outUx = outDx / outLen;
    const outUy = outDy / outLen;
    d += ` L ${cur.x - inUx * cc} ${cur.y - inUy * cc}`;
    d += ` Q ${cur.x} ${cur.y}, ${cur.x + outUx * cc} ${cur.y + outUy * cc}`;
  }
  return d;
}

// The DAG's edges, colored by the step each one belongs to (a pull edge takes
// its consuming step's hue; a feature edge keeps its producing step's hue).
// Six dashed "pull" ties run from the card's boxed fields into the step that
// consumes them; four solid flow edges chain the features through XGBoost to
// the final $score. Each pull edge now attaches to its OWN field box — not to
// the card's outer border — because the card is painted first and the edge
// layer sits on top, so the lead stays visible across the white card. The two
// direct-to-xgboost pulls (semantic_score, rating) arc over the feature column
// from the semantic chip's right edge and the rating box's right edge; the four
// feature-column pulls exit their box's bottom edge and staircase through the
// row gaps out to their step node.
function buildDagEdges(anchors: FlowAnchors): Array<{
  id: string;
  d: string;
  color: string;
  kind: "flow" | "pull";
  direct: boolean;
  landing?: string;
}> {
  const edges: Array<{
    id: string;
    d: string;
    color: string;
    kind: "flow" | "pull";
    direct: boolean;
    landing?: string;
  }> = [];
  const boxes = anchors.boxes;

  // Routing geometry shared by the feature-column pulls, derived from the
  // measured boxes. The horizontal gap channels run between the box rows; the
  // vertical channels descend in the gutter right of the rating/sales column
  // (rating's right edge) and left of the semantic score chip, where no text
  // sits. Falls back to the live-validated constants if a box is unmeasured.
  const ratingBox = boxes.get("rating");
  const releaseBox = boxes.get("release_epoch");
  const clicksBox = boxes.get("clicks_30d");
  const priceChanX = anchors.semantic ? anchors.semantic.l - 3.5 : 368;
  const releaseChanX = ratingBox ? ratingBox.r + 7.5 : 361;
  const gap1Y =
    ratingBox && releaseBox ? (ratingBox.b + releaseBox.t) / 2 : 116.4;
  const gap2Y =
    releaseBox && clicksBox ? (releaseBox.b + clicksBox.t) / 2 : 142.4;

  for (const field of CARD_FIELDS) {
    const box = boxes.get(field.name);
    if (!box) {
      continue;
    }
    const step = FIELD_TO_STEP[field.name];
    const direct = DAG_DIRECT_PULL[field.name];
    const portY = DAG_PULL_PORT[field.name] ?? DAG_STEP_Y[step];

    // semantic_score's chip sits in the card's right metric column: the solid
    // source dot sits just right of the box's outset ring (at box.r + 7), so the
    // line starts there, runs a short horizontal lead, then rises into its
    // corridor and runs over the feature column. rating's box sits higher in the
    // same right-hand column: exit its right edge and run a higher corridor.
    // Both land on the xgboost node's top edge.
    let d: string;
    if (field.name === "semantic_score") {
      d = dagPullOver(
        box.r + 7,
        box.cy,
        direct!.corridor,
        direct!.dropX,
        DAG_XGB_TOP_Y,
        14,
      );
    } else if (field.name === "rating") {
      d = dagPullOver(
        box.r,
        box.cy,
        direct!.corridor,
        direct!.dropX,
        DAG_XGB_TOP_Y,
        6,
      );
    } else if (field.name === "display_price_usd") {
      d = dagStair([
        { x: box.cx, y: box.b },
        { x: box.cx, y: gap1Y },
        { x: priceChanX, y: gap1Y },
        { x: priceChanX, y: portY },
        { x: DAG_STEP_X, y: portY },
      ]);
    } else if (field.name === "release_epoch") {
      d = dagStair([
        { x: box.cx, y: box.b },
        { x: box.cx, y: gap2Y },
        { x: releaseChanX, y: gap2Y },
        { x: releaseChanX, y: portY },
        { x: DAG_STEP_X, y: portY },
      ]);
    } else {
      // clicks_30d / sales_30d: exit bottom-center, drop straight down to their
      // popularity port, then run right into the step node.
      d = dagStair([
        { x: box.cx, y: box.b },
        { x: box.cx, y: portY },
        { x: DAG_STEP_X, y: portY },
      ]);
    }
    edges.push({
      id: `card->${field.name}`,
      d,
      color: fieldColorHex(field.name),
      kind: "pull",
      direct: direct != null,
      // Every pull edge lands on a port via a short solid tail so the dash
      // phase never opens a white gap at the connector. Direct pulls drop
      // straight down onto the xgboost top edge; the feature-column pulls run
      // horizontally into their step node's left edge.
      landing:
        direct != null
          ? dagHop(direct.dropX, DAG_XGB_TOP_Y - 6, direct.dropX, DAG_XGB_TOP_Y)
          : dagHop(DAG_STEP_X - 6, portY, DAG_STEP_X, portY),
    });
  }

  // The derived-feature flow: each operation node feeds a named intermediate
  // feature chip (a data node, same language as $score), and that chip feeds
  // xgboost. Two short horizontal hops per feature, all at the feature's own
  // height, so "price_affinity / freshness / popularity" read as first-class
  // data entering the model rather than an arrow label on an operation node.
  for (const step of ["popularity", "price_affinity", "freshness"]) {
    edges.push({
      id: `step:${step}->feature:${STEP_FLOW[step].output}`,
      d: dagHop(DAG_STEP_R, DAG_STEP_Y[step], DAG_DATA_X, DAG_STEP_Y[step]),
      color: stepColorHex(step),
      kind: "flow",
      direct: false,
    });
    edges.push({
      id: `feature:${STEP_FLOW[step].output}->xgboost_rerank`,
      d: dagHop(DAG_DATA_R, DAG_STEP_Y[step], DAG_XGB_X, DAG_XGB_IN_Y[step]),
      color: stepColorHex(step),
      kind: "flow",
      direct: false,
    });
  }
  edges.push({
    id: "xgboost_rerank->$score",
    d: dagEdgePath(DAG_XGB_R, DAG_XGB_Y, DAG_OUT_X, DAG_OUT_Y),
    color: stepColorHex("xgboost_rerank"),
    kind: "flow",
    direct: false,
  });
  return edges;
}

function FunctionChainFlow({ comparison }: { comparison: SearchComparison }) {
  const [hover, setHover] = useState<HoverTarget>(null);
  const [anchors, setAnchors] = useState<FlowAnchors | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const flowRef = useRef<HTMLDivElement | null>(null);
  const panelRef = useRef<HTMLDivElement | null>(null);
  const hoverNodeRef = useRef<HTMLElement | null>(null);
  const [panelPos, setPanelPos] = useState<{
    top: number;
    left: number;
  } | null>(null);
  const steps = comparison.function_chain.chain_steps;

  // The six in-place field boxes live inside a <foreignObject>, so their
  // positions are measured at runtime (like RankShiftLinks) and scaled from DOM
  // px into viewBox units. The card top is DAG_CARD_TOP in viewBox space; a
  // box's px offset is scaled by DAG_VIEW_W / renderedWidth. Each box stores
  // its full rect (left/right/top/bottom/center) in viewBox units so each pull
  // edge can attach to the box's own edge and steer through the card's gutters.
  // The semantic score chip is measured too: it is the right-hand obstacle the
  // price/release channels must thread past.
  useLayoutEffect(() => {
    function measure() {
      const svg = svgRef.current;
      if (!svg) {
        setAnchors(null);
        return;
      }
      const card =
        svg.ownerDocument.querySelector<HTMLElement>("[data-flow-card]");
      if (!card) {
        setAnchors(null);
        return;
      }
      const rect = svg.getBoundingClientRect();
      const scale = rect.width > 0 ? DAG_VIEW_W / rect.width : 1;
      const cardLeft = card.getBoundingClientRect().left;
      const cardTop = card.getBoundingClientRect().top;
      const boxes = new Map<string, BoxRect>();
      for (const field of CARD_FIELDS) {
        const box = card.querySelector<HTMLElement>(
          `[data-card-field="${field.name}"]`,
        );
        if (!box) {
          continue;
        }
        const boxRect = box.getBoundingClientRect();
        const toVb = (px: number, axis: "x" | "y") =>
          (axis === "x" ? DAG_CARD_X : DAG_CARD_TOP) + px * scale;
        const l = toVb(boxRect.left - cardLeft, "x");
        const t = toVb(boxRect.top - cardTop, "y");
        const r = toVb(boxRect.right - cardLeft, "x");
        const b = toVb(boxRect.bottom - cardTop, "y");
        boxes.set(field.name, {
          l,
          r,
          t,
          b,
          cx: (l + r) / 2,
          cy: (t + b) / 2,
        });
      }
      let semantic: BoxRect | null = null;
      const chip = card.querySelector<HTMLElement>(".card-score--semantic");
      if (chip) {
        const chipRect = chip.getBoundingClientRect();
        const toVb = (px: number, axis: "x" | "y") =>
          (axis === "x" ? DAG_CARD_X : DAG_CARD_TOP) + px * scale;
        const l = toVb(chipRect.left - cardLeft, "x");
        const t = toVb(chipRect.top - cardTop, "y");
        const r = toVb(chipRect.right - cardLeft, "x");
        const b = toVb(chipRect.bottom - cardTop, "y");
        semantic = { l, r, t, b, cx: (l + r) / 2, cy: (t + b) / 2 };
      }
      setAnchors({ boxes, semantic });
    }
    measure();
    window.addEventListener("resize", measure);
    let observer: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined") {
      observer = new ResizeObserver(measure);
      observer.observe(svgRef.current!);
    }
    return () => {
      window.removeEventListener("resize", measure);
      observer?.disconnect();
    };
  }, [comparison]);

  // Ordered steps in execution order.
  const ordered = useMemo(
    () =>
      CHAIN_STEP_PLACEHOLDERS.map((placeholder) => {
        const step = steps?.find(
          (candidate) => candidate.name === placeholder.name,
        );
        const flow = STEP_FLOW[placeholder.name];
        return { placeholder, step, flow };
      }).filter((entry) => entry.step),
    [steps],
  );

  const featureSteps = useMemo(
    () => ordered.filter((entry) => entry.step!.name !== "xgboost_rerank"),
    [ordered],
  );
  const rerankStep = useMemo(
    () => ordered.find((entry) => entry.step!.name === "xgboost_rerank"),
    [ordered],
  );

  // The product card that sources the DAG's raw fields. The demo example is
  // frozen to a single catalog item regardless of the query, so the diagram
  // always shows the same product and numbers on every open.
  const sourceProduct = HARDCODED_SOURCE_PRODUCT;

  // Only operation step nodes reveal the code+graphic overlay; raw input and
  // output chips are data, not operations, so they do not open a panel.
  const activeStep = hover;

  function isStepActive(name: string): boolean {
    if (!activeStep) return false;
    if (name === activeStep) return true;
    const flow = STEP_FLOW[activeStep];
    if (flow?.output && STEP_FLOW.xgboost_rerank.inputs.includes(flow.output)) {
      return name === "xgboost_rerank";
    }
    return false;
  }

  const activeCode =
    activeStep === null
      ? null
      : (ordered.find((entry) => entry.step!.name === activeStep)?.step ??
        null);
  const palette = activeCode
    ? (STEP_COLORS[activeCode.name] ?? {
        main: "#334762",
        text: "#334762",
        code: "#cfe3ff",
      })
    : null;

  const edges = useMemo(
    () => buildDagEdges(anchors ?? { boxes: new Map(), semantic: null }),
    [anchors],
  );

  // Position the code panel after it has rendered at its natural width, so a
  // wide snippet is never shrunk by an earlier (narrower) measurement. The
  // panel sits just below the hovered node and may extend past the DAG's right
  // edge; when the node sits near the right edge we shift the panel LEFT rather
  // than shrink it, so the code stays fully readable.
  useLayoutEffect(() => {
    const node = hoverNodeRef.current;
    const flow = flowRef.current;
    const panel = panelRef.current;
    if (!node || !flow || !panel) return;
    // Clear the previous inline position first so the panel lays out at its
    // natural width for THIS step's content; measuring with a stale left/top
    // would report a width already squeezed by the previous node's offset.
    panel.style.left = "";
    panel.style.top = "";
    const flowRect = flow.getBoundingClientRect();
    const nodeRect = node.getBoundingClientRect();
    const top = nodeRect.bottom - flowRect.top + 8;
    const panelW = panel.offsetWidth;
    const flowW = flowRect.width;
    const idealLeft = nodeRect.left - flowRect.left;
    // Never shrink the panel: align to the node when there is room, otherwise
    // nudge it left until its right edge just touches the flow edge.
    const maxLeft = Math.max(0, flowW - panelW - 8);
    const left = Math.min(Math.max(8, idealLeft), maxLeft);
    setPanelPos({ top, left });
  }, [activeCode?.name]);

  function enterStep(node: HTMLElement, name: string) {
    hoverNodeRef.current = node;
    setHover(name);
  }

  function renderStepNode(step: ChainStep, x: number, y: number, h: number) {
    const stepActive = isStepActive(step.name);
    const isXgb = step.name === "xgboost_rerank";
    const xgbInputs = isXgb ? STEP_FLOW.xgboost_rerank.inputs : [];
    return (
      <foreignObject
        key={step.name}
        x={x}
        y={y - h / 2}
        width={DAG_STEP_W}
        height={h}
      >
        <div className="dag-node-host dag-node-host--step">
          <div
            className={`flow-step-node${stepActive ? " is-active" : ""}`}
            data-flow-step-node={step.name}
            onMouseEnter={(event) => enterStep(event.currentTarget, step.name)}
            onMouseLeave={() => setHover(null)}
          >
            <span className="flow-step-op">{step.operation}</span>
            <span className="flow-step-desc">{step.description}</span>
            {isXgb ? (
              <span className="flow-step-inputs">
                <span className="flow-step-inputs-title">inputs</span>
                {xgbInputs.map((name) => (
                  <span className="flow-step-input" key={name}>
                    <code>{name}</code>
                  </span>
                ))}
              </span>
            ) : null}
          </div>
        </div>
      </foreignObject>
    );
  }

  return (
    <div
      className="function-chain-flow"
      data-testid="function-chain-flow"
      ref={flowRef}
      onMouseLeave={() => setHover(null)}
    >
      <div className="flow-dag">
        <div className="flow-dag-head">
          <span className="flow-dag-title">Function chain rerank demo</span>
        </div>
        <svg
          className="flow-dag-svg"
          viewBox={`0 0 ${DAG_VIEW_W} ${DAG_VIEW_H}`}
          ref={svgRef}
        >
          <defs>
            <marker
              id="dag-arrow"
              viewBox="0 0 10 10"
              refX="10"
              refY="5"
              markerWidth="8"
              markerHeight="8"
              markerUnits="userSpaceOnUse"
              orient="auto"
            >
              <path d="M 0 0 L 10 5 L 0 10 Z" fill="context-stroke" />
            </marker>
          </defs>

          {/* Column 0: the source product card, a pixel-identical replica of
              the search cards below. It is painted FIRST (under the edge
              layer) so the pull edges that start on the card itself — the two
              direct pulls from the right-border SEND ports (semantic_score,
              rating) and the four feature-column leads — stay visible ON the
              card's white surface instead of being hidden behind it. The edges
              are still routed around the card's content, exiting at the card's
              right border. */}
          {sourceProduct ? (
            <foreignObject
              x={DAG_CARD_X}
              y={DAG_CARD_TOP}
              width={DAG_CARD_W}
              height={DAG_CARD_H}
            >
              <FlowProductCard
                product={sourceProduct}
                onEnterField={() => setHover(null)}
              />
            </foreignObject>
          ) : null}

          <g className="flow-dag-edges" aria-hidden="true">
            {edges.map((edge) => (
              <path
                key={edge.id}
                d={edge.d}
                className={`flow-dag-edge${
                  edge.kind === "pull"
                    ? edge.direct
                      ? " flow-dag-edge--pull flow-dag-edge--direct"
                      : " flow-dag-edge--pull"
                    : ""
                }`}
                data-edge={edge.id}
                style={{ stroke: edge.color }}
              />
            ))}
            {edges
              .filter((edge) => edge.landing)
              .map((edge) => (
                <path
                  key={`${edge.id}->landing`}
                  d={edge.landing!}
                  className="flow-dag-landing"
                  style={{ stroke: edge.color }}
                />
              ))}
          </g>

          {/* Column 2: the three feature-producing steps. */}
          {featureSteps.map(({ step }) =>
            renderStepNode(
              step!,
              DAG_STEP_X,
              DAG_STEP_Y[step!.name],
              DAG_STEP_H,
            ),
          )}

          {/* Column 2.5: the intermediate-feature data chips — each operation
              node's output is a named datum (popularity / price_affinity /
              freshness) that flows on into xgboost, mirroring the final $score
              chip. They sit at the same height as their producing operation. */}
          {["price_affinity", "freshness", "popularity"].map((step) => (
            <foreignObject
              key={`data-${step}`}
              x={DAG_DATA_X}
              y={DAG_STEP_Y[step] - DAG_CHIP_H / 2}
              width={DAG_DATA_W}
              height={DAG_CHIP_H}
            >
              <div className="dag-node-host dag-node-host--data">
                <FeatureChip
                  name={STEP_FLOW[step].output}
                  step={step}
                  active={isStepActive(step)}
                  onEnter={() => setHover(null)}
                />
              </div>
            </foreignObject>
          ))}

          {/* Column 3: the single xgboost rerank step. */}
          {rerankStep
            ? renderStepNode(rerankStep.step!, DAG_XGB_X, DAG_XGB_Y, DAG_XGB_H)
            : null}

          {/* Column 4: the final output — the XGBoost business score, styled
              like the "business" score chip on the real rerank cards below. */}
          <foreignObject
            x={DAG_OUT_X}
            y={DAG_OUT_Y - DAG_CHIP_H / 2}
            width={DAG_OUT_W}
            height={DAG_CHIP_H}
          >
            <div className="dag-node-host dag-node-host--output">
              <span className="flow-output-score">
                <small>business</small>
                <strong>{HARDCODED_SOURCE_BUSINESS_SCORE.toFixed(4)}</strong>
              </span>
            </div>
          </foreignObject>

          {/* Port dots: painted LAST, above every node's opaque fill, so each
              connector renders as a full circle sitting on the node border
              (never a half-circle clipped by the node's white background).
              Every flow edge lands on one so lines read as "plugged in". */}
          <g className="flow-dag-ports" aria-hidden="true">
            {["popularity", "price_affinity", "freshness"].map((step) => (
              <circle
                key={`${step}-in`}
                cx={DAG_XGB_X}
                cy={DAG_XGB_IN_Y[step]}
                r={3}
                fill="#fff"
                stroke={stepColorHex(step)}
                strokeWidth={1.75}
              />
            ))}
            {["popularity", "price_affinity", "freshness"].map((step) => (
              <circle
                key={`${step}-data-in`}
                cx={DAG_DATA_X}
                cy={DAG_STEP_Y[step]}
                r={3}
                fill="#fff"
                stroke={stepColorHex(step)}
                strokeWidth={1.75}
              />
            ))}
            {["popularity", "price_affinity", "freshness"].map((step) => (
              <circle
                key={`${step}-data-out`}
                cx={DAG_DATA_R}
                cy={DAG_STEP_Y[step]}
                r={3}
                fill="#fff"
                stroke={stepColorHex(step)}
                strokeWidth={1.75}
              />
            ))}
            <circle
              cx={DAG_DIRECT_PULL.semantic_score.dropX}
              cy={DAG_XGB_TOP_Y}
              r={3}
              fill="#fff"
              stroke={SOURCE_FEATURE_HEX}
              strokeWidth={1.75}
            />
            <circle
              cx={DAG_DIRECT_PULL.rating.dropX}
              cy={DAG_XGB_TOP_Y}
              r={3}
              fill="#fff"
              stroke={SOURCE_FEATURE_HEX}
              strokeWidth={1.75}
            />
            <circle
              cx={DAG_STEP_X}
              cy={DAG_STEP_Y.price_affinity}
              r={3}
              fill="#fff"
              stroke={stepColorHex("price_affinity")}
              strokeWidth={1.75}
            />
            <circle
              cx={DAG_STEP_X}
              cy={DAG_STEP_Y.freshness}
              r={3}
              fill="#fff"
              stroke={stepColorHex("freshness")}
              strokeWidth={1.75}
            />
            <circle
              cx={DAG_STEP_X}
              cy={DAG_STEP_Y.popularity - 18}
              r={3}
              fill="#fff"
              stroke={stepColorHex("popularity")}
              strokeWidth={1.75}
            />
            <circle
              cx={DAG_STEP_X}
              cy={DAG_STEP_Y.popularity + 18}
              r={3}
              fill="#fff"
              stroke={stepColorHex("popularity")}
              strokeWidth={1.75}
            />
            <circle
              cx={DAG_XGB_R}
              cy={DAG_XGB_Y}
              r={3}
              fill="#fff"
              stroke={stepColorHex("xgboost_rerank")}
              strokeWidth={1.75}
            />
            <circle
              cx={DAG_OUT_X}
              cy={DAG_OUT_Y}
              r={3}
              fill="#fff"
              stroke={stepColorHex("xgboost_rerank")}
              strokeWidth={1.75}
            />
          </g>

          {/* The semantic score chip's box and its source port dot. The ring is
              drawn in SVG (not CSS) so it shares no pixels with the pull line,
              and it is painted AFTER every opaque layer so it is never clipped.
              The ring is outset 2px on all sides so the value's last digit keeps
              breathing room, and the SOLID source dot sits in its own clear gap
              just right of the ring — it never touches the ring or the number.
              The pull line starts on the dot's center. */}
          {anchors?.semantic ? (
            <g className="flow-dag-ports" aria-hidden="true">
              <rect
                x={anchors.semantic.l - 2}
                y={anchors.semantic.t - 2}
                width={anchors.semantic.r - anchors.semantic.l + 4}
                height={anchors.semantic.b - anchors.semantic.t + 4}
                rx={6}
                fill="none"
                stroke={SOURCE_FEATURE_HEX}
                strokeWidth={1.5}
              />
              <circle
                cx={anchors.semantic.r + 7}
                cy={anchors.semantic.cy}
                r={2.5}
                fill={SOURCE_FEATURE_HEX}
              />
            </g>
          ) : null}
        </svg>
      </div>

      {/* Single shared overlay: hovered node's executable code + a right-hand
          diagram that pictures the same step. Positioned above the page layer,
          just below the hovered node so the node itself stays visible. */}
      <div
        className={`flow-code-panel${activeCode ? "" : " flow-code-panel--empty"}`}
        data-flow-code={activeCode?.name ?? "none"}
        ref={panelRef}
        style={
          {
            ...(panelPos ? { top: panelPos.top, left: panelPos.left } : {}),
            ...(palette
              ? {
                  "--chip-color": palette.main,
                  "--chip-text": palette.text,
                }
              : {}),
          } as React.CSSProperties
        }
      >
        {activeCode ? (
          <div className="flow-code-panel-grid">
            <div className="flow-code-panel-code">
              <header className="flow-code-card-head">
                <span className="flow-code-step-dot" />
                <code className="flow-code-block-op">
                  {activeCode.operation}
                </code>
                <span className="flow-code-block-desc">
                  {activeCode.description}
                </span>
              </header>
              <pre className="flow-code-block-pre">
                <code>{pythonHighlight(activeCode.code)}</code>
              </pre>
            </div>
            <StepGraphic
              stepName={activeCode.name}
              operation={activeCode.operation}
            />
          </div>
        ) : null}
      </div>
    </div>
  );
}

function DataModelPanel({ comparison }: { comparison: SearchComparison }) {
  return (
    <section
      className="data-model-panel"
      aria-label="Data model"
      data-testid="data-model-panel"
    >
      <div className="data-model-body">
        <FunctionChainFlow comparison={comparison} />
      </div>
    </section>
  );
}

function rankLinkTone(rankChange: number): "up" | "down" | "same" {
  if (rankChange > 0) {
    return "up";
  }
  if (rankChange < 0) {
    return "down";
  }
  return "same";
}

// Draws one SVG ribbon per product, directly connecting the product card in the
// left column to the same product's card in the right column. Unchanged ranks
// draw a faint grey line; moved ranks draw a colored arc.
function RankShiftLinks({ comparison }: { comparison: SearchComparison }) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [geometry, setGeometry] = useState<{
    anchors: Array<{
      item_id: string;
      leftX: number;
      leftY: number;
      rightX: number;
      rightY: number;
      rankChange: number;
    }>;
  } | null>(null);

  useLayoutEffect(() => {
    function measure() {
      const svg = svgRef.current;
      const board = svg?.parentElement;
      if (!svg || !board) {
        setGeometry(null);
        return;
      }
      const anchors = comparison.vector_order
        .map((vectorProduct) => {
          const businessProduct = comparison.business_order.find(
            (candidate) => candidate.item_id === vectorProduct.item_id,
          );
          const leftCard = board.querySelector<HTMLElement>(
            `[data-item-id="${vectorProduct.item_id}"][data-ranking-strategy="vector_order"]`,
          );
          const rightCard = board.querySelector<HTMLElement>(
            `[data-item-id="${vectorProduct.item_id}"][data-ranking-strategy="business_order"]`,
          );
          if (!leftCard || !rightCard || !businessProduct) {
            return null;
          }
          return {
            item_id: vectorProduct.item_id,
            leftX: leftCard.offsetLeft + leftCard.offsetWidth,
            leftY: leftCard.offsetTop + leftCard.offsetHeight / 2,
            rightX: rightCard.offsetLeft,
            rightY: rightCard.offsetTop + rightCard.offsetHeight / 2,
            rankChange: vectorProduct.rank - businessProduct.rank,
          };
        })
        .filter(
          (
            anchor,
          ): anchor is {
            item_id: string;
            leftX: number;
            leftY: number;
            rightX: number;
            rightY: number;
            rankChange: number;
          } => anchor !== null,
        );

      setGeometry({ anchors });
    }
    measure();
    const board = svgRef.current?.parentElement ?? null;
    if (!board) {
      return;
    }
    window.addEventListener("resize", measure);
    let resizeObserver: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined") {
      resizeObserver = new ResizeObserver(measure);
      // The board's overall size stays fixed while the feature column opens;
      // only the three column tracks change width, so observe those too.
      resizeObserver.observe(board);
      board.querySelectorAll(".board-column").forEach((column) => {
        resizeObserver?.observe(column);
      });
    }
    return () => {
      resizeObserver?.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, [comparison]);

  return (
    <svg
      className="rank-shift-links"
      aria-hidden="true"
      ref={svgRef}
      data-testid="rank-shift-links"
    >
      <defs>
        <marker
          id="rank-arrow-up"
          viewBox="0 0 8 8"
          refX="8"
          refY="4"
          markerWidth="9"
          markerHeight="9"
          markerUnits="userSpaceOnUse"
          orient="auto"
        >
          <path d="M 0 0 L 8 4 L 0 8 Z" fill="#0ca678" />
        </marker>
        <marker
          id="rank-arrow-down"
          viewBox="0 0 8 8"
          refX="8"
          refY="4"
          markerWidth="9"
          markerHeight="9"
          markerUnits="userSpaceOnUse"
          orient="auto"
        >
          <path d="M 0 0 L 8 4 L 0 8 Z" fill="#e07a1f" />
        </marker>
        <marker
          id="rank-arrow-same"
          viewBox="0 0 8 8"
          refX="8"
          refY="4"
          markerWidth="9"
          markerHeight="9"
          markerUnits="userSpaceOnUse"
          orient="auto"
        >
          <path d="M 0 0 L 8 4 L 0 8 Z" fill="#c3ccd9" />
        </marker>
      </defs>
      {geometry
        ? geometry.anchors.map((anchor) => {
            const midX = (anchor.leftX + anchor.rightX) / 2;
            const tone = rankLinkTone(anchor.rankChange);
            return (
              <path
                key={anchor.item_id}
                className={`rank-link rank-link--${tone}`}
                d={`M ${anchor.leftX} ${anchor.leftY} C ${midX} ${anchor.leftY}, ${midX} ${anchor.rightY}, ${anchor.rightX} ${anchor.rightY}`}
                markerEnd={`url(#rank-arrow-${tone})`}
                data-item-id={anchor.item_id}
                data-rank-delta={anchor.rankChange}
              />
            );
          })
        : null}
    </svg>
  );
}

// A Demo-3-style query combobox: the text input and the 倒三角 toggle live in
// the SAME bordered control (no separate box), and the dropdown lists the raw
// natural-language queries verbatim (no id-derived summaries). Picking one runs
// it immediately; the input stays editable for custom queries.
function QueryComboBox({
  queries,
  activeQueryId,
  value,
  disabled,
  onChange,
  onSubmit,
  onPick,
}: {
  queries: QueryOption[];
  activeQueryId: string | null;
  value: string;
  disabled: boolean;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onPick: (query: QueryOption) => void;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;
    function onPointer(event: MouseEvent) {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setOpen(false);
      }
    }
    window.addEventListener("pointerdown", onPointer);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("pointerdown", onPointer);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div className="query-combobox" ref={rootRef}>
      <div className="query-combobox-control">
        <input
          id="function-chain-query"
          type="text"
          value={value}
          disabled={disabled}
          aria-label="Query text"
          placeholder="Type any natural-language product query…"
          autoComplete="off"
          spellCheck={false}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              onSubmit();
            }
          }}
        />
        <button
          type="button"
          className="query-combobox-toggle"
          aria-label="Preset queries"
          aria-haspopup="listbox"
          aria-expanded={open}
          onClick={() => setOpen((current) => !current)}
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
      {open ? (
        <ul
          id="preset-query-menu"
          className="query-combobox-menu"
          role="listbox"
          aria-label="Query presets"
        >
          {queries.map((query) => (
            <li key={query.id} role="presentation">
              <button
                type="button"
                role="option"
                aria-selected={activeQueryId === query.id}
                onClick={() => {
                  onPick(query);
                  setOpen(false);
                }}
              >
                <span className="query-combobox-option-label">
                  {query.query_text}
                </span>
              </button>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

interface CompareCell {
  label: string;
  a: string;
  b: string;
  diff: boolean;
  // Which side "wins" this row, when the field has a clear favorable
  // direction. `null` means the row is descriptive (title, brand, date, …) or
  // a tie, so no win/lose marker is shown.
  winner: "a" | "b" | null;
}

function compareWinner(
  aVal: number,
  bVal: number,
  direction: "higher" | "lower",
): "a" | "b" | null {
  if (aVal === bVal) {
    return null;
  }
  const aWins = direction === "higher" ? aVal > bVal : aVal < bVal;
  return aWins ? "a" : "b";
}

interface CompareSection {
  title: string;
  cells: CompareCell[];
}

// Build the three PK sections from two fully resolved rows. Every value is a
// string so the overlay is pure presentation; `diff` marks cells where the two
// products disagree so the reader can spot what actually moved the ranking.
function compareSections(rows: [CompareRow, CompareRow]): CompareSection[] {
  const [a, b] = rows;
  const aFeatures = chainFeatureValues(a.vector, a.vector.score);
  const bFeatures = chainFeatureValues(b.vector, b.vector.score);
  const cell = (
    label: string,
    aText: string,
    bText: string,
    winner: "a" | "b" | null = null,
  ): CompareCell => ({
    label,
    a: aText,
    b: bText,
    diff: aText !== bText,
    winner,
  });

  // Raw business fields mirror the main product card exactly: the same four
  // metrics (clicks, sales, rating as a 0–1 percent, listed) plus the identity
  // attributes and price. `inventory` and `return rate` are NOT model features
  // and never appear on the card, so they are omitted here to stay consistent.
  const raw = [
    cell("title", formatText(a.vector.title), formatText(b.vector.title)),
    cell("type", a.vector.product_type, b.vector.product_type),
    cell("brand", formatText(a.vector.brand), formatText(b.vector.brand)),
    cell(
      "material",
      formatText(a.vector.material),
      formatText(b.vector.material),
    ),
    cell("style", formatText(a.vector.style), formatText(b.vector.style)),
    cell("color", formatText(a.vector.color), formatText(b.vector.color)),
    cell(
      "price",
      formatPrice(a.vector.display_price_usd),
      formatPrice(b.vector.display_price_usd),
      compareWinner(
        a.vector.display_price_usd,
        b.vector.display_price_usd,
        "lower",
      ),
    ),
    cell(
      "rating",
      formatPercentValue(a.vector.rating, 1),
      formatPercentValue(b.vector.rating, 1),
      compareWinner(a.vector.rating, b.vector.rating, "higher"),
    ),
    cell(
      "clicks",
      formatCount(a.vector.clicks_30d),
      formatCount(b.vector.clicks_30d),
      compareWinner(a.vector.clicks_30d, b.vector.clicks_30d, "higher"),
    ),
    cell(
      "sales",
      formatCount(a.vector.sales_30d),
      formatCount(b.vector.sales_30d),
      compareWinner(a.vector.sales_30d, b.vector.sales_30d, "higher"),
    ),
    cell(
      "listed",
      a.vector.release_date,
      b.vector.release_date,
      compareWinner(a.vector.release_epoch, b.vector.release_epoch, "higher"),
    ),
  ];

  const engineered = [
    cell(
      "popularity",
      formatPercentValue(aFeatures.popularity, 0),
      formatPercentValue(bFeatures.popularity, 0),
      compareWinner(aFeatures.popularity, bFeatures.popularity, "higher"),
    ),
    cell(
      "price affinity",
      formatPercentValue(aFeatures.price_affinity, 0),
      formatPercentValue(bFeatures.price_affinity, 0),
      compareWinner(
        aFeatures.price_affinity,
        bFeatures.price_affinity,
        "higher",
      ),
    ),
    cell(
      "freshness",
      formatPercentValue(aFeatures.freshness, 0),
      formatPercentValue(bFeatures.freshness, 0),
      compareWinner(aFeatures.freshness, bFeatures.freshness, "higher"),
    ),
    cell(
      "rating (0–1)",
      formatPercentValue(aFeatures.rating, 1),
      formatPercentValue(bFeatures.rating, 1),
      compareWinner(aFeatures.rating, bFeatures.rating, "higher"),
    ),
    cell(
      "semantic",
      aFeatures.semantic_score.toFixed(4),
      bFeatures.semantic_score.toFixed(4),
      compareWinner(
        aFeatures.semantic_score,
        bFeatures.semantic_score,
        "higher",
      ),
    ),
  ];

  const ranking = [
    cell(
      "semantic rank",
      `#${a.vector.rank}`,
      `#${b.vector.rank}`,
      compareWinner(a.vector.rank, b.vector.rank, "lower"),
    ),
    cell(
      "reranked rank",
      `#${a.business.rank}`,
      `#${b.business.rank}`,
      compareWinner(a.business.rank, b.business.rank, "lower"),
    ),
    cell(
      "rank shift",
      formatSigned(a.vector.rank - a.business.rank),
      formatSigned(b.vector.rank - b.business.rank),
      compareWinner(
        a.vector.rank - a.business.rank,
        b.vector.rank - b.business.rank,
        "higher",
      ),
    ),
    cell(
      "business score",
      a.business.score.toFixed(4),
      b.business.score.toFixed(4),
      compareWinner(a.business.score, b.business.score, "higher"),
    ),
  ];

  return [
    { title: "Raw business fields", cells: raw },
    { title: "Engineered features", cells: engineered },
    { title: "Ranking outcome", cells: ranking },
  ];
}

function CompareColumnHead({
  row,
  winner,
}: {
  row: CompareRow;
  winner: boolean;
}) {
  const accessibleName = row.vector.title ?? row.vector.product_type;
  return (
    <div className="compare-col-head">
      <span className="compare-col-photo">
        <ProductPhoto product={row.vector} />
      </span>
      <span className="compare-col-copy">
        <strong>{accessibleName}</strong>
        <code>{row.itemId}</code>
      </span>
      {winner ? (
        <span className="compare-win-badge" title="Wins the reranked outcome">
          winner
        </span>
      ) : null}
    </div>
  );
}

function CompareOverlay({
  rows,
  queryText,
  onClose,
}: {
  rows: [CompareRow, CompareRow];
  queryText: string;
  onClose: () => void;
}) {
  const closeRef = useRef<HTMLButtonElement | null>(null);
  const sections = useMemo(() => compareSections(rows), [rows]);
  // The overall winner is the product with the lower reranked (business_order)
  // rank — the outcome the Function Chain actually serves to the reader.
  const [left, right] = rows;
  const leftWins = left.business.rank < right.business.rank;
  const rightWins = right.business.rank < left.business.rank;

  useEffect(() => {
    closeRef.current?.focus();
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="compare-overlay" data-testid="compare-overlay">
      <div className="compare-backdrop" aria-hidden="true" onClick={onClose} />
      <div
        className="compare-dialog"
        role="dialog"
        aria-modal="true"
        aria-label="Product comparison"
      >
        <header className="compare-dialog-header">
          <div className="compare-dialog-title">
            <strong>Product PK</strong>
            <span title={queryText}>{queryText}</span>
          </div>
          <button
            type="button"
            className="compare-close"
            ref={closeRef}
            aria-label="Close comparison"
            onClick={onClose}
          >
            ×
          </button>
        </header>

        <div className="compare-grid">
          <div className="compare-corner" aria-hidden="true" />
          <CompareColumnHead row={rows[0]} winner={leftWins} />
          <CompareColumnHead row={rows[1]} winner={rightWins} />

          {sections.map((section) => (
            <div className="compare-section" key={section.title}>
              <h3 className="compare-section-heading">{section.title}</h3>
              {section.cells.map((cell) => (
                <div className="compare-row" key={cell.label}>
                  <span className="compare-row-label">{cell.label}</span>
                  <span
                    className={`compare-cell${cell.diff ? " compare-cell--diff" : ""}`}
                  >
                    {cell.a}
                    {cell.winner === "a" ? (
                      <span
                        className="compare-win-mark"
                        title="Better on this field"
                      >
                        ✓
                      </span>
                    ) : null}
                  </span>
                  <span
                    className={`compare-cell${cell.diff ? " compare-cell--diff" : ""}`}
                  >
                    {cell.b}
                    {cell.winner === "b" ? (
                      <span
                        className="compare-win-mark"
                        title="Better on this field"
                      >
                        ✓
                      </span>
                    ) : null}
                  </span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export function FunctionChainDemoPage() {
  const [queries, setQueries] = useState<QueryOption[]>([]);
  const [queryText, setQueryText] = useState("");
  const [activeQueryId, setActiveQueryId] = useState<string | null>(null);
  const [comparison, setComparison] = useState<SearchComparison | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [failedAction, setFailedAction] = useState<"queries" | "search" | null>(
    null,
  );
  const [compareItems, setCompareItems] = useState<string[]>([]);
  const [compareOpen, setCompareOpen] = useState(false);
  const [comparing, setComparing] = useState(false);
  const requestVersion = useRef(0);
  const requestController = useRef<AbortController | null>(null);

  async function runSearch(text: string, queryId: string | null) {
    const trimmed = text.trim();
    if (!trimmed) {
      return;
    }
    const version = ++requestVersion.current;
    requestController.current?.abort();
    const controller = new AbortController();
    requestController.current = controller;
    setComparison(null);
    setCompareItems([]);
    setCompareOpen(false);
    setLoading(true);
    setError(null);
    setFailedAction(null);
    try {
      const nextComparison = await fetchSearchComparison(
        trimmed,
        queryId,
        controller.signal,
      );
      if (requestVersion.current !== version) {
        return;
      }
      setComparison(nextComparison);
    } catch (caught: unknown) {
      if (
        requestVersion.current === version &&
        !(caught instanceof DOMException && caught.name === "AbortError")
      ) {
        setError("The real Milvus search could not be completed.");
        setFailedAction("search");
      }
    } finally {
      if (requestVersion.current === version) {
        setLoading(false);
      }
    }
  }

  function loadQueries() {
    const version = ++requestVersion.current;
    requestController.current?.abort();
    const controller = new AbortController();
    requestController.current = controller;
    setLoading(true);
    setError(null);
    setFailedAction(null);
    fetchQueryOptions(controller.signal)
      .then((options) => {
        if (requestVersion.current !== version) {
          return;
        }
        setQueries(options);
        const first = options[0];
        if (first) {
          setQueryText(first.query_text);
          setActiveQueryId(first.id);
          // Show results immediately on first load instead of waiting for a
          // manual "Run comparison" click.
          void runSearch(first.query_text, first.id);
        }
      })
      .catch((caught: unknown) => {
        if (
          requestVersion.current === version &&
          !(caught instanceof DOMException && caught.name === "AbortError")
        ) {
          setError("The Function Chain backend is not available.");
          setFailedAction("queries");
        }
      })
      .finally(() => {
        if (requestVersion.current === version) {
          setLoading(false);
        }
      });
  }

  useEffect(() => {
    loadQueries();
    return () => {
      requestVersion.current += 1;
      requestController.current?.abort();
    };
  }, []);

  const vectorByItem = useMemo(
    () =>
      new Map(
        comparison?.vector_order.map((product) => [product.item_id, product]) ??
          [],
      ),
    [comparison],
  );
  const businessByItem = useMemo(
    () =>
      new Map(
        comparison?.business_order.map((product) => [
          product.item_id,
          product,
        ]) ?? [],
      ),
    [comparison],
  );
  const dataset = comparison?.dataset ?? queries[0]?.dataset ?? null;

  // The two products picked for a head-to-head PK, in selection order, each
  // resolved to its vector_order and business_order rows so the overlay can
  // show the full field breakdown side by side.
  const compareProducts = useMemo(() => {
    return compareItems.map((itemId) => {
      const vectorProduct = vectorByItem.get(itemId);
      const businessProduct = businessByItem.get(itemId);
      return {
        itemId,
        vectorProduct,
        businessProduct,
      };
    });
  }, [compareItems, vectorByItem, businessByItem]);

  function toggleCompare(itemId: string) {
    setCompareItems((current) => {
      if (current.includes(itemId)) {
        return current.filter((existing) => existing !== itemId);
      }
      if (current.length >= 2) {
        return current;
      }
      return [...current, itemId];
    });
  }

  function startComparing() {
    setCompareItems([]);
    setComparing(true);
  }

  function cancelComparing() {
    setComparing(false);
    setCompareItems([]);
  }

  function openCompare() {
    if (compareItems.length >= 2) {
      setCompareOpen(true);
      setComparing(false);
    }
  }

  function pickPreset(query: QueryOption) {
    setQueryText(query.query_text);
    setActiveQueryId(query.id);
    void runSearch(query.query_text, query.id);
  }

  function changeQueryText(nextText: string) {
    setQueryText(nextText);
    const match = queries.find((query) => query.query_text === nextText.trim());
    setActiveQueryId(match?.id ?? null);
  }

  function submitQuery(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault();
    const match = queries.find(
      (query) => query.query_text === queryText.trim(),
    );
    void runSearch(queryText, match?.id ?? null);
  }

  function retry() {
    if (failedAction === "queries") {
      loadQueries();
      return;
    }
    const match = queries.find(
      (query) => query.query_text === queryText.trim(),
    );
    void runSearch(queryText, match?.id ?? null);
  }

  const renderRankedProducts = (
    order: RankedProduct[],
    strategy: RankingStrategy,
  ) =>
    order.map((product) => {
      const vectorProduct = vectorByItem.get(product.item_id);
      const businessProduct = businessByItem.get(product.item_id);
      if (!vectorProduct || !businessProduct) {
        return null;
      }
      return (
        <ProductRow
          product={product}
          vectorProduct={vectorProduct}
          businessProduct={businessProduct}
          strategy={strategy}
          compareItems={compareItems}
          comparing={comparing}
          onToggleCompare={toggleCompare}
          key={product.item_id}
        />
      );
    });

  return (
    <main className="function-chain-page">
      <header className="commerce-header">
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
          <h1>Multi-Phase Reranking with Function Chain</h1>
          <a
            className="source-link"
            href="https://github.com/zc277584121/milvus3-demos/tree/main/demos/function-chain-rerank"
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

      <form
        className="search-console"
        aria-labelledby="search-heading"
        onSubmit={submitQuery}
      >
        <h2 id="search-heading" className="search-heading">
          Query
        </h2>
        <QueryComboBox
          queries={queries}
          activeQueryId={activeQueryId}
          value={queryText}
          disabled={loading}
          onChange={changeQueryText}
          onSubmit={submitQuery}
          onPick={pickPreset}
        />
        <button
          type="submit"
          className="search-submit"
          disabled={loading || !queryText.trim()}
        >
          {loading ? "Searching…" : "Search"}
        </button>
      </form>

      {error ? (
        <div className="search-error" role="alert">
          <span>{error}</span>
          <button
            type="button"
            onClick={retry}
            disabled={loading || failedAction === null}
          >
            {loading ? "Retrying…" : "Retry"}
          </button>
        </div>
      ) : null}

      {comparison ? (
        <section className="ranking-workbench" aria-live="polite">
          <DataModelPanel comparison={comparison} />

          <div
            className="ranking-comparison-board"
            data-testid="ranking-comparison-board"
          >
            <section
              className="board-column board-column--left"
              aria-labelledby="vector-column-title"
            >
              <header className="board-column-header">
                <h2 id="vector-column-title">Semantic</h2>
              </header>
              <div
                className="product-list"
                data-testid="vector-ranking-list"
                data-server-order-source="vector_order"
                tabIndex={0}
              >
                {renderRankedProducts(comparison.vector_order, "vector_order")}
              </div>
            </section>

            <section
              className="board-column board-column--middle"
              aria-labelledby="business-column-title"
            >
              <header className="board-column-header">
                <h2 id="business-column-title">Reranked</h2>
              </header>
              <div
                className="product-list"
                data-testid="business-ranking-list"
                data-server-order-source="business_order"
                tabIndex={0}
              >
                {renderRankedProducts(
                  comparison.business_order,
                  "business_order",
                )}
              </div>
            </section>

            <RankShiftLinks comparison={comparison} />
          </div>
        </section>
      ) : (
        <section className="empty-ranking">
          <strong>{dataset?.product_count ?? 240} products</strong>
          <span>Choose a query to compare the two rankings.</span>
        </section>
      )}

      {compareOpen &&
      compareProducts.length === 2 &&
      compareProducts[0].vectorProduct &&
      compareProducts[0].businessProduct &&
      compareProducts[1].vectorProduct &&
      compareProducts[1].businessProduct ? (
        <CompareOverlay
          rows={[
            {
              itemId: compareProducts[0].itemId,
              vector: compareProducts[0].vectorProduct,
              business: compareProducts[0].businessProduct,
            },
            {
              itemId: compareProducts[1].itemId,
              vector: compareProducts[1].vectorProduct,
              business: compareProducts[1].businessProduct,
            },
          ]}
          queryText={comparison?.query_text ?? queryText}
          onClose={() => setCompareOpen(false)}
        />
      ) : null}

      {comparison ? (
        <div className="compare-fab" data-testid="compare-fab">
          {comparing ? (
            <div className="compare-fab--armed">
              <span className="compare-mode-hint" aria-live="polite">
                Pick 2 to compare
              </span>
              <span
                className={`compare-mode-count${
                  compareItems.length >= 2 ? " compare-mode-count--ready" : ""
                }`}
              >
                {compareItems.length}/2
              </span>
              <button
                type="button"
                className="compare-cancel"
                onClick={cancelComparing}
              >
                Cancel
              </button>
              <button
                type="button"
                className="compare-launch"
                data-testid="compare-launch"
                disabled={compareItems.length < 2}
                onClick={openCompare}
              >
                Compare
              </button>
            </div>
          ) : (
            <button
              type="button"
              className="compare-launch compare-fab--round"
              data-testid="compare-launch"
              aria-label="Compare"
              onClick={startComparing}
            >
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M8 5h11M8 12h11M8 19h11" />
                <circle cx="4" cy="5" r="1.4" />
                <circle cx="4" cy="12" r="1.4" />
                <circle cx="4" cy="19" r="1.4" />
              </svg>
              <span>Compare</span>
            </button>
          )}
        </div>
      ) : null}
    </main>
  );
}
