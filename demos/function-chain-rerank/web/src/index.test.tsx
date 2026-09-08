import { readFileSync } from "node:fs";

import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { FunctionChainDemoPage, productImageUrl } from "./index";

const dataset = {
  dataset_id: "synthetic-commerce-catalog" as const,
  revision: "synthetic-commerce-catalog-r1" as const,
  manifest_sha256: "a".repeat(64),
  dataset_name: "Synthetic Commerce Catalog" as const,
  publisher_and_data_credit:
    "milvus3-demos project (no external publisher)" as const,
  license: "CC0-1.0" as const,
  license_url: "https://creativecommons.org/publicdomain/zero/1.0/",
  license_conflict_record: "none" as const,
  publication_review_required: false as const,
  source_type: "synthetic_catalog_with_simulated_operations" as const,
  synthetic: true as const,
  real_product_metadata: false as const,
  real_product_photos: false as const,
  real_transaction_data: false as const,
  contains_simulated_operational_signals: true as const,
  simulated_signal_label: "deterministic_simulated_operational_signal" as const,
  as_of_date: "2026-08-20",
  product_count: 240,
  product_type_count: 20,
  image_count: 240,
  embedding: {
    dimensions: 1024,
    algorithm: "bge-m3-onnx-int8-dense-l2-type-x-attr-anchor",
  },
  semantic_score_preprocessing: { operation: "round_decimal", decimal: 5 },
  relevance_ground_truth: {
    path: "relevance-ground-truth.json",
    evaluation_only: true,
    production_ranking_consumers: [],
  },
  field_provenance: {
    title: "synthetic_authored_metadata",
    inventory_units: "deterministic_simulated_operational_signal",
  },
};

function product(itemId: string, rank: number, score: number) {
  const first = itemId === "SYN-DESK-001";
  return {
    id: first ? 1 : 2,
    rank,
    score,
    item_id: itemId,
    title: first ? "Compact writing desk" : null,
    product_type: "DESK",
    brand: first ? null : "Hearth & Beam",
    color: first ? "Brown" : null,
    material: first ? "Wood" : null,
    style: first ? null : "Modern",
    node_name: "Home & Kitchen > Furniture > Desks",
    description: first ? null : "An authored synthetic catalog description.",
    bullet_points: first ? [] : ["Authored synthetic bullet point"],
    main_image_id: first ? "main-image-1" : "main-image-2",
    selected_image_id: first ? "main-image-1" : "main-image-2",
    image_role: "main",
    source_object_path: `images/small/${first ? "1" : "2"}.jpg`,
    source_url: `synthetic://catalog/${itemId}/placeholder.jpg`,
    display_price_usd: first ? 58 : 62.25,
    rating_value: first ? 3.2 : 4.8,
    clicks_30d: first ? 1141 : 9000,
    sales_30d: first ? 89 : 400,
    inventory_units: first ? 2 : 92,
    inventory_capacity: first ? 91 : 100,
    release_date: first ? "2024-10-29" : "2026-06-01",
    release_epoch: first ? 1730073600 : 1781740800,
    rating: first ? 0.3 : 0.95,
    inventory: first ? 0.021978 : 0.92,
    return_rate: first ? 0.21 : 0.025,
    freshness: first ? 0.09589 : 0.890411,
    image_path: `images/${itemId}.jpg`,
    image_mime: "image/jpeg" as const,
    image_width: 640,
    image_height: 480,
    image_sha256: first ? "b".repeat(64) : "c".repeat(64),
    operational_signal_provenance:
      "deterministic_simulated_operational_signal" as const,
    field_provenance: {
      title: "synthetic_authored_metadata",
      display_price_usd: "deterministic_simulated_operational_signal",
    },
  };
}

const queries = [
  [
    "compact-dark-wood-desk",
    "a compact dark wood desk for a small home office",
  ],
  ["soft-neutral-living-room-rug", "a soft neutral rug for a cozy living room"],
  ["red-weekend-backpack", "a red backpack for the gym and weekend trips"],
  [
    "comfortable-on-ear-headphones",
    "comfortable on-ear headphones for everyday music",
  ],
  [
    "automatic-black-commuter-umbrella",
    "a sturdy automatic black umbrella for commuting in heavy rain",
  ],
  ["warm-bronze-modern-bed", "a warm bronze bed frame for a modern bedroom"],
].map(([id, query_text], index) => ({
  id,
  query_text,
  story: `Natural shopping intent ${index + 1}`,
  split: (index < 4 ? "train" : "validation") as "train" | "validation",
  dataset,
}));

const query = queries[0];

const comparison = {
  query_id: query.id,
  query_text: query.query_text,
  execution_path: "milvus_l0_xgboost_function_chain" as const,
  model_resource_name: "milvus3_demos_function_chain_rerank_model",
  function_chain: {
    stage: "L0_RERANK" as const,
    operation: "xgboost" as const,
    feature_order: [
      "semantic_score",
      "rating",
      "popularity",
      "price_affinity",
      "freshness",
    ],
    parallel_inputs: false as const,
    intermediate_feature_orders: false as const,
    vector_order_score: "semantic_score" as const,
    business_order_score: "business_score" as const,
    chain_steps: [
      {
        name: "popularity",
        operation: "num_combine",
        output: "popularity",
        description: "Weighted blend of 30-day clicks and sales.",
        code: '.map("popularity", fn.num_combine(...))',
      },
      {
        name: "price_affinity",
        operation: "decay",
        output: "price_affinity",
        description: "Linear price decay.",
        code: '.map("price_affinity", fn.decay(...))',
      },
      {
        name: "freshness",
        operation: "decay",
        output: "freshness",
        description: "Exponential recency decay.",
        code: '.map("freshness", fn.decay(...))',
      },
      {
        name: "semantic_round",
        operation: "round_decimal",
        output: "normalized_semantic_score",
        description: "Round the vector similarity score.",
        code: '.map("normalized_semantic_score", fn.round_decimal(...))',
      },
      {
        name: "xgboost_rerank",
        operation: "xgboost",
        output: "$score",
        description: "XGBoost combines the five features.",
        code: '.map("$score", fn.xgboost(...))',
      },
    ],
  },
  dataset,
  data_model: {
    fields: [
      {
        name: "id",
        type: "INT64",
        role: "primary_key",
        note: "stable product id",
      },
      {
        name: "embedding",
        type: "FLOAT_VECTOR(1024)",
        role: "vector",
        note: "pure BGE-M3 dense vector",
      },
      {
        name: "clicks_30d",
        type: "INT64",
        role: "chain_input",
        note: "simulated clicks",
      },
      {
        name: "rating",
        type: "FLOAT",
        role: "model_feature",
        note: "derived rating",
      },
    ],
    chain_trace: {
      item_id: "SYN-DESK-001",
      title: "DESK",
      steps: [
        {
          name: "semantic_score",
          value: 0.91,
          inputs: { $score: "0.91000" },
          note: "vector search cosine similarity",
        },
        {
          name: "popularity",
          value: 0.42,
          inputs: { clicks_30d: 1200, sales_30d: 80 },
          note: "num_combine weighted blend",
        },
        {
          name: "business_score",
          value: 0.48,
          inputs: { semantic_score: 0.91, rating: 0.9 },
          note: "XGBoost combines five features",
        },
      ],
    },
  },
  vector_order: [
    product("SYN-DESK-001", 1, 0.91),
    product("SYN-DESK-002", 2, 0.82),
  ],
  business_order: [
    product("SYN-DESK-002", 1, 0.81),
    product("SYN-DESK-001", 2, 0.48),
  ],
};

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("FunctionChainDemoPage", () => {
  it("renders the data model, chain, rank mapping, and both side-by-side orders", async () => {
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      const payload = url.endsWith("/queries") ? queries : comparison;
      return Promise.resolve(
        new Response(JSON.stringify(payload), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<FunctionChainDemoPage />);

    // Results appear immediately on first load without a manual run click.
    await screen.findAllByText("num_combine");
    expect(screen.getByText("Function Chain product search")).toBeTruthy();
    expect(
      (screen.getByLabelText("Query text") as HTMLInputElement).value,
    ).toBe(query.query_text);
    // The preset listbox lives inside the same bordered combobox as the input;
    // its toggle is an icon-only 倒三角 button, and the options show the raw
    // natural-language queries (not id-derived summaries).
    const presetToggle = screen.getByRole("button", {
      name: "Preset queries",
    });
    expect(presetToggle.getAttribute("aria-expanded")).toBe("false");
    expect(presetToggle.getAttribute("aria-haspopup")).toBe("listbox");
    expect(screen.queryAllByText("round_decimal")).toHaveLength(0);
    expect(screen.getAllByText("xgboost")).toBeTruthy();

    const cards = await screen.findAllByTestId("product-card");
    // Two columns, each holding the two mocked products in its server order.
    expect(cards.map((card) => card.getAttribute("data-item-id"))).toEqual([
      "SYN-DESK-001",
      "SYN-DESK-002",
      "SYN-DESK-002",
      "SYN-DESK-001",
    ]);
    const productImages = screen.getAllByRole("img", {
      name: /Product image for/,
    });
    expect(productImages[0].getAttribute("src")).toBe(
      "/api/function-chain/v1/assets/images/SYN-DESK-001.jpg",
    );
    expect(screen.getAllByAltText("Product image for DESK")).toHaveLength(2);

    // Data model panel renders the DAG: a leading source product card (a
    // pixel-identical replica of the search cards, with six fields boxed in
    // place), four step nodes, and ten connecting edges (six card→step "pull"
    // ties plus the four flow edges), with code hidden until a node is hovered.
    const flow = screen.getByTestId("function-chain-flow");
    expect(flow.querySelectorAll(".flow-step-node")).toHaveLength(4);
    expect(flow.querySelectorAll(".flow-field")).toHaveLength(0);
    expect(flow.querySelectorAll(".flow-dag-edge")).toHaveLength(10);
    expect(flow.querySelectorAll(".flow-dag-edge--pull")).toHaveLength(6);
    expect(flow.textContent).toContain("clicks_30d");
    expect(flow.textContent).toContain("popularity");
    expect(flow.textContent).toContain("$score");

    // The leading product card is the semantic top-1 and boxes each consumed
    // field in place; the two simulated signals are present as dashed tags.
    const sourceCard = flow.querySelector("[data-flow-card]");
    expect(sourceCard).toBeTruthy();
    expect(sourceCard?.getAttribute("data-item-id")).toBe("SYN-DESK-001");
    expect(flow.querySelectorAll("[data-card-field]")).toHaveLength(6);
    expect(
      flow.querySelectorAll("[data-card-field].flow-field-box--simulated"),
    ).toHaveLength(2);
    expect(sourceCard?.textContent).toContain("$58.00");

    // Code is revealed on hover, not always visible per step.
    let codePanel = flow.querySelector(".flow-code-panel");
    expect(codePanel?.getAttribute("data-flow-code")).toBe("none");
    expect(flow.querySelector(".flow-code-block-pre")).toBeNull();

    const popularityNode = flow.querySelector(
      '[data-flow-step-node="popularity"]',
    );
    fireEvent.mouseEnter(popularityNode!);
    codePanel = flow.querySelector(".flow-code-panel");
    expect(codePanel?.getAttribute("data-flow-code")).toBe("popularity");
    expect(
      codePanel?.querySelector(".flow-code-block-pre code")?.textContent,
    ).toContain("fn.num_combine");
    expect(codePanel?.querySelector(".py-fn")?.textContent).toContain("fn");
    expect(codePanel?.querySelectorAll(".py-entity")).toHaveLength(1);

    // The two columns are product cards; the link SVG connects them.
    const board = screen.getByTestId("ranking-comparison-board");
    expect(board).toBeTruthy();
    expect(screen.getByTestId("rank-shift-links")).toBeTruthy();

    expect(cards[0].getAttribute("data-title")).toBe("Compact writing desk");
    expect(cards[0].getAttribute("data-vector-rank")).toBe("1");
    expect(cards[0].getAttribute("data-business-rank")).toBe("2");
    expect(cards[0].getAttribute("data-rank-delta")).toBe("-1");

    // Each card reads like a store listing: score (semantic left, business
    // right) plus To-C facts — price, a 5-star rating, the listed date, and the
    // product type. No engineered feature column; the full breakdown lives
    // behind Compare.
    expect(cards[0].getAttribute("data-score-label")).toBe("semantic");
    expect(cards[2].getAttribute("data-score-label")).toBe("business");
    const features = cards[0].querySelectorAll(".feature");
    expect(features).toHaveLength(4);
    const featureLabels = Array.from(features).map(
      (node) => node.querySelector("small")?.textContent,
    );
    expect(featureLabels).toEqual(["price", "rating", "listed", "type"]);
    // The rating is a 5-star read of the original 1–5 rating value
    // (SYN-DESK-001 = 3.2 → 3.2 stars), not a machine percent.
    const starRating = cards[0].querySelector(".star-rating");
    expect(starRating).toBeTruthy();
    expect(starRating?.getAttribute("aria-label")).toBe("3.2 out of 5 stars");
    // The engineered-feature column is gone: no feature cells, no toggle.
    expect(screen.queryByTestId("feature-cell")).toBeNull();
    expect(
      screen.queryByRole("button", {
        name: "Toggle engineered features column",
      }),
    ).toBeNull();

    const [, request] = fetchMock.mock.calls[1] as [string, RequestInit];
    expect(JSON.parse(String(request.body))).toEqual({
      query_id: query.id,
      query_text: query.query_text,
    });
  });

  it("selects up to two products and opens a three-section PK overlay", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockImplementation((url: string) =>
          Promise.resolve(
            new Response(
              JSON.stringify(url.endsWith("/queries") ? queries : comparison),
              { status: 200, headers: { "Content-Type": "application/json" } },
            ),
          ),
        ),
    );
    render(<FunctionChainDemoPage />);
    await screen.findAllByTestId("product-card");

    // Cards are clean by default: no per-card checkboxes until Compare is armed.
    // The Compare control lives in the bottom-right floating round button and
    // carries its accessible name via aria-label (the visible glyph is an SVG).
    expect(screen.queryAllByTestId("compare-toggle")).toHaveLength(0);
    const launch = screen.getByRole("button", {
      name: "Compare",
    }) as HTMLButtonElement;
    expect(launch.disabled).toBe(false);
    expect(launch.getAttribute("aria-label")).toBe("Compare");

    // Arming Compare reveals a checkbox on every card (two columns × two
    // products = four) and swaps the round button for the selection-mode cluster.
    fireEvent.click(launch);
    const armedLaunch = screen.getByRole("button", {
      name: "Compare",
    }) as HTMLButtonElement;
    expect(armedLaunch.disabled).toBe(true);
    expect(armedLaunch.textContent).toBe("Compare");
    const toggles = screen.getAllByTestId("compare-toggle");
    expect(toggles).toHaveLength(4);
    expect(screen.getByText("Pick 2 to compare")).toBeTruthy();
    expect(screen.getByText("0/2")).toBeTruthy();

    // Toggle the first product (SYN-DESK-001). It appears in both columns, but
    // selection is keyed by item_id, so both of its cards light up.
    fireEvent.click(
      toggles.find((t) => t.getAttribute("data-item-id") === "SYN-DESK-001")!,
    );
    expect(screen.getByText("1/2")).toBeTruthy();
    expect(
      screen
        .getAllByTestId("product-card")
        .filter((c) => c.getAttribute("data-compare-selected") === "true"),
    ).toHaveLength(2);

    // A third product toggle is blocked only once two are chosen; selecting the
    // second distinct product (SYN-DESK-002) fills the pair and enables launch.
    fireEvent.click(
      toggles.find((t) => t.getAttribute("data-item-id") === "SYN-DESK-002")!,
    );
    expect(screen.getByText("2/2")).toBeTruthy();
    expect(armedLaunch.disabled).toBe(false);

    fireEvent.click(armedLaunch);
    const overlay = await screen.findByTestId("compare-overlay");
    expect(overlay).toBeTruthy();
    expect(overlay.textContent).toContain("Raw business fields");
    expect(overlay.textContent).toContain("Engineered features");
    expect(overlay.textContent).toContain("Ranking outcome");
    // The raw "rating" row now uses the same 0–1 percent as the main card
    // (no 1–5 star value), so raw and engineered stay consistent.
    expect(overlay.textContent).not.toContain("★");
    expect(overlay.textContent).toContain("rating (0–1)");
    expect(overlay.textContent).toContain("30.0%");
    expect(overlay.textContent).toContain("95.0%");
    // inventory / return rate are not model features and not on the main card,
    // so they are absent from the PK overlay too.
    expect(overlay.textContent).not.toContain("return rate");
    expect(overlay.textContent).not.toContain("inventory");
    // Each row with a clear direction carries a per-row win mark, and the
    // product that wins the reranked outcome gets a winner badge in its header.
    expect(
      overlay.querySelectorAll(".compare-win-mark").length,
    ).toBeGreaterThan(0);
    const winBadges = overlay.querySelectorAll(".compare-win-badge");
    expect(winBadges).toHaveLength(1);

    // Deselecting a product collapses the pair; the overlay closes on Escape.
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.queryByTestId("compare-overlay")).toBeNull();
  });

  it("shows explicit image loading and error states for JPEG assets", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string) =>
        Promise.resolve(
          new Response(
            JSON.stringify(url.endsWith("/queries") ? queries : comparison),
            {
              status: 200,
              headers: { "Content-Type": "application/json" },
            },
          ),
        ),
      ),
    );
    render(<FunctionChainDemoPage />);
    await screen.findAllByRole("img", { name: /Product image for/ });

    const images = await screen.findAllByRole("img", {
      name: /Product image for/,
    });
    // Two columns render the same two products (four cards) plus the DAG's
    // leading source card, so five loading placeholders.
    expect(screen.getAllByText("Loading image…")).toHaveLength(5);
    fireEvent.load(images[0]);
    expect(screen.getAllByText("Loading image…")).toHaveLength(4);
    fireEvent.error(images[1]);
    expect(screen.getByRole("status").textContent).toContain(
      "image unavailable",
    );
  });

  it("marks a derived English summary as distinct from authored title text", async () => {
    const derivedProduct = {
      ...product("SYN-DESK-001", 1, 0.91),
      title: "Synthetic Automatic Black Folding Travel Umbrella",
      product_type: "UMBRELLA",
      field_provenance: {
        ...product("SYN-DESK-001", 1, 0.91).field_provenance,
        title: "deterministic_english_summary_from_synthetic_authored_metadata",
      },
    };
    const derivedComparison = {
      ...comparison,
      vector_order: [derivedProduct],
      business_order: [derivedProduct],
    };
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockImplementation((url: string) =>
          Promise.resolve(
            Response.json(
              url.endsWith("/queries") ? queries : derivedComparison,
            ),
          ),
        ),
    );

    render(<FunctionChainDemoPage />);

    const derivedNotes = await screen.findAllByText("summary");
    expect(derivedNotes.length).toBeGreaterThanOrEqual(1);
    const cards = await screen.findAllByTestId("product-card");
    expect(cards[0].getAttribute("data-title-provenance")).toBe(
      "deterministic_english_summary_from_synthetic_authored_metadata",
    );
  });

  it("clears stale results immediately when the query changes", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockImplementation((url: string) =>
          Promise.resolve(
            new Response(
              JSON.stringify(url.endsWith("/queries") ? queries : comparison),
              { status: 200, headers: { "Content-Type": "application/json" } },
            ),
          ),
        ),
    );
    render(<FunctionChainDemoPage />);
    await screen.findAllByTestId("product-card");

    // Open the 倒三角 dropdown and pick the second preset. The option shows the
    // raw natural-language query, not an id-derived summary.
    fireEvent.click(screen.getByRole("button", { name: "Preset queries" }));
    fireEvent.click(
      screen.getByRole("option", {
        name: "a soft neutral rug for a cozy living room",
      }),
    );

    // Stale results clear synchronously before the next search resolves.
    expect(screen.queryAllByTestId("product-card")).toHaveLength(0);
    expect(screen.getByText("240 products")).toBeTruthy();
    expect(screen.queryByTitle(query.query_text)).toBeNull();
  });

  it("keeps error, retry, and disabled states consistent", async () => {
    let searchAttempts = 0;
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.endsWith("/queries")) {
        return Promise.resolve(Response.json(queries));
      }
      searchAttempts += 1;
      return Promise.resolve(
        searchAttempts === 1
          ? new Response("unavailable", { status: 503 })
          : Response.json(comparison),
      );
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<FunctionChainDemoPage />);

    // The first search runs automatically on load and fails (503).
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain(
      "The real Milvus search could not be completed.",
    );
    expect(screen.queryAllByTestId("product-card")).toHaveLength(0);

    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(
      (screen.getByLabelText("Query text") as HTMLInputElement).disabled,
    ).toBe(true);
    await screen.findAllByTestId("product-card");
    await waitFor(() => expect(screen.queryByRole("alert")).toBeNull());
    expect(searchAttempts).toBe(2);
  });

  it("keeps client source free of old contracts, answer keys, and reorder calls", () => {
    const source = readFileSync("src/index.tsx", "utf8");
    for (const forbidden of [
      "milvus-commerce-catalog-r1",
      "project_owned_synthetic",
      "image/svg+xml",
      "business_target_id",
      "baseline_reference_id",
      "hard_negative_ids",
      "query_label",
      "subcategory",
      "SHAP",
      "causal",
      "contribution",
    ]) {
      expect(source).not.toContain(forbidden);
    }
    expect(source).not.toMatch(/\.sort\s*\(/);
    expect(source).not.toMatch(/\.reverse\s*\(/);
    expect(source).not.toMatch(/\.toSorted\s*\(/);
    expect(source).not.toMatch(
      /(?:Real|ABO) (?:business|operational) (?:data|signals)/,
    );
  });

  it("encodes every JPEG asset path segment", () => {
    expect(productImageUrl("images/a b.jpg")).toBe(
      "/api/function-chain/v1/assets/images/a%20b.jpg",
    );
  });
});
