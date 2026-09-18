import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  DEFAULT_QUERY,
  EmbeddingListDemoPage,
  QUERY_OPTIONS,
  embeddingListDemo,
} from "./index";

const dataset = {
  dataset_id: "nasa-systems-engineering-handbook-rev2",
  title: "NASA Systems Engineering Handbook",
  revision: "NASA/SP-2016-6105 Rev 2",
  document_identifier: "NASA/SP-2016-6105 Rev 2",
  ntrs_id: 20170001761,
  ntrs_record_url: "https://ntrs.nasa.gov/citations/20170001761",
  official_pdf_url:
    "https://ntrs.nasa.gov/api/citations/20170001761/downloads/20170001761.pdf",
  distribution: "PUBLIC",
  rights_determination: "PUBLIC_USE_PERMITTED",
  contains_third_party_material: false,
  attribution:
    "NASA Systems Engineering Handbook, Rev 2 (NASA/SP-2016-6105 Rev 2), Steven R. Hirshorn, Linda D. Voss, and Linda K. Bromley. NASA Technical Reports Server record 20170001761.",
  authors: [
    { name: "Steven R. Hirshorn", affiliation: "NASA Headquarters" },
    { name: "Linda D. Voss", affiliation: "ASRC" },
    { name: "Linda K. Bromley", affiliation: "ASRC" },
  ],
  render_statement:
    "Page images are unmodified 144-DPI renders from the official PDF.",
  endorsement_statement: "NASA does not endorse Milvus or this demo.",
  renderer: "PyMuPDF",
  renderer_version: "1.28.2",
  render_dpi: 144,
  pdf_page_index_base: 1,
  source_pdf_page_count: 356,
  source_pdf_bytes: 4122125,
  page_count: 40,
  manifest_sha256: "a".repeat(64),
  pdf_sha256: "b".repeat(64),
};

const status = {
  status: "ready",
  implementation_status: "implemented",
  runtime_ready: true,
  collection_name: "milvus3_demos_nasa_seh_colsmol_pages",
  metric_type: "MAX_SIM_COSINE",
  dataset,
  query_presets: QUERY_OPTIONS.map((option) => ({
    query_id: option.id,
    text: option.text,
  })),
  prepare_timings: {
    cold: { total_ms: 28450.2 },
    warm: { total_ms: 34.8 },
  },
  model: {
    model_id: "vidore/colSmol-256M",
    inference_dtype: "float32",
    device_type: "cpu",
    cpu_only: true,
  },
  milvus: { server_version: "3.0.0", target_exists: true },
};

const search = {
  status: "passed",
  execution_path: "milvus_embedding_list_max_sim_cosine",
  query: DEFAULT_QUERY,
  metric_type: "MAX_SIM_COSINE",
  dataset,
  query_embedding: {
    vector_counts: [23],
    vector_dimension: 128,
    inference_device: "cpu",
    inference_dtype: "float32",
  },
  page_vector_counts: Array(40).fill(1139),
  ranking_equal: true,
  max_score_delta: 0.000002,
  score_absolute_tolerance: 0.005,
  timings: {
    query_inference_ms: 27201.2,
    milvus_search_ms: 42.1,
    local_score_ms: 18.2,
    total_ms: 27261.5,
  },
  latency_ms: 27261.5,
  local_explanations: [
    {
      page_id: "nasa-seh-printed-053",
      source: "local_colsmol_query_page_multi_vector",
      rank_source: "milvus_page_level_max_sim_cosine",
      query_vector_count: 23,
      page_vector_count: 1139,
      explained_query_token_count: 11,
      ignored_special_token_count: 10,
      grid: {
        columns: 2,
        rows: 2,
        patch_count: 4,
        coordinate_space: "normalized_unmodified_page",
      },
      aggregation: {
        similarity: "cosine_dot_product_on_l2_normalized_vectors",
        concept: "maximum_over_subword_query_tokens",
        patch: "maximum_over_visible_non_stopword_query_concepts",
        normalization: "clamp((value-page_median)/(page_p98-page_median),0,1)",
        spatial_mapping:
          "ColPali Engine local image-token mask with global image tokens excluded",
      },
      concepts: [
        {
          label: "stakeholders",
          query_token_indices: [2],
          peak_similarity: 0.82,
          peak_patch_index: 1,
        },
      ],
      token_score_bounds: { min: 0.2, max: 0.9 },
      tokens: [
        {
          index: 0,
          text: "Example",
          char_start: 0,
          char_end: 7,
          is_stopword: false,
          max_similarity: 0.5,
          peak_patch_index: 0,
          intensities: [0.0, 0.4],
        },
        {
          index: 1,
          text: " of",
          char_start: 7,
          char_end: 10,
          is_stopword: true,
          max_similarity: 0.1,
          peak_patch_index: 0,
          intensities: [0.2, 0.0],
        },
        {
          index: 2,
          text: "stakeholders",
          char_start: 11,
          char_end: 22,
          is_stopword: false,
          max_similarity: 0.82,
          peak_patch_index: 1,
          intensities: [0.1, 1.0],
        },
      ],
      patches: [
        {
          patch_index: 0,
          model_sequence_index: 10,
          grid_column: 0,
          grid_row: 0,
          x: 0,
          y: 0,
          width: 0.5,
          height: 0.5,
          raw_similarity: 0.4,
          intensity: 0,
        },
        {
          patch_index: 1,
          model_sequence_index: 11,
          grid_column: 1,
          grid_row: 0,
          x: 0.5,
          y: 0,
          width: 0.5,
          height: 0.5,
          raw_similarity: 0.82,
          intensity: 1,
        },
      ],
    },
  ],
  results: [
    {
      page_id: "nasa-seh-printed-053",
      pdf_page_index: 66,
      printed_page: 53,
      title: "Stakeholder Expectations Definition Process",
      section: "4.1 Stakeholder Expectations Definition",
      document_identifier: "NASA/SP-2016-6105 Rev 2",
      ntrs_id: 20170001761,
      ntrs_record_url: "https://ntrs.nasa.gov/citations/20170001761",
      distribution: "PUBLIC",
      rights_determination: "PUBLIC_USE_PERMITTED",
      contains_third_party_material: false,
      score: 18.624748,
      local_score: 18.624746,
      score_delta: 0.000002,
      milvus_rank: 1,
      local_rank: 1,
      image_width: 1224,
      image_height: 1584,
      evidence_label: "Unmodified 144-DPI render from the official NTRS PDF",
    },
  ],
};

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("EmbeddingListDemoPage", () => {
  it("renders eight question presets with the visual query first and shows page evidence", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((input: RequestInfo | URL) => {
        const payload = String(input).endsWith("/status") ? status : search;
        return Promise.resolve(
          new Response(JSON.stringify(payload), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          }),
        );
      }),
    );

    render(<EmbeddingListDemoPage />);

    expect(embeddingListDemo.status).toBe("implemented");
    expect((screen.getByLabelText("Query") as HTMLInputElement).value).toBe(
      DEFAULT_QUERY,
    );
    await screen.findByTestId("embedding-result");
    const toggle = screen.getByRole("button", { name: "Choose a question" });
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    fireEvent.click(toggle);
    const options = screen.getAllByRole("option") as HTMLButtonElement[];
    expect(options).toHaveLength(8);
    expect(options[0].textContent).toBe(QUERY_OPTIONS[0].text);
    expect(options[1].textContent).toBe(QUERY_OPTIONS[1].text);
    expect(DEFAULT_QUERY).toBe(QUERY_OPTIONS[0].text);
    expect(document.querySelector(".embedding-source-details")).toBeNull();
    expect(screen.queryByText("CPU index ready")).toBeNull();

    const card = await screen.findByTestId("embedding-result");
    expect(card.getAttribute("data-page-id")).toBe("nasa-seh-printed-053");
    expect(card.textContent).toContain("printed 53 · PDF index 66");
    expect(card.textContent).toContain(
      "Stakeholder Expectations Definition Process",
    );
    expect(card.textContent).toContain("MAX_SIM");
    expect(screen.queryByTestId("score-verification")).toBeNull();
    expect(screen.queryByText(/expects page/)).toBeNull();
    expect(screen.queryByText(/hard negative/i)).toBeNull();
    expect(screen.getAllByRole("img")[0].getAttribute("src")).toBe(
      "/api/v1/pages/nasa-seh-printed-053",
    );
    const heatmap = screen.getByTestId("local-colsmol-heatmap");
    expect(heatmap.tagName).toBe("CANVAS");
    expect(heatmap.getAttribute("data-page-id")).toBe("nasa-seh-printed-053");
    expect(heatmap.getAttribute("data-patch-count")).toBe("4");
    expect(heatmap.getAttribute("width")).toBe("1224");
    expect(heatmap.getAttribute("height")).toBe("1584");
    const canvas = document.querySelector(".embedding-page-canvas");
    expect(canvas?.getAttribute("data-image-width")).toBe("1224");
    expect(canvas?.getAttribute("data-image-height")).toBe("1584");
    expect(canvas?.getAttribute("style")).toContain(
      "--page-aspect-ratio: 0.7727272727272727",
    );
    fireEvent.change(screen.getByLabelText("Query"), {
      target: { value: "A different question" },
    });
    expect(screen.queryByTestId("local-colsmol-heatmap")).toBeNull();
    expect(screen.queryByTestId("embedding-result")).toBeNull();
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
  });

  it("does not enable search before the CPU index is prepared", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            ...status,
            status: "not_prepared",
            runtime_ready: false,
            milvus: { ...status.milvus, target_exists: false },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      ),
    );

    render(<EmbeddingListDemoPage />);

    expect(
      await screen.findByRole("button", { name: "Prepare index" }),
    ).toBeTruthy();
    expect(
      (
        screen.getByRole("button", {
          name: "Search 40 pages",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
    expect(screen.queryByText("Ready")).toBeNull();
    expect(screen.queryByText("Implemented")).toBeNull();
  });

  it("clears a failed search and succeeds when the user retries", async () => {
    let searchAttempts = 0;
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
          if (String(input).endsWith("/status")) {
            return Promise.resolve(
              new Response(JSON.stringify(status), {
                status: 200,
                headers: { "Content-Type": "application/json" },
              }),
            );
          }
          if (init?.method === "POST") {
            searchAttempts += 1;
            if (searchAttempts === 1) {
              return Promise.resolve(
                new Response(
                  JSON.stringify({
                    detail: [
                      {
                        type: "string_too_long",
                        loc: ["body", "query"],
                        msg: "String should have at most 512 characters",
                      },
                    ],
                  }),
                  {
                    status: 422,
                    headers: { "Content-Type": "application/json" },
                  },
                ),
              );
            }
          }
          return Promise.resolve(
            new Response(JSON.stringify(search), {
              status: 200,
              headers: { "Content-Type": "application/json" },
            }),
          );
        }),
    );

    render(<EmbeddingListDemoPage />);

    expect((await screen.findByRole("alert")).textContent).toContain(
      "String should have at most 512 characters",
    );
    expect(screen.queryByTestId("embedding-result")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Search 40 pages" }));
    expect(
      (await screen.findByTestId("embedding-result")).getAttribute(
        "data-page-id",
      ),
    ).toBe("nasa-seh-printed-053");
    expect(screen.queryByRole("alert")).toBeNull();
    expect(searchAttempts).toBe(2);
  });

  it("keeps implemented status honest when the backend is offline", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));

    render(<EmbeddingListDemoPage />);

    // The status/ready badges are intentionally not shown to users; the only
    // visible signal when offline is the disabled search button.
    expect(
      await screen.findByText("Visual PDF Retrieval with EmbeddingList"),
    ).toBeTruthy();
    expect(screen.queryByText("Implemented")).toBeNull();
    expect(screen.queryByText("Ready")).toBeNull();
    expect(screen.queryByText("Backend offline")).toBeNull();
    expect(
      (
        screen.getByRole("button", {
          name: "Search 40 pages",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
  });

  it("switches the heatmap to a single token's page on hover", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((input: RequestInfo | URL) => {
        const payload = String(input).endsWith("/status") ? status : search;
        return Promise.resolve(
          new Response(JSON.stringify(payload), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          }),
        );
      }),
    );

    render(<EmbeddingListDemoPage />);

    await screen.findByTestId("embedding-result");
    const tokenCell = screen.getByRole("button", {
      name: /stakeholders 0\.820/,
    });
    const heatmap = screen.getByTestId("local-colsmol-heatmap");

    fireEvent.mouseEnter(tokenCell);
    expect(tokenCell.classList.contains("is-active")).toBe(true);
    // The single-patch overlay is gone; the same canvas now renders this
    // token's full-page heatmap.
    expect(screen.queryByTestId("token-peak-overlay")).toBeNull();
    expect(heatmap.getAttribute("data-page-id")).toBe("nasa-seh-printed-053");

    fireEvent.mouseLeave(tokenCell);
    expect(tokenCell.classList.contains("is-active")).toBe(false);
    expect(screen.queryByTestId("token-peak-overlay")).toBeNull();
  });

  it("renders every subword token as a contribution cell", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((input: RequestInfo | URL) => {
        const payload = String(input).endsWith("/status") ? status : search;
        return Promise.resolve(
          new Response(JSON.stringify(payload), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          }),
        );
      }),
    );

    render(<EmbeddingListDemoPage />);

    await screen.findByTestId("embedding-result");
    const strip = screen.getByLabelText("Query token contribution");
    const cells = strip.querySelectorAll(".embedding-token-cell");
    expect(cells.length).toBe(3);
    expect(cells[0].textContent).toContain("Example");
    expect(cells[1].textContent).toContain("of");
    expect(cells[1].classList.contains("is-stopword")).toBe(true);
    expect(cells[2].classList.contains("is-stopword")).toBe(false);
  });
});
