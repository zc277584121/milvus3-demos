import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  StructArrayDemoPage,
  type SearchResponse,
  type StructArrayStatus,
} from "./index";

const status: StructArrayStatus = {
  status: "ready",
  sample_size: 30,
  model: {
    model_id: "gpahal/bge-m3-onnx-int8",
    revision: "2b34e84df040034d4b9eabb62383a87c18955822",
    vector_dimension: 1024,
    dense_output_name: "dense_vecs",
    device: "cpu",
    loaded: true,
    cache_available: true,
  },
  dataset: {
    dataset_id: "synthetic-driving-scenes-r1",
    dataset_version: 1,
    video_count: 30,
    observation_count: 540,
    evidence_frame_count: 30,
    manifest_sha256: "abcd",
  },
  milvus: {
    milvus_uri: "http://127.0.0.1:49530",
    server_version: "3.0.0",
    collection_name: "milvus3_demos_structarray_hybrid_synthetic",
    collection_exists: true,
    raw_collection_names: ["milvus3_demos_structarray_hybrid_synthetic"],
    row_count: 30,
    index_names: ["i1", "i2"],
  },
  query_presets: [
    {
      id: "white-truck-intersection",
      text: "a white truck waiting at an intersection",
      scene_terms: ["intersection"],
      object_terms: ["truck"],
      color_terms: ["white"],
    },
  ],
};

const searchResponse: SearchResponse = {
  status: "passed",
  query: "a white truck waiting at an intersection",
  query_vector_dimension: 1024,
  limit: 8,
  parent_weight: 0.5,
  child_weight: 0.5,
  collapse_strategy: "topk_sum",
  collapse_topk: 3,
  latency_ms: 120,
  score_semantics: "COSINE similarity, not a probability",
  ground_truth: {
    scene_terms: ["intersection"],
    object_terms: ["truck"],
    color_terms: ["white"],
    matched_video_ids: ["001c2d37", "005b7275", "0071e746"],
    matched_count: 3,
  },
  path_recall: {
    parent: { matched: 3, gt_count: 3, recall: 1, ndcg: 0.9469 },
    child: { matched: 2, gt_count: 3, recall: 0.6667, ndcg: 0.8316 },
    fusion: { matched: 3, gt_count: 3, recall: 1, ndcg: 0.9675 },
  },
  paths: {
    parent: {
      anns_field: "summary_vector",
      label: "Parent-only semantic search",
      results: [
        {
          rank: 1,
          score: 0.91,
          video_id: "synthetic-drive-001",
          video_summary:
            "Driving environment: Road type: intersection, Weather: rainy || Driving behaviors: Segment 1 (lane_keep): Driving straight through an intersection. || Detected objects: 1 car, 2 truck",
          source_ordinal: 0,
          scene_match: true,
          object_match: false,
          color_match: false,
          actual_scene: "intersection",
          actual_object: "car",
          actual_color: "black",
          preview: {
            frame_id: 240,
            annotated_frame: "synthetic-drive-001.jpg",
            raw_frame: "synthetic-drive-001.jpg",
          },
        },
      ],
    },
    child: {
      anns_field: "observations[description_vector]",
      group_by_field: "video_id",
      label: "Child-only semantic search, grouped by parent",
      results: [
        {
          rank: 1,
          score: 0.82,
          video_id: "synthetic-drive-001",
          video_summary: "A rainy drive through an intersection.",
          source_ordinal: 0,
          offset: 3,
          scene_match: false,
          object_match: true,
          color_match: true,
          actual_scene: "local_residential",
          actual_object: "truck",
          actual_color: "white",
          observation: {
            description: "A white truck stopped at the intersection.",
            object_type: "truck",
            frame_id: 240,
            image_id: "f",
            bbox: [10, 20, 30, 40],
            vehicle_type: "truck",
            color: "white",
            orientation: "front",
            lights_on: "no",
            v_ego: 13.0,
            a_ego: 0.15,
            clip_id: "synthetic-drive-001-clip-1",
            raw_frame: "synthetic-drive-001.jpg",
            annotated_frame: "synthetic-drive-001.jpg",
          },
        },
      ],
    },
    fusion: {
      label: "Parent + child hybrid with collapse and weighted rerank",
      ranker: "WeightedRanker",
      parent_anns_field: "summary_vector",
      child_anns_field: "observations[description_vector]",
      results: [
        {
          rank: 1,
          score: 0.87,
          video_id: "synthetic-drive-001",
          video_summary:
            "Driving environment: Road type: intersection, Weather: rainy || Driving behaviors: Segment 1 (lane_keep): Driving straight through an intersection. || Detected objects: 1 car, 2 truck",
          source_ordinal: 0,
          scene_match: true,
          object_match: true,
          color_match: true,
          actual_scene: "intersection",
          actual_object: "truck",
          actual_color: "white",
          preview: {
            frame_id: 240,
            annotated_frame: "synthetic-drive-001.jpg",
            raw_frame: "synthetic-drive-001.jpg",
          },
        },
      ],
    },
  },
};

async function jsonResponse(body: unknown): Promise<Response> {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

describe("StructArrayDemoPage", () => {
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("shows the three result columns after a search", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith("/status")) return jsonResponse(status);
        if (url.endsWith("/search")) return jsonResponse(searchResponse);
        return jsonResponse({});
      });

    render(<StructArrayDemoPage />);

    // The first preset query auto-runs once the backend reports ready.
    await waitFor(() => {
      expect(screen.getAllByText(/^NDCG /).length).toBe(3);
    });

    // Each column reports an NDCG score, and recall shows GT matches.
    const ndcgChips = screen.getAllByText(/^NDCG /);
    expect(ndcgChips.length).toBe(3);
    expect(screen.getAllByText(/Recall 3\/3/).length).toBeGreaterThan(0);
    expect(screen.getByText(/Recall 2\/3/)).toBeTruthy();

    // Each column carries its own one-line plan caption (Plan 1/2/3) right
    // above the first result, instead of a separate explainer band.
    const planCaps = document.querySelectorAll(".column-plan");
    expect(planCaps.length).toBe(3);
    expect(planCaps[0].textContent).toContain("summary_vector");
    expect(planCaps[1].textContent).toContain("description_vector");
    expect(planCaps[2].textContent).toContain("topk_sum(3)");
    expect(planCaps[2].textContent).toContain("WeightedRanker");

    // No code is shown until the user hovers a plan caption.
    expect(document.querySelector(".column-plan-row.is-active")).toBeFalsy();

    // Hovering the fusion caption reveals its Python snippet in a floating
    // overlay anchored to the caption.
    fireEvent.mouseEnter(planCaps[2].parentElement as Element);
    expect(document.querySelector(".column-plan-row.is-active")).toBeTruthy();
    expect(
      document.querySelector(".column-plan-row.is-active .pipeline-code")
        ?.textContent,
    ).toContain("hybrid_search");
    expect(
      document.querySelector(".column-plan-row.is-active .pipeline-code")
        ?.textContent,
    ).toContain("WeightedRanker");

    // Exactly the fusion card (scene ∧ object ∧ color) gets the GT highlight outline.
    const gtCards = document.querySelectorAll(".result-card--gt");
    expect(gtCards.length).toBe(1);
    expect(gtCards[0].classList.contains("result-card--fusion")).toBe(true);
    expect(gtCards[0].querySelector(".gt-chip")?.textContent).toBe("GT");

    // Mismatched detected values (object "car" ≠ truck) are struck through in
    // the parent card's summary, while matching intent terms stay highlighted.
    const missMarks = document.querySelectorAll(
      ".result-card--parent .gt-miss",
    );
    expect(missMarks.length).toBeGreaterThan(0);
    expect(missMarks[0].textContent?.toLowerCase()).toBe("car");
    const hitMarks = document.querySelectorAll(".result-card--parent .gt-term");
    expect(
      Array.from(hitMarks).some((mark) => mark.textContent === "truck"),
    ).toBe(true);

    // Every card carries a detected-values row; the parent card's contradicting
    // object/color values are struck there even when the summary text omits them.
    const parentDetected = document.querySelector(
      ".result-card--parent .card-detected",
    );
    expect(parentDetected).toBeTruthy();
    const parentMissValues = parentDetected!.querySelectorAll(
      ".card-detected-value.gt-miss",
    );
    expect(parentMissValues.length).toBeGreaterThan(0);
    expect(
      Array.from(parentMissValues).some(
        (el) => el.textContent === "car" || el.textContent === "black",
      ),
    ).toBe(true);
    // The GT fusion card's detected values are highlighted, not struck.
    const fusionDetected = document.querySelector(
      ".result-card--fusion .card-detected",
    );
    expect(fusionDetected).toBeTruthy();
    expect(
      fusionDetected!.querySelectorAll(".card-detected-value.gt-term").length,
    ).toBe(3);

    fetchMock.mockRestore();
  });

  it("renders the demo title once the status loads", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
      jsonResponse(status),
    );
    render(<StructArrayDemoPage />);
    await waitFor(() => {
      expect(
        screen.getByRole("heading", {
          name: "Data Curation in Autonomous Driving with StructArray",
        }),
      ).toBeTruthy();
    });
  });

  it("shows the schema table and switches pipeline detail on hover", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
      jsonResponse(status),
    );
    render(<StructArrayDemoPage />);

    // The data model is a real table whose header lists every column + type.
    await screen.findByText(/Collection schema/);
    const headerNames = Array.from(
      document.querySelectorAll(".schema-table thead th code"),
    ).map((node) => node.textContent);
    expect(headerNames).toEqual(
      expect.arrayContaining([
        "video_id",
        "video_summary",
        "source_ordinal",
        "summary_vector",
        "observations",
      ]),
    );
    // Example rows show sample values plus an ellipsis for the rest.
    expect(screen.getByText("video_0001")).toBeTruthy();
    expect(screen.getByText(/more videos/i)).toBeTruthy();
    // Vector cells show sample floats, not a bare placeholder.
    const vectorTexts = Array.from(
      document.querySelectorAll(".td-vec code"),
    ).map((node) => node.textContent ?? "");
    expect(vectorTexts.some((text) => /0\.0023/.test(text))).toBe(true);

    // Hovering the observations cell floats a nested StructArray panel above
    // the parent table, instead of inserting rows into the parent grid.
    fireEvent.mouseEnter(screen.getByRole("button", { name: /12 objects/ }));
    expect(screen.getByRole("dialog")).toBeTruthy();
    expect(screen.getByText("description_vector")).toBeTruthy();
  });

  it("merges the preset picker into the query field and drops the sliders", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
      jsonResponse(status),
    );
    render(<StructArrayDemoPage />);

    // One editable box holds the query text plus the preset ▾ and clear (×).
    await screen.findByLabelText("Query");
    const input = screen.getByLabelText("Query") as HTMLInputElement;
    // The first preset auto-runs, so the box reflects that preset text.
    expect(input.value).toBe("a white truck waiting at an intersection");
    const box = document.querySelector(".sa-combobox");
    expect(box?.querySelector("#hybrid-query")).toBeTruthy();
    expect(box?.querySelector(".sa-combobox-toggle")).toBeTruthy();
    expect(box?.querySelector(".sa-combobox-clear")).toBeTruthy();

    // Opening the ▾ reveals a listbox of preset queries.
    fireEvent.click(
      screen.getByRole("button", { name: "Choose a preset query" }),
    );
    const listbox = screen.getByRole("listbox", { name: "Query presets" });
    expect(listbox).toBeTruthy();
    const options = screen.getAllByRole("option");
    expect(
      options.some((option) => option.textContent?.includes("white truck")),
    ).toBe(true);

    // The Results and Fusion weight sliders are gone.
    expect(screen.queryByLabelText(/Fusion weight/)).toBeNull();
    expect(screen.queryByLabelText(/^Results/)).toBeNull();
  });
});
