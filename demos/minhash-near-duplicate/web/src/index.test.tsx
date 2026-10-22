import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  MinHashDemoPage,
  runMinHashSearch,
  type SearchResponse,
} from "./index";

afterEach(() => {
  cleanup();
});

describe("runMinHashSearch", () => {
  it("posts the query, threshold, and limit and parses the response", async () => {
    const response: SearchResponse = {
      status: "passed",
      execution_path: "milvus_minhash_lsh_near_duplicate",
      query: "On Sunday, we followed the river path to the old bridge.",
      collection_name: "milvus3_demos_minhash_near_duplicate",
      index_name: "milvus3_demos_minhash_signature",
      text_field: "content",
      signature_field: "minhash_signature",
      metric_type: "MHJACCARD",
      threshold: 0.85,
      limit: 10,
      above_threshold_count: 2,
      results: [
        {
          doc_id: "river-weekend-003.txt",
          title: "A Weekend by the River",
          content: "On Sunday, we followed the river path to the old bridge.",
          estimated_jaccard: 0.94,
          raw_distance: 0.94,
          above_threshold: true,
        },
        {
          doc_id: "river-weekend-001.txt",
          title: "A Weekend by the River",
          content: "On Saturday, we followed the river path to the red bridge.",
          estimated_jaccard: 0.87,
          raw_distance: 0.87,
          above_threshold: true,
        },
        {
          doc_id: "city-walk-073.txt",
          title: "An Evening Walk Through the City Center",
          content:
            "An evening walk through the city center passed the shuttered bank.",
          estimated_jaccard: 0.43,
          raw_distance: 0.43,
          above_threshold: false,
        },
      ],
      diffs: [
        {
          document_id: "river-weekend-003.txt",
          removed_count: 2,
          added_count: 2,
          segments: [
            { kind: "equal", text: "On" },
            { kind: "removed", text: "Saturday" },
            { kind: "added", text: "Sunday" },
          ],
        },
        {
          document_id: "river-weekend-001.txt",
          removed_count: 2,
          added_count: 2,
          segments: [],
        },
        {
          document_id: "city-walk-073.txt",
          removed_count: 0,
          added_count: 0,
          segments: [],
        },
      ],
      semantics: "Estimated Jaccard similarity between MinHash signatures.",
      latency_ms: 12.3,
    };

    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify(response), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const result = await runMinHashSearch({
      query: "hello",
      threshold: 0.85,
      limit: 10,
    });
    expect(result.status).toBe("passed");
    expect(result.above_threshold_count).toBe(2);
    expect(result.results[0].estimated_jaccard).toBe(0.94);

    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain("/api/v1/search");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(String(init?.body))).toEqual({
      query: "hello",
      threshold: 0.85,
      limit: 10,
    });

    fetchMock.mockRestore();
  });
});

describe("MinHashDemoPage", () => {
  it("renders the topbar title and primary action", async () => {
    const statusPayload = {
      status: "not_prepared",
      implementation_status: "implemented",
      execution_path: "milvus_minhash_lsh_near_duplicate",
      capability: "MinHash Function + MINHASH_LSH",
      collection_name: "milvus3_demos_minhash_near_duplicate",
      index_name: "milvus3_demos_minhash_signature",
      text_field: "content",
      signature_field: "minhash_signature",
      metric_type: "MHJACCARD",
      dataset: {
        dataset_id: "original-short-stories-r1",
        dataset_version: 1,
        document_count: 36,
        documents_sha256: "d",
        presets_sha256: "p",
      },
      query_presets: [
        {
          id: "river-weekend",
          label: "A weekend by the river",
          title: "A Weekend by the River",
          text: "On Sunday, we followed the river path to the old bridge.",
          description: "A lightly edited copy.",
        },
      ],
      milvus: {
        milvus_uri: "http://127.0.0.1:49530",
        server_version: "3.0.0",
        expected_server_version: "3.0.0",
        target_collection: "milvus3_demos_minhash_near_duplicate",
        target_exists: false,
      },
      last_prepare: null,
    };

    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify(statusPayload), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    render(<MinHashDemoPage />);

    await waitFor(() => {
      expect(
        screen.getByText("Near-Duplicate Detection with MinHash"),
      ).toBeTruthy();
    });
    expect(screen.getByText("Detect duplicates")).toBeTruthy();
    expect(
      screen.getByText("Find copied and lightly edited documents."),
    ).toBeTruthy();

    fetchMock.mockRestore();
  });

  it("reveals the MinHash code panel on hover", async () => {
    const statusPayload = {
      status: "not_prepared",
      implementation_status: "implemented",
      execution_path: "milvus_minhash_lsh_near_duplicate",
      capability: "MinHash Function + MINHASH_LSH",
      collection_name: "milvus3_demos_minhash_near_duplicate",
      index_name: "milvus3_demos_minhash_signature",
      text_field: "content",
      signature_field: "minhash_signature",
      metric_type: "MHJACCARD",
      dataset: {
        dataset_id: "original-short-stories-r1",
        dataset_version: 1,
        document_count: 36,
        documents_sha256: "d",
        presets_sha256: "p",
      },
      query_presets: [
        {
          id: "river-weekend",
          label: "A weekend by the river",
          title: "A Weekend by the River",
          text: "On Sunday, we followed the river path to the old bridge.",
          description: "A lightly edited copy.",
        },
      ],
      milvus: {
        milvus_uri: "http://127.0.0.1:49530",
        server_version: "3.0.0",
        expected_server_version: "3.0.0",
        target_collection: "milvus3_demos_minhash_near_duplicate",
        target_exists: false,
      },
      last_prepare: null,
    };

    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify(statusPayload), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    render(<MinHashDemoPage />);

    await waitFor(() => {
      expect(
        screen.getByText("How MinHash finds near-duplicates"),
      ).toBeTruthy();
    });

    const minhashStage = screen.getByText("MinHash");
    fireEvent.mouseEnter(minhashStage.closest(".pipeline-stage")!);

    await waitFor(() => {
      expect(screen.getByText(/BINARY_VECTOR/)).toBeTruthy();
    });

    fetchMock.mockRestore();
  });
});
