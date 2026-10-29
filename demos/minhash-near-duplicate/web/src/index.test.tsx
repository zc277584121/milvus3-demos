import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  buildBands,
  MinHashDemoPage,
  runMinHashSearch,
  runTrial,
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

    const emptySearchPayload = {
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
      above_threshold_count: 0,
      results: [],
      diffs: [],
      semantics: "Estimated Jaccard similarity between MinHash signatures.",
      latency_ms: 8.4,
    };

    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        new Response(JSON.stringify(statusPayload), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify(emptySearchPayload), {
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

  it("runs the first preset automatically on load", async () => {
    const statusPayload = {
      status: "ready",
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
        target_exists: true,
      },
      last_prepare: null,
    };

    const searchPayload = {
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
      above_threshold_count: 1,
      results: [
        {
          doc_id: "river-weekend-003.txt",
          title: "A Weekend by the River",
          content: "On Sunday, we followed the river path to the old bridge.",
          estimated_jaccard: 0.94,
          raw_distance: 0.94,
          above_threshold: true,
        },
      ],
      diffs: [
        {
          document_id: "river-weekend-003.txt",
          removed_count: 0,
          added_count: 0,
          segments: [{ kind: "equal", text: "On Sunday" }],
        },
      ],
      semantics: "Estimated Jaccard similarity between MinHash signatures.",
      latency_ms: 9.1,
    };

    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        new Response(JSON.stringify(statusPayload), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify(searchPayload), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      );

    render(<MinHashDemoPage />);

    // No button click: the first preset runs on mount.
    await waitFor(() => {
      expect(
        screen.getAllByText("river-weekend-003.txt").length,
      ).toBeGreaterThan(0);
    });
    expect(screen.getByText("1 candidate above threshold")).toBeTruthy();
    expect(fetchMock).toHaveBeenCalledTimes(2);
    const [searchUrl, searchInit] = fetchMock.mock.calls[1];
    expect(String(searchUrl)).toContain("/api/v1/search");
    expect(searchInit?.method).toBe("POST");

    fetchMock.mockRestore();
  });
});

describe("MinHash slot and band mechanics", () => {
  it("keeps the same minimum word for identical sets", () => {
    const words = ["river", "path", "bridge"];
    const a = new Set(words);
    const trial = runTrial(a, a, words, 0);

    expect(trial.collision).toBe(true);
    expect(trial.queryPick?.hash).toBe(trial.candidatePick?.hash);
    // The pick is the smallest hash over the set for this slot.
    const smallest = Math.min(
      ...words.map((w) => trial.ordering.find((o) => o.word === w)!.hash),
    );
    expect(trial.queryPick?.hash).toBe(smallest);
  });

  it("orders the vocabulary by hash so the first match is the minimum", () => {
    const trial = runTrial(new Set(["a"]), new Set(["b"]), ["a", "b", "c"], 3);
    const hashes = trial.ordering.map((entry) => entry.hash);
    expect(hashes).toEqual([...hashes].sort((x, y) => x - y));
  });

  it("groups slots into bands and only a complete all-match band hits", () => {
    // 2 bands x 2 slots. Band 0 both match -> hit. Band 1 half match -> no hit.
    const bands = buildBands([true, true, true, false], 2, 2);

    expect(bands).toHaveLength(2);
    expect(bands[0]).toMatchObject({ complete: true, hit: true, filled: 2 });
    expect(bands[1]).toMatchObject({ complete: true, hit: false, filled: 2 });
  });

  it("reports partially filled bands as incomplete and never hitting", () => {
    const bands = buildBands([true, true, true], 2, 2);

    expect(bands[0]).toMatchObject({ complete: true, hit: true });
    expect(bands[1]).toMatchObject({ complete: false, hit: false, filled: 1 });
  });
  it("re-runs automatically when a different example is chosen", async () => {
    const statusPayload = {
      status: "ready",
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
        {
          id: "night-market",
          label: "The night market",
          title: "The Night Market",
          text: "The night market opened under strings of yellow lanterns.",
          description: "A copy with an added paragraph.",
        },
      ],
      milvus: {
        milvus_uri: "http://127.0.0.1:49530",
        server_version: "3.0.0",
        expected_server_version: "3.0.0",
        target_collection: "milvus3_demos_minhash_near_duplicate",
        target_exists: true,
      },
      last_prepare: null,
    };

    const buildSearchPayload = (docId: string, query: string) => ({
      status: "passed",
      execution_path: "milvus_minhash_lsh_near_duplicate",
      query,
      collection_name: "milvus3_demos_minhash_near_duplicate",
      index_name: "milvus3_demos_minhash_signature",
      text_field: "content",
      signature_field: "minhash_signature",
      metric_type: "MHJACCARD",
      threshold: 0.85,
      limit: 10,
      above_threshold_count: 1,
      results: [
        {
          doc_id: docId,
          title: "t",
          content: query,
          estimated_jaccard: 0.99,
          raw_distance: 0.99,
          above_threshold: true,
        },
      ],
      diffs: [],
      semantics: "Estimated Jaccard similarity between MinHash signatures.",
      latency_ms: 5,
    });

    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        new Response(JSON.stringify(statusPayload), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify(
            buildSearchPayload("first-hit.txt", "first query text"),
          ),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify(
            buildSearchPayload("second-hit.txt", "second query text"),
          ),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      );

    render(<MinHashDemoPage />);

    await waitFor(() => {
      expect(screen.getAllByText("first-hit.txt").length).toBeGreaterThan(0);
    });

    // Choosing another example triggers its own request — no button click.
    fireEvent.change(screen.getByRole("combobox"), {
      target: { value: "night-market" },
    });

    await waitFor(() => {
      expect(screen.getAllByText("second-hit.txt").length).toBeGreaterThan(0);
    });
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(
      JSON.parse(String(fetchMock.mock.calls[2][1]?.body)).query,
    ).toContain("night market");
  });
  it("re-judges candidates against the live threshold without a new request", async () => {
    const statusPayload = {
      status: "ready",
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
        target_exists: true,
      },
      last_prepare: null,
    };

    const searchPayload = {
      status: "passed",
      execution_path: "milvus_minhash_lsh_near_duplicate",
      query: "q",
      collection_name: "milvus3_demos_minhash_near_duplicate",
      index_name: "milvus3_demos_minhash_signature",
      text_field: "content",
      signature_field: "minhash_signature",
      metric_type: "MHJACCARD",
      threshold: 0.85,
      limit: 10,
      above_threshold_count: 1,
      results: [
        {
          doc_id: "high.txt",
          title: "High",
          content: "high similarity",
          estimated_jaccard: 0.9,
          raw_distance: 0.9,
          above_threshold: true,
        },
        {
          doc_id: "mid.txt",
          title: "Mid",
          content: "medium similarity",
          estimated_jaccard: 0.45,
          raw_distance: 0.45,
          above_threshold: false,
        },
      ],
      diffs: [],
      semantics: "Estimated Jaccard similarity between MinHash signatures.",
      latency_ms: 4,
    };

    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        new Response(JSON.stringify(statusPayload), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify(searchPayload), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      );

    render(<MinHashDemoPage />);

    await waitFor(() => {
      expect(screen.getByText("1 candidate above threshold")).toBeTruthy();
    });

    // Drag the slider down; the mid candidate must join without a new request.
    fireEvent.change(screen.getByLabelText(/threshold/i), {
      target: { value: "0.4" },
    });

    await waitFor(() => {
      expect(screen.getByText("2 candidates above threshold")).toBeTruthy();
    });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
