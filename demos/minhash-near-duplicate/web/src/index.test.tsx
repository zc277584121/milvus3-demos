import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  MinHashDemoPage,
  runMinHashSearch,
  type DiffSegment,
  type SearchResponse,
} from "./index";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
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
  it("applies editor drafts, keeps failed drafts, and follows preset selection", async () => {
    const presets = [
      {
        id: "river",
        label: "River",
        text: "We followed the river to the bridge.",
      },
      {
        id: "market",
        label: "Market",
        text: "The night market opened under yellow lanterns.",
      },
    ];
    const requests: string[] = [];
    let failNextSearch = false;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (url, init) => {
      if (String(url).endsWith("/status")) {
        return Response.json({ query_presets: presets });
      }
      const { query } = JSON.parse(String(init?.body)) as { query: string };
      requests.push(query);
      if (failNextSearch) {
        failNextSearch = false;
        return Response.json(
          { detail: "Search temporarily unavailable." },
          { status: 503 },
        );
      }
      return Response.json({
        query: query.trim(),
        results: [
          {
            doc_id: "matching-document.txt",
            title: "Matching document",
            content: query.trim(),
            estimated_jaccard: 1,
          },
        ],
        diffs: [
          {
            document_id: "matching-document.txt",
            added_count: 0,
            removed_count: 0,
            segments: [{ kind: "equal", text: query.trim() }],
          },
        ],
      });
    });

    render(<MinHashDemoPage />);
    await screen.findByText(/Estimated Jaccard/);
    const comparison = within(
      screen.getByRole("region", { name: "What changed" }),
    );
    const edit = () =>
      fireEvent.click(
        comparison.getByRole("button", { name: "Edit incoming document" }),
      );
    const change = (value: string) =>
      fireEvent.change(
        comparison.getByRole("textbox", { name: "Incoming document" }),
        { target: { value } },
      );
    const incomingText = () =>
      comparison.getAllByRole("article")[0].querySelector(".document-text")
        ?.textContent;

    edit();
    const revised = "  We followed the river to a new bridge.\n";
    change(revised);
    expect(requests).toEqual([presets[0].text]);
    fireEvent.click(comparison.getByRole("button", { name: "Apply changes" }));
    await waitFor(() => expect(comparison.queryByRole("textbox")).toBeNull());
    expect(requests.at(-1)).toBe(revised);
    expect(incomingText()).toBe(revised);
    expect(comparison.getByText(/Estimated Jaccard/)).toBeTruthy();

    edit();
    change("A draft that should survive a failed search.");
    failNextSearch = true;
    fireEvent.click(comparison.getByRole("button", { name: "Apply changes" }));
    await screen.findByRole("alert");
    expect((comparison.getByRole("textbox") as HTMLTextAreaElement).value).toBe(
      "A draft that should survive a failed search.",
    );
    expect(comparison.queryByText(/Estimated Jaccard/)).toBeNull();
    fireEvent.click(comparison.getByRole("button", { name: "Cancel" }));
    expect(incomingText()).toBe(revised);
    expect(comparison.getByText(/Estimated Jaccard/)).toBeTruthy();

    edit();
    change(" \n\t");
    expect(
      (
        comparison.getByRole("button", {
          name: "Apply changes",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);

    change("A second revision applied from the editor.");
    fireEvent.click(comparison.getByRole("button", { name: "Apply changes" }));
    await waitFor(() => expect(comparison.queryByRole("textbox")).toBeNull());
    expect(incomingText()).toBe("A second revision applied from the editor.");
    expect(requests.at(-1)).toBe("A second revision applied from the editor.");

    edit();
    change("Unsubmitted text.");
    fireEvent.change(screen.getByRole("combobox", { name: "Example case" }), {
      target: { value: "market" },
    });
    await waitFor(() =>
      expect(comparison.getByText(/Estimated Jaccard/)).toBeTruthy(),
    );
    expect(comparison.queryByRole("textbox")).toBeNull();
    expect(incomingText()).toBe(presets[1].text);
    expect(requests.at(-1)).toBe(presets[1].text);
    expect(comparison.getByText(/Estimated Jaccard/)).toBeTruthy();
  });

  it("renders the demo heading, walkthrough, and primary action", async () => {
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
        screen.getByRole("heading", {
          name: "Near-duplicate detection with MinHash",
        }),
      ).toBeTruthy();
    });
    await screen.findByText("0 of 0 above threshold");
    // The walkthrough plays inline, so the player is on the page from the start.
    expect(
      screen.getByLabelText("MinHash walkthrough", { selector: "video" }),
    ).toBeTruthy();
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(screen.getByRole("heading", { name: "What changed" })).toBeTruthy();
    expect(
      screen.getByRole("heading", { name: "Candidate documents" }),
    ).toBeTruthy();
    expect(screen.queryByRole("textbox")).toBeNull();
    const controls = within(
      screen.getByRole("form", { name: "Document controls" }),
    );
    expect(
      controls.getByRole("combobox", { name: "Example case" }),
    ).toBeTruthy();
    fireEvent.click(
      screen.getByRole("button", { name: "Edit incoming document" }),
    );
    expect(
      within(screen.getByRole("region", { name: "What changed" })).getByRole(
        "textbox",
        { name: "Incoming document" },
      ),
    ).toBeTruthy();
    expect(controls.queryByRole("textbox")).toBeNull();

    fetchMock.mockRestore();
  });

  it.each([
    {
      name: "unchanged text",
      query: "On Sunday, we crossed the bridge.",
      document: "On Sunday, we crossed the bridge.",
      segments: [{ kind: "equal", text: "On Sunday, we crossed the bridge." }],
      added: [],
      removed: [],
    },
    {
      name: "insertions at both ends",
      query: "Fresh apples are red. Today.",
      document: "apples are red.",
      segments: [
        { kind: "added", text: "Fresh" },
        { kind: "equal", text: "apples are red." },
        { kind: "added", text: "Today." },
      ],
      added: ["Fresh", "Today."],
      removed: [],
    },
    {
      name: "deletions at both ends",
      query: "apples are red.",
      document: "Fresh apples are red. Today.",
      segments: [
        { kind: "removed", text: "Fresh" },
        { kind: "equal", text: "apples are red." },
        { kind: "removed", text: "Today." },
      ],
      added: [],
      removed: ["Fresh", "Today."],
    },
    {
      name: "replacements with original paragraph spacing",
      query: "\tOn Sunday,\n\nwe crossed  the old bridge.  ",
      document: "On Saturday,\nwe crossed the stone bridge.\n",
      segments: [
        { kind: "equal", text: "On" },
        { kind: "removed", text: "Saturday," },
        { kind: "added", text: "Sunday," },
        { kind: "equal", text: "we crossed the" },
        { kind: "removed", text: "stone" },
        { kind: "added", text: "old" },
        { kind: "equal", text: "bridge." },
      ],
      added: ["Sunday,", "old"],
      removed: ["Saturday,", "stone"],
    },
  ] satisfies {
    name: string;
    query: string;
    document: string;
    segments: DiffSegment[];
    added: string[];
    removed: string[];
  }[])(
    "loads the first preset and displays $name inline",
    async ({ query, document, segments, added, removed }) => {
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
            text: query,
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
            doc_id: "river-weekend-003.txt",
            title: "A Weekend by the River",
            content: document,
            estimated_jaccard: 0.94,
            raw_distance: 0.94,
            above_threshold: true,
          },
        ],
        diffs: [
          {
            document_id: "river-weekend-003.txt",
            removed_count: removed.length,
            added_count: added.length,
            segments,
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
      expect(screen.getByText("1 of 1 above threshold")).toBeTruthy();
      const comparison = within(
        screen.getByRole("region", { name: "What changed" }),
      );
      const documents = comparison.getAllByRole("article");
      const incoming = documents[0].querySelector(".document-text")!;
      const existing = documents[1].querySelector(".document-text")!;
      expect(incoming.textContent).toBe(query);
      expect(existing.textContent).toBe(document);
      expect(
        Array.from(incoming.querySelectorAll("ins"), (el) => el.textContent),
      ).toEqual(added);
      expect(
        Array.from(existing.querySelectorAll("del"), (el) => el.textContent),
      ).toEqual(removed);
      expect(incoming.querySelectorAll("del")).toHaveLength(0);
      expect(existing.querySelectorAll("ins")).toHaveLength(0);
      expect(documents[0].querySelectorAll("p")).toHaveLength(1);
      expect(documents[1].querySelectorAll("p")).toHaveLength(1);
      fireEvent.click(
        comparison.getByRole("button", { name: "Edit incoming document" }),
      );
      fireEvent.change(
        screen.getByRole("textbox", { name: "Incoming document" }),
        {
          target: { value: "An unsent edit" },
        },
      );
      expect(comparison.queryByText(/Estimated Jaccard/)).toBeNull();
      expect(
        screen.queryByRole("button", { name: /river-weekend-003.txt/ }),
      ).toBeNull();
      expect(
        comparison.getByText("Apply changes to update the comparison."),
      ).toBeTruthy();
      fireEvent.click(comparison.getByRole("button", { name: "Cancel" }));
      expect(documents[0].querySelector(".document-text")!.textContent).toBe(
        query,
      );
      expect(comparison.getByText(/Estimated Jaccard/)).toBeTruthy();
      expect(
        comparison.queryByRole("button", { name: "Show full text" }),
      ).toBeNull();
      expect(fetchMock).toHaveBeenCalledTimes(2);
      const [searchUrl, searchInit] = fetchMock.mock.calls[1];
      expect(String(searchUrl)).toContain("/api/v1/search");
      expect(searchInit?.method).toBe("POST");

      fetchMock.mockRestore();
    },
  );
});
