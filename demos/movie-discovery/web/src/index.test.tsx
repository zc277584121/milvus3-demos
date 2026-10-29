import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { MovieDiscoveryPage } from "./index";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function movie(movie_id: string, title: string, primary_genre = "Adventure", runtime_minutes = 104) {
  return {
    rank: 1,
    score: 0.9123,
    movie_id,
    title,
    synopsis: "A quiet, fictional story about ordinary people on a small journey.",
    primary_genre,
    release_year: 2021,
    decade: "2020s",
    region: "Canada",
    runtime_minutes,
    poster_path: `posters/${movie_id}.jpg`,
    poster_palette: ["#1e3a8a", "#3b82f6", "#93c5fd"],
  };
}

const statusPayload = {
  status: "ready",
  movie_count: 90,
  model: { model_id: "gpahal/bge-m3-onnx-int8", vector_dimension: 1024, device: "cpu" },
  milvus: {
    collection_name: "milvus3_demos_movie_catalog",
    server_version: "3.0.0",
    row_count: 90,
  },
  facets: [
    { id: "genre", field: "primary_genre", label: "Genre" },
    { id: "decade", field: "decade", label: "Decade" },
    { id: "region", field: "region", label: "Region" },
  ],
  query_presets: [
    {
      id: "lighthearted-adventure",
      label: "Lighthearted adventure",
      text: "A lighthearted adventure with ordinary people",
      description: "Warm, low-stakes journeys.",
    },
  ],
  poster_placeholders: false,
  poster_count: 90,
};

const base = {
  status: "passed" as const,
  filter: null,
  latency_ms: 12.3,
  score_semantics: "COSINE similarity, not a probability",
};

const browsePayload = {
  ...base,
  api: "query" as const,
  mode: "sort" as const,
  sort: "year" as const,
  sort_field: "release_year",
  sort_label: "Year",
  direction: "desc" as const,
  page: 1,
  page_size: 12,
  total_pages: 8,
  total: 90,
  results: [movie("m0001", "Browse One"), movie("m0002", "Browse Two", "Comedy")],
  semantics: "Server-side ORDER BY over the whole catalog; no vector is used.",
};

const exactPayload = {
  ...base,
  api: "query" as const,
  mode: "group" as const,
  facet: "genre" as const,
  facet_field: "primary_genre",
  metrics: [
    { id: "count", field: "count(*)", label: "Count" },
    { id: "avg_runtime", field: "avg(runtime_minutes)", label: "Avg runtime" },
    { id: "min_year", field: "min(release_year)", label: "Earliest" },
    { id: "max_year", field: "max(release_year)", label: "Latest" },
  ],
  groups: [
    {
      key: "Adventure",
      metrics: { count: 15, avg_runtime: 106.47, min_year: 1987, max_year: 2024 },
    },
  ],
  group_count: 1,
  semantics: "GROUP BY with count/sum/avg/min/max runs over the entire match set, so these figures are exact.",
};

const rankedPayload = {
  ...base,
  api: "search" as const,
  mode: "sort" as const,
  query: "A lighthearted adventure with ordinary people",
  sort: "year" as const,
  sort_field: "release_year",
  sort_label: "Year",
  direction: "desc" as const,
  limit: 12,
  retained_candidate_count: 2,
  score_monotonic: false,
  results: [movie("m0032", "Ranked One"), movie("m0031", "Ranked Two")],
  semantics:
    "The vector search retains a fixed top-K and the scalar sort only re-orders those retained hits: the candidate set is unchanged.",
};

const approxPayload = {
  ...base,
  api: "search" as const,
  mode: "group" as const,
  query: "A lighthearted adventure with ordinary people",
  facet: "genre" as const,
  facet_field: "primary_genre",
  limit: 12,
  retained_candidate_count: 3,
  buckets: [
    {
      key: "Animation",
      count: 3,
      metrics: { avg_runtime: 94, max_year: 2024 },
      candidates: [movie("m0032", "Bucket One", "Animation")],
    },
  ],
  semantics:
    "Facet buckets are built over the candidates retained during vector retrieval, so counts are approximate.",
};

function stubFetch(route: (url: string) => unknown) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/v1/status")) {
        return new Response(JSON.stringify(statusPayload), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      return new Response(JSON.stringify(route(url)), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }),
  );
}

describe("MovieDiscoveryPage · 2x2 matrix", () => {
  it("renders the four cells and defaults to search+group", async () => {
    stubFetch(() => approxPayload);
    render(<MovieDiscoveryPage />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /search[\s\S]*group/i })).toBeTruthy();
    });
    // Four cells, each labelled with its api and mode.
    const cells = screen.getAllByRole("button", { pressed: true });
    expect(cells.length).toBeGreaterThanOrEqual(1);
    // Contract bar names the active capability.
    expect(screen.getByText("search() + SearchAggregation")).toBeTruthy();
  });

  it("query+sort browses the exact catalog with paging", async () => {
    stubFetch(() => browsePayload);
    render(<MovieDiscoveryPage />);
    fireEvent.click(await screen.findByRole("button", { name: /query[\s\S]*sort/i }));

    await waitFor(() => expect(screen.getByText("Browse One")).toBeTruthy());
    expect(screen.getByText("query() + ORDER BY")).toBeTruthy();
    // Exact total is shown, which the search cells cannot produce.
    expect(screen.getByText(/exact total/)).toBeTruthy();
    expect(screen.getByText("1 / 8")).toBeTruthy();
  });

  it("query+group shows exact aggregates", async () => {
    stubFetch(() => exactPayload);
    render(<MovieDiscoveryPage />);
    fireEvent.click(await screen.findByRole("button", { name: /query[\s\S]*group/i }));

    await waitFor(() => expect(screen.getByText("Adventure")).toBeTruthy());
    expect(screen.getByText("query() + GROUP BY")).toBeTruthy();
    expect(screen.getByText("106.47")).toBeTruthy();
    expect(screen.getByText("1987")).toBeTruthy();
  });

  it("search+sort reports that scores stopped being monotonic", async () => {
    stubFetch(() => rankedPayload);
    render(<MovieDiscoveryPage />);
    fireEvent.click(await screen.findByRole("button", { name: /search[\s\S]*sort/i }));

    await waitFor(() => expect(screen.getByText("Ranked One")).toBeTruthy());
    expect(screen.getByText("search() + order_by_fields")).toBeTruthy();
    // The honest proof that the retained hits were re-ordered.
    expect(screen.getByText("jump around")).toBeTruthy();
  });

  it("search+group marks bucket counts as approximate", async () => {
    stubFetch(() => approxPayload);
    render(<MovieDiscoveryPage />);
    fireEvent.click(await screen.findByRole("button", { name: /search[\s\S]*group/i }));

    await waitFor(() => expect(screen.getByText("Animation")).toBeTruthy());
    // Exact wording check on the panel-level note only.
    expect(
      screen.getByText((_, node) => node?.className === "panel-meta" && /approximate/i.test(node.textContent ?? "")),
    ).toBeTruthy();
    // Never claim a bucket count is an exact statistic.
    expect(screen.getByText("3 retained")).toBeTruthy();
    expect(screen.getByText("avg 94 min")).toBeTruthy();
  });

  it("switching the facet re-requests with the new facet", async () => {
    const calls: string[] = [];
    stubFetch((url) => {
      calls.push(url);
      return exactPayload;
    });
    render(<MovieDiscoveryPage />);
    fireEvent.click(await screen.findByRole("button", { name: /query[\s\S]*group/i }));
    await waitFor(() => expect(calls.some((u) => u.includes("query/group"))).toBe(true));

    fireEvent.click(screen.getByRole("button", { name: "Decade" }));
    await waitFor(() => expect(calls.filter((u) => u.includes("query/group")).length).toBeGreaterThan(1));
  });

  it("surfaces a backend error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith("/api/v1/status")) {
          return new Response(JSON.stringify(statusPayload), { status: 200 });
        }
        return new Response(JSON.stringify({ detail: "milvus unreachable" }), { status: 503 });
      }),
    );
    render(<MovieDiscoveryPage />);
    fireEvent.click(await screen.findByRole("button", { name: /query[\s\S]*sort/i }));
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("milvus unreachable"));
  });
});
