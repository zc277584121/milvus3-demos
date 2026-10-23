import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { MovieDiscoveryPage } from "./index";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function movie(
  movie_id: string,
  title: string,
  primary_genre = "Adventure",
  runtime_minutes = 104,
) {
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
  model: {
    model_id: "gpahal/bge-m3-onnx-int8",
    vector_dimension: 1024,
    device: "cpu",
    loaded: true,
    cache_available: true,
  },
  dataset: { dataset_id: "movie-catalog-r1", movie_count: 90 },
  milvus: {
    collection_name: "milvus3_demos_movie_catalog",
    collection_exists: true,
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
  poster_placeholders: true,
  poster_count: 6,
};

const groupedPayload = {
  status: "passed",
  query: "A lighthearted adventure with ordinary people",
  view: "grouped",
  facet: "genre",
  facet_field: "primary_genre",
  filter: null,
  limit: 12,
  latency_ms: 18.2,
  score_semantics: "COSINE similarity, not a probability",
  list: { results: [] },
  grouped: {
    buckets: [
      {
        key: "Adventure",
        count: 2,
        candidates: [
          movie("m0001", "The Lighthouse Keeper"),
          movie("m0002", "River Crossing"),
        ],
      },
      {
        key: "Comedy",
        count: 1,
        candidates: [movie("m0003", "The Mailroom", "Comedy")],
      },
    ],
  },
  retained_candidate_count: 3,
  semantics: "Scores are COSINE similarity between BGE-M3 query and synopsis vectors.",
};

const listPayload = {
  ...groupedPayload,
  view: "list",
  list: {
    results: [
      movie("m0001", "The Lighthouse Keeper"),
      movie("m0002", "River Crossing"),
    ],
  },
  grouped: { buckets: [] },
  retained_candidate_count: 2,
};

function mockApi(status: unknown = statusPayload, search: unknown = groupedPayload) {
  const calls: Array<{ url: string; body: unknown }> = [];
  const spy = vi.spyOn(globalThis, "fetch").mockImplementation(
    async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const body = init?.body ? JSON.parse(String(init.body)) : null;
      calls.push({ url, body });
      const payload = url.includes("/status") ? status : search;
      return new Response(JSON.stringify(payload), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    },
  );
  return { calls, spy };
}

describe("MovieDiscoveryPage", () => {
  it("renders the topbar and query prompt", async () => {
    mockApi();
    render(<MovieDiscoveryPage />);

    expect(
      screen.getByRole("heading", {
        name: "Movie Discovery with Search Aggregation",
      }),
    ).toBeTruthy();
    expect(
      screen.getByLabelText("What would you like to watch?"),
    ).toBeTruthy();

    await waitFor(() => {
      expect(screen.getByText("Adventure")).toBeTruthy();
    });
  });

  it("auto-runs the first preset and renders grouped buckets", async () => {
    mockApi();
    render(<MovieDiscoveryPage />);

    await waitFor(() => {
      expect(screen.getByText("The Lighthouse Keeper")).toBeTruthy();
    });
    expect(screen.getByText("River Crossing")).toBeTruthy();
    expect(screen.getByText("The Mailroom")).toBeTruthy();
    // Bucket count labels cover only retained candidates.
    expect(screen.getByText("2 candidates")).toBeTruthy();
    expect(screen.getByText("1 candidates")).toBeTruthy();
    // Honest footer about retained-candidate statistics.
    expect(screen.getByText(/3 retained candidates across 2 groups/)).toBeTruthy();
  });

  it("switches to the list view on demand", async () => {
    const { calls } = mockApi(statusPayload, listPayload);
    render(<MovieDiscoveryPage />);

    await waitFor(() => {
      expect(screen.getByText("The Lighthouse Keeper")).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("tab", { name: "List" }));

    await waitFor(() => {
      expect(screen.getByText(/2 retained candidates/)).toBeTruthy();
    });

    const searchCalls = calls.filter((call) => call.url.includes("/search"));
    expect(searchCalls.at(-1)?.body).toMatchObject({ view: "list" });
  });

  it("opens the preset menu and submits a chosen preset", async () => {
    const { calls } = mockApi();
    render(<MovieDiscoveryPage />);

    await waitFor(() => {
      expect(screen.getByText("The Lighthouse Keeper")).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: "Choose a preset query" }));
    const option = await screen.findByRole("option", {
      name: /Lighthearted adventure/,
    });
    expect(option).toBeTruthy();

    fireEvent.click(option);
    await waitFor(() => {
      const searchCalls = calls.filter((call) => call.url.includes("/search"));
      expect(searchCalls.length).toBeGreaterThanOrEqual(1);
    });
  });

  it("toggles the runtime filter and sends max_runtime on the next search", async () => {
    const { calls } = mockApi();
    render(<MovieDiscoveryPage />);

    await waitFor(() => {
      expect(screen.getByText("The Lighthouse Keeper")).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: "Under 120 min" }));

    await waitFor(() => {
      const searchCalls = calls.filter((call) => call.url.includes("/search"));
      expect(searchCalls.at(-1)?.body).toMatchObject({ max_runtime: 120 });
    });
  });

  it("shows an empty state before any result arrives", async () => {
    mockApi({ ...statusPayload, status: "not_prepared", query_presets: [] });
    render(<MovieDiscoveryPage />);

    await waitFor(() => {
      expect(
        screen.getByText("Describe a movie, explore by facet."),
      ).toBeTruthy();
    });
  });
});
