import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { StatusBadge } from "./index";

describe("StatusBadge", () => {
  it("labels the implemented foundation honestly", () => {
    render(<StatusBadge status="foundation-ready" />);

    expect(screen.getByText("Foundation ready").className).toContain(
      "foundation-ready",
    );
  });

  it("labels an implemented demo without claiming runtime availability", () => {
    render(<StatusBadge status="implemented" />);

    expect(screen.getByText("Implemented").className).toContain("implemented");
  });
});
