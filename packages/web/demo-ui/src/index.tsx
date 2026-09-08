import type { ReactNode } from "react";

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

export interface DemoStatusPageProps {
  demo: DemoDefinition;
  children?: ReactNode;
}

export function StatusBadge({ status }: { status: DemoStatus }) {
  const labels: Record<DemoStatus, string> = {
    available: "Available",
    implemented: "Implemented",
    "foundation-ready": "Foundation ready",
    planned: "Planned",
  };
  return (
    <span className={`status-badge status-badge--${status}`}>
      {labels[status]}
    </span>
  );
}

export function DemoStatusPage({ demo, children }: DemoStatusPageProps) {
  return (
    <main
      className="demo-page"
      style={{ "--demo-accent": demo.accent } as React.CSSProperties}
    >
      <a className="back-link" href="/">
        ← Back to Portal
      </a>
      <section className="demo-hero">
        <StatusBadge status={demo.status} />
        <p className="eyebrow">Milvus 3.0 capability</p>
        <h1>{demo.title}</h1>
        <p className="demo-lead">{demo.description}</p>
      </section>
      <section
        className="demo-foundation-grid"
        aria-label="Demo foundation status"
      >
        <article>
          <span>Capability</span>
          <strong>{demo.capability}</strong>
        </article>
        <article>
          <span>Service boundary</span>
          <strong>Independent backend, tests, and image</strong>
        </article>
        <article>
          <span>Core implementation</span>
          <strong>Reserved for its dedicated Story</strong>
        </article>
      </section>
      <section className="demo-highlights">
        <h2>What this demo will make visible</h2>
        <ul>
          {demo.highlights.map((highlight) => (
            <li key={highlight}>{highlight}</li>
          ))}
        </ul>
      </section>
      {children}
    </main>
  );
}
