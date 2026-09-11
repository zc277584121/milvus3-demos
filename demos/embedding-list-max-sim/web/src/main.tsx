import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { EmbeddingListDemoPage } from "./index";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <EmbeddingListDemoPage />
  </StrictMode>,
);
