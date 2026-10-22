import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { MinHashDemoPage } from "./index";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <MinHashDemoPage />
  </StrictMode>,
);
