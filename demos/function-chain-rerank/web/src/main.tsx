import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { FunctionChainDemoPage } from "./index";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <FunctionChainDemoPage />
  </StrictMode>,
);
