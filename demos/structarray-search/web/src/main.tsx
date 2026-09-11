import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { StructArrayDemoPage } from "./index";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <StructArrayDemoPage />
  </StrictMode>,
);
