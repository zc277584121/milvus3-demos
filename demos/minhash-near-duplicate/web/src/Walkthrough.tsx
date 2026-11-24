import { useState } from "react";

import videoUrl from "./assets/minhash-walkthrough.mp4";
import posterUrl from "./assets/minhash-walkthrough.webp";

export function Walkthrough() {
  const [failed, setFailed] = useState(false);

  if (failed) {
    return (
      <p className="walkthrough-error" role="status">
        The video could not be played. <a href={videoUrl}>Open the MP4</a>.
      </p>
    );
  }

  return (
    <video
      className="walkthrough-video"
      aria-label="MinHash walkthrough"
      src={videoUrl}
      poster={posterUrl}
      controls
      playsInline
      preload="metadata"
      onError={() => setFailed(true)}
    >
      <a href={videoUrl}>Open the MP4</a>
    </video>
  );
}
