# Synthetic dataset provenance

`synthetic-driving-scenes-r1` was created specifically for the Milvus
StructArray search demo. It does not contain images, text, identifiers, or
annotations copied from the former externally sourced prototype.

- `scenes.json` contains original fictional scene specifications and the exact
  prompt-specific scene descriptions used for image generation.
- The 30 files under `images/` were generated from text only with Codex's
  built-in OpenAI image-generation capability. No source or reference images
  were supplied.
- Images were center-cropped and downscaled to 960×420 JPEG at quality 84 for
  use as low-resolution representative video frames.
- `SHA256SUMS` records the final byte identity of every shipped JPEG.
- The data loader expands each scene's six object specifications across three
  deterministic observation phases. These records are synthetic and are not
  transcriptions of any source video.

The images intentionally use generic roads and unbranded vehicles. Prompts
prohibit recognizable landmarks, readable plates and signs, logos, text, and
watermarks. Human visual review is still required before any public release.
