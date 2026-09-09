# Transformer Studio

An original interactive teaching prototype inspired by 3Blue1Brown's geometric explanations, with a Japanese instructor guide and captioned browser walkthrough.

- Open `index.html` directly: no API key, install, external library or network required for the five chapters.
- `teaching-guide.html`: 18-minute lesson script, questions, limitations and links to real-model follow-up resources.
- `walkthrough.mp4`: recorded browser interaction, Japanese captions, no audio.
- `transformer-studio.zip`: offline copy including video.

## Model boundary

This is not pretrained GPT. Chapter 1 is a concept sketch. Chapters 2–3 use a controlled four-token example with hand-designed 2-D queries, keys and values, computed scaled dot-product softmax attention and a residual update with identity output projection. Chapter 4 demonstrates masking using separate illustrative positional scores. Chapter 5 is a separate tiny readout with a one-bias gradient update. It is not an end-to-end trained Transformer.

## Checks

Verified in Chrome at desktop and 390px mobile widths: all five chapters, query intervention, softmax sum, zero residual update, causal mask toggle, loss decrease after gradient update, sampling, reset and no JavaScript errors. No repository build system is required for this self-contained static path. Video is fully decoded and visually inspected before delivery.

Publication scope: only `17-transformer-studio/`; existing demos remain unchanged. Integration target: `origin/main`. Completion owner: the requesting OpenClaw agent. Canonical delivery is the GitHub Pages path after deployment and browser playback verification.

## Credits

- https://www.3blue1brown.com/lessons/attention/
- https://poloclub.github.io/transformer-explainer/
- https://bbycroft.net/llm

Graphics and code are original; this is not an official 3Blue1Brown reproduction.
