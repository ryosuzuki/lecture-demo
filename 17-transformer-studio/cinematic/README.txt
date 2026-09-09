Attention / Cinema — Transformer cinematic, 2026-09-09

Run this directory with Python 3:
  python3 -m http.server 8000 --bind 127.0.0.1
Then open http://127.0.0.1:8000/

All browser dependencies are in vendor/. No API key, build step, external CDN,
or model download is needed. file:// is not supported for JavaScript modules.

index.html  Japanese interactive explainer
app.js      Six animated Three.js chapters, camera, playback and controls
math.js     Explicit illustrative X/W matrices, attention and multi-head math
notes.html  Sources, inspection scope, mathematical conventions and limits
verification.html  Delivery verification summary
vendor/     Three.js and OrbitControls; MIT license included

4 tokens / width 3 / 3 heads / Q and K width 2.
These are fixed illustrative values, not pretrained language-model weights.
Norm and MLP are diagrammatic; there is no next-token generation.

Public URL: https://ryosuzuki.github.io/lecture-demo/17-transformer-studio/cinematic/
Published on GitHub Pages separately from the unchanged ../v2/.
