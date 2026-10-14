# Physical AI — saved lecture demos

Open `index.html` directly, or serve this directory with `python3 -m http.server 8000`. No API key, GPU, Tailscale, or inference server is required.

Eight recorded model/physics videos and five interactive recorded-output inspectors are retained. Inspectors embed their data/images in the page; they do not run new model inference. Videos loop; the openpilot replay starts after the six-second opening explanation.

Videos are retained in this repository’s existing `physical-ai-demo-videos-20261014` release, not a different project. For fully offline playback, run `python3 download-videos.py` once, then open `offline.html`.

The original live services were retired on 2026-10-14. Webcam/new-video inference and cloud robot planning are intentionally absent. Original reproducibility source remains in the private `lecture-physical-ai-demos` repository. Interactive live MuJoCo source is retained there for optional local CPU use, but this saved page uses recorded physics and is not a live physics engine.
