# Context teaching rehearsal — 2026-09-09

Completion owner: current Transformer continuation session. Integration target: origin/main.
Authoritative route: https://ryosuzuki.github.io/lecture-demo/17-transformer-studio/context/guide.html
Owned paths: context/guide.html; context/index.html and context/notes.html (navigation only); context/rehearsal/ (new media, script, captions, proof record). No existing cinematic/ or v2/ content changed. No new worktrees or branches.

Actual public context UI recording, 16 actions, 5:01, 1920x1080 H.264 High/yuv420p/30fps, 48kHz AAC. Original Japanese narration, Edge NanamiNeural. 58 Japanese cues, burned plus mov_text and separate SRT/VTT. Downloadable script and timestamped player guide. Numerical invariants checked during actual capture. Head selection failure found visually and only that segment re-recorded, with explicit head==1 assertion.

Prepublication validation: full delivery decode; 9030 constant-duration video frames; track end spread 21.3ms; embedded/separate subtitle text parity; no nonfinite/clipped audio or unexpected >=1.5s internal narration silence. -16.0 LUFS / -2.1 dBFS true peak. Full local Whisper transcript reviewed; direct perceptual listening unverified. 32 frames reviewed in 4 contact sheets plus detailed scene images. Guide playback and seeks validated at 1440/390/320px, 15 result groups, zero page/resource errors. 4901 unchanged context numerical checks and existing v2 test-model pass. No repository-wide validator configured; scoped validators used.

Initial local SimpleHTTP preview did not support reliable range seeking; corrected preview to byte-range server and reran all guide checks. Production GitHub Pages endpoint and exact bytes are verified after deployment; immutable tip and readback report are kept by the completion owner outside this commit.

Limits: hand-designed model, not learned embeddings or complete 3B1B recreation; Norm/MLP/training/generation unimplemented. ASR is not direct listening; Safari and physical phones untested. Canonical high-quality master retained in task artifacts, raw capture in temporary workspace. Existing pages preserved.
