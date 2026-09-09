3Blue1Brown Transformer — source-aligned interactive reconstruction

This is not a pixel-identical or all-animation replica. Read provenance.html.
The 52 scene ranges cover the two published videos; each range can contain
more original shots than are recreated in the corresponding web scene.

Local launch (Python 3, no Python dependencies):
  python3 server.py --port 8000
Open http://127.0.0.1:8000/

The server implements byte-range responses so the included MP4 recordings
can be seeked. WebGL is required for the interactive 3D views. YouTube source
embeds require an Internet connection. No GPT API call or credential is used.

Files:
  index.html / app.js / style.css — presentation + interaction
  space.js / vendor/ — Three.js 3D projection and camera
  math.js — normalization, attention, values and numeric checks
  embeddings.json — selected actual GloVe 50D vectors and PCA projection
  scenes.js — 52 source-timestamped scenes and original English teaching cues
  guide.html / media/ — five captioned, silent browser walkthroughs
  compare.html / comparison/ — reference frame vs web reconstruction
  provenance.html / qa.html — scope, data provenance and verification

Attribution: Grant Sanderson / 3Blue1Brown, Deep Learning Chapters 5 and 6,
and the public 3b1b/videos Transformer animation sources.
Changes: rewritten web code, interactive controls, English narration cues,
new illustrative matrices/values, reconstructed projection/camera/layout.
Code and added teaching text: CC BY-NC-SA 4.0.
Third-party dependencies retain their own licenses (Three.js / KaTeX MIT;
GloVe dataset PDDL). Original videos are not included or redistributed.

English conversion — September 9, 2026
The 3d/ directory contains five spatial explanations, a guide, comparisons,
and five additional English-captioned silent recordings. All ten recordings
were captured again with the English interface; none reuse burned Japanese UI.
Original source comparison images remain unchanged.
See english-verification.html for the conversion audit and evidence.
