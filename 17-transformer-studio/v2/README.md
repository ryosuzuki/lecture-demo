# Transformer Studio II

A coherent, inspectable Transformer lesson: one input sentence, a tiny trained decoder, and actual intermediate activations all the way to vocabulary probabilities.

## Use

Open `index.html`. Everything required for inference is local. No install, API key, account or network is needed. The Japanese instructor guide and narrated walkthrough are in `guide.html`.

- Press `F` for compact presentation layout; use arrow keys for chapters.
- Keep block 1 / head 1 and receiver `bank` for the introductory sequence.
- Change context, drag Q, edit V, ablate one head, inspect readout, test the causal mask, or load actual training checkpoints.
- See `model-card.html` for boundaries, architecture, training data and loss decomposition.

## Model

1,296 parameters; two pre-norm decoder blocks; two attention heads per block; 8-D residual stream; each head has exact 2-D Q/K/V; 16-unit ReLU MLPs; learned absolute positions; 21-token word-level vocabulary. Trained from scratch on 81 original synthetic short sentences, with 15 held-out sentences representing five unique held-out prefixes. Not pretrained GPT and not evidence of general English competence.

`model.js` bundles the JSON for offline loading. `model.json` includes weights at six saved checkpoints, corpus and reference PyTorch activations. `engine.js` is an independent JavaScript inference implementation. `train.py` reproduces the source model. `narration.json` contains the authored Japanese voice script; `walkthrough.srt` and `walkthrough.vtt` preserve its text.

## Important causal distinctions

Q editing changes one receiving query. V editing changes one sending value for all receivers of the selected head. That head's Q/K weights remain unchanged, though downstream attention may change. These are interventions, not training updates. Head gain scales a single head's output before concatenation. Context comparison clears interventions. Temperature affects only vocabulary softmax.

## Important evaluation distinction

The held-out split excludes determiner/context combinations. Total validation loss increases with extended training because those missing combinations become unlikely; bank→verb loss improves toward ln(3), appropriate for three equally likely valid verbs. Do not interpret the rising total loss as deterioration of bank disambiguation.

## Provenance and verification

Original graphics and code, inspired by the conceptual sequence in 3Blue1Brown's Attention lesson. No borrowed video frames, private transcripts, student data, or external model weights are published.

`verification.json` records numerical, independent-review, browser and media checks. The JavaScript forward pass agrees with PyTorch on all intermediate values for five reference prompts within 2.7e-6. Source files are self-contained static assets; no repository build system is required.

The narrated video uses AI-generated Japanese speech (Microsoft Edge Nanami Neural), not the instructor's cloned voice. It shows actual browser interactions and is delivered with selectable subtitles and a transcript.

Publication scope is `17-transformer-studio/v2/` plus a version link in the original prototype. Completion owner is the requesting OpenClaw agent; integration target is `origin/main`, followed by GitHub Pages and public browser verification. Existing prototype content is preserved.
