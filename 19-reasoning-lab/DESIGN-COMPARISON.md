# Learning recap: three alternatives

## Why the first version was insufficient

Ryo found the central graph too passive: numerical parameter changes and the causal link from target probability to loss to downhill update were not salient. Passing numerical tests did not establish teaching effectiveness. These alternatives directly address that feedback; none is claimed to be educationally optimal before classroom use.

## Three different questions

- **A / Inside the Layers**: What changes inside a network? Use spatial matrix slabs, selected numeric parameters, before/after updates, live vectors and fixed right-hand probabilities.
- **B / Downhill on Loss**: What does a gradient step actually do? Freeze all but two selected parameters so the surface and trajectory belong to the same function. Connect height to cross-entropy and right-hand target probability.
- **C / One Sentence**: Why update anything at all? Predict first, reveal an observed target, measure its low probability, inspect one gradient, take one update and watch repeated actual updates. A matrix and time-series plot make the progression more salient than decorative 3D.

Suggested order is C (4 minutes) → B (4 minutes) → A (3 minutes), or choose only A for a compact integrated recap. Avoid treating training on one example as a demonstration of held-out generalization. Show the highlighted input context, not an implied full-sentence understanding by these tiny models.

## Reference inspection and what changed

Read the official written adaptations of 3Blue1Brown's *Large Language Models Explained Briefly* and *Gradient Descent, How Neural Networks Learn*. Inspection was the illustrated written explanations, not a complete video playback. The flow used is concrete prediction → observed target → quantified error → parameter sensitivity → update → repeat over data. Their source links remain in the instructor TEACHING-GUIDE, not the live classroom UI. We use cross-entropy for next-token prediction; the digit-classification lesson's squared-error example is not silently equated to it.

## Videos

Videos are screen recordings of the operational apps, with readable on-screen English instructions, not generated mockups. Instructor scripts explain the classroom questions in Japanese. They are silent; they do not imply a narrated lecture recording.
