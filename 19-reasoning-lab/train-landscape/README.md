# SGD: Walk Down the Loss

Alternative B is a causal, classroom-first view of one genuine optimization step:

1. two context tokens and a known next-token label;
2. a fixed two-parameter slice through the tiny model;
3. cross-entropy height and its real gradient;
4. the parameter update and new softmax probability.

Only `O[0,target]` and `c[target]` are trainable in this demo. All embeddings, hidden weights, remaining output weights, and other biases remain frozen. Every surface cell runs the model forward pass from `../train/model.js`; the yellow ball, projected red gradient step, teal history, probability bars, and displayed loss all use those same values.

Run the parent reasoning-lab server and open `/train-landscape/`.

## 60-second teaching script

- **0–10 s:** “The model sees exactly two tokens: *the, cat*. Our label says the next token is *sat*.”
- **10–22 s:** “On the right, the real softmax gives *sat* a probability. Cross-entropy turns that probability into height: lower probability means higher loss.”
- **22–35 s:** Rotate the surface. “This is an honest two-parameter slice, not the whole network. Everything except one target output weight and bias is frozen.”
- **35–48 s:** “The red arrow is the negative gradient: the locally downhill direction.” Press **Take one SGD step**.
- **48–60 s:** “Both numbers changed. The ball moved down, loss fell, and *sat* probability rose. That is the causal chain: gradient → parameters → logits → softmax → loss.”
- **60–75 s (optional):** Drag a parameter uphill, observe the probability/loss response, then reset.

The visual order is informed by the causal progression in 3Blue1Brown’s official gradient-descent lesson: define the cost, locate the current point, use local slope, take a step, repeat. No external video is needed in the classroom UI.
