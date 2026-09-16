# Reason & Tools walkthrough — instructor script

**Teacher question:** How do we separate a plausible explanation from a correct answer?

1. Identify the public local Qwen3 0.6B model and the arithmetic question.
2. Run Direct. Point out the actual generated-token count and measured time.
3. Run Reason aloud. Ask: *Does more text itself establish correctness?*
4. Keep truncation visible if the model reaches its generation limit; never supply a missing final response.
5. Open Tool. Trace model request → application arguments → deterministic calculator result, 1081.
6. Open Search. Combine 13 and 9 with subtraction; contrast explicit exact-rational search with model output.

**Teaching boundary:** Emitted reasoning is observable model output, not privileged access to all hidden computation. Calculator and Game 24 panels are separately labeled deterministic tools.

