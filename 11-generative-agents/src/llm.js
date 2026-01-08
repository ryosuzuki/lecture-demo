// LLM wrapper for WebLLM (single engine).
// Provides a small queue so multiple agent calls are serialized.
//
// WebLLM has OpenAI-compatible chat.completions API.
// Docs: https://webllm.mlc.ai/docs/user/basic_usage.html

export class LLMClient {
  constructor(engine) {
    this.engine = engine;
    this._queue = Promise.resolve();
  }

  async chat({ messages, temperature = 0.7, max_tokens = 256, response_format = undefined }) {
    const task = async () => {
      const req = { messages, temperature, max_tokens };
      if (response_format) req.response_format = response_format;

      const resp = await this.engine.chat.completions.create(req);
      const content = resp?.choices?.[0]?.message?.content ?? "";
      return content;
    };

    // serialize to avoid overlapping WebGPU work
    this._queue = this._queue.then(task, task);
    return this._queue;
  }

  async chatJSON({ messages, schemaObj, temperature = 0.2, max_tokens = 384 }) {
    // WebLLM JSON mode: response_format {type:"json_object", schema?: "<json schema string>"}
    // We still instruct the model to output JSON explicitly in the prompt.
    const schema = schemaObj ? JSON.stringify(schemaObj) : undefined;
    const response_format = schema
      ? { type: "json_object", schema }
      : { type: "json_object" };

    const txt = await this.chat({ messages, temperature, max_tokens, response_format });

    try {
      return JSON.parse(txt);
    } catch (e) {
      // fallback: try extracting {...}
      const start = txt.indexOf("{");
      const end = txt.lastIndexOf("}");
      if (start >= 0 && end > start) {
        const slice = txt.slice(start, end + 1);
        try { return JSON.parse(slice); } catch (_) {}
      }
      console.warn("JSON parse failed:", txt);
      return null;
    }
  }
}
