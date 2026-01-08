// Prompt templates (educational, inspired by the paper's components).
// We keep JSON-mode in mind (WebLLM response_format: json_object).

import { formatSimTime } from "./util.js";

export function systemPersona(agent) {
  // Keep it short to fit small models; user can expand.
  const personaBlock = (agent.persona && agent.persona.trim())
    ? agent.persona.trim()
    : `Name: ${agent.name}\nAge: ${agent.age}\nRole: ${agent.title}\nTraits: ${agent.traits.join(", ")}\nBio: ${agent.bio}`;

  const summaryBlock = (agent.summary && agent.summary.trim())
    ? `\nCurrent self-summary (from reflection):\n${agent.summary.trim()}\n`
    : "";

  return `You are roleplaying as a real person living in Smallville, a small town.
Never mention you are an AI model or that you are in a simulation. Stay in character.

Character sheet (ground truth facts):
${personaBlock}

Long-term goals:
- ${agent.goals.join("\n- ")}
${summaryBlock}

Style:
- Use English only.
- Be concise and concrete.
- If speaking, sound like a real person.
- If thinking, keep it brief (1-2 sentences).`;
}

export const ACTION_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["thought", "action", "target", "utterance", "memories"],
  properties: {
    thought: { type: "string" },
    action: { type: "string", enum: ["move", "interact", "stay"] },
    target: { type: "string" },
    utterance: { type: "string" },
    memories: {
      type: "array",
      maxItems: 3,
      items: {
        type: "object",
        additionalProperties: false,
        required: ["text", "type", "importance"],
        properties: {
          text: { type: "string" },
          type: { type: "string", enum: ["observation", "action"] },
          importance: { type: "integer", minimum: 1, maximum: 10 }
        }
      }
    }
  }
};

export const PLAN_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["date", "blocks"],
  properties: {
    date: { type: "string" },
    blocks: {
      type: "array",
      maxItems: 10,
      items: {
        type: "object",
        additionalProperties: false,
        required: ["start", "end", "location", "activity"],
        properties: {
          start: { type: "string" },
          end: { type: "string" },
          location: { type: "string" },
          activity: { type: "string" }
        }
      }
    }
  }
};

export const REFLECTION_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["insights", "summary_update"],
  properties: {
    insights: { type: "array", minItems: 2, maxItems: 4, items: { type: "string" } },
    summary_update: { type: "string" }
  }
};

export function buildDailyPlanPrompt(agent, world, now) {
  const places = world.places.map(p => p.name).join(", ");
  const dateStr = formatSimTime(now).slice(0, 10);

  const sys = systemPersona(agent);
  const user = `Today is ${dateStr}.
Create a simple plan for today for ${agent.name}.

Constraints:
- Choose location only from: ${places}
- Use time blocks (start/end) in HH:MM (24h).
- Provide 6~10 blocks.
- Output MUST be valid JSON that matches the given schema.
(Important: respond in JSON only, no markdown.)

Context:
- ${agent.name}'s home base: ${agent.home}
- ${agent.name}'s current priorities: ${agent.goals.join("; ")}

JSON schema (for reference, do not include it in output):
${JSON.stringify(PLAN_SCHEMA)}`;

  return [
    { role: "system", content: sys },
    { role: "user", content: user }
  ];
}

export function buildActionPrompt({ agent, world, now, perception, retrieved, mode }) {
  const sys = systemPersona(agent);
  const placeList = world.places.map(p => p.name).join(", ");
  const timeStr = formatSimTime(now);

  const others = perception.others.length
    ? perception.others.map(o => `- ${o.name}: ${o.action}`).join("\n")
    : "(no one nearby)";

  const chats = perception.chat.length
    ? perception.chat.map(c => `- ${c.speaker}: ${c.text}`).join("\n")
    : "(no recent chat)";

  const planSnippet = agent.planSnippetFor(now);

  const memLines = retrieved.length
    ? retrieved.map((m, i) => {
        const rec = m.rec;
        return `${i+1}. [${rec.type}] (imp:${rec.importance}) ${rec.text}`;
      }).join("\n")
    : "(no memories retrieved)";

  const user = `You are ${agent.name}. It is ${timeStr}.
Current location: ${perception.placeName}

Nearby people:
${others}

Recent chat in this place:
${chats}

Your plan (snippet):
${planSnippet}

Retrieved memories (most relevant):
${memLines}

Decide what to do NEXT for the next ~${agent.tickMinutes} minutes.
Allowed actions:
- move: go to a different location (target must be one of: ${placeList})
- interact: say something to a nearby person (target must be a nearby person's name)
- stay: continue what you're doing here (target can be current location)

Also, write 1-3 memory items to store into your memory stream:
- observation: what you noticed (short factual)
- action: what you decided/did (short factual)
Each memory needs an importance rating 1..10 (10 = life-changing, 1 = trivial).

Output MUST be valid JSON ONLY and match schema.
JSON schema:
${JSON.stringify(ACTION_SCHEMA)}`;

  return [
    { role: "system", content: sys },
    { role: "user", content: user }
  ];
}

export function buildImportancePrompt(agent, memoryText) {
  // Paper-style "poignancy" rating prompt (separate call).
  const sys = systemPersona(agent);
  const user = `Rate the importance (poignancy) of the following memory for ${agent.name}'s long-term goals.
Return only an integer 1..10.

Memory:
${memoryText}`;
  return [{ role: "system", content: sys }, { role: "user", content: user }];
}

export function buildReflectionPrompt(agent, now, recentMemories) {
  const sys = systemPersona(agent);
  const timeStr = formatSimTime(now);
  const mem = recentMemories.map((m, i) => `${i+1}. (imp:${m.importance}) ${m.text}`).join("\n");

  const user = `It is ${timeStr}. You are ${agent.name}.
You will reflect on your recent experiences and update your self-understanding.

Recent memories:
${mem}

Task:
- Produce 2-4 high-level insights (short sentences).
- Produce a one-paragraph summary update describing what kind of person ${agent.name} is becoming today.

Output MUST be valid JSON ONLY and match schema.
JSON schema:
${JSON.stringify(REFLECTION_SCHEMA)}`;

  return [{ role: "system", content: sys }, { role: "user", content: user }];
}
