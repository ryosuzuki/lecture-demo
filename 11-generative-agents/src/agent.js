// Agent implementation (simplified generative agents).
// Each agent has:
// - persona (bio, traits, goals)
// - memory stream with (text, time, importance, type)
// - daily plan (JSON schedule)
// - step() that perceives world, retrieves memories, calls LLM to decide action,
//   and appends new memories.
//
// Notes:
// - This is a *mini educational* replication, not a full Smallville.
// - Relevance is TF-IDF, not embeddings.
// - We keep prompts short to fit small local models.

import { MemoryStream, MemoryRecord } from "./memory.js";
import { ACTION_SCHEMA, PLAN_SCHEMA, REFLECTION_SCHEMA,
         buildDailyPlanPrompt, buildActionPrompt, buildImportancePrompt, buildReflectionPrompt } from "./prompts.js";
import { clamp, dist, formatSimTime } from "./util.js";

let _idCounter = 0;
function nextId(prefix = "m") { _idCounter += 1; return `${prefix}_${_idCounter}`; }

export class Agent {
  constructor({
    id,
    name,
    age,
    title,
    traits,
    bio,
    goals,
    home,
    startX,
    startY,
    // Optional paper-style fields
    persona = "",
    seedMemory = "",
  }) {
    this.id = id;
    this.name = name;
    this.age = age;
    this.title = title;
    this.traits = traits;
    this.bio = bio;
    this.goals = goals;
    this.home = home;

    // Paper-style persona string (used in prompts).
    this.persona = persona;

    // Paper-style initial memory seed (semicolon-delimited phrases).
    this.seedMemory = seedMemory;

    this.x = startX;
    this.y = startY;

    this.dest = null; // {x,y, placeId, placeName}
    this.speed = 1.3; // cells per tick

    this.energy = 100;
    this.tickMinutes = 10;

    this.memory = new MemoryStream();
    this.dailyPlan = null; // {date, blocks:[]}
    this.summary = ""; // updated by reflection
    this.importanceSinceReflection = 0;

    this.currentAction = "idle";
    this.lastThought = "";
    this.lastRetrievedMemoriesText = "";
    this.lastPromptText = "";
  }

  seedInitialMemories(now) {
    // Seed the memory stream with biographical facts (paper-style).
    // In the original paper demo, each semicolon-delimited phrase is entered as an initial memory.
    let seeds = [];

    if (this.seedMemory && this.seedMemory.trim()) {
      seeds = this.seedMemory
        .split(";")
        .map((s) => s.trim())
        .filter(Boolean);
    } else {
      // Fallback if no seed memory is provided.
      seeds = [
        `${this.name} is ${this.age} years old.`,
        `${this.name} works as ${this.title}.`,
        `${this.name} lives near ${this.home}.`,
        `${this.name}'s traits: ${this.traits.join(", ")}.`,
        `${this.name}'s long-term goals: ${this.goals.join("; ")}.`,
        this.bio,
      ];
    }

    // Start a little higher so these long-lived persona memories have weight.
    let imp = 8;
    for (const t of seeds) {
      this.memory.add(
        new MemoryRecord({
          id: nextId("seed"),
          time: now,
          text: t,
          importance: imp,
          type: "observation",
        })
      );
      // Gentle decay; keep seeds reasonably important.
      imp = Math.max(5, imp - 1);
    }
  }

  dailyPlanText() {
    if (!this.dailyPlan?.blocks?.length) return "";
    return this.dailyPlan.blocks.map(b => `${b.start}-${b.end} @${b.location}: ${b.activity}`).join("\n");
  }

  planSnippetFor(now) {
    if (!this.dailyPlan?.blocks?.length) return "(no plan)";
    const hhmm = String(now.getHours()).padStart(2,"0") + ":" + String(now.getMinutes()).padStart(2,"0");
    const upcoming = this.dailyPlan.blocks.filter(b => b.end >= hhmm).slice(0, 3);
    if (!upcoming.length) return "(no remaining plan blocks)";
    return upcoming.map(b => `${b.start}-${b.end} @${b.location}: ${b.activity}`).join("\n");
  }

  async initialize(llm, world, now, { mode = "fast" } = {}) {
    this.seedInitialMemories(now);
    await this.makeDailyPlan(llm, world, now);
    world.logEvent(now, `${this.name} wakes up and starts the day.`, world.placeAt(this.x, this.y)?.id ?? null);
  }

  async makeDailyPlan(llm, world, now) {
    const messages = buildDailyPlanPrompt(this, world, now);
    const obj = await llm.chatJSON({ messages, schemaObj: PLAN_SCHEMA, temperature: 0.35, max_tokens: 512 });

    if (obj && Array.isArray(obj.blocks)) {
      this.dailyPlan = obj;
      return;
    }

    // fallback
    const date = formatSimTime(now).slice(0,10);
    const pick = (pred, fallbackName) =>
      world.places.find((p) => pred(p))?.name ?? fallbackName ?? world.places[0]?.name ?? this.home;

    const cafe = pick((p) => /cafe/i.test(p.name), "Hobbs Cafe");
    const library = pick((p) => /library/i.test(p.name), "Oak Hill College Library");
    const park = pick((p) => /park/i.test(p.name), "Johnson Park");
    const market = pick((p) => /market|pharmacy/i.test(p.name), "Willow Market and Pharmacy");

    this.dailyPlan = {
      date,
      blocks: [
        { start: "08:00", end: "09:00", location: this.home, activity: "Morning routine and breakfast" },
        { start: "09:00", end: "11:30", location: library, activity: "Work / study" },
        { start: "11:30", end: "12:30", location: cafe, activity: "Lunch and a short chat" },
        { start: "12:30", end: "15:30", location: library, activity: "More focused work / study" },
        { start: "15:30", end: "16:15", location: park, activity: "Walk and clear my head" },
        { start: "16:15", end: "18:00", location: market, activity: "Errands" },
        { start: "18:00", end: "20:00", location: this.home, activity: "Dinner and downtime" },
      ],
    };
  }

  _moveTowardsDest() {
    if (!this.dest) return;
    const d = dist(this.x, this.y, this.dest.x, this.dest.y);
    if (d < 0.01) {
      this.x = this.dest.x;
      this.y = this.dest.y;
      this.dest = null;
      return;
    }
    const step = Math.min(this.speed, d);
    const dx = (this.dest.x - this.x) / d;
    const dy = (this.dest.y - this.y) / d;
    this.x += dx * step;
    this.y += dy * step;
    this.x = clamp(this.x, 0, 31.9);
    this.y = clamp(this.y, 0, 31.9);
  }

  async _rateImportancePaperStyle(llm, text) {
    // separate LLM call (paper-ish)
    const messages = buildImportancePrompt(this, text);
    const out = await llm.chat({ messages, temperature: 0.0, max_tokens: 16 });
    const m = out.match(/(\d+)/);
    if (!m) return 3;
    const n = parseInt(m[1], 10);
    return clamp(n, 1, 10);
  }

  async maybeReflect(llm, world, now) {
    if (this.importanceSinceReflection < 26) return;
    const recent = this.memory.recent(12).map(r => ({ text: r.text, importance: r.importance, type: r.type }));
    const messages = buildReflectionPrompt(this, now, recent);
    const obj = await llm.chatJSON({ messages, schemaObj: REFLECTION_SCHEMA, temperature: 0.35, max_tokens: 512 });

    if (obj?.insights?.length) {
      for (const ins of obj.insights) {
        const rec = new MemoryRecord({ id: nextId("ref"), time: now, text: `Insight: ${ins}`, importance: 8, type: "reflection" });
        this.memory.add(rec);
      }
      if (obj.summary_update) this.summary = obj.summary_update;
      world.logEvent(now, `${this.name} reflects and gains insights.`, world.placeAt(this.x, this.y)?.id ?? null);
    }
    this.importanceSinceReflection = 0;
  }

  async step(llm, world, allAgents, now, { mode = "fast" } = {}) {
    // Move if traveling
    if (this.dest) {
      this._moveTowardsDest();
      this.currentAction = "walking";
    }

    const perception = world.perceive(this, allAgents);

    // Build retrieval query (current situation)
    const q = `${this.name} at ${perception.placeName}. Plan: ${this.planSnippetFor(now)}. Others: ${perception.others.map(o => o.name).join(", ")}. Chat: ${perception.chat.map(c => c.text).join(" | ")}`;
    const retrieved = this.memory.retrieve(q, now, 8);
    this.lastRetrievedMemoriesText = retrieved.map(r => `[score:${r.score.toFixed(2)}] ${r.rec.text}`).join("\n");

    // Prompt for action decision (single LLM call in fast mode)
    const messages = buildActionPrompt({ agent: this, world, now, perception, retrieved, mode });
    this.lastPromptText = messages.map(m => `${m.role.toUpperCase()}:\n${m.content}`).join("\n\n---\n\n");

    const obj = await llm.chatJSON({ messages, schemaObj: ACTION_SCHEMA, temperature: 0.35, max_tokens: 512 });

    if (!obj) {
      // fallback: random move or stay
      this.lastThought = "I'm not sure what to do next… I'll observe for now.";
      this.currentAction = "idle";
      return;
    }

    this.lastThought = obj.thought || "";

    // Store memories
    if (Array.isArray(obj.memories)) {
      for (const m of obj.memories) {
        const text = String(m.text || "").slice(0, 280);
        const type = m.type === "action" ? "action" : "observation";
        let imp = clamp(parseInt(m.importance ?? 3, 10) || 3, 1, 10);

        if (mode === "paper") {
          // do separate rating call to better emulate the paper's pipeline
          imp = await this._rateImportancePaperStyle(llm, text);
        }

        const rec = new MemoryRecord({ id: nextId("mem"), time: now, text, importance: imp, type });
        this.memory.add(rec);
        this.importanceSinceReflection += imp;
      }
    }

    // Execute action
    const action = obj.action;
    const target = (obj.target || "").trim();
    const utterance = (obj.utterance || "").trim();

    // If interacting, only allow if target is nearby.
    if (action === "interact") {
      const ok = perception.others.some(o => o.name === target);
      if (ok && utterance) {
        world.postChat(perception.placeId, this.name, utterance, now);
        this.currentAction = `talking to ${target}`;
      } else {
        this.currentAction = "idle";
      }
    } else if (action === "move") {
      // target should be a place name
      const place = world.places.find(p => p.name === target) ?? world.places[0];
      this.dest = { x: place.x, y: place.y, placeId: place.id, placeName: place.name };
      this.currentAction = `going to ${place.name}`;
      world.logEvent(now, `${this.name} heads to ${place.name}.`, place.id);
    } else {
      this.currentAction = "staying";
    }

    // Energy drift (toy)
    this.energy = clamp(this.energy - 1, 0, 100);

    // Reflection occasionally
    await this.maybeReflect(llm, world, now);
  }
}
