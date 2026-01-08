// Simple 2D world: named places + chat + event log.
// Rendering uses Canvas 2D.

import { clamp, dist } from "./util.js";

export class World {
  constructor({ width = 32, height = 32 } = {}) {
    this.width = width;
    this.height = height;

    // Named places (a few, for educational demo).
    // Coordinates are in grid space [0,width] x [0,height].
    // Names and landmarks mirror the "Smallville" setting from the Generative Agents paper.
    this.places = [
      { id: "hobbs", name: "Hobbs Cafe", x: 10, y: 10, r: 3.0, desc: "A cozy coffee shop with pastries and regulars." },
      { id: "park", name: "Johnson Park", x: 22, y: 9, r: 3.5, desc: "A green park with benches, flowers, and walking paths." },
      { id: "college", name: "Oak Hill College", x: 20, y: 21, r: 3.2, desc: "A small college campus where students attend classes." },
      { id: "library", name: "Oak Hill College Library", x: 8, y: 22, r: 3.2, desc: "A quiet library used for studying and research." },
      { id: "market", name: "Willow Market and Pharmacy", x: 26, y: 26, r: 3.2, desc: "A neighborhood market with a pharmacy counter." },
      { id: "plaza", name: "Town Plaza", x: 14, y: 28, r: 3.8, desc: "The center of town where people pass by and chat." },
    ];

    this.chatByPlace = new Map(); // placeId -> [{speaker,text,time}]
    for (const p of this.places) this.chatByPlace.set(p.id, []);

    this.log = []; // [{time,text,place}]
  }

  nearestPlace(x, y) {
    let best = null;
    let bestD = 1e9;
    for (const p of this.places) {
      const d = dist(x, y, p.x, p.y);
      if (d < bestD) { best = p; bestD = d; }
    }
    return { place: best, dist: bestD };
  }

  nearestPlaceName(x, y) {
    const { place, dist: d } = this.nearestPlace(x, y);
    if (!place) return "Nowhere";
    return d <= place.r ? place.name : "Street";
  }

  placeAt(x, y) {
    const { place, dist: d } = this.nearestPlace(x, y);
    if (!place) return null;
    return d <= place.r ? place : null;
  }

  perceive(agent, allAgents) {
    const place = this.placeAt(agent.x, agent.y);
    const placeName = place ? place.name : "Street";
    const placeId = place ? place.id : null;

    const others = [];
    for (const a of allAgents) {
      if (a.id === agent.id) continue;
      const samePlace = place && this.placeAt(a.x, a.y)?.id === place.id;
      if (samePlace) {
        others.push({ id: a.id, name: a.name, action: a.currentAction || "idle" });
      }
    }

    const chat = placeId ? (this.chatByPlace.get(placeId) ?? []).slice(-6) : [];

    return { place, placeId, placeName, others, chat };
  }

  postChat(placeId, speaker, text, time) {
    if (!placeId) return;
    const arr = this.chatByPlace.get(placeId);
    if (!arr) return;
    arr.push({ speaker, text, time });
    // Also log to world events
    this.log.push({ time, text: `${speaker}: ${text}`, place: this.places.find(p => p.id === placeId)?.name ?? "" });
  }

  logEvent(time, text, placeId = null) {
    const place = placeId ? (this.places.find(p => p.id === placeId)?.name ?? "") : "";
    this.log.push({ time, text, place });
  }

  cleanupOldChat(now, keepMinutes = 120) {
    const cutoff = now.getTime() - keepMinutes * 60 * 1000;
    for (const [pid, arr] of this.chatByPlace.entries()) {
      const kept = arr.filter(m => m.time.getTime() >= cutoff);
      this.chatByPlace.set(pid, kept);
    }
  }

  // ---------------- rendering ----------------
  render(ctx, agents, selectedAgentId) {
    const W = ctx.canvas.width;
    const H = ctx.canvas.height;

    // bg
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#0a0c0f";
    ctx.fillRect(0, 0, W, H);

    // grid
    const cell = W / this.width;
    ctx.strokeStyle = "rgba(255,255,255,0.04)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= this.width; i++) {
      const x = i * cell;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    for (let j = 0; j <= this.height; j++) {
      const y = j * cell;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }

    // places
    for (const p of this.places) {
      const px = p.x * cell;
      const py = p.y * cell;
      const pr = p.r * cell;

      ctx.fillStyle = "rgba(122,162,255,0.10)";
      ctx.strokeStyle = "rgba(122,162,255,0.55)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(px, py, pr, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = "rgba(232,238,246,0.85)";
      ctx.font = "12px ui-sans-serif, system-ui, -apple-system";
      ctx.fillText(p.name, px - pr + 6, py - pr + 14);

      ctx.fillStyle = "rgba(154,164,175,0.9)";
      ctx.font = "10px ui-sans-serif, system-ui, -apple-system";
      ctx.fillText(p.id, px - pr + 6, py - pr + 28);
    }

    // agents
    for (const a of agents) {
      const ax = a.x * cell;
      const ay = a.y * cell;
      const r = 0.6 * cell;

      const selected = a.id === selectedAgentId;

      ctx.fillStyle = selected ? "rgba(87,209,139,0.95)" : "rgba(255,255,255,0.88)";
      ctx.strokeStyle = selected ? "rgba(87,209,139,0.95)" : "rgba(0,0,0,0.35)";
      ctx.lineWidth = selected ? 4 : 2;

      ctx.beginPath();
      ctx.arc(ax, ay, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      // initials
      const initials = a.name.split(" ").map(s => s[0]).slice(0,2).join("");
      ctx.fillStyle = "#0a0c0f";
      ctx.font = "bold 12px ui-sans-serif, system-ui, -apple-system";
      ctx.fillText(initials, ax - 8, ay + 4);

      // action label
      if (selected) {
        ctx.fillStyle = "rgba(232,238,246,0.85)";
        ctx.font = "12px ui-sans-serif, system-ui, -apple-system";
        ctx.fillText(a.currentAction || "", ax + 10, ay - 10);
      }
    }
  }
}
