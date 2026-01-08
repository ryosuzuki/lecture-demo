// WebLLM Generative Agents Mini
// - HTML+JS only (client-only). No server backend.
// - Uses WebLLM (MLC) via CDN import.
// - Educational replication inspired by "Generative Agents" paper.

import * as webllm from "https://esm.run/@mlc-ai/web-llm";

import { LLMClient } from "./src/llm.js";
import { World } from "./src/world.js";
import { makeAgents } from "./src/scenario.js";
import { formatSimTime, sleep } from "./src/util.js";

const el = (id) => document.getElementById(id);

const modelSelect = el("modelSelect");
const btnLoad = el("btnLoad");
const btnReset = el("btnReset");
const btnStep = el("btnStep");
const btnRun = el("btnRun");
const btnPause = el("btnPause");
const statusEl = el("status");
const simTimeEl = el("simTime");
const speedEl = el("speed");
const speedLabel = el("speedLabel");
const modeSelect = el("modeSelect");

const agentListEl = el("agentList");
const agentKVEl = el("agentKV");
const thoughtBox = el("thoughtBox");
const planBox = el("planBox");
const memoryBox = el("memoryBox");
const logBox = el("logBox");
const promptBox = el("promptBox");

const canvas = el("worldCanvas");
const ctx = canvas.getContext("2d");

let engine = null;
let llm = null;

let world = null;
let agents = [];
let selectedAgentId = null;

let simTime = null;
let running = false;

// Simulation constants
const MINUTES_PER_TICK = 10;

// --------------------------- UI helpers ---------------------------

function setStatus(text, kind = "muted") {
  statusEl.textContent = text;
  statusEl.className = kind;
}

function updateSpeedLabel() {
  speedLabel.textContent = "x" + speedEl.value;
}

function renderAgentButtons() {
  agentListEl.innerHTML = "";
  for (const a of agents) {
    const btn = document.createElement("button");
    btn.className = "agentBtn" + (a.id === selectedAgentId ? " active" : "");
    btn.innerHTML = `<b>${a.name}</b><div class="muted" style="font-size:12px;">${a.title}</div>`;
    btn.onclick = () => {
      selectedAgentId = a.id;
      renderAgentButtons();
      renderSidePanel();
      renderWorld();
    };
    agentListEl.appendChild(btn);
  }
}

function renderSidePanel() {
  const a = agents.find(x => x.id === selectedAgentId) ?? agents[0];
  if (!a) return;

  // Key-values
  const kv = [
    ["Name", a.name],
    ["Role", a.title],
    ["Location", world?.nearestPlaceName(a.x, a.y) ?? "-"],
    ["Action", a.currentAction ?? "-"],
    ["Energy", String(a.energy)],
    ["Memory count", String(a.memory.records.length)],
    ["Mode", modeSelect.value],
  ];
  agentKVEl.innerHTML = kv.map(([k,v]) => `<div>${k}</div><div>${v}</div>`).join("");

  thoughtBox.textContent = a.lastThought || "(no thought yet)";
  planBox.textContent = a.dailyPlanText() || "(no plan yet)";
  memoryBox.textContent = a.lastRetrievedMemoriesText || "(no retrieval yet)";
  promptBox.textContent = a.lastPromptText || "(no prompt yet)";
}

function renderLog() {
  if (!world) return;
  const items = world.log.slice(-80).reverse();
  logBox.innerHTML = items.map(it => {
    const t = formatSimTime(it.time);
    const place = it.place ? ` <span class="tag">${it.place}</span>` : "";
    return `<div class="logItem"><span class="muted">[${t}]</span> ${it.text}${place}</div>`;
  }).join("");
}

function renderWorld() {
  if (!world) return;
  world.render(ctx, agents, selectedAgentId);
}

// --------------------------- Model select ---------------------------

function pickDefaultModel(modelList) {
  // Heuristic: prefer small parameter models (0.5B/1B/2B/3B) and Instruct.
  const scored = modelList.map(r => {
    const id = r.model_id || r.modelId || r.id || "";
    const m = id.match(/(\d+(?:\.\d+)?)B/i);
    const params = m ? parseFloat(m[1]) : 99;
    const instruct = /instruct|chat/i.test(id) ? 0 : 5;
    const tinyBonus = params <= 1 ? -2 : (params <= 2 ? -1 : 0);
    return { id, score: params + instruct + tinyBonus };
  }).sort((a,b) => a.score - b.score);

  return scored[0]?.id || (modelList[0]?.model_id ?? modelList[0]?.modelId ?? "");
}

function populateModelList() {
  const list = webllm.prebuiltAppConfig.model_list;
  if (!list || list.length === 0) {
    setStatus("No models found in prebuiltAppConfig", "bad");
    return;
  }

  modelSelect.innerHTML = "";
  for (const r of list) {
    const id = r.model_id || r.modelId || r.id;
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = id;
    modelSelect.appendChild(opt);
  }

  modelSelect.value = pickDefaultModel(list);
}

// --------------------------- Lifecycle ---------------------------

async function loadModel() {
  const selectedModel = modelSelect.value;
  setStatus("loading… (download + compile)", "warn");

  btnLoad.disabled = true;

  const initProgressCallback = (p) => {
    // p: { progress, text, timeElapsed, ... }
    // show readable text when available.
    if (p?.text) setStatus(p.text, "warn");
    else if (typeof p?.progress === "number") setStatus(`loading… ${(p.progress*100).toFixed(1)}%`, "warn");
  };

  try {
    engine = await webllm.CreateMLCEngine(selectedModel, { initProgressCallback });
    llm = new LLMClient(engine);
    setStatus(`loaded: ${selectedModel}`, "good");

    btnReset.disabled = false;
    btnStep.disabled = false;
    btnRun.disabled = false;
    btnPause.disabled = false;
  } catch (e) {
    console.error(e);
    setStatus("load failed (check WebGPU support, console)", "bad");
    btnLoad.disabled = false;
    return;
  }

  await resetSim();
  btnLoad.disabled = false;
}

async function resetSim() {
  running = false;

  // Feb 13, 2023 is used in the original paper demo; we keep it for familiarity.
  simTime = new Date("2023-02-13T08:00:00");

  world = new World({ width: 32, height: 32 });
  agents = makeAgents(world);

  selectedAgentId = agents[0]?.id ?? null;

  renderAgentButtons();
  renderWorld();
  renderLog();
  simTimeEl.textContent = formatSimTime(simTime);
  renderSidePanel();

  // Generate daily plans once at reset (requires LLM).
  if (llm) {
    setStatus("initializing agents… (daily plans)", "warn");
    const mode = modeSelect.value;
    for (const a of agents) {
      await a.initialize(llm, world, simTime, { mode });
      renderWorld();
      renderSidePanel();
    }
    setStatus("ready", "good");
  }
}

async function tickOnce() {
  if (!world || !agents.length || !llm) return;

  const mode = modeSelect.value;

  // Advance time
  simTime = new Date(simTime.getTime() + MINUTES_PER_TICK * 60 * 1000);
  simTimeEl.textContent = formatSimTime(simTime);

  // One tick: each agent moves/acts sequentially (single shared model).
  for (const a of agents) {
    await a.step(llm, world, agents, simTime, { mode });
    renderWorld();
    renderLog();
    renderSidePanel();
    // Let UI breathe a little.
    await sleep(0);
  }

  world.cleanupOldChat(simTime);
  renderWorld();
  renderLog();
  renderSidePanel();
}

async function runLoop() {
  setStatus("running…", "good");

  while (running) {
    await tickOnce();
    // Speed: x1..x6 controls real-time delay between ticks.
    // (Not a perfect mapping; local inference cost dominates.)
    const delayMs = 900 / Number(speedEl.value);
    await sleep(delayMs);
  }
  setStatus("paused", "muted");
}

// --------------------------- Event handlers ---------------------------

btnLoad.onclick = loadModel;

btnReset.onclick = async () => {
  if (!llm) return;
  setStatus("resetting…", "warn");
  await resetSim();
  setStatus("ready", "good");
};

btnStep.onclick = async () => {
  if (!llm) return;
  setStatus("stepping…", "warn");
  await tickOnce();
  setStatus("ready", "good");
};

btnRun.onclick = async () => {
  if (!llm) return;
  if (running) return;
  running = true;
  runLoop();
};

btnPause.onclick = () => {
  running = false;
};

speedEl.oninput = () => updateSpeedLabel();
modeSelect.onchange = () => {
  // Reset is not required; mode impacts next prompts.
  renderSidePanel();
};

updateSpeedLabel();
populateModelList();
setStatus("choose model → Load", "muted");
