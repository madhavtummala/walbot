const SVG_NS = "http://www.w3.org/2000/svg";
const BUCKET_NAMES = ["buy", "sell"];
const MAX_AMOUNT = 2000;
//: The algorithm declares which purpose-built editor its Tune screen needs, and the config
//: payload carries the answer. This used to be a hardcoded list of "the DCA algorithms" here,
//: which meant the frontend held an idea of an algorithm family that the backend did not.
const BUDGETS_EDITOR = "budgets";
const WHEEL_STEP = 25;
//: Vertical offset of a bubble's amount label from its centre. Shared so the typing caret
//: lands on the number it is editing rather than near it.
const AMOUNT_LABEL_DY = 14;
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));
let BACKTEST_PERIOD = "4m";
let BACKTEST_LABEL = "4M";

//: Selectable backtest windows. The configured default from status still wins on first load;
//: this only lets you look at a different window without editing config.
const BACKTEST_PERIOD_CHOICES = ["1m", "3m", "6m", "12m", "24m"];

const ENABLED_COLORS = {
  buy: "#024c4a",
  sell: "#7a3800",
};

const DISABLED_COLORS = {
  buy: "#668f8b",
  sell: "#a36d3c",
};

const STRATEGIES = [
  {
    key: "bursty_dca",
    blurb: "Sized by distance from the moving average, paced by how far ahead of plan it already is.",
    name: "Bursty DCA",
    status: "Live",
    horizon: "Continuous",
    risk: "Medium",
    logic: "Accrues a monthly budget per symbol, then sizes each order by two multiplied factors: how many standard deviations the price sits from its moving average, and how far ahead of or behind its plan that symbol already is. Cheap names buy more, rich names buy less or sell, and a symbol that has overspent resists spending again until it catches up. The only hard stop is share rounding.",
    signals: ["Distance from moving average", "Scaling factor per σ", "Backlog resistance", "Monthly cap"],
  },
  {
    key: "rally_rotation",
    blurb: "Ranks the field to pick leaders, but only holds the ones already trending on their own.",
    name: "Rally Rotation",
    status: "Paper",
    horizon: "Daily",
    risk: "High",
    logic: "Two questions, asked separately. Which of these is leading? -- a robust cross-sectional z-score blended over four horizons, one day to twelve, weighted toward the slow end. And is it worth holding at all? -- its own trend line, its own 20- and 60-day returns, and a ceiling on its volatility. The ranking decides the order; the floors decide membership, and a name that fails them is unranked, so it cannot be held however well it scores. That is also how a holding is sold: it stops qualifying, drops out of the ranking, and its slot goes to the next name. When too few qualify the book sits in T-bills rather than in the least bad name. Sizing follows score with no per-name cap, so one qualifying name can take the whole book. Ranking, entry and replacement run on the rerank clock; the single-session crash stop runs every session.",
    signals: ["Cross-sectional rank", "Absolute eligibility", "Volatility ceiling", "Replacement margin", "Crash stop"],
  },
  {
    key: "options_flip",
    blurb: "Buys a predicted intraday low, one contract per symbol, bracketed at the exchange",
    name: "Options Flip",
    status: "Live",
    horizon: "1-2 sessions",
    risk: "High",
    logic: "Reads a multi-day trend per symbol and requires pre-market to confirm it -- disagreement means no trade that day. Picks the nearest contract at least min_dte out inside a delta band, then rests a limit buy priced from how far comparable past sessions pulled back before going the right way, walking it in as the day's budget depletes. On a fill an OCO goes to the broker: a profit limit that only ratchets up, and a stop that never moves. Flat within max_hold_sessions.",
    signals: ["Multi-day trend", "Pre-market confirmation", "Excursion budget", "Delta band and spread", "Exchange-side OCO"],
  },
];

//: A saved strategy id that is unknown (or the retired "none") lands here.
const DEFAULT_ALGORITHM_KEY = "bursty_dca";

//: Strategies driven by the DCA plan, so a plan edit invalidates their cached views.

const state = {
  status: null,
  universe: [],
  controls: {
    trading_account_id: "",
    algorithm_enabled: false,
    active_strategy: "rally_rotation",
  },
  accounts: { rows: [] },
  bot: null,
  //: Which algorithm's plan the bubble board is editing, set when the board renders.
  planStrategy: "",
  //: Which algorithm ``state.nodes`` were built from, so they cannot be synced into another.
  nodesStrategy: "",
  //: "", "saving" or the time of the last successful plan save. The board has no save button
  //: -- it writes on every gesture -- so this is the only feedback that an edit landed.
  planSaveStatus: "",
  layout: null,
  nodes: [],
  invalidNodes: [],
  selected: null,
  drag: null,
  pinch: null,
  touchPointers: new Map(),
  boardPointers: new Map(),
  boardPinch: null,
  draft: null,
  //: The bubble whose budget is being typed, or null. Holds the symbol rather than the node,
  //: because the board rebuilds its nodes on every render and the object would go stale.
  amountEdit: null,
  animationId: null,
  backtests: {},
  backtestLoading: {},
  signals: {},
  signalLoading: {},
  //: Symbols whose gate breakdown is open, by symbol rather than by row index: the table
  //: reorders as a run changes what it decided, and an index would expand the wrong name.
  expandedSignals: new Set(),
  universeProposal: null,
  universeRefreshing: false,
  universeApplying: false,
  deckWheelLocked: false,
  schwabAuth: null,
  renderedBindingKey: "",
  algorithmConfigs: {},
  algorithmConfigLoading: {},
  positions: {},
  positionsLoading: {},
  activity: {},
  activityLoading: {},
  algorithmActivity: {},
  algorithmActivityLoading: {},
};

const $ = (selector) => document.querySelector(selector);

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.remove("show"), 2600);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function api(path, options = {}) {
  const apiBase = window.TRADING_API_BASE || "";
  const timeoutMs = options.timeoutMs ?? 6000;
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const { timeoutMs: _timeoutMs, ...fetchOptions } = options;
    const response = await fetch(`${apiBase}${path}`, {
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      ...fetchOptions,
    });
    // Read as text first. A 500 from FastAPI's default handler is plain "Internal Server
    // Error", not JSON, so parsing before checking the status threw the parser's complaint
    // instead of the server's -- in Safari, "The string did not match the expected pattern",
    // which names neither the request nor the cause.
    const body = await response.text();
    let payload;
    try {
      payload = body ? JSON.parse(body) : {};
    } catch {
      if (!response.ok) throw new Error(`Request failed: ${response.status} ${body.trim().slice(0, 200)}`.trim());
      throw new Error(`${path} returned ${response.status} but not JSON: ${body.trim().slice(0, 200)}`);
    }
    // ``detail`` is what FastAPI's HTTPException produces; ``error`` is what the payload
    // builders return for a failure they handled themselves.
    if (!response.ok) {
      throw new Error(payload.error || payload.detail || `Request failed: ${response.status}`);
    }
    return payload;
  } catch (error) {
    // Every browser words an aborted fetch differently -- "fetch aborted", "The user aborted
    // a request", "signal is aborted without reason" -- and none of them say that *we* gave
    // up waiting, which is the only thing the reader can act on. The request itself is still
    // running on the server, so a retry usually finds the answer cached.
    if (error.name === "AbortError") {
      throw new Error(
        `Timed out after ${Math.round(timeoutMs / 1000)}s. The server may still be working; try again in a moment.`,
      );
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

function money(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return Number(value).toLocaleString(undefined, {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: digits,
  });
}

function num(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return Number(value).toFixed(digits);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}
// DCA is live when it is the selected algorithm and the algorithm bot is on -- the same
// condition as any other strategy, now that it runs on the shared loop.
function bucketColor(bucketName) {
  return isDcaEnabled() ? ENABLED_COLORS[bucketName] : DISABLED_COLORS[bucketName];
}

function bucketStrokeColor(bucketName) {
  return shadeColor(bucketColor(bucketName), isDcaEnabled() ? -48 : -34);
}

//: Which algorithm's plan the board is editing. Its plan is ordinary tuning living at
//: algorithms.<id>.plan, so the board always edits exactly the algorithm whose page you are on.
function planStrategyKey() {
  return state.planStrategy || DEFAULT_ALGORITHM_KEY;
}

//: Whether an algorithm wants the budget board, as reported by /api/algorithm-config. False
//: until that config has loaded, which is the same condition the board's own guards test.
function usesBudgetsEditor(strategyKey) {
  return state.algorithmConfigs[strategyKey]?.tune_editor === BUDGETS_EDITOR;
}

//: The plan object inside the loaded config, created empty if this algorithm has none yet.
//: Returns null until the config has arrived, which is what the board's guards test.
function currentPlan() {
  const config = state.algorithmConfigs[planStrategyKey()]?.config;
  if (!config) return null;
  if (!config.plan || typeof config.plan !== "object") config.plan = {};
  BUCKET_NAMES.forEach((bucketName) => {
    if (!config.plan[bucketName] || typeof config.plan[bucketName] !== "object") {
      config.plan[bucketName] = { amount: 0, items: [] };
    }
    if (!Array.isArray(config.plan[bucketName].items)) config.plan[bucketName].items = [];
  });
  return config.plan;
}

function bucketItems(bucketName) {
  return currentPlan()?.[bucketName]?.items || [];
}

function setBucketItems(bucketName, items) {
  const plan = currentPlan();
  if (!plan) return;
  plan[bucketName].items = items.map((item) => ({
    symbol: item.symbol,
    amount: clamp(Number(item.amount || 0), 0, MAX_AMOUNT),
  }));
  plan[bucketName].amount = plan[bucketName].items.reduce(
    (total, item) => total + Number(item.amount || 0),
    0,
  );
}

function assignedSymbols(bucketName) {
  return new Set(bucketItems(bucketName).map((item) => item.symbol));
}

function itemRadius(amount) {
  return 18 + Math.sqrt(clamp(amount, 0, MAX_AMOUNT) / MAX_AMOUNT) * 24.4;
}

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  Object.entries(attrs).forEach(([key, value]) => {
    if (value !== undefined && value !== null) el.setAttribute(key, value);
  });
  return el;
}

function textEl(attrs, content) {
  const el = svgEl("text", attrs);
  el.textContent = content;
  return el;
}

function calculateLayout() {
  const svg = $("#bubbleBoard");
  if (!svg) return;
  const width = svg.clientWidth || 1200;
  const height = svg.clientHeight || 720;
  const stacked = width < 760;
  const baseR = stacked ? Math.min(width * 0.31, height * 0.17, 185) : Math.min(width * 0.18, height * 0.32, 257);
  const maxR = stacked ? Math.min(width * 0.38, height * 0.21, 229) : Math.min(width * 0.24, height * 0.39, 321);

  state.layout = {
    width,
    height,
    stacked,
    buckets: stacked
      ? {
        buy: { cx: width / 2, cy: height * 0.27, r: baseR, baseR, maxR, label: "BUY" },
        sell: { cx: width / 2, cy: height * 0.72, r: baseR, baseR, maxR, label: "SELL" },
      }
      : {
        buy: { cx: width * 0.29, cy: height * 0.51, r: baseR, baseR, maxR, label: "BUY" },
        sell: { cx: width * 0.71, cy: height * 0.51, r: baseR, baseR, maxR, label: "SELL" },
      },
  };
}

function fitBucketRadii() {
  BUCKET_NAMES.forEach((bucketName) => {
    const bucket = state.layout.buckets[bucketName];
    const areaRadius = Math.sqrt(
      bucketItems(bucketName).reduce((sum, item) => sum + (itemRadius(item.amount) + 12) ** 2, 0),
    );
    bucket.r = clamp(Math.max(bucket.baseR, areaRadius * 1.22 + 42), bucket.baseR, bucket.maxR);
  });
}

function pointFromPosition(position, bucket, radius) {
  const usable = Math.max(bucket.r - radius - 18, 1);
  return {
    x: bucket.cx + clamp(Number(position?.x || 0), -1, 1) * usable,
    y: bucket.cy + clamp(Number(position?.y || 0), -1, 1) * usable,
  };
}

function fallbackPosition(index, count, bucketName) {
  if (count <= 1) return { x: 0, y: 0 };
  const ring = Math.sqrt((index + 1) / (count + 1));
  const angle = index * GOLDEN_ANGLE + (bucketName === "sell" ? 0.9 : 0);
  return {
    x: Math.cos(angle) * ring * 0.68,
    y: Math.sin(angle) * ring * 0.68,
  };
}

function clampPointToBucket(point, bucketName, radius) {
  const bucket = state.layout.buckets[bucketName];
  const dx = point.x - bucket.cx;
  const dy = point.y - bucket.cy;
  const maxDistance = Math.max(bucket.r - radius - 18, 1);
  const currentDistance = Math.hypot(dx, dy);
  if (currentDistance <= maxDistance) return { x: point.x, y: point.y };
  const scale = maxDistance / Math.max(currentDistance, 1);
  return {
    x: bucket.cx + dx * scale,
    y: bucket.cy + dy * scale,
  };
}

function distance(point, bucket) {
  return Math.hypot(point.x - bucket.cx, point.y - bucket.cy);
}

function nearestBucket(point) {
  const buckets = state.layout.buckets;
  return distance(point, buckets.buy) <= distance(point, buckets.sell) ? "buy" : "sell";
}

function bucketAtPoint(point) {
  const bucketName = nearestBucket(point);
  const bucket = state.layout.buckets[bucketName];
  return distance(point, bucket) <= bucket.r + 44 ? bucketName : null;
}

function eventToSvgPoint(event) {
  const svg = $("#bubbleBoard");
  const rect = svg.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * state.layout.width,
    y: ((event.clientY - rect.top) / rect.height) * state.layout.height,
  };
}

function organicPath(bucketName, phase = 0) {
  const bucket = state.layout.buckets[bucketName];
  const nodes = state.nodes.filter((node) => node.bucketName === bucketName);
  const points = [];
  const pointCount = 112;
  for (let index = 0; index < pointCount; index += 1) {
    const angle = (index / pointCount) * Math.PI * 2;
    let radius =
      bucket.r +
      Math.sin(angle * 2.1 + phase) * bucket.r * 0.025 +
      Math.cos(angle * 3.4 - phase * 0.7) * bucket.r * 0.02;

    nodes.forEach((node) => {
      const nodeAngle = Math.atan2(node.y - bucket.cy, node.x - bucket.cx);
      const delta = Math.atan2(Math.sin(angle - nodeAngle), Math.cos(angle - nodeAngle));
      const influence = Math.exp(-(delta ** 2) / 0.15);
      const reach = Math.hypot(node.x - bucket.cx, node.y - bucket.cy) + node.radius + 24;
      radius = Math.max(radius, bucket.r + Math.max(0, reach - bucket.r) * influence + node.radius * 0.08 * influence);
    });

    radius = Math.min(radius, bucket.maxR + 58);
    points.push([bucket.cx + Math.cos(angle) * radius, bucket.cy + Math.sin(angle) * radius]);
  }
  return smoothClosedPath(points);
}

function smoothClosedPath(points) {
  if (!points.length) return "";
  const commands = [`M ${points[0][0].toFixed(2)} ${points[0][1].toFixed(2)}`];
  points.forEach((point, index) => {
    const next = points[(index + 1) % points.length];
    const midX = (point[0] + next[0]) / 2;
    const midY = (point[1] + next[1]) / 2;
    commands.push(`Q ${point[0].toFixed(2)} ${point[1].toFixed(2)} ${midX.toFixed(2)} ${midY.toFixed(2)}`);
  });
  return `${commands.join(" ")} Z`;
}

function buildNodes() {
  const previous = new Map(state.nodes.map((node) => [`${node.bucketName}:${node.symbol}`, node]));
  state.nodes = [];
  // Nodes belong to the plan they were built from. Recorded so nothing can write one
  // algorithm's bubbles into another's plan -- see syncNodesToPlan.
  state.nodesStrategy = planStrategyKey();
  BUCKET_NAMES.forEach((bucketName) => {
    const bucket = state.layout.buckets[bucketName];
    const items = bucketItems(bucketName);
    items.forEach((item, index) => {
      const radius = itemRadius(item.amount);
      const id = `${bucketName}:${item.symbol}`;
      const oldNode = previous.get(id);
      const fallback = fallbackPosition(index, items.length, bucketName);
      const point = oldNode || pointFromPosition(fallback, bucket, radius);
      const clamped = clampPointToBucket(point, bucketName, radius);
      state.nodes.push({
        id,
        symbol: item.symbol,
        name: item.name,
        amount: clamp(Number(item.amount || 0), 0, MAX_AMOUNT),
        bucketName,
        targetBucket: bucketName,
        radius,
        x: clamped.x,
        y: clamped.y,
        vx: oldNode?.vx || 0,
        vy: oldNode?.vy || 0,
      });
    });
  });
}

function syncNodeToPlan(node) {
  const item = bucketItems(node.bucketName).find((candidate) => candidate.symbol === node.symbol);
  if (!item) return;
  const clamped = clampPointToBucket({ x: node.x, y: node.y }, node.bucketName, node.radius);
  node.x = clamped.x;
  node.y = clamped.y;
  item.amount = clamp(node.amount, 0, MAX_AMOUNT);
}

function syncNodesToPlan() {
  // Nodes are the board's working copy of one algorithm's plan. Writing them into a
  // different algorithm's plan overwrites it with budgets the reader never typed there --
  // which is what happened on every navigation between two DCA pages, because renderDca
  // syncs before it rebuilds. The nodes are rebuilt from the new plan a moment later, so
  // there is nothing here worth carrying across.
  if (state.nodesStrategy !== planStrategyKey()) return;
  state.nodes.forEach(syncNodeToPlan);
  BUCKET_NAMES.forEach((bucketName) => setBucketItems(bucketName, bucketItems(bucketName)));
}

function renderBoard() {
  if (!currentPlan() || !$("#bubbleBoard")) return;
  window.cancelAnimationFrame(state.animationId);
  document.body.classList.toggle("dca-off", !isDcaEnabled());
  calculateLayout();
  fitBucketRadii();
  buildNodes();

  const svg = $("#bubbleBoard");
  svg.setAttribute("viewBox", `0 0 ${state.layout.width} ${state.layout.height}`);
  svg.classList.toggle("resize-mode", Boolean(state.selected));
  svg.replaceChildren();

  BUCKET_NAMES.forEach((bucketName) => {
    const bucket = state.layout.buckets[bucketName];
    const color = bucketColor(bucketName);
    const blob = svgEl("path", {
      id: `${bucketName}-blob`,
      class: `bucket-blob ${bucketName}`,
      fill: color,
      stroke: bucketStrokeColor(bucketName),
      d: organicPath(bucketName),
    });
    svg.appendChild(blob);
    svg.appendChild(textEl({
      class: "bucket-label",
      x: bucket.cx,
      y: bucket.cy - 14,
      "text-anchor": "middle",
      fill: color,
    }, bucket.label));
    svg.appendChild(textEl({
      id: `${bucketName}-total`,
      class: "bucket-total",
      x: bucket.cx,
      y: bucket.cy + 24,
      "text-anchor": "middle",
      fill: color,
    }, isDcaEnabled() ? money(bucketItems(bucketName).reduce((sum, item) => sum + Number(item.amount || 0), 0)) : "DCA off"));
  });

  state.nodes.forEach((node) => renderAsset(svg, node, "live"));
  state.invalidNodes = state.invalidNodes.filter((node) => node.expiresAt > Date.now());
  state.invalidNodes.forEach((node) => renderInvalidAsset(svg, node));
  if (state.draft) renderDraft(svg, state.draft);

  svg.ondblclick = handleBoardDoubleClick;
  svg.onpointerdown = startBoardPointer;
  svg.onpointermove = handleBoardPointerMove;
  svg.onpointerup = endBoardPointer;
  svg.onpointercancel = endBoardPointer;
  startAnimation();
}

function renderAsset(svg, node, extraClass) {
  const group = svgEl("g", {
    class: `asset ${extraClass} ${state.selected === node.symbol ? "selected" : ""}`,
    transform: `translate(${node.x}, ${node.y})`,
    "data-id": node.id,
  });
  group.appendChild(svgEl("circle", { r: node.radius, fill: bucketColor(node.bucketName) }));
  group.appendChild(textEl({ class: "symbol-label", y: -5 }, node.symbol));
  group.appendChild(textEl({ class: "amount-label", y: AMOUNT_LABEL_DY }, `$${Math.round(node.amount)}`));
  group.addEventListener("pointerdown", (event) => startAssetPointer(event, node));
  group.addEventListener("wheel", (event) => resizeNode(event, node), { passive: false });
  group.addEventListener("dblclick", (event) => {
    event.stopPropagation();
    removeSymbol(node.bucketName, node.symbol);
  });
  svg.appendChild(group);
}

function renderDraft(svg, node) {
  const group = svgEl("g", { class: "asset draft", transform: `translate(${node.x}, ${node.y})` });
  group.appendChild(svgEl("circle", { r: node.radius, fill: bucketColor(node.bucketName) }));
  // show nothing in-SVG when drafting; the visible caret is provided by the positioned #symbolEntry input
  group.appendChild(textEl({ class: "symbol-label", y: -5 }, node.symbol || ""));
  group.appendChild(textEl({ class: "amount-label", y: 14 }, "$25"));
  svg.appendChild(group);
}

function renderInvalidAsset(svg, node) {
  const group = svgEl("g", { class: "asset invalid", transform: `translate(${node.x}, ${node.y})` });
  group.appendChild(svgEl("circle", { r: node.radius, fill: "#b42318" }));
  group.appendChild(textEl({ class: "symbol-label", y: -5 }, node.symbol));
  // Do not render any "not found" text inside the invalid asset; red circle + vanish is sufficient
  svg.appendChild(group);
}

function updateBoardElements(phase = 0) {
  BUCKET_NAMES.forEach((bucketName) => {
    const blob = $(`#${bucketName}-blob`);
    const total = $(`#${bucketName}-total`);
    if (blob) blob.setAttribute("d", organicPath(bucketName, phase));
    if (total) {
      total.textContent = isDcaEnabled()
        ? money(bucketItems(bucketName).reduce((sum, item) => sum + Number(item.amount || 0), 0))
        : "DCA off";
    }
  });
  state.nodes.forEach((node) => {
    const group = Array.from(document.querySelectorAll(".asset.live")).find((element) => element.dataset.id === node.id);
    if (!group) return;
    group.setAttribute("transform", `translate(${node.x}, ${node.y})`);
    group.querySelector("circle")?.setAttribute("r", node.radius);
    const amount = group.querySelector(".amount-label");
    if (amount) amount.textContent = `$${Math.round(node.amount)}`;
  });
}

function startAnimation() {
  const tick = () => {
    stepPhysics();
    updateBoardElements(performance.now() / 1050);
    state.animationId = window.requestAnimationFrame(tick);
  };
  state.animationId = window.requestAnimationFrame(tick);
}

function stepPhysics() {
  BUCKET_NAMES.forEach((bucketName) => {
    const bucket = state.layout.buckets[bucketName];
    const nodes = state.nodes.filter((node) => node.bucketName === bucketName);
    nodes.forEach((node, index) => {
      if (state.drag?.node === node) return;
      node.vx += (bucket.cx - node.x) * 0.002;
      node.vy += (bucket.cy - node.y) * 0.002;
      for (let otherIndex = index + 1; otherIndex < nodes.length; otherIndex += 1) {
        const other = nodes[otherIndex];
        const dx = other.x - node.x;
        const dy = other.y - node.y;
        const dist = Math.max(Math.hypot(dx, dy), 0.1);
        const minDist = node.radius + other.radius + 8;
        if (dist < minDist) {
          const push = (minDist - dist) / dist * 0.018;
          const px = dx * push;
          const py = dy * push;
          if (state.drag?.node !== node) {
            node.vx -= px;
            node.vy -= py;
          }
          if (state.drag?.node !== other) {
            other.vx += px;
            other.vy += py;
          }
        }
      }
      node.vx *= 0.86;
      node.vy *= 0.86;
      node.x += node.vx;
      node.y += node.vy;
      const clamped = clampPointToBucket({ x: node.x, y: node.y }, bucketName, node.radius);
      node.x = clamped.x;
      node.y = clamped.y;
      syncNodeToPlan(node);
    });
  });
}

function pointersForNode(node) {
  return Array.from(state.touchPointers.entries())
    .filter(([_id, pointer]) => pointer.node === node)
    .map(([id, pointer]) => ({ id, ...pointer }));
}

function pinchDistanceForPointers(pointers) {
  if (pointers.length < 2) return 0;
  return Math.hypot(pointers[0].x - pointers[1].x, pointers[0].y - pointers[1].y);
}

function beginPinch(node, element) {
  const pointers = pointersForNode(node).slice(-2);
  const startDistance = pinchDistanceForPointers(pointers);
  if (startDistance <= 0) return;
  if (state.drag?.element) state.drag.element.classList.remove("dragging");
  state.drag = null;
  state.pinch = {
    node,
    element,
    pointerIds: pointers.map((pointer) => pointer.id),
    startDistance,
    startAmount: node.amount,
  };
  element.classList.add("pinching");
}

function startAssetPointer(event, node) {
  event.preventDefault();
  // Capture first. Everything below can re-render the board, and a re-render replaces this
  // group element -- calling setPointerCapture on the detached node throws InvalidStateError
  // and the drag never starts.
  event.currentTarget.setPointerCapture(event.pointerId);
  hideSymbolEntry();
  // Commit whatever was being typed on the previous bubble before the selection moves.
  commitAmountEntry();
  const wasSelected = state.selected === node.symbol;
  state.selected = wasSelected ? null : node.symbol;
  $("#bubbleBoard")?.classList.toggle("resize-mode", Boolean(state.selected));
  if (event.pointerType === "touch") updateBoardElements();
  event.currentTarget.onpointermove = (moveEvent) => handleAssetPointerMove(moveEvent, node);
  event.currentTarget.onpointerup = (upEvent) => endAssetPointer(upEvent, node);
  event.currentTarget.onpointercancel = (upEvent) => endAssetPointer(upEvent, node);

  if (event.pointerType === "touch") {
    state.touchPointers.set(event.pointerId, {
      x: event.clientX,
      y: event.clientY,
      node,
      element: event.currentTarget,
    });
    if (pointersForNode(node).length >= 2) {
      beginPinch(node, event.currentTarget);
      return;
    }
  }

  // ``moved`` separates a click from a drag: only the former opens the budget caret.
  state.drag = { node, pointerId: event.pointerId, element: event.currentTarget, moved: false };
  event.currentTarget.classList.add("dragging");
}

function updateTouchPointer(event, node) {
  if (event.pointerType !== "touch" || !state.touchPointers.has(event.pointerId)) return;
  const current = state.touchPointers.get(event.pointerId);
  state.touchPointers.set(event.pointerId, {
    ...current,
    x: event.clientX,
    y: event.clientY,
    node,
  });
}

//: Set a budget to exactly what was asked for, in whole dollars. Scrolling and pinching go
//: through resizeNodeToAmount instead, which snaps to the WHEEL_STEP grid -- a gesture has no
//: business expressing $310, but a typed number means precisely what it says, so rounding it
//: to the nearest $25 would silently discard what the reader just asked for.
function setNodeAmount(node, amount) {
  node.amount = clamp(Math.round(amount), 0, MAX_AMOUNT);
  node.radius = itemRadius(node.amount);
  syncNodeToPlan(node);
  schedulePlanSave();
  syncAmountEntryTo(node);
}

//: Keep an open budget field showing what the bubble now holds. Scrolling and pinching change
//: the amount behind the caret, and committing afterwards would otherwise write back the value
//: the field was opened with -- undoing the gesture that had just been made.
function syncAmountEntryTo(node) {
  if (state.amountEdit?.symbol !== node.symbol) return;
  const entry = $("#amountEntry");
  if (!entry || document.activeElement === entry) return;
  entry.value = String(Math.round(node.amount));
}

function resizeNodeToAmount(node, amount) {
  setNodeAmount(node, clamp(Math.round(amount / WHEEL_STEP) * WHEEL_STEP, 0, MAX_AMOUNT));
}

function selectedDcaNode() {
  return state.nodes.find((node) => node.symbol === state.selected) || null;
}

function boardPointers() {
  return Array.from(state.boardPointers.entries()).map(([id, pointer]) => ({ id, ...pointer }));
}

function beginBoardPinch() {
  const node = selectedDcaNode();
  const pointers = boardPointers().slice(-2);
  const startDistance = pinchDistanceForPointers(pointers);
  if (!node || startDistance <= 0) return;
  state.boardPinch = {
    node,
    pointerIds: pointers.map((pointer) => pointer.id),
    startDistance,
    startAmount: node.amount,
  };
}

function startBoardPointer(event) {
  if (event.pointerType !== "touch" || !state.selected || event.target.closest(".asset")) return;
  event.preventDefault();
  event.currentTarget.setPointerCapture(event.pointerId);
  state.boardPointers.set(event.pointerId, {
    x: event.clientX,
    y: event.clientY,
  });
  if (state.boardPointers.size >= 2) beginBoardPinch();
}

function handleBoardPointerMove(event) {
  if (event.pointerType !== "touch" || !state.boardPointers.has(event.pointerId)) return;
  event.preventDefault();
  state.boardPointers.set(event.pointerId, {
    x: event.clientX,
    y: event.clientY,
  });
  if (!state.boardPinch) return;
  const pointers = state.boardPinch.pointerIds
    .map((id) => state.boardPointers.get(id))
    .filter(Boolean);
  const currentDistance = pinchDistanceForPointers(pointers);
  if (currentDistance <= 0) return;
  resizeNodeToAmount(state.boardPinch.node, state.boardPinch.startAmount + ((currentDistance - state.boardPinch.startDistance) / 6));
  updateBoardElements();
}

function endBoardPointer(event) {
  if (event.pointerType !== "touch") return;
  state.boardPointers.delete(event.pointerId);
  if (state.boardPinch && state.boardPointers.size < 2) {
    state.boardPinch = null;
    renderDca();
  }
}

function handleAssetPointerMove(event, node) {
  event.preventDefault();
  updateTouchPointer(event, node);
  if (state.pinch?.node === node) {
    const pointers = state.pinch.pointerIds
      .map((id) => state.touchPointers.get(id))
      .filter(Boolean);
    const currentDistance = pinchDistanceForPointers(pointers);
    if (currentDistance > 0) {
      resizeNodeToAmount(node, state.pinch.startAmount + ((currentDistance - state.pinch.startDistance) / 6));
      updateBoardElements();
    }
    return;
  }
  if (state.drag?.node === node && state.drag.pointerId === event.pointerId) {
    state.drag.moved = true;
    dragNode(event, node);
  }
}

//: Dragging inside a bucket moves the bubble between buckets; dragging clear of both is how a
//: symbol leaves the plan. The bubble follows the pointer unclamped out there and greys out, so
//: the drop is committed only once you let go somewhere that is not a bucket.
function dragNode(event, node) {
  const point = eventToSvgPoint(event);
  const bucketName = bucketAtPoint(point);
  node.pendingRemoval = !bucketName;
  if (bucketName) {
    const clamped = clampPointToBucket(point, bucketName, node.radius);
    node.targetBucket = bucketName;
    node.bucketName = bucketName;
    node.x = clamped.x;
    node.y = clamped.y;
  } else {
    node.x = point.x;
    node.y = point.y;
  }
  node.vx = 0;
  node.vy = 0;
  state.drag?.element?.classList.toggle("removing", node.pendingRemoval);
}

function endAssetPointer(event, node) {
  if (event.pointerType === "touch") state.touchPointers.delete(event.pointerId);
  if (state.pinch?.node === node) {
    const remaining = pointersForNode(node);
    if (remaining.length >= 2) {
      beginPinch(node, state.pinch.element);
      return;
    }
    state.pinch.element.classList.remove("pinching");
    state.pinch = null;
    renderDca();
    return;
  }
  if (state.drag?.node === node && state.drag.pointerId === event.pointerId) {
    const moved = state.drag.moved;
    event.currentTarget.classList.remove("dragging", "removing");
    state.drag = null;
    if (node.pendingRemoval) {
      // The caret would otherwise sit over a bubble that is about to stop existing, and
      // committing it would write the removed symbol's budget back into the plan.
      hideAmountEntry();
      node.pendingRemoval = false;
      state.selected = null;
      removeSymbol(node.bucketName, node.symbol);
      showToast(`${node.symbol} removed`);
      return;
    }
    moveAsset(node);
    renderDca();
    // A click that selected this bubble puts the caret straight onto its budget, with the
    // current value selected so typing replaces it. Waiting for the first keystroke worked --
    // the digit was carried into the field -- but showed nothing to say that typing would do
    // anything, so the affordance was invisible.
    if (!moved) {
      if (state.selected === node.symbol) showAmountEntry(node);
      else hideAmountEntry();
    }
  }
}

function resizeNode(event, node) {
  event.preventDefault();
  event.stopPropagation();
  const direction = event.deltaY > 0 ? 1 : -1;
  resizeNodeToAmount(node, Math.round(node.amount / WHEEL_STEP) * WHEEL_STEP + direction * WHEEL_STEP);
  renderDca();
}

function handleBoardDoubleClick(event) {
  if (event.target.closest(".asset")) return;
  const point = eventToSvgPoint(event);
  const bucketName = bucketAtPoint(point);
  if (!bucketName) return;
  showSymbolEntry(bucketName, point);
}

function showSymbolEntry(bucketName, point) {
  const radius = itemRadius(25);
  const clamped = clampPointToBucket(point, bucketName, radius);
  state.draft = { bucketName, x: clamped.x, y: clamped.y, radius, symbol: "" };
  renderBoard();

  const entry = $("#symbolEntry");
  const rect = $("#bubbleBoard").getBoundingClientRect();
  entry.value = "";
  entry.style.left = `${window.scrollX + rect.left + (clamped.x / state.layout.width) * rect.width}px`;
  // nudge vertically so the caret aligns with the SVG symbol-label (which is slightly above center)
  entry.style.top = `${window.scrollY + rect.top + (clamped.y / state.layout.height) * rect.height - 6}px`;
  entry.className = "show";
  window.setTimeout(() => entry.focus(), 0);
}

function hideSymbolEntry(cancelDraft = true) {
  const entry = $("#symbolEntry");
  const hadEntry = entry.classList.contains("show") || Boolean(state.draft);
  entry.className = "";
  if (cancelDraft && hadEntry) {
    state.draft = null;
    renderBoard();
  }
}

// =========================================================================================
// Typing a budget. Scrolling a bubble is quick but imprecise -- reaching $1,600 from $25 is
// 63 notches, and the value can only ever land on a multiple of WHEEL_STEP. Selecting a
// bubble and typing sets it outright.
// =========================================================================================

//: Places the caret over ``node``'s amount label and opens the edit. The input itself is
//: invisible: the number being typed is drawn by the bubble, the same way a new bubble draws
//: the symbol being typed into #symbolEntry.
//:
//: ``seed`` is the first character when the edit was started by typing a digit, because
//: focusing an input during the keydown that caused it does not deliver that keystroke. An
//: edit started with Enter seeds the current amount and selects it, so typing replaces.
function showAmountEntry(node, seed = "") {
  const board = $("#bubbleBoard");
  if (!board || !node || !state.layout) return;
  const entry = $("#amountEntry");
  const rect = board.getBoundingClientRect();
  const scaleX = rect.width / state.layout.width;
  const scaleY = rect.height / state.layout.height;
  state.amountEdit = { symbol: node.symbol, originalAmount: node.amount };
  entry.value = seed || String(Math.round(node.amount));
  entry.style.left = `${window.scrollX + rect.left + node.x * scaleX}px`;
  // AMOUNT_LABEL_DY down from the bubble's centre, so the caret sits on the number it edits.
  entry.style.top = `${window.scrollY + rect.top + (node.y + AMOUNT_LABEL_DY) * scaleY}px`;
  entry.className = "show";
  window.setTimeout(() => {
    entry.focus();
    if (seed) entry.setSelectionRange(entry.value.length, entry.value.length);
    else entry.select();
  }, 0);
  previewAmountEntry();
}

//: Write each keystroke straight through to the plan, exactly as scrolling does. The bubble
//: label, its radius and the bucket total then all follow the typing, and there is no second
//: copy of the number anywhere to disagree with it. The save is debounced, so this costs one
//: request once the typing stops rather than one per character.
function previewAmountEntry() {
  if (!state.amountEdit) return;
  const entry = $("#amountEntry");
  const digits = entry.value.replace(/[^0-9]/g, "").slice(0, 7);
  if (entry.value !== digits) entry.value = digits;
  const node = state.nodes.find((candidate) => candidate.symbol === state.amountEdit.symbol);
  if (!node) return;
  setNodeAmount(node, Number(digits || 0));
  updateBoardElements();
}

function hideAmountEntry() {
  const entry = $("#amountEntry");
  if (!entry) return;
  entry.className = "";
  state.amountEdit = null;
}

//: Escape puts the budget back where it was. The preview has already been written to the
//: plan, so abandoning has to be an edit of its own rather than simply closing the field.
function cancelAmountEntry() {
  const edit = state.amountEdit;
  hideAmountEntry();
  if (!edit) return;
  const node = state.nodes.find((candidate) => candidate.symbol === edit.symbol);
  if (node) setNodeAmount(node, edit.originalAmount);
  renderBoard();
}

function commitAmountEntry() {
  const entry = $("#amountEntry");
  // Guarded rather than assumed: blur, Enter and a click elsewhere can all arrive for one
  // edit, and only the first of them should be the one that closes it.
  if (!state.amountEdit || !entry?.classList.contains("show")) return;
  const edit = state.amountEdit;
  const digits = entry.value.replace(/[^0-9]/g, "");
  hideAmountEntry();

  const node = state.nodes.find((candidate) => candidate.symbol === edit.symbol);
  if (!node) return;
  if (digits === "") {
    // Cleared and confirmed reads as "never mind", not as a budget of zero -- type 0 for that.
    setNodeAmount(node, edit.originalAmount);
  } else if (Number(digits) > MAX_AMOUNT) {
    // Already clamped on screen; say why, so a smaller number than was typed is not a mystery.
    showToast(`${edit.symbol} capped at ${money(MAX_AMOUNT)}/month`);
  }
  renderBoard();
}

function commitSymbolEntry() {
  const entry = $("#symbolEntry");
  if (!state.draft || !entry.classList.contains("show")) return;
  const symbol = entry.value.trim().toUpperCase();
  const draft = { ...state.draft };
  entry.className = "";
  state.draft = null;
  if (!symbol) {
    renderBoard();
    return;
  }

  const row = state.universe.find((item) => item.symbol === symbol);
  const isDuplicate = assignedSymbols(draft.bucketName).has(symbol);
  if (!row || !row.enabled || isDuplicate) {
    state.invalidNodes.push({
      id: `invalid:${symbol}:${Date.now()}`,
      symbol,
      bucketName: draft.bucketName,
      x: draft.x,
      y: draft.y,
      radius: draft.radius,
      expiresAt: Date.now() + 3000,
    });
    renderBoard();
    showToast(isDuplicate ? `${symbol} already in bucket` : `${symbol} is not allowed to trade`);
    window.setTimeout(renderBoard, 3100);
    return;
  }

  const amount = 25;
  currentPlan()[draft.bucketName].items.push({
    symbol: row.symbol,
    amount,
  });
  setBucketItems(draft.bucketName, currentPlan()[draft.bucketName].items);
  renderDca();
  schedulePlanSave();
  showToast(`${row.symbol} added`);
}

function removeSymbol(bucketName, symbol) {
  setBucketItems(
    bucketName,
    bucketItems(bucketName).filter((item) => item.symbol !== symbol),
  );
  renderDca();
  schedulePlanSave();
}

function moveAsset(node) {
  const fromItems = BUCKET_NAMES.flatMap((bucketName) =>
    bucketItems(bucketName).map((item) => ({ bucketName, item })),
  );
  const found = fromItems.find(({ item }) => item.symbol === node.symbol);
  if (!found) return;
  BUCKET_NAMES.forEach((bucketName) => {
    currentPlan()[bucketName].items = bucketItems(bucketName).filter((item) => item.symbol !== node.symbol);
  });
  found.item.amount = node.amount;
  currentPlan()[node.bucketName].items.push(found.item);
  BUCKET_NAMES.forEach((bucketName) => setBucketItems(bucketName, bucketItems(bucketName)));
  schedulePlanSave();
}

function renderDca() {
  if (!currentPlan()) return;
  syncNodesToPlan();
  renderBoard();
}

function dateTicks(rows, zoom) {
  if (rows.length <= 2) return rows;
  const tickCount = zoom >= 4 ? 6 : zoom >= 2 ? 5 : 4;
  const step = Math.max(1, Math.floor((rows.length - 1) / (tickCount - 1)));
  const ticks = rows.filter((_, index) => index % step === 0);
  if (ticks.at(-1) !== rows.at(-1)) ticks.push(rows.at(-1));
  return ticks;
}

function formatDateTick(date, zoom) {
  const options = zoom >= 4 ? { month: "short", day: "numeric" } : { month: "short" };
  return date.toLocaleDateString(undefined, options);
}

//: Whether a backtest's rows are finer than one a day. Asked of the data rather than of the
//: strategy, because it is the replay grid that decides: the same algorithm steps daily or per
//: bar depending on whether it declared an intraday lookback.
function backtestIsIntraday(rows) {
  return rows.some((row, index) => index > 0
    && row.date.getTime() - rows[index - 1].date.getTime() < 20 * 60 * 60 * 1000);
}

//: A date label on a daily chart, a time label on an intraday one. Without the second case
//: every tick and every hover in a session reads as the same date, which is exactly as useful
//: as the stacked x-positions this replaced.
function formatChartStamp(date, intraday) {
  if (!intraday) return formatDateTick(date, 4);
  return date.toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
}

function algorithmChoices() {
  return STRATEGIES;
}

//: Bindings are pairs of algorithm and account. Signals and backtests are keyed by strategy,
//: but they are *computed* per account: a DCA plan is per account, so the account is not only
//: an execution detail. ``accountForStrategy`` is the frontend half of the same rule the API
//: applies in ``controls.account_for_strategy`` -- both pick the same binding, so the plan the
//: board edits is the plan the signal view and the backtest read.
function bindings() {
  return state.controls?.bindings || [];
}

function accountForStrategy(strategyKey) {
  const candidates = bindings().filter((binding) => binding.strategy === strategyKey);
  if (!candidates.length) return "";
  return (candidates.find((binding) => binding.enabled) || candidates[0]).account_id || "";
}

function bindingById(bindingId) {
  return bindings().find((binding) => String(binding.id) === String(bindingId)) || null;
}

function isDcaEnabled() {
  // The board edits one algorithm's plan, so it is that algorithm's bindings that light it up.
  return bindings().some((binding) => binding.enabled && binding.strategy === planStrategyKey());
}

function configFieldKind(value) {
  if (typeof value === "boolean") return "bool";
  if (typeof value === "number") return "number";
  if (typeof value === "string") return "string";
  if (Array.isArray(value) && value.every((item) => typeof item !== "object" || item === null)) return "list";
  return "json";
}

function renderConfigField(key, value, doc) {
  const kind = configFieldKind(value);
  const help = doc
    ? `<em class="fieldWhat">${escapeHtml(doc.what)}</em><em class="fieldEffect">${escapeHtml(doc.effect)}</em>`
    : "";
  const label = `<span title="${escapeHtml(key)}">${escapeHtml(key)}</span>${help}`;
  const attrs = `data-config-key="${escapeHtml(key)}" data-config-kind="${kind}"`;
  if (kind === "bool") {
    return `<label class="configField">${label}
      <input type="checkbox" ${attrs}${value ? " checked" : ""} />
    </label>`;
  }
  if (kind === "number") {
    // `any` keeps fractional knobs like 0.95 editable; integers still type naturally.
    return `<label class="configField">${label}
      <input type="number" step="any" ${attrs} value="${escapeHtml(String(value))}" />
    </label>`;
  }
  if (kind === "string") {
    return `<label class="configField">${label}
      <input type="text" ${attrs} value="${escapeHtml(value)}" />
    </label>`;
  }
  if (kind === "list") {
    return `<label class="configField configField--wide">${label}
      <textarea rows="2" ${attrs}>${escapeHtml(value.join(", "))}</textarea>
    </label>`;
  }
  // Nested structures have no sensible widget, so they keep a JSON box of their own.
  return `<label class="configField configField--wide">${label}
    <textarea rows="3" ${attrs}>${escapeHtml(JSON.stringify(value, null, 1))}</textarea>
  </label>`;
}

function collectConfigValues(host) {
  const values = {};
  host.querySelectorAll("[data-config-key]").forEach((field) => {
    const key = field.dataset.configKey;
    const kind = field.dataset.configKind;
    if (kind === "bool") {
      values[key] = field.checked;
      return;
    }
    if (kind === "number") {
      const parsed = Number(field.value);
      if (Number.isNaN(parsed)) throw new Error(`${key} must be a number`);
      values[key] = parsed;
      return;
    }
    if (kind === "list") {
      values[key] = field.value.split(/[\s,]+/).map((item) => item.trim()).filter(Boolean);
      return;
    }
    if (kind === "json") {
      try {
        values[key] = JSON.parse(field.value);
      } catch (error) {
        throw new Error(`${key} is not valid JSON`);
      }
      return;
    }
    values[key] = field.value;
  });
  return values;
}

function strategyByKey(strategyKey) {
  return algorithmChoices().find((choice) => choice.key === strategyKey) || STRATEGIES[0];
}

function isBacktestPayload(payload) {
  return Boolean(payload && typeof payload === "object" && Array.isArray(payload.rows));
}

function normalizeBacktestPeriod(period) {
  const normalized = String(period || "").trim().toLowerCase();
  return /^[1-9][0-9]*m$/.test(normalized) ? normalized : "4m";
}

function backtestPeriodLabel(period) {
  const match = normalizeBacktestPeriod(period).match(/^([1-9][0-9]*)m$/);
  return match ? `${match[1]}M` : "4M";
}

function configureBacktestPeriod(period) {
  const normalized = normalizeBacktestPeriod(period);
  const label = backtestPeriodLabel(normalized);
  if (normalized !== BACKTEST_PERIOD) {
    state.backtests = {};
  }
  BACKTEST_PERIOD = normalized;
  BACKTEST_LABEL = label;
}

async function loadBacktest(strategyKey, refresh, options = {}) {
  const cacheOnly = Boolean(options.cacheOnly);
  let changed = false;
  if (!cacheOnly) state.backtestLoading[strategyKey] = true;
  if (!cacheOnly && state.backtests[strategyKey]?.error) delete state.backtests[strategyKey];
  if (!cacheOnly) render();
  try {
    const payload = await api("/api/backtest", {
      method: "POST",
      body: JSON.stringify({
        strategy: strategyKey,
        period: BACKTEST_PERIOD,
        refresh,
        cache_only: cacheOnly,
        // Same account as the signal view and the Tune board: a DCA plan is per account, so
        // replaying the default account backtested a plan nobody was editing.
        account_id: accountForStrategy(strategyKey),
      }),
      // A fresh replay is the longest request the dashboard makes, and it grows with the
      // window: the 24M option covers roughly five times the trade dates the 4M one does.
      // The cache probe is a lookup and stays on the short timeout.
      timeoutMs: cacheOnly ? 15000 : 300000,
    });
    if (isBacktestPayload(payload)) {
      state.backtests[strategyKey] = payload;
      changed = true;
    } else if (payload?.supported === false) {
      // Kept even on the cache probe, unlike an ordinary error. "No cached run yet" is a state
      // the next click can change; "this algorithm cannot be replayed" is a permanent property
      // of the strategy, and discarding it leaves the tab offering a button that cannot work.
      state.backtests[strategyKey] = { supported: false, error: payload.error };
      changed = true;
    } else if (payload?.error) {
      if (!cacheOnly) {
        state.backtests[strategyKey] = { error: payload.error };
        changed = true;
      }
    }
  } catch (error) {
    if (!cacheOnly) {
      state.backtests[strategyKey] = { error: error.message };
      changed = true;
    }
  } finally {
    if (!cacheOnly) state.backtestLoading[strategyKey] = false;
    // Every render of the Backtest tab probes the cache, and this used to re-render whatever
    // the probe found -- including "nothing". Repainting the page for a result that changed
    // no state is what closed the period dropdown the instant it was opened.
    if (changed || !cacheOnly) render();
  }
}

async function recommendUniverse() {
  state.universeRefreshing = true;
  render();
  try {
    const payload = await api("/api/universe/recommend", {
      method: "POST",
      body: JSON.stringify({ refresh: true, max_symbols: 12 }),
      timeoutMs: 90000,
    });
    state.universeProposal = payload;
    showToast(`${(payload.rows || []).length} symbols proposed`);
  } catch (error) {
    showToast(error.message);
  } finally {
    state.universeRefreshing = false;
    render();
  }
}

async function applyUniverseProposal() {
  const rows = state.universeProposal?.rows || [];
  if (!rows.length || state.universeApplying) return;
  state.universeApplying = true;
  render();
  try {
    const payload = await api("/api/universe/apply", {
      method: "POST",
      body: JSON.stringify({ rows }),
      timeoutMs: 15000,
    });
    state.universe = payload.universe?.rows || state.universe;
    state.universeProposal = null;
    state.signals = {};
    state.backtests = {};
    // A plan is sanitised against the universe when it is read, so every cached config may
    // now describe a different set of tradable symbols.
    state.algorithmConfigs = {};
    ensureAlgorithmConfig(currentRoute().id);
    render();
    // The universe change invalidates every algorithm's view, so recompute the one on screen.
    loadSignals(currentRoute().id, true);
    showToast("Universe applied");
  } catch (error) {
    showToast(error.message);
  } finally {
    state.universeApplying = false;
    render();
  }
}

function isSignalsPayload(payload) {
  return Boolean(payload && typeof payload === "object" && Array.isArray(payload.rows));
}

//: Live signals, cached exactly like a backtest: the tab opens on a stored snapshot and only
//: recomputes when Refresh asks for it.
//:
//: ``refresh`` and ``cacheOnly`` are sent because the endpoint requires them. Without them the
//: API sees ``refresh=false`` on a cache miss and answers "No cached live signals are
//: available" *without computing* -- which is exactly what it is meant to do for a probe, and
//: meant the Refresh button could never populate the cache it was reporting empty. The two
//: flags were plumbed through the API and never through the one caller that needed them.
//:
//: A background refresh keeps the previous snapshot on screen while the new one computes, and
//: a failure leaves the last good rows up rather than blanking the tab.
async function loadSignals(strategyKey, refresh = false, options = {}) {
  const cacheOnly = Boolean(options.cacheOnly);
  if (state.signalLoading[strategyKey]) return;
  let changed = false;
  if (!cacheOnly) {
    state.signalLoading[strategyKey] = true;
    // A stale error must not survive an explicit refresh, or the tab keeps explaining a
    // failure that is currently being retried.
    if (state.signals[strategyKey]?.error) delete state.signals[strategyKey];
    render();
  }
  try {
    // Sent explicitly so the view is computed against the same account the Tune board writes.
    const account = accountForStrategy(strategyKey);
    const query = new URLSearchParams({
      strategy: strategyKey,
      account_id: account,
      refresh: String(Boolean(refresh)),
      cache_only: String(cacheOnly),
    });
    const payload = await api(`/api/strategy-signals?${query}`, {
      // A recompute runs the algorithm against live market data; the cache probe is a lookup.
      timeoutMs: cacheOnly ? 15000 : 120000,
    });
    if (isSignalsPayload(payload)) {
      state.signals[strategyKey] = payload;
      changed = true;
    } else if (payload?.error && !cacheOnly) {
      // On a probe, "no cached snapshot" is not an error worth showing -- it is the state the
      // Refresh button exists to change, and the tab already says so.
      state.signals[strategyKey] = { error: payload.error };
      changed = true;
      showToast(`Signal refresh failed: ${payload.error}`);
    }
  } catch (error) {
    if (!cacheOnly) {
      if (!state.signals[strategyKey]) state.signals[strategyKey] = { error: error.message };
      changed = true;
      showToast(`Signal refresh failed: ${error.message}`);
    }
  } finally {
    if (!cacheOnly) state.signalLoading[strategyKey] = false;
    // A probe that found nothing changed no state, so repainting for it would be the same
    // spurious re-render the backtest tab's probe had to stop doing.
    if (changed || !cacheOnly) render();
  }
}

//: How each decision reads at a glance. The five actions are the whole vocabulary -- every
//: algorithm classifies its rows into them, so this map is the only place the deck decides what
//: an outcome looks like, and it never learns which algorithm produced one.
const ACTION_STYLE = {
  enter:   { label: "ENTER",   cls: "is-enter",   hint: "Opening a position that was not held" },
  hold:    { label: "HOLD",    cls: "is-hold",    hint: "Kept: clears the exit band, even if it would not be bought today" },
  exit:    { label: "EXIT",    cls: "is-exit",    hint: "Closing a position that was held" },
  blocked: { label: "BLOCKED", cls: "is-blocked", hint: "Wanted, but a gate said no. Expand for which one" },
  idle:    { label: "IDLE",    cls: "is-idle",    hint: "Neither held nor wanted this run" },
};

function actionStyle(action) {
  return ACTION_STYLE[action] || ACTION_STYLE.idle;
}

//: A price that is not a live print gets an age badge, so a stored-bar fallback can never read
//: as a fresh quote. Live prices (price_current true) stay unbadged.
function priceAgeBadge(row) {
  if (row.price_current !== false || !row.price_time) return "";
  const fetched = Date.parse(row.price_time);
  if (Number.isNaN(fetched)) return "";
  const ageMinutes = Math.max(0, Math.round((Date.now() - fetched) / 60000));
  const ageText = ageMinutes >= 1440 ? `${Math.round(ageMinutes / 1440)}d` : ageMinutes >= 60 ? `${Math.round(ageMinutes / 60)}h` : `${ageMinutes}m`;
  return ` <span class="tableNote" title="Not a live print — closest stored price">(${escapeHtml(ageText)} old)</span>`;
}

//: The gate strip: one pip per check, in the order the algorithm applied them. Filled is a pass,
//: hollow a fail, and the blocking one is marked separately -- several gates can fail at once
//: while only the first decided anything.
function gateStrip(checks) {
  if (!checks.length) return `<span class="gateStrip is-empty" title="No gates applied to this row">—</span>`;
  const pips = checks.map((check) => {
    const cls = check.ok ? "is-pass" : check.blocking ? "is-blocking" : "is-fail";
    const limit = check.limit ? ` (needs ${check.limit})` : "";
    return `<i class="gatePip ${cls}" title="${escapeHtml(`${check.label}: ${check.ok ? "PASS" : "FAIL"} — ${check.value}${limit}`)}"></i>`;
  }).join("");
  const failed = checks.filter((check) => !check.ok).length;
  return `<span class="gateStrip">${pips}<span class="gateCount">${escapeHtml(failed ? `${checks.length - failed}/${checks.length}` : "all")}</span></span>`;
}

//: The expanded panel: every gate with what this run measured beside what it had to clear.
//: Passed gates are listed too -- they are how a reader tells "cleared everything but the vol
//: ceiling" from "failed at the first hurdle", which is a different kind of rejection.
function gateDetail(row) {
  if (!row.checks?.length) {
    return `<p class="gateEmpty">No gates were recorded for ${escapeHtml(row.symbol)} on this run.</p>`;
  }
  const items = row.checks.map((check) => {
    const cls = check.ok ? "is-pass" : check.blocking ? "is-blocking" : "is-fail";
    return `<li class="gateItem ${cls}">
      <span class="gateVerdict">${check.ok ? "PASS" : "FAIL"}</span>
      <span class="gateLabel">${escapeHtml(check.label)}</span>
      <span class="gateValue">${escapeHtml(check.value || "—")}</span>
      <span class="gateLimit">${check.limit ? escapeHtml(`needs ${check.limit}`) : ""}</span>
    </li>`;
  }).join("");
  return `<ul class="gateList">${items}</ul>`;
}

function renderSignalTable(rows) {
  // Columns are declared by the algorithm rather than sniffed from the data. This used to probe
  // for a dozen known keys -- peak, vol_5d, rank -- so publishing a new signal meant editing the
  // frontend, and every algorithm's rows had to pretend to be one shape.
  const metricLabels = [];
  rows.forEach((row) => (row.metrics || []).forEach((metric) => {
    // A column every row leaves blank is a column with nothing to say. Rally Rotation ranks
    // only its eligible candidates, so on a defensive day Rank is "--" the whole way down.
    const blank = !metric.value || metric.value === "--";
    if (!blank && !metricLabels.includes(metric.label)) metricLabels.push(metric.label);
  }));
  // toggle, symbol, action, why, ...metrics, gates
  const columns = 5 + metricLabels.length;
  // Alignment is decided per *column*, from every value in it, rather than per cell. Numbers
  // want to be right-aligned so their decimal points line up; prose does not, and a column
  // carrying "$88 call · 04 Sep · 12d" ragged-left against its neighbours is what "the padding
  // looks off" actually is. Deciding per cell would be worse still: Bursty DCA's price column
  // holds "$90.12" on one row and "$86.79 (stale)" on the next, and they must align together.
  const numericColumn = new Map(metricLabels.map((label) => [
    label,
    rows.every((row) => {
      const metric = (row.metrics || []).find((item) => item.label === label);
      return !metric || isNumericValue(metric.value);
    }),
  ]));

  const body = rows.map((row) => {
    const style = actionStyle(row.action);
    // A row with no gates is not a row whose gates failed to record -- Rally Rotation's
    // defensive sleeve is chosen by rule rather than by rank, so it has none to show. Offering
    // a caret that opens onto "no gates were recorded" reads as a data fault; better to not
    // offer the affordance at all.
    const gated = Boolean((row.checks || []).length);
    const open = gated && state.expandedSignals.has(row.symbol);
    const metrics = metricLabels.map((label) => {
      const metric = (row.metrics || []).find((item) => item.label === label);
      const cls = numericColumn.get(label) ? "num" : "text";
      return `<td class="${cls}">${escapeHtml(metric ? metric.value : "--")}</td>`;
    }).join("");
    const detail = open
      ? `<tr class="signalDetailRow"><td colspan="${columns}">${gateDetail(row)}</td></tr>`
      : "";
    return `<tr class="signalRow ${style.cls}${open ? " is-open" : ""}${gated ? "" : " is-ungated"}"
             ${gated ? `data-signal-symbol="${escapeHtml(row.symbol)}" tabindex="0" role="button" aria-expanded="${open}"` : ""}
             title="${escapeHtml(style.hint)}">
        <td class="signalToggle">${gated ? (open ? "▾" : "▸") : ""}</td>
        <td><strong>${escapeHtml(row.symbol)}</strong></td>
        <td><span class="actionBadge ${style.cls}">${style.label}</span></td>
        <td class="signalWhy"><span class="whyText">${escapeHtml(row.headline || "—")}</span></td>
        ${metrics}
        <td class="signalGates">${gateStrip(row.checks || [])}</td>
      </tr>${detail}`;
  }).join("");

  return `<div class="tableWrap is-scroll"><table class="dataTable signalTable">
    <thead><tr>
      <th aria-label="Expand"></th>
      <th>Symbol</th>
      <th>Action</th>
      <th>Why</th>
      ${metricLabels.map((label) => `<th class="${numericColumn.get(label) ? "num" : "text"}">${escapeHtml(label)}</th>`).join("")}
      <th title="One pip per gate, in the order applied. Filled passed, hollow failed, red is the one that decided it.">Gates</th>
    </tr></thead>
    <tbody>${body}</tbody>
  </table>
  <p class="tableNote">Click a row for every gate, with what this run measured beside what it had to clear.</p>
  </div>`;
}

//: Whether a metric value is a bare number the eye should read as a column of figures --
//: currency, percentage, multiple, or a range of two. Anything carrying a word ("$88 call · 04 Sep",
//: "$86.79 (stale)") is prose and reads left-aligned.
//:
//: Placeholders count as numeric so a column of figures with one missing value does not flip
//: itself to left-aligned for the sake of a dash.
const NUMERIC_VALUE = /^[+-]?\$?[\d.,]+\s*[%x×]?(\s*[–—-]\s*[+-]?\$?[\d.,]+\s*[%x×]?)?$/;

function isNumericValue(value) {
  const text = String(value ?? "").trim();
  if (!text || text === "--" || text === "—") return true;
  return NUMERIC_VALUE.test(text);
}

function backtestStatusLabel(backtest, loading) {
  if (loading) return "Refreshing";
  if (backtest?.error) return "Error";
  if (isBacktestPayload(backtest)) return backtest.cached ? "Cached" : "Fresh";
  return "Pending";
}

function backtestCaptionText(backtest, loading) {
  if (backtest?.error) return backtest.error;
  if (isBacktestPayload(backtest)) {
    return "";
  }
  return loading ? `Refreshing ${BACKTEST_LABEL} backtest...` : `No ${BACKTEST_LABEL} backtest yet.`;
}

function backtestEquityText(backtest) {
  if (!isBacktestPayload(backtest)) return "";
  return `Ending equity = ${money(backtest.ending_cash)} cash + ${money(backtest.ending_invested)} holdings / ${percent(backtest.total_return)}`;
}

function backtestOrderText(backtest) {
  if (!isBacktestPayload(backtest) || !backtest.orders) return "";
  const orders = backtest.orders;
  const planned = Number(orders.planned_order_value || 0);
  const traded = Number(orders.total_order_value || 0);
  const skipped = Number(orders.skipped_order_value || 0);
  const peakInvested = Number(orders.max_capital_at_work ?? orders.max_gross_exposure ?? 0);
  const cashCap = Number(orders.capital_limit || backtest.starting_equity || 0);
  const plannedText = planned > traded + 0.01
    ? ` / ${money(planned)} planned / ${money(skipped)} skipped`
    : "";
  const tradingLabel = "cumulative turnover";
  const peakText = cashCap
    ? ` / peak invested ${money(peakInvested)} of ${money(cashCap)} cap`
    : ` / peak invested ${money(peakInvested)}`;
  return `${Number(orders.total_orders || 0)} orders / ${money(traded)} ${tradingLabel}${plannedText}${peakText} / max exposure ${percent(orders.max_gross_exposure_pct)}`;
}

function renderUniverseProposalRows() {
  const proposal = state.universeProposal;
  const rows = proposal?.rows || [];
  const rejectedCount = Number(proposal?.rejected?.length || 0);
  const status = state.universeRefreshing
    ? "Refreshing"
    : proposal
      ? `${rows.length}/${proposal.eligible_count || rows.length}`
      : "Queued";
  if (state.universeRefreshing && !proposal) {
    return `<p>Refreshing universe candidates.</p>`;
  }
  return `
    <article class="universeProposalHeader">
      <strong>Universe</strong>
      <span>${escapeHtml(status)} / ${escapeHtml(proposal?.data_feed || "feed")} / ${rejectedCount} filtered</span>
      <button class="applyUniverse" type="button" data-apply-universe ${state.universeApplying || !rows.length ? 'disabled aria-busy="true"' : ""}>Apply</button>
    </article>
    ${rows.map((row) => `
      <article class="universeProposalRow">
        <strong>${escapeHtml(row.symbol)}</strong>
        <span>${escapeHtml(row.bucket || "")}</span>
        <span>${escapeHtml(row.latest_bar || "")} / ${money(row.avg_dollar_volume, 0)}</span>
      </article>
    `).join("") || "<p>No proposal</p>"}
  `;
}

function renderSignalFallbackRows(selected, payload, signalInputs) {
  // Every algorithm renders its own plan now, so an empty list means the run proposed nothing --
  // never that the strategy is a template with no wiring behind it.
  const detail = payload
    ? "This run produced no rows at all; the summary above says why."
    : "Waiting for the first live snapshot.";
  const inputs = signalInputs.length ? signalInputs : [selected.logic || "Signal configuration pending"];
  return `
    <article class="signalFallback">
      <strong>No rows this run</strong>
      <span>${escapeHtml(detail)}</span>
    </article>
    ${inputs.map((signal) => `
      <article>
        <strong>${escapeHtml(signal)}</strong>
        <span>Signal input</span>
      </article>
    `).join("")}
  `;
}

function renderBacktestChart(payload, svg) {
  if (!svg) return;
  const width = svg.clientWidth || 860;
  const height = svg.clientHeight || 260;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.replaceChildren();
  if (!payload || payload.error) {
    svg.appendChild(textEl({ x: width / 2, y: height / 2, "text-anchor": "middle", class: "empty-chart" }, payload?.error ? "Backtest unavailable" : "Backtest pending"));
    return;
  }
  const rows = (payload?.rows || [])
    .map((row) => ({
      date: new Date(row.timestamp),
      equity: Number(row.equity),
      invested: Number(row.invested ?? row.gross_exposure ?? 0),
      cash: Number(row.cash ?? 0),
      positions: backtestPositions(row.positions),
      dcaContributions: Number(row.dca_contributions ?? 0),
      trade: backtestTrades(row),
    }))
    .filter((row) => Number.isFinite(row.equity) && !Number.isNaN(row.date.getTime()));
  if (rows.length < 2) {
    svg.appendChild(textEl({ x: width / 2, y: height / 2, "text-anchor": "middle", class: "empty-chart" }, `No ${BACKTEST_LABEL} rows`));
    return;
  }

  const left = 16;
  const right = 25;
  const top = 42;
  const bottom = 24;
  const valueLabelY = 16;
  const minX = Math.min(...rows.map((row) => row.date.getTime()));
  const maxX = Math.max(...rows.map((row) => row.date.getTime()));
  const minY = Math.min(...rows.map((row) => row.equity));
  const maxY = Math.max(...rows.map((row) => row.equity));
  const padY = Math.max((maxY - minY) * 0.1, Math.abs(maxY) * 0.01, 1);
  const xScale = (date) => left + ((date.getTime() - minX) / Math.max(maxX - minX, 1)) * (width - left - right);
  const yScale = (equity) => height - bottom - ((equity - minY + padY) / Math.max(maxY - minY + padY * 2, 1)) * (height - top - bottom);
  const points = rows
    .map((row) => ({ x: xScale(row.date), y: yScale(row.equity) }))
    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
  if (points.length < 2) {
    svg.appendChild(textEl({ x: width / 2, y: height / 2, "text-anchor": "middle", class: "empty-chart" }, `No ${BACKTEST_LABEL} chart`));
    return;
  }
  const path = points.map((point, index) => `${index ? "L" : "M"} ${point.x.toFixed(1)} ${point.y.toFixed(1)}`).join(" ");
  const color = rows.at(-1).equity >= rows[0].equity ? "#057a55" : "#b42318";
  const axisY = height - bottom;
  const intraday = backtestIsIntraday(rows);
  svg.appendChild(svgEl("line", { class: "axis-line", x1: left, y1: axisY, x2: width - right, y2: axisY }));
  // ``dateTicks`` samples at a fixed stride and then always appends the last row, so the final
  // pair can land arbitrarily close together -- and both get clamped to the same edge inset,
  // stacking them outright. Tolerable while every label was "Aug 21"; an intraday label is wide
  // enough that the two overprint. Measured in drawn width rather than in rows, because that is
  // what actually collides.
  let previousTickX = -Infinity;
  dateTicks(rows, 3).forEach((row) => {
    const xRaw = xScale(row.date);
    // keep tick labels inside the chart area to avoid overflow at the edges
    const x = Math.min(Math.max(xRaw, left + 12), width - right - 12);
    const label = formatChartStamp(row.date, intraday);
    if (x - previousTickX < label.length * 6.3) return;
    previousTickX = x;
    svg.appendChild(svgEl("line", { class: "axis-line", x1: x, y1: axisY, x2: x, y2: axisY + 4 }));
    svg.appendChild(textEl({ x, y: height - 6, "text-anchor": "middle", class: "axis-label" }, label));
  });
  svg.appendChild(svgEl("path", { class: "growth-line", stroke: color, d: path }));
  // After the line so the marks sit on top of it, before the hover overlay so the hitbox still
  // catches every pointer event -- markers are decoration and must not become targets.
  renderTradeMarkers(svg, rows, { xScale, yScale });
  svg.appendChild(textEl({ x: left, y: valueLabelY, class: "chart-label" }, money(rows[0].equity)));
  svg.appendChild(textEl({ x: width - right, y: valueLabelY, "text-anchor": "end", class: "chart-label" }, money(rows.at(-1).equity)));
  addBacktestChartHover(svg, rows, { width, height, left, right, top, bottom, xScale, yScale, intraday });
}

//: Closest two marks may sit before they are merged. A trading day is worth a mark; a pixel is
//: not worth two. Bursty DCA puts 31 marks on a 12-month curve and Rally Rotation 58, so this
//: never engages for them -- it exists for the intraday grid, where a 3-month replay trades on
//: 7,175 of its 9,047 bars and one mark per step would ink the entire plot area solid.
const MIN_MARKER_SPACING = 7;

//: Half-width of a marker triangle, and how far clear of the curve its centre sits.
//:
//: The offset is doing real work, not just spacing. The equity line is drawn in one of exactly
//: the two colours the markers use -- green when the period made money, red when it lost -- so
//: whichever way the backtest went, one kind of marker is the same hue as the line it sits on.
//: Keeping the line's win/loss colour and separating the marks instead: 9px of offset against a
//: 3px line leaves a clear gap either side, and the white halo in ``.trade-marker`` closes the
//: case where a steep segment passes near one anyway.
const MARKER_SIZE = 4.5;
const MARKER_OFFSET = 9;

function renderTradeMarkers(svg, rows, { xScale, yScale }) {
  // Merged by position rather than sampled: dropping every second mark would misreport *which*
  // way a bucket traded. A bucket that contains a buy and a sell is a bucket that shows both.
  const buckets = [];
  rows.forEach((row) => {
    if (!row.trade.side) return;
    const x = xScale(row.date);
    const y = yScale(row.equity);
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    const last = buckets.at(-1);
    if (last && x - last.x < MIN_MARKER_SPACING) {
      last.buy = last.buy || row.trade.side === "buy" || row.trade.side === "both";
      last.sell = last.sell || row.trade.side === "sell" || row.trade.side === "both";
      last.flat = last.flat || row.trade.side === "flat";
      return;
    }
    buckets.push({
      x,
      y,
      buy: row.trade.side === "buy" || row.trade.side === "both",
      sell: row.trade.side === "sell" || row.trade.side === "both",
      flat: row.trade.side === "flat",
    });
  });
  if (!buckets.length) return;

  const layer = svgEl("g", { class: "trade-markers" });
  buckets.forEach((bucket) => {
    // A step that bought *and* sold is one decision, not two, so it gets one mark. Drawing both
    // triangles said "buy" and "sell" at the same x and left the reader to infer the rotation;
    // amber names it. Rally Rotation is mostly this, so it is the common case, not the corner.
    if (bucket.buy && bucket.sell) {
      layer.appendChild(diamond(bucket.x, bucket.y, "is-rotation"));
    } else if (bucket.buy) {
      // Buys below the curve, sells above, so the two one-sided cases stay distinguishable at a
      // glance even where the line is the same colour as the mark.
      layer.appendChild(marker(bucket.x, bucket.y + MARKER_OFFSET, "up", "is-buy"));
    } else if (bucket.sell) {
      layer.appendChild(marker(bucket.x, bucket.y - MARKER_OFFSET, "down", "is-sell"));
    } else if (bucket.flat) {
      layer.appendChild(svgEl("circle", { class: "trade-marker is-flat", cx: bucket.x, cy: bucket.y, r: 2.5 }));
    }
  });
  svg.appendChild(layer);
}

//: Sits *on* the curve rather than offset from it, because a rotation belongs to neither side.
function diamond(x, y, className) {
  const size = MARKER_SIZE + 0.5;
  return svgEl("polygon", {
    class: `trade-marker ${className}`,
    points: `${x.toFixed(1)},${(y - size).toFixed(1)} ${(x + size).toFixed(1)},${y.toFixed(1)} `
      + `${x.toFixed(1)},${(y + size).toFixed(1)} ${(x - size).toFixed(1)},${y.toFixed(1)}`,
  });
}

function marker(x, y, direction, className) {
  // ``up`` points at the curve from below, ``down`` from above: both apexes face the line, so
  // the mark reads as attached to the point it describes rather than floating near it.
  const tip = direction === "up" ? y - MARKER_SIZE : y + MARKER_SIZE;
  const base = direction === "up" ? y + MARKER_SIZE : y - MARKER_SIZE;
  return svgEl("polygon", {
    class: `trade-marker ${className}`,
    points: `${x.toFixed(1)},${tip.toFixed(1)} ${(x - MARKER_SIZE).toFixed(1)},${base.toFixed(1)} ${(x + MARKER_SIZE).toFixed(1)},${base.toFixed(1)}`,
  });
}

function addBacktestChartHover(svg, rows, chart) {
  const overlay = svgEl("rect", {
    class: "chart-hitbox",
    x: chart.left,
    y: chart.top,
    width: Math.max(chart.width - chart.left - chart.right, 1),
    height: Math.max(chart.height - chart.top - chart.bottom, 1),
  });
  const layer = svgEl("g", { class: "chart-hover-layer", visibility: "hidden" });
  const vertical = svgEl("line", { class: "chart-crosshair" });
  const horizontal = svgEl("line", { class: "chart-crosshair" });
  const point = svgEl("circle", { class: "chart-hover-point", r: 4 });
  const box = svgEl("rect", { class: "chart-tooltip-bg", rx: 6, ry: 6 });
  const positionGroup = svgEl("g", { class: "chart-position-lines" });
  const valueBox = svgEl("rect", { class: "chart-axis-value-bg", rx: 4, ry: 4 });
  const valueText = textEl({ class: "chart-axis-value", "text-anchor": "start" }, "");
  const axisDateBox = svgEl("rect", { class: "chart-axis-value-bg", rx: 4, ry: 4 });
  const axisDateText = textEl({ class: "chart-axis-value", "text-anchor": "middle" }, "");
  [vertical, horizontal, point, box, positionGroup, valueBox, valueText, axisDateBox, axisDateText].forEach((node) => {
    layer.appendChild(node);
  });
  svg.appendChild(layer);
  svg.appendChild(overlay);

  const timestamps = rows.map((row) => row.date.getTime());
  const minX = Math.min(...timestamps);
  const maxX = Math.max(...timestamps);
  //: Through the SVG's own transform, not by rescaling the bounding box by hand.
  //:
  //: The hand-rolled version assumed the viewBox stretched edge to edge, which is what
  //: ``preserveAspectRatio="none"`` would do. The default is ``xMidYMid meet``: when the
  //: element's aspect ratio does not match the viewBox's, the drawing is scaled to whichever
  //: axis binds and *centred* on the other. Measured here at 1440px, a 1104x378 viewBox in a
  //: 1106x366 box came out scaled 0.969 with an 18px inset each side -- so the crosshair sat up
  //: to a full row away from the cursor, and hovering a trade marker resolved to its neighbour.
  //: ``getScreenCTM`` already knows all of it, including any future padding or transform.
  const toSvgPoint = (event) => {
    const matrix = svg.getScreenCTM();
    if (!matrix) return { x: 0, y: 0 };
    const origin = svg.createSVGPoint();
    origin.x = event.clientX;
    origin.y = event.clientY;
    const local = origin.matrixTransform(matrix.inverse());
    return { x: local.x, y: local.y };
  };
  const nearestRow = (x) => {
    const ratio = clamp((x - chart.left) / Math.max(chart.width - chart.left - chart.right, 1), 0, 1);
    const target = minX + ratio * Math.max(maxX - minX, 1);
    let best = rows[0];
    let bestDistance = Math.abs(best.date.getTime() - target);
    rows.forEach((row) => {
      const distance = Math.abs(row.date.getTime() - target);
      if (distance < bestDistance) {
        best = row;
        bestDistance = distance;
      }
    });
    return best;
  };
  const positionText = (text, x, y) => {
    text.setAttribute("x", x);
    text.setAttribute("y", y);
  };

  overlay.addEventListener("pointermove", (event) => {
    const pointInSvg = toSvgPoint(event);
    const row = nearestRow(pointInSvg.x);
    const x = chart.xScale(row.date);
    const y = chart.yScale(row.equity);
    const positionLines = backtestHoldingLines(row);
    const tooltipWidth = Math.max(168, Math.min(260, 64 + Math.max(...positionLines.map((line) => line.length)) * 6));
    const tooltipHeight = 16 + positionLines.length * 14;
    const tooltipX = x + tooltipWidth + 12 > chart.width ? x - tooltipWidth - 10 : x + 10;
    const availableHeight = chart.height - chart.top - chart.bottom;
    const tooltipY = tooltipHeight >= availableHeight
      ? chart.top
      : Math.min(Math.max(y - tooltipHeight / 2, chart.top), chart.height - chart.bottom - tooltipHeight);
    const axisValue = money(row.equity);
    const axisValueWidth = Math.max(58, axisValue.length * 6.3 + 14);
    const axisValueX = chart.left;
    const axisValueY = Math.min(Math.max(y - 9, chart.top), chart.height - chart.bottom - 18);
    const axisDate = formatChartStamp(row.date, chart.intraday);
    const axisDateWidth = Math.max(52, axisDate.length * 6.3 + 16);
    const axisDateX = Math.min(
      Math.max(x - axisDateWidth / 2, chart.left),
      chart.width - chart.right - axisDateWidth,
    );
    const axisDateY = chart.height - chart.bottom + 5;
    vertical.setAttribute("x1", x);
    vertical.setAttribute("x2", x);
    vertical.setAttribute("y1", chart.top);
    vertical.setAttribute("y2", chart.height - chart.bottom);
    horizontal.setAttribute("x1", chart.left);
    horizontal.setAttribute("x2", chart.width - chart.right);
    horizontal.setAttribute("y1", y);
    horizontal.setAttribute("y2", y);
    point.setAttribute("cx", x);
    point.setAttribute("cy", y);
    box.setAttribute("x", tooltipX);
    box.setAttribute("y", tooltipY);
    box.setAttribute("width", tooltipWidth);
    box.setAttribute("height", tooltipHeight);
    valueBox.setAttribute("x", axisValueX);
    valueBox.setAttribute("y", axisValueY);
    valueBox.setAttribute("width", axisValueWidth);
    valueBox.setAttribute("height", 18);
    valueText.setAttribute("x", axisValueX + 7);
    valueText.setAttribute("y", axisValueY + 12);
    valueText.textContent = axisValue;
    axisDateBox.setAttribute("x", axisDateX);
    axisDateBox.setAttribute("y", axisDateY);
    axisDateBox.setAttribute("width", axisDateWidth);
    axisDateBox.setAttribute("height", 18);
    axisDateText.setAttribute("x", axisDateX + axisDateWidth / 2);
    axisDateText.setAttribute("y", axisDateY + 12);
    axisDateText.textContent = axisDate;
    positionGroup.replaceChildren();
    positionLines.forEach((line, index) => {
      const lineText = textEl({ class: "chart-tooltip-text" }, line);
      positionText(lineText, tooltipX + 10, tooltipY + 18 + index * 14);
      positionGroup.appendChild(lineText);
    });
    layer.setAttribute("visibility", "visible");
  });
  overlay.addEventListener("pointerleave", () => {
    layer.setAttribute("visibility", "hidden");
  });
}

function backtestPositions(positions) {
  if (!positions || typeof positions !== "object" || Array.isArray(positions)) return [];
  return Object.entries(positions)
    .map(([symbol, value]) => [symbol, Number(value)])
    .filter(([symbol, value]) => symbol && Number.isFinite(value) && Math.abs(value) > 0.005)
    .sort((left, right) => Math.abs(right[1]) - Math.abs(left[1]));
}

//: What one replay step traded, from the ``trades`` map the backtest already publishes: net
//: notional per symbol, signed, buys positive. Same shape and same filtering as
//: ``backtestPositions`` above, because it comes off the same row and is read the same way.
//:
//: ``order_count`` is carried separately rather than derived from the map. The two can disagree
//: -- a row from an algorithm that never populated ``trades`` still counts its orders -- and
//: when they do, the count is the one that knows a trade happened at all.
function backtestTrades(row) {
  const raw = row?.trades;
  const entries = raw && typeof raw === "object" && !Array.isArray(raw)
    ? Object.entries(raw)
        .map(([symbol, value]) => [symbol, Number(value)])
        .filter(([symbol, value]) => symbol && Number.isFinite(value) && Math.abs(value) > 0.005)
        .sort((left, right) => Math.abs(right[1]) - Math.abs(left[1]))
    : [];
  // Share counts ride alongside rather than replacing the notionals: the marker still sizes and
  // sorts by dollars, while the holding line reports the quantity that actually changed hands.
  const rawShares = row?.trade_shares;
  const shares = new Map(
    rawShares && typeof rawShares === "object" && !Array.isArray(rawShares)
      ? Object.entries(rawShares)
          .map(([symbol, value]) => [symbol, Number(value)])
          .filter(([symbol, value]) => symbol && Number.isFinite(value) && value !== 0)
      : [],
  );
  const count = Number(row?.order_count) || 0;
  const bought = entries.some(([, value]) => value > 0);
  const sold = entries.some(([, value]) => value < 0);
  // "It traded, but the row does not say which way" is a real state, not an error: it is every
  // row written before ``trades`` existed, and every Options Flip row, whose orders currently
  // record no notional at all. A neutral mark says something happened here, which is strictly
  // better than the silence of treating the row as a quiet day.
  const side = bought && sold ? "both" : bought ? "buy" : sold ? "sell" : count > 0 ? "flat" : "";
  return { side, entries, shares, count };
}

//: What the book held after this step, with what changed to get there appended to the line.
//:
//: Inline rather than as its own block underneath. The two used to be separate lists, so a
//: rotation printed the same four symbols twice -- once as holdings, once as orders -- and left
//: the reader to pair them up. One line per symbol says the whole thing: what is held now, and
//: what moved.
function backtestHoldingLines(row) {
  const shares = row.trade?.shares || new Map();
  const held = new Set();
  const lines = row.positions.map(([symbol, value]) => {
    held.add(symbol);
    return `${symbol} : ${money(value)}${shareDelta(shares.get(symbol))}`;
  });
  // A position sold out entirely is absent from ``positions`` -- it is worth nothing, so the
  // replay drops it. Without this the most consequential line of a rotation, the leg that was
  // closed, is the one line the tooltip does not show.
  shares.forEach((quantity, symbol) => {
    if (!held.has(symbol)) lines.push(`${symbol} : ${money(0)}${shareDelta(quantity)}`);
  });
  if (lines.length) return lines;
  // Traded, but with no per-symbol detail to attach to a holding.
  if (row.trade?.side) return [`${row.trade.count} order${row.trade.count === 1 ? "" : "s"}`];
  return ["No positions"];
}

//: The signed quantity that changed hands, as a suffix. Blank when nothing moved, so a quiet
//: day's holdings read exactly as they did before this existed.
function shareDelta(quantity) {
  if (!quantity || !Number.isFinite(quantity)) return "";
  // Fractional brokerages fill to six places and whole-share ones never do, so the precision
  // has to follow the number: "+0.4521" is the honest answer for a $10 slice of a $500 ETF,
  // and "+1" is the honest answer beside it.
  const size = Math.abs(quantity);
  const shown = size >= 1 ? Number(size.toFixed(2)) : Number(size.toFixed(4));
  return ` ${quantity > 0 ? "+" : "−"}${shown}`;
}

function percent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

//: Wheel and pinch fire continuously, so plan edits coalesce into one POST once the gesture
//: settles rather than one per tick.
const PLAN_SAVE_DEBOUNCE_MS = 500;

//: What the board says about its own saving. The board writes on every gesture and has no
//: save button, so without this an edit that reached the server and one that failed silently
//: looked exactly the same -- which is how a plan being written to the wrong account went
//: unnoticed for as long as it did.
function planSaveStatusText() {
  if (state.planSaveStatus === "saving") return "Saving.";
  return state.planSaveStatus ? `Saved ${state.planSaveStatus}` : "Saves automatically.";
}

function setPlanSaveStatus(status) {
  state.planSaveStatus = status;
  const node = $("#planSaveStatus");
  if (node) node.textContent = planSaveStatusText();
}

function schedulePlanSave() {
  window.clearTimeout(schedulePlanSave.timer);
  setPlanSaveStatus("saving");
  schedulePlanSave.timer = window.setTimeout(() => {
    // Saving swaps in the server's copy of the plan and re-renders the board, which would
    // yank it out from under a gesture that is still going. Wait for the hands to come off.
    if (state.drag || state.pinch || state.boardPinch) {
      schedulePlanSave();
      return;
    }
    savePlan();
  }, PLAN_SAVE_DEBOUNCE_MS);
}

async function savePlan(quiet = true) {
  const strategyKey = planStrategyKey();
  const entry = state.algorithmConfigs[strategyKey];
  if (!entry?.config || !currentPlan()) return;
  syncNodesToPlan();
  try {
    // The plan is part of this algorithm's config, so it saves through the same endpoint as
    // every other knob -- there is no DCA-shaped write path any more.
    const payload = await api("/api/algorithm-config", {
      method: "POST",
      body: JSON.stringify({ strategy: strategyKey, config: entry.config }),
      timeoutMs: 5000,
    });
    state.algorithmConfigs[strategyKey] = payload;
    // Only this algorithm's views are stale: the plans are no longer shared.
    delete state.backtests[strategyKey];
    delete state.signals[strategyKey];
    renderDca();
    setPlanSaveStatus(new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }));
    if (!quiet) showToast("Saved");
  } catch (error) {
    setPlanSaveStatus("");
    showToast(`Plan not saved: ${error.message}`);
  }
}

function shadeColor(color, percent) {
  const parsed = color.replace("#", "");
  const number = parseInt(parsed, 16);
  const amount = Math.round(2.55 * percent);
  const red = clamp((number >> 16) + amount, 0, 255);
  const green = clamp(((number >> 8) & 0xff) + amount, 0, 255);
  const blue = clamp((number & 0xff) + amount, 0, 255);
  return `#${(0x1000000 + red * 0x10000 + green * 0x100 + blue).toString(16).slice(1)}`;
}


// =========================================================================================
// Shell: sidebar + hash router. Each algorithm gets a full page whose tabs follow the
// lifecycle -- see what it does, tune it, test it, debug it, deploy it. Deployment targets
// (accounts) get their own pages, so "where does this run" is a first-class question.
// =========================================================================================

//: Overview answers "what is this and what has it done": the explainer panels plus the
//: bot's own order journal, which is the only per-algorithm record of activity there is.
//: Broker holdings live on the account page instead -- see renderAccountPage.
const TABS = [
  { id: "overview", label: "Overview" },
  { id: "signals", label: "Signals" },
  { id: "tune", label: "Tune" },
  { id: "backtest", label: "Backtest" },
];

const DEFAULT_TAB = TABS[0].id;

function currentRoute() {
  const raw = (window.location.hash || "").replace(/^#\/?/, "");
  const parts = raw.split("/").filter(Boolean);
  if (parts[0] === "algo" && parts[1]) {
    const tab = TABS.some((tab) => tab.id === parts[2]) ? parts[2] : DEFAULT_TAB;
    return { page: "algo", id: decodeURIComponent(parts[1]), tab };
  }
  if ((parts[0] === "account" || parts[0] === "target") && parts[1]) {
    return { page: "account", id: decodeURIComponent(parts[1]), tab: "" };
  }
  return { page: "algo", id: DEFAULT_ALGORITHM_KEY, tab: DEFAULT_TAB };
}

function deploymentsFor(strategyKey) {
  return bindings().filter((binding) => binding.strategy === strategyKey);
}

//: One algorithm runs against at most one account, so a deployment is singular. That keeps
//: P/L attributable: the broker blends positions per account, and with one algorithm per
//: account the account's numbers *are* this algorithm's numbers.
function deploymentFor(strategyKey) {
  return deploymentsFor(strategyKey)[0] || null;
}

function accountRows() {
  return state.accounts?.rows || [];
}

function accountLabel(accountId) {
  return accountRows().find((row) => row.id === accountId)?.label || accountId || "unassigned";
}

// -- sidebar -----------------------------------------------------------------------------

//: One colour rule for the whole dashboard: an algorithm is green when it is switched on and
//: on a clock, orange when it is switched on but waiting for the MCP agent to drive it, and
//: dark when it is off. Accounts and the bot pill both derive from this rather than each
//: inventing their own reading -- "deployed but paused" used to show amber, which looked
//: identical to "the agent is driving it".
function deploymentStatus(deployments) {
  const armed = (deployments || []).filter((deployment) => deployment.enabled);
  if (armed.some((deployment) => normalizeBindingCron(deployment.cron))) return "live";
  if (armed.length) return "idle";
  return "off";
}

function renderSidebar() {
  const route = currentRoute();
  const algorithmNav = $("#algorithmNav");
  if (algorithmNav) {
    algorithmNav.innerHTML = algorithmChoices().map((strategy) => {
      const deployments = deploymentsFor(strategy.key);
      const active = route.page === "algo" && route.id === strategy.key;
      const status = deploymentStatus(deployments);
      return `
        <li>
          <a class="navItem${active ? " is-active" : ""}" href="#/algo/${escapeHtml(strategy.key)}/${escapeHtml(route.page === "algo" ? route.tab : DEFAULT_TAB)}">
            <span class="statusDot is-${status}" aria-hidden="true"></span>
            <span class="navItemLabel">${escapeHtml(strategy.name)}</span>
            ${deployments.length ? `<span class="navBadge">${deployments.length}</span>` : ""}
          </a>
        </li>`;
    }).join("");
  }

  renderAccountNav();
  renderNavFooter();
}

//: Accounts sit under the algorithms because that is the reading order of the question the
//: sidebar answers: what runs, and what is it doing to the money. The number shown is day
//: P/L, which is the only figure the broker reports without a cost-basis round trip.
function renderAccountNav() {
  const host = $("#accountNav");
  if (!host) return;
  const rows = accountRows();
  if (!rows.length) {
    host.innerHTML = `<li><span class="navEmpty">No accounts yet</span></li>`;
    return;
  }
  const route = currentRoute();
  host.innerHTML = rows.map((account) => {
    const deployed = bindings().filter((binding) => binding.account_id === account.id);
    // An account is only as live as the algorithms pointed at it, and it cannot be live at
    // all without credentials.
    const status = account.credentials_ready ? deploymentStatus(deployed) : "off";
    const active = route.page === "account" && route.id === account.id;
    const positions = state.positions[account.id];
    const pl = positions && !positions.error ? positions.day_pl : null;
    const stat = pl === null || pl === undefined
      ? `<span class="navStat is-muted">--</span>`
      : `<span class="navStat ${pl >= 0 ? "gain" : "loss"}">${escapeHtml(money(pl, 2))}</span>`;
    const title = account.credentials_ready
      ? `${account.label} · ${deployed.length ? `runs ${deployed.map((binding) => strategyByKey(binding.strategy).name).join(", ")}` : "no algorithm deployed"}`
      : `${account.label} · credentials missing`;
    const inner = `
      <span class="statusDot is-${status}" aria-hidden="true"></span>
      <span class="navItemLabel">${escapeHtml(account.label)}</span>
      ${stat}`;
    return `
      <li>
        <a class="navItem navItem--stat${active ? " is-active" : ""}" href="#/account/${escapeHtml(account.id)}" title="${escapeHtml(title)}">${inner}</a>
      </li>`;
  }).join("");
  // The sidebar is the only place an idle account's P/L shows, so it pulls its own numbers.
  rows.filter((account) => account.credentials_ready).forEach((account) => ensurePositions(account.id));
}

function renderNavFooter() {
  const footer = $("#navFooter");
  if (!footer) return;
  const auth = state.schwabAuth;
  // Present whenever Schwab is wired up as a connector, credentials or not: this row is the
  // control that *starts* consent, so gating it on being connected would hide the only way
  // to connect. The dot carries the state; the tooltip explains what is missing.
  const authRow = auth?.connector_enabled || auth?.configured
    ? `<button class="navHealth is-${escapeHtml(auth.state)}" type="button" id="schwabAuthPill"
         title="${escapeHtml(auth.detail || "")}">
         <span class="statusDot is-${auth.state === "ok" ? "live" : auth.state === "warning" ? "idle" : "off"}" aria-hidden="true"></span>
         <span class="navHealthLabel">Schwab API</span>
       </button>`
    : "";
  const runtime = runtimeSummary();
  footer.innerHTML = `${authRow}
    <span class="navHealth is-muted" title="${escapeHtml(runtime.detail)}">
      <span class="statusDot is-${runtime.status}" aria-hidden="true"></span>
      <span class="navHealthLabel">Bot</span>
    </span>`;
}

//: Every deployment gets its own scheduler loop, so the runtime has one state *per binding*.
//: Reading the first of them -- which is what this did -- reported "paused" whenever the one
//: armed algorithm happened not to be first in the dict.
function runtimeSummary() {
  const bot = state.bot || {};
  const loops = Object.values(bot.bindings || {});
  if (!loops.length && bot.algorithm) loops.push(bot.algorithm);

  // Deliberately not keyed on the container's runtime mode. That only says an MCP server was
  // started alongside the dashboard; it says nothing about whether any algorithm is on, and
  // reporting "MCP mode" with everything switched off described the process rather than the
  // bot. What runs is decided per binding: switched on with a frequency, or switched on and
  // parked on "mcp" to wait for an external request.
  const running = loops.filter((loop) => loop.running);
  const armed = bindings().filter((binding) => binding.enabled);
  // The bot pill takes the same colour as the algorithms: green while anything is on a
  // clock, orange when everything that is on is waiting for the agent instead.
  const status = deploymentStatus(armed);
  const scheduled = armed.filter((binding) => normalizeBindingCron(binding.cron));
  const agentDriven = armed.filter((binding) => !normalizeBindingCron(binding.cron));
  const lastRun = loops
    .map((loop) => loop.last_finished_at)
    .filter(Boolean)
    .sort()
    .pop() || "";
  const error = loops.map((loop) => loop.last_error).filter(Boolean)[0] || "";

  const label = running.length
    ? `Bot running${running.length > 1 ? ` (${running.length})` : ""}`
    : status === "live"
      ? `Bot scheduled${scheduled.length > 1 ? ` (${scheduled.length})` : ""}`
      : status === "idle"
        ? "Bot agent-driven"
        : "Bot off";
  const detail = error
    ? error
    : [
        scheduled.length
          ? `Scheduled: ${scheduled.map((binding) => `${strategyByKey(binding.strategy).name} ${describeCron(binding.cron)}`).join(", ")}`
          : "",
        agentDriven.length
          ? `Agent-driven: ${agentDriven.map((binding) => strategyByKey(binding.strategy).name).join(", ")}`
          : "",
        armed.length ? "" : "No algorithm is switched on",
        lastRun ? `Last run ${formatActivityTime(lastRun)}` : "No run yet",
      ].filter(Boolean).join(" · ");
  // label and note are no longer rendered -- the dot says scheduled, agent-driven or off, and
  // the words only repeated it. Both survive in the tooltip, where the detail belongs.
  return { status, detail: `${label} -- ${detail}` };
}

// -- page frame --------------------------------------------------------------------------

function pageHeader({ title, subtitle, meta = "", actions = "" }) {
  return `
    <header class="pageHead">
      <div class="pageHeadMain">
        <h1>${escapeHtml(title)}</h1>
        ${subtitle ? `<p>${escapeHtml(subtitle)}</p>` : ""}
      </div>
      <div class="pageHeadSide">${meta}${actions}</div>
    </header>`;
}

function tabBar(strategyKey, activeTab) {
  return `<nav class="tabBar" aria-label="Sections">
    ${TABS.map((tab) => `
      <a class="tabLink${tab.id === activeTab ? " is-active" : ""}" href="#/algo/${escapeHtml(strategyKey)}/${tab.id}">${escapeHtml(tab.label)}</a>
    `).join("")}
  </nav>`;
}

//: A repaint that was deferred because a control had focus, and still owes the screen an
//: update. Dropping it outright is what left a freshly selected backtest period showing
//: nothing: the cached payload arrived, went into state, and no paint ever followed.
let renderDeferred = false;

function render(options = {}) {
  const route = currentRoute();
  renderSidebar();
  const content = $("#content");
  if (!content) return;
  // Rebuilding the body under an open <select> closes it mid-click, and under a focused
  // input it discards what is being typed. While the user is holding a control, defer --
  // but remember that a paint is owed, and flush it when focus leaves.
  //
  // ``force`` is for the handler of that control's own change event: a change means the
  // interaction finished, so there is nothing left to disturb.
  const active = document.activeElement;
  if (!options.force && active && active.matches?.("select, input, textarea") && content.contains?.(active)) {
    renderDeferred = true;
    return;
  }
  renderDeferred = false;
  if (route.page === "account") renderAccountPage(content, route.id);
  else renderAlgorithmPage(content, route.id, route.tab);
  // Both entry inputs live outside #content, so a route change rebuilds the page around them
  // and would leave one floating over a board that is no longer there.
  if (!$("#bubbleBoard")) {
    hideAmountEntry();
    hideSymbolEntry();
  }
  closeNavOnMobile();
}

// -- algorithm page ----------------------------------------------------------------------

// A binding's schedule is a cron expression in market time; empty means an agent drives it.
// Validated here only well enough to keep an obviously broken string from being saved -- the
// server re-parses with src/core/cron.py, which is the authority on what will actually run.
// The prose comes from the vendored cronstrue, which reads arbitrary expressions back in
// words; anything it cannot phrase falls back to the expression itself.
const CRON_FIELD_BOUNDS = [[0, 59], [0, 23], [1, 31], [1, 12], [0, 7]];

function normalizeBindingCron(value) {
  return String(value ?? "").trim().replace(/\s+/g, " ");
}

function cronError(expression) {
  const text = normalizeBindingCron(expression);
  if (!text) return "";
  const parts = text.split(" ");
  if (parts.length !== 5) {
    return `Needs 5 fields (minute hour day-of-month month day-of-week), got ${parts.length}.`;
  }
  const names = ["minute", "hour", "day of month", "month", "day of week"];
  for (let index = 0; index < 5; index += 1) {
    const [low, high] = CRON_FIELD_BOUNDS[index];
    for (const item of parts[index].split(",")) {
      const [range, step] = item.split("/");
      if (step !== undefined && !/^\d+$/.test(step)) return `Bad step in the ${names[index]} field.`;
      if (range === "*") continue;
      const bounds = range.split("-");
      if (bounds.length > 2 || bounds.some((bound) => !/^\d+$/.test(bound))) {
        return `Cannot read "${item}" in the ${names[index]} field.`;
      }
      if (bounds.some((bound) => Number(bound) < low || Number(bound) > high)) {
        return `${names[index]} must be ${low}-${high}.`;
      }
    }
  }
  return "";
}

function describeCron(expression) {
  const text = normalizeBindingCron(expression);
  if (!text) return "Agent-driven (MCP)";
  if (cronError(text)) return "Invalid schedule";
  try {
    return window.cronstrue.toString(text, { use24HourTimeFormat: true });
  } catch {
    return text;
  }
}

function renderAlgorithmPage(content, strategyKey, tab) {
  const strategy = strategyByKey(strategyKey);
  const deployments = deploymentsFor(strategy.key);
  const deployment = deploymentFor(strategy.key);
  // Every account is offered to every algorithm. Sharing one account between algorithms is
  // allowed; it only costs attribution, which the overview says plainly when it happens.
  const options = accountRows();
  const savedCron = normalizeBindingCron(deployment?.cron);
  const actions = options.length
    ? `<div class="deployControl"${deployment ? ` data-binding="${escapeHtml(deployment.id)}"` : ""}>
         <button class="ctl powerButton${deployment?.enabled ? " on" : ""}" type="button" data-role="power"
           aria-pressed="${Boolean(deployment?.enabled)}" aria-label="Toggle trading"
           title="${deployment?.enabled ? "Pause" : "Start"} trading"><span aria-hidden="true">&#9211;</span></button>
         <select class="ctl" id="deployTargetSelect" aria-label="Account"${deployment?.enabled ? " disabled" : ""}>
           ${options.map((account) => `<option value="${escapeHtml(account.id)}"${account.id === deployment?.account_id ? " selected" : ""}>${escapeHtml(account.label)}</option>`).join("")}
         </select>
         <span class="cronField">
           <input class="ctl cronInput" id="deployCronInput" type="text" spellcheck="false"
             aria-label="Schedule (cron, market time)" data-binding="${escapeHtml(deployment?.id || "")}"
             value="${escapeHtml(savedCron)}" placeholder="0 11 * * 1-5"
             title="Cron in US Eastern: minute hour day-of-month month day-of-week. Leave blank to let an agent drive it."
             ${!deployment ? "disabled" : ""}>
           <span class="cronHint" id="deployCronHint">${escapeHtml(describeCron(savedCron))}</span>
         </span>
       </div>`
    : `<span class="pill is-idle">No account available</span>`;

  content.innerHTML = `
    ${pageHeader({ title: strategy.name, subtitle: strategy.blurb, actions })}
    ${tabBar(strategy.key, tab)}
    <div class="tabBody" id="tabBody"></div>`;

  const body = $("#tabBody");
  if (tab === "overview") renderOverviewTab(body, strategy, deployment);
  if (tab === "signals") renderSignalsTab(body, strategy);
  if (tab === "tune") renderTuneTab(body, strategy);
  if (tab === "backtest") renderBacktestTab(body, strategy);
}

// -- account page ------------------------------------------------------------------------

//: Holdings and orders live here rather than on the algorithm because the broker reports
//: them per account and knows nothing about which algorithm -- or which hand-placed order --
//: produced them. Attributing an account's blended P/L to one algorithm would be a lie.
function renderAccountPage(content, accountId) {
  const account = accountRows().find((row) => row.id === accountId);
  if (!account) {
    content.innerHTML = `
      ${pageHeader({ title: accountId, subtitle: "Unknown account" })}
      <section class="card"><p class="emptyState">No account with this id is configured.</p></section>`;
    return;
  }

  const positions = state.positions[account.id];
  const activity = state.activity[account.id];
  const deployed = bindings().filter((binding) => binding.account_id === account.id);
  const status = deploymentStatus(deployed);
  const busy = Boolean(state.positionsLoading[account.id] || state.activityLoading[account.id]);
  const meta = `<span class="pill is-${status}">${
    status === "live" ? "Trading" : status === "idle" ? "Agent-driven" : deployed.length ? "Deployed, paused" : "No algorithm"}</span>`;
  const actions = `<button class="ctl" type="button" id="refreshAccountButton" data-account="${escapeHtml(account.id)}"
    ${busy ? "disabled" : ""}>${busy ? "Refreshing." : "Refresh"}</button>`;

  content.innerHTML = `
    ${pageHeader({
      title: account.label,
      subtitle: `${account.broker}${account.data_feed ? ` · ${account.data_feed}` : ""} · ${account.id}`,
      meta,
      actions,
    })}
    <div class="pageBody">
    <section class="card">
      <div class="metricRow">
        <div class="metric"><span>Equity</span><strong>${positions ? escapeHtml(money(positions.equity, 2)) : "--"}</strong></div>
        <div class="metric"><span>Cash</span><strong>${positions ? escapeHtml(money(positions.cash, 2)) : "--"}</strong></div>
        <div class="metric"><span>Day P/L</span><strong class="${(positions?.day_pl || 0) >= 0 ? "gain" : "loss"}">${
          // A local book has no yesterday to compare against, so it reports no day figure.
          positions?.day_pl === null || positions?.day_pl === undefined
            ? "--"
            : `${escapeHtml(money(positions.day_pl, 2))} (${escapeHtml(percent(positions.day_pl_percent))})`}</strong></div>
        <div class="metric"><span>Open P/L</span><strong class="${(positions?.total_pl || 0) >= 0 ? "gain" : "loss"}">${
          positions ? escapeHtml(money(positions.total_pl, 2)) : "--"}</strong></div>
        <div class="metric"><span>Dividends (1y)</span><strong class="${(positions?.dividend_pl || 0) >= 0 ? "gain" : "loss"}">${
          // Reported beside Open P/L, never inside it. Price appreciation and income are
          // different things, and a T-bill sleeve earns almost entirely through this one.
          positions?.dividend_pl === null || positions?.dividend_pl === undefined
            ? "--"
            : escapeHtml(money(positions.dividend_pl, 2))}</strong></div>
      </div>
      ${!account.credentials_ready
        ? `<p class="cardHint">Credentials missing: set <code>${escapeHtml(account.missing_env.join("</code> and <code>"))}</code> in <code>.env</code> and restart. It cannot trade until then.</p>`
        : `<p class="cardHint">${account.broker === "paper"
            // No broker holds this money, so the usual "including your own orders" caveat
            // would be nonsense here: nothing but this bot can touch a local book.
            ? "A local book, not a broker. Orders fill instantly at the last price the algorithm saw, and no real money moves."
            : "Everything the broker reports for this account, including orders you placed yourself."}${
            deployed.length ? ` Algorithms running here: ${deployed.map((binding) => strategyByKey(binding.strategy).name).join(", ")}.` : ""}</p>`}
      ${deployed.length ? `<div class="chipRow">${deployed.map((binding) => `
        <a class="chip is-link" href="#/algo/${escapeHtml(binding.strategy)}/${DEFAULT_TAB}">${escapeHtml(strategyByKey(binding.strategy).name)}</a>`).join("")}</div>` : ""}
    </section>
    <div class="accountLayout">
      <div class="accountStack">
        <section class="card">
          <div class="cardHead">
            <h2>Positions</h2>
            <span class="cardHint">${positions?.rows?.length ? `${positions.rows.length} open` : ""}</span>
          </div>
          ${accountPositionsTable(positions)}
        </section>
        <section class="card">
          <div class="cardHead">
            <h2>Dividends received</h2>
            <span class="cardHint">${positions?.dividend_rows?.length ? `${positions.dividend_rows.length} shown` : ""}</span>
          </div>
          ${accountDividendsTable(positions)}
        </section>
      </div>
      <section class="card">
        <div class="cardHead">
          <h2>Recent orders</h2>
          <span class="cardHint">${activity?.rows?.length ? `${activity.rows.length} shown` : ""}</span>
        </div>
        ${accountOrdersTable(activity)}
      </section>
    </div>
    </div>`;

  if (account.credentials_ready) {
    ensurePositions(account.id);
    ensureActivity(account.id);
  }
}

function accountPositionsTable(positions) {
  if (positions?.error) return `<p class="emptyState">${escapeHtml(positions.error)}</p>`;
  if (!positions) return `<p class="emptyState">Loading positions.</p>`;
  if (!positions.rows?.length) return `<p class="emptyState">No open positions.</p>`;
  return `
    <div class="tableWrap is-scroll">
      <table class="dataTable">
        <thead>
          <tr><th>Symbol</th><th class="num">Qty</th><th class="num">Avg</th><th class="num">Value</th><th class="num">P/L</th></tr>
        </thead>
        <tbody>
          ${positions.rows.map((row) => `
            <tr>
              <td><strong>${escapeHtml(row.symbol)}</strong></td>
              <td class="num">${escapeHtml(num(row.qty, row.qty % 1 ? 3 : 0))}</td>
              <td class="num">${escapeHtml(money(row.avg_entry_price, 2))}</td>
              <td class="num">${escapeHtml(money(row.market_value, 2))}</td>
              <td class="num ${row.unrealized_pl >= 0 ? "gain" : "loss"}">${escapeHtml(money(row.unrealized_pl, 2))}
                <span class="tableNote">${escapeHtml(percent(row.unrealized_plpc))}</span></td>
            </tr>`).join("")}
        </tbody>
      </table>
    </div>`;
}

function accountDividendsTable(positions) {
  if (positions?.error) return `<p class="emptyState">${escapeHtml(positions.error)}</p>`;
  if (!positions) return `<p class="emptyState">Loading dividends.</p>`;
  if (!positions.dividend_rows?.length) {
    // An account that has simply not been paid yet is not an error, and neither is a broker
    // that cannot report income -- say so plainly rather than showing a blank card.
    return `<p class="emptyState">No dividends received in the last year.</p>`;
  }
  return `
    <div class="tableWrap is-scroll">
      <table class="dataTable">
        <thead>
          <tr><th>Date</th><th>Symbol</th><th class="num">Amount</th></tr>
        </thead>
        <tbody>
          ${positions.dividend_rows.map((row) => `
            <tr>
              <td>${escapeHtml(row.date || "")}</td>
              <td><strong>${escapeHtml(row.symbol || "Cash")}</strong>${
                row.description ? `<span class="tableNote">${escapeHtml(row.description)}</span>` : ""}</td>
              <td class="num ${row.amount >= 0 ? "gain" : "loss"}">${escapeHtml(money(row.amount, 2))}</td>
            </tr>`).join("")}
        </tbody>
      </table>
    </div>`;
}

function accountOrdersTable(activity) {
  if (activity?.error) return `<p class="emptyState">${escapeHtml(activity.error)}</p>`;
  if (!activity) return `<p class="emptyState">Loading activity.</p>`;
  if (!activity.rows?.length) return `<p class="emptyState">No orders yet.</p>`;
  return `
    <div class="tableWrap is-scroll">
      <table class="dataTable">
        <thead>
          <tr><th>Time</th><th>Symbol</th><th>Side</th><th>Detail</th><th>Status</th></tr>
        </thead>
        <tbody>
          ${activity.rows.map((row) => `
            <tr>
              <td class="nowrap">${escapeHtml(formatActivityTime(row.submitted_at))}</td>
              <td><strong>${escapeHtml(row.symbol)}</strong></td>
              <td><span class="side is-${escapeHtml(row.side)}">${escapeHtml(row.side)}</span></td>
              <td>${escapeHtml(formatActivityDetail(row))}</td>
              <td class="tableNote">${escapeHtml(row.status)}</td>
            </tr>`).join("")}
        </tbody>
      </table>
    </div>`;
}

//: The cached copies are what make the sidebar cheap, so a manual refresh has to drop them
//: before asking again -- ensurePositions treats "already present" as "nothing to do".
function refreshAccount(accountId) {
  delete state.positions[accountId];
  delete state.activity[accountId];
  ensurePositions(accountId);
  ensureActivity(accountId);
  render();
}


function explainerCard(strategy) {
  const explainer = state.algorithmConfigs[strategy.key]?.explainer;
  if (!explainer?.summary) return "";
  return `
    <section class="card">
      <h2>How it decides</h2>
      <p class="cardBody">${escapeHtml(explainer.summary)}</p>
      ${explainer.formula?.length
        ? `<pre class="formula">${explainer.formula.map((line) => escapeHtml(line)).join("\n")}</pre>`
        : ""}
    </section>`;
}

// The two halves of an algorithm's tuning are separate things and both belong here. Bursty
// DCA's budgets live in its plan (the bubble board); everything else an algorithm exposes
// lives in its config section (the parameter form). It used to get only the board, which left
// its regime gate, scaling factor and cap knobs -- all of them present in config and served by
// /api/algorithm-config -- with no way to edit them at all.
//
// The board is absent on the first paint because the algorithm has not said it wants one yet;
// ``ensureAlgorithmConfig`` re-renders when the payload lands, and the board's own guards
// already tolerate a config that has not arrived.
function renderTuneTab(body, strategy) {
  const hasBudgets = usesBudgetsEditor(strategy.key);
  ensureAlgorithmConfig(strategy.key);
  body.innerHTML = `
    ${explainerCard(strategy)}
    ${hasBudgets ? `
    <section class="card tuneCard">
      <div class="cardHead">
        <h2>Budgets</h2>
        <span class="cardHint" id="tuneHint"></span>
      </div>
      <div class="tuneBody" id="dcaBoard"></div>
    </section>` : ""}
    <section class="card tuneCard">
      <div class="cardHead">
        <h2>${hasBudgets ? "Parameters" : "Configuration"}</h2>
        <span class="cardHint" id="configHint"></span>
      </div>
      <div class="tuneBody" id="tuneBody"></div>
      <div class="cardActions" id="configActions" hidden><button class="ctl" type="button" id="saveConfigButton">Save changes</button></div>
    </section>`;
  if (hasBudgets) renderDcaTuner($("#dcaBoard"), strategy);
  renderConfigForm($("#tuneBody"), strategy);
}

function renderDcaTuner(host, strategy) {
  const hint = $("#tuneHint");
  const entry = state.algorithmConfigs[strategy.key];
  const plan = entry?.explainer?.parameters?.plan;
  // The board edits this algorithm's own plan, so which page you are on decides what you are
  // editing. It used to edit the first DCA binding's account whichever page you were on.
  if (state.planStrategy !== strategy.key) {
    // A different algorithm's board: drop the old bubbles rather than animating them into
    // place as though they were this plan's.
    state.nodes = [];
    state.selected = null;
    hideAmountEntry();
  }
  state.planStrategy = strategy.key;
  if (!entry) {
    host.innerHTML = `<p class="emptyState">Loading budgets.</p>`;
    return;
  }
  if (hint) hint.textContent = `Dollars per month, per symbol · algorithms.${entry.config_key || strategy.key}.plan`;
  host.innerHTML = `<svg class="bubbleBoard" id="bubbleBoard" role="img"
    aria-label="Interactive buy and sell budget bubbles"></svg>
    <p class="cardHint">${escapeHtml(plan?.effect || "")} Scroll a bubble to change its budget, or select one and type the amount. Drag between buckets, drag one off the buckets to remove it, double-click to add.
      <span id="planSaveStatus" class="saveStatus">${escapeHtml(planSaveStatusText())}</span></p>`;
  renderDca();
}

function renderConfigForm(host, strategy) {
  const entry = state.algorithmConfigs[strategy.key];
  const hint = $("#configHint");
  if (!entry) {
    host.innerHTML = `<p class="emptyState">Loading configuration.</p>`;
    ensureAlgorithmConfig(strategy.key);
    return;
  }
  if (hint) hint.textContent = `config/algorithms.yaml · ${entry.config_key || strategy.key}`;
  // The plan has its own editor on this page, so it is not also offered as a raw JSON box.
  // saveCurrentConfig merges over the loaded config rather than replacing it, so leaving it
  // out of the form does not drop it on save.
  const docs = entry.explainer?.parameters || {};
  // Ordered by the explainer, not by the config file. The file's order is whatever the last
  // writer happened to produce -- the dashboard rewrites that section from its own state -- so
  // reading order from it made the form reshuffle itself between saves. The explainer's order
  // is deliberate: the knobs you actually reach for first, and related ones adjacent.
  const fields = orderedConfigFields(entry.config || {}, docs)
    .filter(([key]) => !isPlanField(strategy.key, key));
  host.innerHTML = fields.length
    ? `<div class="configForm">${fields.map(([key, value]) => renderConfigField(key, value, docs[key])).join("")}</div>`
    : `<p class="emptyState">This algorithm has no tunable parameters.</p>`;
  // Plain DCA has no parameters of its own -- its config is the plan on the board above -- so
  // it would otherwise offer a button that saves an empty object over its config section.
  const actions = $("#configActions");
  if (actions) actions.hidden = !fields.length;
}

//: Config entries in the explainer's order, with anything undocumented kept at the end rather
//: than dropped -- a knob with no description is still a knob, and hiding it would make a saved
//: value invisible and unremovable.
function orderedConfigFields(config, docs) {
  const documented = Object.keys(docs);
  const rank = new Map(documented.map((key, index) => [key, index]));
  return Object.entries(config).sort(([a], [b]) => {
    const ra = rank.has(a) ? rank.get(a) : Number.MAX_SAFE_INTEGER;
    const rb = rank.has(b) ? rank.get(b) : Number.MAX_SAFE_INTEGER;
    return ra === rb ? a.localeCompare(b) : ra - rb;
  });
}

function renderBacktestTab(body, strategy) {
  const backtest = state.backtests[strategy.key];
  const loading = Boolean(state.backtestLoading[strategy.key]);
  // The backend says whether a replay is meaningful for this algorithm. Offering the controls
  // anyway would present an action that cannot succeed, and the reason only after it failed.
  if (backtest && backtest.supported === false) {
    body.innerHTML = `
      <section class="card">
        <div class="cardHead"><h2>Backtest</h2></div>
        <p class="emptyState">${escapeHtml(backtest.error || "This algorithm cannot be backtested.")}</p>
      </section>`;
    return;
  }
  body.innerHTML = `
    <section class="card">
      <div class="cardHead">
        <h2>Backtest</h2>
        <div class="cardHeadActions">
          <select class="ctl" id="backtestPeriodSelect" aria-label="Backtest period">
            ${BACKTEST_PERIOD_CHOICES.map((period) => `
              <option value="${escapeHtml(period)}"${period === BACKTEST_PERIOD ? " selected" : ""}>${escapeHtml(backtestPeriodLabel(period))}</option>
            `).join("")}
          </select>
          <button class="ctl is-primary" type="button" id="runBacktestButton" ${loading ? "disabled" : ""}>
            ${loading ? "Running." : "Run backtest"}
          </button>
        </div>
      </div>
      <div class="metricRow">
        <div class="metric"><span>Return</span><strong class="${(backtest?.total_return || 0) >= 0 ? "gain" : "loss"}">${backtest ? escapeHtml(percent(backtest.total_return)) : "--"}</strong></div>
        <div class="metric"><span>Ending equity</span><strong>${backtest ? escapeHtml(money(backtest.ending_equity)) : "--"}</strong></div>
        <div class="metric"><span>Orders</span><strong>${backtest?.orders?.total_orders ?? "--"}</strong></div>
        <div class="metric"><span>Status</span><strong>${escapeHtml(backtestStatusLabel(backtest, loading))}</strong></div>
      </div>
      <svg class="chartLarge" id="backtestChart" role="img" aria-label="Backtest equity curve"></svg>
      ${[backtestCaptionText(backtest, loading), backtestEquityText(backtest), backtestOrderText(backtest)]
        .filter(Boolean)
        .map((line) => `<p class="cardHint">${escapeHtml(line)}</p>`)
        .join("")}
    </section>`;
  renderBacktestChart(backtest, $("#backtestChart"));
  if (!backtest && !loading) loadBacktest(strategy.key, false, { cacheOnly: true });
}

function renderOverviewTab(body, strategy, deployment) {
  const backtest = state.backtests[strategy.key];
  body.innerHTML = `
    <div class="cardGrid">
      <section class="card is-wide">
        <h2>How it works</h2>
        <p class="cardBody">${escapeHtml(strategy.logic)}</p>
        <div class="chipRow">
          ${(strategy.signals || []).map((signal) => `<span class="chip">${escapeHtml(signal)}</span>`).join("")}
        </div>
      </section>
      <section class="card">
        <h2>At a glance</h2>
        <dl class="factList">
          <div><dt>Horizon</dt><dd>${escapeHtml(strategy.horizon)}</dd></div>
          <div><dt>Risk</dt><dd>${escapeHtml(strategy.risk)}</dd></div>
          <div><dt>Account</dt><dd>${deployment
            ? `<a class="factLink" href="#/account/${escapeHtml(deployment.account_id)}">${escapeHtml(accountLabel(deployment.account_id))}</a>`
            : "none"}</dd></div>
          <div><dt>${escapeHtml(BACKTEST_LABEL)} backtest</dt><dd>${backtest ? escapeHtml(percent(backtest.total_return)) : "--"}</dd></div>
        </dl>
      </section>
    </div>
    <section class="card">
      <div class="cardHead">
        <h2>Orders this algorithm placed</h2>
        <div class="cardHeadActions">
          ${deployment ? `<span class="cardHint">on <a class="factLink" href="#/account/${escapeHtml(deployment.account_id)}">${escapeHtml(accountLabel(deployment.account_id))}</a></span>` : ""}
          <button class="ctl" type="button" id="refreshAlgoOrdersButton">Refresh</button>
        </div>
      </div>
      ${algorithmOrdersTable(state.algorithmActivity[strategy.key])}
    </section>`;
  ensureAlgorithmActivity(strategy.key);
}

function renderSignalsTab(body, strategy) {
  // Cache-first: whatever snapshot was last computed stays on screen -- even while a refresh
  // runs in the background -- and nothing is fetched just because the tab opened.
  const payload = state.signals[strategy.key];
  const loading = Boolean(state.signalLoading[strategy.key]);
  // Already ordered by the algorithm: what the run changed first, then holdings, then the rest.
  const rows = payload?.rows || [];
  const asOf = payload && !payload.error && payload.updated_at
    ? `<span class="cardHint">as of ${escapeHtml(formatActivityTime(payload.updated_at))}</span>`
    : "";
  body.innerHTML = `
    <section class="card">
      <div class="cardHead">
        <h2>Live signals</h2>
        <div class="cardHeadActions">
          ${asOf}
          <button class="ctl" type="button" id="refreshUniverseButton" ${state.universeRefreshing ? "disabled" : ""}>Refresh universe</button>
          <button class="ctl" type="button" id="refreshSignalsButton" ${loading ? "disabled" : ""}>${loading ? "Refreshing." : "Refresh"}</button>
        </div>
      </div>
      ${(payload?.summary || []).length ? `<div class="metricRow">
        ${payload.summary.map((item) => `<div class="metric"><span>${escapeHtml(item.label)}</span><strong>${escapeHtml(item.value)}</strong></div>`).join("")}
      </div>` : ""}
      <div class="signalRows" id="signalRows">
        ${state.universeProposal || state.universeRefreshing
          ? renderUniverseProposalRows()
          : rows.length
            ? renderSignalTable(rows)
            : loading
              ? `<p class="emptyState">Fetching live signal snapshot.</p>`
              : payload?.error
                ? `<p class="emptyState">${escapeHtml(payload.error)}</p>`
                : payload
                  ? renderSignalFallbackRows(strategy, payload, (strategy.signals || []).slice(0, 5))
                  : `<p class="emptyState">No live signals cached yet. Hit Refresh to compute them.</p>`}
      </div>
    </section>`;
  // Cache-first, like the Backtest tab: probe for a stored snapshot without ever computing.
  if (!payload && !loading) loadSignals(strategy.key, false, { cacheOnly: true });
}

//: The bot's own journal, not the broker's feed: it is the only record that knows which
//: algorithm asked for an order, so it is the one view that is honestly per-algorithm.
function algorithmOrdersTable(journal) {
  if (journal?.error) return `<p class="emptyState">${escapeHtml(journal.error)}</p>`;
  if (!journal) return `<p class="emptyState">Loading orders.</p>`;
  if (!journal.rows?.length) {
    return `<p class="emptyState">No orders placed yet. This lists what the bot submitted for this algorithm, so orders you place yourself appear only on the account page.</p>`;
  }
  return `
    <div class="tableWrap is-scroll">
      <table class="dataTable">
        <thead>
          <tr><th>Time</th><th>Symbol</th><th>Side</th><th class="num">Qty</th><th>Status</th></tr>
        </thead>
        <tbody>
          ${journal.rows.map((row) => `
            <tr>
              <td class="nowrap">${escapeHtml(formatActivityTime(row.submitted_at))}</td>
              <td><strong>${escapeHtml(row.symbol)}</strong></td>
              <td><span class="side is-${escapeHtml(orderSideClass(row.side))}">${escapeHtml(row.side)}</span></td>
              <td class="num">${escapeHtml(row.quantity ? num(row.quantity, row.quantity % 1 ? 3 : 0) : "--")}</td>
              <td class="tableNote" title="${escapeHtml(row.reason || "")}">${escapeHtml(row.status)}</td>
            </tr>`).join("")}
        </tbody>
      </table>
    </div>`;
}

// The journal records the algorithm's verb ("add"/"trim"), which has to map onto the two
// colours the side chip knows about.
function orderSideClass(side) {
  const value = String(side || "").toLowerCase();
  if (value === "buy" || value === "add") return "buy";
  if (value === "sell" || value === "trim") return "sell";
  return "none";
}

// -- target page -------------------------------------------------------------------------

// -- data ---------------------------------------------------------------------------------

async function ensureAlgorithmConfig(strategyKey) {
  if (state.algorithmConfigs[strategyKey] || state.algorithmConfigLoading[strategyKey]) return;
  state.algorithmConfigLoading[strategyKey] = true;
  try {
    state.algorithmConfigs[strategyKey] = await api(
      `/api/algorithm-config?strategy=${encodeURIComponent(strategyKey)}`, { timeoutMs: 8000 });
  } catch (error) {
    state.algorithmConfigs[strategyKey] = { error: error.message, config: {} };
  } finally {
    state.algorithmConfigLoading[strategyKey] = false;
    render();
  }
}

function formatActivityDetail(row) {
  const filled = Number(row.filled_qty || 0);
  if (filled > 0 && row.filled_avg_price) {
    return `${num(filled, filled % 1 ? 3 : 0)} @ ${money(row.filled_avg_price, 2)}`;
  }
  const qty = Number(row.qty || 0);
  return qty ? `${num(qty, qty % 1 ? 3 : 0)} requested` : "--";
}

function formatActivityTime(value) {
  if (!value) return "--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value).slice(0, 16);
  return date.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

async function ensureActivity(accountId) {
  const key = accountId || "";
  if (state.activity[key] || state.activityLoading[key]) return;
  state.activityLoading[key] = true;
  try {
    state.activity[key] = await api(`/api/activity?account_id=${encodeURIComponent(key)}`, { timeoutMs: 15000 });
  } catch (error) {
    state.activity[key] = { error: error.message, rows: [] };
  } finally {
    state.activityLoading[key] = false;
    render();
  }
}

async function ensureAlgorithmActivity(strategyKey) {
  if (state.algorithmActivity[strategyKey] || state.algorithmActivityLoading[strategyKey]) return;
  state.algorithmActivityLoading[strategyKey] = true;
  try {
    state.algorithmActivity[strategyKey] = await api(
      `/api/algorithm-activity?strategy=${encodeURIComponent(strategyKey)}`, { timeoutMs: 8000 });
  } catch (error) {
    state.algorithmActivity[strategyKey] = { error: error.message, rows: [] };
  } finally {
    state.algorithmActivityLoading[strategyKey] = false;
    render();
  }
}

async function ensurePositions(accountId) {
  const key = accountId || "";
  if (state.positions[key] || state.positionsLoading[key]) return;
  state.positionsLoading[key] = true;
  try {
    state.positions[key] = await api(`/api/positions?account_id=${encodeURIComponent(key)}`, { timeoutMs: 15000 });
  } catch (error) {
    state.positions[key] = { error: error.message, rows: [] };
  } finally {
    state.positionsLoading[key] = false;
    render();
  }
}

async function loadAccounts() {
  try {
    state.accounts = await api("/api/accounts", { timeoutMs: 6000 });
  } catch (error) {
    state.accounts = { rows: [] };
  }
}

//: The plan is config, but it has the bubble board rather than a form field. Everything that
//: walks the config form has to know to leave it alone.
function isPlanField(strategyKey, key) {
  return key === "plan" && usesBudgetsEditor(strategyKey);
}

async function saveCurrentConfig(strategyKey) {
  const host = $("#tuneBody");
  if (!host) return;
  let values;
  try {
    values = collectConfigValues(host);
  } catch (error) {
    showToast(error.message);
    return;
  }
  try {
    // Merged over the loaded config, not sent in its place: the server replaces the whole
    // section, so posting only the rendered fields would delete the plan the board edits.
    const merged = { ...(state.algorithmConfigs[strategyKey]?.config || {}), ...values };
    state.algorithmConfigs[strategyKey] = await api("/api/algorithm-config", {
      method: "POST",
      body: JSON.stringify({ strategy: strategyKey, config: merged }),
      timeoutMs: 8000,
    });
    // Tuning feeds the backtest cache key, so the cached curve no longer describes this config.
    delete state.backtests[strategyKey];
    delete state.signals[strategyKey];
    showToast("Configuration saved");
    render();
  } catch (error) {
    showToast(error.message);
  }
}

async function saveBindings() {
  try {
    const payload = await api("/api/controls", {
      method: "POST",
      body: JSON.stringify({ controls: state.controls }),
      timeoutMs: 6000,
    });
    state.controls = payload.controls;
    state.bot = payload.bot || state.bot;
    await loadAccounts();
  } catch (error) {
    showToast(error.message);
  }
  render();
}

async function setDeploymentAccount(strategyKey, accountId) {
  const deployment = deploymentFor(strategyKey);
  if (deployment) {
    if (deployment.account_id === accountId) return;
    deployment.account_id = accountId;
    await saveBindings();
    return;
  }
  await deployTo(strategyKey, accountId);
}

async function setDeploymentCron(bindingId, expression) {
  const binding = bindingById(bindingId);
  if (!binding) return;
  const cron = normalizeBindingCron(expression);
  const problem = cronError(cron);
  if (problem) {
    // Refused rather than saved-and-corrected. Falling back silently would leave the field
    // showing one schedule while the scheduler ran another, and a schedule is the one setting
    // where being quietly wrong costs a trading day.
    showToast(problem);
    return;
  }
  binding.cron = cron;
  await saveBindings();
  showToast(cron ? `Schedule: ${describeCron(cron)}` : "Schedule cleared -- agent-driven");
}

async function deployTo(strategyKey, accountId) {
  const used = new Set(bindings().map((binding) => String(binding.id)));
  let index = 1;
  while (used.has(`b${index}`)) index += 1;
  // No cron key at all: the server fills in the algorithm's own default. Sending "" here
  // would mean "this binding wants no clock", and a new deployment would sit switched on
  // and never run.
  bindings().push({ id: `b${index}`, strategy: strategyKey, account_id: accountId, enabled: false });
  await saveBindings();
  showToast(`Deployed to ${accountLabel(accountId)}`);
}

async function toggleDeployment(bindingId) {
  const binding = bindingById(bindingId);
  if (!binding) return;
  binding.enabled = !binding.enabled;
  await saveBindings();
}

async function removeDeployment(bindingId) {
  if (bindings().length <= 1) {
    showToast("Keep at least one deployment");
    return;
  }
  state.controls.bindings = bindings().filter((binding) => String(binding.id) !== String(bindingId));
  await saveBindings();
}

async function loadSchwabAuth() {
  try {
    state.schwabAuth = await api("/api/schwab/auth", { timeoutMs: 5000 });
  } catch (error) {
    state.schwabAuth = null;
  }
  renderNavFooter();
}

//: The footer claims to show what the runtime is doing right now, so it cannot rely on the
//: snapshot taken at page load -- a run that started an hour ago would still read "armed".
async function refreshRuntimeStatus() {
  try {
    const payload = await api("/api/status", { timeoutMs: 5000 });
    state.status = payload;
    state.bot = payload.bot || state.bot;
  } catch (error) {
    return;
  }
  renderNavFooter();
}

async function connectSchwab() {
  // The row is shown for a configured *connector*, which may still be missing its
  // credentials. Say which ones rather than opening a popup that can only fail.
  if (state.schwabAuth && !state.schwabAuth.configured) {
    showToast(state.schwabAuth.detail || "Schwab is not configured.");
    return;
  }
  const popup = window.open("", "schwabAuth", "width=560,height=760");
  try {
    const payload = await api("/api/schwab/auth/start", { method: "POST", timeoutMs: 8000 });
    if (popup) popup.location = payload.authorize_url;
    else showToast("Allow popups to authorize Schwab.");
  } catch (error) {
    popup?.close();
    showToast(`Could not start Schwab authorization: ${error.message}`);
  }
}

// -- events -------------------------------------------------------------------------------

function closeNavOnMobile() {
  if (window.innerWidth > 900) return;
  $("#sidebar")?.classList.remove("is-open");
  $("#navToggle")?.setAttribute("aria-expanded", "false");
}

function wireEvents() {
  window.addEventListener("hashchange", render);
  window.addEventListener("resize", () => {
    if ($("#bubbleBoard")) renderDca();
  });

  $("#navToggle")?.addEventListener("click", () => {
    const sidebar = $("#sidebar");
    const open = sidebar?.classList.toggle("is-open");
    $("#navToggle")?.setAttribute("aria-expanded", String(Boolean(open)));
  });

  // Delegated: page bodies are replaced wholesale on every render.
  $("#content")?.addEventListener("click", (event) => {
    const route = currentRoute();
    const refreshAccountButton = event.target.closest("#refreshAccountButton");
    if (refreshAccountButton) return refreshAccount(refreshAccountButton.dataset.account);
    if (event.target.closest("#saveConfigButton")) return saveCurrentConfig(route.id);
    if (event.target.closest("#runBacktestButton")) return loadBacktest(route.id, true);
    if (event.target.closest("#refreshSignalsButton")) {
      // Background refresh: the cached snapshot stays on screen until the new one lands.
      return loadSignals(route.id, true);
    }
    const signalRow = event.target.closest("[data-signal-symbol]");
    if (signalRow) {
      const symbol = signalRow.dataset.signalSymbol;
      if (state.expandedSignals.has(symbol)) state.expandedSignals.delete(symbol);
      else state.expandedSignals.add(symbol);
      return render();
    }
    if (event.target.closest("#refreshAlgoOrdersButton")) {
      delete state.algorithmActivity[route.id];
      return ensureAlgorithmActivity(route.id);
    }
    if (event.target.closest("#refreshUniverseButton")) return recommendUniverse();
    if (event.target.closest("[data-apply-universe]")) return applyUniverseProposal();
    if (event.target.closest('[data-role="power"]')) {
      const deployment = deploymentFor(route.id);
      // Arming with no deployment yet means deploying to whatever the picker shows.
      if (deployment) return toggleDeployment(deployment.id);
      const target = $("#deployTargetSelect")?.value;
      if (target) return deployTo(route.id, target).then(() => toggleDeployment(deploymentFor(route.id)?.id));
      return;
    }
  });

  // Reads back the expression as it is typed, before the change event saves it. A cron string
  // is write-only otherwise: nothing on the screen tells you "0 11 * * 1-5" is 11am weekdays
  // until after you have committed it to a live binding.
  $("#content")?.addEventListener("input", (event) => {
    if (event.target.id !== "deployCronInput") return;
    const hint = $("#deployCronHint");
    if (!hint) return;
    const problem = cronError(event.target.value);
    hint.textContent = problem || describeCron(event.target.value);
    hint.classList.toggle("is-bad", Boolean(problem));
  });

  $("#content")?.addEventListener("change", (event) => {
    if (event.target.id === "deployTargetSelect") {
      setDeploymentAccount(currentRoute().id, event.target.value);
      return;
    }
    if (event.target.id === "deployCronInput") {
      const bindingId = event.target.dataset.binding;
      if (bindingId) setDeploymentCron(bindingId, event.target.value);
      return;
    }
    if (event.target.id === "backtestPeriodSelect") {
      configureBacktestPeriod(event.target.value);
      // Forced: the select still holds focus after its own change event, so an ordinary
      // render would defer, and the cached payload for the newly chosen window would sit in
      // state unpainted until something else moved focus.
      render({ force: true });
      loadBacktest(currentRoute().id, false, { cacheOnly: true });
    }
  });

  // Whatever was deferred while a control had focus still owes the screen a paint. The
  // timeout lets focus settle, since focusout fires before the new activeElement is set.
  $("#content")?.addEventListener("focusout", () => {
    if (!renderDeferred) return;
    window.setTimeout(() => {
      if (renderDeferred) render();
    }, 0);
  });

  // Rows are focusable buttons, so they have to answer the keys a button answers to. Without
  // this the gate breakdown is mouse-only, which is the one part of the deck that explains why
  // the bot did what it did.
  $("#content")?.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const signalRow = event.target.closest?.("[data-signal-symbol]");
    if (!signalRow) return;
    event.preventDefault();
    const symbol = signalRow.dataset.signalSymbol;
    if (state.expandedSignals.has(symbol)) state.expandedSignals.delete(symbol);
    else state.expandedSignals.add(symbol);
    render();
  });

  $("#navFooter")?.addEventListener("click", (event) => {
    if (event.target.closest("#schwabAuthPill")) connectSchwab();
  });

  $("#symbolEntry")?.addEventListener("keydown", (event) => {
    if (event.key === "Enter") commitSymbolEntry();
    if (event.key === "Escape") hideSymbolEntry();
  });
  $("#symbolEntry")?.addEventListener("input", () => {
    if (!state.draft) return;
    state.draft.symbol = $("#symbolEntry").value.trim().toUpperCase().slice(0, 6);
    renderBoard();
  });
  $("#symbolEntry")?.addEventListener("blur", () => window.setTimeout(commitSymbolEntry, 80));

  $("#amountEntry")?.addEventListener("keydown", (event) => {
    // Kept off the window handler below, which ignores events from inputs: while this is open
    // it owns the keyboard, so Escape must abandon rather than fall through to anything else.
    if (event.key === "Enter") {
      event.preventDefault();
      commitAmountEntry();
    }
    if (event.key === "Escape") {
      event.preventDefault();
      cancelAmountEntry();
    }
  });
  $("#amountEntry")?.addEventListener("input", previewAmountEntry);
  $("#amountEntry")?.addEventListener("blur", () => window.setTimeout(commitAmountEntry, 80));

  window.addEventListener("keydown", (event) => {
    const tag = event.target?.tagName?.toLowerCase();
    if (tag === "input" || tag === "textarea" || event.target?.isContentEditable) return;
    // Type on a selected bubble to set its budget outright. A digit starts the edit and is
    // carried into the field, because focusing an input mid-keydown does not deliver the
    // keystroke that caused it; Enter opens the field on the current value instead.
    if (state.selected && !state.draft && (event.key === "Enter" || /^[0-9]$/.test(event.key))) {
      const node = selectedDcaNode();
      if (node) {
        event.preventDefault();
        showAmountEntry(node, event.key === "Enter" ? "" : event.key);
        return;
      }
    }
    if ((event.key === "Delete" || event.key === "Backspace") && state.selected) {
      const found = BUCKET_NAMES.flatMap((bucketName) =>
        bucketItems(bucketName).map((item) => ({ bucketName, item })),
      ).find(({ item }) => item.symbol === state.selected);
      if (found) {
        // The field is over a bubble that is about to stop existing, so it goes first --
        // and without committing, which would write the deleted symbol's budget back.
        hideAmountEntry();
        removeSymbol(found.bucketName, state.selected);
        state.selected = null;
      }
    }
  });

  window.addEventListener("message", (event) => {
    if (event.data?.type !== "schwab-auth") return;
    showToast(event.data.ok ? "Schwab connected." : "Schwab authorization failed.");
    loadSchwabAuth();
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) return;
    loadSchwabAuth();
    refreshRuntimeStatus();
  });
  window.setInterval(loadSchwabAuth, 5 * 60 * 1000);
  window.setInterval(refreshRuntimeStatus, 60 * 1000);
}

async function init() {
  wireEvents();
  loadSchwabAuth();
  render();
  try {
    const [statusPayload, universePayload, controlsPayload] = await Promise.all([
      api("/api/status", { timeoutMs: 5000 }),
      api("/api/universe", { timeoutMs: 5000 }),
      api("/api/controls", { timeoutMs: 5000 }),
    ]);
    state.status = statusPayload;
    configureBacktestPeriod(statusPayload.config?.backtest_period);
    state.universe = universePayload.rows || [];
    state.controls = controlsPayload.controls || state.controls;
    state.bot = controlsPayload.bot || statusPayload.bot || null;
    await loadAccounts();
  } catch (error) {
    showToast(`Could not load dashboard: ${error.message}`);
  }
  render();
}

init();
