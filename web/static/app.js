const SVG_NS = "http://www.w3.org/2000/svg";
const BUCKET_NAMES = ["buy", "sell"];
const MAX_AMOUNT = 2000;
const DCA_ALGORITHM_KEYS = ["dca", "bursty_dca"];
const WHEEL_STEP = 25;
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
    key: "dca",
    blurb: "Buys each symbol's monthly budget as soon as it clears the broker minimum.",
    name: "DCA",
    status: "Live",
    horizon: "Continuous",
    risk: "Lower",
    logic: "Accrues each symbol's monthly budget against elapsed wall-clock time and buys as soon as the accrued amount can clear a broker minimum, so the schedule controls only when it may act, never how much it spends.",
    signals: ["Monthly budget", "Accrued amount", "Minimum executable trade", "Whole-share threshold"],
  },
  {
    key: "bursty_dca",
    blurb: "Same monthly budget as DCA, but only deployed into a dip above the 200-day.",
    name: "Bursty DCA",
    status: "Live",
    horizon: "Continuous",
    risk: "Medium",
    logic: "Accrues the same monthly budget as DCA but only deploys it into a dip: price must be above its 200-day moving average, and Bollinger %B or Connors RSI(2) must be stretched. Trade size follows a value-averaging path, clamped per trade and per month.",
    signals: ["200-day regime gate", "Bollinger %B", "Connors RSI(2)", "Value-averaging path", "Trade and monthly clamps"],
  },
  {
    key: "fast_momentum",
    blurb: "Ranks risk-on and defensive ETFs on multi-horizon momentum, then sizes with caps.",
    name: "Fast Momentum",
    status: "Live",
    horizon: "Intraday",
    risk: "Medium",
    logic: "Ranks risk-on and defensive ETFs with nano, micro, meso, and macro momentum scores, then sizes selected positions dynamically with caps and rebalance thresholds.",
    signals: ["Nano momentum", "Micro momentum", "Meso trend", "Macro trend", "Pullback bonus"],
  },
  {
    key: "dual_momentum",
    blurb: "Relative momentum picks the leaders; absolute momentum decides if it may hold any.",
    name: "Dual Momentum",
    status: "Paper",
    horizon: "Intraday",
    risk: "Medium",
    logic: "Requires each ETF to clear its own absolute-trend test before it is ranked, holds the top few, re-ranks every few sessions rather than every one, and needs a score margin to displace a sitting position. Weights scale toward a portfolio volatility target.",
    signals: ["Absolute eligibility", "Slow rank", "Replacement margin", "Crash stop", "Vol target"],
  },
  {
    key: "spy_rotation",
    blurb: "Classifies the SPY regime, then rotates between growth, income, cash, and hedges.",
    name: "SPY Rotation",
    status: "Live",
    horizon: "Intraday",
    risk: "Medium",
    logic: "Classifies SPY as growing, flat, falling, or crisis using micro, meso, macro, and sentiment signals, then rotates among growth, covered-call income, cash, and capped hedges.",
    signals: ["SPY state", "Micro/meso/macro", "Sentiment", "XYLD flat state", "SH/VXX crisis hedge"],
  },
];

//: A saved strategy id that is unknown (or the retired "none") lands here.
const DEFAULT_ALGORITHM_KEY = "dca";

//: Strategies driven by the DCA plan, so a plan edit invalidates their cached views.
const DCA_STRATEGY_KEYS = ["dca", "bursty_dca"];

const state = {
  status: null,
  universe: [],
  controls: {
    trading_account_id: "",
    algorithm_enabled: false,
    active_strategy: "fast_momentum",
  },
  accounts: { rows: [] },
  bot: null,
  dca: null,
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
  animationId: null,
  backtests: {},
  backtestLoading: {},
  signals: {},
  signalLoading: {},
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

function bucketItems(bucketName) {
  return state.dca?.plan?.[bucketName]?.items || [];
}

function setBucketItems(bucketName, items) {
  state.dca.plan[bucketName].items = items.map((item) => ({
    symbol: item.symbol,
    amount: clamp(Number(item.amount || 0), 0, MAX_AMOUNT),
  }));
  state.dca.plan[bucketName].amount = state.dca.plan[bucketName].items.reduce(
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
  state.nodes.forEach(syncNodeToPlan);
  BUCKET_NAMES.forEach((bucketName) => setBucketItems(bucketName, bucketItems(bucketName)));
}

function renderBoard() {
  if (!state.dca?.plan || !$("#bubbleBoard")) return;
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
  group.appendChild(textEl({ class: "amount-label", y: 14 }, `$${Math.round(node.amount)}`));
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
  hideSymbolEntry();
  const wasSelected = state.selected === node.symbol;
  state.selected = wasSelected ? null : node.symbol;
  $("#bubbleBoard")?.classList.toggle("resize-mode", Boolean(state.selected));
  if (event.pointerType === "touch") updateBoardElements();
  event.currentTarget.setPointerCapture(event.pointerId);
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

  state.drag = { node, pointerId: event.pointerId, element: event.currentTarget };
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

function resizeNodeToAmount(node, amount) {
  node.amount = clamp(Math.round(amount / WHEEL_STEP) * WHEEL_STEP, 0, MAX_AMOUNT);
  node.radius = itemRadius(node.amount);
  syncNodeToPlan(node);
  schedulePlanSave();
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
    dragNode(event, node);
  }
}

function dragNode(event, node) {
  const point = eventToSvgPoint(event);
  const bucketName = nearestBucket(point);
  const clamped = clampPointToBucket(point, bucketName, node.radius);
  node.targetBucket = bucketName;
  node.bucketName = bucketName;
  node.x = clamped.x;
  node.y = clamped.y;
  node.vx = 0;
  node.vy = 0;
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
    event.currentTarget.classList.remove("dragging");
    state.drag = null;
    moveAsset(node);
    renderDca();
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
  state.dca.plan[draft.bucketName].items.push({
    symbol: row.symbol,
    amount,
  });
  setBucketItems(draft.bucketName, state.dca.plan[draft.bucketName].items);
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
    state.dca.plan[bucketName].items = bucketItems(bucketName).filter((item) => item.symbol !== node.symbol);
  });
  found.item.amount = node.amount;
  state.dca.plan[node.bucketName].items.push(found.item);
  BUCKET_NAMES.forEach((bucketName) => setBucketItems(bucketName, bucketItems(bucketName)));
  schedulePlanSave();
}

function renderDca() {
  if (!state.dca?.plan) return;
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

function algorithmChoices() {
  return STRATEGIES;
}

//: Bindings are pairs of algorithm and account. Signals and backtests are keyed by strategy,
//: so two bindings on the same strategy share those views; the account only changes execution.
function bindings() {
  return state.controls?.bindings || [];
}

function primaryDcaBindingId() {
  return bindings().find((binding) => DCA_ALGORITHM_KEYS.includes(binding.strategy))?.id || "";
}

function bindingById(bindingId) {
  return bindings().find((binding) => String(binding.id) === String(bindingId)) || null;
}

function isDcaEnabled() {
  // The bubbles are one shared plan, so any live DCA binding lights them up.
  return bindings().some((binding) => binding.enabled && DCA_ALGORITHM_KEYS.includes(binding.strategy));
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
      body: JSON.stringify({ strategy: strategyKey, period: BACKTEST_PERIOD, refresh, cache_only: cacheOnly }),
      // A fresh replay is the longest request the dashboard makes, and it grows with the
      // window: the 24M option covers roughly five times the trade dates the 4M one does.
      // The cache probe is a lookup and stays on the short timeout.
      timeoutMs: cacheOnly ? 15000 : 300000,
    });
    if (isBacktestPayload(payload)) {
      state.backtests[strategyKey] = payload;
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
    const dcaPayload = await api(
      `/api/dca?account_id=${encodeURIComponent(state.dca?.account_id || bindings()[0]?.account_id || "")}`,
      { timeoutMs: 5000 },
    );
    state.dca = dcaPayload;
    state.dca.plan.max_item_amount = MAX_AMOUNT;
    renderDca();
    render();
    // The universe change invalidates every algorithm's view, so reload the one on screen.
    loadSignals(currentRoute().id);
    showToast("Universe applied");
  } catch (error) {
    showToast(error.message);
  } finally {
    state.universeApplying = false;
    render();
  }
}

async function ensureSignals(strategyKey) {
  if (state.signals[strategyKey] || state.signalLoading[strategyKey]) return;
  await loadSignals(strategyKey);
}

async function loadSignals(strategyKey) {
  state.signalLoading[strategyKey] = true;
  render();
  try {
    state.signals[strategyKey] = await api(`/api/strategy-signals?strategy=${encodeURIComponent(strategyKey)}`, {
      timeoutMs: 60000,
    });
  } catch (error) {
    state.signals[strategyKey] = { error: error.message };
  } finally {
    state.signalLoading[strategyKey] = false;
    render();
  }
}

function formatSignalDetail(strategyKey, row) {
  if (strategyKey === "fast_momentum") {
    const components = row.score_components || {};
    const details = [
      row.reason || "Signal pending",
      `Macro ${signedNum(components.price_macro, 2)}`,
      `Meso ${signedNum(components.price_meso, 2)}`,
      `Micro ${signedNum(components.price_micro, 2)}`,
      `Nano ${signedNum(components.price_nano, 2)}`,
      `Pullback ${signedNum(components.pullback_uptrend, 2)}`,
      `Sentiment ${signedNum(components.sentiment ?? row.sentiment_component ?? row.sentiment_score ?? row.social_score, 2)}`,
    ];
    return details.join(" / ");
  }
  // Each layer that could have stopped this symbol, in the order it is applied -- the point
  // of the algorithm is that you can see which gate rejected a name, not just its score.
  if (strategyKey === "dual_momentum") {
    const details = [
      row.reason || "Signal pending",
      `Eligible ${row.eligible ? "yes" : "no"}`,
      row.rank ? `Rank ${row.rank}` : "Unranked",
      `Vol ${percent(row.annual_volatility)}`,
    ];
    if (row.close) details.push(`Close ${money(row.close, 2)}`);
    return details.join(" / ");
  }
  if (strategyKey === "spy_rotation") {
    const details = [];
    if (row.reason) details.push(row.reason);
    if (row.spy_state) details.push(`SPY ${String(row.spy_state).toLowerCase()}`);
    if (Math.abs(Number(row.pullback_score || 0)) >= 0.01) {
      details.push(`Pullback ${Number(row.pullback_score) > 0 ? "+" : ""}${num(row.pullback_score, 2)}`);
    }
    if (Number(row.sentiment_records || 0) > 0) {
      const providers = Array.isArray(row.sentiment_providers) && row.sentiment_providers.length
        ? row.sentiment_providers.join(", ")
        : "provider";
      details.push(`Sentiment ${num(row.sentiment_component ?? row.sentiment_score ?? row.social_score, 2)} (${providers})`);
    }
    if (row.close) details.push(`Close ${money(row.close, 2)}`);
    return details.join(" / ");
  }
  if (row.reason) {
    const close = row.close ? ` / Close ${money(row.close, 2)}` : "";
    return `${row.reason}${close}`;
  }
  if (DCA_ALGORITHM_KEYS.includes(strategyKey)) {
    const budget = `${money(row.monthly_budget, 0)}/month`;
    return `${budget} / Accrued ${money(row.accrued, 2)} of ${money(row.min_executable, 2)}`;
  }
  if (row.close) {
    return `Close ${money(row.close, 2)}`;
  }
  return "Live feed pending";
}

function signedNum(value, digits = 2) {
  const parsed = Number(value || 0);
  return `${parsed > 0 ? "+" : ""}${num(parsed, digits)}`;
}

function formatSignalHeadline(strategyKey, row) {
  if (DCA_ALGORITHM_KEYS.includes(strategyKey)) {
    const parts = [row.side || row.signal || "Signal", row.reason || ""];
    if (row.close) parts.push(`Close ${money(row.close, 2)}`);
    if (row.warning) parts.push(row.warning);
    return parts.filter(Boolean).join(" / ");
  }
  if (strategyKey === "fast_momentum" || strategyKey === "spy_rotation" || strategyKey === "dual_momentum") {
    const parts = [
      row.side || row.signal || "Signal",
      `Score ${num(row.score, 2)}`,
      `Weight ${percent(row.target_weight)}`,
    ];
    if (row.close) parts.push(`Close ${money(row.close, 2)}`);
    return parts.join(" / ");
  }
  return `${row.side || row.signal} / Score ${num(row.score, 2)} / Weight ${percent(row.target_weight)}`;
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
  const tradingLabel = backtest.source === "dca" ? "filled" : "cumulative turnover";
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
  const isTemplate = payload?.wired === false;
  const heading = isTemplate ? "Template signal model" : "No active live rows";
  const detail = isTemplate
    ? "Signal inputs are shown until this strategy is wired to live market rows."
    : payload
      ? "No symbols currently meet the live criteria; signal inputs are shown below."
      : "Waiting for the first live snapshot.";
  const inputs = signalInputs.length ? signalInputs : [selected.logic || "Signal configuration pending"];
  return `
    <article class="signalFallback">
      <strong>${escapeHtml(heading)}</strong>
      <span>${escapeHtml(detail)}</span>
    </article>
    ${inputs.map((signal) => `
      <article>
        <strong>${escapeHtml(signal)}</strong>
        <span>${isTemplate ? "Template input" : "Signal input"}</span>
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
  svg.appendChild(svgEl("line", { class: "axis-line", x1: left, y1: axisY, x2: width - right, y2: axisY }));
  dateTicks(rows, 3).forEach((row) => {
    const xRaw = xScale(row.date);
    // keep tick labels inside the chart area to avoid overflow at the edges
    const x = Math.min(Math.max(xRaw, left + 12), width - right - 12);
    svg.appendChild(svgEl("line", { class: "axis-line", x1: x, y1: axisY, x2: x, y2: axisY + 4 }));
    svg.appendChild(textEl({ x, y: height - 6, "text-anchor": "middle", class: "axis-label" }, formatDateTick(row.date, 4)));
  });
  svg.appendChild(svgEl("path", { class: "growth-line", stroke: color, d: path }));
  svg.appendChild(textEl({ x: left, y: valueLabelY, class: "chart-label" }, money(rows[0].equity)));
  svg.appendChild(textEl({ x: width - right, y: valueLabelY, "text-anchor": "end", class: "chart-label" }, money(rows.at(-1).equity)));
  addBacktestChartHover(svg, rows, { width, height, left, right, top, bottom, xScale, yScale });
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
  const toSvgPoint = (event) => {
    const rect = svg.getBoundingClientRect();
    return {
      x: ((event.clientX - rect.left) / Math.max(rect.width, 1)) * chart.width,
      y: ((event.clientY - rect.top) / Math.max(rect.height, 1)) * chart.height,
    };
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
    const positionLines = row.positions.length
      ? row.positions.map(([symbol, value]) => `${symbol} : ${money(value)}`)
      : ["No positions"];
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
    const axisDate = formatDateTick(row.date, 4);
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

function percent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

//: Wheel and pinch fire continuously, so plan edits coalesce into one POST once the gesture
//: settles rather than one per tick.
const PLAN_SAVE_DEBOUNCE_MS = 500;

function schedulePlanSave() {
  window.clearTimeout(schedulePlanSave.timer);
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
  if (!state.dca?.plan) return;
  syncNodesToPlan();
  try {
    const [dcaPayload, controlsPayload] = await Promise.all([
      api("/api/dca", {
        method: "POST",
        body: JSON.stringify({ plan: state.dca.plan, account_id: state.dca?.account_id || "" }),
        timeoutMs: 5000,
      }),
      api("/api/controls", { method: "POST", body: JSON.stringify({ controls: state.controls }), timeoutMs: 5000 }),
    ]);
    state.dca = dcaPayload;
    state.controls = controlsPayload.controls;
    // The plan is an input to both DCA strategies, so their cached views are now stale.
    // This used to clear "none", which was where the DCA view lived before it was selectable.
    DCA_STRATEGY_KEYS.forEach((key) => {
      delete state.backtests[key];
      delete state.signals[key];
    });
    renderDca();
    if (!quiet) showToast("Saved");
  } catch (error) {
    showToast(error.message);
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
  if (armed.some((deployment) => normalizeBindingFrequency(deployment.frequency) !== "mcp")) return "live";
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
  const scheduled = armed.filter((binding) => normalizeBindingFrequency(binding.frequency) !== "mcp");
  const agentDriven = armed.filter((binding) => normalizeBindingFrequency(binding.frequency) === "mcp");
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
          ? `Scheduled: ${scheduled.map((binding) => `${strategyByKey(binding.strategy).name} every ${normalizeBindingFrequency(binding.frequency)}`).join(", ")}`
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
  closeNavOnMobile();
}

// -- algorithm page ----------------------------------------------------------------------

function normalizeBindingFrequency(value) {
  const candidate = String(value ?? "1hr").trim().toLowerCase();
  return ["15m", "30m", "1hr", "2hr", "1d", "mcp"].includes(candidate) ? candidate : "1hr";
}

function renderAlgorithmPage(content, strategyKey, tab) {
  const strategy = strategyByKey(strategyKey);
  const deployments = deploymentsFor(strategy.key);
  const deployment = deploymentFor(strategy.key);
  // Every account is offered to every algorithm. Sharing one account between algorithms is
  // allowed; it only costs attribution, which the overview says plainly when it happens.
  const options = accountRows();
  const frequencyOptions = ["15m", "30m", "1hr", "2hr", "1d", "mcp"];
  const savedFrequency = normalizeBindingFrequency(deployment?.frequency || "1hr");
  const actions = options.length
    ? `<div class="deployControl"${deployment ? ` data-binding="${escapeHtml(deployment.id)}"` : ""}>
         <button class="ctl powerButton${deployment?.enabled ? " on" : ""}" type="button" data-role="power"
           aria-pressed="${Boolean(deployment?.enabled)}" aria-label="Toggle trading"
           title="${deployment?.enabled ? "Pause" : "Start"} trading"><span aria-hidden="true">&#9211;</span></button>
         <select class="ctl" id="deployTargetSelect" aria-label="Account"${deployment?.enabled ? " disabled" : ""}>
           ${options.map((account) => `<option value="${escapeHtml(account.id)}"${account.id === deployment?.account_id ? " selected" : ""}>${escapeHtml(account.label)}</option>`).join("")}
         </select>
         <select class="ctl" id="deployFrequencySelect" aria-label="Frequency" data-binding="${escapeHtml(deployment?.id || "")}" ${!deployment ? "disabled" : ""}>
           ${frequencyOptions.map((value) => `<option value="${escapeHtml(value)}"${savedFrequency === value ? " selected" : ""}>${escapeHtml(value)}</option>`).join("")}
         </select>
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
      ${explainer.behavior ? `<p class="cardBody">${escapeHtml(explainer.behavior)}</p>` : ""}
    </section>`;
}

function renderTuneTab(body, strategy) {
  const isDca = DCA_ALGORITHM_KEYS.includes(strategy.key);
  ensureAlgorithmConfig(strategy.key);
  body.innerHTML = `
    ${explainerCard(strategy)}
    <section class="card tuneCard">
      <div class="cardHead">
        <h2>Configuration</h2>
        <span class="cardHint" id="tuneHint"></span>
      </div>
      <div class="tuneBody" id="tuneBody"></div>
      ${isDca ? "" : `<div class="cardActions"><button class="ctl" type="button" id="saveConfigButton">Save changes</button></div>`}
    </section>`;
  const host = $("#tuneBody");
  if (isDca) renderDcaTuner(host, strategy);
  else renderConfigForm(host, strategy);
}

function renderDcaTuner(host, strategy) {
  const hint = $("#tuneHint");
  const plan = state.algorithmConfigs[strategy.key]?.explainer?.parameters?.__plan__;
  if (hint) hint.textContent = `Dollars per month, per symbol · ${accountLabel(state.dca?.account_id)}`;
  host.innerHTML = `<svg class="bubbleBoard" id="bubbleBoard" role="img"
    aria-label="Interactive buy and sell budget bubbles"></svg>
    <p class="cardHint">${escapeHtml(plan?.effect || "")} Scroll a bubble to change its budget, drag between buckets, double-click to add. Saves automatically.</p>`;
  renderDca();
}

function renderConfigForm(host, strategy) {
  const entry = state.algorithmConfigs[strategy.key];
  const hint = $("#tuneHint");
  if (!entry) {
    host.innerHTML = `<p class="emptyState">Loading configuration.</p>`;
    ensureAlgorithmConfig(strategy.key);
    return;
  }
  if (hint) hint.textContent = `config/algorithms.yaml · ${entry.config_key || strategy.key}`;
  const fields = Object.entries(entry.config || {});
  const docs = entry.explainer?.parameters || {};
  host.innerHTML = fields.length
    ? `<div class="configForm">${fields.map(([key, value]) => renderConfigField(key, value, docs[key])).join("")}</div>`
    : `<p class="emptyState">This algorithm has no tunable parameters.</p>`;
}

function renderBacktestTab(body, strategy) {
  const backtest = state.backtests[strategy.key];
  const loading = Boolean(state.backtestLoading[strategy.key]);
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
      <section class="card">
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
  const payload = state.signals[strategy.key];
  const loading = Boolean(state.signalLoading[strategy.key]);
  // Already ordered strongest-first by ``signal_view_from_decision``.
  const leaders = payload?.leaders || [];
  body.innerHTML = `
    <section class="card">
      <div class="cardHead">
        <h2>Live signals</h2>
        <div class="cardHeadActions">
          <button class="ctl" type="button" id="refreshUniverseButton" ${state.universeRefreshing ? "disabled" : ""}>Refresh universe</button>
          <button class="ctl" type="button" id="refreshSignalsButton" ${loading ? "disabled" : ""}>${loading ? "Loading." : "Refresh"}</button>
        </div>
      </div>
      ${(payload?.summary || []).length ? `<div class="metricRow">
        ${payload.summary.map((item) => `<div class="metric"><span>${escapeHtml(item.label)}</span><strong>${escapeHtml(item.value)}</strong></div>`).join("")}
      </div>` : ""}
      <div class="signalRows" id="signalRows">
        ${state.universeProposal || state.universeRefreshing
          ? renderUniverseProposalRows()
          : loading
            ? `<p class="emptyState">Fetching live signal snapshot.</p>`
            : payload?.error
              ? `<p class="emptyState">${escapeHtml(payload.error)}</p>`
              : leaders.length
                ? leaders.map((row) => `
                  <article>
                    <strong>${escapeHtml(row.symbol)}</strong>
                    <span>${escapeHtml(formatSignalHeadline(strategy.key, row))}</span>
                    <span>${escapeHtml(formatSignalDetail(strategy.key, row))}</span>
                  </article>`).join("")
                : renderSignalFallbackRows(strategy, payload, (strategy.signals || []).slice(0, 5))}
      </div>
    </section>`;
  if (!payload && !loading) ensureSignals(strategy.key);
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
    state.algorithmConfigs[strategyKey] = await api("/api/algorithm-config", {
      method: "POST",
      body: JSON.stringify({ strategy: strategyKey, config: values }),
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

async function setDeploymentFrequency(bindingId, frequency) {
  const binding = bindingById(bindingId);
  if (!binding) return;
  binding.frequency = normalizeBindingFrequency(frequency);
  await saveBindings();
}

async function deployTo(strategyKey, accountId) {
  const used = new Set(bindings().map((binding) => String(binding.id)));
  let index = 1;
  while (used.has(`b${index}`)) index += 1;
  bindings().push({ id: `b${index}`, strategy: strategyKey, account_id: accountId, enabled: false, frequency: "1hr" });
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
      delete state.signals[route.id];
      return loadSignals(route.id);
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

  $("#content")?.addEventListener("change", (event) => {
    if (event.target.id === "deployTargetSelect") {
      setDeploymentAccount(currentRoute().id, event.target.value);
      return;
    }
    if (event.target.id === "deployFrequencySelect") {
      const bindingId = event.target.dataset.binding;
      if (bindingId) setDeploymentFrequency(bindingId, event.target.value);
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

  window.addEventListener("keydown", (event) => {
    const tag = event.target?.tagName?.toLowerCase();
    if (tag === "input" || tag === "textarea" || event.target?.isContentEditable) return;
    if ((event.key === "Delete" || event.key === "Backspace") && state.selected) {
      const found = BUCKET_NAMES.flatMap((bucketName) =>
        bucketItems(bucketName).map((item) => ({ bucketName, item })),
      ).find(({ item }) => item.symbol === state.selected);
      if (found) {
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
    // Plans are per account, so the plan to load is only knowable once controls are in.
    const dcaAccount = bindingById(primaryDcaBindingId())?.account_id || "";
    state.dca = await api(`/api/dca?account_id=${encodeURIComponent(dcaAccount)}`, { timeoutMs: 5000 });
    state.dca.plan.max_item_amount = MAX_AMOUNT;
  } catch (error) {
    showToast(`Could not load dashboard: ${error.message}`);
  }
  render();
}

init();
