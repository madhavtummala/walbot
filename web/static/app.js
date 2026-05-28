const SVG_NS = "http://www.w3.org/2000/svg";
const BUCKET_NAMES = ["buy", "sell"];
const MAX_AMOUNT = 50;
const WHEEL_STEP = 5;
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));
const BACKTEST_PERIOD = "6m";
const BACKTEST_LABEL = "6M";
const BACKTEST_STORAGE_KEY = "tradingBot.backtests.6m.v3";

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
    key: "momentum_social",
    name: "Momentum + Social",
    status: "Live",
    horizon: "Daily",
    risk: "Medium",
    logic: "Ranks the ETF universe every day, buys only names with positive medium-term momentum above the long moving average, then sizes positions by score, volatility, and portfolio caps.",
    signals: ["N-day return", "21-day return", "Long SMA trend", "Social attention", "Sentiment/vendor score", "Volume z-score", "Realized volatility"],
  },
  {
    key: "trend_following",
    name: "Trend Following",
    status: "Template",
    horizon: "Daily",
    risk: "Medium",
    logic: "Stays with persistent uptrends and steps out when price loses the major trend, avoiding counter-trend entries.",
    signals: ["Close vs 50/200 SMA", "Moving-average slope", "63-day return", "Trend gap", "Realized volatility"],
  },
  {
    key: "mean_reversion",
    name: "Mean Reversion",
    status: "Template",
    horizon: "Swing",
    risk: "High",
    logic: "Looks for short-term oversold pullbacks inside a larger uptrend, then exits as price reverts back toward its mean.",
    signals: ["RSI reset", "20-day z-score", "Bollinger position", "200-day trend filter", "ATR stop distance"],
  },
  {
    key: "breakout",
    name: "Breakout",
    status: "Template",
    horizon: "Swing",
    risk: "High",
    logic: "Targets fresh range highs when participation expands, with position size tied to stop distance and volatility.",
    signals: ["55-day high", "Volume expansion", "ATR compression", "Range width", "Trailing stop distance"],
  },
  {
    key: "risk_parity",
    name: "Risk Parity",
    status: "Template",
    horizon: "Portfolio",
    risk: "Lower",
    logic: "Balances portfolio risk so high-volatility assets receive smaller allocations and no single sleeve dominates drawdown.",
    signals: ["20/60-day volatility", "Inverse-vol weight", "Correlation stress", "Drawdown guard", "Exposure cap"],
  },
  {
    key: "dual_momentum",
    name: "Dual Momentum",
    status: "Template",
    horizon: "Monthly",
    risk: "Medium",
    logic: "Rotates into the strongest assets only when they also clear an absolute momentum hurdle; otherwise it moves toward cash-like exposure.",
    signals: ["6-month return", "12-month return", "Relative rank", "Absolute momentum hurdle", "Volatility filter"],
  },
  {
    key: "defensive_momentum",
    name: "Defensive Momentum",
    status: "Live",
    horizon: "Intraday",
    risk: "Medium",
    logic: "Ranks the curated ETF universe with 30-minute momentum, daily absolute trend, and recent sentiment; regime rules rotate between risk-on and defensive sleeves.",
    signals: ["30-minute momentum", "Daily trend", "News sentiment", "Market regime", "Defensive sleeve"],
  },
];

const OPTIONS_STRATEGIES = [
  {
    key: "options_swing_dual_momentum",
    name: "Swing Dual Momentum",
    risk: "Directional",
    description: "Uses dual-momentum ETF signals to buy 30-60 DTE calls for bullish underlyings or puts for bearish underlyings, with premium and liquidity caps.",
    config: ["Dual momentum underlyings", "Long calls/puts only", "30-60 DTE", "0.35-0.65 delta", "Premium capped"],
  },
  {
    key: "covered_call",
    name: "Covered Call",
    risk: "Income",
    description: "Sell calls against owned shares to collect premium, accepting capped upside.",
    config: ["Underlying position", "30-45 DTE", "0.20-0.35 delta short call", "Roll rule before assignment"],
  },
  {
    key: "cash_secured_put",
    name: "Cash-Secured Put",
    risk: "Entry",
    description: "Sell puts only when willing to buy the underlying at the strike.",
    config: ["Cash reserved for assignment", "20-45 DTE", "0.15-0.30 delta put", "Max allocation per symbol"],
  },
  {
    key: "protective_put",
    name: "Protective Put",
    risk: "Hedge",
    description: "Buy puts to define downside risk on an equity or ETF holding.",
    config: ["Hedge ratio", "Expiry matching risk window", "Acceptable premium budget", "Strike near loss limit"],
  },
  {
    key: "put_spread",
    name: "Defined-Risk Put Spread",
    risk: "Directional",
    description: "Use a vertical spread to express bullish or bearish views with capped loss.",
    config: ["Long and short strikes", "Debit/credit limit", "DTE", "Profit-taking and stop rules"],
  },
  {
    key: "iron_condor",
    name: "Iron Condor",
    risk: "Neutral",
    description: "Sell an out-of-the-money call spread and put spread when expecting range-bound movement.",
    config: ["IV rank threshold", "Short strike deltas", "Wing width", "Exit at profit target or tested side"],
  },
];

const NONE_ALGORITHM = {
  key: "none",
  name: "None",
  status: "Idle",
  horizon: BACKTEST_LABEL,
  risk: "Flat",
  logic: "Disables algorithmic trading and leaves the right-side snapshot focused on DCA activity or a flat cached baseline.",
  signals: ["DCA schedule", "Planned buys", "Flat exposure"],
};

const NONE_OPTIONS = {
  key: "none",
  name: "None",
  risk: "Idle",
  description: "No options strategy is active.",
  config: ["Options trading disabled"],
};

const state = {
  status: null,
  universe: [],
  controls: {
    trading_account_id: "",
    algorithm_enabled: false,
    options_trading_enabled: false,
    active_strategy: "momentum_social",
    options_strategy: "none",
    options_trading_account_id: "",
  },
  accounts: [],
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
  renderedAlgorithmDeckKey: "",
  renderedOptionsDeckKey: "",
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
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `Request failed: ${response.status}`);
    return payload;
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
function isDcaEnabled() {
  return Boolean(state.dca?.plan?.enabled);
}

function isAlgorithmTradingEnabled() {
  return Boolean(state.controls?.algorithm_enabled) && activeAlgorithmKey() !== "none";
}

function isOptionsTradingEnabled() {
  return Boolean(state.controls?.options_trading_enabled) && activeOptionsKey() !== "none";
}

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
  return 18 + Math.sqrt(clamp(amount, 0, MAX_AMOUNT)) * 3.45;
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
  const width = svg.clientWidth || 1200;
  const height = svg.clientHeight || 720;
  const stacked = width < 760;
  const baseR = stacked ? Math.min(width * 0.35, height * 0.19, 210) : Math.min(width * 0.2, height * 0.36, 292);
  const maxR = stacked ? Math.min(width * 0.43, height * 0.24, 260) : Math.min(width * 0.27, height * 0.44, 365);

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

function pointToPosition(point, bucket, radius) {
  const usable = Math.max(bucket.r - radius - 18, 1);
  return {
    x: clamp((point.x - bucket.cx) / usable, -1, 1),
    y: clamp((point.y - bucket.cy) / usable, -1, 1),
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
  if (!state.dca?.plan) return;
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
  showToast(`${row.symbol} added`);
}

function removeSymbol(bucketName, symbol) {
  setBucketItems(
    bucketName,
    bucketItems(bucketName).filter((item) => item.symbol !== symbol),
  );
  renderDca();
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
}

function renderControls() {
  $("#cronPattern").value = state.dca?.plan?.schedule_pattern || "0 12 * * 1-5";
  renderScheduleDescription();
  renderAlgorithmPower();
  renderAlgorithmDeck();
  renderOptionsDeck();
  updateFeatureToggles();
}

function renderDca() {
  if (!state.dca?.plan) return;
  syncNodesToPlan();
  renderControls();
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
  return [NONE_ALGORITHM, ...STRATEGIES];
}

function optionsChoices() {
  return [NONE_OPTIONS, ...OPTIONS_STRATEGIES];
}

function strategyByKey(strategyKey) {
  return algorithmChoices().find((choice) => choice.key === strategyKey) || NONE_ALGORITHM;
}

function optionsUnderlyingStrategyKey(optionsKey = activeOptionsKey()) {
  if (optionsKey === "options_swing_dual_momentum") return "dual_momentum";
  return "none";
}

function optionsSignalChoice() {
  const selected = optionsChoices().find((choice) => choice.key === activeOptionsKey()) || NONE_OPTIONS;
  const underlyingKey = optionsUnderlyingStrategyKey(selected.key);
  const underlying = strategyByKey(underlyingKey);
  if (underlyingKey === "none") return selected;
  return {
    ...underlying,
    name: selected.name,
    logic: selected.description,
    signals: underlying.signals || selected.config || [],
  };
}

function activeAlgorithmKey() {
  const key = state.controls?.active_strategy || "none";
  return algorithmChoices().some((choice) => choice.key === key) ? key : "none";
}

function activeOptionsKey() {
  const key = state.controls?.options_strategy || "none";
  return optionsChoices().some((choice) => choice.key === key) ? key : "none";
}

function deckClass(offset) {
  if (offset === 0) return "active";
  if (offset === -1) return "before";
  if (offset === 1) return "after";
  if (offset === 2) return "farAfter";
  if (offset === -2) return "farBefore";
  return "hidden";
}

function toneClass(index) {
  return `tone-${index % 5}`;
}

function isBacktestPayload(payload) {
  return Boolean(payload && typeof payload === "object" && Array.isArray(payload.rows));
}

function loadStoredBacktests() {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(BACKTEST_STORAGE_KEY) || "{}");
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return {};
    return Object.fromEntries(
      Object.entries(parsed).filter(([, payload]) => isBacktestPayload(payload)),
    );
  } catch (_error) {
    return {};
  }
}

function enabledUniverseSymbols() {
  return state.universe
    .filter((row) => row.enabled !== false)
    .map((row) => row.symbol)
    .filter(Boolean);
}

function universeSignature() {
  return enabledUniverseSymbols().join(",");
}

function backtestCacheKey(strategyKey) {
  return `${strategyKey}:${universeSignature()}`;
}

function hydrateStoredBacktests() {
  const stored = loadStoredBacktests();
  const hydrated = {};
  for (const strategy of algorithmChoices()) {
    const payload = stored[backtestCacheKey(strategy.key)];
    if (payload) hydrated[strategy.key] = payload;
  }
  state.backtests = hydrated;
}

function storedBacktest(strategyKey) {
  return loadStoredBacktests()[backtestCacheKey(strategyKey)] || null;
}

function storeBacktest(strategyKey, payload) {
  if (!isBacktestPayload(payload)) return;
  try {
    const cache = loadStoredBacktests();
    cache[backtestCacheKey(strategyKey)] = {
      ...payload,
      browser_cached_at: new Date().toISOString(),
    };
    window.localStorage.setItem(BACKTEST_STORAGE_KEY, JSON.stringify(cache));
  } catch (_error) {
    // Browser storage is best-effort; the API cache still handles server-side reuse.
  }
}

function renderAlgorithmDeck() {
  const deck = $("#algorithmDeck");
  if (!deck) return;
  const choices = algorithmChoices();
  const activeKey = activeAlgorithmKey();
  const activeIndex = Math.max(0, choices.findIndex((choice) => choice.key === activeKey));
  const renderKey = `${activeKey}:${choices.length}`;
  deck.classList.toggle("locked", isAlgorithmTradingEnabled());
  deck.setAttribute("aria-disabled", String(isAlgorithmTradingEnabled()));
  if (state.renderedAlgorithmDeckKey === renderKey && deck.children.length) {
    renderAlgorithmSignals();
    return;
  }
  state.renderedAlgorithmDeckKey = renderKey;
  deck.innerHTML = choices.map((strategy, index) => {
    const offset = index - activeIndex;
    const signalPreview = (strategy.signals || []).slice(0, 5);
    const extraSignalCount = Math.max(0, (strategy.signals || []).length - signalPreview.length);
    return `
      <article class="deckCard ${deckClass(offset)} ${toneClass(index)}" data-strategy="${escapeHtml(strategy.key)}" aria-current="${offset === 0 ? "true" : "false"}">
        <header class="deckHeader">
          <div>
            <span class="deckKicker">${offset === 0 ? "Selected" : "Available"}</span>
            <h2>${escapeHtml(strategy.name)}</h2>
          </div>
        </header>
        <section class="strategyLogic" aria-label="${escapeHtml(strategy.name)} logic">
          <span>Logic</span>
          <p>${escapeHtml(strategy.logic)}</p>
        </section>
        <section class="strategySignals" aria-label="${escapeHtml(strategy.name)} signals">
          <span>Signals</span>
          <ul>
            ${signalPreview.map((signal) => `<li>${escapeHtml(signal)}</li>`).join("")}
            ${extraSignalCount ? `<li>+${extraSignalCount} more</li>` : ""}
          </ul>
        </section>
        <div class="strategyMeta">
          <span>${escapeHtml(strategy.status)}</span>
          <span>${escapeHtml(strategy.horizon)}</span>
          <span>${escapeHtml(strategy.risk)}</span>
        </div>
      </article>
    `;
  }).join("");
  renderAlgorithmSignals();
}

async function loadBacktest(strategyKey, refresh, options = {}) {
  const cacheOnly = Boolean(options.cacheOnly);
  if (!cacheOnly) state.backtestLoading[strategyKey] = true;
  if (!cacheOnly && state.backtests[strategyKey]?.error) delete state.backtests[strategyKey];
  if (!cacheOnly) renderAlgorithmDeck();
  try {
    const payload = await api("/api/backtest", {
      method: "POST",
      body: JSON.stringify({ strategy: strategyKey, period: BACKTEST_PERIOD, refresh, cache_only: cacheOnly }),
      timeoutMs: 60000,
    });
    if (isBacktestPayload(payload)) {
      state.backtests[strategyKey] = payload;
      storeBacktest(strategyKey, payload);
    } else if (payload?.error) {
      const fallback = storedBacktest(strategyKey);
      if (fallback) {
        state.backtests[strategyKey] = { ...fallback, cached: true, offline_error: payload.error };
      } else if (!cacheOnly) {
        state.backtests[strategyKey] = { error: payload.error };
      }
    }
  } catch (error) {
    const fallback = storedBacktest(strategyKey);
    if (fallback) {
      state.backtests[strategyKey] = { ...fallback, cached: true, offline_error: error.message };
    } else if (!cacheOnly) {
      state.backtests[strategyKey] = { error: error.message };
    }
  } finally {
    if (!cacheOnly) state.backtestLoading[strategyKey] = false;
    if (!cacheOnly) renderAlgorithmDeck();
    renderAlgorithmSignals();
  }
}

async function recommendUniverse() {
  state.universeRefreshing = true;
  renderAlgorithmSignals();
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
    renderAlgorithmSignals();
  }
}

async function applyUniverseProposal() {
  const rows = state.universeProposal?.rows || [];
  if (!rows.length || state.universeApplying) return;
  state.universeApplying = true;
  renderAlgorithmSignals();
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
    const dcaPayload = await api("/api/dca", { timeoutMs: 5000 });
    state.dca = dcaPayload;
    state.dca.plan.max_item_amount = MAX_AMOUNT;
    hydrateStoredBacktests();
    renderDca();
    renderAlgorithmDeck();
    renderOptionsSignals();
    loadSignals(activeAlgorithmKey());
    const optionsSignalKey = optionsUnderlyingStrategyKey();
    if (optionsSignalKey !== "none" && optionsSignalKey !== activeAlgorithmKey()) loadSignals(optionsSignalKey);
    loadCachedBacktestsForDeck();
    showToast("Universe applied");
  } catch (error) {
    showToast(error.message);
  } finally {
    state.universeApplying = false;
    renderAlgorithmSignals();
  }
}

async function loadCachedBacktestsForDeck() {
  const keys = [...new Set([...algorithmChoices().map((strategy) => strategy.key), optionsUnderlyingStrategyKey()].filter(Boolean))];
  for (const strategyKey of keys) {
    if (!state.backtests[strategyKey] && !state.backtestLoading[strategyKey]) {
      await loadBacktest(strategyKey, false, { cacheOnly: true });
    }
  }
}

async function ensureSignals(strategyKey) {
  if (state.signals[strategyKey] || state.signalLoading[strategyKey]) return;
  await loadSignals(strategyKey);
}

async function loadSignals(strategyKey) {
  state.signalLoading[strategyKey] = true;
  renderAlgorithmSignals();
  renderOptionsSignals();
  try {
    state.signals[strategyKey] = await api(`/api/strategy-signals?strategy=${encodeURIComponent(strategyKey)}`, {
      timeoutMs: 60000,
    });
  } catch (error) {
    state.signals[strategyKey] = { error: error.message };
  } finally {
    state.signalLoading[strategyKey] = false;
    renderAlgorithmSignals();
    renderOptionsSignals();
  }
}

function formatSignalDetail(strategyKey, row) {
  if (strategyKey === "defensive_momentum") {
    const providers = Array.isArray(row.sentiment_providers) && row.sentiment_providers.length
      ? row.sentiment_providers.join(", ")
      : "none";
    const close = row.close ? ` / Close ${money(row.close, 2)}` : "";
    const reason = row.reason ? `${row.reason} / ` : "";
    return `${reason}Sentiment ${num(row.sentiment_score ?? row.social_score, 2)} (${providers}, ${Number(row.sentiment_records || 0)} recs) / Regime ${row.regime || "pending"}${close}`;
  }
  if (row.reason) {
    const close = row.close ? ` / Close ${money(row.close, 2)}` : "";
    return `${row.reason}${close}`;
  }
  if (strategyKey === "momentum_social") {
    return `Price ${num(row.price_score, 2)} / Social ${num(row.social_score, 2)} / Volume ${num(row.volume_score, 2)}`;
  }
  if (strategyKey === "none") {
    return `Weight ${percent(row.target_weight)} / ${row.trend_ok ? "Enabled" : "Inactive"}`;
  }
  if (row.close) {
    return `Close ${money(row.close, 2)}`;
  }
  return "Live feed pending";
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

function renderUniverseReview(label = "Algorithm universe") {
  const proposal = state.universeProposal;
  const currentSymbols = enabledUniverseSymbols();
  const rows = proposal?.rows || [];
  const rejectedCount = Number(proposal?.rejected?.length || 0);
  const status = state.universeRefreshing
    ? "Refreshing"
    : proposal
      ? `${rows.length}/${proposal.eligible_count || rows.length}`
      : `${currentSymbols.length} active`;
  return `
    <section class="universeReview" aria-label="${escapeHtml(label)}">
      <div class="cachedBacktestHeader">
        <span>Universe</span>
        <strong>${escapeHtml(status)}</strong>
      </div>
      ${proposal
      ? `
        <div class="universeRows">
          ${rows.map((row) => `
            <article>
              <strong>${escapeHtml(row.symbol)}</strong>
              <span>${escapeHtml(row.bucket || "")}</span>
              <span>${escapeHtml(row.latest_bar || "")} / ${money(row.avg_dollar_volume, 0)}</span>
            </article>
          `).join("") || "<p>No proposal</p>"}
        </div>
        <div class="universeMeta">
          <span>${escapeHtml(proposal.data_feed || "feed")}</span>
          <span>${rejectedCount} filtered</span>
        </div>
        <button class="applyUniverse" type="button" data-apply-universe ${state.universeApplying || !rows.length ? 'disabled aria-busy="true"' : ""}>Apply</button>
      `
      : `
        <div class="universeChips">
          ${currentSymbols.map((symbol) => `<span>${escapeHtml(symbol)}</span>`).join("")}
        </div>
      `}
    </section>
  `;
}

function renderSignalBacktestCard({
  card,
  strategyKey,
  selected,
  kicker = "Position Signals",
  backtestChartId = "activeBacktestChart",
  universeLabel = "Algorithm universe",
}) {
  if (!card) return;
  const payload = state.signals[strategyKey];
  const backtest = state.backtests[strategyKey];
  const loading = Boolean(state.signalLoading[strategyKey]);
  const backtestLoading = Boolean(state.backtestLoading[strategyKey]);
  const signalInputs = (selected.signals || []).slice(0, 5);
  const backtestCaption = backtestCaptionText(backtest, backtestLoading);
  const equityCaption = backtestEquityText(backtest);
  const orderCaption = backtestOrderText(backtest);
  const refreshAttrs = backtestLoading ? 'disabled aria-busy="true"' : "";
  const universeAttrs = state.universeRefreshing ? 'disabled aria-busy="true"' : "";
  const allLeaders = payload?.leaders || [];
  const activeLeaders = allLeaders.filter((row) => (row.side === "LONG" || row.side === "SHORT" || row.signal === "LONG" || row.signal === "SHORT"));
  const inactiveLeaders = allLeaders.filter((row) => !activeLeaders.includes(row));
  const visibleLeaders = [...activeLeaders, ...inactiveLeaders];
  card.innerHTML = `
    <header class="signalHeader">
      <div>
        <span class="deckKicker">${escapeHtml(kicker)}</span>
      </div>
      <div class="signalActions">
        <button class="refreshUniverse" type="button" data-refresh-universe ${universeAttrs}>Refresh Universe</button>
        <button class="refreshBacktest" type="button" data-refresh-backtest="${escapeHtml(strategyKey)}" ${refreshAttrs}>Backtest</button>
      </div>
    </header>
    <div class="signalBody">
      <div class="signalRows">
        ${loading
      ? "<p>Fetching live signal snapshot.</p>"
      : payload?.error
        ? `<p>${escapeHtml(payload.error)}</p>`
        : visibleLeaders.length
          ? visibleLeaders.map((row) => `
            <article>
              <strong>${escapeHtml(row.symbol)}</strong>
              <span>${escapeHtml(row.side || row.signal)} / Score ${num(row.score, 2)} / Weight ${percent(row.target_weight)}</span>
              <span>${escapeHtml(formatSignalDetail(strategyKey, row))}</span>
            </article>
          `).join("")
          : renderSignalFallbackRows(selected, payload, signalInputs)
    }
      </div>
      <section class="cachedBacktest" aria-label="Cached ${BACKTEST_LABEL} backtest">
        <div class="cachedBacktestHeader">
          <span>${BACKTEST_LABEL} Backtest</span>
          <strong>${escapeHtml(backtestStatusLabel(backtest, backtestLoading))}</strong>
        </div>
        <svg class="deckChart" id="${escapeHtml(backtestChartId)}" role="img" aria-label="${BACKTEST_LABEL} backtest chart"></svg>
        ${backtestCaption ? `<p>${escapeHtml(backtestCaption)}</p>` : ""}
        ${equityCaption ? `<p>${escapeHtml(equityCaption)}</p>` : ""}
        ${orderCaption ? `<p>${escapeHtml(orderCaption)}</p>` : ""}
        ${backtest?.offline_error ? `<p>${escapeHtml(backtest.offline_error)}</p>` : ""}
      </section>
      ${renderUniverseReview(universeLabel)}
    </div>
  `;
  renderBacktestChart(backtest, $(`#${backtestChartId}`));
}

function renderAlgorithmSignals() {
  renderSignalBacktestCard({
    card: $("#algorithmSignalCard"),
    strategyKey: activeAlgorithmKey(),
    selected: strategyByKey(activeAlgorithmKey()),
    kicker: "Position Signals",
    backtestChartId: "activeBacktestChart",
    universeLabel: "Algorithm universe",
  });
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

async function selectAlgorithmStrategy(strategyKey) {
  if (isAlgorithmTradingEnabled()) {
    showToast("Turn algorithm off before changing strategies");
    return;
  }
  if (strategyKey === activeAlgorithmKey()) return;
  state.controls.active_strategy = strategyKey;
  if (strategyKey === "none") state.controls.algorithm_enabled = false;
  renderAlgorithmDeck();
  renderAlgorithmPower();
  const savePromise = saveControlsOnly({ renderDecks: false });
  await Promise.all([savePromise, ensureSignals(strategyKey)]);
}

function canShiftDeck() {
  if (state.deckWheelLocked) return false;
  state.deckWheelLocked = true;
  window.setTimeout(() => {
    state.deckWheelLocked = false;
  }, 760);
  return true;
}

function shiftAlgorithmDeck(direction) {
  if (isAlgorithmTradingEnabled()) {
    showToast("Turn algorithm off before changing strategies");
    return;
  }
  const choices = algorithmChoices();
  const index = Math.max(0, choices.findIndex((choice) => choice.key === activeAlgorithmKey()));
  const nextIndex = clamp(index + direction, 0, choices.length - 1);
  if (nextIndex === index) return; // at ends, no movement or animation
  if (!canShiftDeck()) return;
  const deck = $("#algorithmDeck");
  if (deck) deck.setAttribute("data-shift", direction > 0 ? "down" : "up");
  selectRelativeAlgorithm(direction);
  if (deck) window.setTimeout(() => deck.removeAttribute("data-shift"), 760);
}

function selectRelativeAlgorithm(direction) {
  if (isAlgorithmTradingEnabled()) return;
  const choices = algorithmChoices();
  const index = Math.max(0, choices.findIndex((choice) => choice.key === activeAlgorithmKey()));
  const nextIndex = clamp(index + direction, 0, choices.length - 1);
  const next = choices[nextIndex];
  if (next && next.key !== activeAlgorithmKey()) selectAlgorithmStrategy(next.key);
}

function renderOptionsDeck() {
  const deck = $("#optionsDeck");
  if (!deck) return;
  const choices = optionsChoices();
  const activeKey = activeOptionsKey();
  const activeIndex = Math.max(0, choices.findIndex((choice) => choice.key === activeKey));
  const renderKey = `${activeKey}:${choices.length}`;
  deck.classList.toggle("locked", isOptionsTradingEnabled());
  deck.setAttribute("aria-disabled", String(isOptionsTradingEnabled()));
  if (state.renderedOptionsDeckKey === renderKey && deck.children.length) {
    renderOptionsInsight();
    return;
  }
  state.renderedOptionsDeckKey = renderKey;
  deck.innerHTML = choices.map((strategy, index) => {
    const offset = index - activeIndex;
    return `
      <article class="deckCard optionsDeckCard ${deckClass(offset)} ${toneClass(index + 2)}" data-options-strategy="${escapeHtml(strategy.key)}" aria-current="${offset === 0 ? "true" : "false"}">
        <header class="deckHeader">
          <div>
            <span class="deckKicker">${offset === 0 ? "Active" : "Available"}</span>
            <h2>${escapeHtml(strategy.name)}</h2>
          </div>
        </header>
        <p>${escapeHtml(strategy.description)}</p>
        <div class="strategyMeta">
          <span>${escapeHtml(strategy.risk)}</span>
          ${strategy.config.map((item) => `<span>${escapeHtml(item)}</span>`).join("")}
        </div>
      </article>
    `;
  }).join("");
  renderOptionsInsight();
}

async function selectOptionsStrategy(strategyKey) {
  if (isOptionsTradingEnabled()) {
    showToast("Turn options bot off before changing strategies");
    return;
  }
  if (strategyKey === activeOptionsKey()) return;
  state.controls.options_strategy = strategyKey;
  if (strategyKey === "none") state.controls.options_trading_enabled = false;
  renderOptionsDeck();
  renderOptionsPower();
  const signalKey = optionsUnderlyingStrategyKey(strategyKey);
  const savePromise = saveControlsOnly({ renderDecks: false });
  await Promise.all([savePromise, signalKey === "none" ? Promise.resolve() : ensureSignals(signalKey)]);
}

function shiftOptionsDeck(direction) {
  if (isOptionsTradingEnabled()) {
    showToast("Turn options bot off before changing strategies");
    return;
  }
  const choices = optionsChoices();
  const index = Math.max(0, choices.findIndex((choice) => choice.key === activeOptionsKey()));
  const nextIndex = clamp(index + direction, 0, choices.length - 1);
  if (nextIndex === index) return; // at ends, no movement or animation
  if (!canShiftDeck()) return;
  const deck = $("#optionsDeck");
  if (deck) deck.setAttribute("data-shift", direction > 0 ? "down" : "up");
  selectRelativeOptions(direction);
  if (deck) window.setTimeout(() => deck.removeAttribute("data-shift"), 760);
}

function selectRelativeOptions(direction) {
  if (isOptionsTradingEnabled()) return;
  const choices = optionsChoices();
  const index = Math.max(0, choices.findIndex((choice) => choice.key === activeOptionsKey()));
  const nextIndex = clamp(index + direction, 0, choices.length - 1);
  const next = choices[nextIndex];
  if (next && next.key !== activeOptionsKey()) selectOptionsStrategy(next.key);
}

function renderOptionsInsight() {
  const card = $("#optionsSignalCard");
  const strategyKey = optionsUnderlyingStrategyKey();
  if (strategyKey === "none") {
    if (!card) return;
    const selected = optionsChoices().find((choice) => choice.key === activeOptionsKey()) || NONE_OPTIONS;
    card.innerHTML = `
      <span class="deckKicker">Options Setup</span>
      <h2>${escapeHtml(selected.name)}</h2>
      <p>${escapeHtml(selected.description)}</p>
      <div class="signalRows">
        ${(selected.config || []).map((item) => `
          <article>
            <strong>${escapeHtml(item)}</strong>
            <span>${selected.key === "none" ? "Inactive" : "Review before enabling"}</span>
          </article>
        `).join("")}
      </div>
    `;
    return;
  }
  renderSignalBacktestCard({
    card,
    strategyKey,
    selected: optionsSignalChoice(),
    kicker: "Underlying Signals",
    backtestChartId: "optionsBacktestChart",
    universeLabel: "Options underlying universe",
  });
}

const renderOptionsSignals = renderOptionsInsight;

function renderOptionsPower() {
  const enabled = isOptionsTradingEnabled();
  const button = $("#optionsPowerToggle");
  const panel = $("#optionsRunPanel");
  const deck = $("#optionsDeck");
  const select = $("#optionsTradingAccount");
  if (button) {
    button.setAttribute("aria-pressed", String(enabled));
    button.classList.toggle("on", enabled);
    button.disabled = activeOptionsKey() === "none";
    button.innerHTML = '<span aria-hidden="true">&#9211;</span>';
  }
  if (panel) panel.classList.toggle("is-on", enabled);
  if (deck) {
    deck.classList.toggle("locked", enabled);
    deck.setAttribute("aria-disabled", String(enabled));
  }
  document.querySelectorAll('[data-deck="options"]').forEach((button) => {
    button.disabled = enabled;
  });
  if (select) {
    const accounts = state.accounts.length
      ? state.accounts
      : [{ id: state.controls?.options_trading_account_id || state.controls?.trading_account_id || "default", label: "Default" }];
    const activeAccount = state.controls?.options_trading_account_id || state.controls?.trading_account_id || accounts[0]?.id || "";
    select.innerHTML = accounts.map((account) => `
      <option value="${escapeHtml(account.id)}"${account.id === activeAccount ? " selected" : ""}>
        ${escapeHtml(account.label || account.id)}
      </option>
    `).join("");
    select.disabled = enabled;
  }
}

function updateFeatureToggles() {
  const enabled = isDcaEnabled();
  const button = $("#scheduleToggle");
  const panel = $("#schedulePanel");
  if (button) {
    button.setAttribute("aria-pressed", String(enabled));
    button.classList.toggle("on", enabled);
    button.innerHTML = '<span aria-hidden="true">&#9211;</span>';
  }
  if (panel) panel.classList.toggle("is-on", enabled);
}

function renderAlgorithmPower() {
  const enabled = isAlgorithmTradingEnabled();
  const button = $("#algorithmPowerToggle");
  const panel = $("#algorithmRunPanel");
  const select = $("#tradingAccount");
  const deck = $("#algorithmDeck");
  if (button) {
    button.setAttribute("aria-pressed", String(enabled));
    button.classList.toggle("on", enabled);
    button.disabled = activeAlgorithmKey() === "none";
    button.innerHTML = '<span aria-hidden="true">&#9211;</span>';
  }
  if (panel) panel.classList.toggle("is-on", enabled);
  if (deck) {
    deck.classList.toggle("locked", enabled);
    deck.setAttribute("aria-disabled", String(enabled));
  }
  document.querySelectorAll('[data-deck="algorithm"]').forEach((button) => {
    button.disabled = enabled;
  });
  if (select) {
    const accounts = state.accounts.length
      ? state.accounts
      : [{ id: state.controls?.trading_account_id || "default", label: "Default" }];
    const activeAccount = state.controls?.trading_account_id || accounts[0]?.id || "";
    select.innerHTML = accounts.map((account) => `
      <option value="${escapeHtml(account.id)}"${account.id === activeAccount ? " selected" : ""}>
        ${escapeHtml(account.label || account.id)}
      </option>
    `).join("");
    select.disabled = enabled;
  }
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
  const box = svgEl("rect", { class: "chart-tooltip-bg", rx: 6, ry: 6, width: 154, height: 72 });
  const dateText = textEl({ class: "chart-tooltip-title" }, "");
  const equityText = textEl({ class: "chart-tooltip-text" }, "");
  const investedText = textEl({ class: "chart-tooltip-text" }, "");
  const cashText = textEl({ class: "chart-tooltip-text" }, "");
  [vertical, horizontal, point, box, dateText, equityText, investedText, cashText].forEach((node) => {
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
    const tooltipWidth = 154;
    const tooltipHeight = 72;
    const tooltipX = x + tooltipWidth + 12 > chart.width ? x - tooltipWidth - 10 : x + 10;
    const tooltipY = Math.min(Math.max(y - tooltipHeight / 2, chart.top), chart.height - chart.bottom - tooltipHeight);
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
    dateText.textContent = row.date.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
    equityText.textContent = `Equity ${money(row.equity)}`;
    investedText.textContent = `Invested ${money(row.invested)}`;
    cashText.textContent = `Cash ${money(row.cash)}`;
    positionText(dateText, tooltipX + 10, tooltipY + 17);
    positionText(equityText, tooltipX + 10, tooltipY + 34);
    positionText(investedText, tooltipX + 10, tooltipY + 50);
    positionText(cashText, tooltipX + 10, tooltipY + 66);
    layer.setAttribute("visibility", "visible");
  });
  overlay.addEventListener("pointerleave", () => {
    layer.setAttribute("visibility", "hidden");
  });
}

function activateAlgorithmView() {
  renderAlgorithmDeck();
  const activeKey = activeAlgorithmKey();
  ensureSignals(activeKey);
}

function percent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function renderScheduleDescription() {
  const pattern = $("#cronPattern").value.trim();
  const descriptions = {
    "0 12 * * 1-5": "Weekdays at 12:00 PM",
    "0 9 * * 1-5": "Weekdays at 9:00 AM",
    "0 12 * * 1": "Mondays at 12:00 PM",
    "0 12 1 * *": "First day monthly at 12:00 PM",
  };
  $("#cronDescription").textContent = descriptions[pattern] || "Cron-style schedule pattern";
}

function switchTab(tabName) {
  document.querySelectorAll(".tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tabName);
  });
  document.querySelectorAll(".tabPanel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `${tabName}Tab`);
  });
  if (tabName === "dca") renderDca();
  if (tabName === "algorithm") activateAlgorithmView();
  if (tabName === "options") {
    renderOptionsDeck();
    renderOptionsPower();
    const signalKey = optionsUnderlyingStrategyKey();
    if (signalKey !== "none") ensureSignals(signalKey);
  }
}

async function savePlan(quiet = true) {
  if (!state.dca?.plan) return;
  syncNodesToPlan();
  state.dca.plan.schedule_pattern = $("#cronPattern").value.trim() || "0 12 * * 1-5";
  try {
    const [dcaPayload, controlsPayload] = await Promise.all([
      api("/api/dca", { method: "POST", body: JSON.stringify({ plan: state.dca.plan }), timeoutMs: 5000 }),
      api("/api/controls", { method: "POST", body: JSON.stringify({ controls: state.controls }), timeoutMs: 5000 }),
    ]);
    state.dca = dcaPayload;
    state.controls = controlsPayload.controls;
    delete state.backtests.none;
    delete state.signals.none;
    renderDca();
    if (!quiet) showToast("Saved");
  } catch (error) {
    showToast(error.message);
  }
}

async function saveControlsOnly(options = {}) {
  const renderDecks = options.renderDecks !== false;
  try {
    const payload = await api("/api/controls", {
      method: "POST",
      body: JSON.stringify({ controls: state.controls }),
      timeoutMs: 5000,
    });
    state.controls = payload.controls;
    state.accounts = payload.accounts || state.accounts;
    state.bot = payload.bot || state.bot;
    updateFeatureToggles();
    renderAlgorithmPower();
    renderOptionsPower();
    if (renderDecks) {
      renderAlgorithmDeck();
      renderOptionsDeck();
    } else {
      renderAlgorithmSignals();
      renderOptionsInsight();
    }
  } catch (error) {
    showToast(error.message);
  }
}

function wireEvents() {
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => switchTab(button.dataset.tab));
  });
  $("#algorithmDeck")?.addEventListener("click", (event) => {
    const card = event.target.closest("[data-strategy]");
    if (card) selectAlgorithmStrategy(card.dataset.strategy);
  });
  $("#algorithmSignalCard")?.addEventListener("click", handleSignalCardClick);
  $("#optionsSignalCard")?.addEventListener("click", handleSignalCardClick);
  $("#algorithmDeck")?.addEventListener("wheel", (event) => {
    event.preventDefault();
    shiftAlgorithmDeck(event.deltaY > 0 ? 1 : -1);
  }, { passive: false });
  $("#algorithmDeck")?.addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown") selectRelativeAlgorithm(1);
    if (event.key === "ArrowUp") selectRelativeAlgorithm(-1);
  });
  $("#optionsDeck")?.addEventListener("click", (event) => {
    const card = event.target.closest("[data-options-strategy]");
    if (card) selectOptionsStrategy(card.dataset.optionsStrategy);
  });
  $("#optionsDeck")?.addEventListener("wheel", (event) => {
    event.preventDefault();
    shiftOptionsDeck(event.deltaY > 0 ? 1 : -1);
  }, { passive: false });
  $("#optionsDeck")?.addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown") selectRelativeOptions(1);
    if (event.key === "ArrowUp") selectRelativeOptions(-1);
  });
  $("#optionsPowerToggle")?.addEventListener("click", () => {
    if (activeOptionsKey() === "none") {
      state.controls.options_trading_enabled = false;
      renderOptionsPower();
      return;
    }
    state.controls.options_trading_enabled = !state.controls.options_trading_enabled;
    renderOptionsPower();
    saveControlsOnly({ renderDecks: false });
  });
  $("#optionsTradingAccount")?.addEventListener("change", (event) => {
    state.controls.options_trading_account_id = event.target.value;
    renderOptionsPower();
    saveControlsOnly({ renderDecks: false });
  });
  $("#scheduleToggle")?.addEventListener("click", () => {
    if (!state.dca?.plan) return;
    state.dca.plan.enabled = !state.dca.plan.enabled;
    renderDca();
    savePlan();
  });
  $("#algorithmPowerToggle")?.addEventListener("click", () => {
    if (activeAlgorithmKey() === "none") {
      state.controls.algorithm_enabled = false;
      renderAlgorithmPower();
      return;
    }
    state.controls.algorithm_enabled = !state.controls.algorithm_enabled;
    renderAlgorithmPower();
    saveControlsOnly({ renderDecks: false });
  });
  $("#tradingAccount")?.addEventListener("change", (event) => {
    state.controls.trading_account_id = event.target.value;
    renderAlgorithmPower();
    saveControlsOnly({ renderDecks: false });
  });
  $("#cronPattern")?.addEventListener("input", renderScheduleDescription);
  $("#cronPattern")?.addEventListener("change", () => savePlan());
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
  document.querySelectorAll(".deckArrow").forEach((button) => {
    button.addEventListener("click", (event) => {
      const target = event.currentTarget;
      const direction = Number(target.dataset.direction || 0);
      if (target.dataset.deck === "algorithm") shiftAlgorithmDeck(direction);
      if (target.dataset.deck === "options") shiftOptionsDeck(direction);
    });
  });
  window.addEventListener("resize", () => {
    renderDca();
  });
  // Allow keyboard delete/backspace to remove the currently selected bubble
  window.addEventListener("keydown", (event) => {
    const target = event.target;
    const tag = target && target.tagName && target.tagName.toLowerCase();
    if (tag === "input" || tag === "textarea" || target.isContentEditable) return;
    if ((event.key === "Delete" || event.key === "Backspace") && state.selected) {
      // find the bucket that contains the selected symbol
      const fromItems = BUCKET_NAMES.flatMap((bucketName) =>
        bucketItems(bucketName).map((item) => ({ bucketName, item })),
      );
      const found = fromItems.find(({ item }) => item.symbol === state.selected);
      if (found) {
        removeSymbol(found.bucketName, state.selected);
        state.selected = null;
        return;
      }
      // fallback: if plan not yet loaded or item not found in plan, remove from rendered nodes
      const nodeIndex = state.nodes.findIndex((n) => n.symbol === state.selected);
      if (nodeIndex !== -1) {
        state.nodes.splice(nodeIndex, 1);
        state.selected = null;
        renderBoard();
      }
    }
  });
}

function handleSignalCardClick(event) {
  const universeButton = event.target.closest("[data-refresh-universe]");
  if (universeButton) {
    recommendUniverse();
    return;
  }
  const applyUniverseButton = event.target.closest("[data-apply-universe]");
  if (applyUniverseButton) {
    applyUniverseProposal();
    return;
  }
  const refreshButton = event.target.closest("[data-refresh-backtest]");
  if (!refreshButton) return;
  const strategyKey = refreshButton.dataset.refreshBacktest;
  delete state.signals[strategyKey];
  loadSignals(strategyKey);
  loadBacktest(strategyKey, true);
}

async function init() {
  wireEvents();
  renderAlgorithmDeck();
  renderOptionsDeck();
  renderOptionsPower();
  renderStaticBubbles();
  try {
    const [statusPayload, universePayload, dcaPayload, controlsPayload] = await Promise.all([
      api("/api/status", { timeoutMs: 5000 }),
      api("/api/universe", { timeoutMs: 5000 }),
      api("/api/dca", { timeoutMs: 5000 }),
      api("/api/controls", { timeoutMs: 5000 }),
    ]);
    state.status = statusPayload;
    state.universe = universePayload.rows || [];
    state.dca = dcaPayload;
    state.controls = controlsPayload.controls || state.controls;
    state.accounts = controlsPayload.accounts || [];
    state.bot = controlsPayload.bot || statusPayload.bot || null;
    state.dca.plan.max_item_amount = MAX_AMOUNT;
    hydrateStoredBacktests();
    renderDca();
    renderAlgorithmDeck();
    renderOptionsDeck();
    renderOptionsPower();
    renderOptionsSignals();
    loadCachedBacktestsForDeck();
  } catch (error) {
    showToast(`Could not load DCA data: ${error.message}`);
    return;
  }
}

function renderStaticBubbles() {
  calculateLayout();
  const svg = $("#bubbleBoard");
  svg.setAttribute("viewBox", `0 0 ${state.layout.width} ${state.layout.height}`);
  svg.replaceChildren();
  BUCKET_NAMES.forEach((bucketName) => {
    const bucket = state.layout.buckets[bucketName];
    const color = DISABLED_COLORS[bucketName];
    svg.appendChild(svgEl("circle", {
      class: `bucket-blob ${bucketName}`,
      cx: bucket.cx,
      cy: bucket.cy,
      r: bucket.r,
      fill: color,
      stroke: shadeColor(color, -30),
    }));
    svg.appendChild(textEl({
      class: "bucket-label",
      x: bucket.cx,
      y: bucket.cy,
      "text-anchor": "middle",
      fill: color,
    }, bucket.label));
  });
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

init();
