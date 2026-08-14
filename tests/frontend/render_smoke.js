// Executes web/static/app.js against a minimal DOM stub and walks every route.
// Catches the runtime errors that string-matching tests cannot: a missing state field, a
// dangling call after a refactor, a render that throws before it paints.
// Exits non-zero on any throw or unhandled rejection.
const winListeners = {}, docListeners = {};
const store = {};
function el(sel) {
  if (!store[sel]) store[sel] = {
    id: sel, innerHTML: "", textContent: "", value: "paper", disabled: false,
    dataset: {}, style: {}, clientWidth: 800, clientHeight: 400,
    classList: { add(){}, remove(){}, toggle(){return false;}, contains(){return false;} },
    setAttribute(){}, getAttribute(){return null;}, addEventListener(){}, appendChild(){},
    replaceChildren(){}, querySelector: (s) => el(sel + s), querySelectorAll: () => [],
    closest: () => null, focus(){}, getBoundingClientRect: () => ({left:0,top:0,width:800,height:400}),
  };
  return store[sel];
}
global.document = {
  querySelector: (s) => el(s), querySelectorAll: () => [], createElementNS: () => el("svg"),
  addEventListener: (n, f) => { docListeners[n] = f; }, body: el("body"), hidden: false,
};
global.window = {
  location: { hash: "#/algo/dca/overview" }, innerWidth: 1400,
  addEventListener: (n, f) => { winListeners[n] = f; },
  setTimeout, clearTimeout, setInterval: () => 0,
  requestAnimationFrame: () => 0, cancelAnimationFrame: () => {},
  open: () => null, prompt: () => null, scrollX: 0, scrollY: 0,
};
global.CSS = { escape: (s) => s };
global.AbortController = class { constructor(){ this.signal = {}; } abort(){} };

const PAYLOADS = {
  "/api/status": { config: { backtest_period: "2m" }, runtime_mode: "bot" },
  "/api/universe": { rows: [{ symbol: "SPY", enabled: true }] },
  "/api/controls": { controls: { bindings: [{ id: "b1", strategy: "dca", account_id: "paper", enabled: true }], trading_account_id: "paper" }, bot: {} },
  "/api/accounts": { default: "paper", rows: [{ id: "paper", label: "Alpaca Paper", broker: "alpaca", base_url: "u", data_feed: "iex", api_key_env: "K", api_secret_env: "S", credentials_ready: true, missing_env: [], deployments: ["dca"] }] },
  "/api/dca": { account_id: "paper", plan: { max_item_amount: 2000, buy: { amount: 100, items: [{ symbol: "SPY", amount: 50 }] }, sell: { amount: 0, items: [] } }, available: [], preview: {} },
  "/api/positions": { account_id: "paper", equity: 100, cash: 10, day_pl: 1, day_pl_percent: 0.01, total_pl: 2, rows: [], error: "" },
  "/api/activity": { account_id: "paper", rows: [], error: "" },
  "/api/algorithm-activity": { strategy: "dca", rows: [{ submitted_at: "2026-08-13T14:00:00+00:00", strategy: "dca", account_id: "paper", symbol: "SPY", side: "buy", quantity: 2, status: "submitted", reason: "" }] },
  "/api/schwab/auth": { configured: false, state: "unconfigured", detail: "" },
  "/api/algorithm-config": { strategy: "fast_momentum", config_key: "fast_momentum", config: { max_positions: 4 } },
};
global.fetch = async (url) => {
  const path = String(url).split("?")[0];
  const body = PAYLOADS[path] ?? {};
  return { ok: true, json: async () => body };
};

let failures = 0;
process.on("unhandledRejection", (e) => { failures++; console.log("REJECTION:", e && e.stack ? e.stack.split("\n").slice(0,3).join("\n") : e); });

require(require("path").resolve(__dirname, "../../web/static/app.js"));

setTimeout(() => {
  console.log("hashchange listener registered:", typeof winListeners.hashchange === "function");
  const routes = ["#/algo/fast_momentum/overview", "#/algo/fast_momentum/signals",
                  "#/algo/fast_momentum/tune", "#/algo/dca/tune", "#/algo/fast_momentum/backtest",
                  "#/algo/dca/signals", "#/algo/dca/overview", "#/account/paper",
                  "#/account/missing", "#/target/paper", "#/nonsense"];
  for (const hash of routes) {
    window.location.hash = hash;
    try { winListeners.hashchange && winListeners.hashchange(); console.log("  ok  ", hash); }
    catch (e) { failures++; console.log("  THROW", hash, "->", e.constructor.name + ": " + e.message); }
  }
  console.log(failures ? `\n${failures} failure(s)` : "\nno failures");
  process.exit(failures ? 1 : 0);
}, 400);
