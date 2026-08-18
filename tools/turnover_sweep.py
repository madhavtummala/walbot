#!/usr/bin/env python3
"""Quick turnover sweep: 8 key configs × 3 periods = 24 backtests."""
import subprocess, sys, os

WALBOT = "/Users/madhav/Documents/walbot"
PYTHON = os.path.join(WALBOT, ".venv", "bin", "python")

PERIODS = [
    ("2023-01-01:2023-12-31", "2023"),
    ("2024-01-01:2024-12-31", "2024"),
    ("2025-08-18:2026-08-17", "2025-2026"),
]

configs = [
    ("baseline", {}),
    ("rerank=5d", {"rerank_interval_days": 5}),
    ("rerank=7d", {"rerank_interval_days": 7}),
    ("step=0.5", {"rebalance_step": 0.5}),
    ("threshold=5%", {"rebalance_weight_threshold": 0.05}),
    ("rerank=5 + step=0.5", {"rerank_interval_days": 5, "rebalance_step": 0.5}),
    ("rerank=5 + thresh=5%", {"rerank_interval_days": 5, "rebalance_weight_threshold": 0.05}),
    ("rerank=7 + step=0.5 + thresh=5%", {"rerank_interval_days": 7, "rebalance_step": 0.5, "rebalance_weight_threshold": 0.05}),
]

def run_bt(period, overrides):
    cmd = [PYTHON, "-m", "tools.backtest_monthly_report", "--period", period]
    for k, v in overrides.items():
        cmd += ["--set", f"{k}={v}"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=WALBOT)
    out = r.stdout + r.stderr
    m = {}
    for line in out.split("\n"):
        s = line.strip()
        if "Turnover:" in s and "Max drawdown:" in s:
            for p in s.split("|"):
                p = p.strip()
                if "Turnover:" in p:
                    m["to"] = float(p.split()[1].rstrip("x"))
                elif "Max drawdown:" in p:
                    m["dd"] = float(p.split(":")[-1].strip().rstrip("%"))
        elif s.startswith("Algo:") and "net" not in s.lower() and "Alpha" not in s:
            try: m["a"] = float(s.split(":")[1].strip().rstrip("%"))
            except: pass
        elif "Total return (net):" in s:
            try: m["n"] = float(s.split(":")[1].strip().split("%")[0])
            except: pass
    return m

results = []
for name, ov in configs:
    print(f"\n--- {name} ---", flush=True)
    pd = {}
    for period, label in PERIODS:
        m = run_bt(period, ov)
        pd[label] = m
        print(f"  {label}: Algo={m.get('a',0):+.2f}% Turn={m.get('to',0):.1f}x Net={m.get('n',0):+.2f}%", flush=True)
    results.append({"name": name, "data": pd})

print(f"\n\n{'='*115}")
print(f"{'Config':<40s} | {'23 Algo':>8s} {'23 TO':>6s} {'23 Net':>7s} | {'24 Algo':>8s} {'24 TO':>6s} {'24 Net':>7s} | {'25-26':>9s} {'TO':>6s} {'Net':>7s} | {'AvgTO':>6s}")
print("-" * 115)
for r in results:
    d = r["data"]
    tos = []
    row = f"{r['name']:<40s} |"
    for l in ["2023", "2024", "2025-2026"]:
        m = d.get(l, {})
        row += f" {m.get('a',0):>+7.2f}% {m.get('to',0):>5.1f}x {m.get('n',0):>+6.2f}% |"
        tos.append(m.get("to", 0))
    row += f" {sum(tos)/len(tos):>5.1f}x"
    print(row)
