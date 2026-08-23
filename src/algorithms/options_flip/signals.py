"""What an Options Flip run decided, and the setup behind it.

This algorithm buys at most one contract set per run, so most sessions it picks nothing. That
makes the *rejected* candidates the interesting part of the view: a run that shows one row and
"macro flat" is telling you the direction gate never opened, which is a different thing from
one that scored twelve names and found none aligned.

Every candidate therefore gets a row and its own gate list, not just the winner.
"""

from __future__ import annotations

from typing import Any

from ...core.interfaces import (
    ACTION_BLOCKED,
    ACTION_ENTER,
    ACTION_IDLE,
    AlgorithmPlan,
    Check,
    SignalRow,
    SignalView,
)
from .config import OptionsFlipConfig


def candidate_checks(scored: dict[str, Any], cfg: OptionsFlipConfig, macro: str) -> list[dict[str, Any]]:
    """The four conditions a name has to meet to be worth a contract, as measured.

    Scoring is additive rather than a hard gate -- a name can be picked while failing one of
    these -- so these are stated as conditions with their contribution, not as vetoes. The
    exception is price range, which really is a veto and short-circuits the rest.
    """
    if "price_outside_range" in scored:
        return [_dict(Check(
            label="Price in tradable range",
            ok=False,
            value=f"${float(scored.get('last_price', 0.0)):,.2f}",
            limit=f"${cfg.min_price:,.0f} - ${cfg.max_price:,.0f}",
            blocking=True,
        ))]

    vol_regime = float(scored.get("vol_regime", 0.0))
    range_expansion = float(scored.get("range_expansion", 0.0))
    volume_ratio = float(scored.get("volume_ratio", 0.0))
    aligned = bool(scored.get("direction_aligned"))
    momentum = float(scored.get("intraday_momentum", 0.0))
    return [
        _dict(Check(
            label=f"Direction matches {macro} macro",
            ok=aligned,
            value=f"{momentum:+.2%} intraday, {'above' if scored.get('above_vwap') else 'below'} VWAP",
            limit="momentum and VWAP both with the trend",
            # The only condition that can leave a name unpickable on its own: an unaligned name
            # never earns the 2.0 that carries a score past the field.
            blocking=not aligned,
        )),
        _dict(Check(
            label="Volatility regime expanding",
            ok=vol_regime >= cfg.vol_regime_threshold,
            value=f"{vol_regime:.2f}x",
            limit=f"≥ {cfg.vol_regime_threshold:.2f}x",
        )),
        _dict(Check(
            label="Range expansion",
            ok=range_expansion >= cfg.range_expansion_threshold,
            value=f"{range_expansion:.2f}x ATR",
            limit=f"≥ {cfg.range_expansion_threshold:.2f}x",
        )),
        _dict(Check(
            label="Volume confirming",
            ok=volume_ratio >= cfg.min_volume_ratio,
            value=f"{volume_ratio:.2f}x",
            limit=f"≥ {cfg.min_volume_ratio:.2f}x",
        )),
    ]


def _dict(check: Check) -> dict[str, Any]:
    return check.__dict__


def signal_rows(
    candidates: list[dict[str, Any]],
    picked: str,
    contract: dict[str, Any],
    cfg: OptionsFlipConfig,
    macro: str,
) -> dict[str, dict[str, Any]]:
    """One row per scored candidate; the picked one carries the contract to be ordered."""
    rows: dict[str, dict[str, Any]] = {}
    for scored in candidates:
        symbol = str(scored["symbol"])
        chosen = symbol == picked
        rows[symbol] = {
            "signal": 1 if chosen and contract.get("option_type") == "call" else -1 if chosen else 0,
            "side": "LONG" if contract.get("option_type") == "call" else "SHORT",
            "action": ACTION_ENTER if chosen else ACTION_BLOCKED,
            "score": float(scored.get("score", 0.0)),
            "close": float(scored.get("last_price", 0.0)),
            "macro": macro,
            "reason": _headline(scored, chosen, contract),
            "checks": candidate_checks(scored, cfg, macro),
            "realized_volatility": float(scored.get("vol_short", 0.0)),
            "vol_regime": float(scored.get("vol_regime", 0.0)),
            "range_expansion": float(scored.get("range_expansion", 0.0)),
            "volume_ratio": float(scored.get("volume_ratio", 0.0)),
            # Read by the order path, not only by the deck: ``submit_option_intents`` prices
            # both legs from these, so they belong to the plan rather than to its rendering.
            **(contract if chosen else {}),
        }
    return rows


def _headline(scored: dict[str, Any], chosen: bool, contract: dict[str, Any]) -> str:
    if chosen:
        return (
            f"Best setup - {contract['contracts']}x {contract['option_type']} "
            f"{contract['strike']:g} / {contract['dte']}DTE"
        )
    if "price_outside_range" in scored:
        return "Price outside the tradable range"
    if not scored.get("direction_aligned"):
        return "Not aligned with the macro trend"
    return f"Outscored (score {float(scored.get('score', 0.0)):.2f})"


def signal_view(plan: AlgorithmPlan) -> SignalView:
    """The pick, the runners-up, and -- when there is no pick -- what stopped there being one."""
    rows = [
        SignalRow(
            symbol=symbol,
            action=str(values["action"]),
            headline=str(values["reason"]),
            metrics=[
                {"label": "Score", "value": f"{float(values['score']):.2f}"},
                {"label": "Close", "value": f"${float(values['close']):,.2f}"},
                {"label": "Contract", "value": _contract(values)},
                {"label": "Entry", "value": _entry(values)},
            ],
            checks=[Check(**check) for check in values.get("checks") or []],
        )
        for symbol, values in plan.signals.items()
    ]
    rows.sort(key=lambda row: (-_score(row), -sum(check.ok for check in row.checks), row.symbol))

    macro = str(plan.metadata.get("macro") or "flat")
    summary = [
        {"label": "Macro", "value": macro.title()},
        {"label": "Evaluated", "value": str(int(plan.metadata.get("candidates_evaluated") or 0))},
        {"label": "Picked", "value": str(sum(1 for row in rows if row.action == ACTION_ENTER))},
    ]
    if not any(row.action == ACTION_ENTER for row in rows):
        # On a run that picks nothing the reason *is* the payload, and it is usually a macro
        # gate that never opened rather than anything about the candidates. Said in a word
        # here; the sentence version is the top row's headline, and putting it in a summary
        # tile as well printed it twice and wrapped it over four lines.
        summary.append({"label": "Result", "value": "No pick"})
    return SignalView(rows=rows, summary=summary)


def _contract(values: dict[str, Any]) -> str:
    if not values.get("option_type"):
        return "--"
    return f"{values['option_type'].upper()} {values['strike']:g} / {values['dte']}DTE"


def _entry(values: dict[str, Any]) -> str:
    if not values.get("entry_limit"):
        return "--"
    return f"${values['entry_limit']:.2f} -> ${values['exit_limit']:.2f}"


def _score(row: SignalRow) -> float:
    for metric in row.metrics:
        if metric["label"] == "Score":
            return float(metric["value"])
    return 0.0


def macro_view(plan: AlgorithmPlan) -> SignalView:
    """Used when the macro gate closed before any candidate was scored."""
    macro = str(plan.metadata.get("macro") or "flat")
    return SignalView(
        rows=[SignalRow(
            symbol=str(plan.metadata.get("benchmark") or "--"),
            action=ACTION_IDLE,
            headline=str(plan.metadata.get("reason") or "No pick"),
            checks=[Check(**check) for check in plan.metadata.get("checks") or []],
        )],
        # No "Result" chip. The reason is already the row's headline, verbatim, and a summary
        # tile is sized for a measurement -- a full sentence in one wrapped to four lines and
        # said nothing the row beneath it had not just said.
        summary=[
            {"label": "Macro", "value": macro.title()},
            {"label": "Candidates", "value": "0 scored"},
        ],
    )
