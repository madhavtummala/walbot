from __future__ import annotations
import logging
from ..api.controls import load_controls
from ..core import pipeline
from ..core.config import DEFAULT_STRATEGY_ID, get_config
from ..common.logging_utils import configure_logging, log_signals, log_portfolio, log_orders, log_position_changes
from ..core.strategy_models import STRATEGY_LABELS
from ..data.order_journal import record_orders
from ..algorithms.registry import canonical_algorithm_id

logger = logging.getLogger(__name__)


def run_once(account_id: str | None = None, strategy: str | None = None) -> None:
    """Run one algorithm against one account.

    ``strategy`` is passed explicitly by the runtime, which drives several bindings at once and
    cannot rely on a single selected strategy in controls. It falls back to that selection so a
    bare ``run_once()`` from a shell still does the obvious thing.
    """
    controls = load_controls()
    strategy = canonical_algorithm_id(strategy or controls.get("active_strategy") or DEFAULT_STRATEGY_ID)
    config = (
        get_config(account_id=account_id, strategy_id=strategy)
        if account_id
        else get_config(strategy_id=strategy)
    )
    configure_logging()

    logger.info(
        "Starting live runner for account %s with trading endpoint %s",
        config.account_id,
        config.alpaca_base_url,
    )

    if config.kill_switch:
        logger.warning("Kill switch is enabled. Exiting without sending orders.")
        return

    if not controls["algorithm_enabled"]:
        logger.warning("Algorithm trading is disabled in dashboard controls. Exiting without sending orders.")
        return
    logger.info("Active strategy: %s", STRATEGY_LABELS.get(strategy, strategy))

    try:
        brokerage = pipeline.resolve_brokerage(config)
    except pipeline.UnknownBrokerageError as exc:
        logger.error("%s", exc)
        return

    if not brokerage.is_market_open():
        logger.warning("Market is closed according to brokerage clock. Exiting without sending orders.")
        return

    # Step 1: algorithm + data only. Identical call to the one the MCP agent makes.
    result = pipeline.run_algorithm(strategy, config)

    if result.metadata["requirements"].paper_only and "paper-api.alpaca.markets" not in str(config.alpaca_base_url):
        logger.warning(
            "%s is restricted to Alpaca paper trading by default. Configured endpoint %s is not paper; exiting without orders.",
            STRATEGY_LABELS.get(strategy, strategy),
            config.alpaca_base_url,
        )
        return

    logger.info("%s metadata: %s", STRATEGY_LABELS.get(strategy, strategy), result.metadata)
    log_signals(result.signals, result.latest_prices)

    # Step 2: brokerage + algorithm. The scheduled flow goes straight here; the MCP flow
    # pauses in between so an agent can validate the proposal first.
    outcome = pipeline.place_orders(
        result,
        config,
        brokerage,
        require_approval=config.require_trade_approval,
        approval_timeout_seconds=config.trade_approval_timeout_seconds,
        approval_poll_seconds=config.trade_approval_poll_seconds,
    )
    log_portfolio(outcome["final_weights"], outcome["equity"])
    log_orders(outcome["order_results"])
    log_position_changes(outcome["order_results"])
    # The brokerage feed cannot say which algorithm placed what, so the attribution is
    # recorded here, at the one point where both facts are in hand.
    record_orders(strategy, config.account_id, outcome["order_results"])


def main() -> None:
    run_once()


if __name__ == "__main__":
    main()
