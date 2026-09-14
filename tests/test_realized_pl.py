"""Realized P/L: the profit an account has already banked.

No venue reports this as a field, so it is matched from fills. These pin the arithmetic and,
more importantly, the honesty rules -- what happens when the record is incomplete.
"""

from __future__ import annotations

from src.brokerages.realized import realized_from_fills


def _fill(symbol: str, action: str, quantity: float, price: float, date: str, multiplier: float = 1.0) -> dict:
    return {
        "symbol": symbol, "action": action, "quantity": quantity,
        "price": price, "multiplier": multiplier, "date": date,
    }


def test_an_option_round_trip_carries_the_contract_multiplier() -> None:
    """The live case this was built for: one USO call, bought at 8.85 and sold at 14.10.

    A $5.25 move on one contract is $525, not $5.25 -- the broker quotes the option per share
    and sells it a hundred at a time.
    """
    result = realized_from_fills([
        _fill("USO260916C00142000", "buy", 1, 8.85, "2026-09-09T16:23:09Z", multiplier=100.0),
        _fill("USO260916C00142000", "sell", 1, 14.10, "2026-09-10T16:00:39Z", multiplier=100.0),
    ])

    assert result["realized_pl"] == 525.0
    assert result["closes"] == 1
    assert result["unmatched"] == 0


def test_fills_are_matched_oldest_first_however_the_broker_ordered_them() -> None:
    """Every broker feed here hands back newest-first, which would match each sell against
    buys that had not happened yet."""
    newest_first = [
        _fill("SPY", "sell", 10, 110.0, "2026-09-10"),
        _fill("SPY", "buy", 10, 100.0, "2026-09-01"),
    ]

    assert realized_from_fills(newest_first)["realized_pl"] == 100.0


def test_a_second_buy_moves_the_basis_rather_than_replacing_it() -> None:
    result = realized_from_fills([
        _fill("SPY", "buy", 10, 100.0, "2026-09-01"),
        _fill("SPY", "buy", 10, 120.0, "2026-09-02"),
        _fill("SPY", "sell", 20, 130.0, "2026-09-03"),
    ])

    # Average cost 110 across 20 shares, sold at 130.
    assert result["realized_pl"] == 400.0


def test_a_partial_sell_leaves_the_rest_open() -> None:
    result = realized_from_fills([
        _fill("SPY", "buy", 10, 100.0, "2026-09-01"),
        _fill("SPY", "sell", 4, 110.0, "2026-09-02"),
    ])

    assert result["realized_pl"] == 40.0
    assert result["closes"] == 1


def test_a_sell_with_no_opening_buy_is_counted_rather_than_guessed() -> None:
    """Its cost basis is genuinely unknowable from this window.

    Booking it at a zero basis would report the entire proceeds as profit; dropping it silently
    would report a total that is too small and say nothing about it. Counting it does neither.
    """
    result = realized_from_fills([_fill("SPY", "sell", 10, 110.0, "2026-09-02")])

    assert result["realized_pl"] == 0.0
    assert result["closes"] == 0
    assert result["unmatched"] == 1


def test_a_sell_larger_than_the_open_position_closes_what_it_can() -> None:
    result = realized_from_fills([
        _fill("SPY", "buy", 4, 100.0, "2026-09-01"),
        _fill("SPY", "sell", 10, 110.0, "2026-09-02"),
    ])

    # The four shares this window can account for, and a flag on the six it cannot.
    assert result["realized_pl"] == 40.0
    assert result["unmatched"] == 1


def test_positions_in_different_symbols_do_not_cross() -> None:
    result = realized_from_fills([
        _fill("SPY", "buy", 10, 100.0, "2026-09-01"),
        _fill("QQQ", "sell", 10, 500.0, "2026-09-02"),
    ])

    assert result["realized_pl"] == 0.0
    assert result["unmatched"] == 1


def test_an_account_that_never_closed_anything_has_realized_nothing() -> None:
    result = realized_from_fills([_fill("SPY", "buy", 10, 100.0, "2026-09-01")])

    assert result == {"realized_pl": 0.0, "closes": 0, "unmatched": 0}
