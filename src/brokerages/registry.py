from __future__ import annotations

from .providers.alpaca import AlpacaBrokerage

BROKERAGE_REGISTRY = {
    "alpaca": AlpacaBrokerage
}
