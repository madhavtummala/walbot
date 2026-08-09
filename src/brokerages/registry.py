from __future__ import annotations

from .providers.alpaca import AlpacaBrokerage
from .providers.paper import PaperBrokerage
from .providers.schwab import SchwabBrokerage

BROKERAGE_REGISTRY = {
    "alpaca": AlpacaBrokerage,
    "paper": PaperBrokerage,
    "schwab": SchwabBrokerage,
}
