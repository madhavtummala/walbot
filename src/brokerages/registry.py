"""Which brokerages exist, and how to reach one without loading the others.

Entries resolve on first use, through the same :class:`Registry` the algorithms use. Importing
them eagerly meant that touching *any* brokerage loaded the Alpaca SDK -- on a Schwab-only
deployment, for a class never constructed -- and put ``core.config`` at the end of an import
chain that starts in ``core.pipeline``, which is how that cycle formed.
"""

from __future__ import annotations

from typing import Type

from ..common.registry import Registry
from .base import BaseBrokerage

BROKERAGES: Registry[BaseBrokerage] = Registry(
    "brokerage",
    BaseBrokerage,
    {
        "alpaca": "src.brokerages.alpaca.brokerage:AlpacaBrokerage",
        "paper": "src.brokerages.paper.brokerage:PaperBrokerage",
        "schwab": "src.brokerages.schwab.brokerage:SchwabBrokerage",
    },
)


def get_brokerage_class(broker_type: str) -> Type[BaseBrokerage]:
    """The class registered for ``broker_type``, importing the vendor's module on first use.

    Raises :class:`KeyError` for an unknown id, which ``resolve_brokerage`` turns into
    ``UnknownBrokerageError`` with the configured value named.
    """
    return BROKERAGES.get(broker_type)
