"""Which brokerages exist, and how to reach one without loading the others.

Entries resolve on first use: importing them eagerly would load every vendor SDK for a class
that may never be constructed.
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

    Raises :class:`KeyError` for an unknown id.
    """
    return BROKERAGES.get(broker_type)
