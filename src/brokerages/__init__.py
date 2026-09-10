"""Brokerages: one package per venue.

``brokerage.py`` holds the :class:`BaseBrokerage` subclass; ``client.py`` holds the
authenticated session and the calls built on it (shared with the market-data connectors).
``paper`` has no client -- there is nothing to talk to.

No re-exports here: import from the module that owns the name, or resolve a class through
``registry.get_brokerage_class``.
"""
