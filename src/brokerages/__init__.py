"""Brokerages: one package per venue, each with the same two roles.

``brokerage.py``
    The :class:`BaseBrokerage` subclass -- positions, account state, orders. What the order
    path talks to.
``client.py``
    The authenticated session and the calls built on it. Shared, because "place an order" and
    "fetch a bar" hit the same vendor with the same credentials: ``connectors/market`` reads
    this module too. A venue with nothing to talk to (``paper``) has no client.

Deliberately no re-exports. ``core.pipeline`` imports the registry, and a facade here that
pulled in the concrete classes would load every vendor SDK -- and with it ``core.config`` --
on any import of this package. Import from the module that owns the name, or resolve a class
through ``registry.get_brokerage_class``.
"""
