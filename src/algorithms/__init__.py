"""The algorithms, and the contract they share.

Deliberately empty of re-exports. ``core.config`` canonicalises strategy ids through
``algorithms.ids``, and a facade here would drag the registry -- and through it ``base``, the
order-placing layer and every brokerage -- into that import, closing a cycle back onto
``core.config`` before it has finished loading. Import from the module that owns the name:
``.base`` for ``BaseAlgorithm``, ``.registry`` for the lookup, ``core.interfaces`` for the
dataclasses.
"""
