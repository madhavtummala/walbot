"""Bursty DCA: accrue a monthly budget per symbol, deploy it sized by valuation and by backlog.

``config.py`` holds the sizing knobs and the per-symbol plan, ``algorithm.py`` the accrual and
the decision, ``signals.py`` the rows the deck renders. Import from the module that owns the
name -- a facade here would put the algorithm inside ``core.config``'s import.
"""
