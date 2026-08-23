"""Dual momentum: relative strength selects, absolute momentum permits.

``config.py`` holds the knobs, ``algorithm.py`` the two passes, ``signals.py`` the rows the deck
renders. Three private modules carry what ``algorithm.py`` would otherwise be unreadable with:
``scoring.py`` (features and the cross-sectional score), ``gates.py`` (every test a name must
pass, as data), ``sizing.py`` (weights, and how much of the move to make today), ``memory.py``
(the little this algorithm remembers between runs).

Import from the module that owns the name. This file used to re-export twenty-five of them so
that tests could reach the internals through one import, which made the package facade a second,
silently drifting description of what the algorithm is made of.
"""
