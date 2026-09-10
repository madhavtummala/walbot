"""How Schwab spells an option symbol.

Its own module because both Schwab modules need it and neither may import the other: the
options module already imports the price-history URL from the quote module, so the dependency
only runs one way.
"""

from __future__ import annotations

from ...core.options import osi_symbol, parse_osi


def schwab_osi(symbol: str) -> str:
    """An option symbol in the 21-character padded OSI form Schwab requires.

    Brokers do not agree on this spelling. Alpaca reports a position as
    ``USO260916C00142000``; Schwab wants ``USO   260916C00142000``, the root space-padded to
    six. Asking Schwab for the unpadded form returns **an empty result rather than an error**,
    which is the worst of the available failures -- verified live against the same contract:

        unpadded 'USO260916C00142000'     -> 0 candles, no quote
        padded   'USO   260916C00142000'  -> 46 candles, $13.00

    So a contract held at one broker and priced at another had no history and no live quote,
    and nothing reported a fault: the band fell back to translating the underlying through
    delta, and the mark fell through to whatever the cache last held.

    Re-emitted from the parsed symbol rather than string-padded, so a root of any length lands
    in the right column. Anything that is not a parseable OSI symbol -- an ordinary equity
    ticker, most of all -- is passed through untouched.
    """
    text = str(symbol or "").upper()
    try:
        parsed = parse_osi(text)
    except Exception:  # noqa: BLE001 - not an OSI symbol, so not ours to rewrite
        return text
    return osi_symbol(
        parsed["underlying"], parsed["expiry"], parsed["option_type"], parsed["strike"],
        padded=True,
    )
