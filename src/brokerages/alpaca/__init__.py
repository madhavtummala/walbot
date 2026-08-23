"""Alpaca: the trading account, and the SDK session the market-data feed shares with it.

``brokerage.py`` places orders; ``client.py`` is the transport underneath, used both by that
and by ``connectors/market/alpaca.py``. Import from the module that owns the name.
"""
