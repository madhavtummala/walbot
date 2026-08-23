"""Schwab: the trading account, the authenticated session, and the OAuth consent that mints it.

``brokerage.py`` places orders; ``client.py`` holds the session and token refresh that the
streaming feed and the connectors share; ``auth.py`` is the one-time consent flow the dashboard
drives. Import from the module that owns the name.
"""
