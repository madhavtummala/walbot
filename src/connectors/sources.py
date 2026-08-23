"""Which providers to try, in what order, and whether one is usable at all.

A *category* names a provider-configuration section and a rate-limit namespace -- not a cache
category. Bars all land in one store keyed by resolution, because intraday and EOD were never
two kinds of data, only two grids. See ``src/data/duckdb_store.py``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ..core.config import Config
from ..data.provider_cache import provider_is_limited

logger = logging.getLogger(__name__)

MARKET_CATEGORY = "market_data"
INTRADAY_MARKET_CATEGORY = "intraday_market_data"
EOD_MARKET_CATEGORY = "eod_market_data"
NEWS_CATEGORY = "news_sentiment"
SENTIMENT_CATEGORY = "sentiment_data"
EXTERNAL_AUTH_PROVIDERS = {"alpaca", "schwab"}

_CATEGORY_ALIASES = {
    MARKET_CATEGORY: (MARKET_CATEGORY, EOD_MARKET_CATEGORY),
    EOD_MARKET_CATEGORY: (EOD_MARKET_CATEGORY, MARKET_CATEGORY),
    NEWS_CATEGORY: (NEWS_CATEGORY, SENTIMENT_CATEGORY),
    SENTIMENT_CATEGORY: (SENTIMENT_CATEGORY, NEWS_CATEGORY),
    INTRADAY_MARKET_CATEGORY: (INTRADAY_MARKET_CATEGORY, EOD_MARKET_CATEGORY, MARKET_CATEGORY),
}


class ProviderUnavailable(RuntimeError):
    pass


class ProviderRateLimited(ProviderUnavailable):
    pass


def _provider_config(config: Config, category: str, provider: str) -> dict[str, Any]:
    root = config.data_source_configs if isinstance(config.data_source_configs, dict) else {}
    for candidate_category in _CATEGORY_ALIASES.get(category, (category,)):
        section = root.get(candidate_category, {}) if isinstance(root.get(candidate_category, {}), dict) else {}
        providers = section.get("providers", {}) if isinstance(section.get("providers", {}), dict) else {}
        provider_config = providers.get(provider, {}) if isinstance(providers.get(provider, {}), dict) else {}
        if provider_config or provider in providers:
            return provider_config
    return {}


def _provider_configured(config: Config, category: str, provider: str) -> bool:
    root = config.data_source_configs if isinstance(config.data_source_configs, dict) else {}
    for candidate_category in _CATEGORY_ALIASES.get(category, (category,)):
        section = root.get(candidate_category, {}) if isinstance(root.get(candidate_category, {}), dict) else {}
        providers = section.get("providers", {}) if isinstance(section.get("providers", {}), dict) else {}
        if provider in providers:
            return True
    return False


def _enabled(config: Config, category: str, provider: str, *, uses_external_auth: bool = False) -> bool:
    if not _provider_configured(config, category, provider):
        return False
    provider_config = _provider_config(config, category, provider)
    enabled = provider_config.get("enabled")
    if enabled is not None:
        return bool(enabled)
    return uses_external_auth or bool(_api_key(config, category, provider))


def _next_provider_name(
    providers: list[str],
    current_index: int,
    fetchers: dict[str, Any],
    config: Config,
    category: str,
    external_auth_providers: set[str],
) -> str:
    for candidate in providers[current_index + 1 :]:
        if candidate not in fetchers or provider_is_limited(candidate):
            continue
        if _enabled(config, category, candidate, uses_external_auth=candidate in external_auth_providers):
            return candidate
    return ""


def _fallback_suffix(next_provider: str) -> str:
    if next_provider:
        return f"; falling back to {next_provider}"
    return "; no configured fallback provider remains"


def _api_key(config: Config, category: str, provider: str) -> str:
    provider_config = _provider_config(config, category, provider)
    defaults = {
        "finnhub": "FINNHUB_API_KEY",
        "alpha_vantage": "ALPHA_VANTAGE_API_KEY",
        "marketaux": "MARKETAUX_API_KEY",
        "newsapi": "NEWSAPI_API_KEY",
    }
    env_name = str(provider_config.get("api_key_env") or defaults.get(provider, "")).strip()
    configured_key = str(provider_config.get("api_key") or "").strip()
    return os.getenv(env_name, configured_key).strip() if env_name else configured_key


def _access_token(config: Config, category: str, provider: str) -> str:
    provider_config = _provider_config(config, category, provider)
    env_name = str(provider_config.get("access_token_env") or provider_config.get("bearer_token_env") or "").strip()
    configured_token = str(provider_config.get("access_token") or provider_config.get("bearer_token") or "").strip()
    return os.getenv(env_name, configured_token).strip() if env_name else configured_token


def _schwab_token(config: Config, category: str) -> str:
    """Schwab bearer token, refreshed via OAuth when app credentials are configured.

    Schwab access tokens last ~30 minutes, so a statically configured token goes stale almost
    immediately; it is kept only as a fallback for manual testing.

    Gated on the app credentials alone, not on a configured refresh token: consent completed
    through the dashboard stores its refresh token in the state store, which is where
    ``SchwabSession`` looks when the config has none. Requiring SCHWAB_REFRESH_TOKEN here made
    the connector unusable for exactly the flow the dashboard exists to drive.
    """
    if getattr(config, "schwab_app_key", "") and getattr(config, "schwab_app_secret", ""):
        from src.brokerages.schwab.client import SchwabAuthError, SchwabSession

        try:
            return SchwabSession(config).access_token()
        except SchwabAuthError as error:
            # No consent yet, or the refresh token expired: fall through to the ladder rather
            # than taking down the whole fetch.
            logger.info("Schwab OAuth session unavailable, falling back: %s", error)
            return ""
    return _access_token(config, category, "schwab")
