"""Dual momentum: relative strength selects, absolute momentum permits.

Split from a single 1344-line module along the layer boundaries its own section headers
already described. Imported exactly as before -- the registry resolves
``src.algorithms.dual_momentum:DualMomentumAlgorithm`` unchanged.
"""

from __future__ import annotations

from .config import (  # noqa: F401
    DualMomentumConfig,
)
from .scoring import (  # noqa: F401
    base_scores,
    compute_features,
    zscores,
)
from .layers import (  # noqa: F401
    covariance_matrix,
    defensive_weights,
    eligibility,
    hold_eligibility,
    market_regime,
    park_residual,
    theme_of,
    theme_allocation,
    portfolio_volatility,
    score_to_weights,
    sentiment_adjusted,
    timing,
    volatility_scale,
)
from .proposal import (  # noqa: F401
    allocation_mode,
    analyze_universe,
    build_signals,
    rank_candidates,
)
# The three underscored names are the step-2 internals the tests drive directly; everything
# else private stays inside its module rather than being re-exported for the sake of it.
from .stateful import (  # noqa: F401
    _in_cooldown,
    _record_exits,
    resolve_themes,
    apply_turnover_filters,
    partial_adjustment,
    track_eligibility,
    confirm_regime,
)
from .algorithm import (  # noqa: F401
    DualMomentumAlgorithm,
)


__all__ = [
    "DualMomentumAlgorithm",
    "DualMomentumConfig",
    "allocation_mode",
    "analyze_universe",
    "apply_turnover_filters",
    "partial_adjustment",
    "track_eligibility",
    "base_scores",
    "build_signals",
    "compute_features",
    "confirm_regime",
    "covariance_matrix",
    "defensive_weights",
    "eligibility",
    "hold_eligibility",
    "market_regime",
    "park_residual",
    "theme_of",
    "theme_allocation",
    "portfolio_volatility",
    "rank_candidates",
    "score_to_weights",
    "sentiment_adjusted",
    "timing",
    "volatility_scale",
    "zscores",
]
