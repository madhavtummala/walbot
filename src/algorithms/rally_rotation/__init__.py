"""Dual momentum: relative strength selects, absolute momentum permits.

Split from a single 1344-line module along the layer boundaries its own section headers
already described. Imported exactly as before -- the registry resolves
``src.algorithms.rally_rotation:RallyRotationAlgorithm`` unchanged.
"""

from __future__ import annotations

from .config import (  # noqa: F401
    RallyRotationConfig,
)
from .scoring import (  # noqa: F401
    base_scores,
    compute_features,
    zscores,
)
from .layers import (  # noqa: F401
    defensive_weights,
    eligibility,
    crash_stop,
    climax_top,
    hold_eligibility,
    universe_data_ok,
    park_residual,
    score_to_weights,
    sentiment_adjusted,
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
    _record_exits,
    resolve_positions,
    apply_turnover_filters,
    partial_adjustment,
    action_due,
    advance_run,
    record_action,
    track_eligibility,
)
from .algorithm import (  # noqa: F401
    RallyRotationAlgorithm,
)


__all__ = [
    "RallyRotationAlgorithm",
    "RallyRotationConfig",
    "allocation_mode",
    "analyze_universe",
    "apply_turnover_filters",
    "partial_adjustment",
    "action_due",
    "advance_run",
    "record_action",
    "track_eligibility",
    "base_scores",
    "build_signals",
    "compute_features",
    "defensive_weights",
    "eligibility",
    "crash_stop",
    "hold_eligibility",
    "universe_data_ok",
    "park_residual",
    "rank_candidates",
    "score_to_weights",
    "sentiment_adjusted",
    "zscores",
]
