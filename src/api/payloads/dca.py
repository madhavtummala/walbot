"""Dollar-cost-averaging plans.

Split out of the single ``api_payloads`` module, which had grown to 1253 lines covering nine
unrelated domains. The public names are unchanged and still importable from ``api_payloads``.
"""


from __future__ import annotations

from ...algorithms.dca import allocation_preview
from ...algorithms.dca import load_dca_plan
from ...algorithms.dca import save_dca_plan

import logging
from typing import Any

logger = logging.getLogger(__name__)
# The DCA preview is expressed against the tradable universe, so it reads that view rather
# than re-deriving the symbol list and risking a second, divergent answer.
from .universe import universe_payload





def dca_payload(account_id: str = "") -> dict[str, Any]:
    universe = universe_payload()["rows"]
    plan = load_dca_plan(universe, account_id=account_id)
    available = [row for row in universe if row["enabled"]]
    return {
        "account_id": account_id,
        "plan": plan,
        "available": available,
        "preview": allocation_preview(plan),
    }


def save_dca_payload(body: dict[str, Any]) -> dict[str, Any]:
    universe = universe_payload()["rows"]
    account_id = str(body.get("account_id") or "")
    plan = save_dca_plan(body.get("plan", body), universe, account_id=account_id)
    available = [row for row in universe if row["enabled"]]
    return {
        "account_id": account_id,
        "plan": plan,
        "available": available,
        "preview": allocation_preview(plan),
    }
