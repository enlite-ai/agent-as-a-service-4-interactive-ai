"""Shared KPI computation for PowerGrid recommendations.

Every recommendation source expresses quality through the same KPIs, computed
from an actually-reached observation (never a what-if forecast), so the
definition lives here and is reused by every rollout consumer.
Today a single KPI (``efficiency_of_the_reco``) is produced; the structure is
kept open so additional KPIs can be added in one place later.
"""
from __future__ import annotations

import numpy as np


def compute_kpis(observation) -> dict:
    """Compute the KPIs of a reached Grid2Op observation.

    :param observation: A Grid2Op observation reached by the environment (from
        ``prepare`` or after a ``step``).
    :return: KPI dictionary, currently ``{"efficiency_of_the_reco": <max rho>}``.
    """
    # ``efficiency_of_the_reco`` is the worst line loading in the reached state:
    # the lower the max rho, the more overloads have been relieved.
    return {
        "efficiency_of_the_reco": float(np.float32(observation.rho.max())),
    }
