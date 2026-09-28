# Shared Grid2Op one-step simulation scoring used by the PowerGrid agents.
# Both agent implementations rank candidate actions by the worst line loading
# they lead to, and both must reject the same unusable outcomes, so the rule
# lives here rather than being restated per agent.
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

# Score given to an action whose simulation is unusable (failed, diverged,
# illegal, or ended the episode), high enough to never be picked.
UNUSABLE_RHO = float("inf")


def simulated_rho(
    observation: Any,
    action: Any,
    on_simulate: Optional[Callable[[], None]],
) -> float:
    """
    Worst line loading one timestep after ``action``, or a penalty.

    A simulation that fails, diverges or ends the episode is scored with
    :data:`UNUSABLE_RHO` so it can never win a comparison.

    :param Any observation: The Grid2Op observation to simulate from.
    :param Any action: The Grid2Op action to simulate.
    :param Optional[Callable[[], None]] on_simulate: Called once per attempted
        simulation, including failed ones, so callers can keep a power-flow
        count. ``None`` to count nothing.
    :return float: Simulated worst line loading, or the penalty value.
    """
    try:
        simulated, _, done, info = observation.simulate(action, time_step=1)
    except BaseException:
        # The assistant's own search swallows simulation failures the same
        # way: a candidate that cannot be simulated is simply not chosen.
        return UNUSABLE_RHO
    finally:
        if on_simulate is not None:
            on_simulate()
    if (
        simulated is None
        or done
        or info.get("is_illegal")
        or info.get("is_ambiguous")
        or bool(np.any(np.isnan(simulated.rho)))
    ):
        return UNUSABLE_RHO
    return float(np.max(simulated.rho))


def action_substation(action: Any) -> Optional[int]:
    """
    Substation a topological action acts on, for diversity filtering.

    :param Any action: The Grid2Op action to inspect.
    :return Optional[int]: The substation id, or ``None`` when the action is not
        a bus assignment (a redispatch, for instance).
    """
    impact = action.impact_on_objects()
    if impact["topology"]["changed"]:
        assigned = impact["topology"]["assigned_bus"]
        if assigned:
            return assigned[0]["substation"]
    return None
