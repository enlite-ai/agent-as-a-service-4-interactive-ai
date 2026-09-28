"""Declares the PowerGrid use case to the environment-agnostic A3S core.

This is the whole of the wiring for this integration, and it is deliberately
thin: the core already owns the multi-step projection, so all that is declared
here is *which* environment to project (the Grid2Op adapter) and *which*
recommendation engine supplies the actions to project, plus how to read a
PowerGrid request context. Nothing about rollouts, branching or response shaping
lives here - that is the core's job for every environment alike.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from a3s_core import Agent, Environment, EnvironmentUseCase, SerializedEnvironmentState

from .agent_implementations.t2_1_agent import ExpertPowerGridAgent
from .agent_implementations.xd_agent import PowerGridAgent
from .environment import PowerGridEnvironment
from .serialization import legacy_context_to_state, validate_grid2op_envelope

logger = logging.getLogger(__name__)

# The recommendation engines that can serve this use case, so the two can be
# evaluated against each other on identical requests. Both ship in the image and
# neither loads anything until its first request, so naming one costs nothing at
# startup for the other:
#   "xd"   - the XD_silly_repo assistant's planner (default, the shipped behaviour)
#   "t2.1" - the T2.1_deep_expert PPO policy (see agent_implementations/t2_1_agent.py)
RECOMMENDATION_ENGINES = {
    "xd": PowerGridAgent,
    "t2.1": ExpertPowerGridAgent,
}
DEFAULT_ENGINE = "xd"


class PowerGridUseCase(EnvironmentUseCase):
    """The Grid2Op power-grid deployment of the A3S core."""

    @property
    def name(self) -> str:
        """:return str: The use-case name callers address this backend by."""
        return "PowerGrid"

    def build_environment(self) -> Environment:
        """
        Builds the Grid2Op environment adapter.

        :return Environment: The PowerGrid environment adapter.
        """
        return PowerGridEnvironment()

    def build_agent(self, environment: Environment) -> Agent:
        """
        Builds the recommendation engine named by ``A3S_POWERGRID_AGENT``.

        :param Environment environment: The environment the engine acts on; both
            engines need it for the action space and for what-if simulation.
        :return Agent: The selected recommendation engine.
        :raises ValueError: If the variable names an engine that does not exist.
            This fails at startup rather than silently serving the default,
            because a typo would otherwise be indistinguishable from a valid
            comparison run.
        """
        name = os.environ.get("A3S_POWERGRID_AGENT", DEFAULT_ENGINE).strip().lower()
        engine_class = RECOMMENDATION_ENGINES.get(name)
        if engine_class is None:
            raise ValueError(
                f"Unknown A3S_POWERGRID_AGENT={name!r}; expected one of "
                f"{sorted(RECOMMENDATION_ENGINES)}"
            )
        logger.info("PowerGrid served by the %r recommendation engine", name)
        return engine_class(environment)

    def validate_envelope(self, envelope: SerializedEnvironmentState) -> None:
        """
        Rejects an envelope this build cannot read, at the API edge.

        :param SerializedEnvironmentState envelope: The incoming envelope.
        :return None:
        :raises ValueError: If the envelope was not produced by a Grid2Op
            serializer this build understands.
        """
        validate_grid2op_envelope(envelope)

    def adapt_legacy_context(
        self, context: dict[str, Any]
    ) -> SerializedEnvironmentState | None:
        """
        Wraps a bare T2.1_deep_expert context into a state envelope.

        Keeps this service a drop-in replacement for the external RL agent API,
        whose context carries a raw Grid2Op observation and no envelope.

        :param dict context: The caller's context payload, without an envelope.
        :return SerializedEnvironmentState: The envelope, or ``None`` when the
            context carries no observation either.
        """
        if not context.get("observation"):
            return None
        logger.info("Wrapping legacy PowerGrid context into a state envelope")
        return legacy_context_to_state(context)
