"""Generic request / response interface for Agent-as-a-Service backends.

Defines the whole contract a new use case implements: the normalized request and
recommendation shapes, plus the :class:`Environment` and :class:`Agent` adapters
the rollout core is written against.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .serialization import SerializedEnvironmentState

ActionType = dict[str, Any]
ObservationType = dict[str, Any]
EventType = dict[str, Any]
KPIType = dict[str, Any]


@dataclass(slots=True)
class A3SRecommendationRequest:
    """Normalized request passed from the API layer to an A3S backend."""

    use_case: str
    environment_state: SerializedEnvironmentState | None
    event: EventType = field(default_factory=dict)
    # Number of alternative first actions to fan out at the first rollout step.
    max_recommendations: int = 3
    # Projection depth: how many timesteps the agent's policy is followed *after*
    # each alternative. 0 asks for the alternative's own outcome and nothing more,
    # so a branch spans `n_steps + 1` timesteps.
    n_steps: int = 0


@dataclass(slots=True)
class A3SRecommendation:
    """Normalized recommendation returned by an A3S backend."""

    title: str
    description: str
    actions: list[ActionType]
    agent_type: str
    kpis: KPIType | None = None
    # 0-based index of the fanned-out candidate this recommendation belongs to.
    branch_index: int | None = None
    # 1-based rollout timestep within its branch.
    step: int | None = None
    # Whether the transition into this step terminated the episode. The KPIs of a
    # `done` step describe a *terminal* state, in which an environment may
    # report degenerate values, so they must not be read as ordinary ones.
    done: bool = False
    # Absolute environment time of the observation this step reached, as opposed
    # to `step`, which counts rollout timesteps from 1. `None` when the
    # environment cannot report an absolute clock.
    env_timestep: int | None = None

    def to_dict(self, use_case: str) -> dict[str, Any]:
        """
        Converts the recommendation to the API response shape.

        :param str use_case: Use-case name attached to the response item.
        :return dict: JSON-ready recommendation dictionary.
        """
        return {
            "title": self.title,
            "description": self.description,
            "use_case": use_case,
            "agent_type": self.agent_type,
            "actions": self.actions,
            "kpis": self.kpis,
            "branch_index": self.branch_index,
            "step": self.step,
            # `done` and `env_timestep` are named after the fields the AI4REALNET
            # WP3 reference service reports per simulated step, and carry the
            # same meaning.
            "done": self.done,
            "env_timestep": self.env_timestep,
        }


class AgentAsAService(ABC):
    """Backend able to answer a normalized recommendation request."""

    @abstractmethod
    def get_recommendations(
        self, request: A3SRecommendationRequest
    ) -> list[A3SRecommendation]:
        """
        Returns backend recommendations for the provided request.

        :param A3SRecommendationRequest request: Normalized request.
        :return list: Normalized recommendation objects.
        """


class Environment(ABC):
    """Environment-specific adapter used by the generic rollout core.

    An implementation encapsulates everything specific to a simulated
    environment: how to reconstruct its state, advance it one timestep, branch
    it, what a KPI is, and how to render a native action into the normalized
    recommendation shape. Adding a use case (e.g. Flatland) means providing an
    implementation - not touching the core.

    The core never simulates or copies directly: it advances the environment via
    :meth:`step` and obtains independent branches via :meth:`fork`, leaving each
    environment free to choose the cheap strategy for it: some simulators are
    expensive to step but cheap to copy, and others are the other way round.
    """

    @abstractmethod
    def prepare(
        self, environment_state: SerializedEnvironmentState | None
    ) -> ObservationType | None:
        """
        Reconstructs the environment to the serialized state and returns its obs.

        How the state is decoded (seed, replay history, ...) is entirely this
        environment's concern; those inputs live inside the opaque ``state`` blob.

        :param SerializedEnvironmentState environment_state: State envelope.
        :return ObservationType: Current observation, or ``None`` without state.
        """

    @abstractmethod
    def step(self, action: Any) -> tuple[ObservationType, bool]:
        """
        Advances the environment by one timestep applying ``action``.

        This mutates the environment in place; the core calls it on a branch
        obtained from :meth:`fork`, never on the shared prepared environment.

        :param Any action: A native action for this environment.
        :return tuple: ``(observation, done)`` after the step.
        """

    @abstractmethod
    def fork(self) -> "Environment":
        """
        Returns an independent environment positioned at the current state.

        Stepping the returned environment must not affect this one. Each
        implementation picks its cheapest branching primitive (state copy,
        replay-based reconstruction, ...).

        :return Environment: An independent environment branch.
        """

    @abstractmethod
    def kpis(self, observation: ObservationType) -> KPIType:
        """
        Computes this environment's KPIs from a real observation.

        Derived from an actually-reached observation (from :meth:`prepare` or
        :meth:`step`) - never from a forecast/what-if simulation - so the metric
        is well defined for any environment.

        :param ObservationType observation: An observation reached by the env.
        :return KPIType: The KPI dictionary.
        """

    @abstractmethod
    def timestep(self, observation: ObservationType) -> int | None:
        """
        Reads the absolute environment time of a reached observation.

        The rollout core counts its own timesteps from 1, which says nothing about
        where in the episode they sit; this is the environment's own clock, so a
        projection can be placed on an absolute timeline. Reading a clock off an
        observation is environment-specific, hence it lives here rather than in
        the core.

        :param ObservationType observation: An observation reached by the env.
        :return int | None: The absolute timestep, or ``None`` when this
            environment cannot report one for ``observation``.
        """

    @abstractmethod
    def format_recommendation(
        self,
        observation: ObservationType,
        action: Any,
        kpis: KPIType,
        agent_type: str,
        step: int,
        branch_index: int,
        *,
        done: bool,
        env_timestep: int | None,
    ) -> A3SRecommendation:
        """
        Renders a native action into a normalized recommendation.

        Pure presentation: it packages the already-computed KPIs and derives any
        descriptive labels; it does not simulate.

        :param ObservationType observation: Observation reached after ``action``.
        :param Any action: The native action taken at this rollout step.
        :param KPIType kpis: KPIs already computed for ``observation``.
        :param str agent_type: Source label of the proposing agent.
        :param int step: 1-based rollout timestep, surfaced in the output as a
            ``_step_<step>`` suffix and as the ``step`` field.
        :param int branch_index: 0-based index of the fanned-out candidate this
            step belongs to, surfaced as the ``branch_index`` field.
        :param bool done: Whether stepping into ``observation`` ended the episode.
        :param int | None env_timestep: Absolute environment time of
            ``observation`` (from :meth:`timestep`). ``step`` and ``env_timestep``
            are different clocks: ``step`` counts rollout timesteps from 1, while
            ``env_timestep`` is where the episode actually stands.
        :return A3SRecommendation: The normalized recommendation.
        """


class Agent(ABC):
    """Agent-specific adapter used by the generic rollout core."""

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Source label attached to this agent's recommendations."""

    @abstractmethod
    def propose(self, observation: ObservationType, n_actions: int) -> list[Any]:
        """
        Proposes candidate native actions for the given observation.

        Used to fan out at the first rollout step so the operator is offered
        several alternative first moves.

        :param ObservationType observation: The current environment observation.
        :param int n_actions: Maximum number of actions to propose.
        :return list: Native actions (rendered later by the environment).
        """

    @abstractmethod
    def act(self, observation: ObservationType) -> Any:
        """
        Chooses a single action for the given observation.

        Used to advance each branch after the first step. Must always return a
        valid action (e.g. a native no-op) even when the policy has no
        suggestion, so the rollout can always proceed.

        :param ObservationType observation: The current environment observation.
        :return Any: A single native action.
        """
