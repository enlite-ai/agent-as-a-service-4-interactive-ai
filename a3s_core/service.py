"""Generic, environment- and agent-agnostic multi-step rollout core.

This is the reusable heart of the Agent-as-a-Service. Given a request carrying a
serialized environment state, it:

1. reconstructs the environment to that state (:meth:`Environment.prepare`);
2. fans out at the first timestep into several alternative first actions
   (:meth:`Agent.propose`), so the operator is offered distinct options;
3. rolls each alternative forward on an independent environment branch
   (:meth:`Environment.fork` + :meth:`step`): the alternative is applied once,
   and then the agent's policy is followed for the next ``n_steps`` timesteps
   (:meth:`Agent.act`), so a branch yields ``n_steps + 1`` timesteps;
4. emits one normalized recommendation per (branch, timestep), tagged with a
   ``_step_<k>`` suffix and carrying whether that step ended the episode
   (``done``) plus the environment's own absolute clock (``env_timestep``).

The core knows nothing about any concrete environment or agent: branching,
stepping, KPIs and rendering are all delegated through the interfaces. New use
cases plug in by implementing :class:`Environment` and :class:`Agent`; the core
stays untouched.

One service instance owns one environment, and positioning that environment
(:meth:`Environment.prepare`) mutates it, so a request is served under a lock:
the WSGI server handles requests on threads, and two overlapping rollouts would
otherwise reset and step the same environment out from under each other.
"""
from __future__ import annotations

import threading

from .interface import (
    A3SRecommendation,
    A3SRecommendationRequest,
    Agent,
    AgentAsAService,
    Environment,
)


class RecommendationService(AgentAsAService):
    """Roll out an :class:`Environment` under an :class:`Agent` to produce recos."""

    def __init__(self, environment: Environment, agent: Agent):
        """Wire a concrete environment and agent into the generic rollout.

        :param environment: Environment adapter for the target use case.
        :param agent: Agent adapter for the target use case.
        :return: None.
        """
        self._environment = environment
        self._agent = agent
        # Serializes access to the shared, mutable environment (see module docs).
        self._lock = threading.Lock()

    def get_recommendations(
        self, request: A3SRecommendationRequest
    ) -> list[A3SRecommendation]:
        """Produce multi-step rollout recommendations for the requested state.

        :param request: Normalized recommendation request. ``max_recommendations``
            is the number of alternative first actions to fan out; ``n_steps``
            is how many timesteps the agent's policy is followed *after* each
            alternative, so each branch yields ``n_steps + 1`` timesteps.
        :return: Up to ``max_recommendations`` x (``n_steps`` + 1) recommendations
            (fewer when a branch terminates early or no state is available).
            Every one carries ``done`` and ``env_timestep``; a branch that
            terminates has exactly one ``done`` recommendation, its last.
        """
        with self._lock:
            return self._get_recommendations(request)

    def _get_recommendations(
        self, request: A3SRecommendationRequest
    ) -> list[A3SRecommendation]:
        """Produce the recommendations, with exclusive use of the environment.

        :param request: Normalized recommendation request.
        :return: The recommendations for this request.
        """
        root_observation = self._environment.prepare(request.environment_state)
        if root_observation is None:
            return []

        # `n_steps` is the projection depth *beyond* the recommended action, so 0
        # is meaningful: it asks for the action's own outcome and nothing further.
        n_steps = max(0, int(request.n_steps))
        first_actions = self._agent.propose(
            root_observation, request.max_recommendations
        )

        recommendations: list[A3SRecommendation] = []
        for branch_index, first_action in enumerate(first_actions):
            recommendations.extend(
                self._rollout_branch(first_action, n_steps, branch_index)
            )
        return recommendations

    def _rollout_branch(
        self, first_action, n_steps: int, branch_index: int
    ) -> list[A3SRecommendation]:
        """Apply one alternative, then follow the policy for ``n_steps`` steps.

        The alternative is applied once; from there the agent's policy is followed
        for ``n_steps`` further timesteps, each chosen from the observation reached
        so far. So the branch spans ``n_steps + 1`` timesteps, of which the first
        is the operator-facing recommendation and the rest are the projection of
        what the agent would do next.

        :param first_action: The alternative first action for this branch.
        :param n_steps: How many policy timesteps to follow after that action.
        :param branch_index: 0-based index of this candidate among the fanned-
            out first actions, tagged on every emitted recommendation.
        :return: One recommendation per rolled-out timestep of this branch
            (``n_steps + 1`` of them, fewer when the episode ends first).
        """
        # Fork so stepping this branch never disturbs the prepared environment
        # (or any sibling branch); the environment picks its cheap branching.
        branch = self._environment.fork()
        agent_type = self._agent.agent_type

        recommendations: list[A3SRecommendation] = []
        # The recommended action plus the policy steps that follow it.
        total_steps = n_steps + 1
        action = first_action
        for step in range(1, total_steps + 1):
            observation, done = branch.step(action)
            kpis = branch.kpis(observation)
            recommendations.append(
                branch.format_recommendation(
                    observation,
                    action,
                    kpis,
                    agent_type,
                    step,
                    branch_index,
                    done=done,
                    # The branch is the environment that advanced, so it is the
                    # one holding the clock for this observation.
                    env_timestep=branch.timestep(observation),
                )
            )
            if done:
                break
            # Asking the policy for an action after the final step would throw
            # it away, and for some agents that discarded call is a real cost.
            if step < total_steps:
                action = self._agent.act(observation)
        return recommendations
