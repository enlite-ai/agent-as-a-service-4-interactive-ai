"""PowerGrid implementation of the generic Agent interface.

Wraps the Grid2Op "assistant" submission (a ``grid2op.Agent.BaseAgent`` loaded
from the configured submission folder) behind the generic :class:`Agent`
interface, so the recommendation core stays agnostic to how actions are
produced.

The two things the rollout core asks for are deliberately served by *different*
code paths, because they answer different questions (see :meth:`PowerGridAgent.propose`
and :meth:`PowerGridAgent.act`): the operator-facing alternatives come from the
assistant's full planner, while the greedy continuation of a branch comes from a
cheap one-step policy.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
from grid2op.Agent import BaseAgent

from ..environment import PowerGridEnvironment
from ..simulation import UNUSABLE_RHO, simulated_rho
from a3s_core import Agent

# How many *usable* candidates the greedy continuation policy wants before it
# settles (see :meth:`PowerGridAgent.act`). A candidate is usable when it can
# actually be simulated; ones the network ranks well but that turn out to be
# illegal or to end the episode do not count towards this, because on exactly
# the states that matter - a forecast cascade - the whole top of the ranking is
# unusable, and a policy that gave up there would let the branch die.
_GREEDY_USABLE_CANDIDATES = 8

# How far down the ranking the search may go looking for those candidates. Bounds
# the cost of a state where almost nothing is simulatable; the assistant's own
# planner scans 150 actions at its first depth, so this stays well below it.
_GREEDY_SCAN_LIMIT = 60


def lazy_import_package(package_name: str, package_path: str):
    """Import a package from a filesystem path without installing it.

    :param package_name: Module name to register in `sys.modules`.
    :param package_path: Directory containing the target `__init__.py`.
    :return: Imported module object.
    """
    spec = importlib.util.spec_from_file_location(
        package_name, os.path.join(package_path, "__init__.py")
    )
    if spec and spec.loader:
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        spec.loader.exec_module(package)
        return package
    raise ImportError(
        f"Cannot import package {package_name} from {package_path}"
    )


class PowerGridAgent(Agent):
    """Grid2Op assistant wrapped as a generic recommendation agent."""

    def __init__(self, environment: PowerGridEnvironment):
        """Bind the agent to the environment that hosts its Grid2Op env.

        The assistant is Grid2Op-specific and is built from the environment's
        Grid2Op instance; it is loaded lazily on first use.

        :param environment: The PowerGrid environment providing the Grid2Op env.
        :return: None.
        """
        self._environment = environment
        self._assistant = None
        self._do_nothing = None

    @property
    def agent_type(self) -> str:
        """Source label for this agent's recommendations."""
        return "IA"

    def _ensure_assistant(self) -> None:
        """Load the Grid2Op assistant submission lazily.

        :return: None.
        """
        if self._assistant is not None:
            return
        env_copy, submission_path, assistant_seed = (
            self._environment.assistant_setup()
        )
        submission = lazy_import_package(
            "powergrid_a3s_submission", submission_path
        )
        assistant = submission.make_agent(env_copy, submission_path)
        if not isinstance(assistant, BaseAgent):
            raise RuntimeError(
                "Your assistant must be a grid2op.Agent.BaseAgent"
            )
        assistant.seed(assistant_seed)
        self._assistant = assistant

    def propose(self, observation, n_actions: int) -> list:
        """Propose candidate Grid2Op actions for the current observation.

        This is the operator-facing path: the actions returned here are the ones
        shown as alternatives, so it uses the assistant's full planner (deep and
        diversity-filtered) despite its cost. Advancing a branch afterwards does
        not, see :meth:`act`.

        :param observation: The current Grid2Op observation.
        :param n_actions: Maximum number of actions to propose.
        :return: List of Grid2Op actions.
        """
        self._ensure_assistant()
        recommendations = self._assistant.make_recommandations(
            observation, n_actions
        )
        return [action for action, _ in recommendations]

    def act(self, observation):
        """Choose a single Grid2Op action to advance a rollout branch.

        This is *not* the operator-facing recommendation path: it only has to
        keep a branch moving so the KPI projection has a step 2..n. It therefore
        uses a cheap one-step greedy policy: rank all unitary actions with the
        assistant's own neural network, simulate the best of them, and keep the
        one that beats doing nothing (see :meth:`_greedy_action`).

        Previously this was implemented as ``propose(observation, 1)``, i.e. the
        assistant's full ``make_recommandations`` planner (~150 simulated actions
        at depth 0, then 50 candidates x 50 expansions per level up to depth 4,
        followed by diversity filtering). That cost ~1200 power flows and up to
        ~4.6s *per continuation step*, and almost all of it was thrown away: the
        planner's depth is pointless here (only its first action is ever applied,
        and we re-plan at the next step anyway) and its diversity filtering is
        meaningless when a single action is wanted. With the frontend asking for a
        6-step projection over 3 branches, those 15 calls were about half of a
        ~9.5s request; they now cost ~5 power flows each.

        The trade-off is what the projection *assumes about the future*: the curve
        now reads "the agent keeps picking its best immediate move" rather than
        "a full 4-step planner is re-run at every future timestep" - the less
        optimistic, and more faithful, of the two readings.

        :param observation: The current Grid2Op observation.
        :return: A single Grid2Op action.
        """
        self._ensure_assistant()
        greedy = self._greedy_action(observation)
        if greedy is not None:
            return greedy
        return self._do_nothing_action()

    def _do_nothing_action(self):
        """Return the assistant's do-nothing action, built once and reused.

        :return: A Grid2Op do-nothing action.
        """
        if self._do_nothing is None:
            self._do_nothing = self._assistant.action_space({})
        return self._do_nothing

    def _greedy_action(self, observation):
        """Pick the best simulatable of the network's top-ranked actions.

        Ranks every unitary action with the assistant's network (one forward
        pass), then walks down that ranking simulating candidates until
        ``_GREEDY_USABLE_CANDIDATES`` of them have produced a usable outcome (or
        ``_GREEDY_SCAN_LIMIT`` actions have been looked at), and returns the one
        with the lowest resulting worst line loading. ``None`` means nothing
        usable improved on doing nothing, so the caller should let the grid run.

        Counting *usable* outcomes rather than ranking positions is what keeps
        this honest on the states that matter. When the grid is already heading
        for a cascade, the network's best-ranked actions are typically the ones
        that end the episode in simulation, and doing nothing does too - so a
        fixed top-N budget would find nothing, report "no improvement", and let
        the branch die a step later while a genuinely stabilising action sat at
        rank ~50. On a healthy grid the first candidates simulate fine and the
        search stops almost immediately, which is the common case.

        :param observation: The current Grid2Op observation.
        :return: A Grid2Op action, or ``None`` when do-nothing is at least as good.
        """
        # Doing nothing is the baseline to beat - and it is itself unusable when
        # the grid is forecast to collapse, in which case any action that merely
        # survives is an improvement.
        best_rho = self._simulated_rho(observation, self._do_nothing_action())
        best_action = None

        # `get_action_from_index` reads the per-substation topology of the
        # observation being acted on, so it has to be refreshed here.
        self._assistant.sub_topo_dict = self._assistant.calc_sub_topo_dict(
            observation
        )
        predicted_rho = self._assistant.unitary_es_agent(observation)[0, :]
        usable = 0
        for index in np.argsort(predicted_rho)[:_GREEDY_SCAN_LIMIT]:
            action = self._assistant.get_action_from_index(
                observation, int(index)
            )
            # Illegal under the current cooldowns, so not applicable this step.
            if action is None:
                continue
            rho = self._simulated_rho(observation, action)
            if rho == UNUSABLE_RHO:
                continue
            usable += 1
            if rho < best_rho:
                best_rho, best_action = rho, action
            if usable >= _GREEDY_USABLE_CANDIDATES:
                break
        return best_action

    def _simulated_rho(self, observation, action) -> float:
        """Worst line loading one timestep after ``action``, or a penalty.

        :param observation: The observation to simulate from.
        :param action: The Grid2Op action to simulate.
        :return float: Simulated worst line loading, or the penalty value.
        """
        return simulated_rho(
            observation, action, self._count_simulation
        )

    def _count_simulation(self) -> None:
        """Record one power flow against the assistant's own counter.

        :return: None.
        """
        self._assistant.simulate_times += 1
