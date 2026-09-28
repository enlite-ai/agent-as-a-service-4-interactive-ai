# PowerGrid agent backed by the T2.1_deep_expert PPO policy, offered as an
# alternative to the XD_silly_repo assistant so the two can be compared on the
# same rollouts. Selected at startup by A3S_POWERGRID_AGENT (see api/utils.py).
"""Adapter around the T2.1_deep_expert PPO policy.

This wires *only* the PPO policy head of ``ExpertAgentRL``: an observation is
reduced to its ``rho`` vector, run through the trained network, and the action
logits give a ranking over the 603 discrete actions the policy was trained on.
Candidates are then validated by simulation exactly the way the XD agent's
candidates are, so the comparison between the two is about the ranking each
network produces rather than about one of them proposing inapplicable actions.

What is deliberately *not* wired, and why the comparison is a comparison of
policies rather than of the two deployed services:

* the ``RecoPowerlineModule`` / ``RecoverInitTopoModule`` heuristics and the
  convex ``OptimModule`` that ``ExpertAgentRL.act`` runs around the policy. They
  live in ``LJNAgent``, an external git dependency of T2.1_deep_expert that is
  not vendored here, and the optimiser is the part of that agent whose cost is
  hardest to bound. Without them this agent cannot propose redispatch at all:
  its action space is ``set_bus`` only.
* T2.1's own ``predict``, which returns raw top-k logits with no simulation, no
  cooldown check and no diversity rule. Feeding that straight into the rollout
  would let branches die on inapplicable actions and would say nothing useful
  about the policy.

The policy is loaded lazily on first use, so selecting the XD agent never pays
for Stable-Baselines3 or for reading the ~84 MB checkpoint, even though both ship
in the image.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np

from ..environment import PowerGridEnvironment
from ..simulation import (
    UNUSABLE_RHO,
    action_substation,
    simulated_rho,
)
from a3s_core import Agent

logger = logging.getLogger(__name__)

# The trained PPO checkpoint and the action list its output layer was trained
# against. Both are vendored under `resources/PowerGrid/T2.1_deep_expert`, in the
# layout they have in the T2.1_deep_expert repository, so selecting this agent
# needs nothing mounted into the container - the image is self-contained. They are
# a *copy*: the action list must stay the exact file the policy was trained with,
# so it is never regenerated, only replaced wholesale together with the
# checkpoint (a mismatch is refused at load time - see `_build_spaces`).
_T21_RESOURCE_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "resources",
        "PowerGrid",
        "T2.1_deep_expert",
    )
)
_MODEL_PATH = os.environ.get(
    "A3S_T21_MODEL_PATH",
    os.path.join(_T21_RESOURCE_DIR, "PPO_SB3", "model", "PPO_SB3.zip"),
)
_ACTION_SPACE_PATH = os.environ.get(
    "A3S_T21_ACTION_SPACE_PATH",
    os.path.join(
        _T21_RESOURCE_DIR, "ExpertAgent", "assets", "whole_action_space.npz"
    ),
)

# Observation attributes the policy was trained on. Must match
# `obs_attr_to_keep` in T2.1_deep_expert/app/main.py: the network's input layer
# is sized to exactly this.
_OBS_ATTR_TO_KEEP = ["rho"]

# How far down the policy's ranking each entry point may simulate. `act` mirrors
# T2.1's own `top_k=20` greedy budget (ExpertAgentRL.__init__), which is the
# behaviour being evaluated. `propose` is allowed to look further because it has
# to find candidates on *distinct* substations, and the top of the ranking is
# often several variants of the same switch.
_ACT_TOP_K = int(os.environ.get("A3S_T21_ACT_TOP_K", "20"))
_PROPOSE_SCAN_LIMIT = int(os.environ.get("A3S_T21_PROPOSE_SCAN_LIMIT", "60"))


class ExpertPowerGridAgent(Agent):
    """T2.1_deep_expert's PPO policy behind the generic Agent interface."""

    def __init__(self, environment: PowerGridEnvironment):
        """
        Bind the agent to the environment hosting its Grid2Op env.

        The policy and its gym spaces are built from that environment and are
        loaded lazily on first use.

        :param PowerGridEnvironment environment: The environment providing the
            Grid2Op env the policy will act on.
        """
        self._environment = environment
        self._env = None
        self._model = None
        self._gym_obs_space = None
        self._gym_act_space = None
        self._action_space = None
        self._do_nothing = None
        self.simulate_times = 0

    @property
    def agent_type(self) -> str:
        """
        Source label attached to this agent's recommendations.

        :return str: The label the rollout core stamps on every recommendation.
        """
        return "IA"

    def propose(self, observation: Any, n_actions: int) -> list:
        """
        Propose up to ``n_actions`` alternative first actions.

        Walks the policy's ranking and keeps the best-ranked candidates that are
        both simulatable and act on *distinct* substations, so the operator is
        offered genuinely different moves rather than variants of one switch.
        Unlike the XD planner this is a single forward pass plus at most
        ``_PROPOSE_SCAN_LIMIT`` power flows, with no multi-step lookahead: it
        cannot find a rescue that needs two actions in sequence.

        :param Any observation: The current Grid2Op observation.
        :param int n_actions: Maximum number of actions to propose.
        :return list: Grid2Op actions, at most ``n_actions`` of them.
        """
        self._ensure_policy()
        proposals: list = []
        used_substations: set = set()
        for action in self._ranked_actions(observation, _PROPOSE_SCAN_LIMIT):
            if simulated_rho(observation, action, self._count) == UNUSABLE_RHO:
                continue
            substation = action_substation(action)
            # A non-topological action has no substation to collide on, so it is
            # always a distinct alternative.
            if substation is not None:
                if substation in used_substations:
                    continue
                used_substations.add(substation)
            proposals.append(action)
            if len(proposals) == n_actions:
                break
        # The rollout core needs at least one branch to project; letting the grid
        # run is the honest fallback when nothing is applicable.
        if not proposals:
            return [self._do_nothing_action()]
        return proposals

    def act(self, observation: Any) -> Any:
        """
        Choose a single action to advance a rollout branch.

        Reproduces the greedy step of ``ExpertAgentRL.act``: simulate the
        policy's top ``_ACT_TOP_K`` actions and keep the one reaching the lowest
        worst line loading, provided it beats doing nothing.

        :param Any observation: The current Grid2Op observation.
        :return Any: A single Grid2Op action, possibly the do-nothing action.
        """
        self._ensure_policy()
        best_rho = simulated_rho(
            observation, self._do_nothing_action(), self._count
        )
        best_action = None
        for action in self._ranked_actions(observation, _ACT_TOP_K):
            rho = simulated_rho(observation, action, self._count)
            if rho < best_rho:
                best_rho, best_action = rho, action
        if best_action is None:
            return self._do_nothing_action()
        return best_action

    # -- Internals -------------------------------------------------------------

    def _ranked_actions(self, observation: Any, limit: int):
        """
        Yield the policy's best-ranked Grid2Op actions, best first.

        Actions the policy ranks highly but that are barred by a cooldown, or
        that the gym space cannot render, are skipped rather than counted
        against ``limit``: they are not choices available this timestep.

        :param Any observation: The Grid2Op observation to rank actions for.
        :param int limit: How many ranked action ids to consider.
        :return: Generator of Grid2Op actions.
        """
        gym_obs = self._gym_obs_space.to_gym(observation)
        logits = self._logits(gym_obs)
        yielded = 0
        # argsort is ascending and the best action is the highest logit.
        for action_id in np.argsort(logits)[::-1]:
            if yielded >= limit:
                break
            try:
                action = self._gym_act_space.from_gym(int(action_id))
            except BaseException:
                continue
            if self._blocked_by_cooldown(observation, action):
                continue
            yielded += 1
            yield action

    def _logits(self, gym_obs: np.ndarray) -> np.ndarray:
        """
        Run one forward pass and return the policy's action logits.

        :param np.ndarray gym_obs: The gym-encoded observation.
        :return np.ndarray: One logit per discrete action.
        """
        import torch

        with torch.no_grad():
            batch = torch.from_numpy(gym_obs).reshape((1, len(gym_obs))).float()
            distribution = self._model.policy.get_distribution(batch)
            return distribution.distribution.logits.cpu().numpy()[0]

    @staticmethod
    def _blocked_by_cooldown(observation: Any, action: Any) -> bool:
        """
        Whether a substation cooldown makes this action illegal right now.

        Checked before simulating because it is a cheap, certain rejection: the
        policy's ranking knows nothing about cooldowns.

        :param Any observation: The current Grid2Op observation.
        :param Any action: The Grid2Op action to check.
        :return bool: ``True`` when the action cannot legally be taken.
        """
        substation = action_substation(action)
        if substation is None:
            return False
        return bool(observation.time_before_cooldown_sub[int(substation)] != 0)

    def _count(self) -> None:
        """
        Record one power flow, mirroring the XD assistant's own counter.

        :return: None.
        """
        self.simulate_times += 1

    def _do_nothing_action(self) -> Any:
        """
        Return the do-nothing action, built once and reused.

        :return Any: A Grid2Op do-nothing action.
        """
        if self._do_nothing is None:
            self._do_nothing = self._action_space({})
        return self._do_nothing

    def _ensure_policy(self) -> None:
        """
        Load the PPO checkpoint and rebuild its gym spaces, once.

        The gym spaces are reconstructed here rather than unpickled from the
        checkpoint so that they are bound to *this* service's Grid2Op env. Both
        grids are the same 519-dimension network, but rebuilding makes that an
        assertion rather than an assumption.

        :return: None.
        :raises RuntimeError: If a dependency or asset is missing, or if the
            checkpoint does not match the action list it is loaded with.
        """
        if self._model is not None:
            return
        self._build_spaces()
        self._model = self._load_model()
        self._check_head_matches_action_space()
        logger.info(
            "Loaded T2.1 PPO policy from %s (%d actions)",
            _MODEL_PATH,
            self._gym_act_space.n,
        )

    def _build_spaces(self) -> None:
        """
        Build the gym observation and action spaces the policy expects.

        The Grid2Op env they are derived from is retained on the instance: the
        spaces hold references into it, so letting it go out of scope would be
        borrowing trouble.

        :return: None.
        :raises RuntimeError: If the action list asset is missing.
        """
        from grid2op.gym_compat import BoxGymObsSpace

        # T2.1 builds its action space with the Gymnasium variant; fall back to
        # the auto-selected one on a Grid2Op build that does not expose it.
        try:
            from grid2op.gym_compat import DiscreteActSpaceGymnasium as DiscreteActSpace
        except ImportError:
            from grid2op.gym_compat import DiscreteActSpace

        env, _, assistant_seed = self._environment.assistant_setup()
        env.seed(assistant_seed)
        if not os.path.exists(_ACTION_SPACE_PATH):
            raise RuntimeError(
                f"T2.1 action list not found at {_ACTION_SPACE_PATH}. It ships "
                "under resources/PowerGrid/T2.1_deep_expert; set "
                "A3S_T21_ACTION_SPACE_PATH to use a copy elsewhere."
            )
        action_list = np.load(_ACTION_SPACE_PATH, allow_pickle=True)[
            "action_space"
        ]
        self._env = env
        self._action_space = env.action_space
        self._gym_obs_space = BoxGymObsSpace(
            env.observation_space, attr_to_keep=_OBS_ATTR_TO_KEEP
        )
        self._gym_act_space = DiscreteActSpace(
            env.action_space, action_list=action_list
        )

    def _load_model(self) -> Any:
        """
        Load the PPO checkpoint onto the CPU.

        :return Any: The loaded Stable-Baselines3 PPO model.
        :raises RuntimeError: If Stable-Baselines3 or the checkpoint is missing.
        """
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise RuntimeError(
                "The T2.1 agent needs stable-baselines3, which the image "
                "installs from pyproject.toml's t2-1 extra; rebuild the "
                "image, or select A3S_POWERGRID_AGENT=xd."
            ) from exc
        if not os.path.exists(_MODEL_PATH):
            raise RuntimeError(
                f"T2.1 PPO checkpoint not found at {_MODEL_PATH}. It ships "
                "under resources/PowerGrid/T2.1_deep_expert; set "
                "A3S_T21_MODEL_PATH to use a copy elsewhere."
            )
        # The checkpoint was trained on a newer stack than this service pins
        # (numpy 2.2 / SB3 2.7 vs. numpy 1.24 / SB3 2.2 — the numpy pin has to
        # match the simulator for the replayed power flow to be identical), so
        # some of its cloudpickled entries cannot be unpickled here: the numpy
        # ones reference `numpy._core`, which 1.x does not have, and the
        # schedules are `SB3.common.utils.FloatSchedule`, which 2.2 does not
        # have. Every one of them is training state that inference never reads,
        # so they are replaced rather than deserialized. The spaces are replaced
        # regardless — see `_ensure_policy`. The policy weights themselves are
        # plain tensors and load unchanged; `_check_head_matches_action_space`
        # is what verifies they are the pair we think they are.
        return PPO.load(
            _MODEL_PATH,
            device="cpu",
            custom_objects={
                "observation_space": self._gym_obs_space,
                "action_space": self._gym_act_space,
                "lr_schedule": lambda _: 0.0,
                "clip_range": lambda _: 0.0,
                "_last_obs": None,
                "_last_episode_starts": None,
            },
        )

    def _check_head_matches_action_space(self) -> None:
        """
        Fail loudly when the checkpoint and the action list disagree.

        A silent mismatch here is the worst failure mode available: every action
        id would still resolve, just to a different action than the one the
        policy meant, and the comparison would quietly measure noise.

        :return: None.
        :raises RuntimeError: If the policy's output layer is not the size of the
            action list.
        """
        head = self._model.policy.action_net.out_features
        expected = int(self._gym_act_space.n)
        if head != expected:
            raise RuntimeError(
                f"T2.1 checkpoint has a {head}-action head but "
                f"{_ACTION_SPACE_PATH} defines {expected} actions; they are not "
                "the pair the policy was trained with."
            )
