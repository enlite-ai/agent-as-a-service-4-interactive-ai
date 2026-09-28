"""Tests for the T2.1_deep_expert PPO agent adapter.

The policy and its gym spaces are stubbed, so neither Stable-Baselines3, the
~84 MB checkpoint nor a Grid2Op environment is needed here. What is covered is
the part this repository owns: how the policy's ranking is turned into rollout
branches - the diversity rule and cooldown filter on ``propose``, and the
top-k greedy budget on ``act``.
"""
import numpy as np
import pytest

from integrations.powergrid.agent_implementations import t2_1_agent as expert_module
from integrations.powergrid.agent_implementations.t2_1_agent import ExpertPowerGridAgent


class FakeObservation:
    """Observation whose ``simulate`` returns a scripted rho per action."""

    def __init__(self, rho_by_action: dict, cooldowns: dict):
        """
        :param dict rho_by_action: Maps action name to ``(rho, done, info)``.
        :param dict cooldowns: Maps substation id to its remaining cooldown.
        """
        self.rho_by_action = rho_by_action
        self.simulated = []
        self.time_before_cooldown_sub = _Cooldowns(cooldowns)

    def simulate(self, action, time_step=1):
        """Return the scripted outcome for ``action``."""
        self.simulated.append(action.name)
        rho, done, info = self.rho_by_action[action.name]
        simulated = None if rho is None else _Simulated(rho)
        return simulated, 0.0, done, info


class _Cooldowns:
    """Indexable cooldown vector defaulting to "no cooldown"."""

    def __init__(self, by_substation: dict):
        """:param dict by_substation: Non-zero cooldowns, keyed by substation."""
        self.by_substation = by_substation

    def __getitem__(self, substation):
        return self.by_substation.get(substation, 0)


class _Simulated:
    """Minimal stand-in for the observation ``simulate`` hands back."""

    def __init__(self, rho: float):
        """:param float rho: The worst line loading to report."""
        self.rho = np.array([rho])


class FakeAction:
    """Grid2Op action stub exposing only ``impact_on_objects``."""

    def __init__(self, name: str, substation):
        """
        :param str name: Identifier used to script simulation outcomes.
        :param substation: Substation the action assigns a bus on, or ``None``
            for a non-topological action.
        """
        self.name = name
        self.substation = substation

    def impact_on_objects(self):
        """Report the substation this action touches, Grid2Op-style."""
        if self.substation is None:
            return {"topology": {"changed": False, "assigned_bus": []}}
        return {
            "topology": {
                "changed": True,
                "assigned_bus": [{"substation": self.substation}],
            }
        }


class FakeActSpace:
    """Discrete gym action space returning the scripted action per id."""

    def __init__(self, actions: list):
        """:param list actions: Actions indexed by their discrete id."""
        self.actions = actions
        self.n = len(actions)

    def from_gym(self, action_id):
        """Return the action at ``action_id``."""
        return self.actions[action_id]


def _agent(logits: list, actions: list) -> ExpertPowerGridAgent:
    """
    Build an agent wired to a stubbed policy.

    :param list logits: One logit per discrete action id.
    :param list actions: The action each id decodes to.
    :return ExpertPowerGridAgent: Agent ready to ``act``/``propose``.
    """
    agent = ExpertPowerGridAgent(environment=None)
    agent._gym_act_space = FakeActSpace(actions)
    agent._gym_obs_space = type(
        "ObsSpace", (), {"to_gym": staticmethod(lambda obs: np.zeros(3))}
    )()
    agent._logits = lambda gym_obs: np.asarray(logits, dtype=float)
    agent._do_nothing = FakeAction("do_nothing", None)
    # The policy is already "loaded", so nothing tries to read a checkpoint.
    agent._model = object()
    return agent


def test_act_picks_the_best_simulated_of_the_top_ranked_actions():
    """The simulation decides, not the ranking: the top logit need not win."""
    agent = _agent(
        [0.9, 0.8, 0.1],
        [FakeAction("a", 1), FakeAction("b", 2), FakeAction("c", 3)],
    )
    observation = FakeObservation(
        {
            "do_nothing": (1.5, False, {}),
            "a": (1.2, False, {}),
            "b": (0.8, False, {}),
            "c": (1.4, False, {}),
        },
        cooldowns={},
    )

    assert agent.act(observation).name == "b"


def test_act_falls_back_to_do_nothing_when_nothing_improves():
    """A branch keeps running rather than taking an action that makes it worse."""
    agent = _agent([0.9, 0.8], [FakeAction("a", 1), FakeAction("b", 2)])
    observation = FakeObservation(
        {
            "do_nothing": (0.7, False, {}),
            "a": (0.9, False, {}),
            "b": (1.1, False, {}),
        },
        cooldowns={},
    )

    assert agent.act(observation).name == "do_nothing"


def test_act_respects_its_top_k_budget():
    """``act`` mirrors T2.1's own greedy budget instead of scanning everything."""
    count = expert_module._ACT_TOP_K + 25
    agent = _agent(
        list(np.arange(count)[::-1]),
        [FakeAction(f"a{i}", i) for i in range(count)],
    )
    observation = FakeObservation(
        {"do_nothing": (1.5, False, {})}
        | {f"a{i}": (1.4, False, {}) for i in range(count)},
        cooldowns={},
    )

    agent.act(observation)
    # The top-k candidates plus the do-nothing baseline.
    assert agent.simulate_times == expert_module._ACT_TOP_K + 1


def test_act_skips_actions_barred_by_a_cooldown():
    """An action under cooldown is not a choice this timestep, so it is not simulated."""
    agent = _agent([0.9, 0.8], [FakeAction("blocked", 4), FakeAction("free", 5)])
    observation = FakeObservation(
        {
            "do_nothing": (1.5, False, {}),
            "blocked": (0.1, False, {}),
            "free": (1.3, False, {}),
        },
        cooldowns={4: 3},
    )

    assert agent.act(observation).name == "free"
    assert "blocked" not in observation.simulated


def test_propose_returns_actions_on_distinct_substations():
    """Operators are offered different moves, not variants of the same switch."""
    agent = _agent(
        [0.9, 0.8, 0.7, 0.6],
        [
            FakeAction("sub1_a", 1),
            FakeAction("sub1_b", 1),
            FakeAction("sub2", 2),
            FakeAction("sub3", 3),
        ],
    )
    observation = FakeObservation(
        {
            "sub1_a": (1.0, False, {}),
            "sub1_b": (1.0, False, {}),
            "sub2": (1.0, False, {}),
            "sub3": (1.0, False, {}),
        },
        cooldowns={},
    )

    names = [action.name for action in agent.propose(observation, 3)]
    assert names == ["sub1_a", "sub2", "sub3"]


def test_propose_drops_candidates_it_cannot_simulate():
    """A branch is never opened on an action that fails, diverges or ends the episode."""
    agent = _agent(
        [0.9, 0.8, 0.7],
        [FakeAction("dead", 1), FakeAction("illegal", 2), FakeAction("ok", 3)],
    )
    observation = FakeObservation(
        {
            "dead": (None, True, {}),
            "illegal": (0.5, False, {"is_illegal": True}),
            "ok": (1.0, False, {}),
        },
        cooldowns={},
    )

    assert [a.name for a in agent.propose(observation, 3)] == ["ok"]


def test_propose_falls_back_to_do_nothing_when_nothing_is_applicable():
    """The rollout core always gets at least one branch to project."""
    agent = _agent([0.9], [FakeAction("dead", 1)])
    observation = FakeObservation(
        {"dead": (None, True, {})}, cooldowns={}
    )

    assert [a.name for a in agent.propose(observation, 3)] == ["do_nothing"]


def test_unknown_agent_name_fails_at_startup(monkeypatch):
    """A typo in A3S_POWERGRID_AGENT must not silently serve the default."""
    from integrations.powergrid.use_case import PowerGridUseCase

    monkeypatch.setenv("A3S_POWERGRID_AGENT", "t2")
    with pytest.raises(ValueError, match="Unknown A3S_POWERGRID_AGENT"):
        PowerGridUseCase().build_agent(environment=None)
