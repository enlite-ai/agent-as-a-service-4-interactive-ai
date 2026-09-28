"""Tests for the PowerGrid agent's greedy branch-continuation policy.

The operator-facing ``propose`` path is the assistant's own planner and is
covered end-to-end by the smoke test. What is tested here is ``act``, the cheap
one-step policy that advances a rollout branch: that it picks the best of the
network's top-ranked actions, that it never picks one it could not simulate, and
that it stays within its simulation budget (the reason it exists at all).

The assistant is stubbed, so no Grid2Op environment is built.
"""
import numpy as np

from integrations.powergrid.agent_implementations import xd_agent as agent_module
from integrations.powergrid.agent_implementations.xd_agent import PowerGridAgent


class FakeObservation:
    """Observation whose ``simulate`` returns a scripted rho per action."""

    def __init__(self, rho_by_action):
        self.rho_by_action = rho_by_action
        self.simulated = []

    def simulate(self, action, time_step=1):
        """Return the scripted outcome for ``action``."""
        self.simulated.append(action)
        rho, done, info = self.rho_by_action[action]
        simulated = None if rho is None else FakeSimulated(rho)
        return simulated, 0.0, done, info


class FakeSimulated:
    """Minimal stand-in for the observation ``simulate`` hands back."""

    def __init__(self, rho):
        self.rho = np.array([rho])


class FakeAssistant:
    """Assistant exposing only the surface the greedy policy uses."""

    def __init__(self, ranking, actions):
        self.ranking = np.asarray(ranking, dtype=float)
        self.actions = actions
        self.simulate_times = 0
        self.sub_topo_dict = None
        self.action_space = lambda _: "do_nothing"

    def calc_sub_topo_dict(self, observation):
        """Record that the topology was refreshed for this observation."""
        return {"for": observation}

    def unitary_es_agent(self, observation):
        """Return the predicted rho of every unitary action."""
        return self.ranking[None, :]

    def get_action_from_index(self, observation, index):
        """Return the (possibly illegal, hence ``None``) action at ``index``."""
        return self.actions[index]


def _agent(ranking, actions):
    """Build an agent wired to a stubbed assistant.

    :param ranking: Predicted rho per unitary action index.
    :param actions: Action returned per unitary action index.
    :return: ``(agent, assistant)``.
    """
    powergrid_agent = PowerGridAgent(environment=None)
    assistant = FakeAssistant(ranking, actions)
    powergrid_agent._assistant = assistant
    return powergrid_agent, assistant


def test_act_picks_the_best_simulated_of_the_top_ranked_actions():
    """The chosen action is the one with the lowest simulated worst loading."""
    powergrid_agent, _ = _agent([0.1, 0.2, 0.9], ["a", "b", "c"])
    observation = FakeObservation(
        {
            "do_nothing": (1.5, False, {}),
            "a": (1.2, False, {}),
            "b": (0.8, False, {}),
            "c": (1.4, False, {}),
        }
    )

    # The network's ranking only selects which actions are worth simulating; the
    # simulation decides, so the best-ranked action ("a") does not automatically win.
    assert powergrid_agent.act(observation) == "b"


def test_act_falls_back_to_do_nothing_when_nothing_improves():
    """A branch keeps running on do-nothing rather than taking a worse action."""
    powergrid_agent, _ = _agent([0.1, 0.2], ["a", "b"])
    observation = FakeObservation(
        {
            "do_nothing": (0.7, False, {}),
            "a": (0.9, False, {}),
            "b": (1.1, False, {}),
        }
    )

    assert powergrid_agent.act(observation) == "do_nothing"


def test_act_ignores_unusable_and_illegal_candidates():
    """Illegal, ambiguous, diverged and episode-ending actions never win."""
    powergrid_agent, _ = _agent([0.1, 0.2, 0.3, 0.4, 0.5], [None, "i", "d", "g", "ok"])
    observation = FakeObservation(
        {
            "do_nothing": (1.4, False, {}),
            "i": (0.1, False, {"is_illegal": True}),
            "d": (None, False, {}),
            "g": (0.1, True, {}),
            "ok": (1.3, False, {}),
        }
    )

    assert powergrid_agent.act(observation) == "ok"
    # The action the assistant reported as illegal for this step is not simulated.
    assert None not in observation.simulated


def test_act_settles_once_enough_candidates_simulate():
    """On a healthy grid the search stops as soon as its budget is satisfied."""
    count = 40
    powergrid_agent, assistant = _agent(
        np.arange(count) / count, [f"a{i}" for i in range(count)]
    )
    observation = FakeObservation(
        {"do_nothing": (1.5, False, {})}
        | {f"a{i}": (1.4, False, {}) for i in range(count)}
    )

    powergrid_agent.act(observation)
    assert (
        assistant.simulate_times
        == agent_module._GREEDY_USABLE_CANDIDATES + 1  # + the do-nothing baseline
    )


def test_act_keeps_looking_when_the_top_of_the_ranking_is_unusable():
    """Regression: a forecast cascade must not degrade into doing nothing.

    On a grid heading for a collapse, doing nothing *and* the network's
    best-ranked actions all end the episode in simulation. Budgeting by ranking
    position found nothing there, reported "no improvement", and let the branch
    die a step later - while an action that actually stabilises the grid sat far
    down the ranking. The budget therefore counts usable outcomes.
    """
    count = 30
    # Everything the network likes ends the episode; only the last action holds.
    actions = [f"dead{i}" for i in range(count - 1)] + ["stabilising"]
    powergrid_agent, _ = _agent(np.arange(count) / count, actions)
    observation = FakeObservation(
        {"do_nothing": (None, True, {})}
        | {f"dead{i}": (None, True, {}) for i in range(count - 1)}
        | {"stabilising": (0.95, False, {})}
    )

    assert powergrid_agent.act(observation) == "stabilising"


def test_act_bounds_how_far_down_the_ranking_it_looks():
    """The scan is bounded, so an unsalvageable state cannot cost the planner's budget."""
    count = agent_module._GREEDY_SCAN_LIMIT + 20
    powergrid_agent, assistant = _agent(
        np.arange(count) / count, [f"a{i}" for i in range(count)]
    )
    observation = FakeObservation(
        {"do_nothing": (None, True, {})}
        | {f"a{i}": (None, True, {}) for i in range(count)}
    )

    assert powergrid_agent.act(observation) == "do_nothing"
    assert (
        assistant.simulate_times == agent_module._GREEDY_SCAN_LIMIT + 1
    )
