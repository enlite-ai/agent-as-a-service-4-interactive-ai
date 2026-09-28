"""Tests for the generic, environment/agent-agnostic rollout core.

These use trivial fake Environment/Agent implementations to prove the rollout
orchestration works without any concrete environment (no Grid2Op), which is the
whole point of the design: a new use case only implements the interfaces.
"""
from a3s_core import RecommendationService
from a3s_core import (
    A3SRecommendation,
    A3SRecommendationRequest,
    Agent,
    Environment,
)


# Offset between the fake env's own clock and the rollout step counter, so a test
# asserting on `env_timestep` cannot pass by accidentally reading `step`.
CLOCK_OFFSET = 500


class FakeEnvironment(Environment):
    """Counter-based environment: each step advances an independent branch."""

    def __init__(self, observation, max_steps=None):
        self.observation = observation
        self.max_steps = max_steps
        self.steps_taken = 0
        self.step_actions = []
        self.children = []

    def prepare(self, environment_state):
        """Return the preconfigured root observation (or None)."""
        return self.observation

    def step(self, action):
        """Advance one step; terminate when ``max_steps`` is reached."""
        self.steps_taken += 1
        self.step_actions.append(action)
        done = self.max_steps is not None and self.steps_taken >= self.max_steps
        return {"t": self.steps_taken, "after": action}, done

    def fork(self):
        """Return an independent branch, recorded on the parent for assertions."""
        clone = FakeEnvironment(self.observation, self.max_steps)
        self.children.append(clone)
        return clone

    def kpis(self, observation):
        """Expose the reached timestep as the KPI."""
        return {"t": observation["t"]}

    def timestep(self, observation):
        """Report the branch's own clock, offset so it is not the rollout step."""
        return CLOCK_OFFSET + observation["t"]

    def format_recommendation(
        self, observation, action, kpis, agent_type, step, branch_index,
        *, done, env_timestep,
    ):
        """Render an action into a recommendation carrying the step suffix."""
        return A3SRecommendation(
            title=f"action-{action}_step_{step}",
            description="",
            actions=[{}],
            agent_type=agent_type,
            kpis=kpis,
            branch_index=branch_index,
            step=step,
            done=done,
            env_timestep=env_timestep,
        )


class FakeAgent(Agent):
    """Agent that fans out fixed first actions then acts greedily."""

    def __init__(self, first_actions, greedy="G", agent_type="IA"):
        self.first_actions = first_actions
        self.greedy = greedy
        self._agent_type = agent_type
        self.proposed_with = None
        self.act_observations = []

    @property
    def agent_type(self):
        """Source label for this fake agent."""
        return self._agent_type

    def propose(self, observation, n_actions):
        """Return up to ``n_actions`` preconfigured first actions."""
        self.proposed_with = (observation, n_actions)
        return self.first_actions[:n_actions]

    def act(self, observation):
        """Return the single greedy action, recording the observation."""
        self.act_observations.append(observation)
        return self.greedy


def _request(max_recommendations=3, n_steps=0):
    return A3SRecommendationRequest(
        use_case="Fake",
        environment_state=None,
        max_recommendations=max_recommendations,
        n_steps=n_steps,
    )


def test_core_returns_empty_when_no_observation():
    """Ensure the core returns nothing when the env has no state to act on."""
    service = RecommendationService(FakeEnvironment(None), FakeAgent([1, 2]))
    assert service.get_recommendations(_request()) == []


def test_rollout_fans_out_then_follows_the_policy_for_n_further_steps():
    """Ensure each of N alternatives is followed by n policy steps, step-tagged."""
    env = FakeEnvironment(observation={"root": 1})
    agent = FakeAgent(first_actions=[10, 20], greedy="G")
    service = RecommendationService(env, agent)

    result = service.get_recommendations(
        _request(max_recommendations=2, n_steps=3)
    )

    # 2 alternatives, each applied once and then followed by 3 policy steps.
    assert [r.title for r in result] == [
        "action-10_step_1", "action-G_step_2", "action-G_step_3", "action-G_step_4",
        "action-20_step_1", "action-G_step_2", "action-G_step_3", "action-G_step_4",
    ]
    # Fan-out asked the agent for the alternatives from the root observation.
    assert agent.proposed_with == ({"root": 1}, 2)
    # One independent branch per alternative; the prepared env is never stepped.
    assert len(env.children) == 2
    assert env.step_actions == []
    assert env.children[0].step_actions == [10, "G", "G", "G"]
    assert env.children[1].step_actions == [20, "G", "G", "G"]
    # The policy is asked once per projected step and never after the last one:
    # a discarded action costs a full agent decision.
    assert len(agent.act_observations) == 2 * 3
    # KPIs come from the reached (stepped) observation of each branch.
    assert [r.kpis["t"] for r in result] == [1, 2, 3, 4, 1, 2, 3, 4]
    # branch_index/step let a client regroup the flat list into per-branch series.
    assert [r.branch_index for r in result] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert [r.step for r in result] == [1, 2, 3, 4, 1, 2, 3, 4]
    # `env_timestep` is the environment's own absolute clock, not the rollout
    # step: a branch that survives reports no terminal step.
    assert [r.env_timestep for r in result] == [
        CLOCK_OFFSET + t for t in (1, 2, 3, 4, 1, 2, 3, 4)
    ]
    assert [r.done for r in result] == [False] * 8


def test_zero_steps_returns_the_recommended_action_alone():
    """Ensure a depth of 0 projects nothing beyond the recommendation itself."""
    env = FakeEnvironment(observation={"root": 1})
    agent = FakeAgent(first_actions=[10, 20], greedy="G")
    service = RecommendationService(env, agent)

    result = service.get_recommendations(
        _request(max_recommendations=2, n_steps=0)
    )

    # One item per alternative: its own outcome, and no policy step after it.
    assert [r.title for r in result] == ["action-10_step_1", "action-20_step_1"]
    assert agent.act_observations == []


def test_rollout_stops_a_branch_when_environment_terminates():
    """Ensure a branch ends early when the env reports done before n_steps."""
    env = FakeEnvironment(observation={"root": 1}, max_steps=2)
    agent = FakeAgent(first_actions=[10], greedy="G")
    service = RecommendationService(env, agent)

    result = service.get_recommendations(
        _request(max_recommendations=1, n_steps=5)
    )

    # done fires at step 2, so the branch yields 2 recos despite n_steps=5.
    assert [r.title for r in result] == [
        "action-10_step_1", "action-G_step_2"
    ]
    # The terminal step is emitted rather than dropped, and is the only one
    # flagged: that flag is how a client tells "the branch ended here" from
    # "the rollout was simply shorter".
    assert [r.done for r in result] == [False, True]
