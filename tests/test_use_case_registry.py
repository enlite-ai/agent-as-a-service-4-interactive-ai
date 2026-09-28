"""Tests the environment-agnostic use-case plugin surface and loader.

These exercise the core in isolation: the fake use case below stands in for a
real integration, which is the point - the core must be able to serve one
without any concrete environment being importable.
"""
import pytest

from a3s_core import (
    A3SRecommendation,
    EnvironmentUseCase,
    SerializedEnvironmentState,
    UseCase,
    build_use_cases,
    load_use_case,
)


class FakeEnvironment:
    """Minimal environment recording the state it was prepared with."""

    def prepare(self, environment_state):
        """:return: A one-key observation, or ``None`` without state."""
        return None if environment_state is None else {"t": 0}

    def step(self, action):
        """:return tuple: ``(observation, done)``."""
        return {"t": 1}, True

    def fork(self):
        """:return FakeEnvironment: An independent branch."""
        return FakeEnvironment()

    def kpis(self, observation):
        """:return dict: The observation's KPIs."""
        return {"k": 1}

    def timestep(self, observation):
        """:return int: The absolute environment clock."""
        return observation["t"]

    def format_recommendation(
        self, observation, action, kpis, agent_type, step, branch_index,
        *, done, env_timestep,
    ):
        """:return A3SRecommendation: The normalized recommendation."""
        return A3SRecommendation(
            title=f"a_step_{step}",
            description="",
            actions=[],
            agent_type=agent_type,
            kpis=kpis,
            branch_index=branch_index,
            step=step,
            done=done,
            env_timestep=env_timestep,
        )


class FakeAgent:
    """Agent proposing the same action regardless of observation."""

    def __init__(self, environment):
        """:param environment: The environment this agent acts on."""
        self.environment = environment

    @property
    def agent_type(self):
        """:return str: The source label for this agent."""
        return "FAKE"

    def propose(self, observation, n_actions):
        """:return list: ``n_actions`` candidate actions."""
        return ["a"] * n_actions

    def act(self, observation):
        """:return str: The single chosen action."""
        return "a"


class FakeUseCase(EnvironmentUseCase):
    """A use case wiring the fakes together, as a real integration would."""

    @property
    def name(self):
        """:return str: The use-case name."""
        return "Fake"

    def build_environment(self):
        """:return FakeEnvironment: The environment adapter."""
        return FakeEnvironment()

    def build_agent(self, environment):
        """:return FakeAgent: The recommendation engine."""
        return FakeAgent(environment)


class StrictUseCase(FakeUseCase):
    """Use case rejecting envelopes from an unknown serializer."""

    def validate_envelope(self, envelope):
        """:raises ValueError: If the serializer is not ``"known"``."""
        if envelope.serializer != "known":
            raise ValueError("unknown serializer")


class LegacyUseCase(FakeUseCase):
    """Use case accepting a bare, pre-envelope context."""

    def adapt_legacy_context(self, context):
        """:return SerializedEnvironmentState: The wrapped legacy context."""
        if "raw" not in context:
            return None
        return SerializedEnvironmentState(serializer="legacy", state=context["raw"])


def test_environment_use_case_builds_a_working_service():
    """A use case declaring only env and agent yields a usable backend."""
    service = FakeUseCase().build_service()
    assert service is not None


def test_resolve_reads_the_standard_envelope():
    """An envelope in the context is returned as-is."""
    context = {"environment_state": {"serializer": "known", "state": {"x": 1}}}
    envelope = FakeUseCase().resolve_environment_state(context)
    assert envelope.serializer == "known"
    assert envelope.state == {"x": 1}


def test_resolve_returns_none_without_state():
    """A context carrying no state resolves to nothing, not an error."""
    assert FakeUseCase().resolve_environment_state({}) is None
    assert FakeUseCase().resolve_environment_state(None) is None


def test_validate_envelope_rejects_at_the_edge():
    """A use case can refuse an envelope it cannot read."""
    context = {"environment_state": {"serializer": "other", "state": {}}}
    with pytest.raises(ValueError, match="unknown serializer"):
        StrictUseCase().resolve_environment_state(context)


def test_legacy_context_is_wrapped_into_an_envelope():
    """A pre-envelope context is adapted by the use case, not the API layer."""
    envelope = LegacyUseCase().resolve_environment_state({"raw": {"y": 2}})
    assert envelope.serializer == "legacy"
    assert envelope.state == {"y": 2}


def test_load_use_case_resolves_an_import_spec():
    """A use case is loaded by "module:Class" reference."""
    use_case = load_use_case(f"{__name__}:FakeUseCase")
    assert isinstance(use_case, UseCase)
    assert use_case.name == "Fake"


@pytest.mark.parametrize(
    "spec, match",
    [
        ("no_colon", "Malformed"),
        ("nonexistent.module:Thing", "Cannot import"),
        (f"{__name__}:Missing", "has no attribute"),
        (f"{__name__}:FakeAgent", "does not name a UseCase"),
    ],
)
def test_load_use_case_rejects_bad_specs(spec, match):
    """A misconfigured spec fails loudly rather than serving nothing."""
    with pytest.raises(ValueError, match=match):
        load_use_case(spec)


def test_build_use_cases_keys_by_name():
    """The registry returns contexts readers and services under one name each."""
    use_cases, services = build_use_cases([f"{__name__}:FakeUseCase"])
    assert set(use_cases) == {"Fake"} == set(services)


def test_build_use_cases_rejects_duplicate_names():
    """Two use cases claiming one name would silently drop one, so it errors."""
    spec = f"{__name__}:FakeUseCase"
    with pytest.raises(ValueError, match="Duplicate use-case name"):
        build_use_cases([spec, spec])
