import numpy as np
import pytest

from integrations.powergrid.environment import PowerGridEnvironment
from integrations.powergrid.serialization import (
    build_grid2op_observation_state,
    read_grid2op_state,
)
from a3s_core import SerializedEnvironmentState


def test_read_grid2op_state_returns_blob_and_validates_serializer():
    """Ensure the PowerGrid state blob is returned and the serializer checked."""
    assert read_grid2op_state(None) is None

    envelope = build_grid2op_observation_state(
        {"current_step": ["12"]}, seed=7, replay_actions=[{"vect": [0.0]}]
    )
    blob = read_grid2op_state(envelope)
    assert blob["observation"] == {"current_step": ["12"]}
    assert blob["seed"] == 7
    assert blob["replay_actions"] == [{"vect": [0.0]}]

    with pytest.raises(ValueError):
        read_grid2op_state(
            SerializedEnvironmentState(serializer="other_env_v1", state={})
        )


def test_coerce_seed_normalizes_to_int_or_none():
    """Ensure a carried seed is coerced to int, and absence maps to None."""
    assert PowerGridEnvironment._coerce_seed(None) is None
    assert PowerGridEnvironment._coerce_seed("42") == 42
    assert PowerGridEnvironment._coerce_seed(7) == 7


def test_effective_seed_prefers_request_then_config():
    """Ensure the request seed wins, with the config seed as fallback."""
    service = PowerGridEnvironment()
    service._config_seed = 7
    assert service._effective_seed() == 7  # no request seed -> config
    service._active_seed = 99
    assert service._effective_seed() == 99  # request seed wins


class FakeAction:
    """Minimal fake action exposing the formatter inputs used by the test."""

    _modif_redispatch = False
    _modif_storage = False
    _modif_curtailment = False
    n_gen = 0
    n_storage = 0

    def impact_on_objects(self):
        """Return a deterministic topological impact payload.

        :return: Fake action impact dictionary.
        """
        return {
            "force_line": {
                "changed": False,
                "reconnections": {"count": 0, "powerlines": []},
                "disconnections": {"count": 0, "powerlines": []},
            },
            "switch_line": {
                "changed": False,
                "count": 0,
                "powerlines": [],
            },
            "topology": {
                "bus_switch": [
                    {"substation": 3, "object_type": "load", "object_id": 1}
                ],
                "assigned_bus": [],
                "disconnect_bus": [],
            },
        }

    def to_json(self):
        """Return a JSON-ready action payload.

        :return: Minimal serialized action payload.
        """
        return {"fake": True}


def test_format_recommendation_tags_step_and_packages_kpis():
    """Ensure formatting derives the label, packages KPIs and tags the step."""
    service = PowerGridEnvironment()
    service.action_do_nothing = object()

    # KPIs are computed elsewhere (env.kpis) and passed in; formatting is pure
    # presentation that derives the label, packages the KPIs and tags the step.
    recommendation = service.format_recommendation(
        object(),
        FakeAction(),
        {"efficiency_of_the_reco": 0.8},
        "IA",
        step=2,
        branch_index=1,
        done=False,
        env_timestep=742,
    )

    assert recommendation.title == (
        "Topological recommendation: Schematic acquisition at substation 3"
        "_step_2"
    )
    assert recommendation.description == (
        "Busbar change:\t \t - Switch bus of load id 1 [at station 3]"
    )
    assert recommendation.kpis["type_of_the_reco"] == "Topological"
    assert recommendation.kpis["efficiency_of_the_reco"] == 0.8
    assert recommendation.agent_type == "IA"
    assert recommendation.branch_index == 1
    assert recommendation.step == 2
    assert recommendation.done is False
    # The rollout step and the environment's clock are separate quantities, and
    # formatting stays pure presentation: it reports the clock it was handed
    # rather than reading it off the observation (which is a bare `object()` here).
    assert recommendation.env_timestep == 742


def test_timestep_reads_the_grid2op_clock_and_admits_absence():
    """Ensure the absolute clock is read off the observation, or reported absent."""
    service = PowerGridEnvironment()

    class Obs:
        """Observation stub carrying only a step counter."""

        def __init__(self, current_step):
            self.current_step = current_step

    # numpy integers do not survive `jsonify`, so the clock must be a plain int.
    assert service.timestep(Obs(np.int64(7))) == 7
    assert isinstance(service.timestep(Obs(np.int64(7))), int)
    # A Grid2Op env reports no step before its first reset; that is an honest
    # absence, not timestep 0.
    assert service.timestep(Obs(None)) is None
