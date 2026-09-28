# Tests for reading back the PowerGrid state blob, independently of Grid2Op.
"""Blob-level tests for the PowerGrid serialization contract.

These cover what the service accepts, rejects and normalizes *before* any grid is
built, so they run everywhere - no Grid2Op stack required. The fidelity of the
reconstruction itself is covered by ``test_env_replay.py``.
"""
import base64
import gzip
import json

import pytest

from integrations.powergrid.serialization import (
    GRID2OP_OBSERVATION_V1,
    GRID2OP_OBSERVATION_V2,
    GZIP_BASE64,
    build_grid2op_observation_state,
    legacy_context_to_state,
    read_grid2op_state,
    validate_grid2op_envelope,
)
from a3s_core import SerializedEnvironmentState

OBSERVATION = {"current_step": [42], "rho": [0.5, 0.9]}
HISTORY = [{"vect": [0.0, 1.0, 2.0]}, {"vect": [3.0, 0.0, 0.0]}]


def _encode(history):
    """Encode a history the way the simulator publishes it.

    :param list history: Action history, in step order.
    :return str: The history as gzipped, base64-encoded JSON.
    """
    return base64.b64encode(
        gzip.compress(json.dumps(history).encode("utf-8"))
    ).decode("ascii")


def _v2_envelope(**overrides):
    """Build the envelope a current simulator publishes.

    :param overrides: Fields to replace in the state or metadata.
    :return SerializedEnvironmentState: The published envelope.
    """
    state = {
        "observation": OBSERVATION,
        "replay_actions": _encode(HISTORY),
        "seed": 2118338672,
        "scenario_name": "jan_28_1",
        "grid2op_version": "1.9.8",
        "lightsim2grid_version": "0.7.5",
    }
    metadata = {
        "current_step": 42,
        "replayed_actions": len(HISTORY),
        "compression": GZIP_BASE64,
    }
    state.update(overrides.pop("state", {}))
    metadata.update(overrides.pop("metadata", {}))
    return SerializedEnvironmentState(
        serializer=GRID2OP_OBSERVATION_V2, state=state, metadata=metadata
    )


def test_compressed_history_is_decoded():
    """Ensure a v2 blob normalizes into the plain history plus its identity."""
    state = read_grid2op_state(_v2_envelope())

    assert state["replay_actions"] == HISTORY
    assert state["observation"] == OBSERVATION
    assert state["scenario_name"] == "jan_28_1"
    assert state["seed"] == 2118338672
    assert state["lightsim2grid_version"] == "0.7.5"
    assert state["expected_actions"] == len(HISTORY)
    assert state["expected_step"] == 42


def test_uncompressed_history_from_an_older_producer_is_read():
    """Ensure the v1 shape still normalizes, with no identity to check."""
    envelope = build_grid2op_observation_state(
        OBSERVATION, seed=7, replay_actions=HISTORY
    )
    state = read_grid2op_state(envelope)

    assert envelope.serializer == GRID2OP_OBSERVATION_V1
    assert state["replay_actions"] == HISTORY
    assert state["scenario_name"] is None
    assert state["expected_actions"] is None


def test_legacy_context_without_a_history_is_read():
    """Ensure a bare legacy context becomes an approximate-only state."""
    state = read_grid2op_state(
        legacy_context_to_state({"observation": OBSERVATION})
    )

    assert state["replay_actions"] == []
    assert state["seed"] is None


def test_compression_survives_a_json_round_trip():
    """Ensure the encoded history is transport-safe as published JSON."""
    envelope = _v2_envelope()
    round_tripped = json.loads(json.dumps(envelope.to_dict()))

    assert read_grid2op_state(
        SerializedEnvironmentState(**round_tripped)
    )["replay_actions"] == HISTORY


def test_a_corrupt_history_is_refused_rather_than_truncated():
    """Ensure an undecodable history raises instead of yielding a partial one."""
    envelope = _v2_envelope(state={"replay_actions": "not-actually-gzip"})

    with pytest.raises(ValueError, match="Corrupt"):
        read_grid2op_state(envelope)


def test_a_misframed_history_is_refused():
    """Ensure an encoded history without its framing is not read as a list."""
    envelope = _v2_envelope(metadata={"compression": None})

    with pytest.raises(ValueError, match="framing"):
        read_grid2op_state(envelope)


def test_a_malformed_action_is_refused():
    """Ensure a history whose entries are not action vectors is rejected."""
    envelope = _v2_envelope(
        state={"replay_actions": _encode([{"action": "reconnect line 3"}])}
    )

    with pytest.raises(ValueError, match="not a list"):
        read_grid2op_state(envelope)


def test_an_unknown_serializer_is_rejected_at_the_edge():
    """Ensure a blob shape this build cannot read fails the edge check."""
    envelope = SerializedEnvironmentState(
        serializer="grid2op_observation_v9", state={}, metadata={}
    )

    with pytest.raises(ValueError, match="Unsupported environment serializer"):
        validate_grid2op_envelope(envelope)


def test_an_unknown_compression_is_rejected_at_the_edge():
    """Ensure a framing this build cannot decode fails the edge check."""
    envelope = _v2_envelope(metadata={"compression": "zstd"})

    with pytest.raises(ValueError, match="compression"):
        validate_grid2op_envelope(envelope)


def test_no_state_reads_as_no_state():
    """Ensure an absent envelope is not an error."""
    assert read_grid2op_state(None) is None
    validate_grid2op_envelope(None)
