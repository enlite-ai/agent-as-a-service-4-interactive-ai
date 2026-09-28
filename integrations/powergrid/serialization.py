# Defines and decodes the PowerGrid-specific state blob carried by the generic
# environment-state envelope.
"""Grid2Op-specific serialization for the PowerGrid use case.

The generic ``src/serialization.py`` only knows the framework-agnostic envelope
(:class:`SerializedEnvironmentState`) with an opaque ``state`` blob. This module
defines the *shape* of that blob for PowerGrid and how to read it back - all
PowerGrid concerns kept out of the generic layer.

The blob carries the Grid2Op observation plus the inputs needed to restore the
environment it came from: the environment's identity and the full action-replay
history. Replaying a matching seeded environment through those actions reproduces
it exactly - including the hidden state no observation exposes (opponent budget,
cooldowns, accumulated redispatch) - which is what makes a multi-step KPI
projection faithful. Without a replay history the environment can only be
approximated by fast-forwarding the chronics to the incoming timestep.

Two blob shapes are read back:

``grid2op_observation_v2``
    Current shape. Carries the environment *identity* (seed, scenario name,
    Grid2Op/LightSim2Grid versions) so the consumer can prove it is replaying
    against the same environment, and carries the history gzipped+base64-encoded
    (it compresses ~300x, which matters because the whole context is re-fetched
    by the frontend on every poll and persisted on every push).

``grid2op_observation_v1``
    What earlier simulator builds published: a plain ``replay_actions`` list and
    at most a seed. Still read, so an un-upgraded producer keeps working, but
    without an identity to check the consumer can only trust the reached-state
    verification in :mod:`integrations.powergrid.environment`.
"""
from __future__ import annotations

import base64
import gzip
import json
from typing import Any

from a3s_core import SerializedEnvironmentState

# Serializer ids for a Grid2Op observation payload - the shared contract between
# the producer (which stamps it) and the PowerGrid environment (which checks it).
GRID2OP_OBSERVATION_V1 = "grid2op_observation_v1"
GRID2OP_OBSERVATION_V2 = "grid2op_observation_v2"

# Blob shapes this module can read back.
SUPPORTED_SERIALIZERS = (GRID2OP_OBSERVATION_V1, GRID2OP_OBSERVATION_V2)

# Framings this module can decode a ``replay_actions`` field from. ``None`` (the
# key absent) means a plain, uncompressed list.
GZIP_BASE64 = "gzip+base64"
SUPPORTED_COMPRESSIONS = (None, "", "none", GZIP_BASE64)

# Keys of the environment identity a v2 producer publishes inside the blob.
IDENTITY_KEYS = (
    "seed",
    "scenario_name",
    "grid2op_version",
    "lightsim2grid_version",
)


def build_grid2op_observation_state(
    observation: dict[str, Any],
    *,
    seed: int | None = None,
    replay_actions: list[dict[str, Any]] | str | None = None,
    identity: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> SerializedEnvironmentState:
    """Wrap a Grid2Op observation and reconstruction inputs in an envelope.

    :param observation: Observation payload serialized by Grid2Op.
    :param seed: Optional seed used to reconstruct the env's stochastic parts.
    :param replay_actions: Optional action-replay history for an exact rebuild,
        either as a plain list or already gzipped+base64-encoded.
    :param identity: Optional environment identity (scenario name, library
        versions) the history must be replayed against.
    :param metadata: Optional transport metadata.
    :return: Generic serialized environment-state envelope.
    """
    state: dict[str, Any] = {
        "observation": observation,
        "seed": seed,
        "replay_actions": replay_actions if replay_actions is not None else [],
    }
    state.update(identity or {})
    serializer = (
        GRID2OP_OBSERVATION_V2 if identity else GRID2OP_OBSERVATION_V1
    )
    return SerializedEnvironmentState(
        serializer=serializer,
        state=state,
        metadata=metadata or {},
    )


def legacy_context_to_state(
    context: dict[str, Any],
) -> SerializedEnvironmentState:
    """
    Wraps a legacy (envelope-less) PowerGrid context into a state envelope.

    The legacy T2.1_deep_expert body carries the bare Grid2Op observation under
    ``context["observation"]`` and usually neither a seed nor an action history;
    the environment then falls back to its configured seed and fast-forwards the
    chronics to the incoming timestep. Both keys are still read so a producer can
    start sending them without another API change.

    :param dict context: InteractiveAI context payload carrying an observation.
    :return SerializedEnvironmentState: The equivalent state envelope.
    """
    return build_grid2op_observation_state(
        context["observation"],
        seed=context.get("seed"),
        replay_actions=context.get("replay_actions"),
    )


def validate_grid2op_envelope(
    environment_state: SerializedEnvironmentState | None,
) -> None:
    """
    Checks at the API edge that this build can read the incoming blob's shape.

    Only the shape is checked, not the payload: decoding the history is deferred
    to the environment, which is where a failure to reconstruct has to fail
    closed. A shape this build does not know is instead a caller-side problem
    worth rejecting outright.

    :param SerializedEnvironmentState environment_state: Envelope to check, or
        ``None`` when the context carries no state.
    :raises ValueError: If the serializer id, or the history's framing, is one
        this build cannot read.
    """
    if environment_state is None:
        return
    if environment_state.serializer not in SUPPORTED_SERIALIZERS:
        raise ValueError(
            f"Unsupported environment serializer: {environment_state.serializer}"
        )
    compression = (environment_state.metadata or {}).get("compression")
    if compression not in SUPPORTED_COMPRESSIONS:
        raise ValueError(
            f"Unsupported replay-history compression: {compression}"
        )


def decode_replay_actions(
    replay_actions: list[dict[str, Any]] | str | None,
    compression: str | None,
) -> list[dict[str, Any]]:
    """
    Decodes the published action history into a plain list of actions.

    :param replay_actions: History as published: a plain list, or a
        gzipped+base64-encoded JSON string.
    :param str compression: Framing stamped in the envelope's metadata.
    :return list: The action history, in step order (empty when there is none).
    :raises ValueError: If the history cannot be decoded, or is not a list of
        action vectors once decoded. A history that cannot be read in full is
        never partially accepted: replaying part of one lands on a different
        grid than the producer's.
    """
    if not replay_actions:
        return []

    decoded = replay_actions
    if isinstance(replay_actions, str):
        if compression != GZIP_BASE64:
            raise ValueError(
                "Replay history is a string but its framing is "
                f"{compression!r}, not {GZIP_BASE64!r}"
            )
        try:
            decoded = json.loads(
                gzip.decompress(base64.b64decode(replay_actions)).decode(
                    "utf-8"
                )
            )
        except Exception as exc:
            raise ValueError(
                f"Corrupt {GZIP_BASE64} replay history: {exc}"
            ) from exc

    if not isinstance(decoded, list) or not all(
        isinstance(entry, dict) and isinstance(entry.get("vect"), list)
        for entry in decoded
    ):
        raise ValueError(
            "Replay history is not a list of {'vect': [...]} actions"
        )
    return decoded


def read_grid2op_state(
    environment_state: SerializedEnvironmentState | None,
) -> dict[str, Any] | None:
    """Return the PowerGrid state blob, normalized across blob versions.

    The returned mapping always has the same shape regardless of which serializer
    produced it: the observation, the decoded action history, the environment
    identity (with ``None`` for whatever the producer did not publish) and the
    producer's own account of the state it shipped, which the environment
    cross-checks against what it reaches.

    :param environment_state: Generic serialized environment-state envelope.
    :return: The normalized PowerGrid state, or ``None`` when there is no state.
    :raises ValueError: If the envelope was produced by a different serializer,
        or its action history cannot be decoded.
    """
    if environment_state is None:
        return None
    validate_grid2op_envelope(environment_state)

    state = environment_state.state or {}
    metadata = environment_state.metadata or {}
    normalized = {key: state.get(key) for key in IDENTITY_KEYS}
    normalized["observation"] = state.get("observation")
    normalized["replay_actions"] = decode_replay_actions(
        state.get("replay_actions"), metadata.get("compression")
    )
    # What the producer says it shipped. Checked against the replayed result
    # rather than trusted: a mismatch means the two sides are not on the same
    # grid, whatever the reason.
    normalized["expected_actions"] = metadata.get("replayed_actions")
    normalized["expected_step"] = metadata.get("current_step")
    return normalized
