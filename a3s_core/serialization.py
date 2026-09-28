"""Generic serialized environment-state envelope.

The envelope is intentionally minimal and environment-agnostic: it carries a
``serializer`` id identifying which environment/format produced it and an opaque
``state`` blob whose shape is defined entirely by that environment. Anything a
specific environment needs to reconstruct itself (seeds, action-replay history,
etc.) lives *inside* ``state`` - it is that environment's concern, not the
generic layer's. This keeps the core free to route the blob through untouched
and lets each environment choose its own reconstruction/branching strategy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SerializedEnvironmentState:
    """Environment-agnostic snapshot envelope with an opaque, env-defined state."""

    serializer: str
    state: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the envelope to a JSON-ready dictionary.

        :return: Dictionary representation of the environment state envelope.
        """
        return asdict(self)


def extract_serialized_state(
    context: dict[str, Any] | None,
) -> SerializedEnvironmentState | None:
    """Read the serialized environment-state envelope.

    :param context: The caller's request context payload.
    :return: Parsed environment-state envelope when present, otherwise `None`.
    """
    if not context:
        return None

    env_state = context.get("environment_state")
    if isinstance(env_state, dict) and "serializer" in env_state and "state" in env_state:
        return SerializedEnvironmentState(
            serializer=env_state["serializer"],
            state=env_state["state"],
            metadata=env_state.get("metadata", {}),
        )

    return None
