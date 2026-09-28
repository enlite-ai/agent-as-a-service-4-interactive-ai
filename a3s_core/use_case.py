"""The plugin surface an A3S integration implements.

The core answers requests through :class:`AgentAsAService`, but a deployment
also has to know *which* use cases exist, how each one is built, and how each
one reads an incoming context payload. Those are per-integration facts, so they
are declared here rather than hard-coded into the API layer: a use case is a
:class:`UseCase` object, and the API layer only ever sees this interface.

:class:`EnvironmentUseCase` is the base an integration normally wants. It
already performs the whole wiring - building the projection service around an
environment and its recommendation engine, and reading the standard state
envelope - so a concrete integration reduces to naming its environment and its
agent, plus optional hooks for validating or legacy-adapting a context.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .interface import Agent, AgentAsAService, Environment
from .serialization import SerializedEnvironmentState, extract_serialized_state
from .service import RecommendationService


class UseCase(ABC):
    """A named A3S backend together with how to read its request context."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Use-case name callers address this backend by (e.g. ``"PowerGrid"``)."""

    @abstractmethod
    def build_service(self) -> AgentAsAService:
        """
        Builds the backend that answers recommendation requests.

        Called once at startup. Anything expensive (models, simulators) should be
        deferred to the first request rather than loaded here, so registering a
        use case costs nothing until it is actually used.

        :return AgentAsAService: The backend serving this use case.
        """

    @abstractmethod
    def resolve_environment_state(
        self, context: dict[str, Any] | None
    ) -> SerializedEnvironmentState | None:
        """
        Reads the serialized environment state out of a request context.

        :param dict context: The caller's context payload.
        :return SerializedEnvironmentState: The envelope, or ``None`` when the
            context carries no state.
        :raises ValueError: If the context is shaped in a way this use case
            cannot read; the API layer turns this into a 400.
        """


class EnvironmentUseCase(UseCase):
    """Base use case pairing an environment with a recommendation engine.

    Subclasses declare the two moving parts - :meth:`build_environment` and
    :meth:`build_agent` - and everything else is inherited: the environment and
    agent are wired into a :class:`RecommendationService`, and a context is read
    through the standard envelope. Override :meth:`validate_envelope` to reject
    unreadable envelopes at the edge, and :meth:`adapt_legacy_context` to accept
    a bare, environment-native context that predates the envelope.
    """

    @abstractmethod
    def build_environment(self) -> Environment:
        """
        Builds the environment adapter this use case projects with.

        :return Environment: The environment adapter.
        """

    @abstractmethod
    def build_agent(self, environment: Environment) -> Agent:
        """
        Builds the recommendation engine that supplies actions to project.

        :param Environment environment: The environment the agent acts on, as
            some agents need its action space or a simulation handle.
        :return Agent: The agent adapter.
        """

    def build_service(self) -> AgentAsAService:
        """
        Wires the environment and agent into the generic projection service.

        Stashes the built environment on ``self._environment`` too: it is the
        same instance the service rolls out requests against, and exposing it
        here lets a caller (tests, an admin endpoint) reach the live
        environment without reaching into the service's private state.

        :return AgentAsAService: The backend serving this use case.
        """
        environment = self.build_environment()
        self._environment = environment
        return RecommendationService(environment, self.build_agent(environment))

    def resolve_environment_state(
        self, context: dict[str, Any] | None
    ) -> SerializedEnvironmentState | None:
        """
        Reads the standard envelope, falling back to a legacy context shape.

        :param dict context: The caller's context payload.
        :return SerializedEnvironmentState: The envelope, or ``None`` when absent.
        :raises ValueError: If :meth:`validate_envelope` rejects the envelope.
        """
        envelope = extract_serialized_state(context)
        if envelope is not None:
            self.validate_envelope(envelope)
            return envelope
        return self.adapt_legacy_context(context or {})

    def validate_envelope(self, envelope: SerializedEnvironmentState) -> None:
        """
        Checks that an incoming envelope is one this use case can read.

        Runs at the API edge, so a context produced by an unknown serializer is
        rejected before any environment work starts. This is a *shape* check
        only: whether the reconstructed state is trustworthy can only be decided
        by the environment itself, so it stays there. Accepts anything by
        default.

        :param SerializedEnvironmentState envelope: The incoming envelope.
        :return None:
        :raises ValueError: If the envelope cannot be read by this use case.
        """

    def adapt_legacy_context(
        self, context: dict[str, Any]
    ) -> SerializedEnvironmentState | None:
        """
        Wraps a pre-envelope, environment-native context into an envelope.

        Lets a service stay a drop-in replacement for an older API whose context
        carried a bare native state. Returns ``None`` by default, meaning this
        use case requires a proper envelope.

        :param dict context: The caller's context payload, without an envelope.
        :return SerializedEnvironmentState: The envelope, or ``None``.
        """
        return None
