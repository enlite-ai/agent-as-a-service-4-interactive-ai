"""Resolves which use cases this deployment serves, from configuration alone.

Deliberately environment-agnostic: it names no integration and imports none.
Which ones are served is set by ``A3S_USE_CASES``, and each is loaded by import
path through the core's registry, so serving a different environment is a
configuration change rather than a code change here.
"""
import os

from a3s_core import UseCase, build_use_cases as load_registry
from a3s_core.interface import AgentAsAService
from config import logger

# Use cases this deployment serves, as a comma-separated list of
# "module.path:ClassName" references to `a3s_core.UseCase` implementations.
USE_CASE_SPECS = [
    spec.strip()
    for spec in os.environ.get(
        "A3S_USE_CASES", "integrations.powergrid:PowerGridUseCase"
    ).split(",")
    if spec.strip()
]

# Use case assumed when a caller does not send the `use_case` query parameter.
# The legacy T2.1_deep_expert API this service stands in for has no such
# parameter, so unqualified requests are served by the first registered use case
# unless a name is pinned here.
DEFAULT_USE_CASE = os.environ.get("A3S_DEFAULT_USE_CASE", "").strip()


def build_registry() -> tuple[dict[str, UseCase], dict[str, AgentAsAService]]:
    """
    Loads the configured use cases and builds their backends.

    :return tuple: ``(use_cases, services)``, both keyed by use-case name; the
        first reads request contexts, the second answers requests.
    :raises ValueError: If a configured spec cannot be resolved.
    """
    logger.info("Registering A3S use cases: %s", ", ".join(USE_CASE_SPECS))
    return load_registry(USE_CASE_SPECS)


def default_use_case(use_cases: dict[str, UseCase]) -> str | None:
    """
    Picks the use case that serves requests carrying no ``use_case`` parameter.

    :param dict use_cases: The registered use cases, keyed by name.
    :return str | None: The default use-case name, or ``None`` if none are
        registered.
    """
    if DEFAULT_USE_CASE:
        return DEFAULT_USE_CASE
    return next(iter(use_cases), None)
