"""Resolves A3S use cases from import paths, so the core ships no integrations.

Which integrations a deployment serves is configuration, not code: each is named
as a ``"module.path:ClassName"`` spec and imported on demand. That is what keeps
this core environment-agnostic - it never imports an integration, it is handed
one - and it lets an image serve a different environment by changing a variable
rather than a source file.
"""
from __future__ import annotations

import logging
from importlib import import_module
from typing import Iterable

from .interface import AgentAsAService
from .use_case import UseCase

logger = logging.getLogger(__name__)


def load_use_case(spec: str) -> UseCase:
    """
    Imports and instantiates the use case named by an import spec.

    :param str spec: A ``"module.path:ClassName"`` reference to a
        :class:`UseCase` subclass taking no constructor arguments.
    :return UseCase: The instantiated use case.
    :raises ValueError: If the spec is malformed, cannot be imported, or does
        not name a :class:`UseCase`. Failing here is deliberate: a typo must be
        distinguishable from a deployment that legitimately serves nothing.
    """
    module_path, separator, class_name = spec.partition(":")
    if not separator or not module_path or not class_name:
        raise ValueError(
            f"Malformed use-case spec {spec!r}; expected 'module.path:ClassName'"
        )

    try:
        module = import_module(module_path)
    except ImportError as exc:
        raise ValueError(f"Cannot import use-case module {module_path!r}: {exc}") from exc

    use_case_class = getattr(module, class_name, None)
    if use_case_class is None:
        raise ValueError(f"{module_path!r} has no attribute {class_name!r}")
    if not (isinstance(use_case_class, type) and issubclass(use_case_class, UseCase)):
        raise ValueError(f"{spec!r} does not name a UseCase subclass")

    return use_case_class()


def build_use_cases(
    specs: Iterable[str],
) -> tuple[dict[str, UseCase], dict[str, AgentAsAService]]:
    """
    Loads every named use case and builds its backend.

    :param specs: ``"module.path:ClassName"`` references to load, in order.
    :return tuple: ``(use_cases, services)``, both keyed by use-case name. The
        first is needed to read request contexts, the second to answer requests.
    :raises ValueError: If a spec cannot be resolved, or two use cases claim the
        same name - which would otherwise silently drop one of them.
    """
    use_cases: dict[str, UseCase] = {}
    services: dict[str, AgentAsAService] = {}
    for spec in specs:
        use_case = load_use_case(spec)
        if use_case.name in use_cases:
            raise ValueError(
                f"Duplicate use-case name {use_case.name!r} from spec {spec!r}"
            )
        use_cases[use_case.name] = use_case
        services[use_case.name] = use_case.build_service()
        logger.info("Registered A3S use case %r from %r", use_case.name, spec)
    return use_cases, services
