"""Environment- and model-agnostic Agent-as-a-Service core.

This package is the reusable heart of A3S and is deliberately free of any
reference to a concrete environment, agent or framework. It provides:

* the contract an integration implements - :class:`Environment`, :class:`Agent`,
  and the normalized request/recommendation shapes (:mod:`.interface`);
* the environment-agnostic serialized-state envelope (:mod:`.serialization`);
* the multi-step projection engine that turns a serialized state into
  recommendations (:mod:`.service`);
* the plugin surface an integration declares itself through, and the loader that
  resolves plugins by import path (:mod:`.use_case`, :mod:`.registry`).

A concrete integration (PowerGrid, Flatland, ATM, ...) lives entirely outside
this package, under ``integrations/``: it subclasses :class:`EnvironmentUseCase`,
names its environment and its recommendation engine, and is wired in by import
path at startup. Nothing here changes when one is added.
"""
from .interface import (
    A3SRecommendation,
    A3SRecommendationRequest,
    ActionType,
    Agent,
    AgentAsAService,
    Environment,
    EventType,
    KPIType,
    ObservationType,
)
from .registry import build_use_cases, load_use_case
from .serialization import SerializedEnvironmentState, extract_serialized_state
from .service import RecommendationService
from .use_case import EnvironmentUseCase, UseCase

__all__ = [
    "A3SRecommendation",
    "A3SRecommendationRequest",
    "ActionType",
    "Agent",
    "AgentAsAService",
    "Environment",
    "EnvironmentUseCase",
    "EventType",
    "KPIType",
    "ObservationType",
    "RecommendationService",
    "SerializedEnvironmentState",
    "UseCase",
    "build_use_cases",
    "extract_serialized_state",
    "load_use_case",
]
