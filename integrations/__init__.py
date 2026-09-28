"""Concrete A3S integrations.

Each subpackage is one deployment of the environment- and model-agnostic
:mod:`a3s_core`: it supplies an :class:`~a3s_core.Environment` adapter, the
recommendation engine that proposes actions to project, and a
:class:`~a3s_core.UseCase` declaring the two. Integrations depend on the core;
the core never depends on them, and is wired to one by import path at startup.
"""
