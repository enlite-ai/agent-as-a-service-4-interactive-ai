"""Renders a Grid2Op action into the operator-facing title and description.

Pure presentation for the PowerGrid use case: each renderer below inspects one
kind of action impact and, when that impact is present, returns the label
(``type_of_the_reco``), the title fragment and the description fragment for it.
Keeping them in one ordered table removes the branch-per-impact duplication and
makes the emitted wording easy to review in one place.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# A renderer returns (type_of_the_reco, title, description), or None when its
# impact is absent from the action.
Rendered = tuple[str, str, str] | None


def _redispatch(action: Any, impact: dict) -> Rendered:
    """
    Describes a production-source redispatch.

    :param Any action: The Grid2Op action being rendered.
    :param dict impact: The action's ``impact_on_objects()`` payload.
    :return Rendered: Label, title and description, or ``None``.
    """
    if not action._modif_redispatch:
        return None
    parts = [
        f'"{action.name_gen[gen]}" de {action._redispatch[gen]:.2f} MW'
        for gen in range(action.n_gen)
        if action._redispatch[gen] != 0.0
    ]
    return (
        "Redispatch",
        "Injection recommendation: production source redispatch",
        ", ".join(parts),
    )


def _storage(action: Any, impact: dict) -> Rendered:
    """
    Describes a storage charge/discharge setpoint.

    :param Any action: The Grid2Op action being rendered.
    :param dict impact: The action's ``impact_on_objects()`` payload.
    :return Rendered: Label, title and description, or ``None``.
    """
    if not action._modif_storage:
        return None
    parts = []
    for unit in range(action.n_storage):
        amount = action._storage_power[unit]
        if np.isfinite(amount) and amount != 0.0:
            parts.append(
                f'Ask unit "{action.name_storage[unit]}" to '
                f'{"charge" if amount > 0.0 else "discharge"} '
                f"{abs(amount):.2f} MW "
                f"(setpoint: {amount:.2f} MW)"
            )
    return "Storage", "Storage recommendation", ", ".join(parts)


def _curtailment(action: Any, impact: dict) -> Rendered:
    """
    Describes a renewable-generation curtailment.

    :param Any action: The Grid2Op action being rendered.
    :param dict impact: The action's ``impact_on_objects()`` payload.
    :return Rendered: Label, title and description, or ``None``.
    """
    if not action._modif_curtailment:
        return None
    parts = []
    for gen in range(action.n_gen):
        amount = action._curtail[gen]
        if np.isfinite(amount) and amount != -1.0:
            parts.append(
                f'Limit unit "{action.name_gen[gen]}" to '
                f"{100.0 * amount:.1f}% of its maximum capacity "
                f"(setpoint: {amount:.3f})"
            )
    return "Injection", "Injection recommendation", ", ".join(parts)


def _force_line(action: Any, impact: dict) -> Rendered:
    """
    Describes forced line (dis)connections.

    :param Any action: The Grid2Op action being rendered.
    :param dict impact: The action's ``impact_on_objects()`` payload.
    :return Rendered: Label, title and description, or ``None``.
    """
    force_line = impact["force_line"]
    if not force_line["changed"]:
        return None
    parts = []
    reconnections = force_line["reconnections"]
    if reconnections["count"] > 0:
        parts.append(
            f"Reconnection of {reconnections['count']} lines "
            f"({reconnections['powerlines']})"
        )
    disconnections = force_line["disconnections"]
    if disconnections["count"] > 0:
        parts.append(
            f"Disconnection of {disconnections['count']} lines "
            f"({disconnections['powerlines']})"
        )
    return (
        "Topological",
        "Topological recommendation: connection/disconnection of line",
        "".join(parts),
    )


def _switch_line(action: Any, impact: dict) -> Rendered:
    """
    Describes a line state switch.

    :param Any action: The Grid2Op action being rendered.
    :param dict impact: The action's ``impact_on_objects()`` payload.
    :return Rendered: Label, title and description, or ``None``.
    """
    switch_line = impact["switch_line"]
    if not switch_line["changed"]:
        return None
    return (
        "Topological",
        "Topological: change a line state",
        f"Change the state of {switch_line['count']} lines "
        f"({switch_line['powerlines']})",
    )


def _bus_switch(action: Any, impact: dict) -> Rendered:
    """
    Describes busbar switches at a substation.

    :param Any action: The Grid2Op action being rendered.
    :param dict impact: The action's ``impact_on_objects()`` payload.
    :return Rendered: Label, title and description, or ``None``.
    """
    bus_switch = impact["topology"]["bus_switch"]
    if len(bus_switch) == 0:
        return None
    parts = ["Busbar change:"]
    for switch in bus_switch:
        parts.append(
            f"\t \t - Switch bus of {switch['object_type']} id "
            f"{switch['object_id']} [at station {switch['substation']}]"
        )
    return (
        "Topological",
        "Topological recommendation: Schematic acquisition at substation "
        + str(bus_switch[0]["substation"]),
        "".join(parts),
    )


def _bus_assignment(action: Any, impact: dict) -> Rendered:
    """
    Describes bus assignments and disconnections at a substation.

    :param Any action: The Grid2Op action being rendered.
    :param dict impact: The action's ``impact_on_objects()`` payload.
    :return Rendered: Label, title and description, or ``None``.
    """
    assigned = impact["topology"]["assigned_bus"]
    disconnected = impact["topology"]["disconnect_bus"]
    if len(assigned) == 0 and len(disconnected) == 0:
        return None
    substation = (
        assigned[0]["substation"] if assigned else disconnected[0]["substation"]
    )
    assigned_parts = [
        f" Assign bus {item['bus']} to "
        f"{item['object_type']} id {item['object_id']}"
        for item in assigned
    ]
    disconnected_parts = [
        f"Disconnect {item['object_type']} with id "
        f"{item['object_id']} [at the substation level {item['substation']}]"
        for item in disconnected
    ]
    return (
        "Topological",
        "Topological recommendation: Schematic acquisition at substation "
        + str(substation),
        ", ".join(assigned_parts) + ", ".join(disconnected_parts),
    )


# Applied in order; every renderer that matches contributes its fragments, and
# the last one to match decides the reported `type_of_the_reco`.
_RENDERERS = (
    _redispatch,
    _storage,
    _curtailment,
    _force_line,
    _switch_line,
    _bus_switch,
    _bus_assignment,
)


def describe_action(action: Any, do_nothing: Any) -> tuple[str, str, dict]:
    """
    Builds the title, description and label KPI for a Grid2Op action.

    :param Any action: The Grid2Op action to describe.
    :param Any do_nothing: The environment's do-nothing action, used to
        recognise an action that changes nothing.
    :return tuple: ``(title, description, kpis)`` where ``kpis`` carries the
        ``type_of_the_reco`` label and ``title`` is empty for an unrecognised
        action.
    """
    impact = action.impact_on_objects()
    kpis: dict = {}
    titles: list[str] = []
    descriptions: list[str] = []

    for render in _RENDERERS:
        rendered = render(action, impact)
        if rendered is None:
            continue
        reco_type, title, description = rendered
        kpis["type_of_the_reco"] = reco_type
        titles.append(title)
        descriptions.append(description)

    if not titles and action == do_nothing:
        kpis["type_of_the_reco"] = "Do nothing"
        titles.append("Poursuivre")
        descriptions.append("Continuation of the scenario without operator action")

    return "".join(titles), "".join(descriptions), kpis
