"""End-to-end smoke test of the PowerGrid recommendation pipeline.

Exercises the whole flow the way it runs in production, minus the network hops:
a context is produced exactly as the simulator serializes it, POSTed to the A3S
recommendation endpoint via the Flask test client, deserialized, the environment
is reconstructed, the agent proposes actions, and at least one actionable
recommendation with KPIs comes back.

All accepted request shapes are covered: the envelope the simulator publishes
today, carrying a compressed action-replay history and the environment's identity
(the exact-reconstruction path), an envelope without a history, and the legacy
T2.1_deep_expert body (bare ``context.observation``, no ``use_case`` query
parameter) that makes this service a drop-in replacement for the external RL
agent API.

Requires the Grid2Op stack, so it runs where the service's dependencies are
installed (container / CI), not on a bare host.
"""
import base64
import gzip
import json

import grid2op

from app import create_app
from integrations.powergrid.serialization import (
    GZIP_BASE64,
    build_grid2op_observation_state,
)


def _overloaded_observation(app):
    """Produce an observation of an overloaded grid, as the simulator would.

    Uses a throwaway copy of the service's Grid2Op env (same scenario/seed) to
    generate a realistic observation without disturbing the env that serves the
    request — mirroring the separate producer/consumer processes in production.

    :param app: The A3S Flask app (holds the registered PowerGrid environment).
    :return: ``(observation_json, event_line, seed)`` for the overloaded step.
    """
    environment = app.use_cases["PowerGrid"]._environment
    environment.ensure_initialized()

    gen_env = environment.env.copy()
    obs = gen_env.reset()
    do_nothing = gen_env.action_space({})
    step = 0
    while obs.rho.max() < 1.0 and step < 300:
        obs, _, done, _ = gen_env.step(do_nothing)
        step += 1
        if done:
            obs = gen_env.reset()

    # Overloaded line in the "<or_subid>:<ex_subid>:<name_line>" form the
    # simulator sends as the event line.
    idx = int(obs.rho.argmax())
    line = (
        f"{gen_env.line_or_to_subid[idx]}:"
        f"{gen_env.line_ex_to_subid[idx]}:"
        f"{gen_env.name_line[idx]}"
    )

    return obs, line, int(environment._config_seed)


def _assert_actionable(recommendations):
    """Assert a response carries well-formed, actually-simulated recommendations.

    :param recommendations: Parsed JSON response body.
    :return: None.
    """
    assert isinstance(recommendations, list) and len(recommendations) >= 1

    for reco in recommendations:
        assert reco["use_case"] == "PowerGrid"
        # Only the neural-network agent is registered here.
        assert reco["agent_type"] == "IA"
        assert reco["title"]
        assert "kpis" in reco
        # Asserted on the serialized body, not on the recommendation objects:
        # the output schema is a whitelist, so a field missing from it is
        # dropped on the way out while every object-level test still passes.
        assert "done" in reco
        assert "env_timestep" in reco

    # At least one recommendation carries an efficiency KPI, i.e. the pipeline
    # really did simulate an action.
    assert any(
        "efficiency_of_the_reco" in (reco["kpis"] or {})
        for reco in recommendations
    )


def _replayed_history(app, steps):
    """Act as the simulator: step a private env and record what was applied.

    :param app: The A3S Flask app (holds the registered PowerGrid environment).
    :param int steps: Number of steps to advance before publishing.
    :return dict: The observation, action history and step the producer reached.
    """
    environment = app.use_cases["PowerGrid"]._environment
    environment.ensure_initialized()

    producer = environment.env.copy()
    producer.seed(environment._effective_seed())
    producer.set_id(environment.id_scenario)
    obs = producer.reset()

    history = []
    for _ in range(steps):
        action = producer.action_space({})
        history.append({"vect": action.to_vect().tolist()})
        obs, _, done, _ = producer.step(action)
        if done:
            break

    return {
        "observation": obs.to_json(),
        # Published the way the simulator publishes it: gzipped and base64-encoded.
        "replay_actions": base64.b64encode(
            gzip.compress(json.dumps(history).encode("utf-8"))
        ).decode("ascii"),
        "recorded_actions": len(history),
        "current_step": int(obs.current_step),
    }


def test_pipeline_smoke_context_to_recommendation():
    """Post a simulator-shaped context and get actionable recommendations back."""
    app = create_app("test")
    client = app.test_client()

    obs, line, seed = _overloaded_observation(app)
    # Envelope shaped exactly like Communicate.send_context_online (seed and
    # replay history nested inside the env-defined state blob).
    payload = {
        "context": {
            "environment_state": build_grid2op_observation_state(
                obs.to_json(),
                seed=seed,
                replay_actions=[],
                metadata={"current_step": int(obs.current_step)},
            ).to_dict(),
        },
        "event": {"line": line},
        "options": {"max_recommendations": 3, "n_steps": 1},
    }

    response = client.post(
        "/api/v1/recommendation?use_case=PowerGrid", json=payload
    )

    assert response.status_code == 200
    _assert_actionable(response.get_json())


def test_pipeline_smoke_legacy_rl_agent_payload():
    """Serve a legacy T2.1_deep_expert request unchanged (drop-in replacement).

    Same body recommendation-service sends to the external RL agent API — a bare
    Grid2Op observation under ``context.observation``, no options, and no
    ``use_case`` query parameter — must yield a single-step KPI projection.
    """
    app = create_app("test")
    client = app.test_client()

    obs, line, _ = _overloaded_observation(app)
    payload = {"context": {"observation": obs.to_json()}, "event": {"line": line}}

    response = client.post("/api/v1/recommendation", json=payload)

    assert response.status_code == 200
    recommendations = response.get_json()
    _assert_actionable(recommendations)
    # The projection depth defaults to 0 — no policy step beyond the recommended
    # action — so this stays a drop-in replacement for the legacy single-step API.
    assert all(reco["title"].endswith("_step_1") for reco in recommendations)
    assert all(reco["step"] == 1 for reco in recommendations)


def test_pipeline_smoke_replayed_environment_payload():
    """Post a context carrying a replay history and roll out a KPI projection.

    Mirrors what Communicate.send_context_online publishes once environment
    serialization is enabled: the observation plus the environment's identity and
    its full, compressed action history, under ``context.environment_state``.
    """
    app = create_app("test")
    client = app.test_client()

    environment = app.use_cases["PowerGrid"]._environment
    obs, line, seed = _overloaded_observation(app)
    replay = _replayed_history(app, steps=30)

    payload = {
        "context": {
            "environment_state": build_grid2op_observation_state(
                replay["observation"],
                seed=seed,
                replay_actions=replay["replay_actions"],
                identity={
                    "scenario_name": environment._config["scenario_name"],
                    "grid2op_version": grid2op.__version__,
                },
                metadata={
                    "current_step": replay["current_step"],
                    "replayed_actions": replay["recorded_actions"],
                    "compression": GZIP_BASE64,
                },
            ).to_dict(),
        },
        "event": {"line": line},
        "options": {"max_recommendations": 3, "kpi_prediction_steps": 3},
    }

    response = client.post(
        "/api/v1/recommendation?use_case=PowerGrid", json=payload
    )

    assert response.status_code == 200
    recommendations = response.get_json()
    _assert_actionable(recommendations)
    # A multi-step projection must tag its steps so the UI can group branches.
    assert {reco["step"] for reco in recommendations} - {None}
    # `kpi_prediction_steps` is the depth *beyond* the recommended action, so each
    # branch spans that many policy steps plus the action itself.
    for branch in {reco["branch_index"] for reco in recommendations}:
        steps = sorted(
            reco["step"]
            for reco in recommendations
            if reco["branch_index"] == branch
        )
        assert steps == [1, 2, 3, 4]
    # The env really was rebuilt by replay, so it sits on the producer's step.
    assert int(environment.obs.current_step) == replay["current_step"]

    # `step` and `env_timestep` are different clocks, and only the latter places
    # the projection on the episode's timeline: rolling out from the producer's
    # step S, the k-th step of every branch stands at S + k.
    for reco in recommendations:
        assert reco["env_timestep"] == replay["current_step"] + reco["step"]
    # A branch that blacks out ends there, so `done` marks the last step of its
    # branch and nothing after it.
    for branch in {reco["branch_index"] for reco in recommendations}:
        steps = sorted(
            (reco["step"], reco["done"])
            for reco in recommendations
            if reco["branch_index"] == branch
        )
        assert not any(done for _, done in steps[:-1])


def test_unknown_serializer_is_rejected_with_400():
    """Ensure a context from an unreadable serializer is a client error."""
    app = create_app("test")
    client = app.test_client()

    response = client.post(
        "/api/v1/recommendation?use_case=PowerGrid",
        json={
            "context": {
                "environment_state": {
                    "serializer": "some_other_env_v9",
                    "state": {"whatever": 1},
                }
            },
            "event": {},
        },
    )

    assert response.status_code == 400


def test_a_corrupt_history_returns_no_recommendations():
    """Ensure an unreadable history yields nothing rather than a guess.

    The whole point of shipping a replay history is that the projection is
    computed on the operator's grid. A history that cannot be replayed must not
    quietly fall back to approximating the state.
    """
    app = create_app("test")
    client = app.test_client()

    obs, line, seed = _overloaded_observation(app)
    envelope = build_grid2op_observation_state(
        obs.to_json(),
        seed=seed,
        replay_actions="not-a-gzipped-history",
        identity={"scenario_name": "jan_28_1"},
        metadata={
            "current_step": int(obs.current_step),
            "compression": GZIP_BASE64,
        },
    ).to_dict()

    response = client.post(
        "/api/v1/recommendation?use_case=PowerGrid",
        json={"context": {"environment_state": envelope}, "event": {"line": line}},
    )

    assert response.status_code == 200
    assert response.get_json() == []
