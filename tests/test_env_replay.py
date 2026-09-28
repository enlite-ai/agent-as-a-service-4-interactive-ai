"""Fidelity tests for reconstructing the PowerGrid env from a replay history.

These are the tests that matter for KPI trustworthiness: they check not that the
env *loads*, but that the reconstructed env is the same grid the producer was
standing on and *continues identically*. An observation can match while the
underlying env is wrong, which yields plausible-looking but bogus projections, so
each test compares env-internal state and the next reached step - not just the
observation.

Requires the Grid2Op stack, so it runs where the service's dependencies are
installed (container / CI), not on a bare host.
"""
import base64
import gzip
import json

import numpy as np
import pytest

from integrations.powergrid.serialization import (
    GZIP_BASE64,
    build_grid2op_observation_state,
)

grid2op = pytest.importorskip("grid2op")


@pytest.fixture(scope="module")
def template():
    """Build the service's PowerGrid environment once for the module.

    :return PowerGridEnvironment: An initialized environment adapter.
    """
    from integrations.powergrid.environment import PowerGridEnvironment

    environment = PowerGridEnvironment()
    environment.ensure_initialized()
    return environment


def _produce(template, steps, redispatch_every):
    """Act as the simulator: step a private env, recording every applied action.

    Redispatch is injected periodically because accumulated dispatch is exactly
    the kind of hidden env state that an observation overlay cannot restore.

    :param PowerGridEnvironment template: Adapter holding the template env.
    :param int steps: Number of steps to advance.
    :param int redispatch_every: Apply a redispatch action every N steps.
    :return tuple: ``(producer_env, observation, history)``.
    """
    producer = template.env.copy()
    producer.seed(template._effective_seed())
    producer.set_id(template.id_scenario)
    obs = producer.reset()

    dispatchable = [
        gen for gen in range(producer.n_gen) if producer.gen_redispatchable[gen]
    ]
    history = []
    for step in range(steps):
        if dispatchable and step % redispatch_every == redispatch_every - 1:
            gen = dispatchable[step % len(dispatchable)]
            action = producer.action_space({"redispatch": [(gen, 1.0)]})
        else:
            action = producer.action_space({})
        history.append({"vect": action.to_vect().tolist()})
        obs, _, done, _ = producer.step(action)
        if done:
            pytest.skip("scenario ended before the producer state was built")

    return producer, obs, history


def _envelope(observation, seed, history):
    """Build the (older, uncompressed and identity-less) v1 envelope.

    :param dict observation: Serialized producer observation.
    :param int seed: Seed the producer's env was seeded with.
    :param list history: Recorded action history.
    :return SerializedEnvironmentState: The published envelope.
    """
    return build_grid2op_observation_state(
        observation,
        seed=seed,
        replay_actions=history,
        metadata={"current_step": int(observation["current_step"][0])},
    )


def _published_envelope(observation, seed, history, **identity):
    """Build the envelope a current simulator publishes: compressed, identified.

    Mirrors ``env_serialization.build_environment_state`` on the producer side,
    down to the framing and the recorded action count, so these tests exercise
    the shape that actually arrives in production.

    :param dict observation: Serialized producer observation.
    :param int seed: Seed the producer's env was seeded with.
    :param list history: Recorded action history.
    :param identity: Environment-identity overrides (e.g. ``scenario_name``).
    :return SerializedEnvironmentState: The published envelope.
    """
    published = {
        "seed": seed,
        "scenario_name": "jan_28_1",
        "grid2op_version": grid2op.__version__,
    }
    published.update(identity)
    encoded = base64.b64encode(
        gzip.compress(json.dumps(history).encode("utf-8"))
    ).decode("ascii")
    return build_grid2op_observation_state(
        observation,
        seed=published.pop("seed"),
        replay_actions=encoded,
        identity=published,
        metadata={
            "current_step": int(observation["current_step"][0]),
            "replayed_actions": len(history),
            "compression": GZIP_BASE64,
        },
    )


def test_replay_reproduces_the_producer_exactly(template):
    """Ensure replaying the history rebuilds the producer's grid bit-for-bit."""
    producer, obs, history = _produce(template, steps=60, redispatch_every=7)
    assert (obs.actual_dispatch != 0).any(), "test needs accumulated dispatch"

    consumer = type(template)()
    restored = consumer.prepare(
        _envelope(obs.to_json(), int(template._effective_seed()), history)
    )

    assert restored is not None
    assert int(restored.current_step) == int(obs.current_step)
    assert np.allclose(restored.to_vect(), obs.to_vect(), equal_nan=True)
    # Hidden, env-internal state an observation overlay cannot restore.
    assert np.array_equal(
        consumer.env._times_before_line_status_actionable,
        producer._times_before_line_status_actionable,
    )
    assert np.array_equal(
        consumer.env._times_before_topology_actionable,
        producer._times_before_topology_actionable,
    )
    assert np.allclose(restored.actual_dispatch, obs.actual_dispatch)


def test_replayed_env_continues_identically(template):
    """Ensure the rebuilt env steps to the same future as the producer would.

    This is the property KPI projection depends on, and the one a restored-but-
    broken env fails while still reporting a matching observation.
    """
    producer, obs, history = _produce(template, steps=60, redispatch_every=7)

    consumer = type(template)()
    consumer.prepare(
        _envelope(obs.to_json(), int(template._effective_seed()), history)
    )

    producer_next, _, producer_done, _ = producer.step(producer.action_space({}))
    branch = consumer.fork()
    consumer_next, consumer_done = branch.step(branch.action_do_nothing)

    assert consumer_done == producer_done
    assert not consumer_done, "grid collapsed; cannot compare trajectories"
    assert np.allclose(
        consumer_next.to_vect(), producer_next.to_vect(), equal_nan=True
    )
    # A KPI computed on that step therefore matches the producer's reality.
    assert np.isclose(
        branch.kpis(consumer_next)["efficiency_of_the_reco"],
        float(producer_next.rho.max()),
    )


def test_forking_does_not_disturb_the_prepared_env(template):
    """Ensure rolling out a branch leaves the prepared env where replay put it."""
    _, obs, history = _produce(template, steps=30, redispatch_every=7)

    consumer = type(template)()
    consumer.prepare(
        _envelope(obs.to_json(), int(template._effective_seed()), history)
    )
    prepared_step = int(consumer.obs.current_step)

    branch = consumer.fork()
    branch.step(branch.action_do_nothing)

    assert int(consumer.obs.current_step) == prepared_step


def test_replay_is_preferred_over_fast_forwarding(template):
    """Ensure the fast-forward fallback really is the lower-fidelity path.

    Without a history the env can only be fast-forwarded, which does not restore
    accumulated dispatch — so the same observation reconstructs into a different
    env. This asserts the gap the replay history exists to close.
    """
    _, obs, history = _produce(template, steps=60, redispatch_every=7)
    seed = int(template._effective_seed())

    replayed = type(template)()
    replayed.prepare(_envelope(obs.to_json(), seed, history))

    fast_forwarded = type(template)()
    fast_forwarded.prepare(
        build_grid2op_observation_state(obs.to_json(), seed=seed)
    )

    # Both report the producer's step (the fallback overlays the observation)...
    assert int(fast_forwarded.obs.current_step) == int(obs.current_step)
    # ...but only the replayed env's *internals* carry the accumulated dispatch.
    assert np.allclose(replayed.env._actual_dispatch, obs.actual_dispatch)
    assert not np.allclose(
        fast_forwarded.env._actual_dispatch, obs.actual_dispatch
    )


def test_replay_history_survives_the_json_round_trip(template):
    """Ensure the action vectors stay replayable after JSON serialization."""
    import json

    _, obs, history = _produce(template, steps=20, redispatch_every=7)
    envelope = _envelope(
        obs.to_json(), int(template._effective_seed()), history
    )
    round_tripped = json.loads(json.dumps(envelope.to_dict()))

    from a3s_core import extract_serialized_state

    consumer = type(template)()
    restored = consumer.prepare(
        extract_serialized_state({"environment_state": round_tripped})
    )

    assert restored is not None
    assert int(restored.current_step) == int(obs.current_step)


def test_published_compressed_state_is_replayable(template):
    """Ensure the shape the simulator actually publishes reconstructs exactly."""
    _, obs, history = _produce(template, steps=40, redispatch_every=7)

    consumer = type(template)()
    restored = consumer.prepare(
        _published_envelope(
            obs.to_json(), int(template._effective_seed()), history
        )
    )

    assert restored is not None
    assert int(restored.current_step) == int(obs.current_step)
    assert np.allclose(restored.actual_dispatch, obs.actual_dispatch)


def test_a_history_with_a_gap_is_refused(template):
    """Ensure a history that is not the one the producer applied yields no state.

    This is the failure mode a broken recording produces: the history has the
    right length, so no count check fires, but one of its actions is not the
    action the producer took — and replaying it lands on a grid that merely looks
    plausible. What is verified is the state actually reached, not the history
    itself, so the action dropped here is one that leaves a lasting mark
    (accumulated redispatch) rather than one the env would have converged past.
    """
    _, obs, history = _produce(template, steps=40, redispatch_every=7)
    do_nothing = history[0]
    swapped = next(
        index
        for index in reversed(range(len(history)))
        if history[index] != do_nothing
    )
    with_gap = list(history)
    with_gap[swapped] = do_nothing

    consumer = type(template)()
    restored = consumer.prepare(
        _published_envelope(
            obs.to_json(), int(template._effective_seed()), with_gap
        )
    )

    assert restored is None, "a wrong grid must not be projected from"
    assert consumer.obs is None


def test_an_incomplete_history_is_refused(template):
    """Ensure a history missing actions in transit is refused on the count."""
    _, obs, history = _produce(template, steps=40, redispatch_every=7)
    envelope = _published_envelope(
        obs.to_json(), int(template._effective_seed()), history[:-5]
    )
    # The producer recorded the full history; only part of it arrived.
    envelope.metadata["replayed_actions"] = len(history)

    consumer = type(template)()

    assert consumer.prepare(envelope) is None


def test_a_history_from_another_scenario_is_refused(template):
    """Ensure a history recorded on other chronics is refused on identity."""
    _, obs, history = _produce(template, steps=20, redispatch_every=7)

    consumer = type(template)()
    restored = consumer.prepare(
        _published_envelope(
            obs.to_json(),
            int(template._effective_seed()),
            history,
            scenario_name="feb_40_2",
        )
    )

    assert restored is None


def test_a_corrupt_history_yields_no_state(template):
    """Ensure an undecodable history does not silently fall back to guessing."""
    _, obs, history = _produce(template, steps=20, redispatch_every=7)
    envelope = _published_envelope(
        obs.to_json(), int(template._effective_seed()), history
    )
    envelope.state["replay_actions"] = "}}}not-gzip{{{"

    consumer = type(template)()

    assert consumer.prepare(envelope) is None


def test_a_library_version_mismatch_still_replays(template):
    """Ensure a version difference is reported, not fatal.

    The reached-state check is what decides whether a differing build actually
    changed the grid, so a version mismatch alone must not take the service down.
    """
    _, obs, history = _produce(template, steps=20, redispatch_every=7)

    consumer = type(template)()
    restored = consumer.prepare(
        _published_envelope(
            obs.to_json(),
            int(template._effective_seed()),
            history,
            lightsim2grid_version="0.0.1-from-the-future",
        )
    )

    assert restored is not None


def test_an_extending_history_reuses_the_replayed_prefix(template):
    """Ensure a follow-up context only steps the actions it added.

    Replay dominates request latency and grows with the episode, so the common
    case — the producer pushed a new context a few steps later — must not replay
    from reset again.
    """
    producer, obs, history = _produce(template, steps=40, redispatch_every=7)
    seed = int(template._effective_seed())

    consumer = type(template)()
    assert consumer.prepare(_published_envelope(obs.to_json(), seed, history)) \
        is not None

    # The producer advances a few more steps and publishes again.
    extra = []
    for _ in range(5):
        action = producer.action_space({})
        extra.append({"vect": action.to_vect().tolist()})
        later_obs, _, done, _ = producer.step(action)
        assert not done, "scenario ended; cannot extend the history"

    stepped = []
    original_step = consumer.env.step

    def counting_step(action):
        """Count how many actions the env is asked to apply.

        :param action: Grid2Op action being applied.
        :return tuple: Whatever the wrapped env returns.
        """
        stepped.append(action)
        return original_step(action)

    consumer.env.step = counting_step
    restored = consumer.prepare(
        _published_envelope(later_obs.to_json(), seed, history + extra)
    )
    consumer.env.step = original_step

    assert restored is not None
    assert int(restored.current_step) == int(later_obs.current_step)
    assert len(stepped) == len(extra), (
        f"replayed {len(stepped)} actions instead of the {len(extra)} added"
    )


def test_a_reseeded_request_does_not_reuse_the_prefix(template):
    """Ensure the cached prefix is only reused under the seed it was built with.

    The seed drives the opponent, so the same actions under a different seed are
    a different environment and must be replayed from reset.
    """
    _, obs, history = _produce(template, steps=20, redispatch_every=7)
    seed = int(template._effective_seed())

    consumer = type(template)()
    consumer.prepare(_published_envelope(obs.to_json(), seed, history))

    consumer._active_seed = seed + 1
    assert consumer._replayable_prefix(
        consumer._digests(consumer._action_vectors(history))
    ) == 0
