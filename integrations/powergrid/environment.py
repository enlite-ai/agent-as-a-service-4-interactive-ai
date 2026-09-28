# Grid2Op-backed implementation of the generic Environment interface, including
# the exact reconstruction of a producer's environment from its replay history.
"""PowerGrid (Grid2Op) implementation of the generic Environment interface.

Encapsulates everything Grid2Op-specific: building the simulation env, seeding,
restoring state from a serialized environment (exactly, by replaying its recorded
action history, or approximately by fast-forwarding the chronics), stepping the
env one timestep, branching it (via a cheap Grid2Op ``copy``), deriving KPIs from
a real observation, and rendering a Grid2Op action into the normalized
recommendation shape. The generic rollout core and the agent talk to this only
through the :class:`Environment` interface (plus a small surface the PowerGrid
agent needs to build its assistant).

The identity and action-replay history used to restore the exact state (opponent
budget, cooldowns, accumulated redispatch) are PowerGrid-specific, so they are
carried *inside* the serialized ``state`` blob rather than on the generic
envelope.

Reconstruction fails closed. A KPI projection is only worth anything if it is
computed on the same grid the operator is looking at, and every way of being on a
*different* grid looks exactly like being on the right one:

* the producer replays a different scenario, so the history is applied to other
  chronics (checked against the published environment identity);
* the history arrives incomplete, so the replay stops short of - or diverges
  from - the producer's state (checked against the published action count, and
  against the state actually reached: timestep, topology, line status and line
  loading are all compared to the shipped observation);
* the two sides run different Grid2Op/LightSim2Grid builds, so the same actions
  yield a different power flow (the versions are compared, and any residual
  numeric drift shows up in the reached-state check).

Any of these returns "no usable state" rather than a plausible-looking
projection: the operator gets no recommendation instead of a wrong one.

Replaying is also the dominant cost of a request - about 1.6 ms per recorded step,
so ~3 s by the end of an episode - and a producer's history only ever grows by the
handful of actions taken since the previous context. The replayed prefix is
therefore cached: when the incoming history extends what this env already
replayed, only the new actions are stepped.
"""
from __future__ import annotations

import copy
import hashlib
import logging
import os

import grid2op
import lightsim2grid
import numpy as np
import toml
from grid2op.Chronics import FromHandlers
from grid2op.Chronics.handlers import CSVHandler, PerfectForecastHandler
from lightsim2grid import LightSimBackend

from .formatting import describe_action
from .kpi import compute_kpis
from .serialization import read_grid2op_state
from a3s_core import (
    A3SRecommendation,
    Environment,
    SerializedEnvironmentState,
)

logger = logging.getLogger(__name__)

BkClass = LightSimBackend

# Tolerances for comparing the continuous state this env reaches by replay against
# the state the producer shipped. Identical builds reproduce it bit-for-bit, so
# these only absorb JSON round-tripping and solver noise; a replay that landed on
# a different grid is out by orders of magnitude more.
STATE_ATOL = 1e-4
STATE_RTOL = 1e-3


class PowerGridEnvironment(Environment):
    """Grid2Op-backed environment adapter for the PowerGrid use case."""

    def __init__(self):
        """:return: None."""
        self._initialized = False
        # Seed carried by the current request (aligns the env's stochastic
        # components, e.g. the opponent, with the source environment). Falls back
        # to the config seed when the request does not carry one.
        self._active_seed = None
        self._config_seed = None
        # Action history carried by the current request. When present, the env
        # is reconstructed exactly by seeding + resetting + replaying these
        # actions (restoring hidden state such as opponent budget, cooldowns and
        # accumulated redispatch) instead of fast-forwarding with do-nothing.
        self._active_replay = None
        # Per-action digests of the history already replayed into ``self.env``,
        # in order, and the seed it was replayed under. ``None`` means the env is
        # not at a known replay position, so the next replay starts from reset.
        self._replayed_digests = None
        self._replayed_seed = None

    # -- Environment interface -------------------------------------------------

    def prepare(
        self, environment_state: SerializedEnvironmentState | None
    ) -> dict | None:
        """Ready the env for the serialized state and return the observation.

        Reads the observation and the PowerGrid-specific reconstruction inputs
        (environment identity, action-replay history) from inside the env-defined
        ``state`` blob, and refuses to hand back an observation it cannot vouch
        for.

        :param environment_state: Serialized environment-state envelope.
        :return: The reconstructed live Grid2Op observation, or ``None``.
        """
        try:
            state = read_grid2op_state(environment_state)
        except ValueError as exc:
            # An undecodable history is not a reason to silently fall back to
            # approximating the state: the caller believes it sent a replayable
            # environment, so say nothing rather than project from a guess.
            logger.error("Unusable PowerGrid environment state: %s", exc)
            return None
        if state is None:
            return None
        observation = state.get("observation")
        if not observation:
            return None

        self.ensure_initialized()
        if not self._identity_matches(state):
            return None

        self._active_seed = self._coerce_seed(state.get("seed"))
        self._active_replay = state.get("replay_actions") or None
        # `_sync_obs` reconstructs the live Grid2Op observation (self.obs) from
        # the serialized payload; the agent/simulator operate on that object,
        # not the raw dict. It fails when the state cannot be reconstructed
        # faithfully, in which case there is nothing trustworthy to project from.
        if not self._sync_obs(state):
            return None
        return self.obs

    def step(self, action) -> tuple:
        """Advance the Grid2Op env one timestep and return ``(obs, done)``.

        :param action: Grid2Op action to apply.
        :return: The observation reached and whether the episode terminated.
        """
        # Stepping walks this env off the position its replayed history describes,
        # so that history can no longer be reused as a prefix. (The rollout steps
        # forks, not the prepared env, but nothing here should depend on that.)
        self._invalidate_replay_cache()
        self.obs, _, done, _ = self.env.step(action)
        return self.obs, bool(done)

    def fork(self) -> "PowerGridEnvironment":
        """
        Returns an independent branch positioned at the current state.

        Uses Grid2Op's ``copy`` (cheap relative to stepping), so rolling out a
        branch never disturbs the prepared env or any sibling branch. The shallow
        copy shares the immutable loaded config/resources and carries over the
        rollout bookkeeping; only the live Grid2Op env is duplicated.

        :return PowerGridEnvironment: An independent environment branch.
        """
        self.ensure_initialized()
        clone = copy.copy(self)
        clone.env = self.env.copy()
        clone.obs = clone.env.get_obs()
        clone.action_do_nothing = clone.env.action_space({})
        # A branch is stepped away from the replayed position immediately, so it
        # must never be mistaken for an env sitting on a replayed prefix.
        clone._replayed_digests = None
        clone._replayed_seed = None
        return clone

    def kpis(self, observation) -> dict:
        """Compute PowerGrid KPIs from a real (reached) observation.

        :param observation: A Grid2Op observation reached by the env.
        :return: KPI dictionary (currently the worst line loading).
        """
        return compute_kpis(observation)

    def timestep(self, observation) -> int | None:
        """Read the Grid2Op step counter off a reached observation.

        :param observation: A Grid2Op observation reached by the env.
        :return int | None: The absolute Grid2Op step, or ``None`` when the
            observation carries no step counter (a Grid2Op env reports ``None``
            before its first reset).
        """
        current_step = getattr(observation, "current_step", None)
        if current_step is None:
            return None
        # Cast away numpy integer types: they do not survive `jsonify`.
        return int(current_step)

    def format_recommendation(
        self,
        observation,
        action,
        kpis: dict,
        agent_type: str,
        step: int,
        branch_index: int,
        *,
        done: bool,
        env_timestep: int | None,
    ) -> A3SRecommendation:
        """
        Converts a Grid2Op action into the normalized A3S response format.

        Pure presentation: derives the descriptive labels, packages the
        already-computed ``kpis`` (from :meth:`kpis`), and tags the title with a
        ``_step_<step>`` suffix identifying the rollout timestep. The simulated
        KPIs are attached only for a recognised action, so an unrecognised one
        keeps reporting just its label.

        :param observation: Grid2Op observation reached after the action.
        :param action: Grid2Op action taken at this rollout step.
        :param dict kpis: KPIs already computed for this observation.
        :param str agent_type: Source label of the proposing agent.
        :param int step: 1-based rollout timestep.
        :param int branch_index: 0-based index of the fanned-out candidate.
        :param bool done: Whether this step ended the episode (a Grid2Op game
            over), in which case ``kpis`` describe a terminal grid.
        :param int | None env_timestep: Absolute Grid2Op step of ``observation``.
        :return A3SRecommendation: Normalized recommendation object.
        """
        title, description, recommendation_kpis = describe_action(
            action, self.action_do_nothing
        )
        if title:
            recommendation_kpis.update(kpis)

        return A3SRecommendation(
            title=f"{title}_step_{step}",
            description=description,
            actions=[action.to_json()],
            agent_type=agent_type,
            kpis=recommendation_kpis,
            branch_index=branch_index,
            step=step,
            done=done,
            env_timestep=env_timestep,
        )

    # -- Surface used by the PowerGrid agent to build its assistant ------------

    def assistant_setup(self) -> tuple:
        """Provide what the PowerGrid agent needs to build its assistant.

        :return: ``(grid2op_env_copy, submission_path, assistant_seed)``.
        """
        self.ensure_initialized()
        submission_path = os.path.join(
            self._resource_path(self._config["assistant_path"]), "submission"
        )
        return (
            self.env.copy(),
            submission_path,
            int(self._config["assistant_seed"]),
        )

    # -- Internals -------------------------------------------------------------

    def _resource_path(self, config_value: str) -> str:
        """Resolve a path from ``CONFIG_POWERGRID.toml`` against the resource dir.

        The config file is copied verbatim from the upstream InteractiveAI
        PowerGrid use case, so its paths are relative to *that* checkout and
        carry a ``Ressources/`` prefix. The README has those directories copied
        flat into ``PowerGridgrid2op_poc_simulator/``, so only the last
        component survives - keeping the prefix would point at a directory that
        does not exist here (and Grid2Op would then quietly treat the missing
        path as the *name* of a downloadable environment).

        :param config_value: Path as written in the config file.

        :return: Absolute path to that resource in this checkout.
        """
        return os.path.join(self._resource_dir, os.path.basename(config_value.rstrip("/")))

    @staticmethod
    def _coerce_seed(seed) -> int | None:
        """Normalize a seed value carried in the state blob.

        :param seed: Raw seed value from the serialized state (may be ``None``).
        :return: The seed as an ``int``, or ``None`` when absent.
        """
        return None if seed is None else int(seed)

    def _effective_seed(self) -> int:
        """Return the seed to apply to the env: request seed, else config seed.

        :return: The seed used to make the environment reproducible.
        """
        return (
            self._active_seed
            if self._active_seed is not None
            else self._config_seed
        )

    def ensure_initialized(self) -> None:
        """Load the Grid2Op environment lazily.

        :return: None.
        """
        if self._initialized:
            return

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self._resource_dir = os.path.normpath(
            os.path.join(
                script_dir,
                "..",
                "..",
                "resources",
                "PowerGrid",
                "PowerGridgrid2op_poc_simulator",
            )
        )
        config_path = os.path.join(self._resource_dir, "CONFIG_POWERGRID.toml")
        self._config = toml.load(config_path)
        self._config_seed = int(self._config["env_seed"])

        env_name = self._resource_path(self._config["env_name"])
        if not os.path.isdir(env_name):
            # Grid2Op falls back to looking the argument up in its catalogue of
            # downloadable environments, so a missing directory surfaces as a
            # confusing "unknown environment" - say what is actually missing.
            raise FileNotFoundError(
                f"Grid2Op environment directory not found: {env_name} "
                "- fetch the PowerGrid resources as described in the README."
            )
        forecasts_horizons = [5, 10, 15, 20, 25, 30]
        self.env = grid2op.make(
            env_name,
            backend=BkClass(),
            data_feeding_kwargs={
                "gridvalueClass": FromHandlers,
                "gen_p_handler": CSVHandler("prod_p"),
                "load_p_handler": CSVHandler("load_p"),
                "gen_v_handler": CSVHandler("prod_v"),
                "load_q_handler": CSVHandler("load_q"),
                "h_forecast": forecasts_horizons,
                "gen_p_for_handler": PerfectForecastHandler(
                    "prod_p_forecasted"
                ),
                "load_p_for_handler": PerfectForecastHandler(
                    "load_p_forecasted"
                ),
                "load_q_for_handler": PerfectForecastHandler(
                    "load_q_forecasted"
                ),
            },
        )
        self.env.seed(self._effective_seed())

        self.id_scenario = 0
        for sc_id, subpath in enumerate(
            self.env.chronics_handler.real_data.subpaths
        ):
            if os.path.basename(subpath) == self._config["scenario_name"]:
                self.id_scenario = sc_id
                break

        self.env.set_id(self.id_scenario)
        self.obs = self.env.reset()
        self.previous_step = (
            "1"
            if self.obs.current_step is None
            else str(self.obs.current_step)
        )
        self.nb_timestep = 0
        self.action_do_nothing = self.env.action_space({})
        self._initialized = True

    def _identity_matches(self, state: dict) -> bool:
        """Check the incoming state was produced by an environment we can match.

        The scenario is the identity that must hold: it selects the chronics, so
        replaying a history recorded on another scenario silently lands on a
        different grid. Library versions are reported but not enforced - the
        reached-state check catches whatever numeric difference they cause, and
        failing on a harmless patch bump would take the service down for no
        safety gain.

        :param dict state: Normalized PowerGrid state blob.
        :return bool: Whether this env may replay the incoming history.
        """
        scenario = state.get("scenario_name")
        if scenario is not None and scenario != self._config["scenario_name"]:
            logger.error(
                "Refusing the incoming state: it was produced on scenario %r "
                "but this service plays %r, so its action history belongs to "
                "different chronics",
                scenario,
                self._config["scenario_name"],
            )
            return False

        expected_actions = state.get("expected_actions")
        received = len(state.get("replay_actions") or [])
        if expected_actions is not None and int(expected_actions) != received:
            logger.error(
                "Refusing the incoming state: the producer recorded %s actions "
                "but %d arrived, so the replay history is incomplete",
                expected_actions,
                received,
            )
            return False

        for key, installed in (
            ("grid2op_version", grid2op.__version__),
            ("lightsim2grid_version", lightsim2grid.__version__),
        ):
            produced = state.get(key)
            if produced is not None and produced != installed:
                logger.warning(
                    "%s mismatch: the state was produced with %s, this service "
                    "runs %s - replay may not reproduce the same power flow",
                    key,
                    produced,
                    installed,
                )
        return True

    @staticmethod
    def _action_vectors(replay_actions: list) -> list:
        """Turn the serialized history into Grid2Op action vectors.

        :param list replay_actions: Serialized actions, in step order.
        :return list: One float array per action, in step order.
        """
        return [
            np.asarray(serialized["vect"], dtype=float)
            for serialized in replay_actions
        ]

    @staticmethod
    def _digests(vectors: list) -> list:
        """Digest each action vector, for prefix-matching against a replayed env.

        :param list vectors: Action vectors, in step order.
        :return list: One digest per action, in step order.
        """
        return [
            hashlib.blake2b(vector.tobytes(), digest_size=16).digest()
            for vector in vectors
        ]

    def _replayable_prefix(self, digests: list) -> int:
        """Return how many leading actions this env has already replayed.

        Reusing the env's position is only sound if it was replayed under the
        same seed and the incoming history starts with exactly the actions
        already applied - a history that diverges anywhere has to be replayed
        from reset.

        :param list digests: Digests of the incoming history, in step order.
        :return int: Number of leading actions already applied to ``self.env``.
        """
        cached = self._replayed_digests
        if not cached or self._replayed_seed != self._effective_seed():
            return 0
        if len(cached) > len(digests):
            return 0
        return len(cached) if cached == digests[: len(cached)] else 0

    def _invalidate_replay_cache(self) -> None:
        """Forget the replayed position, forcing the next replay to start fresh.

        :return: None.
        """
        self._replayed_digests = None
        self._replayed_seed = None

    def _reconstruct_via_replay(self, state: dict) -> bool:
        """Rebuild the env exactly by replaying the recorded action history.

        Seeds and resets the environment, then replays every serialized action.
        Because the chronics are deterministic data and every stochastic
        component is driven by the seeded RNG, this reproduces the source
        environment exactly - including the hidden state an observation does not
        expose (opponent budget, cooldowns, accumulated redispatch). That exact
        reconstruction is what makes the multi-step KPI projection faithful.

        When the env already sits on a prefix of this history (the usual case:
        the producer pushed a new context a few steps later), only the actions
        after that prefix are stepped.

        The shipped observation is *verified* against the replayed one rather
        than overlaid onto it: any divergence means the projection would be
        computed on a different grid than the producer is standing on, which must
        fail rather than be papered over.

        :param dict state: Normalized PowerGrid state blob.
        :return bool: Whether the environment was reconstructed successfully.
        """
        observation = state["observation"]
        vectors = self._action_vectors(self._active_replay)
        digests = self._digests(vectors)
        reused = self._replayable_prefix(digests)

        if reused:
            logger.info(
                "Reusing %d already-replayed actions; stepping the remaining %d",
                reused,
                len(vectors) - reused,
            )
        else:
            self._invalidate_replay_cache()
            self.env.seed(self._effective_seed())
            self.env.set_id(self.id_scenario)
            self.env.reset()

        # Cleared for the duration of the stepping: if anything raises midway,
        # the env is left at an unknown position and must not be reused as a
        # prefix of anything.
        self._invalidate_replay_cache()
        for index in range(reused, len(vectors)):
            action = self.env.action_space.from_vect(vectors[index])
            _, _, done, info = self.env.step(action)
            if done:
                # The episode ended inside the history, so there is no live env
                # left to branch from: either the source episode really is over,
                # or the history does not match this environment. Neither can
                # yield a trustworthy projection.
                logger.warning(
                    "Replay hit game over at action %d/%d (%s); cannot project "
                    "from this state",
                    index + 1,
                    len(vectors),
                    info.get("exception"),
                )
                self.obs = None
                return False

        self._replayed_digests = digests
        self._replayed_seed = self._effective_seed()
        self.obs = self.env.get_obs()
        logger.info(
            "Replayed %d actions (%d newly stepped) to step %s (rho.max=%.4f, "
            "cooldowns=%d, dispatch_nonzero=%d)",
            len(vectors),
            len(vectors) - reused,
            int(self.obs.current_step),
            float(self.obs.rho.max()),
            int(self.obs.time_before_cooldown_line.sum()),
            int((self.obs.actual_dispatch != 0).sum()),
        )

        if not self._reached_state_matches(observation):
            self._invalidate_replay_cache()
            self.obs = None
            return False
        current_step = observation.get("current_step")
        if current_step:
            self.previous_step = current_step[0]
        return True

    def _reached_state_matches(self, observation: dict) -> bool:
        """Check the replayed env is standing on the grid the producer shipped.

        Compares *everything the producer published*, attribute by attribute,
        rather than a hand-picked summary: a history that is subtly wrong (an
        action dropped, a redispatch shifted by a step) can leave the timestep,
        the topology and even the line loadings looking right while the env's
        accumulated state has drifted, and any such difference invalidates the
        projection. Discrete attributes must match exactly; continuous ones are
        compared within a tolerance that identical builds beat by orders of
        magnitude.

        :param dict observation: Incoming serialized observation payload.
        :return bool: Whether the reached state matches the shipped one.
        """
        try:
            expected_step = int(observation["current_step"][0])
        except (KeyError, IndexError, TypeError, ValueError):
            # Without a timestep there is nothing to anchor the comparison on;
            # the state cannot be vouched for, so it is not used.
            logger.error(
                "Shipped observation carries no usable current_step; cannot "
                "verify the replayed environment"
            )
            return False

        actual_step = int(self.obs.current_step)
        if actual_step != expected_step:
            logger.error(
                "Replayed env is at step %s but the shipped observation is at "
                "%s: the action history does not match this environment",
                actual_step,
                expected_step,
            )
            return False

        # Read the shipped payload back into an observation of this env's own
        # class, so the comparison is attribute-to-attribute rather than against
        # raw JSON. Starting from a copy of the reached observation means any
        # attribute the producer did not publish simply compares equal to itself
        # and is skipped below.
        shipped = self.obs.copy()
        try:
            shipped.from_json(observation)
        except Exception as exc:
            logger.error(
                "Shipped observation cannot be read back for verification: %s",
                exc,
            )
            return False

        for attribute in type(self.obs).attr_list_vect:
            if attribute not in observation:
                # Only what the producer actually published can be compared.
                continue
            if not self._attribute_matches(
                attribute,
                getattr(self.obs, attribute, None),
                getattr(shipped, attribute, None),
            ):
                return False
        return True

    @staticmethod
    def _attribute_matches(attribute: str, reached, shipped) -> bool:
        """Compare one observation attribute between the reached and shipped state.

        :param str attribute: Name of the observation attribute.
        :param reached: Value the replayed environment reached.
        :param shipped: Value the producer published.
        :return bool: Whether the two agree closely enough to project from.
        """
        if reached is None or shipped is None:
            return True
        reached_array = np.asarray(reached).reshape(-1)
        shipped_array = np.asarray(shipped).reshape(-1)
        if reached_array.shape != shipped_array.shape:
            logger.error(
                "Shipped %s has %d entries but this grid has %d: the state was "
                "produced on a different grid",
                attribute,
                shipped_array.size,
                reached_array.size,
            )
            return False

        reached_array = reached_array.astype(float)
        shipped_array = shipped_array.astype(float)
        # Integer- and boolean-valued attributes (topology, line status,
        # cooldowns, maintenance timers) are identities, not measurements: they
        # either match or the two sides are on different grids.
        exact = np.issubdtype(np.asarray(reached).dtype, np.integer) or np.issubdtype(
            np.asarray(reached).dtype, np.bool_
        )
        if exact:
            matches = np.array_equal(
                reached_array, shipped_array, equal_nan=True
            )
        else:
            matches = np.allclose(
                reached_array,
                shipped_array,
                atol=STATE_ATOL,
                rtol=STATE_RTOL,
                equal_nan=True,
            )
        if not matches:
            difference = np.abs(reached_array - shipped_array)
            worst = int(np.nanargmax(difference))
            logger.error(
                "Replayed env diverges from the shipped observation on %s "
                "(%s vs %s at index %d): refusing to project KPIs from a grid "
                "the operator is not looking at",
                attribute,
                reached_array[worst],
                shipped_array[worst],
                worst,
            )
            return False
        return True

    def _reset_obs_if_needed(self, observation: dict) -> None:
        """Reset the runtime if the received timestep goes backwards.

        :param observation: Incoming serialized observation payload.
        :return: None.
        """
        if self.nb_timestep < 0:
            # Re-seed before reset so the env's stochastic components (opponent)
            # are reproducible and aligned with the active seed.
            self.env.seed(self._effective_seed())
            self.env.set_id(self.id_scenario)
            self.obs = self.env.reset()
            self.previous_step = "1"
            self._get_nb_of_timestep_since_last_obs(observation)

    def _get_nb_of_timestep_since_last_obs(self, observation: dict) -> int:
        """Compute the timestep delta from the last synchronized observation.

        :param observation: Incoming serialized observation payload.
        :return: Timestep delta relative to the last synced state.
        :raises ValueError: If the observation carries no readable timestep.
        """
        try:
            current_step = int(observation["current_step"][0])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(
                "Observation carries no usable current_step"
            ) from exc
        self.nb_timestep = current_step - int(self.previous_step)
        return self.nb_timestep

    def _apply_observation_overlay(self, observation: dict) -> None:
        """Overlay the incoming observable state onto the current env obs.

        :param observation: Incoming serialized observation payload.
        :return: None.
        """
        self.obs = self.env.get_obs()
        self.obs.from_json(observation)
        self.obs._env_internal_params["_line_status_env"] = (
            self.obs.line_status.astype(int)
        )

    def _fast_forward_to(self, observation: dict) -> bool:
        """Approximate the incoming state by fast-forwarding the chronics.

        The fallback for a producer that sends no replay history: the observable
        state is overlaid onto an env advanced to the same timestep, which leaves
        the env's hidden state (opponent budget, cooldowns, accumulated
        redispatch) as whatever do-nothing stepping produced. Projections from
        here are approximate by construction.

        :param dict observation: Incoming serialized observation payload.
        :return bool: Whether the runtime now holds a usable observation.
        """
        # This walks the env away from any replayed position.
        self._invalidate_replay_cache()
        try:
            self._get_nb_of_timestep_since_last_obs(observation)
            self._reset_obs_if_needed(observation)
            if self.nb_timestep > 1:
                self.env.fast_forward_chronics(self.nb_timestep)
                self.previous_step = observation["current_step"][0]
            elif self.nb_timestep == 1:
                self.env.step(self.action_do_nothing)
                self.previous_step = observation["current_step"][0]
            self._apply_observation_overlay(observation)
        except ValueError as exc:
            logger.error("Cannot position the environment: %s", exc)
            return False
        logger.info(
            "No replay history: approximated the state by fast-forwarding to "
            "step %s (hidden env state is not restored)",
            self.previous_step,
        )
        return True

    def _sync_obs(self, state: dict) -> bool:
        """Synchronize the runtime observation with the incoming timestep.

        When the request carries an action-replay history, the env is rebuilt
        exactly from it; otherwise the runtime is fast-forwarded to the incoming
        timestep and the observation is overlaid.

        :param dict state: Normalized PowerGrid state blob.
        :return bool: Whether the runtime now holds a usable observation.
        """
        if self._active_replay:
            return self._reconstruct_via_replay(state)
        return self._fast_forward_to(state["observation"])
