from apiflask import Schema
from apiflask.fields import Boolean, Dict, Integer, List, String
from marshmallow import EXCLUDE


class RecommendationAsk(Schema):
    """Input schema for recommendation requests."""

    class Meta:
        # Callers (e.g. the recommendation-service) forward their full
        # request payload verbatim, which may carry fields this service
        # doesn't use (e.g. cognitive_snapshot) — ignore them instead of
        # rejecting the whole request with a 422.
        unknown = EXCLUDE

    context = Dict()
    event = Dict()
    options = Dict()


class RecommendationOut(Schema):
    """Output schema for normalized recommendation objects."""

    title = String()
    description = String()
    use_case = String()
    agent_type = String()
    actions = List(Dict())
    kpis = Dict(allow_none=True)
    branch_index = Integer(allow_none=True)
    step = Integer(allow_none=True)
    # Named after the equivalent fields of the AI4REALNET WP3 reference service.
    # `step` counts rollout timesteps from 1; `env_timestep` is the environment's
    # own absolute clock.
    done = Boolean()
    env_timestep = Integer(allow_none=True)
