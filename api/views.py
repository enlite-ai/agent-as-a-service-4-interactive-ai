from a3s_core import A3SRecommendationRequest
from apiflask import APIBlueprint, abort
from apiflask.views import MethodView
from flask import current_app, jsonify, request

from .schemas import RecommendationAsk, RecommendationOut

api_bp = APIBlueprint("a3s-api", __name__, url_prefix="/api/v1")


class HealthCheck(MethodView):
    """Expose a minimal liveness endpoint."""

    def get(self):
        """:return: Simple health payload confirming the service is up."""
        return {"message": "Ok"}


class RecommendationView(MethodView):
    """Serve backend recommendations for a given use case."""

    @api_bp.input(RecommendationAsk)
    @api_bp.output(RecommendationOut(many=True), status_code=200)
    def post(self, data):
        """Build an A3S request and return normalized recommendations.

        :param data: Parsed API request body.
        :return: JSON response containing A3S recommendations.
        """
        # The legacy T2.1_deep_expert endpoint this service can stand in for
        # takes no `use_case` query parameter, so fall back to the configured
        # default instead of rejecting the request.
        use_case_name = request.args.get("use_case") or current_app.default_use_case
        use_case = current_app.use_cases.get(use_case_name)
        service = current_app.services.get(use_case_name)
        if use_case is None or service is None:
            return abort(400, f"Unknown use_case: {use_case_name}")

        options = data.get("options") or {}
        try:
            # How a context is read is the use case's own concern, so the API
            # layer delegates rather than knowing any environment's payload shape.
            environment_state = use_case.resolve_environment_state(
                data.get("context") or {}
            )
        except ValueError as exc:
            # A context produced by a serializer this build cannot read is a
            # caller-side problem, not a server fault.
            return abort(400, str(exc))
        recommendations = service.get_recommendations(
            A3SRecommendationRequest(
                use_case=use_case_name,
                environment_state=environment_state,
                event=data.get("event", {}),
                max_recommendations=options.get("max_recommendations", 3),
                # How many policy timesteps to project *after* the recommended
                # action; 0 (the default) reproduces the legacy single-step
                # response. The legacy "kpi_prediction_steps" options key is
                # accepted as a fallback so untouched upstream callers keep
                # working.
                n_steps=options.get(
                    "n_steps", options.get("kpi_prediction_steps", 0)
                ),
            )
        )
        return jsonify(
            [recommendation.to_dict(use_case=use_case_name) for recommendation in recommendations]
        )


api_bp.add_url_rule("/health", view_func=HealthCheck.as_view("health"))
api_bp.add_url_rule(
    "/recommendation", view_func=RecommendationView.as_view("recommendation")
)
