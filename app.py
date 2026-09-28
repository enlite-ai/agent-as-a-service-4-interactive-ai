from api.utils import build_registry, default_use_case
from api.views import api_bp
from apiflask import APIFlask

from config import DevConfig, ProdConfig, TestConfig, logger

config_mapping = {"dev": DevConfig, "test": TestConfig, "prod": ProdConfig}


def create_app(config_mode):
    """Create and configure the A3S Flask application.

    :param config_mode: Named configuration profile to load.
    :return: Configured APIFlask application instance.
    """
    logger.info(f"starting a3s-service in {config_mode} mode")
    app = APIFlask("a3s-service")
    app.register_blueprint(api_bp)
    app.config.from_object(config_mapping.get(config_mode, DevConfig))

    # `use_cases` read request contexts, `services` answer requests; both are
    # keyed by use-case name and neither is known to this module by name.
    app.use_cases, app.services = build_registry()
    app.default_use_case = default_use_case(app.use_cases)

    return app


if __name__ == "__main__":
    create_app("dev")
