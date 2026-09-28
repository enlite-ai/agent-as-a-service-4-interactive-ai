import logging.config

DEFAULT_LOGGING = {
    "version": 1,
    "formatters": {
        "default": {
            "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
        }
    },
    "handlers": {
        "wsgi": {
            "class": "logging.StreamHandler",
            "stream": "ext://flask.logging.wsgi_errors_stream",
            "formatter": "default",
        }
    },
    "root": {"level": "INFO", "handlers": ["wsgi"]},
}
logging.config.dictConfig(DEFAULT_LOGGING)

logger = logging.getLogger(__name__)


# Flask configuration profiles selected by name in app.create_app.
class Config:
    TESTING = False


class DevConfig(Config):
    pass


class TestConfig(Config):
    TESTING = True


class ProdConfig(Config):
    pass
