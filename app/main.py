import logging
import logging.config
from typing import Any, Dict

from fastapi import FastAPI

from app.routes import router
from app.schemas import StatusOutput

LOGGING_CONFIG: Dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "default": {
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
    },
}


def setup_logging():
    """
    Applies the standardized logging configuration.
    """
    logging.config.dictConfig(LOGGING_CONFIG)


setup_logging()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Summarization API",
    description="Abstractive Text Summarization Service powered by BART-Large.",
    version="1.0.0",
)

app.include_router(router)


@app.get("/health", response_model=StatusOutput, tags=["monitoring"])
def health_check():
    """
    Performs a basic liveness check for the API service.
    """
    logger.info("Health check endpoint called.")

    return StatusOutput(status="ok", model_status="Loading (Awaiting first request)")


logger.info("FastAPI application instance created.")
