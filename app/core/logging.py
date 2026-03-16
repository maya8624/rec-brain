import logging
import structlog
from app.core.config import settings


def setup_logging() -> None:
    """
    Configure structlog for structured JSON logging.
    Call once at app startup in main.py.
    """
    log_level_str: str = str(settings.LOG_LEVEL)
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    environment: str = str(settings.ENVIRONMENT)

    logging.basicConfig(
        format="%(message)s",
        level=log_level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            # JSON in production, colored in dev
            structlog.dev.ConsoleRenderer()
            if environment == "development"
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
