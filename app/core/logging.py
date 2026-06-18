import logging
import logging.handlers
import os
import re
from datetime import datetime

import structlog
from app.core.config import settings


def setup_logging() -> None:
    log_level_str: str = str(settings.LOG_LEVEL)
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    environment: str = str(settings.ENVIRONMENT)

    os.makedirs("logs", exist_ok=True)

    # Processors shared between both handlers and foreign (non-structlog) loggers
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    console_renderer = (
        structlog.dev.ConsoleRenderer()
        if environment == "development"
        else structlog.processors.JSONRenderer()
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, console_renderer],
            foreign_pre_chain=shared_processors,
        )
    )

    def _namer(name: str) -> str:
        # "logs/app-2026-06-17.log.2026-06-18" → "logs/app-2026-06-18.log"
        m = re.search(r"\.(\d{4}-\d{2}-\d{2})$", name)
        return f"logs/app-{m.group(1)}.log" if m else name

    today = datetime.now().strftime("%Y-%m-%d")
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=f"logs/app-{today}.log",
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.namer = _namer
    file_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, structlog.processors.JSONRenderer()],
            foreign_pre_chain=shared_processors,
        )
    )

    root = logging.getLogger()
    root.setLevel(log_level)
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
