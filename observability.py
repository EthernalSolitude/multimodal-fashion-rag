"""Observability: structured logs + Prometheus metrics.

Usage:
    from observability import configure_logging, timed, log, search_requests, llm_errors

    configure_logging()
    with timed("dense_search"):
        ...
"""
import logging
import time
import uuid
from contextlib import contextmanager

import structlog
from prometheus_client import Counter, Histogram

search_duration = Histogram(
    "search_duration_seconds",
    "Duration of individual search pipeline stages",
    labelnames=("stage",),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

search_requests = Counter(
    "search_requests_total",
    "Total number of search requests by endpoint and status",
    labelnames=("endpoint", "status"),
)

llm_errors = Counter(
    "llm_errors_total",
    "Total number of LLM call errors by type",
    labelnames=("type",),
)

guardrail_rejections = Counter(
    "guardrail_rejections_total",
    "Requests rejected by the fashion-topic guardrail",
    labelnames=("reason",),
)


def configure_logging() -> None:
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )


log = structlog.get_logger()


def new_request_id() -> str:
    rid = uuid.uuid4().hex[:8]
    structlog.contextvars.bind_contextvars(request_id=rid)
    return rid


def clear_context() -> None:
    structlog.contextvars.clear_contextvars()


@contextmanager
def timed(stage: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dur = time.perf_counter() - t0
        search_duration.labels(stage=stage).observe(dur)
        log.info("stage_complete", stage=stage, duration_ms=round(dur * 1000, 1))
