"""
Retry decorators for OpenAI API calls using tenacity.
Handles rate limits, connection errors, and transient server errors
with exponential backoff.
"""
from __future__ import annotations

import logging
from functools import wraps
from typing import Callable, Type, Tuple

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    RetryError,
)

logger = logging.getLogger(__name__)


def _openai_retryable_exceptions() -> Tuple[Type[Exception], ...]:
    """Return retryable OpenAI exception types (imported lazily)."""
    from openai import (
        RateLimitError,
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
    )
    return (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)


def with_retry(
    max_attempts: int = 4,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
    multiplier: float = 2.0,
):
    """
    Decorator that wraps a function with tenacity retry logic.

    Default behavior: up to 4 attempts, exponential backoff 1s → 2s → 4s → max 60s.
    Only retries on OpenAI rate-limit, connection, timeout, and server errors.

    Usage:
        @with_retry(max_attempts=3)
        def call_openai(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            exceptions = _openai_retryable_exceptions()

            @retry(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(multiplier=multiplier, min=min_wait, max=max_wait),
                retry=retry_if_exception_type(exceptions),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True,
            )
            def _call():
                return func(*args, **kwargs)

            try:
                return _call()
            except RetryError as e:
                logger.error(f"All {max_attempts} retry attempts exhausted for {func.__name__}: {e}")
                raise
            except exceptions as e:
                logger.error(f"Non-retryable error in {func.__name__}: {e}")
                raise

        return wrapper
    return decorator


# Pre-configured variants for convenience
openai_retry = with_retry(max_attempts=4, min_wait=1, max_wait=60)
embedding_retry = with_retry(max_attempts=3, min_wait=2, max_wait=30)
