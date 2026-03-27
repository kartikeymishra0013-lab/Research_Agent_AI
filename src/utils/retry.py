"""
Retry decorators for OpenAI API calls using Tenacity.

Provides:
  - with_retry(...)  — configurable decorator factory
  - openai_retry     — pre-configured for chat/completion calls (4 attempts)
  - embedding_retry  — pre-configured for embedding calls (3 attempts)

Retried exceptions:
  openai.RateLimitError, openai.APIConnectionError,
  openai.APITimeoutError, openai.InternalServerError
"""
from __future__ import annotations

import logging
from functools import wraps

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


def _openai_retryable_exceptions():
    """Return the tuple of OpenAI exceptions that should trigger a retry."""
    try:
        from openai import (
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        )
        return (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
    except ImportError:
        # Fallback: retry on any OSError / ConnectionError so tests still work
        return (OSError, ConnectionError)


def with_retry(
    max_attempts: int = 4,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
    multiplier: float = 2.0,
):
    """
    Decorator factory that wraps a callable with tenacity exponential-backoff retry.

    Args:
        max_attempts : Maximum number of attempts (including the first call).
        min_wait     : Minimum wait between retries in seconds.
        max_wait     : Maximum wait between retries in seconds.
        multiplier   : Exponential backoff multiplier.

    Usage:
        @with_retry(max_attempts=3)
        def call_api(): ...

        # Or inline:
        result = with_retry(max_attempts=2)(my_function)()
    """
    def decorator(func):
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

            return _call()

        return wrapper

    return decorator


# Pre-configured variants
openai_retry   = with_retry(max_attempts=4, min_wait=1,  max_wait=60)
embedding_retry = with_retry(max_attempts=3, min_wait=2,  max_wait=30)
