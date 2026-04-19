"""공용 HTTP 클라이언트와 재시도 데코레이터.

`httpx.Client` 를 팩토리로 노출하고, 논문 검색 API 용 재시도 정책을 `http_retry`
데코레이터로 제공한다. 재시도 정책은 타임아웃과 '재시도할 가치가 있는' HTTP 에러
(429 / 5xx) 에만 반응하며, 그 외 4xx 는 즉시 전파한다.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from research_agent.config import get_settings
from research_agent.logger import logger


def _build_user_agent() -> str:
    """요청 헤더에 실을 User-Agent 를 만든다.

    OpenAlex polite pool 규약처럼 contact email 이 있으면 포함시켜 soft rate-limit 혜택을 받는다.
    """

    mailto = get_settings().openalex_mailto
    if mailto:
        return f"research-agent/0.1 (+contact:{mailto})"
    return "research-agent/0.1"


def get_http_client() -> httpx.Client:
    """타임아웃/UA/리다이렉트가 프로젝트 기본값으로 세팅된 `httpx.Client` 인스턴스.

    호출부에서 `with get_http_client() as client:` 로 컨텍스트 관리하는 것이 안전하다.
    """

    settings = get_settings()
    return httpx.Client(
        timeout=settings.request_timeout_seconds,
        follow_redirects=True,
        headers={"User-Agent": _build_user_agent()},
    )


def _should_retry_http_error(exc: BaseException) -> bool:
    """타임아웃 또는 429/5xx 응답이면 True.

    `HTTPStatusError` 는 `response.raise_for_status()` 후에 발생하는데, 그 중에서도
    4xx 는 대부분 재시도해도 의미가 없으므로 429(rate limit) 만 예외적으로 재시도한다.
    """

    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return False


def _log_before_sleep(retry_state: RetryCallState) -> None:
    """tenacity 가 다음 시도 전에 호출. 경고 로그로 백오프 상황을 남긴다."""

    attempt = retry_state.attempt_number
    outcome = retry_state.outcome
    exc = outcome.exception() if outcome is not None else None
    next_sleep = getattr(retry_state.next_action, "sleep", None)
    logger.warning(
        "HTTP 재시도 예정 (attempt={attempt}, next_sleep={sleep}s, error={err})",
        attempt=attempt,
        sleep=next_sleep,
        err=repr(exc),
    )


def http_retry(func: Callable[..., Any]) -> Callable[..., Any]:
    """논문 검색 HTTP 호출에 공통 적용하는 재시도 데코레이터.

    - 최대 4회 시도
    - 지수 백오프 (1s → 2s → 4s, 상한 8s)
    - 타임아웃 또는 429/5xx 일 때만 재시도 (그 외 4xx 는 즉시 전파)
    """

    decorated = retry(
        retry=retry_if_exception(_should_retry_http_error),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(4),
        before_sleep=_log_before_sleep,
        reraise=True,
    )(func)
    return decorated


__all__ = ["get_http_client", "http_retry"]
