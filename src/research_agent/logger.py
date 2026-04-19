"""loguru 기반 공용 로거.

이 모듈을 import 하는 시점에 loguru 기본 핸들러를 제거하고, stderr sink 하나를
프로젝트 포맷으로 재등록한다. `Settings` 로부터 레벨을 읽지만, `.env` 가 아직 준비되지
않은 환경(테스트, import-time side effect 최소화)에서도 실패하지 않도록 기본 INFO 로
폴백한다.

사용법:
    from research_agent.logger import logger
    logger.info("메시지")

또는:
    from research_agent.logger import get_logger
    log = get_logger(__name__)
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger

_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level:<7}</level> | "
    "<cyan>{name}:{line}</cyan> | "
    "{message}"
)


def _resolve_log_level() -> str:
    """설정에서 로그 레벨을 읽는다. 실패 시 INFO.

    import 타이밍에 `.env` 가 없거나 `ANTHROPIC_API_KEY` 미설정이어도 로거 초기화가
    앱 전체를 죽이면 안 되므로 광범위한 예외 포획이 정당화된다.
    """

    try:
        from research_agent.config import get_settings

        return get_settings().log_level
    except Exception:  # noqa: BLE001
        return "INFO"


def _configure() -> None:
    """기본 핸들러 제거 + stderr sink 재등록."""

    logger.remove()
    logger.add(
        sys.stderr,
        level=_resolve_log_level(),
        format=_LOG_FORMAT,
        enqueue=False,
        backtrace=False,
        diagnose=False,
    )


def get_logger(name: str) -> "Logger":
    """이름 바인딩된 로거를 반환한다.

    loguru 는 전역이지만 `bind(name=...)` 로 컨텍스트를 실어주면 포맷의 `{name}` 을 통해
    모듈 이름이 기록된다.
    """

    return logger.bind(name=name)


_configure()

__all__ = ["get_logger", "logger"]
