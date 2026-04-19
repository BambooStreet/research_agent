"""애플리케이션 설정 로더.

환경 변수와 `.env` 파일을 읽어 `Settings` 객체로 노출한다.
`get_settings()` 는 `lru_cache` 기반 싱글톤이므로 여러 번 호출해도 동일 인스턴스를 돌려준다.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """런타임 설정.

    필드 이름은 snake_case 이며, 대응 환경 변수는 대문자(pydantic-settings 기본 규칙).
    예: `anthropic_api_key` <-> `ANTHROPIC_API_KEY`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: SecretStr
    anthropic_model: str = "claude-sonnet-4-6"

    semantic_scholar_api_key: SecretStr | None = None
    openalex_mailto: str | None = None

    sessions_dir: Path = Path("data/sessions")
    log_level: str = "INFO"

    request_timeout_seconds: float = 20.0
    max_retry_loops: int = 3


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """프로세스 전역 싱글톤 `Settings` 반환.

    `lru_cache` 가 없으면 여러 모듈에서 각자 `Settings()` 를 만들 때 `.env` 를 중복 파싱한다.
    """

    return Settings()
