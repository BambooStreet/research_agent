"""pytest 공용 fixture.

테스트 환경에서는 `.env` 가 없어도 `Settings` 가 로드되도록 필수 환경변수를 주입한다.
또한 `get_settings()` 의 lru_cache 를 테스트 간에 초기화해 monkeypatch 된 env var 이
반영되도록 한다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pytest

# `get_settings` 가 import 되기 전 단계에서 환경변수를 세팅해 둔다.
os.environ.setdefault("OPENAI_API_KEY", "test-key")

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> Iterator[None]:
    """매 테스트마다 Settings 싱글톤을 초기화한다.

    `openalex_mailto` 같은 값을 monkeypatch 로 바꿀 때 캐시된 인스턴스가 남아있으면
    변경이 반영되지 않는다.
    """

    from research_agent.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def fixtures_dir() -> Path:
    """테스트 리소스 디렉토리 경로."""

    return FIXTURES_DIR
