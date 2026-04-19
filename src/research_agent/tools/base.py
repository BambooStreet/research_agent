"""논문 검색 클라이언트가 따라야 하는 구조적 타입 정의.

구현체는 클래스가 아니라 모듈 단위로 `SOURCE` 상수와 `search()` 함수를 제공하고,
호출부는 `from research_agent.tools import arxiv_client` 처럼 모듈을 주입받아 쓴다.
Protocol 을 두는 이유는 mypy/pyright 로 시그니처 누락을 잡기 위함.
"""

from __future__ import annotations

from typing import Protocol

from research_agent.models.paper import Paper, PaperSource


class PaperSearchClient(Protocol):
    """논문 검색 구현체가 노출해야 하는 최소 인터페이스."""

    SOURCE: PaperSource

    def search(
        self,
        query: str,
        *,
        is_survey: bool = True,
        limit: int = 10,
    ) -> list[Paper]:
        """쿼리에 대응하는 논문 후보를 반환한다.

        실패 시 빈 리스트 대신 예외를 던져야 한다 (상위 노드가 소스별 실패를 로깅/스킵).
        """

        ...


__all__ = ["PaperSearchClient"]
