"""Semantic Scholar Graph API 클라이언트.

API 키가 있으면 `x-api-key` 헤더를 실어 soft rate-limit 완화. 429 / 5xx 는 공용
`http_retry` 로 지수 백오프 재시도한다.
"""

from __future__ import annotations

from typing import Any

from research_agent.config import get_settings
from research_agent.logger import logger
from research_agent.models.paper import Paper, PaperSource
from research_agent.tools.http import get_http_client, http_retry

SOURCE: PaperSource = PaperSource.SEMANTIC_SCHOLAR

_BASE = "https://api.semanticscholar.org/graph/v1"
_FIELDS = "title,authors,year,abstract,externalIds,venue,url"


def _build_headers() -> dict[str, str]:
    """`x-api-key` 헤더를 설정이 있으면 추가한다.

    `SecretStr.get_secret_value()` 는 반드시 호출 직전에만 사용하고, 로그/예외 메시지로
    흘리지 않는다.
    """

    settings = get_settings()
    headers: dict[str, str] = {}
    if settings.semantic_scholar_api_key is not None:
        headers["x-api-key"] = settings.semantic_scholar_api_key.get_secret_value()
    return headers


def _to_paper(item: dict[str, Any]) -> Paper:
    """Semantic Scholar 응답 한 건을 `Paper` 로 변환.

    누락되기 쉬운 필드(`authors`, `externalIds`, `venue`)는 방어적으로 `.get()` 체인을
    사용한다.
    """

    external_ids = item.get("externalIds") or {}
    authors_raw = item.get("authors") or []
    authors: list[str] = [a.get("name", "") for a in authors_raw if a.get("name")]

    # SS 는 venue 가 빈 문자열이거나 타입이 다를 때(drop) 가 있어 항상 방어한다.
    raw_venue = item.get("venue")
    venue: str | None
    if isinstance(raw_venue, str) and raw_venue.strip():
        venue = raw_venue
    else:
        venue = None

    return Paper(
        source=SOURCE,
        external_id=item["paperId"],
        doi=external_ids.get("DOI"),
        title=item.get("title") or "",
        authors=authors,
        year=item.get("year"),
        venue=venue,
        abstract=item.get("abstract"),
        url=item.get("url"),
    )


@http_retry
def _fetch(query: str, *, is_survey: bool, limit: int) -> dict[str, Any]:
    """Semantic Scholar `paper/search` 엔드포인트를 호출한다.

    함수를 분리해 `@http_retry` 를 데코레이터로 붙인다. 응답 파싱은 상위에서.
    """

    params: dict[str, Any] = {
        "query": query,
        "limit": limit,
        "fields": _FIELDS,
    }
    if is_survey:
        params["publicationTypes"] = "Review"

    with get_http_client() as client:
        response = client.get(
            f"{_BASE}/paper/search",
            params=params,
            headers=_build_headers(),
        )
        response.raise_for_status()
        return response.json()


def search(query: str, *, is_survey: bool = True, limit: int = 10) -> list[Paper]:
    """Semantic Scholar 에서 논문을 검색해 `Paper` 리스트로 반환."""

    logger.info(
        "semantic_scholar 검색 {limit}건 요청: {query}", limit=limit, query=query
    )

    data = _fetch(query, is_survey=is_survey, limit=limit)
    items: list[dict[str, Any]] = data.get("data") or []
    papers = [_to_paper(item) for item in items if item.get("paperId")]

    logger.info("semantic_scholar 검색 결과 {count}건", count=len(papers))
    return papers


__all__ = ["SOURCE", "search"]
