"""OpenAlex Works API 클라이언트.

`mailto` 파라미터를 설정이 있으면 붙여 polite pool 에 진입한다.
`abstract_inverted_index` 필드를 평문 초록으로 복원하는 로직이 핵심.
"""

from __future__ import annotations

from typing import Any

from research_agent.config import get_settings
from research_agent.logger import logger
from research_agent.models.paper import Paper, PaperSource
from research_agent.tools.http import get_http_client, http_retry

SOURCE: PaperSource = PaperSource.OPENALEX

_BASE = "https://api.openalex.org"


def _reconstruct_abstract(inverted: dict[str, list[int]] | None) -> str | None:
    """OpenAlex 의 `abstract_inverted_index` 를 원문 초록 문자열로 복원한다.

    구조: `{"word": [pos1, pos2, ...]}`. 모든 위치를 한 번에 펼쳐 인덱스로 정렬한 뒤
    토큰들을 공백으로 연결한다.

    방어 로직:
    - None / 빈 dict → None
    - positions 가 list 가 아닌 경우 해당 word 무시
    - 음수 또는 비-int 위치 무시
    - 동일 position 중복 시 첫 등장만 채택 (결정적 순서 보장)
    - 유효 단어가 하나도 없으면 None
    """

    if not inverted:
        return None

    pairs: list[tuple[int, str]] = []
    for word, positions in inverted.items():
        if not isinstance(positions, list):
            continue
        for p in positions:
            if isinstance(p, int) and p >= 0:
                pairs.append((p, word))

    if not pairs:
        return None

    # 같은 position 이 중복되면 (sorted 순서에서) 첫 번째 것만 채택해 비결정성 방지.
    seen: set[int] = set()
    deduped: list[str] = []
    for pos, word in sorted(pairs, key=lambda pair: pair[0]):
        if pos in seen:
            continue
        seen.add(pos)
        deduped.append(word)

    return " ".join(deduped) if deduped else None


def _extract_venue(work: dict[str, Any]) -> str | None:
    """OpenAlex 응답의 venue 를 추출한다.

    최근 스키마는 `primary_location.source.display_name` 로 이동했지만, 구 버전 응답에는
    `host_venue.display_name` 이 남아있을 수 있어 둘 다 시도한다.
    """

    primary = work.get("primary_location")
    if isinstance(primary, dict):
        source = primary.get("source")
        if isinstance(source, dict):
            display_name = source.get("display_name")
            if display_name:
                return display_name

    host_venue = work.get("host_venue")
    if isinstance(host_venue, dict):
        display_name = host_venue.get("display_name")
        if display_name:
            return display_name

    return None


def _to_paper(work: dict[str, Any]) -> Paper:
    """OpenAlex Work 한 건을 `Paper` 로 정규화."""

    # id 는 `https://openalex.org/W1234567890` 형태. 뒤의 ID 부분만 external_id 로 사용.
    raw_id: str = work.get("id") or ""
    external_id = raw_id.rsplit("/", 1)[-1] if raw_id else ""

    authorships = work.get("authorships") or []
    authors: list[str] = []
    for authorship in authorships:
        author_obj = authorship.get("author") or {}
        name = author_obj.get("display_name")
        if name:
            authors.append(name)

    title = work.get("title") or work.get("display_name") or ""
    abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))

    # landing url 로는 DOI 가 있으면 그것을, 없으면 OpenAlex id 자체를 사용.
    url = work.get("doi") or raw_id or None

    return Paper(
        source=SOURCE,
        external_id=external_id,
        doi=work.get("doi"),
        title=title,
        authors=authors,
        year=work.get("publication_year"),
        venue=_extract_venue(work),
        abstract=abstract,
        url=url,
    )


@http_retry
def _fetch(query: str, *, is_survey: bool, limit: int) -> dict[str, Any]:
    """OpenAlex `/works` 엔드포인트를 호출한다."""

    settings = get_settings()
    params: dict[str, Any] = {
        "search": query,
        "per-page": limit,
    }
    if is_survey:
        params["filter"] = "type:review"
    if settings.openalex_mailto:
        params["mailto"] = settings.openalex_mailto

    with get_http_client() as client:
        response = client.get(f"{_BASE}/works", params=params)
        response.raise_for_status()
        return response.json()


def search(query: str, *, is_survey: bool = True, limit: int = 10) -> list[Paper]:
    """OpenAlex 에서 논문을 검색해 `Paper` 리스트로 반환."""

    logger.info("openalex 검색 {limit}건 요청: {query}", limit=limit, query=query)

    data = _fetch(query, is_survey=is_survey, limit=limit)
    results: list[dict[str, Any]] = data.get("results") or []
    papers = [_to_paper(work) for work in results if work.get("id")]

    logger.info("openalex 검색 결과 {count}건", count=len(papers))
    return papers


__all__ = ["SOURCE", "search"]
