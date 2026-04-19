"""arXiv 검색 클라이언트.

`arxiv` 라이브러리를 감싸 모듈 싱글톤 `arxiv.Client` 를 사용한다.
서베이 모드에서는 title/abstract 에 'survey' 또는 'review' 가 포함된 것만 필터링한다.
"""

from __future__ import annotations

import re

import arxiv

from research_agent.logger import logger
from research_agent.models.paper import Paper, PaperSource

SOURCE: PaperSource = PaperSource.ARXIV

# 모듈 싱글톤. `delay_seconds` 는 arxiv 가 공개한 polite crawl 지침에 맞춘 기본값.
_CLIENT = arxiv.Client(page_size=20, delay_seconds=3, num_retries=3)

# arXiv 쿼리 예약 문자: 괄호/콜론/큰따옴표 는 파서에 의미가 있어 LLM 이 만든 사용자
# 쿼리에 섞이면 문법 오류를 낸다. 안전하게 공백으로 치환한다.
_ARXIV_RESERVED_RE = re.compile(r'[():"]')


def _sanitize_user_query(query: str) -> str:
    """사용자(혹은 LLM) 쿼리에서 arXiv 예약 문자를 제거한다.

    괄호, 콜론, 큰따옴표를 공백으로 치환하고 연속 공백을 하나로 축약한다.
    `cat:cs.CL` 같은 필터는 `_merge_arxiv_categories` 가 별도로 합성하므로 여기서는
    순수 키워드 텍스트만 남기면 된다.
    """

    replaced = _ARXIV_RESERVED_RE.sub(" ", query)
    return re.sub(r"\s+", " ", replaced).strip()


def _build_query(query: str, *, is_survey: bool) -> str:
    """arXiv query syntax 문자열을 구성한다.

    서베이 모드는 제목/초록에 survey/review 키워드가 들어간 것만 매칭되도록 AND 결합한다.
    사용자 쿼리 부분만 이스케이프 처리하고, `cat:...` 필터는 그대로 둔다.
    """

    # 카테고리 필터가 포함된 쿼리(노드 단의 _merge_arxiv_categories 산출물) 구조는
    # `(keywords) AND cat:xx.YY` 형태. 이 경우 keyword 부분만 분리해 이스케이프한다.
    sanitized = _sanitize_composite_query(query)
    if is_survey:
        return f'(ti:"survey" OR ti:"review" OR abs:"survey") AND ({sanitized})'
    return sanitized


def _sanitize_composite_query(query: str) -> str:
    """키워드 + 카테고리 필터 조합 쿼리에서 키워드 부분만 예약 문자 제거.

    `(keywords) AND cat:cs.CL` 또는 `(keywords) AND (cat:a OR cat:b)` 형태를 보존한다.
    카테고리 토큰이 없으면 전체를 키워드로 간주해 `_sanitize_user_query` 를 적용한다.
    """

    match = re.match(r"^\((.+)\)\s+AND\s+(.+)$", query.strip(), flags=re.DOTALL)
    if match:
        keywords = _sanitize_user_query(match.group(1))
        filter_part = match.group(2).strip()
        return f"({keywords}) AND {filter_part}"
    return _sanitize_user_query(query)


def _to_paper(result: arxiv.Result) -> Paper:
    """`arxiv.Result` 를 공통 `Paper` 로 정규화한다.

    entry_id 는 `http://arxiv.org/abs/2301.00001v1` 형태라 '/abs/' 기준으로 뒤쪽을
    external_id 로 쓴다 (버전 접미사 포함).
    """

    external_id = result.entry_id.split("/abs/")[-1]
    return Paper(
        source=SOURCE,
        external_id=external_id,
        doi=result.doi,
        title=result.title.strip(),
        authors=[author.name for author in result.authors],
        year=result.published.year if result.published else None,
        venue=None,
        abstract=result.summary,
        url=result.entry_id,
    )


def search(query: str, *, is_survey: bool = True, limit: int = 10) -> list[Paper]:
    """arXiv 에서 논문을 검색해 `Paper` 리스트로 반환한다.

    실패 시 예외는 상위로 전파한다 (`nodes` 레벨에서 소스별 isolation 처리).
    """

    final_query = _build_query(query, is_survey=is_survey)
    logger.info("arxiv 검색 {limit}건 요청: {query}", limit=limit, query=final_query)

    search_request = arxiv.Search(
        query=final_query,
        max_results=limit,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    results = list(_CLIENT.results(search_request))
    papers = [_to_paper(r) for r in results]

    logger.info("arxiv 검색 결과 {count}건", count=len(papers))
    return papers


__all__ = ["SOURCE", "search"]
