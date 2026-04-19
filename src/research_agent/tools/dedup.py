"""세 소스 병합 결과에서 중복 논문을 제거한다.

1차 키는 DOI(정규화), DOI 가 없으면 title(정규화) 로 그룹핑한다.
동일 그룹에서는 abstract 유무 → 소스 우선순위 순으로 하나만 남긴다.
소스 우선순위: Semantic Scholar > OpenAlex > arXiv (SS 가 가장 메타데이터가 풍부).
"""

from __future__ import annotations

import re

from research_agent.models.paper import Paper, PaperSource

# DOI URL 프리픽스. 대소문자 무시하기 위해 소문자 비교 전에 입력도 소문자로 내린다.
_DOI_PREFIXES: tuple[str, ...] = (
    "https://doi.org/",
    "http://doi.org/",
    "doi:",
)

# 정렬/랭크에 쓰는 소스별 순위. 값이 작을수록 우선.
_SOURCE_RANK: dict[PaperSource, int] = {
    PaperSource.SEMANTIC_SCHOLAR: 0,
    PaperSource.OPENALEX: 1,
    PaperSource.ARXIV: 2,
}

# 알파벳/숫자/공백만 남기는 정규식. 유니코드 알파벳은 논문 타이틀에 드물다고 가정하고
# ASCII 에 한정해 단순화 (특수기호/구두점/하이픈 제거).
_TITLE_KEEP_RE = re.compile(r"[^a-z0-9\s]")
_MULTI_WS_RE = re.compile(r"\s+")


def normalize_doi(doi: str | None) -> str | None:
    """DOI 문자열을 비교 가능한 형태로 정규화한다.

    - None / 공백 문자열 → None
    - 소문자화 + 앞뒤 공백 제거
    - `https://doi.org/`, `http://doi.org/`, `doi:` 프리픽스 제거
    """

    if doi is None:
        return None
    cleaned = doi.strip().lower()
    if not cleaned:
        return None
    for prefix in _DOI_PREFIXES:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    return cleaned or None


def normalize_title(title: str) -> str:
    """타이틀을 비교 키로 쓸 정규화 형태로 변환한다.

    - 소문자화
    - 알파벳/숫자/공백을 제외한 모든 문자 제거
    - 연속 공백을 하나로 축소 + 양끝 공백 제거
    """

    lowered = title.lower()
    stripped = _TITLE_KEEP_RE.sub(" ", lowered)
    collapsed = _MULTI_WS_RE.sub(" ", stripped).strip()
    return collapsed


def _rank(paper: Paper) -> tuple[int, int]:
    """그룹 대표 선정용 정렬 키. 작을수록 우선 (abstract 있는 쪽 → 소스 우선순위 순)."""

    has_abstract = 0 if paper.abstract else 1
    source_rank = _SOURCE_RANK.get(paper.source, 99)
    return (has_abstract, source_rank)


def dedupe_papers(papers: list[Paper]) -> list[Paper]:
    """DOI / 정규화 타이틀 기준으로 중복 논문을 한 건으로 합친다.

    원래 입력 순서를 유지하기 위해, 각 그룹이 처음 등장한 인덱스를 기억해 그 순서로
    결과를 재배열한다.
    """

    # 그룹 키 → (대표 Paper, 최초 등장 인덱스)
    groups: dict[str, tuple[Paper, int]] = {}

    for idx, paper in enumerate(papers):
        doi_key = normalize_doi(paper.doi)
        if doi_key:
            key = f"doi:{doi_key}"
        else:
            key = f"title:{normalize_title(paper.title)}"

        existing = groups.get(key)
        if existing is None:
            groups[key] = (paper, idx)
            continue

        # 기존 대표와 비교해 더 나은 쪽을 선택하되 등장 인덱스는 최초 것을 유지한다.
        current_best, first_idx = existing
        if _rank(paper) < _rank(current_best):
            groups[key] = (paper, first_idx)

    ordered = sorted(groups.values(), key=lambda pair: pair[1])
    return [paper for paper, _ in ordered]


__all__ = ["dedupe_papers", "normalize_doi", "normalize_title"]
