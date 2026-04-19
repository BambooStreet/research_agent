"""`tools/dedup.py` 에 대한 단위 테스트.

외부 의존성이 없으므로 순수 로직만 검증한다.
"""

from __future__ import annotations

from typing import Any

from research_agent.models.paper import Paper, PaperSource
from research_agent.tools.dedup import dedupe_papers, normalize_doi, normalize_title


def _make_paper(**overrides: Any) -> Paper:
    """테스트용 Paper 팩토리. 필수 필드만 채우고 나머지는 기본값."""

    defaults: dict[str, Any] = {
        "source": PaperSource.ARXIV,
        "external_id": "2301.00001",
        "title": "A Survey on Something",
    }
    defaults.update(overrides)
    return Paper(**defaults)


# ------------- normalize_doi -------------


def test_normalize_doi_none() -> None:
    assert normalize_doi(None) is None


def test_normalize_doi_empty_and_whitespace() -> None:
    assert normalize_doi("") is None
    assert normalize_doi("   ") is None


def test_normalize_doi_strips_https_prefix() -> None:
    assert normalize_doi("https://doi.org/10.1000/XYZ") == "10.1000/xyz"


def test_normalize_doi_strips_http_prefix() -> None:
    assert normalize_doi("http://doi.org/10.1000/xyz") == "10.1000/xyz"


def test_normalize_doi_strips_doi_colon_prefix() -> None:
    assert normalize_doi("doi:10.1000/XyZ") == "10.1000/xyz"


def test_normalize_doi_plain_doi_lowercased() -> None:
    assert normalize_doi("10.1000/ABC") == "10.1000/abc"


# ------------- normalize_title -------------


def test_normalize_title_basic() -> None:
    assert normalize_title("Hello World") == "hello world"


def test_normalize_title_strips_punctuation() -> None:
    assert normalize_title("Hello, World! (Survey)") == "hello world survey"


def test_normalize_title_collapses_whitespace() -> None:
    assert normalize_title("A   Survey\ton\n\nLLMs") == "a survey on llms"


def test_normalize_title_case_insensitive() -> None:
    assert normalize_title("A SURVEY on LLMs") == normalize_title("a survey on llms")


# ------------- dedupe_papers -------------


def test_dedupe_same_doi_different_sources_prefers_semantic_scholar() -> None:
    doi = "10.1000/shared"
    arxiv_paper = _make_paper(
        source=PaperSource.ARXIV, external_id="arx-1", title="T", doi=doi, abstract="a"
    )
    openalex_paper = _make_paper(
        source=PaperSource.OPENALEX, external_id="oa-1", title="T", doi=doi, abstract="a"
    )
    ss_paper = _make_paper(
        source=PaperSource.SEMANTIC_SCHOLAR,
        external_id="ss-1",
        title="T",
        doi=doi,
        abstract="a",
    )

    result = dedupe_papers([arxiv_paper, openalex_paper, ss_paper])

    assert len(result) == 1
    assert result[0].source == PaperSource.SEMANTIC_SCHOLAR


def test_dedupe_same_title_no_doi_prefers_abstract_present() -> None:
    no_abs = _make_paper(
        source=PaperSource.SEMANTIC_SCHOLAR,
        external_id="ss-2",
        title="Great Survey on LLMs",
        abstract=None,
    )
    with_abs = _make_paper(
        source=PaperSource.ARXIV,
        external_id="arx-2",
        title="Great Survey on LLMs!",  # 특수문자 달라도 정규화 후 동일
        abstract="abstract body",
    )

    result = dedupe_papers([no_abs, with_abs])

    assert len(result) == 1
    assert result[0].abstract == "abstract body"
    assert result[0].source == PaperSource.ARXIV


def test_dedupe_keeps_distinct_papers() -> None:
    a = _make_paper(title="Paper A", external_id="1", doi="10.1/a")
    b = _make_paper(title="Paper B", external_id="2", doi="10.1/b")
    c = _make_paper(title="Paper C", external_id="3")

    result = dedupe_papers([a, b, c])

    assert [p.title for p in result] == ["Paper A", "Paper B", "Paper C"]


def test_dedupe_preserves_first_occurrence_order() -> None:
    # arxiv 가 먼저, ss 가 나중에 들어오지만 결과는 최초 등장 위치를 유지해야 한다.
    shared_doi = "10.5/shared"
    first = _make_paper(source=PaperSource.ARXIV, external_id="arx", title="A", doi=shared_doi)
    middle = _make_paper(
        source=PaperSource.OPENALEX, external_id="oa-m", title="Middle", doi="10.5/other"
    )
    last_ss = _make_paper(
        source=PaperSource.SEMANTIC_SCHOLAR, external_id="ss", title="A", doi=shared_doi
    )

    result = dedupe_papers([first, middle, last_ss])

    # 그룹 대표는 SS 지만 위치는 첫 번째(인덱스 0) 자리.
    assert len(result) == 2
    assert result[0].source == PaperSource.SEMANTIC_SCHOLAR
    assert result[1].title == "Middle"


def test_dedupe_doi_prefix_variants_are_merged() -> None:
    a = _make_paper(
        source=PaperSource.ARXIV,
        external_id="arx",
        title="X",
        doi="https://doi.org/10.1/abc",
    )
    b = _make_paper(
        source=PaperSource.SEMANTIC_SCHOLAR,
        external_id="ss",
        title="X",
        doi="10.1/ABC",
    )

    result = dedupe_papers([a, b])

    assert len(result) == 1
    assert result[0].source == PaperSource.SEMANTIC_SCHOLAR
