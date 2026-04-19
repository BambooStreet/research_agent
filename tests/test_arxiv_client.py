"""`tools/arxiv_client.py` 에 대한 단위 테스트.

arxiv 라이브러리의 네트워크 호출은 `_CLIENT.results` 를 monkeypatch 로 교체해 차단한다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from research_agent.models.paper import PaperSource
from research_agent.tools import arxiv_client


def _fake_result(
    *,
    entry_id: str,
    title: str,
    summary: str,
    published: datetime | None,
    authors: list[str],
    doi: str | None = None,
) -> Any:
    """`arxiv.Result` 최소 스펙을 흉내낸 DuckType 객체를 만든다.

    `arxiv.Result` 를 직접 인스턴스화하려면 수많은 필수 인자가 필요해 테스트가 취약해진다.
    대신 `_to_paper` 가 접근하는 속성만 가진 SimpleNamespace 를 쓴다.
    """

    return SimpleNamespace(
        entry_id=entry_id,
        title=title,
        summary=summary,
        published=published,
        authors=[SimpleNamespace(name=n) for n in authors],
        doi=doi,
    )


def test_search_returns_papers_with_expected_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_results = [
        _fake_result(
            entry_id="http://arxiv.org/abs/2309.00001v1",
            title="  A Survey on LLM Hallucination  ",
            summary="We survey hallucination in LLMs.",
            published=datetime(2023, 9, 1, tzinfo=timezone.utc),
            authors=["Alice", "Bob"],
            doi="10.1000/arxiv.1",
        ),
        _fake_result(
            entry_id="http://arxiv.org/abs/2401.00010v2",
            title="Review of RAG",
            summary="RAG review body.",
            published=datetime(2024, 1, 1, tzinfo=timezone.utc),
            authors=["Carol"],
            doi=None,
        ),
    ]

    def fake_results_iter(_search: Any) -> Any:
        return iter(fake_results)

    monkeypatch.setattr(arxiv_client._CLIENT, "results", fake_results_iter)

    papers = arxiv_client.search("LLM hallucination", limit=5)

    assert len(papers) == 2

    first = papers[0]
    assert first.source == PaperSource.ARXIV
    assert first.external_id == "2309.00001v1"
    assert first.title == "A Survey on LLM Hallucination"  # strip 확인
    assert first.authors == ["Alice", "Bob"]
    assert first.year == 2023
    assert first.abstract == "We survey hallucination in LLMs."
    assert first.doi == "10.1000/arxiv.1"
    assert first.url == "http://arxiv.org/abs/2309.00001v1"

    second = papers[1]
    assert second.doi is None
    assert second.year == 2024


def test_search_handles_missing_published(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_results = [
        _fake_result(
            entry_id="http://arxiv.org/abs/9999.00001",
            title="No Date Paper",
            summary="Body",
            published=None,
            authors=[],
        )
    ]
    monkeypatch.setattr(arxiv_client._CLIENT, "results", lambda _s: iter(fake_results))

    papers = arxiv_client.search("x", is_survey=False, limit=1)

    assert len(papers) == 1
    assert papers[0].year is None
    assert papers[0].authors == []


def test_build_query_survey_mode() -> None:
    built = arxiv_client._build_query("retrieval augmented generation", is_survey=True)
    assert 'ti:"survey"' in built
    assert "retrieval augmented generation" in built


def test_build_query_non_survey_mode() -> None:
    built = arxiv_client._build_query("topic", is_survey=False)
    assert built == "topic"


def test_build_query_sanitizes_reserved_chars() -> None:
    """괄호/콜론/큰따옴표 가 들어간 사용자 쿼리도 예외 없이 처리되어야 한다.

    arXiv 쿼리 파서가 해석해서 문법 오류를 내지 않도록 공백으로 치환된다.
    """

    built = arxiv_client._build_query(
        'LLM (hallucination): "survey"', is_survey=True
    )
    # 원본의 괄호/콜론/큰따옴표는 키워드 부분에서 제거되어야 한다.
    # 외곽의 `(...)` 구조는 템플릿이 추가한 것이지 원문 괄호가 아니다.
    assert 'LLM hallucination survey' in built
    # 템플릿 헤더는 유지 (survey / review 조건 AND).
    assert 'ti:"survey"' in built


def test_build_query_preserves_category_filter() -> None:
    """`(keywords) AND cat:cs.CL` 형태가 들어오면 카테고리 부분은 보존된다."""

    composed = "(LLM hallucination) AND cat:cs.CL"
    built = arxiv_client._build_query(composed, is_survey=False)
    # sanitize 후에도 카테고리 필터는 원형을 유지.
    assert "cat:cs.CL" in built
    assert "LLM hallucination" in built


def test_sanitize_user_query_removes_reserved() -> None:
    assert (
        arxiv_client._sanitize_user_query('foo (bar): "baz"')
        == "foo bar baz"
    )
