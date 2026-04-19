"""`tools/semantic_scholar_client.py` 테스트.

`respx` 로 httpx 요청을 가로채 fixture JSON 으로 응답한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from research_agent.models.paper import PaperSource
from research_agent.tools import semantic_scholar_client


@pytest.fixture
def fixture_payload(fixtures_dir: Path) -> dict:
    with (fixtures_dir / "semantic_scholar_search_response.json").open(encoding="utf-8") as f:
        return json.load(f)


def test_search_parses_fixture(fixture_payload: dict) -> None:
    with respx.mock(assert_all_called=True) as mock:
        mock.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=httpx.Response(200, json=fixture_payload)
        )

        papers = semantic_scholar_client.search("hallucination", limit=10)

    assert len(papers) == 2

    first = papers[0]
    assert first.source == PaperSource.SEMANTIC_SCHOLAR
    assert first.external_id == "ss-paper-1"
    assert first.doi == "10.1000/ss.1"
    assert first.title == "A Survey on Hallucination in Large Language Models"
    assert first.authors == ["Alice Chen", "Bob Lee"]
    assert first.year == 2023
    assert first.venue == "ACL"
    assert first.abstract and "hallucination" in first.abstract
    assert first.url == "https://www.semanticscholar.org/paper/ss-paper-1"


def test_search_handles_missing_optional_fields(fixture_payload: dict) -> None:
    with respx.mock() as mock:
        mock.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=httpx.Response(200, json=fixture_payload)
        )
        papers = semantic_scholar_client.search("x")

    second = papers[1]
    assert second.external_id == "ss-paper-2"
    assert second.authors == []
    assert second.abstract is None
    assert second.venue is None  # 빈 문자열은 None 으로 정규화
    assert second.year is None
    assert second.url is None
    assert second.doi is None


def test_search_sends_review_filter_when_is_survey() -> None:
    payload = {"data": []}

    with respx.mock() as mock:
        route = mock.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=httpx.Response(200, json=payload)
        )
        semantic_scholar_client.search("topic", is_survey=True, limit=3)

    assert route.called
    request = route.calls.last.request
    assert "publicationTypes=Review" in str(request.url)
    assert "limit=3" in str(request.url)


def test_search_skips_review_filter_when_not_survey() -> None:
    payload = {"data": []}

    with respx.mock() as mock:
        route = mock.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=httpx.Response(200, json=payload)
        )
        semantic_scholar_client.search("topic", is_survey=False)

    request = route.calls.last.request
    assert "publicationTypes" not in str(request.url)
