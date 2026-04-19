"""`tools/openalex_client.py` 테스트."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from research_agent.models.paper import PaperSource
from research_agent.tools import openalex_client
from research_agent.tools.openalex_client import _reconstruct_abstract


@pytest.fixture
def fixture_payload(fixtures_dir: Path) -> dict:
    with (fixtures_dir / "openalex_works_response.json").open(encoding="utf-8") as f:
        return json.load(f)


def test_reconstruct_abstract_basic() -> None:
    inverted = {"Hello": [0, 3], "world": [1], "again": [4], "cruel": [2]}
    assert _reconstruct_abstract(inverted) == "Hello world cruel Hello again"


def test_reconstruct_abstract_none() -> None:
    assert _reconstruct_abstract(None) is None
    assert _reconstruct_abstract({}) is None


def test_reconstruct_abstract_tolerates_gaps() -> None:
    """중간에 빠진 position 이 있어도 예외 없이 등장 순서대로 연결된다."""

    # position 0, 2 만 존재 (1 번 빠짐) — 길이 기반 placeholder 없이 2-token 문자열.
    assert _reconstruct_abstract({"a": [0], "b": [2]}) == "a b"


def test_reconstruct_abstract_dedupes_same_position() -> None:
    """같은 position 이 중복 등록되면 첫 것만 유지된다."""

    # 같은 position=0 에 두 번 등장 → 한 번만 나와야 함.
    assert _reconstruct_abstract({"x": [0, 0]}) == "x"


def test_reconstruct_abstract_filters_negative_positions() -> None:
    """음수 위치는 버려지고 유효한 위치만 남는다."""

    assert _reconstruct_abstract({"y": [-1, 1]}) == "y"


def test_reconstruct_abstract_invalid_positions_type() -> None:
    """positions 가 list 가 아닌 경우 그 word 는 무시된다."""

    # "bad" 는 int 직접 값(비-list) → 무시. "ok" 만 반영.
    assert _reconstruct_abstract({"bad": 5, "ok": [0]}) == "ok"  # type: ignore[dict-item]


def test_reconstruct_abstract_all_invalid_returns_none() -> None:
    """유효 단어가 하나도 없으면 None 을 반환한다."""

    assert _reconstruct_abstract({"nope": [-1, -2]}) is None


def test_search_parses_fixture(fixture_payload: dict) -> None:
    with respx.mock(assert_all_called=True) as mock:
        mock.get("https://api.openalex.org/works").mock(
            return_value=httpx.Response(200, json=fixture_payload)
        )
        papers = openalex_client.search("hallucination", limit=5)

    assert len(papers) == 2

    first = papers[0]
    assert first.source == PaperSource.OPENALEX
    assert first.external_id == "W1234567890"
    assert first.doi == "https://doi.org/10.1000/oa.1"
    assert first.title == "A Comprehensive Survey of LLM Hallucination"
    assert first.authors == ["Carol Park", "Dan Kim"]
    assert first.year == 2024
    assert first.venue == "arXiv"  # primary_location 우선
    assert first.abstract == "We survey hallucination in LLMs"


def test_search_falls_back_to_host_venue_and_display_name(fixture_payload: dict) -> None:
    with respx.mock() as mock:
        mock.get("https://api.openalex.org/works").mock(
            return_value=httpx.Response(200, json=fixture_payload)
        )
        papers = openalex_client.search("x")

    second = papers[1]
    assert second.external_id == "W9999999999"
    assert second.title == "Minimal Work Record"  # display_name 으로 폴백
    assert second.venue == "Legacy Venue"  # host_venue 폴백
    assert second.abstract is None
    assert second.authors == []
    assert second.year is None
    assert second.doi is None
    # doi 가 없으면 url 은 OpenAlex id 로 폴백.
    assert second.url == "https://openalex.org/W9999999999"


def test_search_passes_mailto_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENALEX_MAILTO", "test@example.com")

    payload = {"results": []}
    with respx.mock() as mock:
        route = mock.get("https://api.openalex.org/works").mock(
            return_value=httpx.Response(200, json=payload)
        )
        openalex_client.search("topic")

    request_url = str(route.calls.last.request.url)
    assert "mailto=test%40example.com" in request_url or "mailto=test@example.com" in request_url


def test_search_sends_review_filter() -> None:
    payload = {"results": []}
    with respx.mock() as mock:
        route = mock.get("https://api.openalex.org/works").mock(
            return_value=httpx.Response(200, json=payload)
        )
        openalex_client.search("topic", is_survey=True, limit=7)

    request_url = str(route.calls.last.request.url)
    assert "filter=type%3Areview" in request_url or "filter=type:review" in request_url
    assert "per-page=7" in request_url
