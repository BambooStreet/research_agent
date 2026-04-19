"""`models/paper.py` 와 `models/session.py` 에 대한 단위 테스트.

초점은 다음 네 가지:
- 기본값이 계획대로 세팅되는가
- JSON 라운드트립이 손실 없이 동작하는가
- StrEnum 이 직렬화 시 문자열(`"approved"`) 로 나가는가
- Session 이 중첩된 Paper 리스트까지 그대로 보존하는가
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from research_agent.models.paper import Paper, PaperRole, PaperSource, PaperStatus
from research_agent.models.session import Session, SessionStage


def _make_paper(**overrides: Any) -> Paper:
    """테스트용 Paper 팩토리. 필수 필드만 채우고 나머지는 기본값.

    pydantic `BaseModel` 은 `**kwargs` 형태로 인스턴스화되기 때문에 테스트 헬퍼에서
    `Any` 유니언이 가장 자연스럽다. 프로덕션 코드에서는 `Any` 를 쓰지 않는다.
    """

    defaults: dict[str, Any] = {
        "source": PaperSource.ARXIV,
        "external_id": "2301.00001",
        "title": "A Survey on X",
    }
    defaults.update(overrides)
    return Paper(**defaults)


def test_paper_defaults() -> None:
    paper = _make_paper()

    assert isinstance(paper.paper_id, UUID)
    assert paper.status == PaperStatus.PENDING
    assert paper.role == PaperRole.SURVEY
    assert paper.authors == []
    assert paper.doi is None
    assert paper.year is None


def test_paper_json_roundtrip() -> None:
    original = _make_paper(
        doi="10.1000/xyz",
        authors=["Alice", "Bob"],
        year=2024,
        venue="ACL",
        abstract="abstract body",
        summary_ko="한국어 요약",
        url="https://arxiv.org/abs/2301.00001",
        status=PaperStatus.APPROVED,
        role=PaperRole.SURVEY,
        notes="메모",
    )

    dumped = original.model_dump_json()
    restored = Paper.model_validate_json(dumped)

    assert restored == original


def test_paper_strenum_serializes_as_string() -> None:
    paper = _make_paper(status=PaperStatus.APPROVED, role=PaperRole.SURVEY)

    payload = json.loads(paper.model_dump_json())

    assert payload["status"] == "approved"
    assert payload["role"] == "survey"
    assert payload["source"] == "arxiv"


def test_session_defaults() -> None:
    session = Session(raw_topic="LLM hallucination")

    assert isinstance(session.session_id, UUID)
    assert session.schema_version == 1
    assert session.current_stage == SessionStage.TOPIC
    assert session.retry_count == 0
    assert session.candidates == []
    assert session.approved == []
    assert session.rejected == []
    assert session.refined_topic is None


def test_session_json_roundtrip_with_papers() -> None:
    paper_a = _make_paper(title="Survey A", status=PaperStatus.APPROVED)
    paper_b = _make_paper(
        source=PaperSource.SEMANTIC_SCHOLAR,
        external_id="ss-123",
        title="Survey B",
        status=PaperStatus.REJECTED,
    )

    original = Session(
        raw_topic="LLM hallucination",
        refined_topic="LLM 환각 탐지",
        candidates=[paper_a, paper_b],
        approved=[paper_a],
        rejected=[paper_b],
        current_stage=SessionStage.REVIEW,
        retry_count=1,
    )

    dumped = original.model_dump_json()
    restored = Session.model_validate_json(dumped)

    assert restored == original
    # 중첩 필드 직렬화 형태 확인
    payload = json.loads(dumped)
    assert payload["current_stage"] == "review"
    assert payload["candidates"][0]["source"] == "arxiv"
    assert payload["candidates"][0]["status"] == "approved"
