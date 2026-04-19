"""`storage/session_store.py` 에 대한 단위 테스트.

`tmp_path` 를 `sessions_dir` 인자로 주입해 파일시스템을 격리한다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from research_agent.models.paper import Paper, PaperSource, PaperStatus
from research_agent.models.session import Session, SessionStage
from research_agent.storage.session_store import (
    list_sessions,
    load_session,
    save_session,
)


def _make_session(raw_topic: str = "LLM hallucination") -> Session:
    paper = Paper(
        source=PaperSource.ARXIV,
        external_id="2301.00001",
        title="A Survey on X",
        status=PaperStatus.APPROVED,
    )
    return Session(
        raw_topic=raw_topic,
        refined_topic="LLM 환각",
        candidates=[paper],
        approved=[paper],
        current_stage=SessionStage.REVIEW,
    )


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    session = _make_session()

    path = save_session(session, sessions_dir=tmp_path)

    assert path.exists()
    assert path.suffix == ".json"
    assert str(session.session_id) in path.name

    loaded = load_session(session.session_id, sessions_dir=tmp_path)
    assert loaded.session_id == session.session_id
    assert loaded.raw_topic == session.raw_topic
    assert loaded.current_stage == SessionStage.REVIEW
    assert len(loaded.candidates) == 1
    assert loaded.candidates[0].title == "A Survey on X"


def test_save_updates_updated_at(tmp_path: Path) -> None:
    session = _make_session()
    before = session.updated_at

    save_session(session, sessions_dir=tmp_path)

    # `save_session` 내부에서 갱신되었으므로 이후 값이 이전보다 크거나 같아야 한다.
    assert session.updated_at >= before


def test_load_session_with_string_id(tmp_path: Path) -> None:
    session = _make_session()
    save_session(session, sessions_dir=tmp_path)

    loaded = load_session(str(session.session_id), sessions_dir=tmp_path)
    assert loaded.session_id == session.session_id


def test_load_session_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_session("00000000-0000-0000-0000-000000000000", sessions_dir=tmp_path)


def test_load_session_rejects_non_uuid_string(tmp_path: Path) -> None:
    """비-UUID 문자열은 glob 인젝션 위험이 있으니 진입 시 거부된다."""

    with pytest.raises(ValueError, match="UUID"):
        load_session("../../etc/passwd", sessions_dir=tmp_path)


def test_list_sessions_sorted(tmp_path: Path) -> None:
    sessions = [_make_session(f"topic-{i}") for i in range(3)]
    for s in sessions:
        save_session(s, sessions_dir=tmp_path)

    listed = list_sessions(tmp_path)
    assert len(listed) == 3
    # 정렬된 리스트여야 한다.
    assert listed == sorted(listed)


def test_list_sessions_missing_dir_returns_empty(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    assert list_sessions(missing) == []


def test_save_creates_parent_directory(tmp_path: Path) -> None:
    nested = tmp_path / "nested" / "sessions"
    session = _make_session()

    path = save_session(session, sessions_dir=nested)

    assert path.exists()
    assert path.parent == nested
