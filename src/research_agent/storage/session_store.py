"""세션 JSON 영속화.

`data/sessions/<YYYY-MM-DD>_<session_id>.json` 형태로 저장한다. 파일명에 날짜를 붙여
사용자가 디렉토리만 봐도 최신 세션을 찾기 쉽게 한다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from research_agent.config import get_settings
from research_agent.logger import logger
from research_agent.models.session import Session


def _resolve_dir(sessions_dir: Path | None) -> Path:
    """인자로 받은 경로가 있으면 그대로, 없으면 설정 기본값 사용."""

    return sessions_dir if sessions_dir is not None else get_settings().sessions_dir


def _make_filename(session: Session) -> str:
    """`<YYYY-MM-DD>_<session_id>.json` 파일명을 생성한다."""

    date_prefix = session.updated_at.astimezone(timezone.utc).strftime("%Y-%m-%d")
    return f"{date_prefix}_{session.session_id}.json"


def save_session(session: Session, *, sessions_dir: Path | None = None) -> Path:
    """세션을 JSON 으로 직렬화해 디스크에 저장하고 그 경로를 돌려준다.

    저장 직전 `updated_at` 을 현재 UTC 시각으로 덮어쓴다 — 파일명 날짜와 내용이 일관된다.
    """

    target_dir = _resolve_dir(sessions_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    session.updated_at = datetime.now(timezone.utc)
    path = target_dir / _make_filename(session)

    payload = session.model_dump_json(indent=2)
    path.write_text(payload, encoding="utf-8")

    logger.info("세션 저장 완료: {path}", path=str(path))
    return path


def load_session(
    session_id: UUID | str, *, sessions_dir: Path | None = None
) -> Session:
    """세션 ID 로 JSON 파일을 찾아 `Session` 으로 복원한다.

    파일명 패턴이 `*_<session_id>.json` 이므로 glob 로 탐색한다. 못 찾으면 `FileNotFoundError`.
    임의 문자열이 glob 패턴으로 섞여 디렉토리 순회 공격이 되지 않도록 진입 시 UUID 형식
    검증을 수행한다.
    """

    target_dir = _resolve_dir(sessions_dir)
    key = str(session_id)

    # UUID 문자열이 아니면 예외 — glob 인젝션/경로 탐색 방지.
    try:
        UUID(key)
    except ValueError as exc:
        raise ValueError(f"session_id 는 UUID 형식이어야 합니다: {session_id}") from exc

    matches = sorted(target_dir.glob(f"*_{key}.json"))
    if not matches:
        raise FileNotFoundError(f"session_id={key} 파일을 {target_dir} 에서 찾지 못함")

    path = matches[-1]
    payload = path.read_text(encoding="utf-8")
    logger.info("세션 로드: {path}", path=str(path))
    return Session.model_validate_json(payload)


def list_sessions(sessions_dir: Path | None = None) -> list[Path]:
    """디렉토리의 세션 JSON 파일 경로를 정렬된 리스트로 반환한다.

    디렉토리가 없으면 빈 리스트를 돌려준다 (예외 발생 X).
    """

    target_dir = _resolve_dir(sessions_dir)
    if not target_dir.exists():
        return []
    return sorted(target_dir.glob("*.json"))


__all__ = ["list_sessions", "load_session", "save_session"]
