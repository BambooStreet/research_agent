"""세션 도메인 모델.

한 번의 CLI 실행에 대응하는 작업 세션. 승인/거부된 논문들과 현재 단계를 담고,
`storage/session_store.py` 가 JSON 으로 직렬화해 디스크에 보관한다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from research_agent.models.paper import Paper


class SessionStage(StrEnum):
    """세션 진행 단계. 그래프의 `current_stage` 와 1:1 대응."""

    TOPIC = "topic"
    REFINE = "refine"
    SEARCH = "search"
    REVIEW = "review"
    DONE = "done"


def _utc_now() -> datetime:
    """기본 타임스탬프 생성기. UTC 고정으로 직렬화 일관성 확보."""

    return datetime.now(timezone.utc)


class Session(BaseModel):
    """디스크에 저장되는 세션 스냅샷.

    `schema_version` 을 1 부터 시작해 향후 모델 변경 시 로더에서 분기할 수 있게 한다.
    """

    model_config = ConfigDict(use_enum_values=False)

    session_id: UUID = Field(default_factory=uuid4)
    schema_version: int = 1
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    raw_topic: str
    refined_topic: str | None = None

    candidates: list[Paper] = Field(default_factory=list)
    approved: list[Paper] = Field(default_factory=list)
    rejected: list[Paper] = Field(default_factory=list)
    # 보류(deferred) 논문은 거부와 의미가 달라 별도 컬렉션으로 보관한다.
    # 향후 재검토/resume 단계에서 다시 꺼내 볼 수 있다.
    deferred: list[Paper] = Field(default_factory=list)

    current_stage: SessionStage = SessionStage.TOPIC
    retry_count: int = 0
