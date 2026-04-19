"""LangGraph 에서 흐르는 상태 스키마.

LangGraph 는 각 노드 반환 dict 를 state 에 shallow merge 한다. 모든 필드에 기본값을
두지 않고 `total=False` 로 선언해 선택적 필드로 관리한다 — 노드가 자신이 쓰는 키만
반환해도 나머지 값이 유지된다.

주의: LangGraph state 는 체크포인트 직렬화 시 dict 로 변환되지만, 노드 실행 중에는
Python 객체가 그대로 보존된다. 따라서 `Paper` (pydantic BaseModel) 를 리스트에 담아도
노드 간 전달에 문제없다. 디스크 저장은 `storage/session_store.py` 가 별도로 담당한다.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from research_agent.models.paper import Paper

StageLiteral = Literal["topic", "refine", "search", "review", "done"]


class ResearchState(TypedDict, total=False):
    """연구 에이전트 그래프의 전체 상태.

    필드 구분:
    - 식별: `session_id`, `created_at`
    - 주제: `raw_topic`, `refined_topic`, `topic_options`, `topic_confirmed`
    - 검색: `search_queries`, `candidates`
    - 리뷰: `approved`, `rejected`, `deferred`, `pending_indices`
    - 제어: `retry_count`, `current_stage`, `error`
    """

    session_id: str
    created_at: str

    raw_topic: str
    refined_topic: str | None
    topic_options: list[str]
    topic_reason: str
    topic_confirmed: bool

    search_queries: list[str]

    candidates: list[Paper]
    approved: list[Paper]
    rejected: list[Paper]
    # 보류(deferred) 는 거부와 의미가 달라 별도 리스트로 관리한다.
    deferred: list[Paper]
    pending_indices: list[int]

    retry_count: int
    current_stage: StageLiteral
    error: str | None


__all__ = ["ResearchState", "StageLiteral"]
