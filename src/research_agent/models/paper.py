"""논문 도메인 모델.

세 개의 외부 소스(arXiv / Semantic Scholar / OpenAlex)에서 가져온 결과를
공통 `Paper` 로 정규화해 그래프 상태와 세션 JSON 에 동일한 모양으로 흐르게 한다.
"""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class PaperSource(StrEnum):
    """논문을 가져온 외부 소스."""

    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    OPENALEX = "openalex"


class PaperStatus(StrEnum):
    """HITL 리뷰 상태. 기본값은 `pending`."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEFERRED = "deferred"


class PaperRole(StrEnum):
    """파이프라인 단계 상의 역할.

    MVP 에서는 `survey` 만 사용하지만 후속 단계(선행연구/후속논문 분류)까지 대비해
    enum 을 미리 선언한다.
    """

    SURVEY = "survey"
    PROBLEM = "problem"
    METHOD = "method"
    RECENT = "recent"


class Paper(BaseModel):
    """세 소스를 정규화한 논문 레코드.

    `url` 은 `HttpUrl` 대신 `str | None` 을 사용한다. 외부 소스의 URL 포맷이
    제각각이라 pydantic 의 엄격한 URL 검증에서 탈락하는 경우가 많기 때문.
    """

    model_config = ConfigDict(use_enum_values=False)

    paper_id: UUID = Field(default_factory=uuid4)
    source: PaperSource
    external_id: str
    doi: str | None = None
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    abstract: str | None = None
    summary_ko: str | None = None
    url: str | None = None
    status: PaperStatus = PaperStatus.PENDING
    role: PaperRole = PaperRole.SURVEY
    notes: str | None = None
