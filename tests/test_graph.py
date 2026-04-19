"""그래프 end-to-end 테스트.

모든 외부 의존(LLM / 검색 클라이언트 / CLI / storage) 을 fake 로 교체하여 그래프를
한 번 돌리고, 최종 상태가 기대대로 approved/rejected 로 분류되는지 검증한다.

시나리오:
    1. 사용자가 "LLM hallucination" 입력
    2. refine 응답: too_broad, 옵션 3개 제시 → 사용자 "1" 선택
    3. query 응답: primary/alternative/categories 돌려줌
    4. 3개 클라이언트가 각각 Paper 2~3건씩 반환 (일부 중복 포함)
    5. 요약 LLM 호출은 각 Paper 마다 더미 한국어 문자열 반환
    6. review 단계에서 처음 2건 "y", 나머지 "n"
    7. 최종 상태: approved=2, rejected=나머지, current_stage="done"
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator
from uuid import uuid4

import pytest

from research_agent.agents.graph import build_graph
from research_agent.agents.nodes import PaperDecision
from research_agent.models.paper import Paper, PaperSource, PaperStatus
from research_agent.models.session import Session


class FakeLLM:
    """`LLMClient` 의 인터페이스 중 노드가 실제로 호출하는 두 메서드만 스텁.

    `complete_json` 은 호출 순서에 따라 미리 정해진 dict 를 돌려준다.
    `complete_text` 는 summarize 용이므로 항상 같은 한국어 더미 문자열을 반환.
    """

    def __init__(self, json_responses: list[dict]) -> None:
        self._json_responses: list[dict] = list(json_responses)
        self.text_calls: int = 0
        self.json_calls: int = 0

    def complete_json(
        self, *, system: str, user: str, schema_hint: str | None = None
    ) -> dict:
        self.json_calls += 1
        if not self._json_responses:
            raise AssertionError("FakeLLM: 추가 JSON 응답이 없음")
        return self._json_responses.pop(0)

    def complete_text(self, *, system: str, user: str) -> str:
        self.text_calls += 1
        return "테스트 요약: 이 논문은 환각 문제를 다룬다. 주요 기여는 평가 프레임워크 제시."


class FakeSearchClient:
    """모듈 단위 검색 클라이언트의 duck typing 대체물."""

    def __init__(self, source: PaperSource, papers: list[Paper]) -> None:
        self.SOURCE = source
        self._papers = papers
        self.calls: int = 0

    def search(
        self, query: str, *, is_survey: bool = True, limit: int = 10
    ) -> list[Paper]:
        self.calls += 1
        return list(self._papers[:limit])


class RoundFakeSearchClient:
    """round 마다 다른 결과를 반환하는 fake 클라이언트.

    retry 재시도 경로를 검증하려면 round 1 과 round 2 가 서로 다른 후보 집합을
    내놓아야 한다. `per_round` 리스트의 인덱스가 호출 순서에 대응된다.
    """

    def __init__(
        self, source: PaperSource, per_round: list[list[Paper]]
    ) -> None:
        self.SOURCE = source
        self._per_round: list[list[Paper]] = per_round
        self.calls: int = 0

    def search(
        self, query: str, *, is_survey: bool = True, limit: int = 10
    ) -> list[Paper]:
        idx = min(self.calls, len(self._per_round) - 1)
        self.calls += 1
        return list(self._per_round[idx][:limit])


class FakeCLI:
    """스크립트된 입력 시퀀스를 돌려주는 CLI 구현."""

    def __init__(
        self,
        *,
        topic: str,
        topic_choice: str,
        topic_direct: str = "",
        paper_decisions: list[PaperDecision],
    ) -> None:
        self._topic = topic
        self._topic_choice = topic_choice
        self._topic_direct = topic_direct
        self._paper_decisions: list[PaperDecision] = list(paper_decisions)

        self.notifications: list[str] = []
        self.headers: list[tuple[int, dict[str, int]]] = []
        self.decision_calls: int = 0

    def ask_topic(self) -> str:
        return self._topic

    def ask_topic_choice(self, options: list[str], reason: str) -> str:
        return self._topic_choice

    def ask_topic_direct(self) -> str:
        return self._topic_direct

    def show_candidates_header(self, total: int, per_source: dict[str, int]) -> None:
        self.headers.append((total, dict(per_source)))

    def ask_paper_decision(
        self, paper: Paper, index: int, total: int
    ) -> PaperDecision:
        self.decision_calls += 1
        if self._paper_decisions:
            return self._paper_decisions.pop(0)
        # 기본값: 나머지는 전부 거부
        return "n"

    def notify(self, message: str) -> None:
        self.notifications.append(message)


@pytest.fixture
def saved_sessions() -> list[Session]:
    """테스트에서 저장된 세션을 캡처하는 컨테이너."""

    return []


@pytest.fixture
def fake_storage_save(
    saved_sessions: list[Session], tmp_path: Path
) -> Iterator:
    """실제 파일 I/O 없이 메모리 리스트에 세션을 담는 save 함수."""

    def _save(session: Session) -> Path:
        saved_sessions.append(session)
        return tmp_path / f"{session.session_id}.json"

    yield _save


def _make_paper(
    source: PaperSource, external_id: str, title: str, doi: str | None = None
) -> Paper:
    """테스트용 Paper 팩토리."""

    return Paper(
        source=source,
        external_id=external_id,
        doi=doi,
        title=title,
        authors=["Test Author"],
        year=2024,
        abstract="This survey covers hallucination in LLMs.",
        url=f"https://example.org/{external_id}",
    )


def test_graph_end_to_end_happy_path(
    saved_sessions: list[Session], fake_storage_save
) -> None:
    """정상 경로 1회 실행: 주제 → 옵션 선택 → 검색 → 요약 → 2건 승인 → 저장."""

    # --- 준비: fake LLM 응답 시퀀스 ---------------------------------------
    json_responses = [
        # 1) refine_topic
        {
            "scope": "too_broad",
            "reason": "LLM hallucination 은 분야 전체를 포괄함",
            "options": [
                "LLM 환각 탐지 방법론",
                "LLM 환각 완화 기법",
                "LLM 환각 평가 벤치마크",
            ],
        },
        # 2) build_queries
        {
            "primary_query": "LLM hallucination detection",
            "alternative_queries": ["large language model factuality"],
            "arxiv_categories": ["cs.CL"],
        },
    ]
    llm = FakeLLM(json_responses=json_responses)

    # --- 준비: fake 검색 클라이언트 --------------------------------------
    # 일부 중복(같은 DOI) 을 포함시켜 dedupe 경로도 태운다.
    arxiv_papers = [
        _make_paper(PaperSource.ARXIV, "2301.00001", "Survey on LLM Hallucination"),
        _make_paper(
            PaperSource.ARXIV,
            "2302.00002",
            "Detecting Factual Errors in LLMs",
            doi="10.1234/xyz",
        ),
    ]
    ss_papers = [
        _make_paper(
            PaperSource.SEMANTIC_SCHOLAR,
            "ss-1",
            "Detecting Factual Errors in LLMs",
            doi="10.1234/xyz",  # arxiv[1] 과 중복
        ),
        _make_paper(
            PaperSource.SEMANTIC_SCHOLAR, "ss-2", "Hallucination Benchmarks Overview"
        ),
    ]
    oa_papers = [
        _make_paper(
            PaperSource.OPENALEX, "oa-1", "A Review of Mitigation Strategies"
        ),
    ]
    clients = [
        FakeSearchClient(PaperSource.ARXIV, arxiv_papers),
        FakeSearchClient(PaperSource.SEMANTIC_SCHOLAR, ss_papers),
        FakeSearchClient(PaperSource.OPENALEX, oa_papers),
    ]

    # --- 준비: fake CLI ---------------------------------------------------
    # dedupe 후 4건 기대 (arxiv 2 + ss 2 + oa 1 - 중복 1 = 4).
    # 처음 2건 y, 나머지 n.
    cli = FakeCLI(
        topic="LLM hallucination",
        topic_choice="1",
        paper_decisions=["y", "y", "n", "n", "n", "n"],
    )

    # --- 실행 -------------------------------------------------------------
    graph = build_graph(
        llm=llm,  # type: ignore[arg-type]
        search_clients=clients,  # type: ignore[arg-type]
        cli=cli,
        storage_save=fake_storage_save,
    )

    thread_id = str(uuid4())
    final_state = graph.invoke(
        {}, config={"configurable": {"thread_id": thread_id}}
    )

    # --- 검증 -------------------------------------------------------------
    assert final_state["current_stage"] == "done"
    assert final_state["refined_topic"] == "LLM 환각 탐지 방법론"

    approved = final_state["approved"]
    rejected = final_state["rejected"]
    candidates = final_state["candidates"]

    assert len(approved) == 2, f"approved 2건 기대, 실제 {len(approved)}"
    # 승인되지 않은 나머지 후보는 모두 rejected
    assert len(approved) + len(rejected) == len(candidates)
    # 승인 상태가 정확히 APPROVED 로 기록되었는지
    for p in approved:
        assert p.status == PaperStatus.APPROVED
    # 저장 호출이 1회 이뤄졌는지
    assert len(saved_sessions) == 1
    saved = saved_sessions[0]
    assert saved.refined_topic == "LLM 환각 탐지 방법론"
    assert len(saved.approved) == 2
    # CLI 결정이 후보 수만큼 호출되었는지
    assert cli.decision_calls == len(candidates)
    # 헤더 출력 1회
    assert len(cli.headers) == 1


def test_graph_retry_round_does_not_accumulate_previous_rejections(
    saved_sessions: list[Session], fake_storage_save
) -> None:
    """retry 분기 시 이전 round 의 rejected 가 누적되지 않아야 한다.

    시나리오:
        - refine: scope=ok (선택지 없음)
        - build_queries: primary + alternative 1개 → search_queries 2개
        - round 1: 검색 결과 3건, 사용자가 전부 "n" → 승인 0, retry 증가 → build_queries 재진입
        - round 2: 검색 결과 2건, 사용자가 모두 "y" → persist
        - 기대 최종 상태: approved=2, rejected=0 (2 round 에는 거부가 없음)
          + 이전 round 의 3건 거부는 상태에 남아있지 않아야 한다.
    """

    # build_queries 가 retry 시에도 매 round 호출되는데, 실제 구현에서는 retry 분기로
    # 재진입할 때마다 LLM 쿼리 생성을 다시 호출한다. 따라서 json_responses 도 두 번 쓴다.
    query_payload: dict[str, object] = {
        "primary_query": "retry primary",
        "alternative_queries": ["retry alternative"],
        "arxiv_categories": [],
    }
    json_responses: list[dict] = [
        # refine_topic
        {"scope": "ok", "reason": "적절", "options": []},
        # build_queries (round 1)
        dict(query_payload),
        # build_queries (round 2, retry 로 재진입)
        dict(query_payload),
    ]
    llm = FakeLLM(json_responses=json_responses)

    # round 별 서로 다른 후보군.
    round1 = [
        _make_paper(PaperSource.ARXIV, "r1-a", "Round1 Paper A"),
        _make_paper(PaperSource.ARXIV, "r1-b", "Round1 Paper B"),
        _make_paper(PaperSource.ARXIV, "r1-c", "Round1 Paper C"),
    ]
    round2 = [
        _make_paper(PaperSource.ARXIV, "r2-a", "Round2 Paper A"),
        _make_paper(PaperSource.ARXIV, "r2-b", "Round2 Paper B"),
    ]
    clients = [
        RoundFakeSearchClient(PaperSource.ARXIV, [round1, round2]),
        # 다른 두 소스는 비어있음 (retry 트리거 관찰에만 집중).
        FakeSearchClient(PaperSource.SEMANTIC_SCHOLAR, []),
        FakeSearchClient(PaperSource.OPENALEX, []),
    ]

    cli = FakeCLI(
        topic="retry-topic",
        topic_choice="k",
        paper_decisions=["n", "n", "n", "y", "y"],
    )

    graph = build_graph(
        llm=llm,  # type: ignore[arg-type]
        search_clients=clients,  # type: ignore[arg-type]
        cli=cli,
        storage_save=fake_storage_save,
    )
    final_state = graph.invoke(
        {}, config={"configurable": {"thread_id": "t-retry"}}
    )

    # --- 검증 -------------------------------------------------------------
    assert final_state["current_stage"] == "done"
    approved = final_state["approved"]
    rejected = final_state["rejected"]
    deferred = final_state["deferred"]
    candidates = final_state["candidates"]

    # 2 round 결과만 남아야 한다.
    assert len(candidates) == 2, f"candidates 2건 기대, 실제 {len(candidates)}"
    assert len(approved) == 2, f"approved 2건 기대, 실제 {len(approved)}"
    # 누적 방지 핵심 검증: round 1 에서 거부된 3건이 남아있으면 안 된다.
    assert rejected == [], f"rejected 비어야 함, 실제 {rejected!r}"
    assert deferred == []

    # 저장된 세션의 rejected/deferred 도 비어야 한다.
    assert len(saved_sessions) == 1
    saved = saved_sessions[0]
    assert [p.title for p in saved.approved] == ["Round2 Paper A", "Round2 Paper B"]
    assert saved.rejected == []
    assert saved.deferred == []
    # retry_count 는 round 1 종료 시 한 번 증가했다.
    assert saved.retry_count == 1


def test_graph_scope_ok_skips_topic_choice(
    saved_sessions: list[Session], fake_storage_save
) -> None:
    """scope=ok 인 경우 사용자 선택 없이 raw_topic 이 그대로 refined_topic 이 된다."""

    json_responses = [
        # refine_topic → scope=ok, options 빈 배열
        {"scope": "ok", "reason": "적절한 범위", "options": []},
        # build_queries
        {
            "primary_query": "retrieval augmented generation survey",
            "alternative_queries": [],
            "arxiv_categories": [],
        },
    ]
    llm = FakeLLM(json_responses=json_responses)

    papers = [
        _make_paper(PaperSource.ARXIV, "2401.00001", "A Survey on RAG"),
    ]
    clients = [
        FakeSearchClient(PaperSource.ARXIV, papers),
        FakeSearchClient(PaperSource.SEMANTIC_SCHOLAR, []),
        FakeSearchClient(PaperSource.OPENALEX, []),
    ]

    # topic_choice 는 사용되지 않아야 한다 (confirm_topic 이 options 가 비어서 바로 확정).
    cli = FakeCLI(
        topic="retrieval augmented generation survey",
        topic_choice="SHOULD_NOT_BE_USED",
        paper_decisions=["y"],
    )

    graph = build_graph(
        llm=llm,  # type: ignore[arg-type]
        search_clients=clients,  # type: ignore[arg-type]
        cli=cli,
        storage_save=fake_storage_save,
    )
    final_state = graph.invoke(
        {}, config={"configurable": {"thread_id": "t-ok"}}
    )

    assert final_state["current_stage"] == "done"
    assert final_state["refined_topic"] == "retrieval augmented generation survey"
    assert final_state["topic_confirmed"] is True
    assert len(final_state["approved"]) == 1
