"""LangGraph StateGraph 빌더.

`build_graph(...)` 는 외부 의존을 주입받아 컴파일된 그래프를 반환한다. MemorySaver
체크포인터를 기본 장착해 동일 thread_id 로 재호출하면 상태가 복원된다. 단, MVP 는
프로세스 종료 시 체크포인트가 사라지므로 영속 저장은 `storage/session_store.py` 담당.

엣지 구조:
    START
      └─ start_session
           └─ collect_topic
                └─ refine_topic
                     └─ confirm_topic
                          ├─(confirmed)─ build_queries
                          │                └─ search_surveys
                          │                     └─ summarize_candidates
                          │                          └─ present_and_review
                          │                               ├─(persist)─ persist_session ─ END
                          │                               └─(retry)─ build_queries
                          └─(not confirmed)─ refine_topic  (재진입)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from research_agent.agents.nodes import (
    CLIInterface,
    make_nodes,
    route_after_confirm,
    route_after_review,
)
from research_agent.agents.state import ResearchState
from research_agent.llm.client import LLMClient
from research_agent.models.session import Session
from research_agent.tools.base import PaperSearchClient


def build_graph(
    *,
    llm: LLMClient | None = None,
    search_clients: Sequence[PaperSearchClient] | None = None,
    cli: CLIInterface,
    storage_save: Callable[[Session], Path] | None = None,
    checkpointer: MemorySaver | None = None,
):
    """연구 에이전트 StateGraph 를 컴파일해 반환한다.

    Args:
        llm: LLM 래퍼. None 이면 `LLMClient()` 기본 생성.
        search_clients: 검색 클라이언트 시퀀스. None 이면 기본 3소스.
        cli: CLI 인터페이스 구현 (필수).
        storage_save: 세션 저장 함수. None 이면 `save_session`.
        checkpointer: LangGraph 체크포인터. None 이면 새 `MemorySaver`.

    Returns:
        `StateGraph.compile(...)` 결과 (`CompiledGraph`). `invoke(...)` 로 실행.
    """

    nodes = make_nodes(
        llm=llm,
        search_clients=search_clients,
        cli=cli,
        storage_save=storage_save,
    )

    graph = StateGraph(ResearchState)

    # 노드 등록
    graph.add_node("start_session", nodes["start_session"])
    graph.add_node("collect_topic", nodes["collect_topic"])
    graph.add_node("refine_topic", nodes["refine_topic"])
    graph.add_node("confirm_topic", nodes["confirm_topic"])
    graph.add_node("build_queries", nodes["build_queries"])
    graph.add_node("search_surveys", nodes["search_surveys"])
    graph.add_node("summarize_candidates", nodes["summarize_candidates"])
    graph.add_node("present_and_review", nodes["present_and_review"])
    graph.add_node("persist_session", nodes["persist_session"])

    # 엣지
    graph.add_edge(START, "start_session")
    graph.add_edge("start_session", "collect_topic")
    graph.add_edge("collect_topic", "refine_topic")
    graph.add_edge("refine_topic", "confirm_topic")

    # confirm_topic → 확정 여부에 따라 build_queries / refine_topic 분기
    graph.add_conditional_edges(
        "confirm_topic",
        route_after_confirm,
        {
            "build_queries": "build_queries",
            "refine_topic": "refine_topic",
        },
    )

    graph.add_edge("build_queries", "search_surveys")
    graph.add_edge("search_surveys", "summarize_candidates")
    graph.add_edge("summarize_candidates", "present_and_review")

    # present_and_review → persist 또는 retry (build_queries 재진입)
    graph.add_conditional_edges(
        "present_and_review",
        route_after_review,
        {
            "persist": "persist_session",
            "retry": "build_queries",
        },
    )

    graph.add_edge("persist_session", END)

    memory_saver = checkpointer if checkpointer is not None else MemorySaver()
    return graph.compile(checkpointer=memory_saver)


__all__ = ["build_graph"]
