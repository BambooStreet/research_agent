"""대화형 REPL 진입점.

click 엔트리(`cli/app.py`) 에서 호출하며, 내부에서 의존(LLM, 검색 클라이언트, CLI)
을 조립해 LangGraph 를 실행한다. 사용자 친화적 에러 메시지 처리는 여기서 담당.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast
from uuid import uuid4

import anthropic
import httpx
from pydantic import ValidationError
from rich.console import Console

from research_agent.agents.graph import build_graph
from research_agent.cli.views import RichCLI
from research_agent.llm.client import LLMClient
from research_agent.logger import logger
from research_agent.tools import arxiv_client, openalex_client, semantic_scholar_client
from research_agent.tools.base import PaperSearchClient


def _default_clients() -> Sequence[PaperSearchClient]:
    """운영 기본 검색 클라이언트. 모듈을 Protocol 로 cast."""

    return (
        cast(PaperSearchClient, semantic_scholar_client),
        cast(PaperSearchClient, openalex_client),
        cast(PaperSearchClient, arxiv_client),
    )


def run_interactive_session() -> Path | None:
    """새 세션을 실행하고 저장 경로(또는 None) 를 반환한다.

    실패 경로(ValidationError, KeyboardInterrupt, API 에러) 는 모두 친화적 메시지로
    안내하고 `None` 을 돌려준다. 정상 종료 시에도 Path 반환 여부는 persist 노드가
    상태에 경로를 담지 않기 때문에 항상 None; 실제 저장 경로는 `cli.notify` 로
    이미 사용자에게 출력되었다.
    """

    cli = RichCLI()
    console: Console = cli.console

    # LLM 초기화 단계에서 API 키 누락 시 ValidationError 가 터진다.
    try:
        llm = LLMClient()
    except ValidationError as exc:
        console.print(
            "[bold red]환경 변수 설정 오류.[/bold red] "
            "`.env` 파일에 ANTHROPIC_API_KEY 가 설정돼 있는지 확인하세요."
        )
        logger.error("LLMClient 초기화 실패: {err}", err=str(exc))
        return None

    clients = _default_clients()
    graph = build_graph(llm=llm, search_clients=clients, cli=cli)
    thread_id = str(uuid4())

    try:
        graph.invoke({}, config={"configurable": {"thread_id": thread_id}})
    except KeyboardInterrupt:
        console.print("\n[yellow]사용자에 의해 중단되었습니다. (Ctrl+C)[/yellow]")
        logger.info("세션 중단 (KeyboardInterrupt): thread_id={tid}", tid=thread_id)
        return None
    except anthropic.APIError as exc:
        # 인증/레이트리밋/서버 에러 모두 여기로 모인다. 사용자에게는 요지만 노출.
        console.print(
            f"[bold red]Anthropic API 호출 실패:[/bold red] {exc.__class__.__name__} — {exc}"
        )
        logger.error("Anthropic API 오류: {err}", err=repr(exc))
        return None
    except httpx.HTTPError as exc:
        # 외부 검색 API 네트워크 오류 (Semantic Scholar / OpenAlex 등).
        console.print(f"[bold red]외부 검색 API 오류:[/bold red] {exc}")
        logger.error("httpx 오류: {err}", err=repr(exc))
        return None
    except Exception as exc:  # noqa: BLE001
        # 마지막 방어선. 스택트레이스는 로그로만 남기고, 화면에는 요지만.
        console.print(f"[bold red]예상치 못한 오류:[/bold red] {exc}")
        logger.exception("예상치 못한 오류")
        return None

    # 정상 종료 — 저장 경로는 persist_session 노드가 이미 `cli.notify` 로 출력함.
    cli.notify("\n[bold green][완료][/bold green] 세션이 종료되었습니다.")
    return None


__all__ = ["run_interactive_session"]
