"""click 기반 CLI 엔트리포인트.

`pyproject.toml` 의 `research-agent = "research_agent.cli.app:main"` 와 연결된다.
서브커맨드:
    - start: 새 대화형 세션을 시작
    - list : 저장된 세션 목록 출력
"""

from __future__ import annotations

import click

from research_agent.cli.repl import run_interactive_session
from research_agent.storage.session_store import list_sessions


@click.group()
def main() -> None:
    """논문 서베이 검색 연구 에이전트."""


@main.command()
def start() -> None:
    """새 서베이 검색 세션을 시작한다."""

    run_interactive_session()


@main.command(name="list")
def list_cmd() -> None:
    """저장된 세션 목록을 출력한다."""

    sessions = list_sessions()
    if not sessions:
        click.echo("저장된 세션이 없습니다.")
        return
    for p in sessions:
        click.echo(str(p))


if __name__ == "__main__":
    main()
