"""rich 기반 CLI 뷰.

`agents.nodes.CLIInterface` Protocol 을 만족하는 `RichCLI` 를 제공한다.
입력/출력 표현을 한 곳에 몰아 두어 노드 코드가 rich 의존을 직접 갖지 않게 한다.
"""

from __future__ import annotations

from typing import Literal

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from research_agent.models.paper import Paper, PaperSource

# 후보 카드 상단에 노출할 저자 최대 수. 초과분은 "외 N인" 으로 축약.
_MAX_AUTHORS_SHOWN: int = 3

PaperDecision = Literal["y", "n", "s", "q"]


def _format_authors(authors: list[str]) -> str:
    """저자 리스트를 카드 헤더용 문자열로 축약한다.

    없으면 "-" 반환. 3명 초과 시 앞 3명 + "외 N인" 으로 표시.
    """

    if not authors:
        return "-"
    if len(authors) <= _MAX_AUTHORS_SHOWN:
        return ", ".join(authors)
    shown = ", ".join(authors[:_MAX_AUTHORS_SHOWN])
    return f"{shown} 외 {len(authors) - _MAX_AUTHORS_SHOWN}인"


def _format_source(source: PaperSource | str) -> str:
    """Paper.source 를 안전하게 문자열로 변환 (enum/plain str 양쪽 대응)."""

    if isinstance(source, PaperSource):
        return source.value
    return str(source)


class RichCLI:
    """rich.Prompt + rich.Panel 로 구성된 CLIInterface 구현.

    `Console` 을 외부에서 주입할 수 있어 테스트에서 record=True 콘솔로 캡처 가능.
    """

    def __init__(self, console: Console | None = None) -> None:
        self.console: Console = console or Console()

    # ----- 주제 수집 ---------------------------------------------------------

    def ask_topic(self) -> str:
        """초기 연구 주제를 받는다. 빈 입력이면 재질문."""

        while True:
            topic = Prompt.ask("[bold cyan]> 연구 주제를 입력하세요[/bold cyan]")
            stripped = topic.strip()
            if stripped:
                return stripped
            self.console.print("[red]빈 입력입니다. 주제를 한 줄 이상 입력해주세요.[/red]")

    def ask_topic_choice(self, options: list[str], reason: str) -> str:
        """리파인 옵션 중 하나를 숫자/d/k 로 선택받는다."""

        self.console.print(f"\n[bold yellow][주제 좁히기][/bold yellow] {reason}")
        for i, opt in enumerate(options):
            self.console.print(f"  [bold]{i + 1}[/bold]) {opt}")
        self.console.print("  [dim]d: 직접 입력 / k: 원래 주제 그대로[/dim]")

        valid = [str(i + 1) for i in range(len(options))] + ["d", "k"]
        return Prompt.ask("선택", choices=valid)

    def ask_topic_direct(self) -> str:
        """사용자가 직접 입력한 주제를 받는다. 빈 입력 방지."""

        while True:
            topic = Prompt.ask("직접 입력할 주제")
            stripped = topic.strip()
            if stripped:
                return stripped
            self.console.print("[red]빈 입력입니다. 주제를 다시 입력해주세요.[/red]")

    # ----- 검색/리뷰 ---------------------------------------------------------

    def show_candidates_header(self, total: int, per_source: dict[str, int]) -> None:
        """검색 결과 요약 헤더."""

        if per_source:
            parts = " · ".join(f"{k}({v})" for k, v in per_source.items())
        else:
            parts = "(소스별 결과 없음)"
        self.console.print(
            f"\n[bold green][검색 완료][/bold green] {parts} → 중복 제거 후 {total}건\n"
        )

    def ask_paper_decision(self, paper: Paper, index: int, total: int) -> PaperDecision:
        """한 후보에 대한 y/n/s/q 결정을 받는다."""

        self.console.print(self._render_paper_panel(paper, index, total))
        choice = Prompt.ask(
            f"[{index + 1}/{total}] 승인/거부/보류",
            choices=["y", "n", "s", "q"],
            default="n",
        )
        # Prompt.ask 는 choices 제한 덕분에 여기서 이미 4종 중 하나. 타입만 좁혀 반환.
        return choice  # type: ignore[return-value]

    def notify(self, message: str) -> None:
        """일반 알림 메시지."""

        self.console.print(message)

    # ----- 렌더링 헬퍼 --------------------------------------------------------

    def _render_paper_panel(self, paper: Paper, index: int, total: int) -> Panel:
        """Paper 를 카드형 Panel 로 조립한다.

        구성 (위→아래):
            제목 (bold)
            저자 요약
            year · venue · source
            summary_ko (있으면)
            URL (dim)
        """

        body = Text()

        body.append(paper.title, style="bold")
        body.append("\n")

        authors = _format_authors(paper.authors)
        body.append(authors, style="italic")
        body.append("\n")

        year_str = str(paper.year) if paper.year is not None else "-"
        venue_str = paper.venue or "-"
        source_str = _format_source(paper.source)
        body.append(f"{year_str} · {venue_str} · {source_str}\n")

        if paper.summary_ko:
            body.append("\n")
            body.append(paper.summary_ko)
            body.append("\n")

        if paper.url:
            body.append("\n")
            body.append(paper.url, style="dim")

        title_line = f"[후보 {index + 1}/{total}]"
        return Panel(body, title=title_line, border_style="cyan", expand=True)


__all__ = ["RichCLI", "PaperDecision"]
