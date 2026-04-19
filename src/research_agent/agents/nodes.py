"""LangGraph 노드 구현.

테스트 용이성을 위해 모든 외부 의존(LLM / 검색 클라이언트 / CLI / storage) 을 인자로 주입한다.
`make_nodes(...)` 팩토리가 실제 노드 함수들을 클로저로 감싸 반환하므로, 테스트에서는
fake 객체를 주입하고 운영에서는 기본 구현을 쓰면 된다.

노드는 모두 `(state: ResearchState) -> dict` 형태로, LangGraph 가 반환 dict 를 state 에
머지하는 방식에 맞춘다. Paper 객체는 pydantic BaseModel 그대로 state 에 담기고,
최종 직렬화는 `persist_session` 노드에서 `Session.model_dump_json` 으로 수행된다.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol, TypedDict, cast
from uuid import UUID, uuid4

from research_agent.agents.state import ResearchState
from research_agent.config import get_settings
from research_agent.llm.client import LLMClient
from research_agent.llm.prompts import (
    QUERY_BUILD_SYSTEM,
    QUERY_BUILD_USER_TEMPLATE,
    SUMMARIZE_SYSTEM,
    SUMMARIZE_USER_TEMPLATE,
    TOPIC_REFINE_SYSTEM,
    TOPIC_REFINE_USER_TEMPLATE,
)
from research_agent.logger import logger
from research_agent.models.paper import Paper, PaperSource, PaperStatus
from research_agent.models.session import Session, SessionStage
from research_agent.storage.session_store import save_session
from research_agent.tools import arxiv_client, openalex_client, semantic_scholar_client
from research_agent.tools.base import PaperSearchClient
from research_agent.tools.dedup import dedupe_papers

# 노드당 외부 검색 소스별로 요청할 건수 (중복 제거 전 합산 15건 목표).
_PER_SOURCE_LIMIT: int = 5
# 중복 제거 후 사용자에게 제시할 최대 후보 수.
_CANDIDATE_LIMIT: int = 10

PaperDecision = Literal["y", "n", "s", "q"]

# arXiv 카테고리는 `archive.group` 형식(`cs.CL`, `stat.ML`, `cond-mat.mes-hall` 등).
# LLM 이 간혹 이상한 문자열을 내놓을 수 있어 정규식으로 강 검증한다.
_ARXIV_CAT_RE = re.compile(r"^[a-z\-]+\.[a-zA-Z\-]+$")


class CLIInterface(Protocol):
    """사용자 입출력을 캡슐화한 Protocol.

    Phase C 에서는 시그니처만 정의하고, Phase D 에서 rich + click 기반 실구현을
    `cli/views.py` 에 둔다. 모든 메서드는 동기 호출이며, 블로킹 입력을 사용한다.
    """

    def ask_topic(self) -> str:
        """초기 주제를 받는다."""

    def ask_topic_choice(self, options: list[str], reason: str) -> str:
        """리파인 옵션 중 선택을 받는다. 반환값: '1'/'2'/'3'/'d'(직접)/'k'(그대로)."""

    def ask_topic_direct(self) -> str:
        """사용자가 직접 입력하는 주제 텍스트를 받는다."""

    def show_candidates_header(self, total: int, per_source: dict[str, int]) -> None:
        """검색 결과 요약 헤더를 화면에 출력한다."""

    def ask_paper_decision(self, paper: Paper, index: int, total: int) -> PaperDecision:
        """한 건의 논문에 대한 승인/거부/보류/종료 입력을 받는다."""

    def notify(self, message: str) -> None:
        """일반 메시지를 화면에 출력한다."""


# LangGraph 는 노드가 반환할 수 있는 dict 의 구조를 강제하지 않는다. 여기서는 가독성을 위해
# NodeResult 라는 alias 만 두고, 실제 반환은 노드별로 부분 dict 다.
NodeResult = dict


class NodesBundle(TypedDict):
    """`build_graph` 가 StateGraph 에 등록할 노드 함수 묶음."""

    start_session: Callable[[ResearchState], NodeResult]
    collect_topic: Callable[[ResearchState], NodeResult]
    refine_topic: Callable[[ResearchState], NodeResult]
    confirm_topic: Callable[[ResearchState], NodeResult]
    build_queries: Callable[[ResearchState], NodeResult]
    search_surveys: Callable[[ResearchState], NodeResult]
    summarize_candidates: Callable[[ResearchState], NodeResult]
    present_and_review: Callable[[ResearchState], NodeResult]
    persist_session: Callable[[ResearchState], NodeResult]


def _default_search_clients() -> list[PaperSearchClient]:
    """운영 기본 클라이언트 세트 (arxiv, semantic_scholar, openalex).

    모듈을 그대로 `PaperSearchClient` 로 취급한다. Protocol 은 구조적 타이핑이라
    SOURCE 상수와 `search` 함수만 있으면 만족한다. 타입 체커는 module 객체가
    Protocol 에 부합함을 추론하지 못하므로 `cast` 로 의도를 명시한다.
    """

    return [
        cast(PaperSearchClient, arxiv_client),
        cast(PaperSearchClient, openalex_client),
        cast(PaperSearchClient, semantic_scholar_client),
    ]


def _merge_arxiv_categories(query: str, categories: list[str]) -> str:
    """arXiv 쿼리에만 카테고리 필터를 AND 결합한다.

    카테고리가 여러 개면 괄호 내부에서 OR 로 묶는다. 빈 배열이면 원 쿼리 그대로.
    정규식에 맞지 않는 항목은 걸러내 arXiv 쿼리 파서 오류를 피한다.
    """

    valid = [
        c.strip()
        for c in categories
        if c and c.strip() and _ARXIV_CAT_RE.match(c.strip())
    ]
    if not valid:
        return query
    if len(valid) == 1:
        return f"({query}) AND cat:{valid[0]}"
    joined = " OR ".join(f"cat:{c}" for c in valid)
    return f"({query}) AND ({joined})"


def make_nodes(
    *,
    llm: LLMClient | None = None,
    search_clients: Sequence[PaperSearchClient] | None = None,
    cli: CLIInterface | None = None,
    storage_save: Callable[[Session], Path] | None = None,
) -> NodesBundle:
    """외부 의존을 주입받아 노드 함수 묶음을 반환한다.

    기본값 해석:
    - `llm` 미지정 시 `LLMClient()` 로 실 Anthropic 호출 (ANTHROPIC_API_KEY 필요)
    - `search_clients` 미지정 시 arxiv/openalex/semantic_scholar 모듈 사용
    - `cli` 는 필수 (Phase D 의 CLI 구현 주입 기대) — None 이면 사용 시점에 AttributeError
    - `storage_save` 미지정 시 `save_session` 사용
    """

    # 주입값 바인딩. 클로저가 아래 노드 함수들에서 참조한다.
    llm_client = llm if llm is not None else LLMClient()
    clients = list(search_clients) if search_clients is not None else _default_search_clients()
    save_fn = storage_save if storage_save is not None else save_session

    if cli is None:
        # CLI 없이 노드만 유닛 테스트하는 경우를 위해 Placeholder 대신 즉시 에러.
        raise ValueError("cli must be provided (CLIInterface implementation required)")

    cli_impl = cli

    def start_session(state: ResearchState) -> NodeResult:
        """세션 식별자와 빈 컬렉션들을 초기화한다."""

        now_iso = datetime.now(timezone.utc).isoformat()
        sid = str(uuid4())
        logger.info("세션 시작: {sid}", sid=sid)
        return {
            "session_id": sid,
            "created_at": now_iso,
            "topic_options": [],
            "topic_reason": "",
            "topic_confirmed": False,
            "search_queries": [],
            "candidates": [],
            "approved": [],
            "rejected": [],
            "deferred": [],
            "pending_indices": [],
            "retry_count": 0,
            "current_stage": "topic",
            "error": None,
        }

    def collect_topic(state: ResearchState) -> NodeResult:
        """CLI 에서 초기 주제 문자열을 받는다."""

        topic = cli_impl.ask_topic().strip()
        logger.info("주제 입력: {topic}", topic=topic)
        return {"raw_topic": topic, "current_stage": "topic"}

    def refine_topic(state: ResearchState) -> NodeResult:
        """LLM 으로 주제 범위를 평가하고 옵션을 준비한다.

        LLM 응답 실패 시 scope=ok 로 폴백해 워크플로우를 막지 않는다.
        """

        topic = state.get("raw_topic", "")
        user_msg = TOPIC_REFINE_USER_TEMPLATE.format(topic=topic)

        try:
            parsed = llm_client.complete_json(
                system=TOPIC_REFINE_SYSTEM, user=user_msg
            )
        except ValueError as exc:
            logger.warning("주제 리파인 JSON 파싱 실패, ok 로 폴백: {err}", err=str(exc))
            return {
                "topic_options": [],
                "refined_topic": topic,
                "current_stage": "refine",
            }

        scope = parsed.get("scope", "ok")
        reason = parsed.get("reason", "")
        options = parsed.get("options", []) or []
        # options 는 list[str] 이어야 한다. 타입 이상 시 무시.
        clean_options = [str(o) for o in options if isinstance(o, (str, int, float))]

        logger.info(
            "주제 평가: scope={scope}, options={n}, reason={r}",
            scope=scope,
            n=len(clean_options),
            r=reason,
        )

        update: NodeResult = {
            "topic_options": clean_options,
            "topic_reason": str(reason),
            "current_stage": "refine",
        }
        # scope=ok 면 옵션 없이 바로 refined_topic 확정 단계로 넘긴다.
        if scope == "ok" or not clean_options:
            update["refined_topic"] = topic
        return update

    def confirm_topic(state: ResearchState) -> NodeResult:
        """옵션이 있으면 CLI 로 선택받고, 없으면 raw_topic 을 그대로 확정한다."""

        options = state.get("topic_options", []) or []
        raw_topic = state.get("raw_topic", "")

        if not options:
            # refine_topic 이 scope=ok 로 판정한 경우.
            return {
                "refined_topic": raw_topic,
                "topic_confirmed": True,
                "current_stage": "refine",
            }

        reason = state.get("topic_reason", "") or ""
        choice = cli_impl.ask_topic_choice(options, reason).strip().lower()

        # 숫자 선택 → 해당 옵션
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                chosen = options[idx]
                return {
                    "refined_topic": chosen,
                    "topic_confirmed": True,
                    "current_stage": "refine",
                }
            # 범위를 벗어난 숫자면 그대로 유지 (재진입).
            logger.warning("잘못된 선택 인덱스: {c}", c=choice)
            return {"topic_confirmed": False, "current_stage": "refine"}

        if choice == "d":
            direct = cli_impl.ask_topic_direct().strip()
            return {
                "refined_topic": direct,
                "topic_confirmed": True,
                "current_stage": "refine",
            }

        if choice == "k":
            return {
                "refined_topic": raw_topic,
                "topic_confirmed": True,
                "current_stage": "refine",
            }

        # 알 수 없는 입력 → 재진입 (route_after_confirm 이 refine_topic 으로 돌린다).
        logger.warning("알 수 없는 선택 입력: {c}", c=choice)
        return {"topic_confirmed": False, "current_stage": "refine"}

    def build_queries(state: ResearchState) -> NodeResult:
        """refined_topic 으로부터 검색 쿼리 JSON 을 생성한다.

        LLM 실패 시 refined_topic 자체를 primary 로 사용해 워크플로우를 진행한다.
        arxiv_categories 는 primary 쿼리에만 병합하고, 최종 `search_queries` 리스트의
        첫 번째 원소로 넣는다.

        진입 시 이전 round 의 후보/분류 결과를 전부 초기화한다. retry 분기로 재진입할
        때 이전 round 의 approved/rejected/deferred 가 섞여 route_after_review 판정과
        세션 파일에 오염되는 것을 방지한다. 첫 진입 시에는 이미 빈 상태라 부작용 없음.
        """

        topic = state.get("refined_topic") or state.get("raw_topic", "")
        user_msg = QUERY_BUILD_USER_TEMPLATE.format(topic=topic)

        primary = topic
        alternatives: list[str] = []
        categories: list[str] = []

        try:
            parsed = llm_client.complete_json(
                system=QUERY_BUILD_SYSTEM, user=user_msg
            )
            primary = str(parsed.get("primary_query") or topic).strip() or topic
            raw_alts = parsed.get("alternative_queries", []) or []
            alternatives = [str(a).strip() for a in raw_alts if str(a).strip()]
            raw_cats = parsed.get("arxiv_categories", []) or []
            categories = [str(c).strip() for c in raw_cats if str(c).strip()]
        except ValueError as exc:
            logger.warning("쿼리 생성 JSON 파싱 실패, topic 사용: {err}", err=str(exc))

        primary_with_cats = _merge_arxiv_categories(primary, categories)
        queries = [primary_with_cats, *alternatives]

        logger.info(
            "검색 쿼리 {n}개 생성: primary={p}", n=len(queries), p=primary_with_cats
        )
        return {
            "search_queries": queries,
            # round 리셋: retry 재진입에서도 동일 로직. 첫 진입 시 빈 값과 동일해 무해.
            "candidates": [],
            "approved": [],
            "rejected": [],
            "deferred": [],
            "pending_indices": [],
            "current_stage": "search",
        }

    def search_surveys(state: ResearchState) -> NodeResult:
        """3개 클라이언트를 병렬 호출하고 결과를 중복 제거해 candidates 로 담는다.

        한 소스가 실패해도 전체를 죽이지 않고 빈 리스트로 대체한다.
        retry_count 에 따라 `search_queries` 를 회전해 매 round 다른 alternative 를
        사용한다. (첫 round = index 0 = primary, 그 다음 round 부터 alternatives.)
        """

        queries = state.get("search_queries") or []
        if not queries:
            logger.warning("search_queries 비어있음, 검색 건너뜀")
            return {
                "candidates": [],
                "pending_indices": [],
                "current_stage": "search",
            }

        retry = int(state.get("retry_count", 0) or 0)
        # retry 가 alternative 수를 넘어서면 마지막 쿼리로 고정 (out-of-range 방지).
        current_query = queries[min(retry, len(queries) - 1)]
        per_source: dict[str, int] = {}
        collected: list[Paper] = []

        def run_client(client: PaperSearchClient) -> tuple[str, list[Paper]]:
            source_name = client.SOURCE.value if hasattr(client, "SOURCE") else "unknown"
            try:
                papers = client.search(
                    current_query, is_survey=True, limit=_PER_SOURCE_LIMIT
                )
                return source_name, papers
            except Exception as exc:  # noqa: BLE001
                # 개별 소스 실패는 전체 워크플로우를 막지 않는다.
                logger.warning(
                    "검색 실패 (source={src}): {err}", src=source_name, err=repr(exc)
                )
                return source_name, []

        # ThreadPoolExecutor 로 병렬 제출하되, 수집은 원래 `clients` 순서대로 해
        # dedup 결과(최초 등장 인덱스 기반 정렬)가 결정적이도록 유지한다.
        with ThreadPoolExecutor(max_workers=max(1, len(clients))) as pool:
            futures = [pool.submit(run_client, c) for c in clients]
            for fut in futures:
                source_name, papers = fut.result()
                per_source[source_name] = len(papers)
                collected.extend(papers)

        deduped = dedupe_papers(collected)
        limited = deduped[:_CANDIDATE_LIMIT]

        logger.info(
            "검색 완료: per_source={per}, dedup_before={before}, final={after}",
            per=per_source,
            before=len(collected),
            after=len(limited),
        )

        return {
            "candidates": limited,
            "pending_indices": list(range(len(limited))),
            "current_stage": "search",
        }

    def summarize_candidates(state: ResearchState) -> NodeResult:
        """각 후보의 초록을 한국어 2~3문장으로 요약해 summary_ko 필드를 채운다.

        순차 호출 (병렬 아님) — 레이트 리밋과 토큰 비용을 보수적으로 관리.
        abstract 가 없으면 LLM 호출을 생략하고 플레이스홀더 문자열을 넣는다.
        """

        candidates: list[Paper] = list(state.get("candidates", []) or [])
        updated: list[Paper] = []

        for paper in candidates:
            if not paper.abstract:
                # model_copy 로 얕은 복제 후 summary_ko 갱신. 원본 Paper 는 불변 유지.
                updated.append(paper.model_copy(update={"summary_ko": "초록 없음"}))
                continue
            user_msg = SUMMARIZE_USER_TEMPLATE.format(
                title=paper.title, abstract=paper.abstract
            )
            try:
                summary = llm_client.complete_text(
                    system=SUMMARIZE_SYSTEM, user=user_msg
                ).strip()
            except Exception as exc:  # noqa: BLE001
                # 요약 실패는 전체를 막지 않고 플레이스홀더로 대체.
                logger.warning("요약 실패 (title={t}): {err}", t=paper.title, err=repr(exc))
                summary = "요약 생성 실패"
            updated.append(paper.model_copy(update={"summary_ko": summary}))

        logger.info("요약 완료 {n}건", n=len(updated))
        return {"candidates": updated, "current_stage": "search"}

    def present_and_review(state: ResearchState) -> NodeResult:
        """후보를 하나씩 제시해 y/n/s/q 입력을 받고 approved/rejected/deferred 로 분류한다.

        "q" 입력 시 즉시 중단(남은 후보는 pending_indices 에 남김).
        모두 거부/보류되어 승인 0 건이면 retry_count 를 증가시켜 조건부 엣지가 재검색으로
        분기하게 한다. build_queries 가 재진입 시 이전 round 의 분류 컬렉션을 리셋하므로
        여기서는 이번 round 결과만 state 에 넣으면 된다.
        """

        candidates: list[Paper] = list(state.get("candidates", []) or [])
        total = len(candidates)
        per_source: dict[str, int] = {}
        for p in candidates:
            key = p.source.value if isinstance(p.source, PaperSource) else str(p.source)
            per_source[key] = per_source.get(key, 0) + 1

        cli_impl.show_candidates_header(total, per_source)

        # 이번 round 결과만 담는다 (이전 round 는 build_queries 에서 리셋됨).
        approved: list[Paper] = []
        rejected: list[Paper] = []
        deferred: list[Paper] = []
        pending_indices: list[int] = []

        stopped_early = False
        for idx, paper in enumerate(candidates):
            decision = cli_impl.ask_paper_decision(paper, idx, total)
            if decision == "y":
                approved.append(paper.model_copy(update={"status": PaperStatus.APPROVED}))
            elif decision == "n":
                rejected.append(paper.model_copy(update={"status": PaperStatus.REJECTED}))
            elif decision == "s":
                # 보류: DEFERRED 상태로 별도 컬렉션(deferred) 에 담는다.
                deferred.append(paper.model_copy(update={"status": PaperStatus.DEFERRED}))
            elif decision == "q":
                # 남은 후보는 pending 으로 기록하고 중단.
                pending_indices = list(range(idx, total))
                stopped_early = True
                break
            else:
                # 알 수 없는 입력은 보수적으로 보류 처리.
                logger.warning("알 수 없는 결정 입력: {d}", d=decision)
                deferred.append(paper.model_copy(update={"status": PaperStatus.DEFERRED}))

        # 재시도 카운터 증가 조건: "승인 0 건 + q 로 조기 종료되지 않음 + 전원 처리 완료".
        # stopped_early=False 면 for 루프가 break 없이 끝난 것이므로 len(pending)=0 이며
        # 전원 처리가 완료되었음을 의미한다. 이 시점이 정확히 retry_count 증가 지점.
        new_retry = state.get("retry_count", 0) or 0
        if not stopped_early and not approved:
            new_retry += 1
            logger.info("승인 0건 → 재시도 카운터 증가: {n}", n=new_retry)

        return {
            "approved": approved,
            "rejected": rejected,
            "deferred": deferred,
            "pending_indices": pending_indices,
            "retry_count": new_retry,
            "current_stage": "review",
        }

    def persist_session(state: ResearchState) -> NodeResult:
        """현재 state 를 `Session` 으로 조립해 JSON 으로 저장한다."""

        # Session.session_id 는 UUID 타입이므로 state 의 문자열을 UUID 로 복원한다.
        raw_sid = state.get("session_id")
        sid = _coerce_uuid(raw_sid) if raw_sid else uuid4()

        # start_session 이 기록한 created_at (ISO 문자열) 을 datetime 으로 되살린다.
        # 저장 중에 시계가 흘러도 세션 생성 시각은 보존되어야 감사/목록 정렬이 정확하다.
        created_at_iso = state.get("created_at")
        try:
            created_at = (
                datetime.fromisoformat(created_at_iso)
                if created_at_iso
                else datetime.now(timezone.utc)
            )
        except ValueError:
            logger.warning(
                "created_at ISO 파싱 실패, 현재 시각으로 대체: {v}", v=created_at_iso
            )
            created_at = datetime.now(timezone.utc)

        session = Session(
            session_id=sid,
            created_at=created_at,
            raw_topic=state.get("raw_topic", ""),
            refined_topic=state.get("refined_topic"),
            candidates=list(state.get("candidates", []) or []),
            approved=list(state.get("approved", []) or []),
            rejected=list(state.get("rejected", []) or []),
            deferred=list(state.get("deferred", []) or []),
            current_stage=SessionStage.DONE,
            retry_count=state.get("retry_count", 0) or 0,
        )
        path = save_fn(session)
        cli_impl.notify(f"세션 저장 완료: {path}")
        logger.info("세션 저장 경로: {p}", p=str(path))
        return {"current_stage": "done"}

    return NodesBundle(
        start_session=start_session,
        collect_topic=collect_topic,
        refine_topic=refine_topic,
        confirm_topic=confirm_topic,
        build_queries=build_queries,
        search_surveys=search_surveys,
        summarize_candidates=summarize_candidates,
        present_and_review=present_and_review,
        persist_session=persist_session,
    )


def _coerce_uuid(value: str) -> UUID:
    """state 에 담긴 session_id 문자열을 UUID 로 복원한다.

    Session.session_id 는 UUID 타입이므로 변환 실패 시 새 UUID 로 폴백한다 (상태 무결성 우선).
    """

    try:
        return UUID(value)
    except (ValueError, TypeError):
        logger.warning("session_id UUID 변환 실패, 새 UUID 발급: {v}", v=value)
        return uuid4()


# --- 조건부 엣지 라우팅 함수 ---------------------------------------------------


def route_after_confirm(state: ResearchState) -> str:
    """confirm_topic 이후 분기.

    topic_confirmed=True → build_queries 로, 아니면 refine_topic 재진입.
    """

    return "build_queries" if state.get("topic_confirmed") else "refine_topic"


def route_after_review(state: ResearchState) -> str:
    """present_and_review 이후 분기.

    - approved 가 있고 모든 후보가 처리되었으면 persist
    - 승인 0 건이며 모든 후보가 처리(rejected + deferred) 되었고 retry 여유가 있으면 retry
    - 그 외는 persist (q 로 조기 종료된 경우 포함)
    """

    approved = state.get("approved", []) or []
    candidates = state.get("candidates", []) or []
    rejected = state.get("rejected", []) or []
    deferred = state.get("deferred", []) or []
    retry = state.get("retry_count", 0) or 0
    max_retry = get_settings().max_retry_loops

    # deferred 도 "처리된" 것으로 간주해 all_reviewed 계산에 포함.
    all_reviewed = len(approved) + len(rejected) + len(deferred) >= len(candidates)
    if approved and all_reviewed:
        return "persist"
    if all_reviewed and not approved and retry < max_retry:
        return "retry"
    return "persist"


__all__ = [
    "CLIInterface",
    "NodesBundle",
    "PaperDecision",
    "make_nodes",
    "route_after_confirm",
    "route_after_review",
]
