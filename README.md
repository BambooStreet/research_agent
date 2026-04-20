# research-agent

논문 서베이 워크플로우를 반자동으로 수행하는 개인용 CLI 도구. 주제를 좁히고, 서베이 논문을 자동 검색한 뒤, human-in-the-loop 로 승인/거부한다.

## 주요 기능 (MVP)

- **주제 좁히기**: 입력한 주제가 너무 넓으면 LLM 이 3~5개의 하위 주제를 제안하고 사용자가 선택/직접 입력/원래 주제 유지 중 고른다.
- **서베이 자동 검색**: arXiv · Semantic Scholar · OpenAlex 3개 소스를 병렬 호출해 서베이/리뷰 논문을 모은 뒤 DOI/제목 기반으로 중복 제거한다.
- **한국어 요약**: 각 후보의 영어 초록을 한국어 2~3문장으로 요약해 화면에 표시한다.
- **HITL 검증**: 한 건씩 `y(승인) / n(거부) / s(보류) / q(종료)` 로 분류한다. 모두 거부되면 새 쿼리로 자동 재검색(최대 3회).
- **JSON 세션 저장**: 승인/거부 결과를 `data/sessions/` 아래 JSON 으로 기록한다.

## 설치

의존성 관리는 [uv](https://docs.astral.sh/uv/) 로 한다. Python 3.11 이상이 필요하다.

```bash
# uv 가 없으면 먼저 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 의존성 설치 (.venv 자동 생성, uv.lock 기반 재현 가능한 설치)
uv sync --extra dev

cp .env.example .env
# .env 파일에 ANTHROPIC_API_KEY 입력
```

의존성 변경 시:

```bash
uv add <package>          # 런타임 의존성 추가
uv add --dev <package>    # 개발 의존성 추가
uv lock                   # pyproject.toml 변경 후 lockfile 재생성
uv sync                   # lockfile 기준으로 .venv 재동기화
```

`uv.lock` 은 커밋한다 (재현 가능한 빌드를 위해).

`.env` 에서 선택적으로 설정할 수 있는 키:

- `SEMANTIC_SCHOLAR_API_KEY` — 있으면 Semantic Scholar 요청에 쓰여 레이트 리밋이 완화된다.
- `OPENALEX_MAILTO` — OpenAlex polite pool 용 이메일.
- `ANTHROPIC_MODEL` — 기본값은 최신 Sonnet. 모델 변경 시 덮어쓴다.
- `LOG_LEVEL` — 기본 `INFO`.

## 실행

```bash
uv run research-agent start  # 새 세션 시작
uv run research-agent list   # 저장된 세션 목록 출력
```

세션 JSON 은 `data/sessions/<YYYY-MM-DD>_<session_id>.json` 경로에 저장된다. 파일명에 날짜 접두사가 붙어 있어 디렉토리만 훑어도 최신 세션을 찾기 쉽다.

## 개발

```bash
uv run pytest -q
uv run pytest --cov=src/research_agent
uv run ruff check .
uv run ruff format .
```

테스트는 외부 네트워크 호출 없이 모두 모킹되어 오프라인에서 돈다 (`respx` 로 httpx 주입, LLM/검색 클라이언트는 fake 구현 주입).

## 아키텍처

LangGraph `StateGraph` 한 개로 전체 대화 흐름을 관리한다. 각 노드는 `ResearchState` 를 읽고 부분 업데이트 dict 를 돌려준다.

```
START
  └─ start_session         # UUID 발급, 상태 초기화
       └─ collect_topic    # CLI 에서 raw_topic 입력
            └─ refine_topic    # LLM 이 scope 평가 + 옵션 제안
                 └─ confirm_topic
                      ├─(confirmed)─ build_queries
                      │                └─ search_surveys      # 3소스 병렬 + dedupe
                      │                     └─ summarize_candidates  # 한국어 요약
                      │                          └─ present_and_review # y/n/s/q HITL
                      │                               ├─(persist)─ persist_session ─ END
                      │                               └─(retry)─ build_queries (재검색, 최대 3회)
                      └─(not confirmed)─ refine_topic (재진입)
```

주요 디렉토리:

```
src/research_agent/
├── agents/       # state, nodes, graph (LangGraph)
├── cli/          # app (click), repl, views (rich)
├── llm/          # Anthropic 래퍼, 프롬프트 상수
├── models/       # Paper, Session (pydantic v2)
├── storage/      # 세션 JSON save/load/list
└── tools/        # arxiv/semantic_scholar/openalex 클라이언트, dedup, http
```

## 향후 계획

MVP 는 서베이 검색 + HITL 까지를 다룬다. 노드를 추가하는 방향으로 확장한다.

- **2-3 핵심 선행연구 추출**: 승인된 서베이의 PDF 에서 참고문헌 목록을 뽑고 빈도가 높은 논문을 선행연구 후보로 제시.
- **2-4 후속 논문 수집**: Semantic Scholar `/paper/{id}/citations` 로 인용 논문 수집.
- **2-5 역할 분류**: LLM 으로 각 논문을 `survey / problem / method / recent` 중 하나로 라벨링 (`Paper.role` enum 이 이미 선언됨).
- **2-6~2-8**: 노트 작성, 스크리닝 보드, 리포트 생성 등.

Protocol (`PaperSearchClient`) 와 `Paper.role` enum 이 이미 확장 지점을 마련해 두어 소스/역할 추가는 기존 코드를 건드리지 않고 가능하다.
