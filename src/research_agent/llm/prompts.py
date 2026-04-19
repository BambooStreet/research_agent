"""LLM 프롬프트 상수.

각 노드가 쓰는 system / user 메시지를 분리해 보관한다. user 템플릿은 `.format()`
으로 변수를 주입하므로 JSON 스키마 안의 중괄호는 `{{ }}` 로 이스케이프한다.
모두 순수 문자열 상수이므로 이 모듈은 side effect 가 전혀 없다.
"""

from __future__ import annotations

TOPIC_REFINE_SYSTEM = """당신은 학술 연구 주제 코치입니다. 사용자가 제시한 주제가 서베이 논문을 검색하기에 적절한 범위인지 평가합니다.

평가 기준:
- too_broad: 한 분야 전체를 포괄해 서베이 수십 건이 나올 정도로 넓은 주제
- ok: 1~2년치 서베이 5~15건 범위로 수렴 가능한 주제
- too_narrow: 서베이가 거의 존재하지 않을 만큼 지엽적인 주제

출력은 반드시 지시된 JSON 형식으로만 반환하고, 설명 문장이나 코드블록 감싸기를 추가하지 마세요."""

TOPIC_REFINE_USER_TEMPLATE = """사용자 주제: {topic}

출력은 아래 JSON 스키마 하나만 반환하세요. 다른 텍스트, 마크다운, 코드블록 감싸기 모두 금지.
{{
  "scope": "too_broad" | "ok" | "too_narrow",
  "reason": "한국어 1문장",
  "options": ["좁힌 옵션1", "좁힌 옵션2", "좁힌 옵션3"]
}}
scope=ok 이면 options는 빈 배열. scope=too_narrow 이면 options에 더 넓힌 주제 제안."""


QUERY_BUILD_SYSTEM = """당신은 학술 검색 전문가입니다. 한국어 연구 주제를 받아 arXiv / Semantic Scholar / OpenAlex 에서 통하는 영어 키워드 기반 검색 쿼리로 변환합니다.

규칙:
- primary_query 는 핵심 개념 2~4개를 공백 또는 AND 로 연결한 간결한 쿼리
- alternative_queries 는 동의어/상위어를 활용한 대체 쿼리 1~2개
- arxiv_categories 는 관련된 arXiv 카테고리 코드 배열 (예: "cs.CL", "cs.LG"). 관련 없으면 빈 배열
- 모든 쿼리는 영어로, 불필요한 따옴표/특수문자 최소화
- 출력은 지시된 JSON 형식만. 설명/코드블록 금지."""

QUERY_BUILD_USER_TEMPLATE = """입력 주제: {topic}

영어 키워드 기반의 학술 검색 쿼리를 만드세요. 출력은 아래 JSON 만:
{{
  "primary_query": "...",
  "alternative_queries": ["..."],
  "arxiv_categories": []
}}"""


SUMMARIZE_SYSTEM = """당신은 논문 초록을 한국어로 요약하는 연구 보조입니다.

규칙:
- 한국어 2~3문장, 평문만 (마크다운, 번호 목록, 따옴표 강조 금지)
- 1문장: 다루는 문제, 2문장: 핵심 기여, (선택) 3문장: 한계 또는 적용 분야
- 초록에 없는 내용을 추측해 추가하지 말 것"""

SUMMARIZE_USER_TEMPLATE = """제목: {title}
초록: {abstract}

위 초록을 한국어 2~3문장으로 요약하세요. 평문으로만, 마크다운/번호 금지."""


__all__ = [
    "QUERY_BUILD_SYSTEM",
    "QUERY_BUILD_USER_TEMPLATE",
    "SUMMARIZE_SYSTEM",
    "SUMMARIZE_USER_TEMPLATE",
    "TOPIC_REFINE_SYSTEM",
    "TOPIC_REFINE_USER_TEMPLATE",
]
