"""Anthropic Claude API 래퍼.

노드 코드에서 SDK 의존을 숨기고 `complete_text` / `complete_json` 두 메서드만 노출한다.
JSON 응답은 모델이 종종 ```json``` 코드블록으로 감싸 보내므로 파싱 전 울타리를 제거한다.
"""

from __future__ import annotations

import json
import re
from typing import Any

import anthropic

from research_agent.config import get_settings
from research_agent.logger import logger

# 선행/후행 코드블록 제거용 정규식. 문자열 양끝 앵커(\A, \Z) 로 고정해 본문 중간
# 백틱 펜스는 건드리지 않는다. (MULTILINE 을 쓰면 JSON 문자열 값 안의 ``` 가 매칭되어
# 원문을 깨는 버그가 발생한다.)
_FENCE_OPEN_RE = re.compile(r"\A\s*```(?:[a-zA-Z0-9_\-]+)?\s*\n?")
_FENCE_CLOSE_RE = re.compile(r"\n?\s*```\s*\Z")


def _strip_json_fence(text: str) -> str:
    """LLM 이 반환한 텍스트에서 앞뒤 마크다운 코드 펜스를 제거한다.

    ```json ... ``` / ``` ... ``` 패턴만 처리한다. 중간의 코드블록까지 건드리면
    본문 JSON 을 깨뜨릴 수 있으므로 문자열 양끝만 손본다.
    """

    cleaned = text.strip()
    cleaned = _FENCE_OPEN_RE.sub("", cleaned, count=1)
    cleaned = _FENCE_CLOSE_RE.sub("", cleaned, count=1)
    return cleaned.strip()


class LLMClient:
    """Anthropic messages API 의 얇은 래퍼.

    JSON 응답은 ```json``` 코드 펜스를 제거한 뒤 `json.loads` 로 파싱한다.
    """

    def __init__(self, *, model: str | None = None, max_tokens: int = 1024) -> None:
        settings = get_settings()
        self.model: str = model or settings.anthropic_model
        self.max_tokens: int = max_tokens
        self._client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key.get_secret_value()
        )

    def complete_text(self, *, system: str, user: str) -> str:
        """system + user 메시지를 보내 평문 응답을 받는다.

        응답이 비어있거나 text 블록이 없으면 빈 문자열을 반환한다. 호출부가 JSON 파서에
        넘길 때 빈 문자열이면 명확한 에러로 떨어지므로 별도 예외를 던지지 않는다.
        """

        logger.debug(
            "LLM 요청: model={model}, system_len={s}, user_len={u}",
            model=self.model,
            s=len(system),
            u=len(user),
        )

        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )

        content = getattr(resp, "content", None) or []
        text_parts: list[str] = []
        for block in content:
            # Anthropic SDK 는 text 블록을 `TextBlock(type="text", text=...)` 로 보낸다.
            text = getattr(block, "text", None)
            if isinstance(text, str):
                text_parts.append(text)

        result = "".join(text_parts)
        logger.debug("LLM 응답 길이: {n}", n=len(result))
        return result

    def complete_json(
        self, *, system: str, user: str, schema_hint: str | None = None
    ) -> dict[str, Any]:
        """system + user 로 JSON 응답을 요청하고 파싱한 dict 를 반환한다.

        `schema_hint` 가 주어지면 user 프롬프트 말미에 스키마 힌트를 덧붙인다.
        파싱 실패 시 `ValueError` 로 원인을 밝혀 상위 노드가 로깅 후 폴백할 수 있게 한다.
        """

        final_user = user
        if schema_hint:
            final_user = f"{user}\n\nJSON schema hint:\n{schema_hint}"

        raw = self.complete_text(system=system, user=final_user)
        cleaned = _strip_json_fence(raw)

        if not cleaned:
            raise ValueError("LLM did not return valid JSON: (empty response)")

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            # 원문이 과도하게 길면 앞 200자만 노출. 비밀 정보가 섞일 여지가 없는 응답이라
            # 로그에 포함해도 무방.
            excerpt = cleaned[:200].replace("\n", " ")
            raise ValueError(f"LLM did not return valid JSON: {excerpt!r}") from exc

        if not isinstance(parsed, dict):
            raise ValueError(
                f"LLM JSON must be an object, got {type(parsed).__name__}: {cleaned[:200]!r}"
            )
        return parsed


__all__ = ["LLMClient"]
