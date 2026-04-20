"""OpenAI Chat Completions API 래퍼.

노드 코드에서 SDK 의존을 숨기고 `complete_text` / `complete_json` 두 메서드만 노출한다.
JSON 응답은 OpenAI 네이티브 JSON 모드(`response_format={"type": "json_object"}`)로 받으며,
혹시 모델이 펜스를 감싸는 경우를 대비해 파싱 전 코드펜스 제거를 수행한다.
"""

from __future__ import annotations

import json
import re
from typing import Any

import openai

from research_agent.config import get_settings
from research_agent.logger import logger

# 선행/후행 코드블록 제거용 정규식. 문자열 양끝 앵커(\A, \Z) 로 고정해 본문 중간
# 백틱 펜스는 건드리지 않는다. (MULTILINE 을 쓰면 JSON 문자열 값 안의 ``` 가 매칭되어
# 원문을 깨는 버그가 발생한다.)
_FENCE_OPEN_RE = re.compile(r"\A\s*```(?:[a-zA-Z0-9_\-]+)?\s*\n?")
_FENCE_CLOSE_RE = re.compile(r"\n?\s*```\s*\Z")


def _strip_json_fence(text: str) -> str:
    """LLM 이 반환한 텍스트에서 앞뒤 마크다운 코드 펜스를 제거한다.

    OpenAI JSON 모드에선 펜스가 거의 붙지 않지만, 예외 케이스를 위한 방어선으로 유지.
    """

    cleaned = text.strip()
    cleaned = _FENCE_OPEN_RE.sub("", cleaned, count=1)
    cleaned = _FENCE_CLOSE_RE.sub("", cleaned, count=1)
    return cleaned.strip()


class LLMClient:
    """OpenAI chat completions API 의 얇은 래퍼.

    JSON 응답은 네이티브 JSON 모드를 쓰고, 혹시 모를 펜스는 `_strip_json_fence` 로 제거한 뒤
    `json.loads` 로 파싱한다.
    """

    def __init__(self, *, model: str | None = None, max_tokens: int = 1024) -> None:
        settings = get_settings()
        self.model: str = model or settings.openai_model
        self.max_tokens: int = max_tokens
        self._client = openai.OpenAI(
            api_key=settings.openai_api_key.get_secret_value()
        )

    def _create(
        self,
        *,
        system: str,
        user: str,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Chat completions 호출 후 첫 choice 의 content 를 문자열로 반환.

        응답이 비어있거나 choices 가 없으면 빈 문자열을 반환 — 상위 파서가 명확한 에러로 처리한다.
        """

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        logger.debug(
            "LLM 요청: model={model}, system_len={s}, user_len={u}, json_mode={j}",
            model=self.model,
            s=len(system),
            u=len(user),
            j=response_format is not None,
        )

        resp = self._client.chat.completions.create(**kwargs)

        choices = getattr(resp, "choices", None) or []
        if not choices:
            logger.debug("LLM 응답에 choices 없음")
            return ""
        content = getattr(choices[0].message, "content", None)
        result = content or ""
        logger.debug("LLM 응답 길이: {n}", n=len(result))
        return result

    def complete_text(self, *, system: str, user: str) -> str:
        """system + user 메시지를 보내 평문 응답을 받는다."""

        return self._create(system=system, user=user)

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

        raw = self._create(
            system=system,
            user=final_user,
            response_format={"type": "json_object"},
        )
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
