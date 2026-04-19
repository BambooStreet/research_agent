"""llm/client.py 의 순수 함수 유닛 테스트.

실제 Anthropic 호출은 하지 않는다. `_strip_json_fence` 와 같은 파서 헬퍼만 검증.
"""

from __future__ import annotations

import pytest

from research_agent.llm.client import _strip_json_fence


class TestStripJsonFence:
    """다양한 코드 펜스 패턴이 올바르게 제거되는지 확인."""

    def test_no_fence_unchanged(self) -> None:
        text = '{"a": 1}'
        assert _strip_json_fence(text) == '{"a": 1}'

    def test_removes_json_fence(self) -> None:
        text = '```json\n{"a": 1}\n```'
        assert _strip_json_fence(text) == '{"a": 1}'

    def test_removes_bare_fence(self) -> None:
        text = '```\n{"a": 1}\n```'
        assert _strip_json_fence(text) == '{"a": 1}'

    def test_uppercase_language_hint(self) -> None:
        text = '```JSON\n{"a": 1}\n```'
        assert _strip_json_fence(text) == '{"a": 1}'

    def test_trims_surrounding_whitespace(self) -> None:
        text = '   \n```json\n{"a": 1}\n```\n  '
        assert _strip_json_fence(text) == '{"a": 1}'

    def test_only_leading_fence(self) -> None:
        text = '```json\n{"a": 1}'
        assert _strip_json_fence(text) == '{"a": 1}'

    def test_inline_backticks_in_string_value_preserved(self) -> None:
        """응답 본문 중간에 백틱 3개가 있는 JSON 값도 원문을 유지해야 한다.

        MULTILINE 플래그로 `^```...```$` 가 본문 중간 라인을 매칭해 깨뜨리지 않도록
        문자열 양끝 앵커(\\A, \\Z)로 바뀐 뒤에는 이 케이스가 통과해야 한다.
        """

        # JSON 값 안에 코드 블록 예시가 들어간 응답.
        text = (
            '```json\n'
            '{"code": "prefix\\n```python\\nprint(1)\\n```\\nsuffix"}\n'
            '```'
        )
        cleaned = _strip_json_fence(text)
        # 양끝 펜스만 제거되고 본문의 이스케이프된 백틱 블록은 그대로 보존되어야 한다.
        assert cleaned.startswith('{"code":')
        assert cleaned.endswith('"}')
        assert "```python" in cleaned


class TestLLMClientJsonParsing:
    """`complete_json` 이 파서 오류를 명확한 ValueError 로 바꿔주는지.

    실제 HTTP 호출을 피하기 위해 `complete_text` 만 monkeypatch 한다.
    """

    def test_complete_json_parses_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from research_agent.llm.client import LLMClient

        # __init__ 에서 anthropic.Anthropic 이 실제 API 키 검증을 하지 않도록 패치.
        import research_agent.llm.client as mod

        class FakeAnth:
            def __init__(self, *args, **kwargs) -> None:
                pass

        monkeypatch.setattr(mod.anthropic, "Anthropic", FakeAnth)

        client = LLMClient(model="test-model")
        monkeypatch.setattr(
            client, "complete_text", lambda **_: '```json\n{"scope": "ok"}\n```'
        )
        assert client.complete_json(system="s", user="u") == {"scope": "ok"}

    def test_complete_json_empty_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from research_agent.llm.client import LLMClient
        import research_agent.llm.client as mod

        class FakeAnth:
            def __init__(self, *args, **kwargs) -> None:
                pass

        monkeypatch.setattr(mod.anthropic, "Anthropic", FakeAnth)

        client = LLMClient()
        monkeypatch.setattr(client, "complete_text", lambda **_: "")
        with pytest.raises(ValueError, match="empty response"):
            client.complete_json(system="s", user="u")

    def test_complete_json_invalid_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from research_agent.llm.client import LLMClient
        import research_agent.llm.client as mod

        class FakeAnth:
            def __init__(self, *args, **kwargs) -> None:
                pass

        monkeypatch.setattr(mod.anthropic, "Anthropic", FakeAnth)

        client = LLMClient()
        monkeypatch.setattr(client, "complete_text", lambda **_: "not json at all")
        with pytest.raises(ValueError, match="did not return valid JSON"):
            client.complete_json(system="s", user="u")

    def test_complete_json_non_object_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from research_agent.llm.client import LLMClient
        import research_agent.llm.client as mod

        class FakeAnth:
            def __init__(self, *args, **kwargs) -> None:
                pass

        monkeypatch.setattr(mod.anthropic, "Anthropic", FakeAnth)

        client = LLMClient()
        monkeypatch.setattr(client, "complete_text", lambda **_: "[1, 2, 3]")
        with pytest.raises(ValueError, match="must be an object"):
            client.complete_json(system="s", user="u")
