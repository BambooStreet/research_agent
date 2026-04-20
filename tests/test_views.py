"""cli/views.py 순수 함수 유닛 테스트."""

from __future__ import annotations

from research_agent.cli.views import _strip_surrogates


class TestStripSurrogates:
    """WSL/Windows 한글 IME 에서 섞여 들어오는 unpaired surrogate 방어."""

    def test_plain_ascii_unchanged(self) -> None:
        assert _strip_surrogates("hello world") == "hello world"

    def test_korean_unchanged(self) -> None:
        assert _strip_surrogates("페르소나 UX 평가") == "페르소나 UX 평가"

    def test_removes_lone_high_surrogate(self) -> None:
        # \udce3 같은 low surrogate 가 단독으로 끼어든 케이스 (실제 로그에서 발견).
        text = "ㅕ페르소나 기반 \udce3\udc85UX 평가하는 에이전\udced트"
        cleaned = _strip_surrogates(text)
        assert cleaned == "ㅕ페르소나 기반 UX 평가하는 에이전트"
        # 정리된 문자열은 UTF-8 로 round-trip 가능해야 한다.
        assert cleaned.encode("utf-8").decode("utf-8") == cleaned

    def test_full_surrogate_range_stripped(self) -> None:
        # D800 ~ DFFF 전체가 제거 대상.
        text = "a\ud800b\udbffc\udc00d\udfffe"
        assert _strip_surrogates(text) == "abcde"

    def test_empty_string(self) -> None:
        assert _strip_surrogates("") == ""

    def test_only_surrogates_becomes_empty(self) -> None:
        assert _strip_surrogates("\ud800\udfff") == ""
