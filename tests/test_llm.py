import httpx
import pytest
from groq import RateLimitError

from core.llm import GroqLLM


def _rate_limit_error():
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com/x"))
    return RateLimitError("rate limited", response=response, body=None)


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class _FakeClient:
    def __init__(self, responses):
        self.chat = self
        self.completions = _FakeCompletions(responses)


def _fake_success_response(text):
    class _Message:
        content = text

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]

    return _Response()


def test_generate_returns_content_on_success():
    client = _FakeClient([_fake_success_response("hello")])
    llm = GroqLLM(client=client, sleep_fn=lambda seconds: None)

    assert llm.generate("some prompt") == "hello"


def test_generate_retries_once_after_rate_limit_then_succeeds():
    client = _FakeClient([_rate_limit_error(), _fake_success_response("recovered")])
    llm = GroqLLM(client=client, sleep_fn=lambda seconds: None)

    result = llm.generate("some prompt")

    assert result == "recovered"
    assert client.completions.calls == 2


def test_generate_raises_after_exhausting_retries():
    client = _FakeClient([_rate_limit_error(), _rate_limit_error(), _rate_limit_error()])
    llm = GroqLLM(client=client, max_retries=3, sleep_fn=lambda seconds: None)

    with pytest.raises(RateLimitError):
        llm.generate("some prompt")

    assert client.completions.calls == 3
