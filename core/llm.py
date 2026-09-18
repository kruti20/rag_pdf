import os
import time

from groq import APIConnectionError, Groq, InternalServerError, RateLimitError

RETRYABLE_ERRORS = (RateLimitError, InternalServerError, APIConnectionError)

# Groq's free-tier model lineup shifts over time (check client.models.list()
# if these stop working). llama-3.3-70b-versatile from the original tech
# stack doc no longer exists; gpt-oss-120b/20b are the current equivalents.
MODEL_PRIMARY = "openai/gpt-oss-120b"
MODEL_FALLBACK = "openai/gpt-oss-20b"
MAX_TOKENS = 1024


class GroqLLM:
    def __init__(
        self,
        client=None,
        model: str = MODEL_PRIMARY,
        max_retries: int = 3,
        backoff_base_seconds: float = 2.0,
        sleep_fn=time.sleep,
    ):
        self.client = client or Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model = model
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.sleep_fn = sleep_fn

    def generate(self, prompt: str) -> str:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=MAX_TOKENS,
                )
                return response.choices[0].message.content
            except RETRYABLE_ERRORS as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    self.sleep_fn(self.backoff_base_seconds * (2**attempt))
        raise last_error
