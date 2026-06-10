"""Cached + back-off OpenRouter model wrappers for the RAPTOR harness.

These subclass the RAPTOR base classes so they slot directly into a
``RetrievalAugmentationConfig``. Both summarization and QA go through the SAME
gpt-oss model (via OpenRouter) so chunking-arm results stay comparable.

A small disk cache makes runs cheap and reproducible: identical (model,
messages, max_tokens, temperature) requests are served from a JSON file rather
than re-hitting the network. Network calls are wrapped with tenacity
exponential back-off, which covers transient errors such as HTTP 429.
"""

import hashlib
import json
import logging
import os

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from raptor.SummarizationModels import BaseSummarizationModel
from raptor.QAModels import BaseQAModel

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Disk cache (self-contained: no imports from raptor.chunking)
# ---------------------------------------------------------------------------
class DiskCache:
    """A trivial one-JSON-file-per-key disk cache."""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def get(self, key: str):
        path = self._path(key)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def set(self, key: str, value) -> None:
        # Ensure dir still exists (in case it was removed mid-run).
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self._path(key), "w", encoding="utf-8") as fh:
            json.dump(value, fh)


def make_cache_key(model: str, messages: list, **extra) -> str:
    """Deterministic sha256 hex key over (model, messages, extra)."""
    canonical = json.dumps(
        {"model": model, "messages": messages, "extra": extra},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _make_openai_client(base_url: str, api_key_env: str):
    """Create a real OpenAI client. Imported lazily so tests stay offline."""
    from openai import OpenAI

    return OpenAI(api_key=os.environ[api_key_env], base_url=base_url)


# ---------------------------------------------------------------------------
# Moderation / retry classification
# ---------------------------------------------------------------------------
class ModerationBlocked(Exception):
    """Raised when the provider rejects input via content moderation (HTTP 403).

    Permanent: must NOT be retried. Carries the human-readable reason(s).
    """

    def __init__(self, message, reason=None):
        super().__init__(message)
        self.reason = reason


def _extract_moderation_reason(exc):
    """Best-effort human-readable moderation reason. Defensive: returns None."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        metadata = body.get("metadata")
        if isinstance(metadata, dict):
            reasons = metadata.get("reasons")
            if isinstance(reasons, (list, tuple)) and reasons:
                return ", ".join(str(r) for r in reasons)
            if isinstance(reasons, str) and reasons:
                return reasons
    # Fall back to a quoted reason inside the message, e.g. flagged for "x".
    text = str(exc)
    if '"' in text:
        parts = text.split('"')
        if len(parts) >= 3 and parts[1].strip():
            return parts[1].strip()
    return None


def _as_moderation_blocked(exc):
    """Return a ModerationBlocked if ``exc`` is a moderation 403, else None."""
    if getattr(exc, "status_code", None) == 403:
        lowered = str(exc).lower()
        if "moderat" in lowered or "flagged" in lowered:
            return ModerationBlocked(str(exc), reason=_extract_moderation_reason(exc))
    return None


def _should_retry(exc):
    """tenacity predicate: True => retry. False => re-raise immediately."""
    if isinstance(exc, ModerationBlocked):
        return False
    status = getattr(exc, "status_code", None)
    if status is None:
        return True  # generic transient error -> keep retrying
    if status == 429:
        return True
    if 400 <= status < 500:
        return False  # permanent client errors
    return True  # 5xx and anything else


def _extractive_summary_fallback(context, max_tokens):
    """Cheap extractive summary: leading slice of the context (~4 chars/token)."""
    return context.strip()[: max_tokens * 4]


def _coerce_content(response, model, max_tokens) -> str:
    """Return the completion text as a GUARANTEED string.

    gpt-oss is a reasoning model: its OpenAI-compatible response can carry
    ``message.content = None`` (e.g. the token budget was spent on reasoning and
    no visible content was emitted). If that None propagates, the RAPTOR tree
    builder calls ``tiktoken.encode(None)`` -> ``TypeError: expected string or
    buffer`` and the whole tree build dies. Coerce to a string here (preferring a
    reasoning field if the provider exposes one), and warn on an empty completion
    so it is visible in the run output.
    """
    message = response.choices[0].message
    content = getattr(message, "content", None)
    if content:
        return content
    for attr in ("reasoning_content", "reasoning"):
        alt = getattr(message, attr, None)
        if alt:
            return alt
    _log.warning(
        "empty completion (content=None) from model=%s (max_tokens=%s); returning "
        "empty string. If frequent, raise max_tokens or lower the reasoning effort.",
        model,
        max_tokens,
    )
    return ""


# ---------------------------------------------------------------------------
# Summarization model
# ---------------------------------------------------------------------------
class CachedOpenRouterSummarizationModel(BaseSummarizationModel):
    def __init__(
        self,
        model: str = "openai/gpt-oss-120b:free",
        base_url: str = "https://openrouter.ai/api/v1",
        api_key_env: str = "OPENROUTER_API_KEY",
        cache: DiskCache = None,
        cache_dir: str = ".llm_cache",
        temperature: float = 0.0,
        client=None,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.cache = cache if cache is not None else DiskCache(cache_dir)
        self.temperature = temperature
        self._client = client  # may be None -> created lazily on first use

    @property
    def client(self):
        if self._client is None:
            self._client = _make_openai_client(self.base_url, self.api_key_env)
        return self._client

    @retry(
        retry=retry_if_exception(_should_retry),
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def _create(self, messages, max_tokens):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.temperature,
            )
        except Exception as exc:
            blocked = _as_moderation_blocked(exc)
            if blocked is not None:
                raise blocked from exc
            raise
        return _coerce_content(response, self.model, max_tokens)

    def summarize(self, context, max_tokens=150, stop_sequence=None) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": (
                    "Write a summary of the following, including as many key "
                    f"details as possible: {context}:"
                ),
            },
        ]
        key = make_cache_key(
            self.model, messages, max_tokens=max_tokens, temperature=self.temperature
        )
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        try:
            content = self._create(messages, max_tokens)
        except ModerationBlocked as exc:
            _log.warning(
                "summarization moderation-blocked (%s); using extractive fallback",
                exc.reason or exc,
            )
            return _extractive_summary_fallback(context, max_tokens)
        if content:  # never cache an empty completion -> a re-run can retry it
            self.cache.set(key, content)
        return content


# ---------------------------------------------------------------------------
# QA model
# ---------------------------------------------------------------------------
class CachedOpenRouterQAModel(BaseQAModel):
    def __init__(
        self,
        model: str = "openai/gpt-oss-120b:free",
        base_url: str = "https://openrouter.ai/api/v1",
        api_key_env: str = "OPENROUTER_API_KEY",
        cache: DiskCache = None,
        cache_dir: str = ".llm_cache",
        temperature: float = 0.0,
        client=None,
        system_prompt: str = "You are Question Answering Portal",
        user_template: str = (
            "Given Context: {context} Give the best full answer amongst the "
            "option to question {question}"
        ),
    ):
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.cache = cache if cache is not None else DiskCache(cache_dir)
        self.temperature = temperature
        self._client = client
        self.system_prompt = system_prompt
        self.user_template = user_template

    @property
    def client(self):
        if self._client is None:
            self._client = _make_openai_client(self.base_url, self.api_key_env)
        return self._client

    @retry(
        retry=retry_if_exception(_should_retry),
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def _create(self, messages, max_tokens):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.temperature,
            )
        except Exception as exc:
            blocked = _as_moderation_blocked(exc)
            if blocked is not None:
                raise blocked from exc
            raise
        return _coerce_content(response, self.model, max_tokens)

    def answer_question(self, context, question, max_tokens=150, stop_sequence=None) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self.user_template.format(context=context, question=question),
            },
        ]
        key = make_cache_key(
            self.model, messages, max_tokens=max_tokens, temperature=self.temperature
        )
        cached = self.cache.get(key)
        if cached is not None:
            return cached.strip()

        content = self._create(messages, max_tokens)
        if content:  # never cache an empty completion -> a re-run can retry it
            self.cache.set(key, content)
        return content.strip()


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------
def build_models(cfg):
    """Construct (summarization_model, qa_model) sharing one DiskCache."""
    cache = DiskCache(cfg.cache_dir)
    summarization_model = CachedOpenRouterSummarizationModel(
        model=cfg.model,
        base_url=cfg.base_url,
        api_key_env=cfg.api_key_env,
        cache=cache,
        temperature=cfg.temperature,
    )
    qa_model = CachedOpenRouterQAModel(
        model=cfg.model,
        base_url=cfg.base_url,
        api_key_env=cfg.api_key_env,
        cache=cache,
        temperature=cfg.temperature,
    )
    return summarization_model, qa_model


def make_qa_model(cfg, system_prompt=None, user_template=None):
    """Build a QA model from a config, optionally with dataset-specific prompts."""
    kwargs = dict(
        model=cfg.model,
        base_url=cfg.base_url,
        api_key_env=cfg.api_key_env,
        cache=DiskCache(cfg.cache_dir),
        temperature=cfg.temperature,
    )
    if system_prompt is not None:
        kwargs["system_prompt"] = system_prompt
    if user_template is not None:
        kwargs["user_template"] = user_template
    return CachedOpenRouterQAModel(**kwargs)
