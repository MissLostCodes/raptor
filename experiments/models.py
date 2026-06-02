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
import os

from tenacity import retry, stop_after_attempt, wait_random_exponential

from raptor.SummarizationModels import BaseSummarizationModel
from raptor.QAModels import BaseQAModel


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

    @retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(6))
    def _create(self, messages, max_tokens):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

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

        content = self._create(messages, max_tokens)
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

    @retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(6))
    def _create(self, messages, max_tokens):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

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
