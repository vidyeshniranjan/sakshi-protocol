"""
models.py — Sakshi-Protocol V3
Multi-provider model abstraction layer.

Supports four providers:
    OpenAI   — GPT-5.5 (and other OpenAI models)
    Anthropic — Claude Sonnet 4.6 (and other Anthropic models)
    Together  — Llama 3.1 8B, Qwen 3 (open models via Together AI)

All providers implement the same ModelClient interface:
    generate(prompt)              → str
    get_log_probs(prompt, output) → float | None

The pipeline and benchmark runner call only ModelClient methods —
they have no knowledge of which provider is active.

Log probability access:
    OpenAI   — available via logprobs=True on chat completions
    Anthropic — not exposed on the API; returns None
    Together  — available on most open models via logprobs=True

Log probabilities are used for the sequence log-probability baseline
in the MVE comparison (one of the five trigger signal conditions).
When unavailable (Anthropic), the log-prob condition is skipped for
that model and noted in results.

Usage:
    from sakshi.models import get_model_client

    client = get_model_client("gpt-5.5")
    output = client.generate("What is the capital of France?")
    lp     = client.get_log_probs("What is the capital of France?", output)

Model identifiers:
    "gpt-5.5"           → OpenAI GPT-5.5
    "gpt-4o"            → OpenAI GPT-4o (fallback)
    "claude-sonnet-4-6" → Anthropic Claude Sonnet 4.6
    "llama-3.1-8b"      → Together AI Llama 3.1 8B Instruct
    "qwen-3-14b"        → Together AI Qwen 3 14B
    "qwen-3-32b"        → Together AI Qwen 3 32B

Environment variables required:
    OPENAI_API_KEY
    ANTHROPIC_API_KEY
    TOGETHER_API_KEY
"""

import os
import math
from abc import ABC, abstractmethod
from typing import Optional


# =============================================================================
# BASE INTERFACE
# =============================================================================

class ModelClient(ABC):
    """
    Abstract base class for all model providers.
    All pipeline components interact only with this interface.
    """

    def __init__(self, model_id: str, temperature: float = 0.2, max_tokens: int = 500):
        self.model_id    = model_id
        self.temperature = temperature
        self.max_tokens  = max_tokens

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response to the prompt.
        Returns the response string.
        Returns "ERROR" on any failure.
        """
        pass

    @abstractmethod
    def get_log_probs(self, prompt: str, output: str) -> Optional[float]:
        """
        Get the mean token log-probability for the output given the prompt.
        Returns a float in (-inf, 0] where higher (less negative) = more confident.
        Returns None when log probabilities are not available for this provider.

        Used for the sequence log-probability baseline in the MVE comparison.
        """
        pass

    @property
    def supports_log_probs(self) -> bool:
        """Whether this provider/model exposes log probabilities."""
        return False

    def __repr__(self):
        return f"{self.__class__.__name__}(model_id={self.model_id!r})"


# =============================================================================
# OPENAI PROVIDER
# Supports: GPT-5.5, GPT-4o, and other OpenAI chat models
# Log probs: available via logprobs=True on chat completions
# =============================================================================

class OpenAIClient(ModelClient):
    """
    OpenAI API client.
    Requires OPENAI_API_KEY environment variable.
    """

    # Map short model identifiers to full OpenAI model strings
    MODEL_MAP = {
        "gpt-5.5":    "gpt-5.5",
        "gpt-4o":     "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
    }

    def __init__(self, model_id: str = "gpt-5.5", temperature: float = 0.2, max_tokens: int = 500):
        super().__init__(model_id, temperature, max_tokens)
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        self._model_str = self.MODEL_MAP.get(model_id, model_id)

    @property
    def supports_log_probs(self) -> bool:
        return True

    # Models that only support temperature=1 (reasoning models)
    _FIXED_TEMP_MODELS = {"gpt-5.5", "o1", "o1-mini", "o3", "o3-mini", "o4-mini"}

    def generate(self, prompt: str) -> str:
        import time as _time

        kwargs = {
            "model":                self._model_str,
            "messages":            [{"role": "user", "content": prompt}],
            "max_completion_tokens": self.max_tokens,
        }
        if self._model_str not in self._FIXED_TEMP_MODELS:
            kwargs["temperature"] = self.temperature

        # Exponential backoff — handles rate limits (429) and transient errors
        max_retries = 5
        backoff     = 15  # seconds, doubles each retry (5→15→30→60→120)
        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or "ERROR"
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = "429" in err_str or "rate limit" in err_str or "too many" in err_str
                is_transient  = "500" in err_str or "503" in err_str or "timeout" in err_str or "connection" in err_str
                if (is_rate_limit or is_transient) and attempt < max_retries - 1:
                    wait = backoff * (2 ** attempt)
                    print(f"[OpenAI] {'rate limit' if is_rate_limit else 'transient error'} — retry {attempt+1}/{max_retries-1} in {wait}s")
                    _time.sleep(wait)
                else:
                    print(f"[OpenAI] generate error (attempt {attempt+1}): {e}")
                    return "ERROR"
        return "ERROR"

    def get_log_probs(self, prompt: str, output: str) -> Optional[float]:
        """
        Returns mean token log-probability for the output.
        Uses a second API call with logprobs=True.
        Note: reasoning models (GPT-5.5 etc) may not support logprobs.
        """
        try:
            kwargs = {
                "model":                self._model_str,
                "messages":            [{"role": "user", "content": prompt}],
                "max_completion_tokens": self.max_tokens,
                "logprobs":            True,
                "top_logprobs":        1,
            }
            if self._model_str not in self._FIXED_TEMP_MODELS:
                kwargs["temperature"] = 0.0
            response = self._client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            if choice.logprobs and choice.logprobs.content:
                lp_values = [t.logprob for t in choice.logprobs.content
                             if t.logprob is not None]
                if lp_values:
                    return float(sum(lp_values) / len(lp_values))
            return None
        except Exception as e:
            print(f"[OpenAI] log_probs error: {e}")
            return None


# =============================================================================
# ANTHROPIC PROVIDER
# Supports: Claude Sonnet 4.6, Claude Opus 4.6
# Log probs: NOT available on Anthropic API
# =============================================================================

class AnthropicClient(ModelClient):
    """
    Anthropic API client.
    Requires ANTHROPIC_API_KEY environment variable.
    Log probabilities are not exposed by the Anthropic API.
    The log-prob baseline condition is skipped for this provider.
    """

    MODEL_MAP = {
        "claude-sonnet-4-6": "claude-sonnet-4-6",
        "claude-opus-4-6":   "claude-opus-4-6",
        "claude-sonnet":     "claude-sonnet-4-6",  # shorthand
        "claude-opus":       "claude-opus-4-6",    # shorthand
    }

    def __init__(self, model_id: str = "claude-sonnet-4-6", temperature: float = 0.2, max_tokens: int = 500):
        super().__init__(model_id, temperature, max_tokens)
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

        self._model_str = self.MODEL_MAP.get(model_id, model_id)

    @property
    def supports_log_probs(self) -> bool:
        # Anthropic API does not expose token log probabilities
        return False

    def generate(self, prompt: str) -> str:
        try:
            message = self._client.messages.create(
                model=self._model_str,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            if not message.content:
                # Empty content — Claude declined internally (safety/policy)
                # Return empty string so pipeline can evaluate the non-response
                return ""
            return message.content[0].text or "ERROR"
        except Exception as e:
            print(f"[Anthropic] generate error: {e}")
            return "ERROR"

    def get_log_probs(self, prompt: str, output: str) -> Optional[float]:
        # Not available on Anthropic API
        return None


# =============================================================================
# TOGETHER AI PROVIDER
# Supports: Llama 3.1 8B, Qwen 3 14B/32B, and other Together-hosted models
# Log probs: available on most models via logprobs parameter
# Uses OpenAI-compatible API endpoint
# =============================================================================

class TogetherClient(ModelClient):
    """
    Together AI client for open models.
    Requires TOGETHER_API_KEY environment variable.
    Uses Together's OpenAI-compatible endpoint.
    """

    # Together AI model strings for our target models
    MODEL_MAP = {
        # Serverless endpoints — confirmed via api.together.ai/models (May 2026)
        # Confirmed serverless — verified May 2026
        "llama-3.3-70b":  "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "llama-3.1-8b":   "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "qwen-3.5-9b":    "Qwen/Qwen3.5-9B",
        "qwen-2.5-7b":    "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "gemma-2-27b":    "google/gemma-2-27b-it",
        "deepseek-v3":    "deepseek-ai/DeepSeek-V3",
    }

    def __init__(self, model_id: str = "llama-3.1-8b", temperature: float = 0.2, max_tokens: int = 500):
        super().__init__(model_id, temperature, max_tokens)
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=os.environ.get("TOGETHER_API_KEY"),
                base_url="https://api.together.xyz/v1",
            )
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        self._model_str = self.MODEL_MAP.get(model_id, model_id)

    @property
    def supports_log_probs(self) -> bool:
        return True

    # Models that require thinking mode to be disabled explicitly
    # These are reasoning/hybrid models that default to thinking mode
    # and return empty content if thinking consumes all tokens
    _THINKING_MODELS = {"Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-9B-FP8"}

    def generate(self, prompt: str) -> str:
        import time as _time

        kwargs = {
            "model":       self._model_str,
            "messages":   [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens":  self.max_tokens,
        }
        if self._model_str in self._THINKING_MODELS:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }

        max_retries = 5
        backoff     = 10  # seconds, doubles each retry
        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or "ERROR"
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = "429" in err_str or "rate limit" in err_str or "too many" in err_str
                is_transient  = "500" in err_str or "503" in err_str or "timeout" in err_str or "connection" in err_str
                if (is_rate_limit or is_transient) and attempt < max_retries - 1:
                    wait = backoff * (2 ** attempt)
                    print(f"[Together] {'rate limit' if is_rate_limit else 'transient error'} — retry {attempt+1}/{max_retries-1} in {wait}s")
                    _time.sleep(wait)
                else:
                    print(f"[Together] generate error (attempt {attempt+1}): {e}")
                    return "ERROR"
        return "ERROR"

    def get_log_probs(self, prompt: str, output: str) -> Optional[float]:
        """
        Returns mean token log-probability.
        Together AI supports logprobs on most open models.
        Falls back to None on models that don't support it.
        """
        try:
            kwargs = {
                "model":      self._model_str,
                "messages":  [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens":  self.max_tokens,
                "logprobs":   True,
            }
            if self._model_str in self._THINKING_MODELS:
                kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            response = self._client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            if choice.logprobs and choice.logprobs.content:
                lp_values = [t.logprob for t in choice.logprobs.content
                             if t.logprob is not None]
                if lp_values:
                    return float(sum(lp_values) / len(lp_values))
            return None
        except Exception as e:
            print(f"[Together] log_probs error for {self._model_str}: {e}")
            return None


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

# Registry mapping model identifiers to (provider_class, model_id)
_MODEL_REGISTRY = {
    # OpenAI
    "gpt-5.5":           (OpenAIClient,    "gpt-5.5"),
    "gpt-4o":            (OpenAIClient,    "gpt-4o"),
    "gpt-4o-mini":       (OpenAIClient,    "gpt-4o-mini"),

    # Anthropic
    "claude-sonnet-4-6": (AnthropicClient, "claude-sonnet-4-6"),
    "claude-opus-4-6":   (AnthropicClient, "claude-opus-4-6"),
    "claude-sonnet":     (AnthropicClient, "claude-sonnet-4-6"),

    # Together AI — confirmed serverless (api.together.ai/models, May 2026)
    "llama-3.3-70b":  (TogetherClient, "llama-3.3-70b"),
    "llama-3.1-8b":   (TogetherClient, "llama-3.1-8b"),
    "qwen-3.5-9b":    (TogetherClient, "qwen-3.5-9b"),
    "qwen-2.5-7b":    (TogetherClient, "qwen-2.5-7b"),
    "gemma-2-27b":    (TogetherClient, "gemma-2-27b"),
    "deepseek-v3":    (TogetherClient, "deepseek-v3"),
}

# Primary evaluation models for V3
# These are the four models used in the main evaluation
EVALUATION_MODELS = [
    "gpt-5.5",
    "claude-sonnet-4-6",
    "llama-3.3-70b",
    "qwen-3.5-9b",
]

# Free tier model for connectivity testing — no cost
TEST_MODEL = "llama-3.3-70b-free"

# Legacy model retained for V1/V2 appendix comparison only
LEGACY_MODEL = "gpt-4o-mini"


def get_model_client(
    model_id: str,
    temperature: float = 0.2,
    max_tokens: int = 500,
) -> ModelClient:
    """
    Factory function. Returns a ModelClient for the given model identifier.

    Args:
        model_id    : one of the keys in _MODEL_REGISTRY
        temperature : generation temperature (default 0.2)
        max_tokens  : maximum output tokens (default 500)

    Returns:
        ModelClient instance for the requested model

    Raises:
        ValueError  : if model_id is not in the registry
        ImportError : if the required provider library is not installed
        ValueError  : if the required API key is not set
    """
    if model_id not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model_id: {model_id!r}\n"
            f"Available models: {available}"
        )

    provider_class, resolved_id = _MODEL_REGISTRY[model_id]

    # Check API key before instantiation
    key_map = {
        OpenAIClient:    "OPENAI_API_KEY",
        AnthropicClient: "ANTHROPIC_API_KEY",
        TogetherClient:  "TOGETHER_API_KEY",
    }
    key_name = key_map.get(provider_class)
    if key_name and not os.environ.get(key_name):
        raise ValueError(
            f"Environment variable {key_name} is not set. "
            f"Required for model {model_id!r}."
        )

    return provider_class(
        model_id=resolved_id,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def list_models() -> dict:
    """
    Returns a summary of available models grouped by provider.
    Useful for debugging and documentation.
    """
    grouped = {
        "openai":    [],
        "anthropic": [],
        "together":  [],
    }
    provider_key = {
        OpenAIClient:    "openai",
        AnthropicClient: "anthropic",
        TogetherClient:  "together",
    }
    for model_id, (cls, resolved) in _MODEL_REGISTRY.items():
        key = provider_key[cls]
        grouped[key].append({
            "id":              model_id,
            "model_string":    resolved,
            "log_probs":       cls != AnthropicClient,
            "evaluation_model": model_id in EVALUATION_MODELS,
        })
    return grouped
