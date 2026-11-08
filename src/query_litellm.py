import base64
import os
from pathlib import Path

from dotenv import load_dotenv
import litellm

# Load .env from project root (src/ is one level below the repo root)
load_dotenv(Path(__file__).parent.parent / ".env")

# Get base URL from env (for LiteLLM proxy)
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY") or os.getenv("GEMINI_API_KEY")


def query_text(
    prompt: str,
    model: str = "gemini/gemini-2.5-flash",
    system_prompt: str | None = None,
    response_schema: dict | None = None,
) -> str:
    """Query LiteLLM with a text prompt, optional system prompt, and optional JSON schema."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs = {"model": model, "messages": messages}
    if LITELLM_BASE_URL:
        # Proxy path (unchanged behavior when a proxy URL is configured)
        kwargs["api_base"] = LITELLM_BASE_URL
        kwargs["api_key"] = LITELLM_API_KEY
        kwargs["custom_llm_provider"] = "openai"
    # else: litellm routes `gemini/*` to Google natively using GEMINI_API_KEY from the env

    if response_schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "strict": True,
                "schema": response_schema,
            },
        }

    response = litellm.completion(**kwargs)
    return response.choices[0].message.content


def query_image(
    prompt: str,
    image_path: str,
    model: str = "gemini/gemini-2.5-flash",
    system_prompt: str | None = None,
    response_schema: dict | None = None,
) -> str:
    """Query LiteLLM with text, an image file, optional system prompt, and optional JSON schema."""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
        ],
    })

    kwargs = {"model": model, "messages": messages}
    if LITELLM_BASE_URL:
        # Proxy path (unchanged behavior when a proxy URL is configured)
        kwargs["api_base"] = LITELLM_BASE_URL
        kwargs["api_key"] = LITELLM_API_KEY
        kwargs["custom_llm_provider"] = "openai"
    # else: litellm routes `gemini/*` to Google natively using GEMINI_API_KEY from the env

    if response_schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "strict": True,
                "schema": response_schema,
            },
        }

    response = litellm.completion(**kwargs)
    return response.choices[0].message.content


def query_image_pair(
    prompt: str,
    image_path_1: str,
    image_path_2: str,
    model: str = "gemini/gemini-2.5-flash",
    system_prompt: str | None = None,
    response_schema: dict | None = None,
) -> str:
    """Query LiteLLM with text and two image files, optional system prompt, and optional JSON schema."""
    with open(image_path_1, "rb") as f:
        image_data_1 = base64.b64encode(f.read()).decode("utf-8")
    with open(image_path_2, "rb") as f:
        image_data_2 = base64.b64encode(f.read()).decode("utf-8")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data_1}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data_2}"}},
        ],
    })

    kwargs = {"model": model, "messages": messages}
    if LITELLM_BASE_URL:
        # Proxy path (unchanged behavior when a proxy URL is configured)
        kwargs["api_base"] = LITELLM_BASE_URL
        kwargs["api_key"] = LITELLM_API_KEY
        kwargs["custom_llm_provider"] = "openai"
    # else: litellm routes `gemini/*` to Google natively using GEMINI_API_KEY from the env

    if response_schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "strict": True,
                "schema": response_schema,
            },
        }

    response = litellm.completion(**kwargs)
    return response.choices[0].message.content


# --- Target planner queries ---------------------------------------------------------------------
#
# The target model is named by a LiteLLM model string whose prefix selects the provider; credentials
# and endpoints come from the environment (or the repo-root .env) under LiteLLM's variable names:
#
#   gemini/<model>         Google AI Studio      GEMINI_API_KEY
#   anthropic/<model>      Anthropic             ANTHROPIC_API_KEY
#   openai/<model>         OpenAI                OPENAI_API_KEY
#   hosted_vllm/<model>    a `vllm serve` server HOSTED_VLLM_API_BASE (and HOSTED_VLLM_API_KEY if set)
#   litellm_proxy/<model>  a LiteLLM gateway     LITELLM_PROXY_API_BASE, LITELLM_PROXY_API_KEY
#
# Each provider receives the subset of the decoding recipe its API accepts; `target_params` is the
# single place that mapping lives, and it is printed once per run so the log records what was sent.

TARGET_PROVIDERS = ("gemini", "anthropic", "openai", "hosted_vllm", "litellm_proxy")


def target_params(
    model: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 1,
    seed: int = 42,
    max_tokens: int = 4096,
    thinking_budget: int = 2048,
    reasoning_effort: str = "medium",
) -> dict:
    """The request parameters `complete` sends for `model`, by provider.

    gemini         temperature, top_p, top_k, seed, max_tokens, and a thinking budget with the
                   thoughts returned (passed as Gemini's own `thinkingConfig`).
    anthropic      max_tokens and extended thinking with the same budget; the Anthropic API does not
                   accept temperature, top_p or seed alongside thinking.
    openai         max_tokens, seed and `reasoning_effort`; reasoning models take no temperature or
                   top_p, and the API returns no reasoning text.
    hosted_vllm    temperature, top_p, top_k, seed, max_tokens; reasoning is configured on the server.
    litellm_proxy  the full recipe, mapped by the gateway for whichever provider serves the model.
    """
    _, provider, _, _ = litellm.get_llm_provider(model)
    if provider not in TARGET_PROVIDERS:
        raise ValueError(
            f"{model!r}: the model string must start with one of "
            f"{', '.join(p + '/' for p in TARGET_PROVIDERS)}"
        )
    if provider == "gemini":
        return dict(temperature=temperature, top_p=top_p, top_k=top_k, seed=seed,
                    max_tokens=max_tokens,
                    thinkingConfig={"includeThoughts": True, "thinkingBudget": thinking_budget})
    if provider == "anthropic":
        return dict(max_tokens=max_tokens,
                    thinking={"type": "enabled", "budget_tokens": thinking_budget})
    if provider == "openai":
        return dict(max_tokens=max_tokens, seed=seed, reasoning_effort=reasoning_effort)
    if provider == "hosted_vllm":
        return dict(temperature=temperature, top_p=top_p, seed=seed, max_tokens=max_tokens,
                    extra_body={"top_k": top_k})
    return dict(temperature=temperature, top_p=top_p, top_k=top_k, seed=seed, max_tokens=max_tokens,
                thinking={"type": "enabled", "budget_tokens": thinking_budget})


def complete(model: str, system_prompt: str, user_text: str, image_path: str | None = None,
             **kwargs) -> dict:
    """One target query: the system prompt, then a user turn of the image (if any) and the text.

    Decoding comes from `target_params(model)`. Extra keyword arguments are passed to
    `litellm.completion`. Returns {"text": ..., "thinking": ...}, with the provider's reasoning
    output in "thinking" when it returns one and "" otherwise.
    """
    content = []
    if image_path is not None:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"}})
    content.append({"type": "text", "text": user_text})
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": content}]

    response = litellm.completion(model=model, messages=messages, **target_params(model), **kwargs)
    message = response.choices[0].message
    thinking = getattr(message, "reasoning_content", None)
    if not thinking:
        thinking = (getattr(message, "provider_specific_fields", None) or {}).get("reasoning_content")
    return {"text": message.content or "", "thinking": thinking or ""}
