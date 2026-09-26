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
