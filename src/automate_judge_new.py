import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from argparse import ArgumentParser
from dotenv import load_dotenv
from filelock import FileLock
from google import genai
from google.genai import types
from tqdm import tqdm

from system_prompts_2 import llm_judge_system_prompt, llm_judge_user_prompt

# Load GENAI_API_KEY (or GEMINI_API_KEY) from .devcontainer/.env (also injected as
# real env vars via the devcontainer's --env-file at container start) or a repo-root
# .env. override=False so an already-set environment variable always wins. Loading
# the file directly means a freshly-added key works without rebuilding the container.
_REPO_ROOT = Path(__file__).parent.parent
load_dotenv(_REPO_ROOT / ".devcontainer" / ".env")
load_dotenv(_REPO_ROOT / ".env")

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds: sleeps 1s, 2s between attempts

_CLIENT = None


def _get_client():
    """Lazily create a shared google.genai client (safe to reuse across threads)."""
    global _CLIENT
    if _CLIENT is None:
        api_key = os.getenv("GENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No Gemini API key found. Set GENAI_API_KEY (or GEMINI_API_KEY) in a "
                ".env file at the repo root or in the environment."
            )
        _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT


def query_text(
    prompt: str,
    model: str,
    system_prompt: str | None = None,
    response_schema: dict | None = None,
    thinking_budget: int = 1024,
    max_output_tokens: int = 2048,
) -> str:
    """Query the Gemini API directly (google.genai) and return the raw text response.

    Deterministic judging: temperature 0, fixed seed. JSON is requested via
    response_schema so the caller gets a parseable object.
    """
    config_kwargs = {
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "max_output_tokens": max_output_tokens,
    }
    if system_prompt:
        config_kwargs["system_instruction"] = system_prompt
    if response_schema:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = response_schema
    if thinking_budget is not None:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)

    response = _get_client().models.generate_content(
        model=model,
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
        config=types.GenerateContentConfig(**config_kwargs),
    )
    if response.text is None:
        raise ValueError("Gemini API returned no text")
    return response.text


def parse_json_response(response_text: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks and key=value format."""
    import re

    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    # Try JSON first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON object from text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    # Try key=value format: "isAccepted = 0" or "isAccepted = 1"
    is_accepted_match = re.search(r'isAccepted\s*=\s*(-?\d+)', text)
    if is_accepted_match:
        is_accepted = int(is_accepted_match.group(1))
        # Extract reason - look for "reason = ..." pattern
        reason_match = re.search(r'reason\s*=\s*(.+)', text, re.DOTALL)
        reason = reason_match.group(1).strip() if reason_match else ""
        return {"isAccepted": is_accepted, "reason": reason}

    return {"isAccepted": -1, "reason": f"Failed to parse: {text[:200]}"}


# google.genai's response_schema follows a restricted OpenAPI subset: integer
# `enum` and `additionalProperties` aren't supported, so the -1/0/1 constraint is
# carried by the judge system prompt instead. Mirrors llm_2.query_gemini's schema,
# which is known to work against the Gemini API.
JUDGEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "isAccepted": {"type": "integer"},
        "reason": {"type": "string"},
    },
    "required": ["isAccepted", "reason"],
}


def _judge_single_task(args) -> tuple[int, dict]:
    """Judge a single task - helper for parallel execution."""
    idx, task, model, system_prompt = args
    user_prompt = llm_judge_user_prompt(
        model_response=task["Response"],
        seed_task=task["Instruction"],
    )
    # Retry transient API failures (rate limits, timeouts) with backoff so a
    # blip doesn't silently poison a row's label with the -1 error sentinel.
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            response_text = query_text(
                prompt=user_prompt,
                model=model,
                system_prompt=system_prompt,
                response_schema=JUDGEMENT_SCHEMA,
            )
            return idx, parse_json_response(response_text)
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE ** attempt)
    print(f"[{idx}] Error after {MAX_RETRIES} attempts: {last_err}", flush=True)
    return idx, {"isAccepted": -1, "reason": f"Error after retries: {last_err}"}


def automated_judge(
    input_csv_path: str,
    model: str,
    rang: tuple | None = None,
    max_workers: int = 10,
):
    """Run automated judging on model responses using LiteLLM."""
    print(f"Running automated judge with model: {model}, workers: {max_workers}", flush=True)
    df = pd.read_csv(input_csv_path)

    if rang is not None:
        start, end = rang
        df = df.iloc[start:end]

    system_prompt, _ = llm_judge_system_prompt(llm_model=model)

    tasks = df.to_dict("records")
    work_items = [(idx, task, model, system_prompt) for idx, task in enumerate(tasks)]

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_judge_single_task, item): item[0] for item in work_items}
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Judging",
            file=sys.stdout,
            mininterval=30,
            maxinterval=60,
        ):
            idx, judgment = future.result()
            results[idx] = judgment

    # Reconstruct judgments in original order
    judgments = [results[i] for i in range(len(tasks))]

    assert len(judgments) == len(df)
    write_judgements_to_csv(input_csv_path, judgments, rang)
    print(f"Done. Processed {len(judgments)} rows.", flush=True)


def write_judgements_to_csv(output_path: str, judgments: list, rang: tuple | None = None):
    """
    Write judgement outputs to CSV with file locking.
    Acquires lock, reads latest CSV, updates Judgement columns, and writes back.
    """
    lock_path = output_path + ".lock"

    with FileLock(lock_path):
        df = pd.read_csv(output_path)

        # Ensure columns exist and are string type to avoid dtype conflicts
        if "Judgement" not in df.columns:
            df["Judgement"] = ""
        else:
            df["Judgement"] = df["Judgement"].astype(str)
        if "Judgement Reason" not in df.columns:
            df["Judgement Reason"] = ""
        else:
            df["Judgement Reason"] = df["Judgement Reason"].astype(str)

        if rang is not None:
            start, end = rang
            index_range = df.index[start:end]

            assert len(judgments) == len(index_range), (
                "Number of judgments does not match the number of rows in the specified range."
            )
            for i, judgement in enumerate(judgments):
                if i < len(index_range):
                    df.loc[index_range[i], "Judgement"] = str(judgement.get("isAccepted", "-1"))
                    df.loc[index_range[i], "Judgement Reason"] = str(judgement.get("reason", ""))
        else:
            assert len(judgments) == len(df), (
                "Number of judgments does not match the number of rows in the CSV."
            )
            df["Judgement"] = [str(j.get("isAccepted", "-1")) for j in judgments]
            df["Judgement Reason"] = [str(j.get("reason", "")) for j in judgments]

        df.to_csv(output_path, index=False)
        print(f"Judgements written to {output_path}", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description="Run automated judging on model responses via LiteLLM")
    parser.add_argument(
        "--input_csv_path",
        type=str,
        required=True,
        help="Path to input CSV file with model responses to judge",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model name for the google.genai API (e.g., gemini-2.5-flash, gemini-2.5-pro)",
    )
    parser.add_argument(
        "--rang",
        type=str,
        default=None,
        help="Range of rows to process, format: start,end",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10)",
    )
    args = parser.parse_args()

    rang = args.rang
    rang = tuple(map(int, rang.split(","))) if rang is not None else None

    automated_judge(
        input_csv_path=args.input_csv_path,
        model=args.model,
        rang=rang,
        max_workers=args.workers,
    )
