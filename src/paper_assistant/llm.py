"""Thin wrapper around the Anthropic API: forced tool calls (guaranteed JSON shape) + disk cache.

Why a forced tool call? Asking a model to "reply in JSON" sometimes yields prose around the
JSON. Declaring a tool with an input schema and forcing the model to call it makes the API
return arguments that match the schema, which is far more reliable.

Why a cache? Evals re-run the same prompts many times while you tweak one thing. Caching by
a hash of (model, prompt, tool) makes reruns free and, importantly, repeatable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from paper_assistant import config


class DiskCache:
    def __init__(self, directory: Path | None = None):
        self.directory = Path(directory or config.CACHE_DIR / "llm")
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.directory / f"{hashlib.sha256(key.encode()).hexdigest()}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def set(self, key: str, value: Any) -> None:
        self._path(key).write_text(json.dumps(value), encoding="utf-8")


def get_client():
    """Create an Anthropic client. Reads ANTHROPIC_API_KEY from the environment or a .env file."""
    from anthropic import Anthropic
    from dotenv import load_dotenv

    load_dotenv()
    return Anthropic()


def call_tool(
    client,
    *,
    model: str,
    system: str,
    user: str,
    tool: dict,
    max_tokens: int = 1024,
    cache: DiskCache | None = None,
) -> dict:
    """Send one prompt and return the arguments of the forced tool call as a dict."""
    key = json.dumps([model, system, user, tool], sort_keys=True)
    if cache is not None and (hit := cache.get(key)) is not None:
        return hit

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        tools=[tool],
        tool_choice={"type": "tool", "name": tool["name"]},
    )
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == tool["name"]:
            payload = dict(block.input)
            if cache is not None:
                cache.set(key, payload)
            return payload
    raise RuntimeError("Model did not return the expected tool call")
