"""Thin Ollama client with retry, JSON-array parsing and model unloading."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import requests

OLLAMA_URL = "http://localhost:11434"
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


@dataclass
class LLMStats:
    requests: int = 0
    retries: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    failures: int = 0
    errors: list[str] = field(default_factory=list)


class OllamaClient:
    """Stateless client: every request carries only system+user messages, so the
    model never sees accumulated conversation history (no context growth, no
    drift/hallucination from stale context). Each call is a fresh inference."""

    def __init__(self, model: str, temperature: float = 0.3,
                 num_ctx: int = 4096, num_predict: int = 2048,
                 think: bool = False, timeout: int = 900,
                 url: str = OLLAMA_URL):
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.think = think
        self.timeout = timeout
        self.url = url
        self.stats = LLMStats()

    def chat(self, system: str, user: str, num_predict: int | None = None,
             retries: int = 4) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": self.think,
            "keep_alive": -1,  # keep loaded while the run continues; we unload explicitly
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": num_predict or self.num_predict,
                "repeat_penalty": 1.05,
            },
        }
        last_err = ""
        for attempt in range(retries):
            try:
                t0 = time.time()
                r = requests.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                elapsed = time.time() - t0
                self.stats.requests += 1
                self.stats.seconds += elapsed
                self.stats.prompt_tokens += data.get("prompt_eval_count", 0)
                self.stats.output_tokens += data.get("eval_count", 0)
                return data["message"]["content"]
            except Exception as e:  # noqa: BLE001 - network/server hiccups are all retryable
                last_err = str(e)
                self.stats.retries += 1
                time.sleep(min(2 ** attempt * 2, 20))
        self.stats.failures += 1
        self.stats.errors.append(last_err)
        raise RuntimeError(f"Ollama request failed after {retries} attempts: {last_err}")

    def unload(self) -> bool:
        """Eject the model from VRAM/RAM (clears weights + KV cache)."""
        try:
            r = requests.post(f"{self.url}/api/generate",
                              json={"model": self.model, "keep_alive": 0, "prompt": ""},
                              timeout=60)
            return r.ok
        except Exception:  # noqa: BLE001
            return False


def parse_array(text: str, expected: int) -> list[str] | None:
    """Extract a JSON array of `expected` strings from a model response."""
    if not text:
        return None
    cleaned = FENCE_RE.sub(r"\1", text).strip()
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end <= start:
        return None
    candidate = cleaned[start:end + 1]
    for cand in (candidate, candidate.replace("\n", " ").replace("'", '"')):
        try:
            arr = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(arr, list) and len(arr) == expected and all(isinstance(x, str) for x in arr):
            return arr
    return None


def translate_batch(client: OllamaClient, system: str, texts: list[str],
                    prev: str = "", full_text: str = "") -> list[str] | None:
    """Translate segments of ONE paragraph unit.

    ``texts`` are the segments (one per XML text node, alignment preserved),
    ``full_text`` is the complete paragraph for context. Returns aligned output
    or None on parse failure.
    """
    parts = []
    if prev:
        parts.append("Previous paragraph translation (continuity only, do not repeat it):\n"
                     + prev)
    if full_text:
        parts.append(f"The {len(texts)} strings below are consecutive segments of ONE "
                     f"paragraph unit. Full paragraph for context:\n{full_text}")
    parts.append(f"Translate each of the {len(texts)} segment(s) into Nepali. "
                 f"Return a JSON array of exactly {len(texts)} string(s), same order:")
    parts.append(json.dumps(texts, ensure_ascii=False))
    user = "\n\n".join(parts)

    # size output budget to the source text (fresh context every call)
    est = max(768, min(4096, int(len(full_text or texts[0]) * 0.6) + 250))

    for _ in range(3):  # whole-batch retries with stricter instructions
        raw = client.chat(system, user, num_predict=est)
        out = parse_array(raw, len(texts))
        if out is not None:
            return out
        user = (f"Your reply was not a valid JSON array of exactly {len(texts)} string(s). "
                f"Reply with ONLY the JSON array, no prose, no markdown.\n\n" + user)
    return None
