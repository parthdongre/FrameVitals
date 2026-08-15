"""
DataLens AI — Agentic AI Layer (WS-9)
======================================
Planner → Executor → Critic → Writer loop running on a local Ollama model
with OpenRouter and a deterministic fallback.

Flow:
    1. PLANNER decides which 1-3 tools to call (JSON mode)
    2. EXECUTOR runs the tools and collects their outputs
    3. CRITIC checks for hallucinations vs. tool outputs (JSON mode)
    4. WRITER composes the final markdown answer

The whole pipeline degrades gracefully:
    - If JSON parsing fails, we recover the best plan we can
    - If Ollama is offline, we fall back to OpenRouter, then to a heuristic
      writer that uses retrieved RAG facts directly

Public API:
    answer_with_agent(question, df, analysis_result) -> dict
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import pandas as pd
import pydantic
from pydantic import BaseModel, Field, ValidationError

from framevitals.rag_index import (
    build_fact_index,
    render_facts_block,
)
from framevitals.rag_index import (
    retrieve as rag_retrieve,
)

from framevitals.agent_tools import (
    AgentContext,
    list_tools,
    run_tool,
)


_DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:4b")
_DEFAULT_OPENROUTER_MODEL = os.environ.get(
    "OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free"
)
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_OPENROUTER_SITE_URL = os.environ.get("OPENROUTER_SITE_URL", "http://127.0.0.1:5055")
_OPENROUTER_APP_NAME = os.environ.get("OPENROUTER_APP_NAME", "DataLens AI")

# Per-call Ollama timeout. Cloud models (deepseek 671b, gpt-oss 120b, kimi)
# can take 30-60s end-to-end. Override with DATALENS_OLLAMA_TIMEOUT (seconds).
_OLLAMA_TIMEOUT_SECONDS = float(os.environ.get("DATALENS_OLLAMA_TIMEOUT", "120"))

# When True, treat Ollama Cloud entries (remote_model set, name suffix "-cloud"
# or ":cloud") as valid candidates. They route through ollama.com's GPUs and
# require `ollama signin`. Set DATALENS_OLLAMA_ALLOW_CLOUD=1 if you want the
# big models like deepseek-v3.1:671b-cloud or gpt-oss:120b-cloud.
_OLLAMA_ALLOW_CLOUD = os.environ.get("DATALENS_OLLAMA_ALLOW_CLOUD", "0").strip() in {
    "1", "true", "yes",
}

# Performance tuning specific to data-analysis Q&A on a small local model
# (qwen3:4b). The defaults below are tight on purpose:
#
#   - num_ctx=2048: the agent only ever sees ~2 KB of retrieved facts plus a
#     small system prompt. Larger contexts cost prompt-eval time per token
#     and the model never uses them.
#   - num_predict (output cap): the writer produces a structured 4-section
#     answer; 320 tokens is enough. Limits worst-case generation time hard.
#   - temperature=0 + top_k=1: greedy decoding is deterministic, faster, and
#     *more* accurate for fact-quoting work.
#   - num_thread=auto: let llama.cpp pick a sensible thread count.
#   - repeat_penalty=1.05: keeps the model from repeating retrieved bullets.
#
# Override any of these via env if you have a roomier machine (e.g. a 4050).
_OLLAMA_NUM_CTX = int(os.environ.get("DATALENS_OLLAMA_NUM_CTX", "3072"))
_OLLAMA_NUM_PREDICT = int(os.environ.get("DATALENS_OLLAMA_NUM_PREDICT", "600"))
_OLLAMA_TEMPERATURE = float(os.environ.get("DATALENS_OLLAMA_TEMPERATURE", "0.2"))
_OLLAMA_TOP_K = int(os.environ.get("DATALENS_OLLAMA_TOP_K", "40"))
_OLLAMA_TOP_P = float(os.environ.get("DATALENS_OLLAMA_TOP_P", "0.9"))
_OLLAMA_REPEAT_PENALTY = float(os.environ.get("DATALENS_OLLAMA_REPEAT_PENALTY", "1.05"))
_OLLAMA_KEEP_ALIVE = os.environ.get("DATALENS_OLLAMA_KEEP_ALIVE", "10m")

# Qwen3 reasons in <think>...</think> blocks by default. For structured
# fact-quoting work we don't want chain-of-thought leaking into the answer
# (it doubles latency for no quality gain on this task), so we prepend a
# system message that tells Qwen to skip thinking mode. The flag itself is
# only injected for known qwen3 family models.
_QWEN3_DISABLE_THINKING = "/no_think"

# LLM_PREFER controls the routing policy in _call_llm:
#   "auto"        — try Ollama first, then OpenRouter, then heuristic (default)
#   "ollama"      — only try Ollama
#   "openrouter"  — only try OpenRouter
#   "off"         — skip both, go straight to heuristic
_LLM_PREFER = os.environ.get("LLM_PREFER", "auto").strip().lower()

# Preference list for auto-selecting an Ollama model when the configured one
# isn't pulled. Smallest-first so the demo stays snappy.
#
# When DATALENS_OLLAMA_ALLOW_CLOUD=1 the cloud entries are tried first —
# gpt-oss:120b-cloud is on the free Ollama Cloud tier and answers in ~1-2s,
# while the 671b deepseek and kimi require a paid subscription.
_OLLAMA_MODEL_PREFERENCE = [
    "gpt-oss:120b-cloud",         # cloud, free tier, ~1.5s
    "deepseek-v3.1:671b-cloud",   # cloud, paid
    "kimi-k2.5:cloud",            # cloud, paid
    "qwen3:4b",                   # local, ~10-20s on CPU
    "qwen3.5:4b",
    "llama3.2",
    "llama3.1:8b",
    "deepseek-r1:14b",
]

# Cached list of models actually installed on the local Ollama daemon. Filled
# lazily by `_resolve_ollama_model()`.
_AVAILABLE_OLLAMA_MODELS: list[str] | None = None

# Whether the warm-up has already run this process. We do it once on demand
# the first time anyone resolves an Ollama model, never on import (so unit
# tests stay fast and Flask doesn't pay the cost when the LLM isn't used).
_OLLAMA_WARMED = False


# ---------------------------------------------------------------------------
# Pydantic schemas for structured LLM output
# ---------------------------------------------------------------------------

class ToolCall(BaseModel):
    name: str
    args: dict = Field(default_factory=dict)


class Plan(BaseModel):
    reasoning: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)


class Verdict(BaseModel):
    accept: bool
    reason: str = ""
    repair: list[ToolCall] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Low-level LLM calls
# ---------------------------------------------------------------------------

def _ollama_reachable() -> bool:
    """Cheap socket probe — under 200ms when Ollama is up or down."""
    import socket
    try:
        with socket.create_connection(("127.0.0.1", 11434), timeout=0.3):
            return True
    except OSError:
        return False


def _list_ollama_models(force: bool = False) -> list[str]:
    """Return the names of models the local Ollama daemon knows about.

    By default we filter out:
      - embedding-only models (nomic-bert) — they can't chat
      - cloud shim entries (`*-cloud`, `*:cloud`, or remote_model set) —
        these proxy to ollama.com and need `ollama signin`

    Set DATALENS_OLLAMA_ALLOW_CLOUD=1 to keep cloud entries in the list.
    """
    global _AVAILABLE_OLLAMA_MODELS
    if _AVAILABLE_OLLAMA_MODELS is not None and not force:
        return _AVAILABLE_OLLAMA_MODELS

    if not _ollama_reachable():
        _AVAILABLE_OLLAMA_MODELS = []
        return _AVAILABLE_OLLAMA_MODELS

    import urllib.request
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        names: list[str] = []
        for entry in data.get("models", []) or []:
            name = entry.get("name") or ""
            if not name:
                continue
            family_list = (entry.get("details") or {}).get("families") or []
            if "nomic-bert" in family_list:
                continue
            is_cloud = (
                entry.get("remote_model")
                or "cloud" in name.lower()
            )
            if is_cloud and not _OLLAMA_ALLOW_CLOUD:
                continue
            names.append(name)
        _AVAILABLE_OLLAMA_MODELS = names
    except Exception:
        _AVAILABLE_OLLAMA_MODELS = []
    return _AVAILABLE_OLLAMA_MODELS


def _is_cloud_model(name: str) -> bool:
    return bool(name) and ("cloud" in name.lower())


def _resolve_ollama_model(requested: str | None) -> str | None:
    """Pick a model name that's actually installed on the local daemon.

    Priority:
      1. The caller's `requested` model, if it's installed (cloud entries
         are honored even when DATALENS_OLLAMA_ALLOW_CLOUD is off, because
         the user asked for one specifically).
      2. The env-configured `_DEFAULT_OLLAMA_MODEL`, if installed.
      3. The first match from `_OLLAMA_MODEL_PREFERENCE` that's installed.
      4. Whatever's installed first.
      5. None — caller should fall back to OpenRouter / heuristic.
    """
    available = _list_ollama_models()

    # If the caller named a cloud model directly, look it up in the raw
    # /api/tags response so we can route to it even when ALLOW_CLOUD is off.
    if requested and _is_cloud_model(requested):
        try:
            import urllib.request
            with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1.5) as resp:
                raw = json.loads(resp.read().decode("utf-8")).get("models", []) or []
            raw_names = [(e.get("name") or "") for e in raw]
            if requested in raw_names:
                return requested
            if f"{requested}:latest" in raw_names:
                return f"{requested}:latest"
        except Exception:
            pass

    if not available:
        return None

    def _match(name: str) -> str | None:
        # Exact match wins.
        if name in available:
            return name
        # Ollama tags often include `:latest` we didn't ask for.
        if f"{name}:latest" in available:
            return f"{name}:latest"
        # Or the user asked for `model:tag` and `model` is installed bare.
        if ":" in name and name.split(":", 1)[0] in available:
            return name.split(":", 1)[0]
        return None

    if requested:
        hit = _match(requested)
        if hit:
            return hit

    if _DEFAULT_OLLAMA_MODEL:
        hit = _match(_DEFAULT_OLLAMA_MODEL)
        if hit:
            return hit

    for name in _OLLAMA_MODEL_PREFERENCE:
        hit = _match(name)
        if hit:
            return hit

    return available[0]


def _maybe_warm_ollama(model: str) -> None:
    """One-shot model warm-up. Pings Ollama with an empty prompt so the
    weights are paged into RAM before the first real request lands. Harmless
    if Ollama is offline (we already checked) or if the model is already
    warm.
    """
    global _OLLAMA_WARMED
    if _OLLAMA_WARMED or not model:
        return
    _OLLAMA_WARMED = True  # mark first so a failure doesn't loop forever
    try:
        from ollama import Client
        client = Client(host="http://127.0.0.1:11434", timeout=2.0)
        client.generate(
            model=model,
            prompt="ok",
            options={"num_predict": 1, "temperature": 0.0, "num_ctx": 256},
            keep_alive=_OLLAMA_KEEP_ALIVE,
        )
    except Exception:
        pass


def _is_qwen3(model_name: str) -> bool:
    name = (model_name or "").lower()
    return name.startswith("qwen3") or "qwen3" in name


def _maybe_inject_no_think(messages: list[dict], model_name: str) -> list[dict]:
    """For Qwen3 models, prepend a tiny system message that disables the
    `<think>...</think>` chain-of-thought block. This roughly halves the
    output token count on this kind of structured Q&A workload because the
    model otherwise spends 200-600 tokens reasoning in private before it
    starts writing the answer.
    """
    if not _is_qwen3(model_name):
        return messages
    if any(
        msg.get("role") == "system" and _QWEN3_DISABLE_THINKING in (msg.get("content") or "")
        for msg in messages
    ):
        return messages
    head = {
        "role": "system",
        "content": (
            f"{_QWEN3_DISABLE_THINKING}\n"
            "Answer directly. Do not use <think> blocks. "
            "Quote retrieved facts verbatim where possible."
        ),
    }
    return [head, *messages]


def _build_ollama_options(json_mode: bool, model_name: str | None = None) -> dict:
    """Produce the per-call options dict.

    For cloud models (deepseek 671b, gpt-oss 120b, kimi) we keep the request
    minimal — they're huge models served on remote GPUs and our local-tuning
    knobs (small num_ctx, top_k=1) would hurt rather than help. For the local
    qwen3:4b we apply the tighter set.
    """
    if model_name and _is_cloud_model(model_name):
        opts: dict[str, Any] = {"temperature": 0.2 if not json_mode else 0.0}
        return opts

    opts = {
        "temperature": _OLLAMA_TEMPERATURE,
        "top_k": _OLLAMA_TOP_K,
        "top_p": _OLLAMA_TOP_P,
        "repeat_penalty": _OLLAMA_REPEAT_PENALTY,
        "num_ctx": _OLLAMA_NUM_CTX,
        "num_predict": _OLLAMA_NUM_PREDICT,
    }
    # JSON mode benefits from slightly more headroom (the planner emits a
    # short tool-call list, but we want it to never get truncated).
    if json_mode:
        opts["num_predict"] = max(_OLLAMA_NUM_PREDICT, 384)
    return opts


def _call_ollama_chat(messages: list[dict], json_mode: bool = False, model: str | None = None) -> str:
    if not _ollama_reachable():
        raise RuntimeError("Ollama not reachable on 127.0.0.1:11434")

    resolved = _resolve_ollama_model(model)
    if not resolved:
        raise RuntimeError(
            "Ollama is reachable but no compatible chat model is installed. "
            "Try `ollama pull qwen3:4b`."
        )

    import ollama  # local import so missing lib doesn't break module load
    from ollama import Client

    options = _build_ollama_options(json_mode, model_name=resolved)
    # NOTE: We used to prepend a system message with `/no_think` here, but
    # qwen3:4b is sensitive to certain system-message phrasings and can go
    # completely silent. Callers that need the directive should embed it as
    # the first line of their user message instead — that consistently works
    # across qwen3, llama3, and OpenRouter.
    payload_messages = messages

    # Build a per-call client with a tight timeout so a slow daemon can't
    # block the request handler.
    client = Client(host="http://127.0.0.1:11434", timeout=_OLLAMA_TIMEOUT_SECONDS)

    def _do_chat(c) -> str:
        return c.chat(
            model=resolved,
            messages=payload_messages,
            format="json" if json_mode else None,
            options=options,
            keep_alive=_OLLAMA_KEEP_ALIVE,
        )["message"]["content"].strip()

    try:
        text = _do_chat(client)
    except Exception as exc:
        # Fall back to module-level call if the Client interface ever changes.
        try:
            text = ollama.chat(
                model=resolved,
                messages=payload_messages,
                format="json" if json_mode else None,
                options=options,
                keep_alive=_OLLAMA_KEEP_ALIVE,
            )["message"]["content"].strip()
        except Exception:
            raise exc

    # Strip any leftover <think>...</think> blocks just in case the model
    # ignored the directive (older Qwen3 builds sometimes do).
    if "<think>" in text:
        import re
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


def _call_openrouter_chat(messages: list[dict], json_mode: bool = False, model: str | None = None) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    import urllib.request
    import urllib.error

    body: dict[str, Any] = {
        "model": model or _DEFAULT_OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 1400,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    req = urllib.request.Request(
        _OPENROUTER_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": _OPENROUTER_SITE_URL,
            "X-Title": _OPENROUTER_APP_NAME,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {exc.reason}") from exc


def _call_llm(messages: list[dict], json_mode: bool = False) -> tuple[str, str]:
    """Route the call according to LLM_PREFER. Returns (text, source).

    Policies:
      auto       — Ollama (if reachable) → OpenRouter (if API key) → raise
      ollama     — Ollama only
      openrouter — OpenRouter only
      off        — always raise (forces the heuristic writer)
    """
    if _LLM_PREFER == "off":
        raise RuntimeError("LLM_PREFER=off — both backends disabled by config.")

    if _LLM_PREFER == "openrouter":
        return _call_openrouter_chat(messages, json_mode=json_mode), "openrouter"

    if _LLM_PREFER == "ollama":
        return _call_ollama_chat(messages, json_mode=json_mode), "ollama"

    # auto: try Ollama first only if it's reachable; otherwise skip the
    # several-second timeout and go straight to OpenRouter.
    if _ollama_reachable():
        try:
            return _call_ollama_chat(messages, json_mode=json_mode), "ollama"
        except Exception as ollama_err:
            try:
                return _call_openrouter_chat(messages, json_mode=json_mode), "openrouter"
            except Exception as openrouter_err:
                raise RuntimeError(
                    f"Both LLM backends failed. ollama={ollama_err}; openrouter={openrouter_err}"
                )

    return _call_openrouter_chat(messages, json_mode=json_mode), "openrouter"


# ---------------------------------------------------------------------------
# Resilient JSON parsing
# ---------------------------------------------------------------------------

def _extract_json_object(text: str) -> dict | None:
    """Try hard to recover a JSON object even if the model wrapped it in prose."""
    text = (text or "").strip()
    if not text:
        return None

    # Direct parse first
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    # Strip markdown code fences
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1).strip())
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass

    # First brace-balanced object
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        parsed = json.loads(candidate)
                        return parsed if isinstance(parsed, dict) else None
                    except Exception:
                        break
        start = text.find("{", start + 1)

    return None


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def _planner_prompt(question: str, retrieved_block: str) -> list[dict]:
    tools_json = json.dumps(list_tools(), indent=2)
    system = (
        "You are the planning module of a data analysis agent. Decide which tools "
        "to call to gather evidence for the user's question. Respond with strict JSON only."
    )
    user = (
        f"Available tools:\n{tools_json}\n\n"
        f"Most-relevant facts retrieved from the analysis:\n{retrieved_block}\n\n"
        f"User question:\n{question}\n\n"
        "Return JSON with this exact shape:\n"
        "{\n"
        "  \"reasoning\": \"<short reasoning>\",\n"
        "  \"tool_calls\": [\n"
        "    {\"name\": \"<tool_name>\", \"args\": {<arg_name>: <value>}}\n"
        "  ]\n"
        "}\n"
        "Use 1-3 tool_calls. Prefer the most direct tool. Use search_facts only if "
        "no other tool fits."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _critic_prompt(question: str, plan_text: str, evidence_text: str) -> list[dict]:
    system = (
        "You are the critic module. Decide whether the gathered evidence is sufficient "
        "to answer the user's question without making up information. Respond JSON only."
    )
    user = (
        f"Question:\n{question}\n\n"
        f"Plan reasoning:\n{plan_text}\n\n"
        f"Evidence collected from tools:\n{evidence_text}\n\n"
        "Return JSON with this exact shape:\n"
        "{\n"
        "  \"accept\": true|false,\n"
        "  \"reason\": \"<short>\",\n"
        "  \"repair\": [{\"name\": \"<tool>\", \"args\": {...}}]\n"
        "}\n"
        "Set accept=true if the evidence is enough. Otherwise set accept=false and "
        "list 1-2 repair tool_calls. Do not invent tool names."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _writer_prompt(question: str, evidence_text: str) -> list[dict]:
    # NOTE: keep this prompt structurally simple. qwen3:4b becomes silent on
    # certain dual-system arrangements ("You are the writer module..." was a
    # repro). A single concise instruction baked into the user message works
    # consistently across qwen3 / llama3 / openrouter.
    #
    # The leading `/no_think` is a Qwen3 slash command that disables the
    # `<think>...</think>` chain-of-thought block. On non-Qwen models it is
    # ignored as a benign 9-character preamble.
    user = (
        "/no_think\n"
        "Compose a concise markdown answer to the question below. "
        "Quote concrete numbers from the evidence. If something cannot be "
        "answered from the evidence, say so plainly. Do not invent values.\n\n"
        f"Question:\n{question}\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        "Format your answer as:\n"
        "## Answer\n"
        "## Evidence\n"
        "## Recommendation\n"
        "## Limitation"
    )
    return [{"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Plan / verdict parsers with fallbacks
# ---------------------------------------------------------------------------

def _parse_plan(raw: str) -> Plan:
    obj = _extract_json_object(raw) or {}
    try:
        return Plan.model_validate(obj)
    except ValidationError:
        pass

    # Salvage: keep whatever tool_calls look reasonable
    tool_calls = []
    raw_calls = obj.get("tool_calls", []) if isinstance(obj, dict) else []
    if isinstance(raw_calls, list):
        for entry in raw_calls:
            if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                tool_calls.append(
                    ToolCall(
                        name=entry["name"],
                        args=entry.get("args", {}) if isinstance(entry.get("args"), dict) else {},
                    )
                )
    return Plan(reasoning=str(obj.get("reasoning", "")), tool_calls=tool_calls)


def _parse_verdict(raw: str) -> Verdict:
    obj = _extract_json_object(raw) or {}
    try:
        return Verdict.model_validate(obj)
    except ValidationError:
        pass

    accept = bool(obj.get("accept", True))
    repair_raw = obj.get("repair", [])
    repair = []
    if isinstance(repair_raw, list):
        for entry in repair_raw:
            if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                repair.append(
                    ToolCall(
                        name=entry["name"],
                        args=entry.get("args", {}) if isinstance(entry.get("args"), dict) else {},
                    )
                )
    return Verdict(accept=accept, reason=str(obj.get("reason", "")), repair=repair)


# ---------------------------------------------------------------------------
# Heuristic fallback writer (no LLM available)
# ---------------------------------------------------------------------------

def _render_fast_evidence(tool_records: list[dict]) -> str:
    """Flatten the search_facts tool output into plain bullet lines so the
    small local model can read it cleanly. Falls back to the JSON dump if the
    output shape is something we don't recognize.
    """
    lines: list[str] = []
    for rec in tool_records or []:
        out = rec.get("output") or {}
        facts = out.get("facts") if isinstance(out, dict) else None
        if isinstance(facts, list):
            for f in facts[:8]:
                text = (f.get("text") or "").strip() if isinstance(f, dict) else str(f).strip()
                if text:
                    lines.append(f"- {text}")
        elif isinstance(out, dict):
            for k, v in list(out.items())[:8]:
                if k in {"ok", "backend"}:
                    continue
                lines.append(f"- {k}: {v}")
    if not lines:
        return json.dumps(tool_records, default=str, indent=2)[:1200]
    body = "\n".join(lines[:12])
    return body[:1800]


def _heuristic_writer(question: str, evidence_text: str, retrieved_facts: list[dict]) -> str:
    bullet_facts = "\n".join(f"- {f['text']}" for f in retrieved_facts[:6])
    return (
        f"## Answer\n"
        f"No LLM endpoint is currently reachable, so this answer is assembled from the "
        f"top retrieved facts that match your question.\n\n"
        f"## Evidence\n{bullet_facts or '_no facts retrieved_'}\n\n"
        f"## Recommendation\n"
        f"Open the matching dashboard tabs (Anomalies, Leaderboard, Explainability) to "
        f"verify these facts visually.\n\n"
        f"## Limitation\n"
        f"This response was generated without an LLM. The agentic critic and writer "
        f"steps were skipped."
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def answer_with_agent(
    question: str,
    df: pd.DataFrame | None,
    analysis_result: dict,
    max_repairs: int = 1,
    fast: bool = True,
) -> dict:
    """
    Run the planner → executor → critic → writer loop.

    When `fast=True` (the default), runs a single-pass variant that skips the
    planner and critic/repair stages. The fast path issues exactly one writer
    LLM call after RAG retrieval, which is what the Ask Anything chat panel
    actually needs. Set `fast=False` to get the full multi-stage agent.

    Returns a JSON-safe dict:
        {
            "source": "ollama" | "openrouter" | "fallback",
            "answer": str,
            "trace": {
                "plan": {...},
                "tool_calls": [...],
                "verdict": {...},
                "repaired": bool,
            }
        }
    """
    question = (question or "").strip()
    if not question:
        return {"source": "fallback", "answer": "Please provide a question.", "trace": {}}

    facts = build_fact_index(analysis_result)
    ctx = AgentContext(df=df, analysis_result=analysis_result, facts=facts)

    # Always seed with retrieved facts so the writer has real evidence.
    retrieved = rag_retrieve(question, facts, k=10)
    retrieved_block = render_facts_block(retrieved, max_chars=2400)

    trace: dict[str, Any] = {
        "rag_backend": retrieved.get("backend"),
        "rag_top_k": len(retrieved.get("facts", [])),
        "tool_calls": [],
        "repaired": False,
        "fast": fast,
    }

    # ------------------------------------------------------------------
    # Fast path — single writer call, no planner / critic / repair.
    # ------------------------------------------------------------------
    if fast:
        # 1. Pack the dataset brief — schema, quality, signals, ML headlines,
        #    sample rows, AI narrative. This is the "what is this dataset"
        #    grounding the model needs to answer concretely.
        from framevitals.agent_brief import (
             build_dataset_brief,
             render_brief_block,
        )

        brief = build_dataset_brief(analysis_result)
        brief_block = render_brief_block(brief)

        # 2. Then a single deterministic search_facts to surface anything
        #    the brief truncated that's relevant to this question.
        single_call = ToolCall(name="search_facts", args={"question": question, "k": 5})
        tool_records = [{
            "name": single_call.name,
            "args": single_call.args,
            "output": run_tool(single_call.name, single_call.args, ctx),
        }]
        trace["tool_calls"].extend(tool_records)
        trace["brief_chars"] = len(brief_block)

        # Combined evidence: the brief gives broad context, the search_facts
        # output adds question-specific bullets. The brief comes first so
        # cloud / large models keep it in attention range easily.
        facts_text = _render_fast_evidence(tool_records)
        evidence_text = f"{brief_block}\n\n### Question-targeted facts\n{facts_text}"
        if os.environ.get("DATALENS_AGENT_DEBUG", "0") == "1":
            import sys
            print(f"[agent] fast-path evidence ({len(evidence_text)} chars):", file=sys.stderr)
            print(evidence_text, file=sys.stderr)
        msgs = _writer_prompt(question, evidence_text)
        if os.environ.get("DATALENS_AGENT_DEBUG", "0") == "1":
            import sys
            print(f"[agent] fast-path messages ({len(msgs)}):", file=sys.stderr)
            for m in msgs:
                print(f"  {m['role']}: {m['content'][:200]}...", file=sys.stderr)
        try:
            answer, writer_source = _call_llm(msgs, json_mode=False)
            if os.environ.get("DATALENS_AGENT_DEBUG", "0") == "1":
                import sys
                print(f"[agent] writer raw output ({len(answer)} chars): {answer[:200]!r}", file=sys.stderr)
            return {"source": writer_source, "answer": answer, "trace": trace}
        except Exception as exc:
            trace["error"] = f"writer failed: {exc}"
            return {
                "source": "fallback",
                "answer": _heuristic_writer(question, evidence_text, retrieved.get("facts", [])),
                "trace": trace,
            }

    # ------------------------------------------------------------------
    # Full path — planner → executor → critic → repair → writer.
    # ------------------------------------------------------------------

    # ---- 1. Plan -----------------------------------------------------------
    try:
        planner_text, llm_source = _call_llm(_planner_prompt(question, retrieved_block), json_mode=True)
    except Exception as exc:
        trace["error"] = f"planner failed: {exc}"
        return {
            "source": "fallback",
            "answer": _heuristic_writer(question, retrieved_block, retrieved.get("facts", [])),
            "trace": trace,
        }

    plan = _parse_plan(planner_text)
    trace["plan"] = plan.model_dump()
    trace["planner_source"] = llm_source

    # Ensure at least one tool call — fall back to RAG search if planner failed
    if not plan.tool_calls:
        plan.tool_calls = [ToolCall(name="search_facts", args={"question": question, "k": 8})]

    # ---- 2. Execute --------------------------------------------------------
    def _execute(calls: list[ToolCall]) -> list[dict]:
        records: list[dict] = []
        for call in calls[:5]:  # hard cap
            output = run_tool(call.name, call.args, ctx)
            records.append({"name": call.name, "args": call.args, "output": output})
        return records

    tool_records = _execute(plan.tool_calls)
    trace["tool_calls"].extend(tool_records)

    # ---- 3. Critic ---------------------------------------------------------
    evidence_text = json.dumps(tool_records, default=str, indent=2)[:6000]
    try:
        critic_text, _ = _call_llm(
            _critic_prompt(question, plan.reasoning, evidence_text), json_mode=True
        )
        verdict = _parse_verdict(critic_text)
    except Exception:
        verdict = Verdict(accept=True, reason="critic skipped: LLM error")

    trace["verdict"] = verdict.model_dump()

    # ---- 4. Repair (optional) ---------------------------------------------
    if not verdict.accept and verdict.repair and max_repairs > 0:
        repair_records = _execute(verdict.repair)
        tool_records.extend(repair_records)
        trace["tool_calls"].extend(repair_records)
        trace["repaired"] = True
        evidence_text = json.dumps(tool_records, default=str, indent=2)[:6000]

    # ---- 5. Writer ---------------------------------------------------------
    try:
        answer, writer_source = _call_llm(_writer_prompt(question, evidence_text), json_mode=False)
        return {
            "source": writer_source,
            "answer": answer,
            "trace": trace,
        }
    except Exception as exc:
        trace["error"] = f"writer failed: {exc}"
        return {
            "source": "fallback",
            "answer": _heuristic_writer(question, evidence_text, retrieved.get("facts", [])),
            "trace": trace,
        }
