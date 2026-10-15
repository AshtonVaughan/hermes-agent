"""jev plugin — TypeSafe Jev decides, Hermes acts, DeepSeek writes.

Per user turn (``pre_llm_call``): one Jev request scores how much reasoning the turn needs and
ranks the skill roster; a second re-checks the top three. The winning skill rides into the turn
as a ``<skill_relevance>`` line, and the reasoning level is stamped onto every DeepSeek request of
that turn by ``llm_request`` middleware (thinking off / low / high).

Per gated tool call (``pre_tool_call``): one Noul asks whether the call serves what the user asked
for. Below ``action_block_below`` the call is blocked with a message to the model. Nothing here
ever asks the user anything.

Fail-open everywhere: any Jev error, timeout or missing key means "no decision" and stock Hermes
behavior. ``pre_tool_call`` must never raise, because Hermes fails a raising gate closed.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Any

from .decide import (
    Settings,
    TurnPlan,
    block_message,
    judge_action,
    plan_turn,
    skill_context,
)

logger = logging.getLogger(__name__)

_MAX_TURNS = 512
_lock = threading.Lock()
_turns: OrderedDict[str, TurnPlan] = OrderedDict()
_latest_by_session: dict[str, str] = {}
_verdicts: OrderedDict[tuple, float] = OrderedDict()
_clients: dict[tuple[str, str], Any] = {}
_settings = Settings()
# computer_use actions with no side effects: judging them would only add latency.
_READ_ONLY_COMPUTER_ACTIONS = frozenset({"capture", "list_apps", "list_windows", "wait"})


# ── state ────────────────────────────────────────────────────────────────────

def _remember(turn_id: str, session_id: str, plan: TurnPlan) -> None:
    key = turn_id or session_id
    if not key:
        return
    with _lock:
        _turns[key] = plan
        _turns.move_to_end(key)
        while len(_turns) > _MAX_TURNS:
            _turns.popitem(last=False)
        if session_id:
            _latest_by_session[session_id] = key


def _plan_for(turn_id: str, session_id: str) -> TurnPlan | None:
    with _lock:
        if turn_id and turn_id in _turns:
            return _turns[turn_id]
        key = _latest_by_session.get(session_id or "")
        return _turns.get(key) if key else None


def _trace(event: str, **fields: Any) -> None:
    try:
        from plugins.plugin_storage import plugin_data_dir

        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **fields}
        with (plugin_data_dir("jev") / "trace.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        logger.debug("jev: trace write failed", exc_info=True)


# ── clients and inputs ───────────────────────────────────────────────────────

def _client() -> Any | None:
    """TypeSafe client for the active profile's key, or None when no key is configured."""
    try:
        from agent.secret_scope import get_secret

        key = get_secret("TYPESAFE_API_KEY") or ""
        base_url = get_secret("TYPESAFE_BASE_URL") or ""
    except Exception:
        return None
    if not key:
        return None
    cache_key = (key, base_url)
    with _lock:
        client = _clients.get(cache_key)
        if client is None:
            try:
                from typesafe_sdk import RetryPolicy, TypeSafeClient
            except ImportError:
                logger.warning("jev: typesafe-sdk is not installed; run `hermes plugins enable jev`")
                return None
            client = TypeSafeClient(
                api_key=key,
                base_url=base_url or None,
                model=_settings.jev_model,
                timeout=_settings.timeout,
                retry=RetryPolicy(max_retries=1, timeout=_settings.timeout),
            )
            _clients[cache_key] = client
    return client


def _skills() -> list[dict[str, Any]]:
    """Loadable skills as {name, description, path}. Internal API, so guarded."""
    try:
        from tools.skills_tool import _skill_catalog

        return [
            {"name": s["load_name"], "description": s.get("description") or "", "path": s.get("path")}
            for s in _skill_catalog()
            if s.get("load_name")
        ]
    except Exception:
        logger.debug("jev: skill catalog unavailable", exc_info=True)
        return []


def _is_deepseek(provider: str, base_url: str) -> bool:
    return (provider or "").lower() == "deepseek" or "deepseek.com" in (base_url or "").lower()


# ── hooks and middleware ─────────────────────────────────────────────────────

def on_pre_llm_call(
    session_id: str = "",
    turn_id: str = "",
    user_message: Any = "",
    conversation_history: list | None = None,
    **_: Any,
) -> dict[str, str] | None:
    text = user_message if isinstance(user_message, str) else ""
    if not text.strip():
        return None
    # History includes the current user message last; drop it so it isn't counted twice.
    history = list(conversation_history or [])
    if history and history[-1].get("role") == "user":
        history = history[:-1]
    client = _client()
    plan: TurnPlan
    if client is None:
        from .decide import recent_context

        plan = TurnPlan(user_message=text, recent_context=recent_context(history), error="no TYPESAFE_API_KEY")
    else:
        started = time.perf_counter()
        try:
            plan = plan_turn(client, text, history, _skills(), _settings)
        except Exception as exc:  # fail open
            from .decide import recent_context

            logger.warning("jev: turn plan failed, using stock behavior: %s", exc)
            plan = TurnPlan(user_message=text, recent_context=recent_context(history), error=repr(exc))
        _trace("plan", session_id=session_id, turn_id=turn_id,
               ms=round((time.perf_counter() - started) * 1000), **plan.summary())
    _remember(turn_id, session_id, plan)
    return {"context": skill_context(plan.skill)} if plan.skill else None


def on_llm_request(
    request: dict[str, Any],
    session_id: str = "",
    turn_id: str = "",
    provider: str = "",
    base_url: str = "",
    **_: Any,
) -> dict[str, Any] | None:
    if not _settings.reasoning_gate or not _is_deepseek(provider, base_url):
        return None
    plan = _plan_for(turn_id, session_id)
    if plan is None or plan.reasoning_effort is None:
        return None
    req = dict(request)
    extra_body = dict(req.get("extra_body") or {})
    if plan.reasoning_effort == "disabled":
        extra_body["thinking"] = {"type": "disabled"}
        req.pop("reasoning_effort", None)
    else:
        extra_body["thinking"] = {"type": "enabled"}
        req["reasoning_effort"] = plan.reasoning_effort
    req["extra_body"] = extra_body
    return {
        "request": req,
        "source": "jev",
        "reason": f"thinking={plan.reasoning_effort} (score {plan.reasoning_score:.2f})",
    }


def on_pre_tool_call(
    tool_name: str = "",
    args: dict | None = None,
    session_id: str = "",
    turn_id: str = "",
    **_: Any,
) -> dict[str, str] | None:
    try:
        if not _settings.action_gate or tool_name not in _settings.gated_tools:
            return None
        if tool_name == "computer_use" and (args or {}).get("action") in _READ_ONLY_COMPUTER_ACTIONS:
            return None
        plan = _plan_for(turn_id, session_id)
        client = _client()
        if plan is None or client is None:
            return None
        try:
            fingerprint = json.dumps(args or {}, sort_keys=True, default=str)
        except Exception:
            fingerprint = str(args)
        cache_key = (turn_id or session_id, tool_name, fingerprint)
        with _lock:
            p = _verdicts.get(cache_key)
        if p is None:
            started = time.perf_counter()
            p = judge_action(client, plan, tool_name, args, _settings)
            with _lock:
                _verdicts[cache_key] = p
                while len(_verdicts) > _MAX_TURNS * 4:
                    _verdicts.popitem(last=False)
            blocked = p < _settings.action_block_below
            _trace("action", session_id=session_id, turn_id=turn_id, tool=tool_name, p=round(p, 3),
                   blocked=blocked, ms=round((time.perf_counter() - started) * 1000))
        if p < _settings.action_block_below:
            return {"action": "block", "message": block_message(tool_name, p)}
        return None
    except Exception as exc:  # fail open: a raising pre_tool_call hook would block the tool
        logger.warning("jev: action check failed, allowing %s: %s", tool_name, exc)
        return None


def _jev_command(raw_args: str = "") -> str:
    """/jev [n] — the last n Jev decisions (default 8)."""
    try:
        from plugins.plugin_storage import plugin_data_dir

        n = int(raw_args.strip() or 8)
        lines = (plugin_data_dir("jev") / "trace.jsonl").read_text(encoding="utf-8").splitlines()[-n:]
    except FileNotFoundError:
        return "Jev has made no decisions yet."
    except Exception as exc:
        return f"Could not read the Jev trace: {exc}"
    out = []
    for line in lines:
        r = json.loads(line)
        if r["event"] == "plan":
            if r.get("error"):
                out.append(f"{r['ts']}  plan  error: {r['error']}")
                continue
            score = r.get("reasoning_score")
            reasoning = f"thinking={r.get('reasoning_effort')} ({score:.2f})" if score is not None else "thinking=unchanged"
            gate = r.get("skill_gate")
            skill = r.get("skill") or ("none" if gate is None else f"none (gate {gate:.2f})")
            out.append(f"{r['ts']}  plan  {reasoning}  skill={skill}  {r['ms']}ms  «{r['user_message'][:60]}»")
        else:
            verdict = "BLOCKED" if r["blocked"] else "allowed"
            out.append(f"{r['ts']}  tool  {r['tool']} {verdict} p={r['p']:.2f}  {r['ms']}ms")
    return "\n".join(out) or "Jev has made no decisions yet."


def _load_settings(ctx: Any) -> Settings:
    defaults = Settings()
    values: dict[str, Any] = {}
    for name in Settings.__dataclass_fields__:
        try:
            value = ctx.get_config(name, getattr(defaults, name))
        except Exception:
            value = getattr(defaults, name)
        values[name] = tuple(value) if name == "gated_tools" else value
    return Settings(**values)


def register(ctx: Any) -> None:
    global _settings
    _settings = _load_settings(ctx)
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_middleware("llm_request", on_llm_request)
    ctx.register_command("jev", _jev_command, description="Show recent Jev decisions", args_hint="[n]")
