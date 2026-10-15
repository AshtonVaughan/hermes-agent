"""jeeves_presence — tells the Jeeves orb app what the agent is doing.

Events go to 127.0.0.1:<port> as one JSON object per UDP datagram. UDP keeps this
fire-and-forget: a closed port costs nothing and never blocks or fails a turn.

computer_use pointer actions are paced: the target is emitted first, the input waits
``cursor_lead_ms`` so the ghost cursor visibly arrives, then the background input runs.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from typing import Any, Callable

from .geometry import KEY_ACTIONS, CaptureState, plan_pointer, update_from_result

logger = logging.getLogger(__name__)

_MAX_DATAGRAM = 60_000
_lock = threading.Lock()
_captures: dict[str, CaptureState] = {}
_sock: socket.socket | None = None
_port = 47800
_lead_seconds = 0.42


def emit(event: str, **fields: Any) -> None:
    global _sock
    try:
        payload = json.dumps({"event": event, "ts": time.time(), **fields}, ensure_ascii=False, default=str)
        data = payload.encode("utf-8")
        if len(data) > _MAX_DATAGRAM:
            data = json.dumps({"event": event, "ts": time.time(), "truncated": True}).encode()
        with _lock:
            if _sock is None:
                _sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                _sock.setblocking(False)
            _sock.sendto(data, ("127.0.0.1", _port))
    except OSError:
        pass  # nobody listening, buffer full, ... — presence is best-effort
    except Exception:
        logger.debug("jeeves_presence: emit failed", exc_info=True)


def _capture_state(session_id: str) -> CaptureState:
    with _lock:
        return _captures.setdefault(session_id or "", CaptureState())


def _tool_summary(tool_name: str, args: dict[str, Any]) -> str:
    if tool_name == "terminal":
        return str(args.get("command") or "")[:160]
    if tool_name in ("web_search", "x_search"):
        return str(args.get("query") or "")[:160]
    if tool_name in ("web_extract", "browser_navigate"):
        return str(args.get("url") or args.get("urls") or "")[:160]
    if tool_name in ("read_file", "write_file", "patch"):
        return str(args.get("path") or "")[:160]
    if tool_name == "computer_use":
        action = str(args.get("action") or "")
        detail = args.get("text") or args.get("keys") or args.get("app") or args.get("value") or ""
        return f"{action} {detail}".strip()[:160]
    return ""


# ── hooks ────────────────────────────────────────────────────────────────────

def on_pre_llm_call(session_id: str = "", user_message: Any = "", platform: str = "", **_: Any) -> None:
    emit("turn_start", session_id=session_id, platform=platform,
         text=user_message[:400] if isinstance(user_message, str) else "")


def on_pre_api_request(session_id: str = "", api_call_count: int = 0, **_: Any) -> None:
    emit("thinking", session_id=session_id, step=api_call_count)


def on_post_llm_call(session_id: str = "", assistant_response: Any = "", platform: str = "", **_: Any) -> None:
    emit("reply", session_id=session_id, platform=platform,
         text=assistant_response[:6000] if isinstance(assistant_response, str) else "")


def on_session_end(session_id: str = "", interrupted: bool = False, **_: Any) -> None:
    emit("idle", session_id=session_id, interrupted=bool(interrupted))


def on_tool_execution(tool_name: str = "", args: Any = None, next_call: Callable[..., Any] | None = None,
                      session_id: str = "", **_: Any) -> Any:
    args = args if isinstance(args, dict) else {}
    assert next_call is not None
    emit("tool_start", session_id=session_id, tool=tool_name, summary=_tool_summary(tool_name, args))
    cursor = None
    if tool_name == "computer_use":
        action = str(args.get("action") or "")
        cursor = plan_pointer(_capture_state(session_id), args)
        if cursor:
            emit("cursor", session_id=session_id, lead_ms=int(_lead_seconds * 1000), **cursor)
            time.sleep(_lead_seconds)
        elif action in KEY_ACTIONS:
            emit("keys", session_id=session_id, action=action,
                 text=str(args.get("text") or args.get("keys") or "")[:200])
    ok = False
    try:
        result = next_call()
        ok = True
        return result
    finally:
        if tool_name == "computer_use" and ok:
            try:
                if update_from_result(_capture_state(session_id), result):
                    emit("capture", session_id=session_id, app=_capture_state(session_id).app)
            except Exception:
                logger.debug("jeeves_presence: capture parse failed", exc_info=True)
        if cursor:
            emit("cursor_done", session_id=session_id, action=cursor["action"], ok=ok)
        emit("tool_end", session_id=session_id, tool=tool_name, ok=ok)


def register(ctx: Any) -> None:
    global _port, _lead_seconds
    try:
        _port = int(ctx.get_config("port", 47800))
        _lead_seconds = max(0.0, int(ctx.get_config("cursor_lead_ms", 420)) / 1000)
    except Exception:
        pass
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("pre_api_request", on_pre_api_request)
    ctx.register_hook("post_llm_call", on_post_llm_call)
    ctx.register_hook("on_session_end", on_session_end)
    ctx.register_middleware("tool_execution", on_tool_execution)
