"""Tests for the bundled jeeves_presence plugin (plugins/jeeves_presence/).

The plugin loads through real discovery. Events are read off a real UDP socket standing in for
the Jeeves app, and computer_use results use the shapes tools/computer_use/tool.py produces.
"""

from __future__ import annotations

import json
import socket
import time

import pytest

from hermes_cli import plugins
from hermes_cli.middleware import run_tool_execution_middleware
from hermes_cli.plugins import PluginManager

SOM_TEXT = (
    "capture mode=som 1440x900 app=Safari window='Inbox'\n"
    "3 interactable element(s):\n"
    "  #1 AXButton 'Back' @ (100, 200, 40, 20) [Safari]\n"
    "  #2 AXTextField 'Search' @ (300, 120, 400, 30) [Safari]\n"
    "  #3 AXGroup 'empty' @ bounds-unknown (click by element index)\n"
)
AX_JSON = json.dumps({
    "mode": "ax", "width": 1440, "height": 900, "app": "Notes", "window_title": "Recipes",
    "elements": [{"index": 1, "role": "AXButton", "label": "New", "bounds": [10, 20, 30, 40], "app": "Notes"},
                 {"index": 2, "role": "AXRow", "label": "x", "bounds": None, "app": "Notes"}],
    "total_elements": 2, "summary": "",
})


@pytest.fixture
def listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    yield sock
    sock.close()


@pytest.fixture
def presence(tmp_path, monkeypatch, listener):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "plugins:\n  enabled: [jeeves_presence]\n"
        "  entries:\n    jeeves_presence:\n      settings:\n"
        f"        port: {listener.getsockname()[1]}\n        cursor_lead_ms: 30\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    manager.discover_and_load()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    loaded = manager._plugins["jeeves_presence"]
    assert loaded.enabled, loaded.error
    loaded.module._captures.clear()
    return loaded.module


def drain(sock):
    events = []
    sock.settimeout(0.2)
    while True:
        try:
            events.append(json.loads(sock.recv(65536)))
        except (TimeoutError, BlockingIOError):
            return events


def run_tool(name, args, result, session_id="s1"):
    """The real middleware chain around a fake tool that returns *result*."""
    return run_tool_execution_middleware(name, args, lambda a: result, session_id=session_id)


class TestGeometry:
    def test_parses_som_text_capture(self, presence):
        from plugins.jeeves_presence.geometry import CaptureState, update_from_result

        state = CaptureState()
        assert update_from_result(state, SOM_TEXT)
        assert (state.app, state.window_title, state.width, state.height) == ("Safari", "Inbox", 1440, 900)
        assert state.center(1) == (120, 210)
        assert state.center(3) is None  # bounds-unknown element has no position

    def test_parses_json_capture_and_nulls(self, presence):
        from plugins.jeeves_presence.geometry import CaptureState, update_from_result

        state = CaptureState()
        assert update_from_result(state, AX_JSON)
        assert state.center(1) == (25, 40)
        assert state.center(2) is None

    def test_multimodal_envelope_uses_text_summary(self, presence):
        from plugins.jeeves_presence.geometry import CaptureState, update_from_result

        state = CaptureState()
        assert update_from_result(state, {"_multimodal": True, "text_summary": SOM_TEXT, "content": []})
        assert state.center(2) == (500, 135)

    def test_non_capture_result_leaves_state(self, presence):
        from plugins.jeeves_presence.geometry import CaptureState, update_from_result

        state = CaptureState()
        assert not update_from_result(state, '{"ok": true, "message": "clicked"}')
        assert state.elements == {}

    def test_plan_pointer_variants(self, presence):
        from plugins.jeeves_presence.geometry import CaptureState, plan_pointer, update_from_result

        state = CaptureState()
        update_from_result(state, SOM_TEXT)
        assert plan_pointer(state, {"action": "click", "element": 1}) == {
            "action": "click", "button": "left", "point": (120, 210)}
        assert plan_pointer(state, {"action": "right_click", "element": 2})["button"] == "right"
        by_coord = plan_pointer(state, {"action": "click", "coordinate": [50, 60]})
        assert by_coord["window_point"] == (50, 60) and by_coord["app"] == "Safari"
        drag = plan_pointer(state, {"action": "drag", "from_element": 1, "to_element": 2})
        assert drag["from"] == {"point": (120, 210)} and drag["to"] == {"point": (500, 135)}
        assert plan_pointer(state, {"action": "scroll", "direction": "down"})["direction"] == "down"
        assert plan_pointer(state, {"action": "capture"}) is None
        assert plan_pointer(state, {"action": "click", "element": 99}) is None


class TestEvents:
    def test_hooks_emit_turn_lifecycle(self, presence, listener):
        presence.on_pre_llm_call(session_id="s1", user_message="open Notes", platform="cli")
        presence.on_pre_api_request(session_id="s1", api_call_count=1)
        presence.on_post_llm_call(session_id="s1", assistant_response="Done.")
        presence.on_session_end(session_id="s1", interrupted=False)
        kinds = [e["event"] for e in drain(listener)]
        assert kinds == ["turn_start", "thinking", "reply", "idle"]

    def test_plain_tool_emits_start_and_end(self, presence, listener):
        assert run_tool("terminal", {"command": "ls"}, "ok") == "ok"
        events = drain(listener)
        assert [e["event"] for e in events] == ["tool_start", "tool_end"]
        assert events[0]["summary"] == "ls" and events[1]["ok"] is True

    def test_failed_tool_reports_not_ok_and_propagates(self, presence, listener):
        def boom(_):
            raise ValueError("nope")

        with pytest.raises(ValueError):
            run_tool_execution_middleware("terminal", {"command": "x"}, boom, session_id="s1")
        events = drain(listener)
        assert [e["event"] for e in events] == ["tool_start", "tool_end"]
        assert events[-1]["ok"] is False

    def test_no_listener_never_breaks_the_tool(self, presence, listener):
        listener.close()
        assert run_tool("terminal", {"command": "ls"}, "fine") == "fine"

    def test_oversized_event_is_truncated_not_dropped(self, presence, listener):
        presence.emit("reply", text="x" * 200_000)
        event = drain(listener)[0]
        assert event["event"] == "reply" and event["truncated"] is True


class TestCursorChoreography:
    def test_click_by_element_leads_with_cursor_event(self, presence, listener):
        run_tool("computer_use", {"action": "capture", "mode": "som"}, SOM_TEXT)
        drain(listener)
        started = time.perf_counter()
        run_tool("computer_use", {"action": "click", "element": 2}, '{"ok": true}')
        elapsed = time.perf_counter() - started
        events = drain(listener)
        assert [e["event"] for e in events] == ["tool_start", "cursor", "cursor_done", "tool_end"]
        assert events[1]["point"] == [500, 135] and events[1]["action"] == "click"
        assert elapsed >= 0.03  # the configured lead, so the cursor arrives before the input

    def test_capture_updates_state_and_emits_capture_event(self, presence, listener):
        run_tool("computer_use", {"action": "capture"}, SOM_TEXT)
        events = drain(listener)
        assert [e["event"] for e in events] == ["tool_start", "capture", "tool_end"]
        assert events[1]["app"] == "Safari"

    def test_typing_emits_keys_without_cursor(self, presence, listener):
        run_tool("computer_use", {"action": "type", "text": "hello"}, '{"ok": true}')
        events = drain(listener)
        assert [e["event"] for e in events] == ["tool_start", "keys", "tool_end"]
        assert events[1]["text"] == "hello"

    def test_sessions_do_not_share_captures(self, presence, listener):
        run_tool("computer_use", {"action": "capture"}, SOM_TEXT, session_id="a")
        drain(listener)
        run_tool("computer_use", {"action": "click", "element": 1}, "{}", session_id="b")
        assert "cursor" not in [e["event"] for e in drain(listener)]
