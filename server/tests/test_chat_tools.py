"""Unit tests for chat tool-call parsing, confirmation, and MCP helpers."""

from types import SimpleNamespace

import pytest

from server.chat.tools.accumulator import ToolResultAccumulator
from server.chat.tools.call_parser import (
    accumulate_tool_call,
    finalized_tool_calls,
    parse_tool_arguments,
    tool_function_name,
)
from server.chat.tools.executor import requires_user_approval
from server.chat.tools.pending_actions import pop_pending_action, register_pending_action
from server.mcp.transport import (
    SSE,
    STREAMABLE_HTTP,
    mcp_tool_requires_confirmation,
    mcp_transport_order,
    sanitize_mcp_identifier,
)


class _DeltaFn:
    def __init__(self, name="", arguments=""):
        self.name = name
        self.arguments = arguments


class _DeltaCall:
    def __init__(self, index, call_id="", name="", arguments=""):
        self.index = index
        self.id = call_id
        self.type = "function"
        self.function = _DeltaFn(name, arguments)


def test_accumulate_openai_deltas_and_anthropic_dicts():
    accumulated = {}
    accumulate_tool_call(accumulated, _DeltaCall(0, "c1", "search_", '{"q":'))
    accumulate_tool_call(accumulated, _DeltaCall(0, "", "", '"hbA1c"}'))
    accumulate_tool_call(
        accumulated,
        {
            "id": "c2",
            "type": "function",
            "function": {"name": "wiki_search", "arguments": '{"query": "cll"}'},
        },
    )
    calls = finalized_tool_calls(accumulated)
    assert [tool_function_name(c) for c in calls] == ["search_", "wiki_search"]
    assert parse_tool_arguments(calls[0])["q"] == "hbA1c"
    assert calls[0]["id"] == "c1"


def test_finalized_tool_calls_drops_nameless_fragments():
    accumulated = {0: {"id": "x", "function": {"name": "", "arguments": ""}}}
    assert finalized_tool_calls(accumulated) == []


def test_register_pending_action_stores_patient_context():
    action = register_pending_action(
        tool_call={"function": {"name": "create_note", "arguments": "{}"}},
        tool_name="create_note",
        patient_context={"name": "Doe, Jane", "ur_number": "UR1"},
    )
    assert action.patient_context == {"name": "Doe, Jane", "ur_number": "UR1"}
    popped = pop_pending_action(action.id)
    assert popped is not None
    assert popped.patient_context is not None
    assert popped.patient_context["ur_number"] == "UR1"


@pytest.mark.asyncio
async def test_mutating_tool_is_parked_not_executed():
    from server.chat.tools.executor import execute_tool_streaming

    chunks = []
    async for chunk in execute_tool_streaming(
        tool_call={"function": {"name": "create_note", "arguments": "{}"}},
        llm_client=None,
        config={},
        message_list=[],
        context_question_options={},
    ):
        chunks.append(chunk)

    types = [c.get("type") for c in chunks]
    assert "confirmation" in types
    assert types[-1] == "end"
    confirm = next(c for c in chunks if c.get("type") == "confirmation")
    assert confirm["tool"] == "create_note"
    assert pop_pending_action(confirm["action_id"]) is not None


@pytest.mark.asyncio
async def test_accumulator_keeps_artifacts():
    async def stream():
        yield {"type": "artifact", "artifact": {"type": "form_fill", "template_id": "t1"}}
        yield {
            "type": "end",
            "function_response": {"content": "filled", "citations": ["c1"]},
        }

    acc = ToolResultAccumulator()
    content, citations = await acc.consume_stream(stream())
    assert content == "filled"
    assert citations == ["c1"]
    assert acc.artifacts[0]["type"] == "form_fill"


def test_mcp_transport_order_prefers_streamable_http():
    assert mcp_transport_order("http://localhost:3000/mcp") == [STREAMABLE_HTTP, SSE]
    assert mcp_transport_order("http://localhost:3000/sse") == [SSE, STREAMABLE_HTTP]


def test_sanitize_mcp_identifier_strips_non_ascii():
    assert sanitize_mcp_identifier("My Server/v2") == "my_server_v2"
    assert sanitize_mcp_identifier("جستجو.web") == "web"
    assert sanitize_mcp_identifier("...") == "tool"


def test_mcp_tool_requires_confirmation_from_annotations():
    destructive = SimpleNamespace(annotations={"destructiveHint": True})
    read_only = SimpleNamespace(annotations={"readOnlyHint": True})
    writable = SimpleNamespace(annotations={"readOnlyHint": False})
    plain = SimpleNamespace(annotations=None)
    no_hint = SimpleNamespace(annotations={})
    assert mcp_tool_requires_confirmation(destructive) is True
    assert mcp_tool_requires_confirmation(read_only) is False
    assert mcp_tool_requires_confirmation(writable) is True
    # Fail closed: unknown/absent annotations must NOT run without approval.
    assert mcp_tool_requires_confirmation(plain) is True
    assert mcp_tool_requires_confirmation(no_hint) is True


def test_requires_user_approval_builtin():
    assert requires_user_approval("create_note") is True
    assert requires_user_approval("wiki_search") is False
    assert requires_user_approval("create_note", {"_approved": True}) is False


def test_requires_user_approval_mcp(monkeypatch):
    pytest.importorskip("mcp")
    monkeypatch.setattr(
        "server.mcp.client.get_mcp_tools_sync",
        lambda: [
            {
                "function": {"name": "mcp_lab_write"},
                "_mcp_requires_confirmation": True,
            }
        ],
    )
    assert requires_user_approval("mcp_lab_write") is True
    assert requires_user_approval("mcp_lab_read") is False
