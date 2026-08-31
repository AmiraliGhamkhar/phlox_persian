"""Central tool execution dispatch.

This module provides a unified entry point for executing tools in both
streaming (ChatEngine) and non-streaming (reasoning) contexts.
"""

import logging
from collections.abc import AsyncGenerator
from typing import Any

from .accumulator import ToolResultAccumulator

logger = logging.getLogger(__name__)

# Tools that write to the medical record. They are never executed on the
# model's say-so alone: the call is parked as a pending action and only runs
# after the user approves the confirmation card in the UI (LLM03:2026
# Excessive Agency / human-approval gate for high-stakes actions).
MUTATING_TOOLS = {"create_note", "complete_job", "fill_pdf_form"}


def requires_user_approval(function_name: str, tool_call: dict[str, Any] | None = None) -> bool:
    """Return True when this tool must be parked behind the confirmation card.

    Built-in record-writing tools always require approval. MCP tools require it
    when the server advertised a destructive / non-read-only annotation.
    """
    if tool_call and tool_call.get("_approved"):
        return False
    if function_name in MUTATING_TOOLS:
        return True
    if function_name.startswith("mcp_"):
        try:
            from server.mcp.client import get_mcp_tools_sync

            for tool in get_mcp_tools_sync():
                if tool.get("function", {}).get("name") == function_name:
                    return bool(tool.get("_mcp_requires_confirmation"))
        except Exception:
            return False
    return False


async def execute_tool_streaming(
    tool_call: dict[str, Any],
    llm_client,
    config: dict[str, Any],
    message_list: list,
    context_question_options: dict[str, Any],
    vector_store_manager=None,
    conversation_history: list | None = None,
    raw_transcription: str | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Execute a tool with streaming response.

    Central dispatch supporting all tools including MCP.
    Used by ChatEngine for real-time streaming responses.

    Args:
        tool_call: The tool call to execute
        llm_client: The LLM client instance
        config: The configuration dictionary
        message_list: The current message list
        context_question_options: The context question options
        vector_store_manager: Optional VectorStoreManager for literature search
        conversation_history: The conversation history (for transcript search)
        raw_transcription: The raw transcription (for transcript search)

    Yields:
        Dict[str, Any]: Streaming response chunks
    """
    function_name = tool_call["function"]["name"]
    logger.info(f"Executing tool (streaming): {function_name}")

    # Human-approval gate: park record-writing tools until the user approves.
    # Approved re-runs carry the "_approved" marker set by the confirm endpoint.
    if requires_user_approval(function_name, tool_call):
        from server.chat.tools.pending_actions import register_pending_action
        from server.chat.streaming.response import confirmation_message, end_message
        from server.utils.request_context import get_request_actor

        from .sanitization import get_active_patient_context

        action = register_pending_action(
            tool_call=tool_call,
            tool_name=function_name,
            llm_client=llm_client,
            config=config,
            message_list=message_list,
            context_question_options=context_question_options,
            vector_store_manager=vector_store_manager,
            conversation_history=conversation_history,
            raw_transcription=raw_transcription,
            patient_context=get_active_patient_context(),
            actor=get_request_actor(),
        )
        logger.info(f"Tool '{function_name}' requires user approval (action {action.id})")
        yield confirmation_message(action.id, function_name, action.summary)
        yield end_message(
            function_response={
                "content": (
                    f"The '{function_name}' action requires the user's explicit "
                    "approval before it can run. A confirmation card has been "
                    "shown in the UI. Do NOT call this tool again; briefly tell "
                    "the user you are waiting for their approval."
                ),
                "citations": [],
                "pending_action_id": action.id,
            }
        )
        return

    if function_name == "direct_response":
        from .direct_response import execute

        async for result in execute(
            tool_call, llm_client, config, message_list, context_question_options
        ):
            yield result

    elif function_name == "transcript_search":
        from .transcript_search import execute

        async for result in execute(
            tool_call,
            llm_client,
            config,
            message_list,
            conversation_history or [],
            raw_transcription,
            context_question_options,
        ):
            yield result

    elif function_name == "pubmed_search":
        from .pubmed_search import execute

        async for result in execute(
            tool_call, llm_client, config, message_list, context_question_options
        ):
            yield result

    elif function_name == "wiki_search":
        from .wiki_search import execute

        async for result in execute(
            tool_call, llm_client, config, message_list, context_question_options
        ):
            yield result

    elif function_name == "get_previous_encounter":
        from .previous_encounter import execute

        async for result in execute(
            tool_call, llm_client, config, message_list, context_question_options
        ):
            yield result

    elif function_name == "create_note":
        from .create_note import execute

        async for result in execute(
            tool_call, llm_client, config, message_list, context_question_options
        ):
            yield result

    elif function_name == "get_patient_jobs":
        from .patient_jobs import execute

        async for result in execute(
            tool_call, llm_client, config, message_list, context_question_options
        ):
            yield result

    elif function_name == "search_patient":
        from .search_patient import execute

        async for result in execute(
            tool_call, llm_client, config, message_list, context_question_options
        ):
            yield result

    elif function_name == "search_patients_by_condition":
        from .search_patients_by_condition import execute

        async for result in execute(
            tool_call, llm_client, config, message_list, context_question_options
        ):
            yield result

    elif function_name == "todo_list":
        from .todo_list import execute

        async for result in execute(
            tool_call, llm_client, config, message_list, context_question_options
        ):
            yield result

    elif function_name == "search_patient_notes":
        from .search_patient_notes import execute

        async for result in execute(
            tool_call, llm_client, config, message_list, context_question_options
        ):
            yield result

    elif function_name == "list_outstanding_jobs":
        from .list_outstanding_jobs import execute

        async for result in execute(
            tool_call, llm_client, config, message_list, context_question_options
        ):
            yield result

    elif function_name == "complete_job":
        from .complete_job import execute

        async for result in execute(
            tool_call, llm_client, config, message_list, context_question_options
        ):
            yield result

    elif function_name == "list_pdf_form_templates":
        from .pdf_forms import list_templates as execute_list

        async for result in execute_list(
            tool_call, llm_client, config, message_list, context_question_options
        ):
            yield result

    elif function_name == "fill_pdf_form":
        from .pdf_forms import fill_form as execute_fill

        async for result in execute_fill(
            tool_call, llm_client, config, message_list, context_question_options
        ):
            yield result

    elif function_name == "get_relevant_literature":
        from .direct_response import execute as execute_direct
        from .literature_search import execute as execute_literature

        if vector_store_manager is None:
            logger.warning(
                "Literature search requested but vector_store_manager not available. "
                "Falling back to direct response."
            )
            async for result in execute_direct(
                tool_call, llm_client, config, message_list, context_question_options
            ):
                yield result
        else:
            async for result in execute_literature(
                tool_call,
                llm_client,
                config,
                vector_store_manager,
                message_list,
                context_question_options,
            ):
                yield result

    elif function_name.startswith("mcp_"):
        from .mcp_tool import execute

        async for result in execute(
            tool_call, llm_client, config, message_list, context_question_options
        ):
            yield result

    else:
        logger.warning(f"Unknown tool requested: {function_name}")
        from server.chat.streaming.response import end_message, status_message

        yield status_message(f"Tool '{function_name}' not found")
        yield end_message(
            function_response={
                "content": (
                    f"Error: tool '{function_name}' does not exist. "
                    "Do not call it again. For literature/web searches use "
                    "'wiki_search' (general background, drugs, disease overviews), "
                    "'pubmed_search' (research articles), or "
                    "'get_relevant_literature' (clinical guidelines in the local "
                    "knowledge base). For patients use 'search_patient' or "
                    "'search_patients_by_condition'."
                ),
                "citations": [],
            }
        )


async def execute_tool_non_streaming(
    tool_call: dict[str, Any],
    config: dict[str, Any],
    vector_store_manager=None,
) -> tuple[str, list[str] | None]:
    """Execute a tool without streaming.

    This function consumes the streaming output and accumulates the result.
    Used by reasoning context where we need to collect results
    before generating the final structured output.

    Args:
        tool_call: The tool call to execute
        config: The configuration dictionary
        vector_store_manager: Optional VectorStoreManager for literature search

    Returns:
        Tuple of (result_string, citations_list) where citations_list
        contains formatted citation strings for display, or None if
        no citations are available.
    """
    function_name = tool_call["function"]["name"]
    logger.info(f"Executing tool (non-streaming via accumulator): {function_name}")

    # Human-approval gate: the reasoning context has no confirmation card to
    # approve with, so record-writing tools must never run here (and must not
    # register an unconfirmable pending action with llm_client=None).
    if function_name in MUTATING_TOOLS and not tool_call.get("_approved"):
        logger.warning(f"Blocked mutating tool '{function_name}' in non-streaming context")
        return (
            f"The '{function_name}' tool writes data and requires explicit user "
            "approval in the chat flow, so it was NOT executed here. Summarize "
            "what the user would need to confirm instead.",
            None,
        )

    accumulator = ToolResultAccumulator()

    stream = execute_tool_streaming(
        tool_call=tool_call,
        llm_client=None,
        config=config,
        message_list=[],
        context_question_options={},
        vector_store_manager=vector_store_manager,
    )

    # Consume the stream and return accumulated result
    return await accumulator.consume_stream(stream)
