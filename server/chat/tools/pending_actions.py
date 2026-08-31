"""Pending action store for the tool-confirmation gate.

State-changing agent tools (create_note, complete_job, fill_pdf_form) never run
directly when the model calls them. Instead the executor registers the call
here, the UI shows a confirmation card, and the action only executes after the
user explicitly approves it (or is dropped on cancel/expiry).

This is the code-level counterpart of the prompt-level rule in the system
prompt: an LLM (or injected content steering it) cannot write to the medical
record without a human clicking "Approve".
"""

import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

ACTION_TTL_SECONDS = 15 * 60
MAX_PENDING_ACTIONS = 100

_pending: dict[str, "PendingAction"] = {}


@dataclass
class PendingAction:
    """A tool call captured from the model, awaiting user approval."""

    id: str
    tool_call: dict[str, Any]
    tool_name: str
    summary: str
    llm_client: Any = None
    config: dict[str, Any] = field(default_factory=dict)
    message_list: list = field(default_factory=list)
    context_question_options: dict[str, Any] = field(default_factory=dict)
    vector_store_manager: Any = None
    conversation_history: list = field(default_factory=list)
    raw_transcription: str | None = None
    patient_context: dict[str, Any] | None = None
    # AuthN identity that queued the action; confirm/cancel require a match
    # (API1:2023 object-level authz). None = pre-auth desktop/local.
    actor: str | None = None
    created_at: float = field(default_factory=time.time)


def summarize_tool_args(tool_call: dict[str, Any], max_len: int = 300) -> str:
    """Human-readable one-line summary of the tool's arguments."""
    import json

    raw = tool_call.get("function", {}).get("arguments", "") or ""
    try:
        args = json.loads(raw) if isinstance(raw, str) and raw.strip() else raw
    except json.JSONDecodeError:
        args = raw
    if not isinstance(args, dict):
        args = {"arguments": str(args)[:120]}
    parts = []
    for key, value in args.items():
        text = str(value).replace("\n", " ")
        if len(text) > 80:
            text = text[:77] + "…"
        parts.append(f"{key}: {text}")
    summary = "; ".join(parts) if parts else "(no arguments)"
    return summary[:max_len]


def register_pending_action(
    *,
    tool_call: dict[str, Any],
    tool_name: str,
    llm_client=None,
    config: dict[str, Any] | None = None,
    message_list: list | None = None,
    context_question_options: dict[str, Any] | None = None,
    vector_store_manager=None,
    conversation_history: list | None = None,
    raw_transcription: str | None = None,
    patient_context: dict[str, Any] | None = None,
    actor: str | None = None,
) -> PendingAction:
    """Store a tool call awaiting approval and return it."""
    _prune_expired()
    action = PendingAction(
        id=secrets.token_urlsafe(16),
        tool_call=tool_call,
        tool_name=tool_name,
        summary=summarize_tool_args(tool_call),
        llm_client=llm_client,
        config=config or {},
        message_list=message_list or [],
        context_question_options=context_question_options or {},
        vector_store_manager=vector_store_manager,
        conversation_history=conversation_history or [],
        raw_transcription=raw_transcription,
        patient_context=dict(patient_context) if isinstance(patient_context, dict) else None,
        actor=actor,
    )
    if len(_pending) >= MAX_PENDING_ACTIONS:
        # Drop the oldest action to bound memory.
        oldest = min(_pending.values(), key=lambda a: a.created_at)
        _pending.pop(oldest.id, None)
    _pending[action.id] = action
    logger.info(f"Pending action registered: {tool_name} ({action.id})")
    return action


def pop_pending_action(action_id: str) -> PendingAction | None:
    """Remove and return a pending action (None if unknown/expired)."""
    action = _pending.pop(action_id, None)
    if action is None:
        return None
    if time.time() - action.created_at > ACTION_TTL_SECONDS:
        return None
    return action


def _prune_expired() -> None:
    now = time.time()
    expired = [aid for aid, a in _pending.items() if now - a.created_at > ACTION_TTL_SECONDS]
    for aid in expired:
        _pending.pop(aid, None)
