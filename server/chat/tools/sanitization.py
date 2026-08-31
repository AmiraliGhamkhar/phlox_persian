"""Shared sanitization utilities for tool queries.

This module provides sanitization functions to clean queries before sending
to external services, removing potential PHI and other problematic content.

In addition to pattern-based scrubbing, the ChatEngine registers the active
patient's identifiers in a ContextVar (``set_active_patient_context``) so
every outbound external search strips that specific patient's name/UR number
even when the pattern rules would miss them.
"""

import logging
import re
from contextvars import ContextVar
from typing import Any

logger = logging.getLogger(__name__)

# Active patient for the current chat turn (set by ChatEngine).
_active_patient: ContextVar[dict[str, Any] | None] = ContextVar(
    "phlox_active_patient", default=None
)


def set_active_patient_context(patient_context: dict[str, Any] | None) -> None:
    """Record the patient whose record is open for the current chat turn.

    Called by the ChatEngine; external-search sanitization strips these exact
    identifiers from queries in addition to the generic patterns.
    """
    _active_patient.set(patient_context)


def get_active_patient_context() -> dict[str, Any] | None:
    """Return the patient context registered for the current task, if any."""
    return _active_patient.get()


def _fold_persian_digits(text: str) -> str:
    """Map Persian/Arabic-Indic digits to ASCII so the patterns below match."""
    digits = "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩"
    table = str.maketrans(digits, "01234567890123456789")
    return text.translate(table)


def sanitize_query_for_external_search(
    query: str, patient_context: dict[str, Any] | None = None
) -> str:
    """Remove potential PHI from search queries before sending to external services.

    This is a defense-in-depth measure to prevent accidental PHI leakage even if
    the LLM includes it in search queries despite prompt instructions.

    Args:
        query: The original search query
        patient_context: Optional dict with patient info to specifically remove
            (ur_number, name, etc.). When omitted, the ChatEngine-registered
            active patient is used if one exists.

    Returns:
        Sanitized query string with PHI patterns removed
    """
    if patient_context is None:
        patient_context = _active_patient.get()

    # Fold Persian/Arabic-Indic digits first so numeric PHI patterns match
    # regardless of the digit script the model used (Persian-first app).
    sanitized = _fold_persian_digits(query)

    # Remove common PHI patterns
    # UR numbers (various formats)
    sanitized = re.sub(r"\bUR\s*[:#]?\s*\d{4,}\b", "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\bUR\s*[:#]?\s*[A-Z]?\d{4,}[A-Z]?\b", "", sanitized, flags=re.IGNORECASE)

    # Medical record numbers
    sanitized = re.sub(r"\bMRN\s*[:#]?\s*\d+\b", "", sanitized, flags=re.IGNORECASE)

    # Dates of birth (Gregorian, various formats)
    sanitized = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "", sanitized)
    sanitized = re.sub(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", "", sanitized)

    # Jalali / Shamsi dates (Persian calendar, e.g. 1370/05/12 or 1370-5-3).
    # Years 1300-1500 cover all living patients.
    sanitized = re.sub(r"\b1[3-5]\d{2}[/-]\d{1,2}[/-]\d{1,2}\b", "", sanitized)

    # Phone numbers
    sanitized = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "", sanitized)
    sanitized = re.sub(r"\b\+?\d{1,3}[-.\s]?\d{2,4}[-.\s]?\d{3,4}[-.\s]?\d{4}\b", "", sanitized)

    # Email addresses
    sanitized = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", sanitized)

    # Addresses (simple pattern for street addresses)
    sanitized = re.sub(
        r"\b\d+\s+[A-Za-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl)\b",
        "",
        sanitized,
        flags=re.IGNORECASE,
    )

    # Medicare/health card numbers (Australian format)
    sanitized = re.sub(r"\b\d{4}\s?\d{4,5}\s?\d{1,2}\b", "", sanitized)

    # Iranian national ID (کد ملی: 10 digits) — folded digits make this match
    sanitized = re.sub(r"\b\d{3}-?\d{6}-?\d\b", "", sanitized)

    # Remove specific patient context if provided
    context = patient_context or _active_patient.get()
    if context:
        if context.get("ur_number"):
            sanitized = sanitized.replace(str(context["ur_number"]), "")
        if context.get("name"):
            # Remove each part of the name
            for name_part in str(context["name"]).split():
                sanitized = re.sub(
                    rf"\b{re.escape(name_part)}\b", "", sanitized, flags=re.IGNORECASE
                )

    # Clean up extra whitespace and punctuation
    sanitized = re.sub(r"\s+", " ", sanitized)
    sanitized = re.sub(r"^[\s,-]+|[\s,-]+$", "", sanitized)

    if sanitized != query:
        logger.info(
            "Sanitized outbound search query (%d -> %d chars)",
            len(query),
            len(sanitized),
        )

    # Fail closed: if the only content of the query was PHI, return an empty
    # string so callers skip the outbound request entirely. Returning the
    # original would send exactly the identifier we were asked to scrub
    # (LLM02:2026).
    return sanitized.strip()


def sanitize_pubmed_query(query: str) -> str:
    """Sanitize query by removing standalone years that PubMed interprets as keywords.

    PubMed E-utilities treats bare years (e.g., "2024") as required keyword search terms,
    not date filters. This removes standalone years while preserving years that are part
    of disease names (e.g., "COVID-19").

    Args:
        query: The raw search query

    Returns:
        Sanitized query with standalone years removed
    """

    sanitized = re.sub(r"(?<!\S)(20[0-2][0-9]|19[8-9][0-9])(?!\S)", "", query)

    # Clean up extra whitespace that may result from removal
    sanitized = re.sub(r"\s+", " ", sanitized).strip()

    return sanitized
