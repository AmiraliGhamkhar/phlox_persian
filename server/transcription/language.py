"""Language handling and Unicode normalization for Persian/mixed-language ASR."""

import re

# Arabic code points commonly returned by ASR engines for Persian characters.
_ARABIC_TO_PERSIAN = str.maketrans(
    {
        "ي": "ی",
        "ى": "ی",
        "ئ": "ی",
        "ك": "ک",
        "ة": "ه",
        "ۀ": "ه",
        "ؤ": "و",
        "إ": "ا",
        "أ": "ا",
        "ٱ": "ا",
        "ـ": "",
        "ً": "",
        "ٌ": "",
        "ٍ": "",
        "َ": "",
        "ُ": "",
        "ِ": "",
        "ّ": "",
        "ْ": "",
        "ٰ": "",
    }
)

_PERSIAN_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "۰۱۲۳۴۵۶۷۸۹")


def normalize_persian_text(text: str) -> str:
    """Normalize Arabic/Persian Unicode without changing Latin words or numbers.

    ASR providers differ in whether they emit Arabic ``ي``/``ك`` and Persian
    ``ی``/``ک``. Normalizing only these known code points keeps mixed clinical
    text such as ``HbA1c 7.2%`` and drug names intact.
    """
    if not text:
        return text
    normalized = text.translate(_ARABIC_TO_PERSIAN).translate(_PERSIAN_DIGITS)
    normalized = re.sub(r"[ \t\u00a0]+", " ", normalized)
    normalized = re.sub(r" *\n *", "\n", normalized).strip()
    # Keep punctuation attached to the preceding word while avoiding any
    # changes inside Latin identifiers or decimal values.
    normalized = re.sub(r"\s+([،؛؟٪%،.!?:;])", r"\1", normalized)
    return normalized


def resolve_asr_language(config: dict) -> str:
    """Return the configured ASR language, defaulting to mixed-language auto.

    ``WHISPER_LANGUAGE`` is retained as a migration alias because existing
    installations may already have that key. ``auto`` is intentionally the
    default: it supports Persian-only recordings and recordings that mix
    Persian with English medical terms.
    """
    language = config.get("ASR_LANGUAGE") or config.get("WHISPER_LANGUAGE") or "auto"
    language = str(language).strip().lower()
    if language in {"automatic", "detect", "mixed", "fa-en", "fa_en"}:
        return "auto"
    if language.startswith("fa-"):
        return "fa"
    if language.startswith("en-"):
        return "en"
    if language not in {"auto", "fa", "en"}:
        return "auto"
    return language


def streaming_asr_language(config: dict) -> str:
    """Return the language for a *streaming* (Realtime) ASR session.

    Automatic language identification (``auto``) is only supported by
    Speechmatics in Batch mode; the Realtime engine requires an explicit ISO
    language code. Persisting the configured default of ``auto`` silently
    kills every live session, so map it to the app's primary language (``fa``)
    for streaming providers. Users who need English can still pick ``en`` in
    the settings.
    """
    language = resolve_asr_language(config)
    return "fa" if language == "auto" else language
