from __future__ import annotations

from typing import Literal, cast

from langchain_core.prompts import PromptTemplate

DocumentType = Literal[
    "pdf",
    "docx",
    "csv",
    "json",
    "txt",
    "web",
    "database",
    "generic",
]

_COMMON_INSTRUCTIONS = """You answer questions using only the supplied context.
If the context does not contain enough information, say that you do not have
enough information instead of guessing. Keep the answer focused and accurate.
When the context contains source metadata, mention the relevant source in the
answer. Do not claim to have used a source that is not present in the context.
"""


def _template(instructions: str) -> PromptTemplate:
    return PromptTemplate.from_template(
        f"""{_COMMON_INSTRUCTIONS}
{instructions}

Context:
{{context}}

Question:
{{question}}

Answer:"""
    )


PDF_PROMPT = _template(
    """The context comes from one or more PDF documents. Preserve the meaning
of headings, page references, definitions, tables, and footnotes. If pages or
document names are available in the context metadata, use them when referring
to supporting information."""
)

DOCX_PROMPT = _template(
    """The context comes from Word documents. Respect the document structure,
including headings, paragraphs, lists, and table contents. Distinguish
explicit statements from your own short explanation."""
)

CSV_PROMPT = _template(
    """The context comes from CSV rows. Treat column names as field names and
do not combine values from different rows unless the context supports that
combination. For calculations, show the relevant values and state any
assumption. If a requested row or field is absent, say so."""
)

JSON_PROMPT = _template(
    """The context comes from JSON data. Preserve the relationship between
objects, arrays, and nested fields. Use the field names when they make the
answer clearer. Do not invent missing keys or values."""
)

TXT_PROMPT = _template(
    """The context comes from plain-text files. Answer directly from the
provided passages and preserve important wording, names, dates, and numbers.
Do not infer structure that is not present in the text."""
)

WEB_PROMPT = _template(
    """The context comes from web pages. Prefer information explicitly stated
in the retrieved page content. Treat navigation text, menus, advertisements,
and unrelated page fragments as noise. If a URL or page title is available,
use it as the source reference."""
)

DATABASE_PROMPT = _template(
    """The context comes from database records. Treat each record as
structured data and do not infer records that were not retrieved. For
aggregations or comparisons, identify the records and fields used. Respect
the meaning of null, missing, and empty values."""
)

GENERIC_PROMPT = _template(
    """The context may contain mixed document types. Use the source metadata
and document structure when available, and distinguish facts from
interpretation."""
)


PROMPT_TEMPLATES: dict[DocumentType, PromptTemplate] = {
    "pdf": PDF_PROMPT,
    "docx": DOCX_PROMPT,
    "csv": CSV_PROMPT,
    "json": JSON_PROMPT,
    "txt": TXT_PROMPT,
    "web": WEB_PROMPT,
    "database": DATABASE_PROMPT,
    "generic": GENERIC_PROMPT,
}


def get_prompt_template(document_type: str = "generic") -> PromptTemplate:
    """Return the answer prompt for a supported document type."""
    normalised_type = document_type.lower().lstrip(".")
    if normalised_type == "doc":
        normalised_type = "docx"
    if normalised_type in {"html", "htm", "url", "webpage"}:
        normalised_type = "web"

    if normalised_type not in PROMPT_TEMPLATES:
        supported = ", ".join(PROMPT_TEMPLATES)
        raise ValueError(
            f"Unsupported document type '{document_type}'. "
            f"Expected one of: {supported}"
        )
    return PROMPT_TEMPLATES[cast(DocumentType, normalised_type)]


def format_prompt(
    question: str,
    context: str,
    document_type: str = "generic",
) -> str:
    """Render a document-specific answer prompt."""
    if not question.strip():
        raise ValueError("question must not be empty")
    if not context.strip():
        raise ValueError("context must not be empty")

    return get_prompt_template(document_type).format(
        question=question,
        context=context,
    )


__all__ = [
    "DATABASE_PROMPT",
    "DOCX_PROMPT",
    "DocumentType",
    "CSV_PROMPT",
    "GENERIC_PROMPT",
    "JSON_PROMPT",
    "PDF_PROMPT",
    "PROMPT_TEMPLATES",
    "TXT_PROMPT",
    "WEB_PROMPT",
    "format_prompt",
    "get_prompt_template",
]
