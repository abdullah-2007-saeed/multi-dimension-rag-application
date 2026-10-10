from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from langchain_core.documents import Document


def load_web_page(url: str) -> list[Document]:
    try:
        from langchain_community.document_loaders import AsyncHtmlLoader
    except ModuleNotFoundError as error:
        if error.name in {"bs4", "beautifulsoup4"}:
            raise RuntimeError(
                "Web-page support requires beautifulsoup4. Install it with "
                "`uv pip install beautifulsoup4`."
            ) from error
        raise

    return AsyncHtmlLoader([url]).load()


def load_pdf(file_path: str) -> list[Document]:
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError as error:
        raise RuntimeError(
            "PDF support requires pypdf. Install it with "
            "`uv pip install pypdf`."
        ) from error

    return PyPDFLoader(file_path).load()


def load_docs(file_path: str) -> list[Document]:
    try:
        from langchain_community.document_loaders import Docx2txtLoader
    except ImportError as error:
        raise RuntimeError(
            "DOCX support requires docx2txt. Install it with "
            "`uv pip install docx2txt`."
        ) from error

    return Docx2txtLoader(file_path).load()


def load_dir(dir_path: str, globdata: str = "**/*") -> Iterator[Document]:
    from langchain_community.document_loaders import DirectoryLoader, TextLoader

    return DirectoryLoader(dir_path, glob=globdata, loader_cls=TextLoader).lazy_load()


def load_csv(file_path: str) -> list[Document]:
    from langchain_community.document_loaders import CSVLoader

    return CSVLoader(file_path, encoding="utf-8").load()


def load_json(file_path: str) -> list[Document]:
    path = Path(file_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        Document(
            page_content=json.dumps(item, ensure_ascii=False),
            metadata={"source": str(path), "format": "json"},
        )
        for item in (data if isinstance(data, list) else [data])
    ]


def load_text(file_path: str) -> list[Document]:
    from langchain_community.document_loaders import TextLoader

    return TextLoader(file_path, encoding="utf-8").load()


def fallback(
    file_path: str | None = None,
    url: str | None = None,
) -> list[Document]:
    """Load one supported source and fail clearly for unsupported input."""
    if file_path and url:
        raise ValueError("Provide either file_path or url, not both")
    if url:
        return load_web_page(url)
    if not file_path:
        raise ValueError("Either file_path or url must be provided")

    suffix = Path(file_path).suffix.lower()
    loaders = {
        ".pdf": load_pdf,
        ".docx": load_docs,
        ".csv": load_csv,
        ".json": load_json,
        ".txt": load_text,
    }
    try:
        return loaders[suffix](file_path)
    except KeyError as error:
        raise ValueError(f"Unsupported file format: {suffix or '<none>'}") from error


def load_relational_database(connection_string: str) -> list[Document]:
    del connection_string
    raise NotImplementedError("Relational database loading is not configured")


def load_non_relational_database(connection_string: str) -> list[Document]:
    del connection_string
    raise NotImplementedError("Non-relational database loading is not configured")


__all__ = [
    "fallback",
    "load_csv",
    "load_dir",
    "load_docs",
    "load_json",
    "load_non_relational_database",
    "load_pdf",
    "load_relational_database",
    "load_text",
    "load_web_page",
]
