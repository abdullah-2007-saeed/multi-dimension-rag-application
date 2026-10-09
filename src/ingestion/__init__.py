from .loader import (
    fallback,
    load_csv,
    load_dir,
    load_docs,
    load_json,
    load_non_relational_database,
    load_pdf,
    load_relational_database,
    load_text,
    load_web_page,
)

__all__ = ["fallback", "load_web_page", "load_non_relational_database",
"load_pdf",
"load_relational_database", "load_docs", "load_dir", "load_csv", "load_json", "load_text"]