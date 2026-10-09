

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.api import router
from src.embeddings import get_embeddings
from src.ingestion import fallback
from src.llm import get_llm
from src.prompts import get_prompt_template
from src.utils import RAGPipeline, build_rag_pipeline, infer_document_type




logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@dataclass
class ComponentStatus:
    status: str
    error: str | None = None


@dataclass
class ApplicationState:
    rag: RAGPipeline | None = None
    components: dict[str, ComponentStatus] = field(default_factory=dict)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    document_type: str = "generic"
    k: int = Field(default=4, gt=0)


class IngestRequest(BaseModel):
    file_path: str | None = None
    url: str | None = None
    document_type: str | None = None


def _check_component(
    state: ApplicationState,
    name: str,
    check: Any,
) -> None:
    try:
        check()
        state.components[name] = ComponentStatus(status="ready")
    except Exception as error:
        state.components[name] = ComponentStatus(
            status="failed",
            error=f"{type(error).__name__}: {error}",
        )
        logger.exception("Component '%s' failed during startup", name)


def initialize_application() -> ApplicationState:
    """Initialize independent components and retain actionable debug status."""
    state = ApplicationState()

    _check_component(state, "ingestion", lambda: fallback)
    _check_component(state, "prompts", lambda: get_prompt_template("generic"))
    _check_component(state, "embeddings", lambda: get_embeddings)
    _check_component(state, "llm", lambda: get_llm)

    def initialize_rag() -> None:
        state.rag = build_rag_pipeline(
            collection_name=os.getenv("CHROMA_COLLECTION", "documents"),
            db_type=os.getenv("CHROMA_DB_TYPE", "persistent"),
            db_path=os.getenv("CHROMA_PATH", "./chroma"),
        )

    _check_component(state, "rag", initialize_rag)
    return state


def create_app() -> FastAPI:
    state = initialize_application()
    app = FastAPI(title="Local RAG API")
    app.state.rag_state = state
    app.include_router(router)

    @app.get("/debug/components")
    def component_status() -> dict[str, dict[str, str | None]]:
        return {
            name: {"status": item.status, "error": item.error}
            for name, item in state.components.items()
        }

    @app.post("/ingest")
    def ingest(request: IngestRequest) -> dict[str, Any]:
        if state.rag is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "RAG pipeline is unavailable",
                    "error": state.components.get("rag", ComponentStatus("unknown")).error,
                },
            )
        try:
            documents = fallback(file_path=request.file_path, url=request.url)
            source = request.document_type or request.file_path or request.url
            ids = state.rag.index_documents(
                documents,
                source_type=infer_document_type(source),
            )
            return {"indexed": len(ids), "ids": ids}
        except Exception as error:
            logger.exception("Ingestion failed")
            raise HTTPException(
                status_code=400,
                detail=f"{type(error).__name__}: {error}",
            ) from error

    @app.post("/ingest-upload")
    async def ingest_upload(
        file: UploadFile = File(...),
        document_type: str | None = None,
    ) -> dict[str, Any]:
        if state.rag is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "RAG pipeline is unavailable",
                    "error": state.components.get("rag", ComponentStatus("unknown")).error,
                },
            )
        if not file.filename:
            raise HTTPException(status_code=400, detail="Uploaded file has no filename")

        suffix = Path(file.filename).suffix
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
                temporary_path = temporary_file.name
                while chunk := await file.read(1024 * 1024):
                    temporary_file.write(chunk)

            documents = fallback(file_path=temporary_path)
            source = document_type or file.filename
            ids = state.rag.index_documents(
                documents,
                source_type=infer_document_type(source),
            )
            return {"indexed": len(ids), "ids": ids, "filename": file.filename}
        except Exception as error:
            logger.exception("Uploaded file ingestion failed")
            raise HTTPException(
                status_code=400,
                detail=f"{type(error).__name__}: {error}",
            ) from error
        finally:
            if temporary_path:
                Path(temporary_path).unlink(missing_ok=True)
            await file.close()

    @app.post("/ask")
    def ask(request: AskRequest) -> dict[str, str]:
        if state.rag is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "RAG pipeline is unavailable",
                    "error": state.components.get("rag", ComponentStatus("unknown")).error,
                },
            )
        try:
            answer = state.rag.ask(
                request.question,
                document_type=request.document_type,
                k=request.k,
            )
            return {"answer": answer}
        except Exception as error:
            logger.exception("Question answering failed")
            raise HTTPException(
                status_code=503,
                detail=f"{type(error).__name__}: {error}",
            ) from error

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )