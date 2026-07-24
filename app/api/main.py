import logging
import time
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.logging_config import configure_logging
from app.core.orchestrator import AgentOrchestrator
from app.services.ingestion import extract_text
from app.services.chunking import chunk_text
from app.services.vectorstore import add_chunks, get_vectorstore
from app.services.rag import answer_question

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="GenAI Doc Assistant")
_orchestrator = AgentOrchestrator()

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".csv", ".xlsx", ".xls", ".json", ".yaml", ".yml"}
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

MIN_QUESTION_LENGTH = 3
MAX_QUESTION_LENGTH = 500


# ---------------------------------------------------------------------------
# Request/response logging middleware
# ---------------------------------------------------------------------------
# Runs for EVERY request to EVERY endpoint, regardless of whether it succeeds,
# gets rejected by a guard (400/422), or errors out (500). This is what makes
# sure nothing is invisible in app.log - the per-endpoint logger.info() calls
# below add extra detail (the actual question text) on top of this baseline.

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    logger.info(
        "%s %s -> %d (%.1fms)",
        request.method, request.url.path, response.status_code, duration_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Catches FastAPI/Pydantic request-shape errors (e.g. missing required form field)."""
    logger.warning("Validation error on %s: %s", request.url.path, exc.errors())
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request.", "errors": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Last-resort catch-all. HTTPException (used for our own validation errors below)
    is handled by FastAPI's default handler before this ever runs, so this only
    fires for genuinely unexpected errors - which is exactly when you don't want
    to leak a raw stack trace to the client, but do want it in your logs.
    """
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again."},
    )


# ---------------------------------------------------------------------------
# Shared input guards
# ---------------------------------------------------------------------------

def _validate_question(question: str) -> str:
    question = question.strip()
    if len(question) < MIN_QUESTION_LENGTH:
        logger.warning("Rejected question (too short, len=%d): %r", len(question), question)
        raise HTTPException(
            status_code=400,
            detail=f"Question is too short (minimum {MIN_QUESTION_LENGTH} characters).",
        )
    if len(question) > MAX_QUESTION_LENGTH:
        logger.warning(
            "Rejected question (too long, len=%d): %r...",
            len(question), question[:80],
        )
        raise HTTPException(
            status_code=400,
            detail=f"Question is too long (maximum {MAX_QUESTION_LENGTH} characters).",
        )
    return question


def _ensure_store_has_documents():
    """Fails fast with a clear message instead of silently returning 'I don't know'."""
    store = get_vectorstore()
    existing = store.get(limit=1)
    if not existing["ids"]:
        logger.warning("Rejected request: no documents in vector store yet")
        raise HTTPException(
            status_code=400,
            detail="No documents have been uploaded yet. Upload a document via /upload-document first.",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health-check")
def health_check():
    """Simple liveness check."""
    return {"status": "ok"}


@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """
    Validates, saves, extracts, chunks, embeds, and stores an uploaded document.
    """
    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        logger.warning("Rejected upload '%s': unsupported extension %s", file.filename, extension)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    contents = await file.read()

    if len(contents) == 0:
        logger.warning("Rejected upload '%s': empty file", file.filename)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(contents) > MAX_FILE_SIZE_BYTES:
        logger.warning(
            "Rejected upload '%s': %d bytes exceeds limit", file.filename, len(contents)
        )
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(contents) / (1024*1024):.2f} MB). Max allowed: {MAX_FILE_SIZE_MB} MB",
        )

    save_path = DATA_DIR / file.filename
    save_path.write_bytes(contents)
    logger.info("Saved upload: %s (%d bytes)", file.filename, len(contents))

    try:
        extracted_text = extract_text(str(save_path))
    except Exception as e:
        logger.exception("Failed to extract text from %s", file.filename)
        raise HTTPException(status_code=422, detail=f"Failed to extract text: {e}")

    chunks = chunk_text(extracted_text)
    if not chunks:
        logger.warning("No chunks produced for '%s' (extracted_length=%d)", file.filename, len(extracted_text))
        raise HTTPException(status_code=422, detail="No text could be extracted/chunked from this file.")

    try:
        stored_count = add_chunks(chunks, filename=file.filename)
    except Exception as e:
        logger.exception("Failed to embed/store chunks for %s", file.filename)
        raise HTTPException(status_code=500, detail=f"Failed to embed/store chunks: {e}")

    logger.info("Stored %d chunks for %s", stored_count, file.filename)

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(contents),
        "saved_to": str(save_path),
        "extracted_length": len(extracted_text),
        "chunks_created": len(chunks),
        "chunks_stored": stored_count,
    }


@app.post("/ask-question")
async def ask_question(question: str = Form(...)):
    """Answers a question using single-shot RAG: retrieves relevant chunks and generates a grounded answer."""
    question = _validate_question(question)
    _ensure_store_has_documents()

    result = answer_question(question)
    logger.info(
        "Answered /ask-question: question=%r answer=%r",
        question, result["answer"][:150],
    )
    return {
        "question": question,
        "answer": result["answer"],
        "sources": result["sources"],
    }


@app.post("/ask-agent")
async def ask_agent(question: str = Form(...)):
    """
    Answers using the full multi-agent pipeline:
    Planner -> Retriever -> Reasoner -> Responder -> Verifier.
    Returns the answer, sources, and a full trace of what each agent did.
    """
    question = _validate_question(question)
    _ensure_store_has_documents()

    result = _orchestrator.run(question)
    logger.info(
        "Answered /ask-agent: question=%r answer=%r",
        question, result["answer"][:150],
    )
    return result