from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException

from app.services.ingestion import extract_text

app = FastAPI(title="GenAI Doc Assistant")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".csv", ".xlsx", ".xls", ".json", ".yaml", ".yml"}
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


@app.get("/health-check")
def health_check():
    """Simple liveness check."""
    return {"status": "ok"}


@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """
    Validates, saves, and extracts text from an uploaded document.
    """
    # 1. Validate extension
    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    # 2. Read contents and validate size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(contents) / (1024*1024):.2f} MB). Max allowed: {MAX_FILE_SIZE_MB} MB",
        )

    # 3. Save to data/
    save_path = DATA_DIR / file.filename
    save_path.write_bytes(contents)

    # 4. Extract text using the ingestion service
    try:
        extracted_text = extract_text(str(save_path))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to extract text: {e}")

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(contents),
        "saved_to": str(save_path),
        "extracted_preview": extracted_text[:300],
        "extracted_length": len(extracted_text),
    }


@app.post("/ask-question")
async def ask_question(question: str = Form(...)):
    """Accepts a question string and (eventually) returns an answer from the RAG pipeline."""
    return {
        "question": question,
        "answer": "TODO: wire this up to the RAG pipeline",
    }