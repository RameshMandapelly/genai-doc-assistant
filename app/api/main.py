from fastapi import FastAPI, UploadFile, File, Form

app = FastAPI(title="GenAI Doc Assistant")


@app.get("/health-check")
def health_check():
    """Simple liveness check."""
    return {"status": "ok"}


@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """Accepts a file upload (e.g. PDF/DOCX) for later processing."""
    contents = await file.read()
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(contents),
    }


@app.post("/ask-question")
async def ask_question(question: str = Form(...)):
    """Accepts a question string and (eventually) returns an answer from the RAG pipeline."""
    return {
        "question": question,
        "answer": "TODO: wire this up to the RAG pipeline",
    }