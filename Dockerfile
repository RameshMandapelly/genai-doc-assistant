FROM python:3.11-slim

WORKDIR /app

# build-essential is needed for a couple of packages that compile native
# extensions (e.g. some chromadb/sentence-transformers dependencies).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch FIRST, from PyTorch's own CPU wheel index.
# sentence-transformers depends on torch, and without this step pip resolves
# the default GPU/CUDA build of torch - pulling several GB of NVIDIA CUDA
# libraries that are completely unused on a CPU-only host like Render's free
# tier. Installing the CPU wheel first satisfies that dependency before
# requirements.txt is processed, so pip never reaches for the CUDA version.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# data/ and chroma_db/ are in .gitignore/.dockerignore, so they won't exist
# yet inside the image - create them so the app doesn't fail on first request.
RUN mkdir -p data chroma_db

EXPOSE 8000

# Render (and most PaaS platforms) inject a $PORT env var at runtime and
# expect the app to listen on it, not on a hardcoded port. ${PORT:-8000}
# falls back to 8000 for local `docker run` where $PORT isn't set.
CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
