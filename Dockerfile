# KVForge — Progressive RAG with KV-Cache Injection
#
# Build (CPU-only, for indexing / search / tests):
#   docker build --target cpu -t kvforge:cpu .
#
# Build (GPU, for KV computation / LoRA training):
#   docker build --target gpu -t kvforge:gpu .
#
# Run Studio (CPU image):
#   docker run --rm -p 8080:8080 kvforge:cpu python kvforge_portal.py --port 8080
#
# Run Studio (GPU image):
#   docker run --rm --gpus all -p 8080:8080 kvforge:gpu python kvforge_portal.py --port 8080

# ── Base: shared system deps ───────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System libraries needed by torch, bitsandbytes, and chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── CPU stage: indexing, search, Studio UI, tests (no GPU required) ────────────
FROM base AS cpu

COPY requirements_gpu.txt .

# Install CPU-only torch first (much smaller than CUDA build)
RUN pip install --no-cache-dir \
    torch>=2.3.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies from requirements file
RUN pip install --no-cache-dir \
    transformers>=4.45.0 \
    peft>=0.12.0 \
    accelerate>=0.30.0 \
    datasets>=2.19.0 \
    fastapi>=0.111.0 \
    uvicorn>=0.29.0 \
    httpx>=0.27.0 \
    qdrant-client>=1.9.0 \
    "chromadb>=0.5.0,<1.0.0" \
    faiss-cpu>=1.8.0 \
    fastembed>=0.3.0 \
    anthropic>=0.30.0 \
    pypdf \
    pydantic \
    beautifulsoup4 \
    sentence-transformers \
    openai \
    pytest

COPY . .

EXPOSE 8080 8081 8082 8083 8084

CMD ["python", "kvforge_portal.py", "--port", "8080"]

# ── GPU stage: KV computation, LoRA training (requires NVIDIA runtime) ─────────
FROM base AS gpu

# CUDA 12.1 runtime is pulled transitively by the CUDA-enabled torch wheel;
# no separate cuda base image is needed when using the PyPI CUDA wheel.
COPY requirements_gpu.txt .

# Install CUDA-enabled torch
RUN pip install --no-cache-dir \
    torch>=2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

# bitsandbytes requires CUDA torch to be present first
RUN pip install --no-cache-dir \
    transformers>=4.45.0 \
    peft>=0.12.0 \
    bitsandbytes>=0.43.0 \
    accelerate>=0.30.0 \
    datasets>=2.19.0 \
    fastapi>=0.111.0 \
    uvicorn>=0.29.0 \
    httpx>=0.27.0 \
    qdrant-client>=1.9.0 \
    "chromadb>=0.5.0,<1.0.0" \
    faiss-cpu>=1.8.0 \
    fastembed>=0.3.0 \
    anthropic>=0.30.0 \
    pypdf \
    pydantic \
    beautifulsoup4 \
    sentence-transformers \
    openai \
    pytest

COPY . .

EXPOSE 8080 8081 8082 8083 8084

CMD ["python", "kvforge_portal.py", "--port", "8080"]
