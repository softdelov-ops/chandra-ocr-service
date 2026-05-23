# ── Chandra OCR 2 — Dockerfile ────────────────────────────────────────────────
# Base: PyTorch con CUDA 12.1 + Python 3.11
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Evitar prompts interactivos de apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Dependencias del sistema (poppler para pdf2image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Código del servicio
COPY app/ ./app/

# Directorios de datos (sobreescritos por volúmenes en docker-compose)
RUN mkdir -p /data/input /data/output /data/cache

# Variables de entorno por defecto
ENV CHANDRA_MODEL="datalab-to/chandra-ocr-2"
ENV OCR_DPI=96
ENV MAX_IMG_SIDE=1600
ENV BATCH_SIZE=2
ENV PREFETCH_WORKERS=2
ENV MIN_CHARS=50
ENV LOAD_4BIT=true
ENV INPUT_DIR=/data/input
ENV OUTPUT_DIR=/data/output
ENV CACHE_DIR=/data/cache

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
