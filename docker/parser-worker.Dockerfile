FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    DOCLING_CACHE_DIR=/opt/docling-cache \
    HF_HOME=/opt/hf-cache \
    HOME=/tmp

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/docling-cache /opt/hf-cache /work

WORKDIR /app

COPY requirements-parser-worker.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-parser-worker.txt

COPY src ./src
COPY main.py ./

# Fetch and initialize model assets at image-build time
RUN python -c "from src.ingestion.parsers import StructuredParser; a=StructuredParser(); a.initialize(); b=StructuredParser(do_ocr=True); b.initialize(); a.close(); b.close()"

RUN useradd --system --uid 10001 --no-create-home parser
USER 10001:10001

ENTRYPOINT ["python", "-m", "src.ingestion.parser_worker.worker"]
